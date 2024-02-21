from pathlib import Path
import pickle
from typing import Dict, List, Union
import math

import dgl
from dgl.dataloading import GraphDataLoader
import torch
import numpy as np

from data_processing.pdbbind_processing import (build_initial_complex_graph,
                                                get_pocket_atoms, parse_ligand,
                                                parse_protein, get_ot_loss_weights)

class ProteinLigandDataset(dgl.data.DGLDataset):

    def __init__(self, name: str, 
        processed_data_file: str,
        data_files: str,
        rec_file: str,
        prot_elements: List[str],
        graph_cutoffs: dict,
        load_data: bool = True,
        subsample_pharms: bool = False,
        **kwargs):

        self.graph_cutoffs = graph_cutoffs
        self.prot_elements = prot_elements

        # If load_data is true, tensor will be constructed from data pkl files
        self.load_data = load_data
        self.data_files = data_files
        self.rec_file = rec_file

        if self.load_data:
            if not self.data_files or not self.rec_file:
                raise ValueError('load_data set to True but no data or receptor files provided in config')
            for file in self.data_files:
                with open(file, 'rb') as f:
                    data = pickle.load(f)
                
                ## Collect data entries into lists
                rec_file_name = [prot[0] for prot in data]
                lig_file_name = [ph[1] for ph in data]
                lig_obj = [ph[2] for ph in data]
                lig_pos_arr = [lig[3][0] for lig in data]
                lig_feat_arr = [lig[3][1] for lig in data]
                rec_pos_arr = [rec[4][0] for rec in data]
                rec_feat_arr = [rec[4][1] for rec in data]

                ## Convert lig_feat_arr and lig_pos_arr into single list
                ## And encode pharmacophore feats as one-hot
                lig_idx = []
                lig_pos = []
                lig_feat = []
                idx = 0
                
                for i, entry in enumerate(lig_pos_arr):
                    lig_pos.extend(entry)
                    lig_idx.append(idx)
                    idx += len(entry) - 1
                    lig_idx.append(idx)
                    idx += 1

                    lig_feat_arr.extend(one_hot_encode_pharms(lig_feat[i]))
                
                ## Create dictionary to encode protein elements
                ele_idx_map = { element: idx for idx, element in enumerate(self.prot_elements) }

                ## Convert rec_feat_arr and rec_pos_arr into single list
                ## And encode protein elements as one-hot 
                prot_idxs = []
                prot_pos = []
                prot_feat = []
                idx = 0
                for i, entry in enumerate(rec_pos_arr):
                    prot_pos.extend(entry)
                    prot_idxs.append(idx)
                    idx += len(entry) - 1
                    prot_idxs.append(idx)
                    idx += 1

                    prot_feat.extend(one_hot_encode_prots(rec_feat_arr[i], ele_idx_map, len(self.prot_elements)))

                ## Make data into tensor
                

                # ## Set up dictionary of relevant data
                # data_dict = {}
                # data_dict['prot_file_name'] = rec_file_name
                # data_dict['pharm_file_name'] = lig_file_name
                # data_dict['pharm_obj'] = lig_obj
                # data_dict['pharm_pos'] = lig_pos
                # data_dict['pharm_feat'] = lig_feat
                # data_dict['prot_pos'] = rec_pos
                # data_dict['prot_feat'] = rec_feat

        # define filepath of data
        self.processed_data_file: Path = Path(processed_data_file)

        # if subsample_pharms is True, getitem will return random resamplings of pharmacophores to augment dataset
        self.subsample_pharms = subsample_pharms


        super().__init__(name=name) # this has to happen last because this will call self.process()

    def __getitem__(self, i):

        pharm_pos = self.pharm_pos[i]
        pharm_feat = self.pharm_feat[i]
        prot_pos = self.prot_pos[i]
        prot_feat = self.prot_feat[i]

        ## Subsample pharmacophore features if subsample_pharms is True
        # select random indices, grab for both pos and feats, build graph with each of those (re-call this method??)

        complex_graph = build_initial_complex_graph(prot_pos, prot_feat, cutoffs=self.graph_cutoffs, pharm_atom_positions=pharm_pos, pharm_atom_features=pharm_feat)

        # complex_graph = self.data['complex_graph'][i]

        complex_graph.nodes['pharm'].data['h_0'] = pharm_feat
        
        for ntype in ['pharm', 'prot']:
            complex_graph.nodes[ntype].data['h_0'] = complex_graph.nodes[ntype].data['h_0'].float()

        return complex_graph

    def __len__(self):
        return self.pharm_files.shape[0] - 1

    def process(self):
        # load data into memory
        if not self.load_data:
            self.pharm_segments = torch.tensor([0])
        else:

            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
            
            self.pharm_pos = data['pharm_pos']
            self.pharm_feat = data['pharm_feat']
            self.prot_pos = data['prot_pos']
            self.prot_feat = data['prot_feat']
            self.prot_files = data['prot_file_name'] ## filepath to receptor file (only for val/test splits)
            self.pharm_files = data['pharm_file_name'] ## filepath to ligand file (only for val/test splits)

    def lig_atom_idx_to_element(self, element_idxs: List[int]):
        atom_elements = [ self.lig_reverse_map[element_idx] for element_idx in element_idxs ]
        return atom_elements

    @property
    def type_counts_file(self) -> Path:
        dataset_split = self.data_file.name.split('_')[0]
        types_file = self.data_file.parent / f'{dataset_split}_type_counts.pkl'
        return types_file

    @property
    def dataset_dir(self) -> Path:
        return self.data_file.parent

    def get_files(self, idx: int):
        """Given an index of the dataset, return the filepath of the receptor pdb and ligand sdf."""

        return self.prot_files[idx], self.lig_files[idx]

## Fn for one-hot encoding pharmacophore features
## Aromatic, HydrogenDonor, HydrogenAcceptor, PositiveIon, NegativeIon, Hydrophobic
def one_hot_encode_pharms(arr):
    one_hot = np.zeros((len(arr), 6))
    one_hot[np.arange(len(arr)), arr] = 1
    return one_hot

## Fn for one-hot encoding protein features
def one_hot_encode_prots(arr, ele_idx_map, num_ele):
    one_hot = np.zeros((len(arr), num_ele))
    for i, e in enumerate(arr):
        one_hot[i, ele_idx_map[e]] = 1
    return one_hot

def collate_fn(examples: list):

    # break receptor graphs, ligand positions, and ligand features into separate lists
    complex_graphs, interface_points = zip(*examples)

    # batch the receptor graphs together
    complex_graphs = dgl.batch(complex_graphs)
    return complex_graphs, interface_points

def get_dataloader(dataset: ProteinLigandDataset, batch_size: int, num_workers: int = 1, **kwargs) -> GraphDataLoader:

    dataloader = GraphDataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=num_workers, collate_fn=collate_fn, **kwargs)
    return dataloader