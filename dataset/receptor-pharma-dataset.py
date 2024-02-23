from pathlib import Path
import pickle
from typing import Dict, List, Union
import math

import dgl
from dgl.dataloading import GraphDataLoader
import torch
import numpy as np
from torch_cluster import radius_graph

# from data_processing.pdbbind_processing import (build_initial_complex_graph,
#                                                 get_pocket_atoms, parse_ligand,
#                                                 parse_protein, get_ot_loss_weights)

class ProteinPharmacophoreDataset(dgl.data.DGLDataset):

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

        # if subsample_pharms is True, getitem will return random resamplings of pharmacophores to augment dataset
        self.subsample_pharms = subsample_pharms

        # If load_data is true, tensor will be constructed from data pkl files
        self.load_data = load_data
        self.data_files = data_files
        self.rec_file = rec_file

        # define filepath of data
        self.processed_data_file: Path = Path(processed_data_file)

        if self.load_data:
            if not self.data_files or not self.rec_file:
                raise ValueError('load_data set to True but no data or receptor files provided in config')
            for file in self.data_files:
                with open(file, 'rb') as f:
                    data = pickle.load(f)
                
                ## Collect data entries into lists
                prot_file_name = [prot[0] for prot in data]
                pharm_file_name = [ph[1] for ph in data]
                pharm_obj = [ph[2] for ph in data]
                pharm_pos_arr = [pharm[3][0] for pharm in data]
                pharm_feat_arr = [pharm[3][1] for pharm in data]
                prot_pos_arr = [prot[4][0] for prot in data]
                prot_feat_arr = [prot[4][1] for prot in data]

                ## Convert pharm_feat_arr and pharm_pos_arr into single list
                ## And encode pharmacophore feats as one-hot
                pharm_idx = []
                pharm_pos = []
                pharm_feat = []
                idx = 0
                
                for i, entry in enumerate(pharm_pos_arr):
                    pharm_pos.extend(entry)
                    pharm_idx.append(idx)
                    idx += len(entry) - 1
                    pharm_idx.append(idx)
                    idx += 1

                    pharm_feat.extend(one_hot_encode_pharms(pharm_feat_arr[i]))
                
                ## Create dictionary to encode protein elements
                ele_idx_map = { element: idx for idx, element in enumerate(self.prot_elements) }

                ## Convert prot_feat_arr and prot_pos_arr into single list
                ## And encode protein elements as one-hot 
                prot_idx = []
                prot_pos = []
                prot_feat = []
                idx = 0
                for i, entry in enumerate(prot_pos_arr):
                    prot_pos.extend(entry)
                    prot_idx.append(idx)
                    idx += len(entry) - 1
                    prot_idx.append(idx)
                    idx += 1

                    prot_feat.extend(one_hot_encode_prots(prot_feat_arr[i], ele_idx_map, len(self.prot_elements)))

                ## Make data into tensor
                self.prot_file_name = torch.tensor(prot_file_name)
                self.pharm_file_name = torch.tensor(pharm_file_name)
                self.pharm_obj = torch.tensor(pharm_obj)
                self.pharm_pos = torch.tensor(pharm_pos)
                self.pharm_feat = torch.tensor(pharm_feat)
                self.prot_pos = torch.tensor(prot_pos)
                self.prot_feat = torch.tensor(prot_feat)
                self.pharm_idx = torch.tensor(pharm_idx)
                self.prot_idx = torch.tensor(prot_idx)

                ## Save list of tensors to processed_data_file
                torch.save([self.prot_file_name, self.pharm_file_name, self.pharm_obj, self.pharm_pos, self.pharm_feat,
                            self.prot_pos, self.prot_feat, self.pharm_idx, self.prot_idx], processed_data_file)
        else:
            data = torch.load(processed_data_file)
            self.prot_file_name = data[0]
            self.pharm_file_name = data[1]
            self.pharm_obj = data[2]
            self.pharm_pos = data[3]
            self.pharm_feat = data[4]
            self.prot_pos = data[5]
            self.prot_feat = data[6]
            self.pharm_idx = data[7]
            self.prot_idx = data[8]

        super().__init__(name=name) # this has to happen last because this will call self.process()

    def __getitem__(self, i):
        
        pharm_start_idx, pharm_end_idx = self.pharm_idx[i:i+2]
        prot_start_idx, prot_end_idx = self.prot_idx[i:i+2]

        pharm_pos = self.pharm_pos[pharm_start_idx:pharm_end_idx]
        pharm_feat = self.pharm_feat[pharm_start_idx:pharm_end_idx]
        prot_pos = self.prot_pos[prot_start_idx:prot_end_idx]
        prot_feat = self.prot_feat[prot_start_idx:prot_end_idx]
        # rec_res_idx = self.rec_res_idx[rec_start_idx:rec_end_idx]

        ## Subsample pharmacophore features if subsample_pharms is True
        # select random indices, grab for both pos and feats, build graph with each of those (re-call this method??)

        complex_graph = build_initial_complex_graph(prot_pos, prot_feat, cutoffs=self.graph_cutoffs, pharm_atom_positions=pharm_pos, pharm_atom_features=pharm_feat)

        # complex_graph = self.data['complex_graph'][i]

        complex_graph.nodes['pharm'].data['h_0'] = pharm_feat
        
        for ntype in ['pharm', 'prot']:
            complex_graph.nodes[ntype].data['h_0'] = complex_graph.nodes[ntype].data['h_0'].float()

        return complex_graph

    def __len__(self):
        return self.prot_file_name.shape[0] - 1

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

def build_initial_complex_graph(rec_atom_positions: torch.Tensor, rec_atom_features: torch.Tensor, pocket_res_idx: torch.Tensor, n_keypoints: int, cutoffs: dict, lig_atom_positions: torch.Tensor = None, lig_atom_features: torch.Tensor = None):

    if (lig_atom_positions is not None) ^ (lig_atom_features is not None):
        raise ValueError('ligand position and features must be either be both supplied or both left as None')

    n_rec_atoms = rec_atom_positions.shape[0]

    if lig_atom_positions is None:
        n_lig_atoms = 0
    else:
        n_lig_atoms = lig_atom_positions.shape[0]
    

    # i've initialized this as an empty dict just to make clear the different types of edges in graph and their names
    no_edges = ([], [])
    graph_data = {
        ('prot', 'pp', 'prot'): no_edges,
        ('prot', 'pf', 'pharm'): no_edges,
        ('pharm', 'ff', 'pharm'): no_edges,
        ('pharm', 'fp', 'prot'): no_edges
    }

    # compute rec atom -> rec atom edges
    pp_edges = radius_graph(rec_atom_positions, r=cutoffs['pp'], max_num_neighbors=100)
    graph_data[('prot', 'pp', 'prot')] = (pp_edges[0], pp_edges[1])

    # compute "same residue" feature ofr every rr edge
    same_res_edge = pocket_res_idx[pp_edges[0]] == pocket_res_idx[pp_edges[1]]

    num_nodes_dict = {
        'rec': n_rec_atoms,'lig': n_lig_atoms
    }

    # create graph object
    g = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)

    # add node data
    if lig_atom_positions is not None:
        g.nodes['lig'].data['x_0'] = lig_atom_positions
        g.nodes['lig'].data['h_0'] = lig_atom_features
    g.nodes['rec'].data['x_0'] = rec_atom_positions
    g.nodes['rec'].data['h_0'] = rec_atom_features
    
    # add edge data
    g.edges['pp'].data['same_res'] = same_res_edge.view(-1, 1)

    return g

def collate_fn(examples: list):

    # break receptor graphs, ligand positions, and ligand features into separate lists
    complex_graphs, interface_points = zip(*examples)

    # batch the receptor graphs together
    complex_graphs = dgl.batch(complex_graphs)
    return complex_graphs, interface_points

def get_dataloader(dataset: ProteinPharmacophoreDataset, batch_size: int, num_workers: int = 1, **kwargs) -> GraphDataLoader:

    dataloader = GraphDataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=num_workers, collate_fn=collate_fn, **kwargs)
    return dataloader