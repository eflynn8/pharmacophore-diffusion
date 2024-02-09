from pathlib import Path
import pickle
from typing import Dict, List, Union
import math

import dgl
from dgl.dataloading import GraphDataLoader
import torch

from data_processing.pdbbind_processing import (build_initial_complex_graph,
                                                get_pocket_atoms, parse_ligand,
                                                parse_protein, get_ot_loss_weights)

# TODO: with the current implementation of fake atoms, the code will not function properly if max_fake_atom_frac = 0

class ProteinLigandDataset(dgl.data.DGLDataset):

    def __init__(self, name: str, 
        processed_data_file: str,
        prot_elements: List[str],
        pharm_elements: List[str],
        graph_cutoffs: dict,
        pocket_cutoff: Union[int, float] = 4,
        load_data: bool = True,
        subsample_pharms: bool = False,
        **kwargs):

        self.graph_cutoffs = graph_cutoffs

        # if load_data is false, we don't want to actually process any data
        self.load_data = load_data

        # define filepath of data
        self.data_file: Path = Path(processed_data_file)

        # if subsample_pharms is True, getitem will return random resamplings of pharmacophores to augment dataset
        self.subsample_pharms = subsample_pharms

         ### Don't think this is needed anymore
        # atom typing configurations
        # self.prot_elements = prot_elements
        # self.pharm_elements = pharm_elements

        ### Don't think this is needed anymore
        # self.prot_element_map: Dict[str, int] = { element: idx for idx, element in enumerate(self.prot_elements) }
        # self.prot_element_map['other'] = len(self.prot_elements)

        # self.pharm_elements = pharm_elements
        # self.pharm_element_map: Dict[str, int] = { element: idx for idx, element in enumerate(self.pharm_elements) }
        # self.pharm_element_map['other'] = len(self.pharm_elements)

        # self.pharm_reverse_map = {v:k for k,v in self.pharm_element_map.items()}

        ### Should be able to remove this
        # self.pharm_box_padding: Union[int, float] = pharm_box_padding
        # self.pocket_cutoff: Union[int, float] = pocket_cutoff

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

        
def collate_fn(examples: list):

    # break receptor graphs, ligand positions, and ligand features into separate lists
    complex_graphs, interface_points = zip(*examples)

    # batch the receptor graphs together
    complex_graphs = dgl.batch(complex_graphs)
    return complex_graphs, interface_points

def get_dataloader(dataset: ProteinLigandDataset, batch_size: int, num_workers: int = 1, **kwargs) -> GraphDataLoader:

    dataloader = GraphDataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=num_workers, collate_fn=collate_fn, **kwargs)
    return dataloader