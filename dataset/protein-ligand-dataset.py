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
        rec_elements: List[str],
        lig_elements: List[str],
        graph_cutoffs: dict,
        lig_box_padding: Union[int, float] = 6, # an argument that is only useful for processing crossdocked data, and it is never actually used by this class 
        pocket_cutoff: Union[int, float] = 4,
        load_data: bool = True,
        max_fake_atom_frac: float = 0.0,
        **kwargs):

        self.max_fake_atom_frac = max_fake_atom_frac
        self.graph_cutoffs = graph_cutoffs

        # if load_data is false, we don't want to actually process any data
        self.load_data = load_data

        # define filepath of data
        self.data_file: Path = Path(processed_data_file)

        # atom typing configurations
        self.rec_elements = rec_elements
        self.rec_element_map: Dict[str, int] = { element: idx for idx, element in enumerate(self.rec_elements) }
        self.rec_element_map['other'] = len(self.rec_elements)

        self.lig_elements = lig_elements
        self.lig_element_map: Dict[str, int] = { element: idx for idx, element in enumerate(self.lig_elements) }
        self.lig_element_map['other'] = len(self.lig_elements)

        self.lig_reverse_map = {v:k for k,v in self.lig_element_map.items()}

        # hyperparameters for protein graph. these are never actually used. but they're considered "Dataset" parameters i suppose??
        self.lig_box_padding: Union[int, float] = lig_box_padding
        self.pocket_cutoff: Union[int, float] = pocket_cutoff

        super().__init__(name=name) # this has to happen last because this will call self.process()

    def __getitem__(self, i):

        lig_start_idx, lig_end_idx = self.lig_segments[i:i+2]
        rec_start_idx, rec_end_idx = self.rec_segments[i:i+2]

        lig_pos = self.lig_pos[lig_start_idx:lig_end_idx]
        lig_feat = self.lig_feat[lig_start_idx:lig_end_idx]
        rec_pos = self.rec_pos[rec_start_idx:rec_end_idx]
        rec_feat = self.rec_feat[rec_start_idx:rec_end_idx]


        complex_graph = build_initial_complex_graph(rec_pos, rec_feat, cutoffs=self.graph_cutoffs, lig_atom_positions=lig_pos, lig_atom_features=lig_feat)

        complex_graph = self.data['complex_graph'][i]

        complex_graph.nodes['lig'].data['h_0'] = lig_feat
        
        for ntype in ['lig', 'rec']:
            complex_graph.nodes[ntype].data['h_0'] = complex_graph.nodes[ntype].data['h_0'].float()

        return complex_graph

    def __len__(self):
        return self.lig_segments.shape[0] - 1

    def process(self):
        # load data into memory
        if not self.load_data:
            self.lig_segments = torch.tensor([0])
        else:

            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
            

            self.lig_pos = [l[3][0] for l in data]
            self.lig_feat = [l[3][1] for l in data]
            self.rec_pos = [p[4][0] for p in data]
            self.rec_feat = [p[4][1] for p in data]
            self.rec_segments = data['rec_segments']
            self.lig_segments = data['lig_segments']
            # self.ip_segments = data['ip_segments']
            self.rec_files = [p[0] for p in data] ## filepath to receptor file (only for val/test splits)
            self.lig_files = [l[1] for l in data] ## filepath to ligand file (only for val/test splits)

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

        return self.rec_files[idx], self.lig_files[idx]

        
def collate_fn(examples: list):

    # break receptor graphs, ligand positions, and ligand features into separate lists
    complex_graphs, interface_points = zip(*examples)

    # batch the receptor graphs together
    complex_graphs = dgl.batch(complex_graphs)
    return complex_graphs, interface_points

def get_dataloader(dataset: ProteinLigandDataset, batch_size: int, num_workers: int = 1, **kwargs) -> GraphDataLoader:

    dataloader = GraphDataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=num_workers, collate_fn=collate_fn, **kwargs)
    return dataloader