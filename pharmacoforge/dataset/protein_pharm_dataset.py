from pathlib import Path
import pickle
from typing import Dict, List, Union
import math
import os
import rdkit

import dgl
from dgl.dataloading import GraphDataLoader
import torch
import numpy as np
import random
from torch_cluster import radius_graph
import gzip
from torch.nn.functional import one_hot
from pharmacoforge.utils.relative_paths import fix_relative_path
from pharmacoforge.models.priors import com_free_gaussian, align_prior

class ProteinPharmacophoreDataset(dgl.data.DGLDataset):

    def __init__(self,
        name: str,
        split_idxs: List[int],
        raw_data_dir: str,
        processed_data_dir: str,
        graph_cutoffs: dict,
        prot_elements: List[str],
        ph_type_map: List[str],
        subsample_pharms: bool = False,
        subsample_min: int = 3,
        subsample_max: int = 9,  
        model_class: str = 'diffusion',
        **kwargs):

        self.graph_cutoffs = graph_cutoffs
        self.prot_elements = prot_elements
        self.ph_type_map = ph_type_map
        self.raw_data_dir=raw_data_dir
        # if subsample_pharms is True, getitem will return random resamplings of pharmacophores to augment dataset
        self.subsample_pharms = subsample_pharms
        self.subsample_min = subsample_min
        self.subsample_max = subsample_max
        self.model_class = model_class

        # define filepath of data
        self.processed_data_dir: Path = Path(processed_data_dir)

        # fix relative filepath issues for the processed dataset
        if not self.processed_data_dir.exists():
            self.processed_data_dir = Path(fix_relative_path(processed_data_dir))
            if not self.processed_data_dir.exists():
                raise FileNotFoundError(f'Could not find processed data directory at {self.processed_data_dir}')
            
        # fix relative filepath issues for the raw dataset
        if not Path(self.raw_data_dir).exists():
            self.raw_data_dir = fix_relative_path(raw_data_dir)
            if not Path(self.raw_data_dir).exists():
                raise FileNotFoundError(f'Could not find raw data directory at {raw_data_dir}')

        prot_file_names = []
        lig_rdmol_objects = []
        pharm_pos_arrs = []
        pharm_feat_arrs = []
        prot_pos_arrs = []
        prot_feat_arrs = []
        pharm_idx_arrs = []
        prot_idx_arrs = []
        prot_ph_feat_arrs = []
        prot_ph_pos_arrs= []
        prot_ph_idx_arrs = []
        for split_dir in self.processed_data_dir.iterdir():

            split_idx = int(split_dir.name.split('_')[-1][-1])
            if split_idx not in split_idxs:
                continue
            
            #get file names and rdkit objects
            with gzip.open(split_dir / 'prot_file_names.pkl.gz', 'rb') as f:
                prot_file_names.extend(pickle.load(f))
            with gzip.open(split_dir / 'lig_rdmol.pkl.gz', 'rb') as f:
                lig_rdmol_objects.extend(pickle.load(f))
            
            # get filepath of data with the tensors in it
            tensor_file = split_dir / 'prot_pharm_tensors.npz'

            # load the tensors
            data = np.load(tensor_file)

            pharm_pos_arrs.append(data['pharm_pos'])
            pharm_feat_arrs.append(data['pharm_feat'])
            prot_pos_arrs.append(data['prot_pos'])
            prot_feat_arrs.append(data['prot_feat'])
            pharm_idx_arrs.append(data['pharm_idx'])
            prot_idx_arrs.append(data['prot_idx'])
            prot_ph_feat_arrs.append(data['prot_ph_feat'])
            prot_ph_pos_arrs.append(data['prot_ph_pos'])
            prot_ph_idx_arrs.append(data['prot_ph_idx'])
        
        self.pharm_pos = np.concatenate(pharm_pos_arrs, axis=0)
        self.pharm_feat = np.concatenate(pharm_feat_arrs, axis=0)
        self.prot_pos = np.concatenate(prot_pos_arrs, axis=0)
        self.prot_feat = np.concatenate(prot_feat_arrs, axis=0)
        self.prot_ph_feat = np.concatenate(prot_ph_feat_arrs, axis=0)
        self.prot_ph_pos = np.concatenate(prot_ph_pos_arrs, axis=0)

        # convert pharm_idx_arrs to one array, but make sure the indicies are global
        self.pharm_idx = np.concatenate(pharm_idx_arrs, axis=0)
        for i in range(1, len(pharm_idx_arrs)):
            n_graphs_prev = np.sum([len(arr) for arr in pharm_idx_arrs[:i]])
            n_graphs_this_arr = len(pharm_idx_arrs[i])
            self.pharm_idx[n_graphs_prev:n_graphs_prev+n_graphs_this_arr] += self.pharm_idx[n_graphs_prev-1, 1]

        # do the same conversion for prot_idx_arrs
        self.prot_idx = np.concatenate(prot_idx_arrs, axis=0)
        for i in range(1, len(prot_idx_arrs)):
            n_graphs_prev = np.sum([len(arr) for arr in prot_idx_arrs[:i]])
            n_graphs_this_arr = len(prot_idx_arrs[i])
            self.prot_idx[n_graphs_prev:n_graphs_prev+n_graphs_this_arr] += self.prot_idx[n_graphs_prev-1, 1]
        
        # same conversion for prot_ph_idx_arrs
        self.prot_ph_idx = np.concatenate(prot_ph_idx_arrs, axis=0)
        for i in range(1, len(prot_ph_idx_arrs)):
            n_graphs_prev = np.sum([len(arr) for arr in prot_ph_idx_arrs[:i]])
            n_graphs_this_arr = len(prot_ph_idx_arrs[i])
            self.prot_ph_idx[n_graphs_prev:n_graphs_prev+n_graphs_this_arr] += self.prot_ph_idx[n_graphs_prev-1, 1]

        # convert pos, feat, and idx arrays into torch tensors
        self.pharm_pos = torch.from_numpy(self.pharm_pos)
        self.pharm_feat = torch.from_numpy(self.pharm_feat)
        self.prot_pos = torch.from_numpy(self.prot_pos)
        self.prot_feat = torch.from_numpy(self.prot_feat)
        self.pharm_idx = torch.from_numpy(self.pharm_idx)
        self.prot_idx = torch.from_numpy(self.prot_idx)
        self.prot_ph_feat = torch.from_numpy(self.prot_ph_feat)
        self.prot_ph_pos = torch.from_numpy(self.prot_ph_pos)
        self.prot_ph_idx = torch.from_numpy(self.prot_ph_idx)

        # save prot_file_names and ligand rdkit objects as class attributes
        self.prot_file_names = prot_file_names
        self.lig_rdmol_objects = lig_rdmol_objects

        super().__init__(name=name) # this has to happen last because this will call self.process()

    def __getitem__(self, i):
        
        pharm_start_idx, pharm_end_idx = self.pharm_idx[i]
        prot_start_idx, prot_end_idx = self.prot_idx[i]
        prot_ph_start_idx, prot_ph_end_idx = self.prot_ph_idx[i]

        pharm_pos = self.pharm_pos[pharm_start_idx:pharm_end_idx]
        pharm_feat = self.pharm_feat[pharm_start_idx:pharm_end_idx]
        prot_pos = self.prot_pos[prot_start_idx:prot_end_idx]
        prot_feat = self.prot_feat[prot_start_idx:prot_end_idx]
        prot_ph_pos = self.prot_ph_pos[prot_ph_start_idx:prot_ph_end_idx]
        prot_ph_feat = self.prot_ph_feat[prot_ph_start_idx:prot_ph_end_idx]

        # for diffusion (for which we only have continuous diffusion implemented) - one-hot encode categorical features
        if self.model_class == 'diffusion':
            prot_feat = one_hot(prot_feat.long(), num_classes=len(self.prot_elements)).float()
            pharm_feat = one_hot(pharm_feat.long(), num_classes=len(self.ph_type_map)).float()
        elif self.model_class == 'flow-matching':
            # for flow-matching, we keep categorical features as tokens
            pharm_feat = pharm_feat.unsqueeze(-1)
            prot_feat = prot_feat.unsqueeze(-1)

        # in either case, downstream code needs prot_ph_feat just to compute validity (complementarity)
        # and this code just assumes it is a one-hot encoded tensor
        prot_ph_feat = one_hot(prot_ph_feat.long(), num_classes=len(self.ph_type_map)).float()

        ## Subsample pharmacophore features if subsample_pharms is True
        if self.subsample_pharms:
            if len(pharm_pos) > self.subsample_min-1:
                subsample_max= min(self.subsample_max, len(pharm_pos))
                if self.subsample_min==subsample_max:
                    n_pharm_centers = self.subsample_min
                else:     
                    n_pharm_centers = random.randint(self.subsample_min, subsample_max)
                pharm_idxs = random.sample(range(len(pharm_pos)), n_pharm_centers)
                pharm_pos = pharm_pos[pharm_idxs]
                pharm_feat = pharm_feat[pharm_idxs]

        complex_graph = build_initial_complex_graph(prot_pos, prot_feat, cutoffs=self.graph_cutoffs, pharm_atom_positions=pharm_pos, pharm_atom_features=pharm_feat, prot_ph_pos=prot_ph_pos, prot_ph_feat=prot_ph_feat)

        #complex_graph.nodes['pharm'].data['h_0'] = pharm_feat
        
        #TODO turn on for hinge loss
        # for ntype in ['pharm', 'prot', 'prot_ph']:
        if self.model_class == 'diffusion':
            for ntype in ['pharm', 'prot']:
                complex_graph.nodes[ntype].data['h_0'] = complex_graph.nodes[ntype].data['h_0'].float()

        # if we are doing flow matching, sample the prior here and do alignment for positons
        if self.model_class == 'flow-matching':

            # all pharmacophore types are set to the mask token
            complex_graph['pharm'].data['h_1'] = torch.ones_like(complex_graph['pharm'].data['h_0'])*len(self.prot_elements)

            # get ground-truth pharmacophore positions
            x_0 = complex_graph['pharm'].data['x_0']

            # sample gaussian with zero COM
            x_1 = com_free_gaussian(*x_0.shape)

            # move to the center of the true pharmacophore
            x_1 += x_0.mean(dim=0, keepdim=True)

            # perform equivarint-OT alignment
            x_1 = align_prior(x_1, x_0, permutation=True, rigid_body=True)

            complex_graph.nodes['pharm'].data['x_1'] = x_1


        return complex_graph

    def __len__(self):
        return self.prot_idx.shape[0]

    def lig_atom_idx_to_element(self, element_idxs: List[int]):
        atom_elements = [ self.lig_reverse_map[element_idx] for element_idx in element_idxs ]
        return atom_elements

    #remove - there is not data_file attribute??
    # @property
    # def type_counts_file(self) -> Path:
    #     dataset_split = self.data_file.name.split('_')[0]
    #     types_file = self.data_file.parent / f'{dataset_split}_type_counts.pkl'
    #     return types_file

    #remove - there is not data_file attribute??
    # @property
    # def dataset_dir(self) -> Path:
    #     return self.data_file.parent

    #
    def get_files(self, idx: int):
        """Given an index of the dataset, return the filepath of the receptor pdb and ligand rdkit object."""

        raw_data_dir, prot_file_name, lig_rdmol = self.raw_data_dir, self.prot_file_names[idx], self.lig_rdmol_objects[idx]


        return raw_data_dir, prot_file_name, lig_rdmol


def build_initial_complex_graph(prot_atom_positions: torch.Tensor, prot_atom_features: torch.Tensor, cutoffs: dict, pharm_atom_positions: torch.Tensor = None, pharm_atom_features: torch.Tensor = None, prot_ph_pos: torch.Tensor = None, prot_ph_feat: torch.Tensor = None):

    if (pharm_atom_positions is not None) ^ (pharm_atom_features is not None):
        raise ValueError('pharmacophore position and features must be either be both supplied or both left as None')

    n_prot_atoms = prot_atom_positions.shape[0]

    if pharm_atom_positions is None:
        n_pharm_atoms = 0
    else:
        n_pharm_atoms = pharm_atom_positions.shape[0]
    
    

    # i've initialized this as an empty dict just to make clear the different types of edges in graph and their names
    no_edges = ([], [])
    graph_data = {
        ('prot', 'pp', 'prot'): no_edges,
        ('prot', 'pf', 'pharm'): no_edges,
        ('pharm', 'ff', 'pharm'): no_edges,
        ('pharm', 'fp', 'prot'): no_edges
    }

    # compute prot atom -> prot atom edges
    if cutoffs['pp'] > 0:
        pp_edges = radius_graph(prot_atom_positions, r=cutoffs['pp'], max_num_neighbors=100)
        graph_data[('prot', 'pp', 'prot')] = (pp_edges[0].cpu(), pp_edges[1].cpu())

    if prot_ph_pos is not None:
        assert prot_ph_feat is not None
        n_prot_ph_nodes = prot_ph_pos.shape[0]
    else:
        n_prot_ph_nodes = 0

    num_nodes_dict = {
        'prot': n_prot_atoms,'pharm': n_pharm_atoms, 'prot_ph': n_prot_ph_nodes
        }

    # create graph object
    g = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)

    # add pharmacophore node data
    if pharm_atom_positions is not None:
        g.nodes['pharm'].data['x_0'] = pharm_atom_positions
        g.nodes['pharm'].data['h_0'] = pharm_atom_features


    # add protein node data
    g.nodes['prot'].data['x_0'] = prot_atom_positions
    g.nodes['prot'].data['h_0'] = prot_atom_features

    # add protein pharmacophore node data
    if prot_ph_pos is not None:
        g.nodes['prot_ph'].data['x_0'] = prot_ph_pos
        g.nodes['prot_ph'].data['h_0'] = prot_ph_feat

    return g

def collate_fn(complex_graphs: list):
    # batch the graphs together
    complex_graphs = dgl.batch(complex_graphs)
    return complex_graphs

def get_dataloader(dataset: ProteinPharmacophoreDataset, batch_size: int, num_workers: int = 1, **kwargs) -> GraphDataLoader:

    dataloader = GraphDataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=num_workers, collate_fn=collate_fn, **kwargs)
    return dataloader