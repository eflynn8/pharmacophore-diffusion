from collections import Counter
import numpy as np
from pathlib import Path
import pickle
import random
from typing import Dict, List
import torch
import matplotlib.pyplot as plt

class PharmSizeDistribution:

    def __init__(self, dataset_dir: Path, split_idxs: List[int]):
        # define filepath of data
        self.processed_data_dir: Path = Path(dataset_dir)
        
        pharm_idx_arrs = []
        prot_idx_arrs = []
        # pharm_feat_arrs = []
        for split_dir in self.processed_data_dir.iterdir():

            split_idx = int(split_dir.name.split('_')[-1][-1])
            if split_idx not in split_idxs:
                continue
            
            # get filepath of data with the tensors in it
            tensor_file = split_dir / 'prot_pharm_tensors.npz'
            # load the tensors
            data = np.load(tensor_file)
            # append index data to arrays of pharmacophore indices
            pharm_idx_arrs.append(data['pharm_idx'])
            prot_idx_arrs.append(data['prot_idx'])
            # pharm_feat_arrs.append(data['pharm_feat'])
        
        # self.pharm_feat = np.concatenate(pharm_feat_arrs, axis=0)
        
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
        
        self.sizes_dict = self.construct_n_nodes_dict()
        # print(self.sizes_dict)
        # self.plot_pharm_size_dist()
    
    def construct_n_nodes_dict(self):
        sizes = []
        prot_sizes = {}
        for i, idxs in enumerate(self.pharm_idx):
            pharm_start_idx, pharm_end_idx = idxs
            prot_start_idx, prot_end_idx = self.prot_idx[i]
            prot_size = prot_end_idx - prot_start_idx
            pharm_size = pharm_end_idx - pharm_start_idx
            ## Dict solution
            # if pharm_size < 9:
            #     if prot_size not in prot_sizes:
            #         prot_sizes[prot_size] = {}
            #     if pharm_size not in prot_sizes[prot_size]:
            #         prot_sizes[prot_size][pharm_size] = 1
            #     else:
            #         prot_sizes[prot_size][pharm_size] = prot_sizes[prot_size][pharm_size] + 1

            ## List solution
            if pharm_size < 9:
                if prot_size not in prot_sizes:
                    prot_sizes[prot_size] = []
                prot_sizes[prot_size].append(pharm_size)
        
        return prot_sizes

        # return dict(Counter(sizes))

    def plot_pharm_size_dist(self):
        plt.bar(self.sizes_dict.keys(), self.sizes_dict.values())
        plt.xlabel("Number of Pharmacophore Centers")
        plt.ylabel("Count in Train Dataset")
        plt.savefig("Pharm Size Distribution.png")


    def sample(self, n_nodes_rec: torch.Tensor, n_replicates) -> torch.Tensor:
        print("SAMPLING THE PHARMACOPHORE SIZE")

        pharm_sizes = np.ones(n_replicates)

        for i, rec in enumerate(n_nodes_rec):
            pharm_sizes[i] = random.choice(self.sizes_dict[rec])
        
        pharm_sizes = torch.from_numpy(pharm_sizes)
        return pharm_sizes
    
    def sample_uniformly(self, n_replicates) -> torch.Tensor:
        return torch.from_numpy(np.random.randint(3, 9, n_replicates))

        