from pathlib import Path
import pickle
from typing import Dict, List, Union
import math

import dgl
from dgl.dataloading import GraphDataLoader
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
import random
from dataset.protein_pharm_dataset import ProteinPharmacophoreDataset, get_dataloader, collate_fn
from typing import List

class CrossdockedDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_config: dict, 
                 batch_size: int,
                 num_workers: int,
                 validation_splits: List[int] = [],):
        
        super().__init__()
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.num_workers = num_workers

        if len(validation_splits) == 0:
            raise NotImplementedError("training without a validation split has not yet been implemented")

        if len(validation_splits) >= 3:
            raise ValueError("validation split indices must be a subset of [0, 1, 2]")

        # check that provided split indicies are valid
        for split_idx in validation_splits:
            if split_idx not in [0, 1, 2]:
                raise ValueError("validation split index must be 0, 1, or 2")

        split_idxs = [0, 1, 2]
        train_split_idxs = []
        val_split_idxs = []
        for split_idx in split_idxs:
            if split_idx not in validation_splits:
                train_split_idxs.append(split_idx)
            else:
                val_split_idxs.append(split_idx)

        # record the splits which we are using for training and validation
        self.train_split_idxs = train_split_idxs
        self.val_split_idxs = val_split_idxs
        

    def setup(self, stage: str='fit'):

        if stage == 'fit':
            self.train_dataset = ProteinPharmacophoreDataset(name='train', split_idxs=self.train_split_idxs, **self.dataset_config)
            self.val_dataset = ProteinPharmacophoreDataset(name='val', split_idxs=self.val_split_idxs, **self.dataset_config)
        if stage == 'test':
            self.val_dataset = ProteinPharmacophoreDataset(name='val', split_idxs=self.val_split_idxs, **self.dataset_config)
    
    def train_dataloader(self):
        return get_dataloader(self.train_dataset, self.batch_size, self.num_workers)

    def val_dataloader(self):
        return get_dataloader(self.val_dataset, self.batch_size, self.num_workers)
