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

class ProtPharmDataModule(pl.LightningDataModule):
    def __init__(self,
                dataset: ProteinPharmacophoreDataset, 
                 batch_size: int,
                 num_workers: int,
                 **kwargs):
        
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def train_dataloader(self):
        return get_dataloader(self.dataset, self.batch_size, self.num_workers)
