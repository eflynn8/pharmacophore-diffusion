from pathlib import Path
import pickle
from typing import Dict
import torch

class PharmSizeDistribution:

    def __init__(self, dataset_dir: Path):
        #TODO: implement
        pass

    def sample(self, n_nodes_rec: torch.Tensor, n_replicates) -> torch.Tensor:
        #TODO: implement, currently just returning a tensor of 6s
        return torch.ones(n_replicates, dtype=torch.long) * 6
        