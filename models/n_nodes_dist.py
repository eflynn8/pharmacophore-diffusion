import numpy as np
from pathlib import Path
import pickle
from typing import Dict
import torch

class PharmSizeDistribution:

    def __init__(self, dataset_dir: Path):
        #TODO: implement
        pass

    def sample_uniformly(self, n_replicates) -> torch.Tensor:
        return torch.from_numpy(np.random.randint(3, 9, n_replicates))
        