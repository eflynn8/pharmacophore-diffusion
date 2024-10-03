import torch
import torch.nn as nn
import dgl
from typing import Dict

class FMVectorField(nn.Module):

    def __init__(self,
                 n_pharm_types: int,
                 rec_nf: int,
    ):
        super().__init__()


    def forward(self, g: dgl.DGLHeteroGraph, t: torch.Tensor, batch_idxs: Dict[str, torch.Tensor]):


        raise NotImplementedError