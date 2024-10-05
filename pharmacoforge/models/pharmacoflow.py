import torch.nn as nn
import torch
import dgl
from typing import List, Dict
from pathlib import Path

from pharmacoforge.utils import get_batch_info, get_nodes_per_batch, copy_graph, get_batch_idxs
from pharmacoforge.utils.graph_ops import remove_com
from pharmacoforge.models.fm_vector_field import FMVectorField

class PharmacoFlow(nn.Module):

    def __init__(self,
        n_pharm_types: int,  # number of pharmacophore types
        rec_nf: int,  # number of scalar features for receptor nodes
        ph_type_map: List[str], 
        processed_data_dir: Path,
        time_scaled_loss: bool = False,
        graph_config: dict = {},
        vf_config: dict = {},                  
    ):
        super().__init__()

        self.n_pharm_types = n_pharm_types
        self.rec_nf = rec_nf
        self.ph_type_map = ph_type_map  
        self.processed_data_dir = processed_data_dir
        self.graph_config = graph_config
        self.vf_config = vf_config
        self.time_scaled_loss = time_scaled_loss

        self.vector_field = FMVectorField(
                            n_pharm_types=n_pharm_types,
                            rec_nf=rec_nf,
                            graph_config=self.graph_config,
                            **vf_config)
        

        # configure loss functions
        if self.time_scaled_loss:
            reduction = 'none'
        else:
            reduction = 'mean'

        self.loss_ignore_idx = -10
        self.loss_fns = {
            'h': nn.CrossEntropyLoss(reduction=reduction, ignore_index=self.loss_ignore_idx),
            'x': nn.MSELoss(reduction=reduction),
        }

    
    def forward(self, g: dgl.DGLHeteroGraph):
        # get batch size and device
        batch_size = g.batch_size
        device = g.device

        # get batch info
        batch_idxs = get_batch_idxs(g)

        # sample t for each graph in the batch
        t = torch.rand(batch_size, device=device, dtype=float)

        # sample p_t(g|g_0,g_1)
        g = self.sample_conditional_path(g, t, batch_idxs)

        # predict final pharmacophore
        dst_dict = self.vector_field(g, t)

        time_weights = self.loss_weights(t)
        if self.time_scaled_loss:
            time_weights = time_weights[batch_idxs['pharm']]

        losses = {}
        for key in self.loss_fns:

            target = g.nodes['pharm'].data[f'{key}_0']

            # set pharmacophore centers in x_t that were unmasked to the ignore index
            if key == 'h':
                xt = g.nodes['pharm'].data['h_t']
                unmasked_at_t = xt != len(self.ph_type_map)
                target[unmasked_at_t] = self.loss_ignore_idx

            losses[key] = time_weights * self.loss_fns[key](dst_dict[key], target)

            # if time_scaled_loss is True, we need to do the reduction ourselves
            if self.time_scaled_loss:
                losses[key] = losses[key].mean()

        # TODO: fix loss names

        return losses

    def sample_conditional_path(self, g: dgl.DGLHeteroGraph, t: torch.Tensor, batch_idxs: Dict[str, torch.Tensor]) -> dgl.DGLHeteroGraph:
        # we assume t is flow-matching time
        device = g.device

        # get interpolant value
        # TODO: implement general interpolant schedules, modality-dependent schedules
        kappa = t.unsqueeze(1) # has shape (batch_size, 1)
        kappa_h = kappa[batch_idxs['pharm']]
        kappa_x = kappa_h

        # sample conditional path for pharmacophore types
        h_t = g.nodes['pharm'].data['h_0']
        is_masked = torch.rand(g.num_nodes('pharm'), device=device) > kappa_h
        h_t[is_masked] = self.n_pharm_types
        g.nodes['pharm'].data['h_t'] = h_t

        # sample conditional path for pharmacophore positons
        # this is confusing: graph data keys use diffusion time, but we are using flow-matching time
        x_0 = g.nodes['pharm'].data['x_1']
        x_1 = g.nodes['pharm'].data['x_0']
        x_t = (1 - kappa_x) * x_0 + kappa_x * x_1
        g.nodes['pharm'].data['x_t'] = x_t

        return g
    
    def loss_weights(self, t):

        if not self.time_scaled_loss:
            return 1
        
        # TODO: implement abitrary interpolant schedule
        kappa = t
        kappa_prime = 1
        weights = kappa_prime / (1 - kappa)
        weights = torch.clamp(weights, min=0.05, max=1.5)


        