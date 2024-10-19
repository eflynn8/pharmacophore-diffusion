import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import dgl
from typing import Dict, Union, List, Tuple, Callable
from tqdm import tqdm
import dgl.function as fn

from pharmacoforge.utils.embedding import get_time_embedding
from pharmacoforge.models.gvp import GVPMultiEdgeConv, GVP, _norm_no_nan, _rbf
from pharmacoforge.utils.graph_ops import add_pharm_edges, remove_pharm_edges
from pharmacoforge.utils.ctmc_utils import purity_sampling


class FMVectorField(nn.Module):

    pharmacophore_edges = [
            ('pharm', 'ff', 'pharm'),
            ('prot', 'pf', 'pharm'),
        ]
    protein_edges = [
            ('pharm', 'fp', 'prot'),
            ('prot', 'pp', 'prot'),
        ]
    all_edges = pharmacophore_edges + protein_edges
    ntypes = ['pharm', 'prot']

    def __init__(self,
                n_pharm_types: int,
                rec_nf: int,
                node_scalar_dim: int = 64,
                node_vector_dim: int = 16,
                convs_per_update: int = 1,
                n_recycles: int = 1,
                n_updates: int = 3, 
                separate_updaters: bool = True,
                time_embedding_dim: int = 1,
                type_embedding_dim: int = 64,
                cat_temperature_schedule: Union[str, Callable, float] = 0.05,
                cat_temp_decay_max: float = 0.8,
                cat_temp_decay_a: float = 2,
                stochasticity: float = 20,
                high_confidence_threshold: float = 0.9,
                conv_config: dict = {},
                graph_config: dict = {}
    ):
        super().__init__()

        self.n_pharm_types = n_pharm_types
        self.rec_nf = rec_nf
        self.node_scalar_dim = node_scalar_dim
        self.node_vector_dim = node_vector_dim
        self.convs_per_update = convs_per_update
        self.n_recycles = n_recycles
        self.n_updates = n_updates
        self.separate_updaters = separate_updaters
        self.time_embedding_dim = time_embedding_dim
        self.type_embedding_dim = type_embedding_dim
        self.rbf_dmax = conv_config['rbf_dmax']
        self.rbf_dim = conv_config['rbf_dim']
        self.graph_config = graph_config

        # default sampling settings for cateogrical features
        self.stochasticity = stochasticity
        self.high_confidence_threshold = high_confidence_threshold


        # configure categorical feature temperature schedule
        self.cat_temp_func = cat_temperature_schedule
        self.cat_temp_decay_max = cat_temp_decay_max
        self.cat_temp_decay_a = cat_temp_decay_a
        self.cat_temp_func = self.build_cat_temp_schedule(
            cat_temperature_schedule=cat_temperature_schedule,
            cat_temp_decay_max=cat_temp_decay_max,
            cat_temp_decay_a=cat_temp_decay_a)

        self.scalar_embedding_fns = nn.ModuleDict({})


        self.conv_layers = nn.ModuleList([])
        for conv_idx in range(convs_per_update*n_updates):
            self.conv_layers.append(GVPMultiEdgeConv(
                etypes=self.all_edges,
                scalar_size=node_scalar_dim,
                vector_size=node_vector_dim,
                **conv_config
            ))

        # fish cp_dim out of conv_config
        cp_dim = conv_config['cp_dim']

        # create molecule update layers
        self.node_position_updaters = nn.ModuleList([])
        if self.separate_updaters:
            n_updaters = n_updates
        else:
            n_updaters = 1
        for _ in range(n_updaters):
            self.node_position_updaters.append(NodePositionUpdate(node_scalar_dim, node_vector_dim, n_gvps=3, n_cp_feats=cp_dim))


        # create embedding layers for pharmacophore and protein atom types
        self.type_embeddings = {}
        self.num_types_dict = {
            'pharm': n_pharm_types,
            'prot': rec_nf
        }
        for node_type in self.ntypes:
            num_types = self.num_types_dict[node_type] + 1 # +1 for mask token
            self.type_embeddings[node_type] = nn.Embedding(num_types, type_embedding_dim)
        self.type_embedddings = nn.ModuleDict(self.type_embeddings)

        # create embedding functions for scalar features
        self.node_embed_fns = {}
        for ntype in self.ntypes:
            self.node_embed_fns[ntype] = nn.Sequential(
                nn.Linear(type_embedding_dim+time_embedding_dim, node_scalar_dim),
                nn.SiLU(),
                nn.Linear(node_scalar_dim, node_scalar_dim),
                nn.SiLU(),
                nn.LayerNorm(node_scalar_dim)
            )
        self.node_embed_fns = nn.ModuleDict(self.node_embed_fns)

        mid_layer_size = (node_scalar_dim+self.n_pharm_types)//2
        self.pharm_scalar_readout = nn.Sequential(
            nn.Linear(node_scalar_dim, node_scalar_dim),
            nn.SiLU(),
            nn.Linear(node_scalar_dim, self.n_pharm_types),
        )

    def build_cat_temp_schedule(self, cat_temperature_schedule, cat_temp_decay_max, cat_temp_decay_a):

        if cat_temperature_schedule == 'decay':
            cat_temp_func = lambda t: cat_temp_decay_max*torch.pow(1-t, cat_temp_decay_a)
        elif isinstance(cat_temperature_schedule, (float, int)):
            cat_temp_func = lambda t: cat_temperature_schedule
        elif callable(cat_temperature_schedule):
            cat_temp_func = cat_temperature_schedule
        else:
            raise ValueError(f"Invalid cat_temperature_schedule: {cat_temperature_schedule}")
        
        return cat_temp_func

    def forward(self, g: dgl.DGLHeteroGraph, 
                t: torch.Tensor, 
                batch_idxs: Dict[str, torch.Tensor],
                apply_softmax: bool = False):

        # remove pharm edges, add new ones
        g = remove_pharm_edges(g)
        g = add_pharm_edges(g, batch_idxs['pharm'], batch_idxs['prot'], self.graph_config)

        # get scalar feature embeddings
        scalar_feats = {}
        scalar_feats['prot'] = [
            self.type_embeddings['prot'](g.nodes['prot'].data['h_0'].squeeze(-1))
        ]
        scalar_feats['pharm'] = [
            self.type_embeddings['pharm'](g.nodes['pharm'].data['h_t'].squeeze(-1))
        ]
        for ntype in scalar_feats:
            t_batch = t[batch_idxs[ntype]]
            time_embedding = get_time_embedding(t_batch, self.time_embedding_dim)
            scalar_feats[ntype].append(time_embedding)
            scalar_feats[ntype] = self.node_embed_fns[ntype](
                torch.cat(scalar_feats[ntype], dim=-1)
            )

        # get position features
        pos_feats = {
            'prot': g.nodes['prot'].data['x_0'],
            'pharm': g.nodes['pharm'].data['x_t']
        }

        # get vector features
        vec_feats = {}
        for ntype in self.ntypes:
            vec_feats[ntype] = torch.zeros(
                (g.num_nodes(ntype), self.node_vector_dim, 3),
                device=g.device, 
                dtype=scalar_feats[ntype].dtype
            )

        # precompute distances
        x_diff, d = self.precompute_distances(g, node_positions=pos_feats)

        # repack features so they can go into the conv layers
        conv_feats = {}
        for ntype in self.ntypes:
            conv_feats[ntype] = [scalar_feats[ntype], pos_feats[ntype], vec_feats[ntype]]

        for recycle_idx in range(self.n_recycles):
            for conv_idx, conv in enumerate(self.conv_layers):
                conv_feats = conv(g, conv_feats, x_diff, d)

                if (conv_idx+1) % self.convs_per_update == 0:
                    if self.separate_updaters:
                        update_idx = conv_idx // self.convs_per_update
                    else:
                        update_idx = 0

                    # update positions
                    pharm_pos = self.node_position_updaters[update_idx](*conv_feats['pharm'])
                    conv_feats['pharm'][1] = pharm_pos

                    # recompute distances
                    node_positions = {
                        key: conv_feats[key][1] for key in self.ntypes
                    }
                    x_diff, d = self.precompute_distances(g, node_positions=node_positions)
        

        # readout pharmacophore types
        ph_type_logits = self.pharm_scalar_readout(conv_feats['pharm'][0])
        if apply_softmax:
            ph_type_logits = torch.softmax(ph_type_logits, dim=-1)

        # collect outputs
        dst_dict = {
            'h': ph_type_logits,
            'x': conv_feats['pharm'][1]
        }

        # remove pharm edges
        remove_pharm_edges(g)

        return dst_dict
    
    def precompute_distances(self, g: dgl.DGLGraph, node_positions: Dict[str, torch.Tensor]=None):
        """Precompute the pairwise distances between all nodes in the graph."""

        with g.local_scope():

            for etype in self.all_edges:
                src_ntype, _, dst_ntype = etype

                if node_positions is None:
                    g.nodes[src_ntype].data['x_d'] = g.nodes[src_ntype].data['x_t']
                    g.nodes[dst_ntype].data['x_d'] = g.nodes[dst_ntype].data['x_t']
                else:
                    g.nodes[src_ntype].data['x_d'] = node_positions[src_ntype]
                    g.nodes[dst_ntype].data['x_d'] = node_positions[dst_ntype]

                g.apply_edges(fn.u_sub_v("x_d", "x_d", "x_diff"), etype=etype)
                dij = _norm_no_nan(g.edges[etype].data['x_diff'], keepdims=True) + 1e-8
                x_diff = g.edges[etype].data['x_diff'] / dij
                d = _rbf(dij.squeeze(1), D_max=self.rbf_dmax, D_count=self.rbf_dim)
        
        return x_diff, d
    
    def integrate(self, 
        g: dgl.DGLGraph, 
        batch_idxs: Dict[str, torch.Tensor],
        n_timesteps: int, 
        visualize=False, 
        stochasticity=None, 
        high_confidence_threshold=None,
        cat_temp_func=None,
        tspan=None):
        """Integrate the trajectories of molecules along the vector field."""
        if cat_temp_func is None:
            cat_temp_func = self.cat_temp_func
        if stochasticity is None:
            stochasticity = self.stochasticity
        if high_confidence_threshold is None:
            high_confidence_threshold = self.high_confidence_threshold


        # get the timepoint for integration
        if tspan is None:
            t = torch.linspace(0, 1, n_timesteps, device=g.device)
        else:
            t = tspan

        # get the corresponding alpha values for each timepoint
        # TODO: implement interpolant scheduler
        kappa_t = torch.stack([t,t], dim=-1) # has shape (n_timetsteps, n_features)
        kappa_t_prime = torch.ones_like(kappa_t)

        # set x_t = x_0 (flow matching time in comments but diffusion time in data keys)
        g.nodes['pharm'].data['x_t'] = g.nodes['pharm'].data['x_1']
        g.nodes['pharm'].data['h_t'] = g.nodes['pharm'].data['h_1']

        # if visualizing the trajectory, create a datastructure to store the trajectory
        if visualize:
            traj_frames = {}
            for feat in 'xh':
                split_sizes = g.batch_num_nodes(ntype='pharm')
                split_sizes = split_sizes.detach().cpu().tolist()
                init_frame = g.nodes['pharm'].data[f'{feat}_1'].detach().cpu()
                init_frame = torch.split(init_frame, split_sizes)
                traj_frames[feat] = [ init_frame ]
                traj_frames[f'{feat}_0_pred'] = []
    
        for s_idx in range(1,t.shape[0]):

            # get the next timepoint (s) and the current timepoint (t)
            s_i = t[s_idx]
            t_i = t[s_idx - 1]
            kappa_t_i = kappa_t[s_idx - 1]
            kappa_s_i = kappa_t[s_idx]
            kappa_t_prime_i = kappa_t_prime[s_idx - 1]

            # determine if this is the last integration step
            if s_idx == t.shape[0] - 1:
                last_step = True
            else:
                last_step = False

            # compute next step and set x_t = x_s
            g = self.step(g, s_i, t_i, kappa_t_i, kappa_s_i, 
                kappa_t_prime_i, 
                batch_idxs=batch_idxs,
                cat_temp_func=cat_temp_func,
                stochasticity=stochasticity, 
                high_confidence_threshold=high_confidence_threshold,
                last_step=last_step)

            if visualize:
                for feat in 'xh':

                    frame = g.nodes['pharm'][f'{feat}_t'].detach().cpu()
                    split_sizes = g.batch_num_nodes(ntype='pharm')
                    split_sizes = split_sizes.detach().cpu().tolist()
                    frame = g.nodes['pharm'].data[f'{feat}_t'].detach().cpu()
                    frame = torch.split(frame, split_sizes)
                    traj_frames[feat].append(frame)

                    ep_frame = g.nodes['pharm'].data[f'{feat}_0_pred'].detach().cpu()
                    ep_frame = torch.split(ep_frame, split_sizes)
                    traj_frames[f'{feat}_0_pred'].append(ep_frame)

        # set x_1 = x_t
        for feat in 'xh':
            g.nodes['pharm'].data[f'{feat}_0'] = g.nodes['pharm'].data[f'{feat}_t']

        if visualize:

            # currently, traj_frames[key] is a list of lists. each sublist contains the frame for every molecule in the batch
            # we want to rearrange this so that traj_frames is a list of dictionaries, where each dictionary contains the frames for a single molecule
            reshaped_traj_frames = []
            for mol_idx in range(g.batch_size):
                molecule_dict = {}
                for feat in traj_frames.keys():
                    feat_traj = []
                    n_frames = len(traj_frames[feat])
                    for frame_idx in range(n_frames):
                        feat_traj.append(traj_frames[feat][frame_idx][mol_idx])
                    molecule_dict[feat] = torch.stack(feat_traj)
                reshaped_traj_frames.append(molecule_dict)
        
        if visualize:
            return g, reshaped_traj_frames
        else:
            return g
        

    def step(self, 
             g: dgl.DGLGraph, 
             s_i: torch.Tensor, 
             t_i: torch.Tensor,
             kappa_t_i: torch.Tensor, 
             kappa_s_i: torch.Tensor, 
             kappa_t_prime_i: torch.Tensor,
             batch_idxs: Dict[str, torch.Tensor],
             cat_temp_func: Callable,
             stochasticity: float,
             high_confidence_threshold: float, 
             last_step: bool = False,):

        device = g.device

        eta = stochasticity
        hc_thresh = high_confidence_threshold
        
        # predict the destination of the trajectory given the current timepoint
        dst_dict = self(
            g, 
            t=torch.full((g.batch_size,), t_i, device=g.device),
            batch_idxs=batch_idxs,
            apply_softmax=True,
        )
        
        dt = s_i - t_i

        # take integration step for positions
        x_1 = dst_dict['x']
        x_t = g.nodes['pharm'].data['x_t']
        vf = self.vector_field(x_t, x_1, kappa_t_i[0], kappa_t_prime_i[0])
        g.nodes['pharm'].data['x_t'] = x_t + dt*vf

        # record predicted endpoint for visualization
        g.nodes['pharm'].data['x_0_pred'] = x_1.detach().clone()

        # take integration step for pharmacphore types
        ht = g.nodes['pharm'].data[f'h_t'] # has shape (num_nodes,1)

        p_s_1 = dst_dict['h']
        temperature = cat_temp_func(t_i)
        p_s_1 = F.softmax(torch.log(p_s_1)/temperature, dim=-1) # log probabilities

        ht, h_1_sampled = \
        self.campbell_step(p_1_given_t=p_s_1, 
                        xt=ht, 
                        stochasticity=eta, 
                        hc_thresh=hc_thresh, 
                        kappa_t=kappa_t_i[1], 
                        kappa_t_prime=kappa_t_prime_i[1],
                        dt=dt, 
                        batch_size=g.batch_size, 
                        batch_num_nodes=g.batch_num_nodes('pharm'), 
                        n_classes=self.n_pharm_types+1,
                        mask_index=self.n_pharm_types,
                        last_step=last_step,
                        batch_idx=batch_idxs['pharm'],
                        )
        
        # record the updated pharmacophore types
        g.nodes['pharm'].data[f'h_t'] = ht

        # record predicted final pharmacophore types for visualization
        g.nodes['pharm'].data[f'h_0_pred'] = h_1_sampled

        return g

        
    def campbell_step(self, p_1_given_t: torch.Tensor,
                      xt: torch.Tensor, 
                      stochasticity: float, 
                      hc_thresh: float, 
                      kappa_t: float, 
                      kappa_t_prime: float,
                      dt,
                      batch_size: int,
                      batch_num_nodes: torch.Tensor,
                      n_classes: int,
                      mask_index:int,
                      last_step: bool, 
                      batch_idx: torch.Tensor,
):
        x1 = Categorical(p_1_given_t).sample() # has shape (num_nodes,)

        unmask_prob = dt*( kappa_t_prime + stochasticity*kappa_t  ) / (1 - kappa_t)
        mask_prob = dt*stochasticity

        unmask_prob = torch.clamp(unmask_prob, min=0, max=1)
        mask_prob = torch.clamp(mask_prob, min=0, max=1)

        # sample which nodes will be unmasked
        if hc_thresh > 0:
            # select more high-confidence predictions for unmasking than low-confidence predictions
            will_unmask = purity_sampling(
                xt=xt.squeeze(-1), 
                x1=x1, 
                x1_probs=p_1_given_t, 
                unmask_prob=unmask_prob,
                mask_index=mask_index, batch_size=batch_size, batch_num_nodes=batch_num_nodes,
                node_batch_idx=batch_idx, hc_thresh=hc_thresh, device=xt.device)
        else:
            # uniformly sample nodes to unmask
            will_unmask = torch.rand(xt.shape[0], device=xt.device) < unmask_prob
            will_unmask = will_unmask * (xt == mask_index) # only unmask nodes that are currently masked

        if not last_step:
            # compute which nodes will be masked
            will_mask = torch.rand(xt.shape[0], 1, device=xt.device) < mask_prob
            will_mask = will_mask * (xt != mask_index) # only mask nodes that are currently unmasked

            # mask the nodes
            xt[will_mask] = mask_index

        # unmask the nodes
        xt[will_unmask] = x1[will_unmask].unsqueeze(-1)

        return xt, x1
    
    def vector_field(self, x_t, x_1, kappa_t, kappa_t_prime):
        vf = kappa_t_prime/(1 - kappa_t) * (x_1 - x_t)
        return vf
  

class NodePositionUpdate(nn.Module):

    def __init__(self, n_scalars, n_vec_channels, n_gvps: int = 3, n_cp_feats: int = 0):
        super().__init__()

        self.gvps = []
        for i in range(n_gvps):

            if i == n_gvps - 1:
                vectors_out = 1
                vectors_activation = nn.Identity()
            else:
                vectors_out = n_vec_channels
                vectors_activation = nn.Sigmoid()

            self.gvps.append(
                GVP(
                    dim_feats_in=n_scalars,
                    dim_feats_out=n_scalars,
                    dim_vectors_in=n_vec_channels,
                    dim_vectors_out=vectors_out,
                    n_cp_feats=n_cp_feats,
                    vectors_activation=vectors_activation,
                    vector_gating=True,
                )
            )
        self.gvps = nn.Sequential(*self.gvps)

    def forward(self, scalars: torch.Tensor, positions: torch.Tensor, vectors: torch.Tensor):
        _, vector_updates = self.gvps((scalars, vectors))
        return positions + vector_updates.squeeze(1)