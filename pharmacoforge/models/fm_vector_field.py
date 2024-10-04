import torch
import torch.nn as nn
import dgl
from typing import Dict, Union, List, Tuple

from pharmacoforge.utils.embedding import get_time_embedding
from pharmacoforge.models.gvp import GVPMultiEdgeConv, GVP, _norm_no_nan, _rbf

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
                conv_config: dict = {},
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
        self.type_embeddings = nn.ModuleDict({})
        self.num_types_dict = {
            'pharm': n_pharm_types,
            'prot': rec_nf
        }
        for node_type in self.ntypes:
            num_types = self.num_types_dict[node_type] + 1 # +1 for mask token
            self.type_embeddings[node_type] = nn.Embedding(num_types, type_embedding_dim)

        # create embedding functions for scalar features
        self.node_embed_fns = nn.ModuleDict({})
        for ntype in self.ntypes:
            self.node_embed_fns[ntype] = nn.Sequential(
                nn.Linear(type_embedding_dim+time_embedding_dim, node_scalar_dim),
                nn.SiLU(),
                nn.Linear(node_scalar_dim, node_scalar_dim),
                nn.SiLU(),
                nn.LayerNorm(node_scalar_dim)
            )

        mid_layer_size = (node_scalar_dim+self.n_pharm_types)//2
        self.pharm_scalar_readout = nn.Sequential(
            nn.Linear(node_scalar_dim, node_scalar_dim),
            nn.SiLU(),
            nn.Linear(node_scalar_dim, mid_layer_size),
            nn.SiLU(),
            nn.Linear(mid_layer_size, self.n_pharm_types)
        )



    def forward(self, g: dgl.DGLHeteroGraph, 
                t: torch.Tensor, 
                batch_idxs: Dict[str, torch.Tensor],
                apply_softmax: bool = False):

        # TODO: add edges to graph

        # get scalar feature embeddings
        scalar_feats = {}
        scalar_feats['prot'] = [
            self.type_embeddings['prot'](g.nodes['prot'].data['h_0'])
        ]
        scalar_feats['pharm'] = [
            self.type_embeddings['pharm'](g.nodes['pharm'].data['h_t'])
        ]
        for ntype in scalar_feats:
            scalar_feats[ntype] = self.node_embed_fns[ntype](
                torch.cat([scalar_feats[ntype], get_time_embedding(t, self.time_embedding_dim)], dim=-1)
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
        


        ph_type_logits = self.pharm_scalar_readout(conv_feats['pharm'][0])
        if apply_softmax:
            ph_type_logits = torch.softmax(ph_type_logits, dim=-1)

        dst_dict = {
            'h': ph_type_logits,
            'x': conv_feats['pharm'][1]
        }

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
                dij = _norm_no_nan(g.edata['x_diff'], keepdims=True) + 1e-8
                x_diff = g.edata['x_diff'] / dij
                d = _rbf(dij.squeeze(1), D_max=self.rbf_dmax, D_count=self.rbf_dim)
        
        return x_diff, d
    

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