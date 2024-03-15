import torch.nn as nn
import dgl
import torch
from typing import Dict, List, Tuple, Union

from utils import get_batch_info, get_edges_per_batch, get_batch_idxs
from torch_cluster import radius_graph, knn_graph, knn, radius
from .gvp import GVPMultiEdgeConv, GVP

class NoisePredictionBlock(nn.Module):

    def __init__(self, in_scalar_dim: int, out_scalar_dim: int, vector_size: int, n_gvps: int = 3, intermediate_scalar_dim: int = 64):
        super().__init__()

        self.gvps = []
        for i in range(n_gvps):
            if i == n_gvps - 1:
                dim_vectors_out = 1
                dim_feats_out = intermediate_scalar_dim
                vectors_activation = nn.Identity()
            else:
                dim_vectors_out = vector_size
                dim_feats_out = in_scalar_dim
                vectors_activation = nn.Sigmoid()

            self.gvps.append(GVP(
                dim_vectors_in=vector_size,
                dim_vectors_out=dim_vectors_out,
                dim_feats_in=in_scalar_dim,
                dim_feats_out=dim_feats_out,
                vectors_activation=vectors_activation
            ))
        self.gvps = nn.Sequential(*self.gvps)

        self.to_scalar_output = nn.Linear(intermediate_scalar_dim, out_scalar_dim)

    def forward(self, pharm_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        scalars, _, vectors = pharm_data
        scalars, vectors = self.gvps((scalars, vectors))
        scalars = self.to_scalar_output(scalars)
        vectors = vectors.squeeze(1)
        return scalars, vectors
    
class PharmRecGVP(nn.Module):

    pharmacophore_edges = [
            ('pharm', 'ff', 'pharm'),
            ('prot', 'pf', 'pharm'),
        ]
    protein_edges = [
            ('pharm', 'fp', 'prot'),
            ('prot', 'pp', 'prot'),
        ]
    all_edges = pharmacophore_edges + protein_edges
    
    def __init__(self, in_scalar_dim: int, in_vector_dim: int, out_scalar_dim: int, n_convs: int = 4,
                 n_message_gvps: int = 3, n_update_gvps: int = 2, message_norm: Union[float, str, Dict] = 10, n_noise_gvps: int = 3, dropout: float = 0.0):
        
        super().__init__()

        self.conv_layers = nn.ModuleList()
        for i in range(n_convs):

            #TODO implement different edge types for a param sweep
            edge_types = self.all_edges

            self.conv_layers.append(GVPMultiEdgeConv(
                etypes=edge_types,
                scalar_size=in_scalar_dim,
                vector_size=in_vector_dim,
                n_message_gvps=n_message_gvps,
                n_update_gvps=n_update_gvps,
                message_norm=message_norm,
                dropout=dropout
            ))

            self.noise_predictor = NoisePredictionBlock(
            in_scalar_dim=in_scalar_dim,
            out_scalar_dim=out_scalar_dim,
            vector_size=in_vector_dim,
            n_gvps=n_noise_gvps
        )

    def forward(self, g: dgl.DGLHeteroGraph, node_data: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], batch_idxs: Dict[str, torch.Tensor]):

        # do message passing between ligand atoms and keypoints
        for conv_layer in self.conv_layers:
            node_data = conv_layer(g, node_data, batch_idxs)

        # predict noise on ligand atoms
        scalar_noise, vector_noise = self.noise_predictor(node_data['pharm'])
        return scalar_noise, vector_noise
    
class PharmRecDynamicsGVP(nn.Module):

    def __init__(self, n_pharm_scalars, n_prot_scalars, vector_size: int = 16, n_convs=4, n_hidden_scalars=128, act_fn=nn.SiLU,
                 message_norm=1,  graph_cutoffs: dict = {}, n_message_gvps: int = 3, n_update_gvps: int = 2, n_noise_gvps: int = 3, dropout: float = 0.0, ff_k: int = 0, pf_k: int = 0):
        
        super().__init__()
        self.graph_cutoffs = graph_cutoffs
        self.n_pharm_scalars = n_pharm_scalars
        self.n_prot_scalars = n_prot_scalars
        self.vector_size = vector_size
        self.ff_k=ff_k
        self.pf_k=pf_k

        self.pharm_encoder = nn.Sequential(
            nn.Linear(n_pharm_scalars+1, n_hidden_scalars),
            act_fn(),
            nn.LayerNorm(n_hidden_scalars)
        )

        self.prot_encoder = nn.Sequential(
            nn.Linear(n_prot_scalars+1, n_hidden_scalars),
            act_fn(),
            nn.LayerNorm(n_hidden_scalars)
        )

        self.noise_predictor = PharmRecGVP(
            in_scalar_dim=n_hidden_scalars,
            in_vector_dim=vector_size,
            out_scalar_dim=n_pharm_scalars,
            n_convs=n_convs,
            n_message_gvps=n_message_gvps,
            n_update_gvps=n_update_gvps,
            n_noise_gvps=n_noise_gvps,
            message_norm=message_norm,
            dropout=dropout
        )

    def forward(self, g: dgl.DGLHeteroGraph, timestep: torch.Tensor, batch_idxs: Dict[str, torch.Tensor]):

        pharm_batch_idx = batch_idxs['pharm']
        prot_batch_idx = batch_idxs['prot']

        with g.local_scope():

            # get initial lig and rec features from graph
            pharm_scalars = g.nodes['pharm'].data['h_0']
            prot_scalars = g.nodes['prot'].data['h_0']

            # add timestep to node features
            t_pharm = timestep[pharm_batch_idx].view(-1, 1)
            pharm_scalars = torch.concatenate([pharm_scalars, t_pharm], dim=1)
            
            t_prot = timestep[prot_batch_idx].view(-1, 1)
            prot_scalars = torch.concatenate([prot_scalars, t_prot], dim=1)

            # encode lig/kp scalars into a space of the same dimensionality
            pharm_scalars = self.pharm_encoder(pharm_scalars)
            prot_scalars = self.prot_encoder(prot_scalars)

            # set lig/kp features in graph
            g.nodes['pharm'].data['h_0'] = pharm_scalars
            g.nodes['prot'].data['h_0'] = prot_scalars
            g.nodes['pharm'].data['v_0'] = torch.zeros((pharm_scalars.shape[0], self.vector_size, 3),
                                                     device=g.device, dtype=pharm_scalars.dtype)
            #TODO: add prot vector features
            #TODO clean up redundancy
            # construct node data for noise predictor
            node_data = {}
            node_data['pharm'] = (
                pharm_scalars,
                g.nodes['pharm'].data['x_0'],
                torch.zeros((pharm_scalars.shape[0], self.vector_size, 3),
                            device=g.device, dtype=pharm_scalars.dtype)
            )
            node_data['prot'] = (
                prot_scalars,
                g.nodes['prot'].data['x_0'],
                torch.zeros((prot_scalars.shape[0], self.vector_size, 3),
                            device=g.device, dtype=prot_scalars.dtype)
            )

            # add pharm-pharm and prot<->pharm edges to graph
            self.remove_pharm_edges(g)
            g = self.add_pharm_edges(g, pharm_batch_idx, prot_batch_idx)

            # predict noise
            # eps_h, eps_x = self.noise_predictor(g, g.nodes, batch_idxs)
            eps_h, eps_x = self.noise_predictor(g, node_data, batch_idxs)

            self.remove_pharm_edges(g)

            return eps_h, eps_x
        
    def add_pharm_edges(self, g: dgl.DGLHeteroGraph, pharm_batch_idx, prot_batch_idx) -> dgl.DGLHeteroGraph:

        batch_num_nodes, batch_num_edges = get_batch_info(g)
        batch_size = g.batch_size

        # add pharm-pharm edges
        if self.ff_k > 0:
            ff_idxs = knn_graph(g.nodes['pharm'].data['x_0'], k=self.ff_k, batch=pharm_batch_idx)
        else:
            ff_idxs = radius_graph(g.nodes['pharm'].data['x_0'], r=self.graph_cutoffs['ff'], batch=pharm_batch_idx, max_num_neighbors=200)
        g.add_edges(ff_idxs[0], ff_idxs[1], etype='ff')

        # add prot-pharm edges
        if self.pf_k > 0:
            ### Change to knn instead of knn_graph and check which idxs belong to prots and which to pharms
            pf_idxs = knn(g.nodes['pharm'].data['x_0'], g.nodes['prot'].data['x_0'], k=self.pf_k, batch_x=pharm_batch_idx, batch_y=prot_batch_idx)
            print("VERIFY PF EDGES!")
            print("PF Idxs: ", pf_idxs)
        else:     
            pf_idxs = radius(x=g.nodes['pharm'].data['x_0'], y=g.nodes['prot'].data['x_0'], batch_x=pharm_batch_idx, batch_y=prot_batch_idx, r=self.graph_cutoffs['pf'], max_num_neighbors=100)
        g.add_edges(pf_idxs[0], pf_idxs[1], etype='pf')

        # add pharm-prot edges  
        g.add_edges(pf_idxs[1], pf_idxs[0], etype='fp')


        # compute batch information
        batch_num_edges[('pharm', 'ff', 'pharm')] = get_edges_per_batch(ff_idxs[0], batch_size, pharm_batch_idx)
        batch_num_edges[('prot', 'pf', 'pharm')] = get_edges_per_batch(pf_idxs[0], batch_size, prot_batch_idx)
        batch_num_edges[('pharm', 'fp', 'prot')] = batch_num_edges[('prot', 'pf', 'pharm')]
        
        # update the graph's batch information
        g.set_batch_num_edges(batch_num_edges)
        g.set_batch_num_nodes(batch_num_nodes)

        return g
    
    def remove_pharm_edges(self, g: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:

        etypes_to_remove = ['ff', 'pf', 'fp']
        
        batch_num_nodes, batch_num_edges = get_batch_info(g)

        for canonical_etype in batch_num_edges:
            if canonical_etype[1] in etypes_to_remove:
                batch_num_edges[canonical_etype] = torch.zeros_like(batch_num_edges[canonical_etype])

        for etype in etypes_to_remove:
            eids = g.edges(form='eid', etype=etype)
            g.remove_edges(eids, etype=etype)
        
        g.set_batch_num_nodes(batch_num_nodes)
        g.set_batch_num_edges(batch_num_edges)

        return g