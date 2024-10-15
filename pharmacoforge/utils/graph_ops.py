import torch
import dgl
from typing import Dict

from pharmacoforge.utils import get_batch_info, get_edges_per_batch, copy_graph
from torch_cluster import radius_graph, knn_graph, knn, radius

def remove_com(protpharm_graphs: dgl.DGLHeteroGraph, 
               pharm_batch_idx: torch.Tensor, 
               prot_batch_idx: torch.Tensor, 
               com: str = None, 
               pharm_feat='x_t'):
    """Remove center of mass from ligand atom positions and receptor keypoint positions.

    This method can remove either the ligand COM, receptor keypoint COM or the complex COM.
    """               
    if com is None:
        raise NotImplementedError('removing COM of receptor/ligand complex not implemented')
    elif com == 'pharmacophore':
        ntype = 'pharm'
        com_feat = pharm_feat
    elif com == 'protein':
        ntype = 'prot'
        com_feat = 'x_0'
    else:
        raise ValueError(f'invalid value for com: {com=}')
    
    com = dgl.readout_nodes(protpharm_graphs, feat=com_feat, ntype=ntype, op='mean')

    protpharm_graphs.nodes['pharm'].data[pharm_feat] = protpharm_graphs.nodes['pharm'].data[pharm_feat] - com[pharm_batch_idx]
    protpharm_graphs.nodes['prot'].data['x_0'] = protpharm_graphs.nodes['prot'].data['x_0'] - com[prot_batch_idx]
    return protpharm_graphs

# TODO: better pharm edges options
def add_pharm_edges(g: dgl.DGLHeteroGraph, 
                    pharm_batch_idx, 
                    prot_batch_idx,
                    graph_config: dict) -> dgl.DGLHeteroGraph:

    batch_num_nodes, batch_num_edges = get_batch_info(g)
    batch_size = g.batch_size
    graph_types = graph_config['graph_types']

    # add pharm-pharm edges
    if graph_types['ff'] == 'knn':
        ff_k = graph_config['knn']['ff']
        ff_idxs = knn_graph(g.nodes['pharm'].data['x_t'], k=ff_k, batch=pharm_batch_idx)
    elif graph_types['ff'] == 'radius':
        ff_r = graph_config['radius']['ff']
        ff_idxs = radius_graph(g.nodes['pharm'].data['x_t'], r=ff_r, batch=pharm_batch_idx, max_num_neighbors=200)
    g.add_edges(ff_idxs[0], ff_idxs[1], etype='ff')

    # add prot-pharm edges
    if graph_types['pf'] == 'knn':
        ### Change to knn instead of knn_graph and check which idxs belong to prots and which to pharms
        pf_k = graph_config['knn']['pf']
        pf_idxs = knn(
            g.nodes['prot'].data['x_0'], 
            g.nodes['pharm'].data['x_t'], 
            k=pf_k, 
            batch_x=prot_batch_idx, 
            batch_y=pharm_batch_idx)
        # pf_idxs[0] -> pharmacophore nodes
        # pf_idxs[1] -> protein nodes
        g.add_edges(pf_idxs[1], pf_idxs[0], etype='pf')

        # add pharm-prot edges  
        g.add_edges(pf_idxs[0], pf_idxs[1], etype='fp')
    elif graph_types['pf'] == 'radius':     
        pf_r = graph_config['radius']['pf']
        pf_idxs = radius(
            x=g.nodes['prot'].data['x_0'], 
            y=g.nodes['pharm'].data['x_t'], 
            batch_x=prot_batch_idx, 
            batch_y=pharm_batch_idx, 
            r=pf_r, 
            max_num_neighbors=100)
        # pf_idxs[0] -> pharmacophore nodes
        # pf_idxs[1] -> protein nodes
        g.add_edges(pf_idxs[1], pf_idxs[0], etype='pf')

        # add pharm-prot edges  
        g.add_edges(pf_idxs[0], pf_idxs[1], etype='fp')

    # compute batch information
    batch_num_edges[('pharm', 'ff', 'pharm')] = get_edges_per_batch(ff_idxs[0], batch_size, pharm_batch_idx)
    batch_num_edges[('prot', 'pf', 'pharm')] = get_edges_per_batch(pf_idxs[0], batch_size, prot_batch_idx)
    batch_num_edges[('pharm', 'fp', 'prot')] = batch_num_edges[('prot', 'pf', 'pharm')]
    
    # update the graph's batch information
    g.set_batch_num_edges(batch_num_edges)
    g.set_batch_num_nodes(batch_num_nodes)

    return g
    
def remove_pharm_edges(g: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:

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

def translate_pharmacophore_to_init_frame(
        g:dgl.DGLHeteroGraph, 
        init_prot_com: torch.Tensor, 
        batch_idxs: Dict[str, torch.Tensor],
        normalization_constant: float = None):
    
    #make a copy of g
    g_frame=copy_graph(g,n_copies=1,batched_graph=True)[0]

    #unnormalize the features if necessary
    if normalization_constant is not None:
        g_frame.nodes['pharm'].data['h_0'] = g_frame.nodes['pharm'].data['h_0'] * normalization_constant

    #move features back to initial frame of reference
    prot_com = dgl.readout_nodes(g_frame, feat='x_0', ntype='prot', op='mean')
    delta=init_prot_com-prot_com
    delta = delta[batch_idxs['pharm']]
    g_frame.nodes['pharm'].data['x_t'] = g_frame.nodes['pharm'].data['x_t']+delta

    g_frame=g_frame.to('cpu')
    pharm_pos, pharm_feat =[],[]
    for g_i in dgl.unbatch(g_frame):
        pharm_pos.append(g_i.nodes['pharm'].data['x_t'])
        pharm_feat.append(g_i.nodes['pharm'].data['h_t'])
    return pharm_pos, pharm_feat