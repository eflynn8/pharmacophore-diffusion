import torch
import dgl

from pharmacoforge.utils import get_batch_info, get_edges_per_batch
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

    ff_k = graph_config.get('ff_k', 0)
    pf_k = graph_config.get('pf_k', 0)
    graph_cutoffs = graph_config['graph_cutoffs']

    # add pharm-pharm edges
    if ff_k > 0:
        ff_idxs = knn_graph(g.nodes['pharm'].data['x_t'], k=ff_k, batch=pharm_batch_idx)
    else:
        ff_idxs = radius_graph(g.nodes['pharm'].data['x_t'], r=graph_cutoffs['ff'], batch=pharm_batch_idx, max_num_neighbors=200)
    g.add_edges(ff_idxs[0], ff_idxs[1], etype='ff')

    # add prot-pharm edges
    if pf_k > 0:
        ### Change to knn instead of knn_graph and check which idxs belong to prots and which to pharms
        pf_idxs = knn(g.nodes['prot'].data['x_0'], g.nodes['pharm'].data['x_t'], k=pf_k, batch_x=prot_batch_idx, batch_y=pharm_batch_idx)
        # print("VERIFY PF EDGES!")
        # print("PF Idxs: ", pf_idxs)
        ## The edge lists are flipped for knn vs radius
        g.add_edges(pf_idxs[1], pf_idxs[0], etype='pf')

        # add pharm-prot edges  
        g.add_edges(pf_idxs[0], pf_idxs[1], etype='fp')
    else:     
        pf_idxs = radius(x=g.nodes['pharm'].data['x_t'], y=g.nodes['prot'].data['x_0'], batch_x=pharm_batch_idx, batch_y=prot_batch_idx, r=graph_cutoffs['pf'], max_num_neighbors=100)
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