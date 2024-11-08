import torch
import dgl
from torch_cluster import radius_graph, knn_graph
import networkx as nx

def build_initial_complex_graph(
        prot_atom_positions: torch.Tensor, 
        prot_atom_features: torch.Tensor, 
        graph_config: dict = {},
        pharm_atom_positions: torch.Tensor = None, 
        pharm_atom_features: torch.Tensor = None, 
        prot_ph_pos: torch.Tensor = None, 
        prot_ph_feat: torch.Tensor = None
        ):

    if (pharm_atom_positions is not None) ^ (pharm_atom_features is not None):
        raise ValueError('pharmacophore position and features must be either be both supplied or both left as None')

    n_prot_atoms = prot_atom_positions.shape[0]

    if pharm_atom_positions is None:
        n_pharm_atoms = 0
    else:
        n_pharm_atoms = pharm_atom_positions.shape[0]
    
    # i've initialized this as an empty dict just to make clear the different types of edges in graph and their names
    no_edges = ([], [])
    graph_data = {
        ('prot', 'pp', 'prot'): no_edges,
        ('prot', 'pf', 'pharm'): no_edges,
        ('pharm', 'ff', 'pharm'): no_edges,
        ('pharm', 'fp', 'prot'): no_edges
    }

    # determine pp-graph type
    pp_graph_type = graph_config['graph_types']['pp']
    
    # compute prot atom -> prot atom edges
    if pp_graph_type == 'radius':
        pp_cutoff = graph_config['radius']['pp']
        pp_edges = radius_graph(prot_atom_positions, r=pp_cutoff, max_num_neighbors=100)
    elif pp_graph_type == 'knn':
        pp_k = graph_config['knn']['pp']
        pp_edges = knn_graph(prot_atom_positions, k=pp_k)
    elif pp_graph_type == 'random-knn':
        k_local = graph_config['knn']['pp']
        k_random = graph_config['krandom']['pp']
        graph_temp = graph_config['temp']['pp']
        pp_edges = build_random_knn(prot_atom_positions, k_local=k_local, k_random=k_random, graph_temp=graph_temp)
    elif pp_graph_type == 'clustered':
        pp_cutoff = graph_config['radius']['pp']
        k_local = graph_config['knn']['pp']
        pp_edges = build_clustered_graph(prot_atom_positions, cutoff=pp_cutoff, k_local=k_local)
    else:
        raise ValueError(f'pp_graph_type {pp_graph_type} not recognized')

    graph_data[('prot', 'pp', 'prot')] = (pp_edges[0], pp_edges[1])

    if prot_ph_pos is not None:
        assert prot_ph_feat is not None
        n_prot_ph_nodes = prot_ph_pos.shape[0]
    else:
        n_prot_ph_nodes = 0

    num_nodes_dict = {
        'prot': n_prot_atoms,'pharm': n_pharm_atoms, 'prot_ph': n_prot_ph_nodes
        }

    # create graph object
    g = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)

    # add pharmacophore node data
    if pharm_atom_positions is not None:
        g.nodes['pharm'].data['x_0'] = pharm_atom_positions
        g.nodes['pharm'].data['h_0'] = pharm_atom_features


    # add protein node data
    g.nodes['prot'].data['x_0'] = prot_atom_positions
    g.nodes['prot'].data['h_0'] = prot_atom_features

    # add protein pharmacophore node data
    if prot_ph_pos is not None:
        g.nodes['prot_ph'].data['x_0'] = prot_ph_pos
        g.nodes['prot_ph'].data['h_0'] = prot_ph_feat

    return g
    
def build_random_knn(x: torch.Tensor, k_local: int, k_random: int, graph_temp: float = 1.0, return_sparse: bool = True):
    """Build a semi-random KNN graph.
    
    This algorithm is based off of the random graph construction of Ingraham et al. (2023).
    Refer to this code https://github.com/generatebio/chroma/blob/929407c605013613941803c6113adefdccaad679/chroma/layers/structure/protein_graph.py#L467
    Or Appendix E in the SI of the paper.
    """
    n_nodes = x.shape[0]

    # compute pairwise distances
    D = torch.cdist(x, x)

    # make self-distances very large
    D[torch.eye(n_nodes).bool()] = 1.0e5

    # get indicies of k nearest neighbors
    local_idxs = torch.topk(D, k_local, largest=False, dim=1).indices # has shape (n_nodes, k)

    # compute random graph edge propensities
    logp_edge = -3*torch.log(D)
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(D)))
    logp_edge = logp_edge/graph_temp + gumbel_noise

    # get k largest propensities excluding those chosen by local_idxs
    local_mask = torch.zeros_like(D)
    local_mask.scatter_(1, local_idxs, 1)
    logp_edge = logp_edge - 1.0e3*local_mask
    random_idxs = torch.topk(logp_edge, k_random, largest=True, dim=1).indices

    if not return_sparse:
        return local_idxs, random_idxs

    # convert to sparse representation
    src_idxs = torch.arange(n_nodes).repeat_interleave(k_local+k_random)
    dst_idxs = torch.cat([local_idxs, random_idxs], dim=1).flatten()

    return torch.stack([src_idxs, dst_idxs], dim=0)

def build_clustered_graph(x: torch.Tensor, cutoff: float,  k_local: int):
    
    n_nodes = x.shape[0]

    # compute pairwise distances
    D = torch.cdist(x, x)

    # make self-distances very large
    D[torch.eye(n_nodes).bool()] = 1.0e5

    local_idxs_src = torch.arange(n_nodes).repeat_interleave(k_local)

    # get indicies of k nearest neighbors
    local_idxs_dst = torch.topk(D, k_local, largest=False, dim=1).indices # has shape (n_nodes, k)

    radius_edges = radius_graph(x, r=cutoff, max_num_neighbors=100)
    radius_edges = (radius_edges[0], radius_edges[1])
    graph=dgl.graph(radius_edges)

    network_x_graph = dgl.to_networkx(graph).to_undirected()
    

    centroid_indices=[]
    for c in nx.connected_components(network_x_graph):
        indices=list(c)
        positions=x[indices]
        centroid_position=positions.mean(dim=0)
        centroid_closest_idx = torch.argmin(torch.cdist(positions, centroid_position.view(1, -1)))
        centroid_idx = indices[centroid_closest_idx]
        centroid_indices.append(centroid_idx)
    
    centroid_indices = torch.tensor(centroid_indices)
    centroid_indices_src = centroid_indices.repeat_interleave(centroid_indices.shape[0])
    centroid_indices_dst = centroid_indices.repeat(centroid_indices.shape[0])

    #remove self edges
    mask = centroid_indices_src != centroid_indices_dst
    centroid_indices_src = centroid_indices_src[mask]
    centroid_indices_dst = centroid_indices_dst[mask]

    src_idxs = torch.cat([local_idxs_src, centroid_indices_src])
    dst_idxs = torch.cat([local_idxs_dst.flatten(), centroid_indices_dst])

    return torch.stack([src_idxs, dst_idxs], dim=0)