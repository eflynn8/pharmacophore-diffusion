import torch
import dgl
from torch_cluster import radius_graph, knn_graph

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