import torch
import dgl
from torch_cluster import radius_graph

def build_initial_complex_graph(
        prot_atom_positions: torch.Tensor, 
        prot_atom_features: torch.Tensor, 
        cutoffs: dict, 
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

    # compute prot atom -> prot atom edges
    if cutoffs['pp'] > 0:
        pp_edges = radius_graph(prot_atom_positions, r=cutoffs['pp'], max_num_neighbors=100)
        graph_data[('prot', 'pp', 'prot')] = (pp_edges[0].cpu(), pp_edges[1].cpu())

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