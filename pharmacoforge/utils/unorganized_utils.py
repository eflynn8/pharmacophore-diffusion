import torch
from pathlib import Path
from typing import Dict, List, Tuple
import dgl

def get_batch_info(g: dgl.DGLHeteroGraph) -> Tuple[dict,dict]:
    batch_num_nodes = {}
    for ntype in g.ntypes:
        batch_num_nodes[ntype] = g.batch_num_nodes(ntype)

    batch_num_edges = {}
    for etype in g.canonical_etypes:
        batch_num_edges[etype] = g.batch_num_edges(etype)

    return batch_num_nodes, batch_num_edges

def get_edges_per_batch(edge_node_idxs: torch.Tensor, batch_size: int, node_batch_idxs: torch.Tensor):
    device = edge_node_idxs.device
    batch_idxs = torch.arange(batch_size, device=device)
    batches_with_edges, edges_per_batch = torch.unique_consecutive(node_batch_idxs[edge_node_idxs], return_counts=True)
    edges_per_batch_full = torch.zeros_like(batch_idxs)
    edges_per_batch_full[batches_with_edges] = edges_per_batch
    return edges_per_batch_full

def get_nodes_per_batch(node_idxs: torch.Tensor, batch_size: torch.Tensor, node_batch_dxs: torch.Tensor):
    return get_edges_per_batch(node_idxs, batch_size, node_batch_dxs)

def copy_graph(g: dgl.DGLHeteroGraph, n_copies: int, pharm_feats_per_copy: torch.Tensor = None, batched_graph=False) -> List[dgl.DGLHeteroGraph]:
    
    # get edge data
    e_data_dict = {}
    for etype in g.canonical_etypes:
        e_data_dict[etype] = g.edges(form='uv', etype=etype)

    # get number of nodes
    num_nodes_dict = {}
    for ntype in g.ntypes:
        num_nodes_dict[ntype] = g.num_nodes(ntype=ntype)

    # make copies of graph
    if pharm_feats_per_copy is None:
        g_copies = [ dgl.heterograph(e_data_dict, num_nodes_dict=num_nodes_dict, device=g.device) for _ in range(n_copies) ]
    else:
        g_copies = []
        for copy_idx in range(n_copies):

            num_nodes_clone = { k:v for k,v in num_nodes_dict.items() }
            num_nodes_clone['pharm'] = int(pharm_feats_per_copy[copy_idx])

            g_copies.append( dgl.heterograph(e_data_dict, num_nodes_dict=num_nodes_clone, device=g.device) )

    # if the input graph g was a batched graph, we must add the batch information into each of the copies
    if batched_graph:
        batch_num_nodes, batch_num_edges = get_batch_info(g)
        for gidx in range(n_copies):
            g_copies[gidx].set_batch_num_nodes(batch_num_nodes)
            g_copies[gidx].set_batch_num_edges(batch_num_edges)


    # transfer over ligand, receptor, and keypoint features
    # TODO: should we clone the tensors when putting them in the new graph? right now we only use this function
    # when doing sampling, so maintaining the computational graph is not important, so cloning has no effect
    for idx in range(n_copies):
        for ntype in g.ntypes:
            for feat in g.nodes[ntype].data.keys():

                if ntype == 'pharm' and pharm_feats_per_copy is not None:
                    dims = g.nodes[ntype].data[feat].shape[1:]
                    val = torch.zeros(pharm_feats_per_copy[idx], *dims, device=g.device)
                else:
                    val = g.nodes[ntype].data[feat].detach().clone()

                g_copies[idx].nodes[ntype].data[feat] = val

    # transfer over edge features
    for etype in g.canonical_etypes:
        for feat in g.edges[etype].data.keys():
            for idx in range(n_copies):
                g_copies[idx].edges[etype].data[feat] = g.edges[etype].data[feat].detach().clone()

    return g_copies

def get_batch_idxs(g: dgl.DGLHeteroGraph) -> Dict[str, torch.Tensor]:
        
    batch_size = g.batch_size
    device = g.device

    batch_idx = torch.arange(batch_size, device=device)

    # iterate over node types in complex_graphs
    batch_idxs = {}
    for ntype in g.ntypes:
        batch_idxs[ntype] = batch_idx.repeat_interleave(g.batch_num_nodes(ntype))

    return batch_idxs

def get_prot_atom_ph_type_maps(dataset_config: dict):
    # construct atom typing maps
    prot_elements = dataset_config['prot_elements']
    prot_element_map: Dict[str, int] = { element: idx for idx, element in enumerate(prot_elements) }
    ## TODO: Do we use other??
    prot_element_map['other'] = len(prot_elements)

    ph_types = dataset_config['ph_type_map']
    ph_type_map: Dict[str, int] = { element: idx for idx, element in enumerate(ph_types) }
    return prot_element_map, ph_type_map

def save_model(model, output_file: Path):
    torch.save(model.state_dict(), str(output_file))

def write_pharmacophore_file(coords_list: List[torch.tensor], atom_types_list: List[list], pharm_type_map: list , filename: str = None, ):

    type_idx_to_elem = ['P', 'S', 'F', 'N', 'O', 'C',]
    out=""
    for coords, atom_types in zip(coords_list, atom_types_list):
        assert len(coords) == len(atom_types)
        atom_types= [type_idx_to_elem[i] for i in atom_types]
        out += f"{len(coords)}\n"
        for i in range(len(coords)):
            out += f"{atom_types[i]} {coords[i, 0]:.3f} {coords[i, 1]:.3f} {coords[i, 2]:.3f}\n"

    if filename is None:
        return out

    fp = Path(filename)
    
    with open(filename, 'w') as f:
        f.write(out)

