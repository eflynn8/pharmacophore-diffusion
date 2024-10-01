import torch
import dgl

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