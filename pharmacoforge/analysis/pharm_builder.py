import torch
import dgl
from typing import List
from rdkit import Chem
from pathlib import Path
from pharmacoforge.constants import ph_idx_to_type
from pharmacoforge.analysis.validity import compute_complementarity


class SampledPharmacophore:

    _type_idx_to_elem = ['P', 'S', 'F', 'N', 'O', 'C',]

    def __init__(self, g: dgl.DGLHeteroGraph, 
                 pharm_type_map: List[str], 
                 traj_frames = None, 
                 ref_prot_file: Path = None, 
                 ref_rdkit_lig=None,
                 has_mask=False):
        self.g = g
        self._pharm_type_map = deepcopy(pharm_type_map)
        self.has_mask = has_mask


        # TODO: maybe we should suck the "make pocket file" and stuff into this class..tbd
        self.ref_prot_file = ref_prot_file
        self.ref_rdkit_lig = ref_rdkit_lig

        type_idx_to_elem = self.get_type_idx_to_elem()
        pharm_type_map = self.get_pharm_type_map()

        # unpack the final coordiantes and types of the pharmacophore
        self.ph_coords = g.nodes['pharm'].data['x_0'].cpu().clone()
        self.ph_feats_idxs = g.nodes['pharm'].data['h_0'].cpu().clone()
        if self.ph_feats_idxs.shape[1] > 1:
            self.ph_feats_idxs = self.ph_feats_idxs.argmax(-1)
        else:
            self.ph_feats_idxs = self.ph_feats_idxs.flatten()
        self.ph_types = [pharm_type_map[int(idx)] for idx in self.ph_feats_idxs]
        self.n_ph_centers = self.ph_coords.shape[0]

        # unpack trajectory frames if they were passed
        self.traj_frames = traj_frames

        # assert len(pharm_type_map) == len(self.type_idx_to_elem), f"pharm_type_map must have {len(self.type_idx_to_elem)} elements"

        # construct a map from pharmacophore type to elements, for writing to xyz files
        self.ph_type_to_elem = {pharm_type_map[i]: type_idx_to_elem[i] for i in range(len(pharm_type_map))}

    def get_pharm_type_map(self):
        if self.has_mask:
            return self._pharm_type_map + ['mask']
        else:
            return self._pharm_type_map
        
    def get_type_idx_to_elem(self):
        if self.has_mask:
            return self._type_idx_to_elem + ['Se']
        else:
            return self._type_idx_to_elem

    def pharm_to_xyz(self, pos: torch.Tensor, types: List[str]):
        out = f'{len(pos)}\n'
        for i in range(len(pos)):
            elem = self.ph_type_to_elem[types[i]]
            out += f"{elem} {pos[i, 0]:.3f} {pos[i, 1]:.3f} {pos[i, 2]:.3f}\n"
        return out


    def to_xyz_file(self, filename: str = None):

        out = self.pharm_to_xyz(self.ph_coords, self.ph_types)

        if filename is None:
            return out

        with open(filename, 'w') as f:
            f.write(out)

    def traj_to_xyz(self, filename: str = None, xt=True):

        pharm_type_map = self.get_pharm_type_map()

        if xt:
            pos_frames = self.traj_frames['x']
            feat_frames = self.traj_frames['h']
        else:
            pos_frames = self.traj_frames['x_0_pred']
            feat_frames = self.traj_frames['h_0_pred']

        if feat_frames.shape[-1] > 1:
            frame_type_idxs = feat_frames.argmax(dim=2)
        else:
            frame_type_idxs = feat_frames.squeeze(-1)

        out = ""
        n_frames = pos_frames.shape[0]
        for i in range(n_frames):
            frame_types = [pharm_type_map[int(idx)] for idx in frame_type_idxs[i]]
            out += self.pharm_to_xyz(pos_frames[i], frame_types)

        if filename is None:
            return out

        with open(filename, 'w') as f:
            f.write(out)


    def compute_n_valid_centers(self):
        prot_ph_feat = self.g.nodes['prot_ph'].data['h_0']
        prot_ph_pos = self.g.nodes['prot_ph'].data['x_0']

        # convert protein pharmacophore feature indices to types
        prot_ph_types = [ph_idx_to_type[int(idx)] for idx in prot_ph_feat.argmax(dim=1)]

        kwargs = {
            'pharm_pos': self.ph_coords,
            'pharm_types': self.ph_types,
            'prot_ph_pos': prot_ph_pos,
            'prot_ph_types': prot_ph_types
        }
        n_valid_ph_nodes = compute_complementarity(**kwargs, return_count=True)
        return n_valid_ph_nodes