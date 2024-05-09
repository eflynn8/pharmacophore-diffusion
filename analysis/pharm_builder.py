import torch
import dgl
from typing import List
from rdkit import Chem
from pathlib import Path

class SampledPharmacophore:

    type_idx_to_elem = ['P', 'S', 'F', 'N', 'O', 'C',]

    def __init__(self, g: dgl.DGLHeteroGraph, pharm_type_map: List[str], traj_frames = None, ref_prot_file: Path = None, ref_rdkit_lig=None):
        self.g = g
        self.pharm_type_map = pharm_type_map

        # TODO: maybe we should suck the "make pocket file" and stuff into this class..tbd
        self.ref_prot_file = ref_prot_file
        self.ref_rdkit_lig = ref_rdkit_lig

        # unpack the final coordiantes and types of the pharmacophore
        self.ph_coords = g.nodes['pharm'].data['x_0']
        self.ph_feats_idxs = g.nodes['pharm'].data['h_0'].argmax(dim=1)
        self.ph_types = [self.pharm_type_map[int(idx)] for idx in self.ph_feats_idxs]
        self.n_ph_centers = self.ph_coords.shape[0]

        # unpack trajectory frames if they were passed
        if traj_frames is None:
            self.pos_frames = None
            self.feat_frames = None
        else:
            self.pos_frames, self.feat_frames = traj_frames

        assert len(pharm_type_map) == len(self.type_idx_to_elem), f"pharm_type_map must have {len(self.type_idx_to_elem)} elements"

        # construct a map from pharmacophore type to elements, for writing to xyz files
        self.ph_type_to_elem = {pharm_type_map[i]: self.type_idx_to_elem[i] for i in range(len(pharm_type_map))}

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

    def traj_to_xyz(self, filename: str = None):

        if self.pos_frames is None:
            raise ValueError("Cannot write trajectory because no trajectory frames were passed to the SampledPharmacophore object")

        out = ""
        n_frames = self.pos_frames.shape[0]
        frame_type_idxs = self.feat_frames.argmax(dim=2)
        for i in range(n_frames):
            frame_types = [self.pharm_type_map[int(idx)] for idx in frame_type_idxs[i]]
            out += self.pharm_to_xyz(self.pos_frames[i], frame_types)

        if filename is None:
            return out

        with open(filename, 'w') as f:
            f.write(out)


