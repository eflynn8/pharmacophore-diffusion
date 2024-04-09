from rdkit.Chem import MolFromSmarts, MolFromPDBFile, rdmolfiles
import numpy as np
try:
    from molgrid.openbabel import pybel
except ImportError:
    from openbabel import pybel
import rdkit.Chem as Chem
from Bio.PDB import PDBParser, PDBIO, MMCIFIO
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.PDBIO import Select
from pathlib import Path
from scipy.spatial.distance import cdist

# suppress openbabel warnings
pybel.ob.obErrorLog.StopLogging()
pybel.ob.obErrorLog.SetOutputLevel(0)

def get_mol_pharm(pdb_file_path: str):
    """Given the filepath of a pdb file, return the location of pharmacophore features in the protein."""

    rdmol = rdmolfiles.MolFromPDBFile(pdb_file_path, sanitize=True)
    obmol = next(pybel.readfile("pdb", pdb_file_path))


    # define smarts strings
    #use both openbabel and rdkit to get the smarts matches
    smarts={}
    #arginine and lysine added to aromatic for cation pi interactions
    smarts['Aromatic']=["a1aaaaa1", "a1aaaa1",]
    smarts['PositiveIon'] = ['[+,+2,+3,+4]',"[$(C(N)(N)=N)]", "[$(n1cc[nH]c1)]"]
    smarts['NegativeIon'] = ['[-,-2,-3,-4]',"C(=O)[O-,OH,OX1]"]
    smarts['HydrogenAcceptor']=["[#7&!$([nX3])&!$([NX3]-*=[!#6])&!$([NX3]-[a])&!$([NX4])&!$(N=C([C,N])N)]","[$([O])&!$([OX2](C)C=O)&!$(*(~a)~a)]"]
    smarts['HydrogenDonor']=["[#7!H0&!$(N-[SX4](=O)(=O)[CX4](F)(F)F)]", "[#8!H0&!$([OH][C,S,P]=O)]","[#16!H0]"]
    smarts['Hydrophobic']=["a1aaaaa1","a1aaaa1","[$([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])&!$(**[CH3X4,CH2X3,CH1X2,F,Cl,Br,I])]",
                            "[$(*([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I])&!$(*([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I])]([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I]",
                            "[CH2X4,CH1X3,CH0X2]~[CH3X4,CH2X3,CH1X2,F,Cl,Br,I]","[$([CH2X4,CH1X3,CH0X2]~[$([!#1]);!$([CH2X4,CH1X3,CH0X2])])]~[CH2X4,CH1X3,CH0X2]~[CH2X4,CH1X3,CH0X2]",
                            "[$([S]~[#6])&!$(S~[!#6])]"]

    atoms = obmol.atoms
    pharmit_feats={}
    for key in smarts.keys():
        for smart in smarts[key]:
            obsmarts = pybel.Smarts(smart) # Matches an ethyl group
            matches = obsmarts.findall(obmol)
            for match in matches:
                positions=[]
                for idx in match:
                    positions.append(np.array(atoms[idx-1].coords))
                positions=np.array(positions).mean(axis=0)
                if key in pharmit_feats.keys():
                    pharmit_feats[key].append(positions)
                else:
                    pharmit_feats[key]=[positions]
            try: 
                smarts_mol=MolFromSmarts(smart)
                rd_matches=rdmol.GetSubstructMatches(smarts_mol,uniquify=True)
                for match in rd_matches:
                    positions=[]
                    for idx in match:
                        positions.append(np.array(atoms[idx].coords))
                    positions=np.array(positions).mean(axis=0)
                    if key in pharmit_feats.keys():
                        if positions not in pharmit_feats[key]:
                            pharmit_feats[key].append(positions)
                    else:
                        pharmit_feats[key]=[positions]
            except:
                pass
    return pharmit_feats

class PocketSelector(Select):

    def __init__(self, residues: list):
        super().__init__()
        self.residues = residues

    def accept_residue(self, residue):
        return residue in self.residues

class Unparsable(Exception):
    pass

def write_pocket_file(rec_file: Path, lig_rdmol: Chem.Mol, output_pocket_file: Path, cutoff: float = 5):

    # parse pdb file
    pdb_struct = PDBParser(QUIET=True).get_structure('', rec_file)

    # get ligand positions
    ligand_conformer = lig_rdmol.GetConformer()
    atom_positions = ligand_conformer.GetPositions()

    # get binding pocket residues
    pocket_residues = []
    for residue in pdb_struct[0].get_residues():
        if not is_aa(residue.get_resname(), standard=True):
            continue
        res_coords = np.array([a.get_coord() for a in residue.get_atoms()])
        is_pocket_residue = cdist(atom_positions, res_coords).min() < cutoff
        if is_pocket_residue:
            pocket_residues.append(residue)

    # save just the pocket residues
    pocket_selector = PocketSelector(pocket_residues)
    pdb_io = PDBIO()
    pdb_io.set_structure(pdb_struct)
    pdb_io.save(str(output_pocket_file), pocket_selector)