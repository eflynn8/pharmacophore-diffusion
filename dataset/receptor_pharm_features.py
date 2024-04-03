from rdkit.Chem import MolFromSmarts, MolFromPDBFile, rdmolfiles
import numpy as np
try:
    from molgrid.openbabel import pybel
except ImportError:
    from openbabel import pybel

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