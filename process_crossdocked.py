import re, subprocess, os, gzip, json, glob, multiprocessing
import numpy as np
from rdkit.Chem import AllChem as Chem
import tempfile
import pickle
from tqdm import tqdm
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB import PDBIO
from scipy.spatial.distance import cdist
import Bio
import Bio.SeqUtils
import argparse
from pathlib import Path
import yaml
from functools import partial
from constants import ph_type_to_idx
from tqdm.contrib.concurrent import process_map
from typing import Dict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to config file", required=True, type=Path)
    parser.add_argument('--max_workers', type=int, default=None, help='Number of workers to use for multiprocessing, default to max available.')
    args = parser.parse_args()
    return args

def element_fixer(element: str):

    if len(element) > 1:
        element = element[0] + element[1:].lower()
    
    return element


def getfeatures(reclig, crossdocked_data_dir: Path, pocket_cutoff: int = 8):
    
    # reclig is a tuple of length 2. The first entry is the receptor file name and the second entry is the ligand file name

    rec,glig = reclig
    rec = rec.replace('_0.gninatypes','.pdb')            
    m = re.search(r'(\S+)_(\d+)\.gninatypes',glig)
    prefix = m.group(1)
    num = int(m.group(2))
    lig = prefix+'.sdf.gz'
    
    rec_path = crossdocked_data_dir / rec
    lig_path = crosdocked_data_dir / lig
    rec_path = str(rec_path)
    lig_path = str(lig_path)
    
    # why are we doing this? should we instead print some warning? something more informative?
    if not os.path.exists(rec_path):
        print(rec_path)
    if not os.path.exists(lig_path):
        print(lig_path)

    with tempfile.TemporaryDirectory() as tmp:
        try:
            if num != 0:
                #extract num conformer
                #avoid chemical parsing for speed reasons
                sdf = gzip.open(lig_path).read().split(b'$$$$\n')[num]+b'$$$$\n'
                lig_path = os.path.join(tmp,'lig.sdf')
                with open(lig_path, 'wb') as out:
                    out.write(sdf)

            phfile = os.path.join(tmp,"ph.json")
            cmd = f'pharmit pharma -receptor {rec_path} -in {lig_path} -out {phfile}'
            subprocess.check_call(cmd,shell=True)

            #some files have another json object in them - only take first
            #in actuality, it is a bug with how pharmit/openbabel is dealing
            #with gzipped sdf files that causes only one molecule to be read
            decoder = json.JSONDecoder()
            ph = decoder.raw_decode(open(phfile).read())[0]

            if ph['points']:
                feature_coords = np.array([(p['x'],p['y'],p['z']) for p in ph['points'] if p['enabled']])
                feature_kind = np.array([ph_type_to_idx[p['name']] for p in ph['points'] if p['enabled']])
            else:
                feature_coords = []
                feature_kind = []
                
            #extract pocket using Ian's approach
            pdb_struct = PDBParser(QUIET=True).get_structure('', rec_path)
            if lig_path.endswith('.gz'):
                with gzip.open(lig_path) as f:
                    supp = Chem.ForwardSDMolSupplier(f,sanitize=False)
                    ligand = next(supp)
                    del supp
                
            else:
                supp = Chem.ForwardSDMolSupplier(lig_path,sanitize=False)
                ligand = next(supp)
                del supp

            lig_coords = ligand.GetConformer().GetPositions()
             # get residues which constitute the binding pocket
            pocket_residues = []
            for residue in pdb_struct[0].get_residues():
                # get atomic coordinates of residue
                res_coords = np.array([a.get_coord() for a in residue.get_atoms()])

                # skip nonstandard residues
                is_residue = is_aa(residue.get_resname(), standard=True)
                if not is_residue:
                    continue
                min_rl_dist = cdist(lig_coords, res_coords).min()
                if min_rl_dist < pocket_cutoff:
                    pocket_residues.append(residue)

            atom_filter = lambda a: a.element != "H"
            pocket_atomres = []
            for res in pocket_residues:
                atom_list = res.get_atoms()
                pocket_atomres.extend([(a, res) for a in atom_list if atom_filter(a) ])

            pocket_atoms, atom_residues = list(map(list, zip(*pocket_atomres)))

            pocket_coords = np.array([ar[0].get_coord() for ar in pocket_atomres])
            pocket_elements = np.array([ element_fixer(ar[0].element) for ar in pocket_atomres ])
            pocket_anames = np.array([ar[0].name for ar in pocket_atomres])
            pocket_res = np.array([Bio.PDB.Polypeptide.three_to_index(ar[1].resname) for ar in pocket_atomres])
            pocket_rid = np.array([ar[1].id[1] for ar in pocket_atomres])
            #receptor and features are needed for training
            #glig is included for reference back to original data for debugging

            # rec is the filepath of the receptor pdb file
            # glig is the filepath of the ligand gninatypes file
            # ligand is the ligand molecule as an rdkit object
            # feature_coords is the cartesian coordinates of the pharmacophore centers
            # feature_kind is the integer type of each pharmacophore center

            return ((rec,glig,ligand,(feature_coords, feature_kind),(pocket_coords, pocket_elements, pocket_anames, pocket_res, pocket_rid)))    
        except Exception as e:
            print(e)
            print(rec,glig)
            return((rec,glig,None,None,None))
        
def write_processed_dataset(processed_data_dir: str, types_file_path: str, data: list, pocket_element_map: list, min_pharm_centers = 3):
    
    pocket_element_to_idx = {element: idx for idx, element in enumerate(pocket_element_map)}
    
    prot_file_name = []
    pharm_file_name = []
    lig_rdmol = []
    pharm_pos_arr = []
    pharm_feat_arr = []
    prot_pos_arr = []
    prot_feat_arr = []

    for item in data:

        pharm_types = item[3][1] # np array of pharmacophore types as integers
        n_pharmacophore_centers = pharm_types.shape[0]
        if n_pharmacophore_centers < min_pharm_centers:
            continue

        prot_file_name.append(item[0])
        pharm_file_name.append(item[1])
        lig_rdmol.append(item[2])
        pharm_pos_arr.append(item[3][0])
        pharm_feat_arr.append(item[3][1])
        prot_pos_arr.append(item[4][0])
        prot_feat_arr.append(item[4][1])

    # get the number of pharmacophore centers in every example
    n_centers = np.array([len(x) for x in pharm_pos_arr])

    # get the number of receptor atoms in every example
    n_atoms = np.array([len(x) for x in prot_pos_arr])

    # concatenate pharm_pos, pharm_feat, prot_pos, prot_feat into single arrays
    pharm_pos = np.concatenate(pharm_pos_arr, axis=0, dtype=np.float32)
    pharm_feat = np.concatenate(pharm_feat_arr, axis=0, dtype=np.int32)
    prot_pos = np.concatenate(prot_pos_arr, axis=0, dtype=np.float32)

    # convert pocket elements from strings to integers and concatenate into a single array
    prot_feat = np.concatenate(prot_feat_arr, axis=0)
    prot_feat_idxs = np.array([pocket_element_to_idx[el] for el in prot_feat])
    prot_feat = np.array(prot_feat_idxs, dtype=np.int32)

    # create an array of indicies to keep track of the start_idx and end_idx of each pharmacophore
    pharm_idx_array = np.zeros((len(pharm_pos_arr), 2), dtype=int)
    pharm_idx_array[:, 1] = np.cumsum(n_centers)
    pharm_idx_array[1:, 0] = pharm_idx_array[:-1, 1]

    # create an array of indicies to keep track of the start_idx and end_idx of each receptor
    prot_idx_array = np.zeros((len(prot_pos_arr), 2), dtype=int)
    prot_idx_array[:, 1] = np.cumsum(n_atoms)
    prot_idx_array[1:, 0] = prot_idx_array[:-1, 1]

    # get the processed output directory for this types file
    types_file_stem = Path(types_file_path).name.split('.types')[0]
    output_dir = Path(processed_data_dir) / types_file_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # use np.savez_compressed to save protein positions/features, pharmacophore positions/features, and the index arrays
    # to a .npz file
    prot_pharm_tensors_file = output_dir / 'prot_pharm_tensors.npz'
    np.savez_compressed(prot_pharm_tensors_file,
                        prot_pos=prot_pos, prot_feat=prot_feat, prot_idx=prot_idx_array,
                        pharm_pos=pharm_pos, pharm_feat=pharm_feat, pharm_idx=pharm_idx_array,)
    

    # write the ligand rdkit molecules to a .pkl.gz file
    lig_rdmol_file = output_dir / 'lig_rdmol.pkl.gz'
    with gzip.open(lig_rdmol_file, 'wb') as f:
        pickle.dump(lig_rdmol, f)

    # write the protein file names to a .pkl.gz file
    prot_file_name_file = output_dir / 'prot_file_names.pkl.gz'
    with gzip.open(prot_file_name_file, 'wb') as f:
        pickle.dump(prot_file_name, f)


if __name__ == "__main__":

    args = parse_args()

    # process config file into dictionary
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    crossdocked_path = config['dataset']['raw_data_dir']
    crosdocked_data_dir = Path(crossdocked_path) / 'CrossDocked2020'
    output_path = config['dataset']['processed_data_dir']
    dataset_size = config['dataset']['dataset_size']

    # all inputs is a list of tuples. Each tuple has length 2.
    # the first entry in the tuple is the filepath of the types file for which this data point came from
    # the second entry in the tuple is itself a list of tuples. Each tuple has length 2.
    # the first entry in the tuple is the filepath of the receptor file, the second entry is the filepath of the ligand file
    allinputs = []
    #should path to types be a separate argument ?
    types_files = os.path.join(crossdocked_path,'types','it2_tt_v1.3_0_test*types')
    for fname in glob.glob(types_files):
        #pull out good rmsd lines only
        f = open(fname)
        # inputs is a list which contains tuples of length 2. the first item in each tuple is the receptor file name and the second item is the ligand file name
        inputs = [] 
        for idx, line in enumerate(f):
            label,affinity,rmsd,rec,glig,_ = line.split()
            if label == '1':
                inputs.append((rec,glig))

            if dataset_size is not None and idx > dataset_size:
                break

        allinputs.append((fname,inputs))

    #collect and parse all receptors
    # I don't think this is actually necessary right now
    # leaving it here incase we change our minds
    # but for now the startegy is to read the the original receptor pdb file if its needed
    # the filepath of which is made available as part of the processed dataset
    # receptors = dict()
    # for fname, inputs in allinputs:
    #     for rec,lig in tqdm(inputs):
    #         rec = rec.replace('_0.gninatypes','.pdb')           
    #         rec_path = os.path.join(data_path, rec)
            
    #         if rec not in receptors:
    #             receptors[rec] = Chem.MolFromPDBBlock(open(rec_path).read(),sanitize=False)

    # receptors_path = os.path.join(output_path, 'receptors.pkl.gz')
    # with gzip.open(receptors_path,'wb') as recout:
    #     pickle.dump(receptors,recout)
        

    # set the arguments from config which need to be passed to the getfeatures function
    getfeatures_partial = partial(getfeatures, crossdocked_data_dir=crosdocked_data_dir, pocket_cutoff=config['dataset']['pocket_cutoff'])

    allphdata = []
    for fname, inputs in allinputs:

        # print the fname we are processing
        print(f'processing types file {fname}')

        # extract features for each protein-ligand pair
        if args.max_workers:
            phdata = process_map(getfeatures_partial, inputs, max_workers=args.max_workers)
        else:
            phdata = process_map(getfeatures_partial, inputs)

        # the third entry in each tuple is the ligand molecule as an rdkit object, which is None if 
        # the ligand molecule could not be parsed. we filter out these examples
        phdata = [ex for ex in phdata if ex[2]]


        # process into tensors
        write_processed_dataset(output_path, fname, phdata,
                                pocket_element_map=config['dataset']['prot_elements'],
                                min_pharm_centers=config['dataset']['min_pharm_centers'])
