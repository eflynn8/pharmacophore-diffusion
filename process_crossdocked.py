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

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--crossdocked_dir", help="Path to crossdocked directory", required=True, type=Path)
    parser.add_argument("--config", help="Path to config file", required=True, type=Path)
    args = parser.parse_args()
    return args

def element_fixer(element: str):

    if len(element) > 1:
        element = element[0] + element[1:].lower()
    
    return element


def getfeatures(reclig, pocket_cutoff: int = 8):
    
    # reclig is a tuple of length 2. The first entry is the receptor file name and the second entry is the ligand file name

    rec,glig = reclig
    rec = rec.replace('_0.gninatypes','.pdb')            
    m = re.search(r'(\S+)_(\d+)\.gninatypes',glig)
    prefix = m.group(1)
    num = int(m.group(2))
    lig = prefix+'.sdf.gz'
    
    rec_path = os.path.join(data_path, rec)
    lig_path = os.path.join(data_path, lig)
    
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
        
def construct_dataset_tensors():
    pass


if __name__ == "__main__":

    args = parse_args()

    # process config file into dictionary
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_path = args.crossdocked_dir
    output_path = config['dataset']['processed_data_dir']

    # all inputs is a list of tuples. Each tuple has length 2.
    # the first entry in the tuple is the filepath of the types file for which this data point came from
    # the second entry in the tuple is itself a list of tuples. Each tuple has length 2.
    # the first entry in the tuple is the filepath of the receptor file, the second entry is the filepath of the ligand file
    allinputs = []
    #should path to types be a separate argument ?
    types_files = os.path.join(data_path,'types','it2_tt_v1.3_0_test*types')
    for fname in glob.glob(types_files):
        #pull out good rmsd lines only
        f = open(fname)
        # inputs is a list which contains tuples of length 2. the first item in each tuple is the receptor file name and the second item is the ligand file name
        inputs = [] 
        for line in f:
            label,affinity,rmsd,rec,glig,_ = line.split()
            if label == '1':
                inputs.append((rec,glig))
        allinputs.append((fname,inputs))

    #collect and parse all receptors
    # I don't think this is actually necessary right now
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
    getfeatures_partial = partial(getfeatures, pocket_cutoff=config['dataset']['pocket_cutoff'])

    pool = multiprocessing.Pool()
    allphdata = []
    for fname, inputs in allinputs:
        #get features
        phdata = pool.map(getfeatures_partial, inputs, )
        #filter out empty
        # the third entry in each tuple is the ligand molecule as an rdkit object, which is None if the ligand molecule could not be parsed
        phdata = [ex for ex in phdata if ex[2]]
        allphdata += phdata
        #writeout pickle
        outname = os.path.basename(fname).replace('.types','_ph.pkl.gz')
        outpath = os.path.join(output_path, outname)
        with gzip.open(outpath,'wb') as out:
            pickle.dump(phdata,out)
