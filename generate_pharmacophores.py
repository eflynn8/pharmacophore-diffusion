import argparse
import math
import pickle
import shutil
import time
from pathlib import Path

import dgl
import numpy as np
import torch
import yaml
from Bio.PDB import MMCIFIO, PDBIO, MMCIFParser, PDBParser
from Bio.PDB.Polypeptide import is_aa, protein_letters_3to1
from rdkit import Chem
from scipy.spatial.distance import cdist
from torch.nn.functional import one_hot
from tqdm import trange
from typing import Dict, Iterable, List

from dataset.protein_pharm_dataset import build_initial_complex_graph
from constants import ph_idx_to_type
from config_utils.load_from_config import model_from_config, data_module_from_config
from dataset.receptor_utils import PocketSelector, write_pocket_file
from analysis.pharm_builder import SampledPharmacophore
from analysis.metrics import SampleAnalyzer
from models.pharmacodiff import PharmacophoreDiff
from utils import write_pharmacophore_file, copy_graph, get_prot_atom_ph_type_maps


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('receptor_file', type=Path, help='PDB file of the receptor')
    p.add_argument('ref_ligand_file', type=Path, help='sdf file of ligand used to define the pocket')
    p.add_argument('--ckpt', type=Path, help='Path to checkpoint file. Must be inside model dir.', default=None)
    p.add_argument('--model_dir', type=Path, default=None, help='Directory of output from a training run. Will use last.ckpt in this directory.')
    p.add_argument('--samples_per_pocket', type=int, default=1, help="number of samples generated per pocket")
    p.add_argument('--pharm_sizes', nargs="*", type=int, default=[], help="number of pharmacophore centers in each sample, must be of length samples per pocket")
    p.add_argument('--output_dir', type=str, default='generated_pharms/')
    p.add_argument('--max_batch_size', type=int, default=128, help='maximum feasible batch size due to memory constraints')
    p.add_argument('--seed', type=int, default=42, help='random seed as an integer. by default, no random seed is set.')
    # p.add_argument('--use_ref_pharm_com', action='store_true', help="Initialize each pharmacophore's position at the reference pharmacophore's center of mass" )
    p.add_argument('--visualize_trajectory', action='store_true', help="Visualize trajectories of generated pharmacophores" )
    p.add_argument('--metrics', action='store_true', help='compute metrics on generated pharmacophores')
    
    args = p.parse_args()

    if args.ckpt is not None and args.model_dir is not None:
        raise ValueError('only model_file or model_dir can be specified but not both')

    if args.ckpt is None and args.model_dir is None:
        raise ValueError('one of model_file or model_dir must be specified')
    
    # check samples per pocket and pharm sizes match
    if args.pharm_sizes:
        if len(args.pharm_sizes) != args.samples_per_pocket:
            raise ValueError("If pharm_sizes list is provided, must be of length sample per pocket")

    return args


def parse_ligand(ligand_path: Path, remove_hydrogen=False):
    """Load ligand file into rdkit, retrieve atom positions.

    Args:
        ligand_path (Path): Filepath of ligand SDF file

    Returns:
        ligand: rdkit molecule of the ligand
        atom_positions: (N, 3) torch tensor with coordinates of all ligand atoms
    """
    # read ligand into a rdkit mol
    suppl = Chem.SDMolSupplier(str(ligand_path), sanitize=False, removeHs=remove_hydrogen)
    ligands = list(suppl)
    if len(ligands) > 1:
        raise NotImplementedError('Multiple ligands found. Code is not written to handle multiple ligands.')
    ligand = ligands[0]

    # actually remove all hydrogens - setting removeHs=True still preserves hydrogens that are necessary for specifying stereochemistry
    # note that therefore, this step destroys stereochemistry
    if remove_hydrogen:
        ligand = Chem.RemoveAllHs(ligand, sanitize=False)

    # get atom positions
    ligand_conformer = ligand.GetConformer()
    atom_positions = ligand_conformer.GetPositions()
    atom_positions = torch.tensor(atom_positions).float()
    
    return ligand, atom_positions

def element_fixer(element: str):

    if len(element) > 1:
        element = element[0] + element[1:].lower()
    
    return element

def onehot_encode_elements(atom_elements: Iterable, element_map: Dict[str, int]) -> np.ndarray:

    def element_to_idx(element_str, element_map=element_map):
        try:
            return element_map[element_str]
        except KeyError:
            # print(f'other element found: {element_str}')
            return element_map['other']

    element_idxs = np.fromiter((element_to_idx(element) for element in atom_elements), int)
    onehot_elements = np.zeros((element_idxs.size, len(element_map)))
    onehot_elements[np.arange(element_idxs.size), element_idxs] = 1

    return onehot_elements

## Function to take in pocket PDB or mmcif and the ligand SDF to build the graph based on the atoms in the pocket
def process_ligand_and_pocket(rec_file: Path, lig_file: Path, output_dir: Path,
                                  prot_element_map,
                                  graph_cutoffs: dict,
                                  pocket_cutoff: float, remove_hydrogen: bool = True):
    
    if rec_file.suffix == '.pdb':
        parser = PDBParser(QUIET=True)
    elif rec_file.suffix == '.mmcif':
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError(f'unsupported receptor file type: {rec_file.suffix}, must be .pdb or .mmcif')

    rec_struct = parser.get_structure('', rec_file)

    _, lig_coords = parse_ligand(lig_file, remove_hydrogen=remove_hydrogen)

    # make ligand data into torch tensors
    lig_coords = torch.tensor(lig_coords, dtype=torch.float32)

    # get residues which constitute the binding pocket
    pocket_residues = []
    for residue in rec_struct.get_residues():

        # check if residue is a standard amino acid
        is_residue = is_aa(residue.get_resname(), standard=True)
        if not is_residue:
            continue

        # get atomic coordinates of residue
        res_coords = np.array([a.get_coord() for a in residue.get_atoms()])

        # check if residue is interacting with protein
        min_rl_dist = cdist(lig_coords, res_coords).min()
        if min_rl_dist < pocket_cutoff:
            pocket_residues.append(residue)

    if len(pocket_residues) == 0:
        raise ValueError(f'no valid pocket residues found.')

    if remove_hydrogen:
        atom_filter = lambda a: a.element != "H"
    else:
        atom_filter = lambda a: True

    pocket_atomres = []
    for res in pocket_residues:

        atom_list = res.get_atoms()

        pocket_atomres.extend([(a, res) for a in atom_list if atom_filter(a) ])

    pocket_atoms, atom_residues = list(map(list, zip(*pocket_atomres)))
    res_to_idx = { res:i for i, res in enumerate(pocket_residues) }
    pocket_res_idx = list(map(lambda res: res_to_idx[res], atom_residues)) #  list containing the residue of every atom using integers to index pocket residues
    pocket_res_idx = torch.tensor(pocket_res_idx)

    pocket_coords = torch.tensor(np.array([a.get_coord() for a in pocket_atoms]))
    pocket_elements = np.array([ element_fixer(a.element) for a in pocket_atoms ])

    onehot_elements = onehot_encode_elements(pocket_elements, prot_element_map)
    other_atoms_mask = torch.tensor(onehot_elements[:, -1] == 1).bool()
    pocket_atom_features = onehot_elements[:, :-1]

    pocket_atom_features = torch.tensor(pocket_atom_features).float()

    # remove other atoms from pocket
    pocket_coords = pocket_coords[~other_atoms_mask]
    pocket_atom_features = pocket_atom_features[~other_atoms_mask]

    # build graph and represent pharmacophore intial positions and features as 0s
    g: dgl.DGLHeteroGraph = build_initial_complex_graph(
        prot_atom_positions=pocket_coords,
        prot_atom_features=pocket_atom_features,
        cutoffs=graph_cutoffs,
        pharm_atom_positions=torch.zeros((1, 3)),
        pharm_atom_features=torch.zeros((1, 6))
    )

    # save the pocket file
    pocket_file = output_dir / f'pocket.pdb'
    pocket_selector = PocketSelector(pocket_residues)
    io_object = PDBIO()
    io_object.set_structure(rec_struct)
    io_object.save(str(pocket_file), pocket_selector)

    return g


def main():
    
    args = parse_arguments()

    # get output dir path and create the directory
    ## TODO: Fix where we put all the files to be in separate directories
    output_dir = Path(args.output_dir)
    pharm_dir = output_dir
    output_dir.mkdir(exist_ok=True)

    # get filepath of config file within model_dir
    if args.ckpt is not None:
        run_dir = args.ckpt.parent.parent
        model_file = args.ckpt
    elif args.model_dir is not None:
        run_dir = args.model_dir
        model_file = run_dir / 'checkpoints' / 'last.ckpt'
    
    # get config file
    config_file = run_dir / 'config.yaml'
    if not config_file.exists():
        config_file = run_dir / 'config.yml'
        if not config_file.exists():
            raise FileNotFoundError(f'config file not found in {run_dir}')

    # load model configuration
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'{device=}', flush=True)

    # set random seeds
    torch.manual_seed(args.seed)

    # isolate dataset config
    dataset_config = config['dataset']

    prot_element_map, ph_type_map = get_prot_atom_ph_type_maps(dataset_config)

    #create diffusion model
    # TODO: remove this try/except, it is only for backwards compatibility for models trained before i added the ph_type_map argument to the model class
    try:
        model = PharmacophoreDiff.load_from_checkpoint(model_file).to(device)
    except TypeError:
        model = PharmacophoreDiff.load_from_checkpoint(model_file, ph_type_map=config['dataset']['ph_type_map']).to(device)
    model.eval()

    ## Store times for sampling pockets
    pocket_sampling_times=[]
    pocket_sample_start = time.time()


    # process the receptor and pocket files
    rec_file = args.receptor_file
    ref_lig_file = args.ref_ligand_file

    if not rec_file.exists() or not ref_lig_file.exists():
        raise ValueError('receptor or reference ligand file does not exist')
    
    rec_name = rec_file.name.split(".")[0]

    pocket_dir = pharm_dir / f'rec_{rec_name}'
    pocket_dir.mkdir(exist_ok=True)
    
    ## TODO: Fix to be correct configs
    ref_graph: dgl.DGLHeteroGraph = process_ligand_and_pocket(
                                rec_file, ref_lig_file, pocket_dir,
                                prot_element_map=prot_element_map,
                                graph_cutoffs=config['graph']['graph_cutoffs'],
                                pocket_cutoff=dataset_config['pocket_cutoff'], 
                                remove_hydrogen=True)

    ref_graph = ref_graph.to(device)

    # get the number of nodes in the binding pocket
    n_rec_nodes = ref_graph.num_nodes('prot')
    n_rec_nodes = torch.tensor([n_rec_nodes], device=device)

    all_pharms = []

    pocket_sample_start = time.time()

    ## TODO: Is this possible with pdb outside of dataset??
    # if args.use_ref_pharm_com:
    #     ref_init_pharm_com = dgl.readout_nodes(ref_graph, ntype='pharm', feat='x_0', op='mean')
    #     assert ref_init_pharm_com.shape == (1,3)
    
    ref_init_pharm_com = None

    sampled_pharms: List[SampledPharmacophore] = []

    while True:
        n_pharmacophores_needed = args.samples_per_pocket - len(sampled_pharms)
        batch_size = min(n_pharmacophores_needed, args.max_batch_size)

        # collect just the batch_size graphs and init_pharm_coms that we need
        # make copies of graph based on number of pharmacophore centers requested
        if not args.pharm_sizes:
            pharm_sizes = model.pharm_size_dist.sample_uniformly(args.samples_per_pocket)
        else:
            pharm_sizes = args.pharm_sizes
        g_batch = copy_graph(ref_graph, batch_size, pharm_feats_per_copy=pharm_sizes)
        g_batch = dgl.batch(g_batch)

        # if args.use_ref_pharm_com:
        #     init_pharm_com = ref_init_pharm_com.repeat(batch_size,1)
        init_pharm_com = None

        #sample pharmacophores
        with g_batch.local_scope():
            batch_pharms = model.sample_given_receptor(g_batch, init_pharm_com=init_pharm_com,visualize_trajectory=args.visualize_trajectory)
            sampled_pharms.extend(batch_pharms)

        #break out of loop when we have enough pharmacophores
        if len(sampled_pharms) >= args.samples_per_pocket:
            break
    
    pocket_sample_time = time.time() - pocket_sample_start
    pocket_sampling_times.append(pocket_sample_time)

    # add sampled pharms to list of all pharms
    all_pharms.extend(sampled_pharms)

    # save pocket sample time
    with open(pocket_dir / 'sample_time.txt', 'w') as f:
        f.write(f'{pocket_sample_time:.2f}')
    with open(pocket_dir / 'sample_time.pkl', 'wb') as f:
        pickle.dump(pocket_sampling_times, f)

    #print the sampling time
    print(f'Pocket {rec_name} sampling time: {pocket_sample_time:.2f} seconds')

    #print the sampling time per pharmacophore
    print(f'Pocket {rec_name} sampling time per pharmacophore: {pocket_sample_time/len(sampled_pharms):.2f} seconds')

    #save reference files
    ref_files_dir=pocket_dir / 'reference_files'
    ref_files_dir.mkdir(exist_ok=True)
    shutil.copy(rec_file, ref_files_dir / rec_file.name)
    shutil.copy(ref_lig_file, ref_files_dir / ref_lig_file.name)

    #write out pharmacophores
    ph_files = []
    if args.visualize_trajectory:
        # write a trajectory file for each sampled pharmacophore
        for pharm_idx, sampled_pharm in enumerate(sampled_pharms):
            pharm_file = pocket_dir / f'pharm_{pharm_idx}_traj.xyz'
            ph_files.append(pharm_file)
            sampled_pharm.traj_to_xyz(pharm_file)
    else:
        # write a single file that contains all sampled pharmacophores
        pharm_file = pocket_dir / 'pharms.xyz'
        ph_files.append(pharm_file)
        pharm_file_content = ''
        for sampled_pharm in sampled_pharms:
            pharm_file_content += sampled_pharm.to_xyz_file()
        with open(pharm_file, 'w') as f:
            f.write(pharm_file_content)

if __name__ == "__main__":

    with torch.no_grad():
        main()