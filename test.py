import argparse
import time
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from rdkit import Chem
import shutil
import pickle
from tqdm import trange
import dgl
from models.pharmacodiff import PharmacophoreDiff

from constants import ph_idx_to_type
from config_utils.load_from_config import model_from_config, data_module_from_config
from dataset.receptor_utils import write_pocket_file
from analysis.pharm_builder import SampledPharmacophore
from analysis.metrics import SampleAnalyzer
from utils import write_pharmacophore_file, copy_graph

def parse_arguments():
    p = argparse.ArgumentParser()
    
    p.add_argument('--ckpt', type=Path, help='Path to checkpoint file. Must be inside model dir.', default=None)
    p.add_argument('--model_dir', type=Path, default=None, help='Directory of output from a training run. Will use last.ckpt in this directory.')
    
    
    p.add_argument('--samples_per_pocket', type=int, default=1, help="number of samples generated per pocket")
    p.add_argument('--pharm_sizes', nargs="*", type=int, default=[], help="number of pharmacophore centers in each sample, must be of length samples per pocket")
    p.add_argument('--max_batch_size', type=int, default=128, help='maximum feasible batch size due to memory constraints')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', type=Path, default=None)
    p.add_argument('--max_tries', type=int, default=1, help='maximum number of batches to sample per pocket')
    p.add_argument('--dataset_size', type=int, default=None, help='truncate test dataset')
    p.add_argument('--dataset_idx', type=int, default=None)
    p.add_argument('--dataset_idx_as_start', action='store_true', help="Use dataset idx as starting index and sample dataset size")
    p.add_argument('--split', type=str, default='val', help="Specifying which data split to use; options are val or train")
    p.add_argument('--use_ref_pharm_com', action='store_true', help="Initialize each pharmacophore's position at the reference pharmacophore's center of mass" )
    p.add_argument('--visualize_trajectory', action='store_true', help="Visualize trajectories of generated pharmacophores" )

    p.add_argument('--metrics', action='store_true', help='compute metrics on generated pharmacophores')
    
    args = p.parse_args()

    if args.ckpt is None and args.model_dir is None:
        raise ValueError('Must provide either --ckpt or --model_dir')
    
    # check samples per pocket and pharm sizes match
    if args.pharm_sizes:
        if len(args.pharm_sizes) != args.samples_per_pocket:
            raise ValueError("If pharm_sizes list is provided, must of length sample per pocket")
    
    return args

def main():

    args = parse_arguments()

    # get filepath of config file within model_dir
    if args.ckpt is not None:
        run_dir = args.ckpt.parent.parent
        model_file = args.ckpt
    elif args.model_dir is not None:
        run_dir = args.model_dir
        model_file = run_dir / 'checkpoints' / 'last.ckpt'

    #get output dir path and create the directory
    if args.output_dir is None:
        output_dir = run_dir / 'samples'
    else:
        output_dir = args.output_dir

    output_dir.mkdir(exist_ok=True) 
    pharm_dir = output_dir
    pharm_dir.mkdir(exist_ok=True)


    # get config file
    #be robust to yml vs yaml lol
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

    # create test dataset object
    test_data_module = data_module_from_config(config)
    if args.split == 'train':
        test_data_module.setup('fit')
        test_dataset = test_data_module.train_dataset
    else:
        test_data_module.setup('test')
        test_dataset = test_data_module.val_dataset

    #create diffusion model
    # TODO: remove this try/except, it is only for backwards compatibility for models trained before i added the ph_type_map argument to the model class
    try:
        model = PharmacophoreDiff.load_from_checkpoint(model_file).to(device)
    except TypeError:
        model = PharmacophoreDiff.load_from_checkpoint(model_file, ph_type_map=config['dataset']['ph_type_map']).to(device)
    model.eval()

    pocket_sampling_times=[]

    if args.dataset_idx is None:

        if args.dataset_size is not None:
            dataset_size = args.dataset_size
        else:
            dataset_size = len(test_dataset)
        dataset_iterator = trange(dataset_size)
    elif args.dataset_idx is not None and args.dataset_idx_as_start:
        if args.dataset_size is not None:
            dataset_size = args.dataset_size
            dataset_iterator = trange(args.dataset_idx, args.dataset_idx + dataset_size)
        else:
            raise ValueError('Must provide dataset size if dataset_idx_as_start is used')
    else:
        dataset_size = 1
        dataset_iterator = trange(args.dataset_idx, args.dataset_idx+1)


    all_pharms = []
    for dataset_idx in dataset_iterator:

        pocket_sample_start = time.time()
        #get receptor graph and reference pharmacophore positions/features from test set

        ref_graph = test_dataset[dataset_idx]
        raw_data_dir, ref_prot_file, ref_lig_rdmol = test_dataset.get_files(dataset_idx)
        raw_data_dir = Path(raw_data_dir)
        ref_prot_file = raw_data_dir / ref_prot_file

        ref_graph = ref_graph.to(device)

        if args.use_ref_pharm_com:
            ref_init_pharm_com = dgl.readout_nodes(ref_graph, ntype='pharm', feat='x_0', op='mean')
            assert ref_init_pharm_com.shape == (1,3)
        else:
            ref_init_pharm_com = None

        sampled_pharms: List[SampledPharmacophore] = []

        while True:
            n_pharmacophores_needed = args.samples_per_pocket - len(sampled_pharms)
            batch_size = min(n_pharmacophores_needed, args.max_batch_size)

            #collect just the batch_size graphs and init_pharm_coms that we need
            if not args.pharm_sizes:
                pharm_sizes = model.pharm_size_dist.sample_uniformly(args.samples_per_pocket)
            else:
                pharm_sizes = args.pharm_sizes
            g_batch = copy_graph(ref_graph, batch_size, pharm_feats_per_copy=pharm_sizes)
            g_batch = dgl.batch(g_batch)

            if args.use_ref_pharm_com:
                init_pharm_com = ref_init_pharm_com.repeat(batch_size,1)
            else:
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

        pocket_dir = pharm_dir / f'pocket_{dataset_idx}'
        pocket_dir.mkdir(exist_ok=True)

        # add sampled pharms to list of all pharms
        all_pharms.extend(sampled_pharms)

        # save pocket sample time
        with open(pocket_dir / 'sample_time.txt', 'w') as f:
            f.write(f'{pocket_sample_time:.2f}')
        with open(pocket_dir / 'sample_time.pkl', 'wb') as f:
            pickle.dump(pocket_sampling_times, f)

        #print the sampling time
        print(f'Pocket {dataset_idx} sampling time: {pocket_sample_time:.2f} seconds')

        #print the sampling time per pharmacophore
        print(f'Pocket {dataset_idx} sampling time per pharmacophore: {pocket_sample_time/len(sampled_pharms):.2f} seconds')


        # write the protein pocket file
        pocket_file = pocket_dir / 'pocket.pdb'
        write_pocket_file(ref_prot_file,ref_lig_rdmol, pocket_file,cutoff=config['dataset']['pocket_cutoff'])

        #save reference files
        ref_files_dir=pocket_dir / 'reference_files'
        ref_files_dir.mkdir(exist_ok=True)
        shutil.copy(ref_prot_file, ref_files_dir / ref_prot_file.name)
        sdfwriter = Chem.SDWriter(ref_files_dir / 'ligand.sdf')            
        sdfwriter.write(ref_lig_rdmol,confId=0)

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

    # compute metrics if requested
    if args.metrics:
        metrics = SampleAnalyzer().analyze(all_pharms)
        print(metrics)
        with open(output_dir / 'metrics.txt', 'w') as f:
            metrics_strs = [ f'{k}: {v:.3f}' for k,v in metrics.items() ]
            f.write('\n'.join(metrics_strs))
        with open(output_dir / 'metrics.pkl', 'wb') as f:
            pickle.dump(metrics, f)
        
        freqs = SampleAnalyzer().pharm_feat_freq(all_pharms)
        with open(output_dir / f'pharm_counts_{args.dataset_idx}.txt', 'w') as f:
            f.write(str(freqs.data))
        with open(output_dir / f'pharm_counts_{args.dataset_idx}.pkl', 'wb') as f:
            pickle.dump(freqs, f)

        plt.bar(ph_idx_to_type, freqs)
        plt.xticks(rotation=90)
        plt.xlabel("Pharmacophore Feature")
        plt.ylabel("Feature Count")
        plt.title(f"Pharmacophore Type Counts for {dataset_size} Pockets")
        plt.tight_layout()
        plt.savefig(output_dir / f"pharm_freq_plot_{args.dataset_idx}.png")


if __name__ == '__main__':
    main()

