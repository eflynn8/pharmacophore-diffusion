# PharmacoForge

### **Read the full paper here: [PharmacoForge: pharmacophore generation with diffusion models](https://www.frontiersin.org/journals/bioinformatics/articles/10.3389/fbinf.2025.1628800/full)**

PharmacoForge is a diffusion model capable of generating pharmacophores of user-specified size for a receptor protein pocket. Pharmacophores are made up of centers describing areas of interaction between a ligand and a receptor; each center has 3D coordinates and a feature type (aromatic, hydrogen acceptor, hydrogen donor, hydrophobic, negative ion, positive ion). Pharmacophore size is defined as the number of pharmacophore centers. Generated pharmacophores can be used to screen databases using the [Pharmit command line tool](https://github.com/dkoes/pharmit).

## Colab Notebook

Try out PharmacoForge using the Colab notebook below! From here, you can use a pre-trained model to generate pharmacophores by loading in your receptor PDB and a reference ligand (SDF file) to specify the binding pocket.

[PharmacoForge Colab](https://colab.research.google.com/drive/1XZViTC6BiNN1tF0PIhZ48j3wHocUklHJ?usp=sharing)

## Generating new pharmacophores for a pocket

To generate new pharmacophores for a receptor, you will need a PDB file of the receptor and to designate the desired binding pocket with either a reference ligand as an SDF file or to specify the residues that make up the binding pocket as a list. Below is a template command for generating 3 pharmacophores for a pocket specified by a reference ligand:

`python generate_pharmacophores.py receptor_file /path/to/receptor.pdb --ref_ligand_file /path/to/ligand.sdf --model_dir /path/to/trained/model --samples_per_pocket 3 --pharm_sizes 3 4 5 --visualize_trajectory`

Here is a full example command for generating 30 pharmacophores of sizes 3-8 centers:

`python generate_pharmacophores.py receptor.pdb --ref_ligand_file crystal_ligand.sdf --ckpt runs/trained_model/checkpoints/last.ckpt --visualize_trajectory --output_dir generated_pharms/samples --receptor_name xiap --max_batch_size 32 --samples_per_pocket 30 --pharm_sizes 3 3 3 3 3 4 4 4 4 4 5 5 5 5 5 6 6 6 6 6 7 7 7 7 7 8 8 8 8 8 --metrics`

Here is an example command to generate 3 pharmacophores for a pocket specified by a residue list:

`python generate_pharmacophores.py receptor_file /path/to/receptor.pdb --residue_list "A:10" "A:11" "A:12" "A:13" "A:14" --model_dir /path/to/trained/model --samples_per_pocket 3 --pharm_sizes 3 4 5 --visualize_trajectory`

The full options for generating pharmacophores are as follows:
- samples_per_pocket: Number of pharmacophore samples generated for each receptor pocket
- pharm_sizes: list of pharmacophore centers for each sample, must be of length samples_per_pocket
- output_dir: file path to directory for generated samples
- max_batch_size: maximum feasible batch size due to memory constraints
- use_ref_lig_com: random pharmacophore centers will be initalized in the reference ligand's center of mass (COM) instead of the pocket's COM
- visualize_trajectory: if included, will output xyz trajectory files; can be visualized with pymol_generated.py

## Training PharmacoForge
To train a new PharmacoForge model, first follow the directions at [getting_the_data.md](getting_the_data.md) to get the data and configure the datasets for your specific work environment. Create a new yaml to specify your desired training parameters, and train using those options with the following example command:

`python train.py --config=configs/endpoint_param.yaml`

An example yaml file is provided in configs/

## Visualize pharmacophores in Pymol

`pymol pymol_generated.py --pocket_dir /path/to/pocket --load_reference`

## Making this repo pip installable

You have to install the dependencies yourself. But, once you do, while you're inside the conda environment, just run `pip install -e ./` from the root of this repo.
