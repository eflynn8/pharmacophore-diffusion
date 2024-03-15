# pharmacophore-diffusion

# todo
- [ ] write some complimentary instructions for downloading crossdocked and running riya's script
- [ ] riya's script should take as input a configuration file, and the dataset settings in the configuration file should be used by the script (for example, pocket cutoff, which is probably hard-coded right now)

# our data processing pipeline sucks! here's why

## number of steps in the pipeline

We currently have 2 "processing" steps in the pipeline:
1. we use `pharma_process.py` to crawl over the crossdocked dataset, and extract pharmacophores/pocket features. The output of this script is a couple of pickle files.
2. The pickle files are read in by the dataset class (`dataset.protein_pharm_dataset.ProteinPharmacophoreDataset`). The dataset class is responsible for processing these pickle files into the big tensors that are ultimately fed into the model. It also saves the processed tensors to disk. Logistically, this gets really messy. Because now we have arguments to the dataset class to specify a few things. First, whether to load the processed tensosr from disk, or to process the pickle files. Also, we now have to specify both the location of the pickle files and the location of the processed data. 

## Mixing together the "Dataset class" and the second "data processing" step

As mentioned above, the dataset class is now responsible for both typical dataset class things AND processing the pickle files. What this means is, if I want to process the dataset, I need to run the training script. Logistically this is kind of annoying. Now our config file needs to contain information about whether when we run the training script, are we processing the pickle files first? or using some existing pickle files? What if I just want to process the dataset once, and then have many models use the same processed dataset? I would need to have a special config file that is just used for data processing, and a separate config file for the models that are ACTUALLY training, and these separate config files are just so we can accomoplish completeley different tasks using the same script (`train.py`). This is confusing and annoying and just bad design. 

I would rather have a separate script that processes the data, and then the dataset class only reads from the processed data. If we have these things as two separate scripts, then the arugments that need to be passed / the logic that needs to be performed within each script is a lot more clear. It results in a more concrete delegation of responsiblities to various components of the codebase. 


## ease of extension to other datasets

if we ever try to train on a dataset other than crossdocked, then these pickle files would be unncessarily convulted to recreate for a different dataset
this is why I am in favor of the strategy of "data processing" being a separate step in the pipeline from "data loading" that is, for each dataset, there should be a data processing script which writes all the data in a pre-determined format then the actual dataset class (this class) only ever reads from that format and then there is a separate "data processing" script for each dataset that we intend to train on


# notes on a dev configuration file which can work across all of our development environments

ok so Emma has been doing most of the coding here (go emma!) and as a result her development config file `configs/dev.yml` specifies filepaths for the data which are specific to her workstation.
but we are also going to run code on the cluster and also each of us will probably be setting up a development environment on each of our workstations. so we need a solution where the config file
saved in this repository (`configs/dev.yml`) can read the data in correctly regardless of the specific development environment. i have a solution for this!!

## if you are developing locally (not on the cluster)

run these commands:

```console
mkdir data/crossdocked_pharm_extracted
scp cluster:/net/galaxy/home/koes/paf46_shared/cd2020_v1.3/types/*.pkl.gz data/crossdocked_pharm_extracted/
```

## if you are developing on the cluster

run these commands:

```console
ln -s /net/galaxy/home/koes/paf46_shared/cd2020_v1.3/types/ data/crossdocked_pharm_extracted/
```

## what `configs/dev.yml` should look like

now, we set the dataset.data_files parameter to be `[ 'data/crossdocked_pharm_extracted/it2_tt_v1.3_0_test0_ph.pkl.gz' ]`
and the receptor file parameter to be: `data/crossdocked_pharm_extracted/receptors.pkl.gz` 

now the config file will work regardless of the development environment

I also changed the processed_data_dir to `data/crossdock_processed`.