# pharmacophore-diffusion

# todo
- [ ] write some complimentary instructions for downloading crossdocked and running riya's script
- [ ] riya's script should take as input a configuration file, and the dataset settings in the configuration file should be used by the script (for example, pocket cutoff, which is probably hard-coded right now)

# processing crossdocked

example command:

```console
python process_crossdocked.py --config=configs/dev.yml
```

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