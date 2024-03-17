# Starting from scratch, downloading the crossdocked dataset

1. Run `mkdir data/crossdocked_raw` to create a directory to store the raw data.

2. Then, within `data/crossdocked_raw`, run the following commands [here](https://github.com/gnina/models/tree/master/data/CrossDocked2020#extracting-the-tarballs) to download the crossdocked dataset.

# Starting on a machine which already has the crossdocked dataset downloaded somewhere

Determine the directory which contains the crossdocked dataset on your machine. Then, create a symlink to that directory in the `data` directory of this repository.  For example, if the crossdocked dataset is located at `/path/to/crossdocked`, then run the following command:

```console
ln -s /path/to/crossdocked data/crossdocked_raw
```

## which directory is the correct crossdocked directory???

Ok so all of the actual protein/ligand structures are located within a directory named `CrossDocked2020`. But the directory containing the `CrossDocked2020` directory is the one that should be symlinked to `data/crossdocked_raw`. This is because the directory containing `CrossDocked2020` also contains the types files.

# A Caveat

You can actually specify any crossdocked location and set the output of the processing script to any location,
but for consistency across the people developing, we will try to just stick to using `data/crossdocked_raw` as the input and `data/crossdocked_processed` as the output.

# Processing the dataset
The script `process_crossdocked.py` is ultimately responsible for processing the crossdocked dataset. However, it calls pharmit in a subprocess to extract pharmacophores.  So we need to install pharmit.

Here is a link for installing pharmit: [pharmit](https://github.com/dkoes/pharmit)

Once pharmit is installed, you can proces the entire crossdocked dataset by just running the following command:

```console
python process_crossdocked.py --config=configs/dev.yml
```

# What I don't want to process the crossdocked dataset myself?

Processing the crossdocked dataset requries downloading it. The code itself also is very io-intensive. Fortunately, to train a model, you really only need access to the processed version of the dataset. So we can process the crossdocked dataset once somewhere else (on Ian's workstation or David's workstation) and then just copy the fully processed dataset to the `processed_data_dir` specified in the model config files, which by convention will always be `data/crossdocked_processed` or `data/crossdocked_processed_dev`.