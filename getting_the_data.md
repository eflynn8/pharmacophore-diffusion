# Starting from scratch, downloading the crossdocked dataset

1. Run `mkdir data/crossdocked_raw` to create a directory to store the raw data.

2. Then, within `data/crossdocked_raw`, run the following commands [here](https://github.com/gnina/models/tree/master/data/CrossDocked2020#extracting-the-tarballs) to download the crossdocked dataset.

# Starting on a machine which already has the crossdocked dataset downloaded somewhere

Determine the directory which contains the crossdocked dataset on your machine. Then, create a symlink to that directory in the `data` directory of this repository.  For example, if the crossdocked dataset is located at `/path/to/crossdocked`, then run the following command:

```console
ln -s /path/to/crossdocked data/crossdocked_raw
```

# A Caveat

You can actually specify any crossdocked location and set the output of the processing script to any location,
but for consistency across the people developing, we tried to just stick to using `data/crossdocked_raw` as the input and `data/crossdocked_processed` as the output.

# Processing the dataset
The script `process_crossdocked.py` is ultimately responsible for processing the crossdocked dataset. However, it calls pharmit in a subprocess to extract pharmacophores.  So we need to install pharmit.

Here is a link for installing pharmit: [pharmit](https://github.com/dkoes/pharmit)