# Downloading the CrossDocked Dataset

Directions are conveinlently located [here](https://github.com/gnina/models/tree/master/data/CrossDocked2020#extracting-the-tarballs).

# Processing the dataset
The script `process_crossdocked.py` is ultimately responsible for processing the crossdocked dataset. However, it calls pharmit in a subprocess to extract pharmacophores.  So we need to install pharmit.

Here is a link for installing pharmit: [pharmit](https://github.com/dkoes/pharmit)