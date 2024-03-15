# pharmacophore-diffusion

# todo
- [ ] convert david's notebook `pharmacophores.ipynb` to a processing script
- [ ] write some complimentary instructions for downloading crossdocked and running this script
- [ ] make it so that this processing script gets arguments from model config files
- [ ] how does evaluation hyperparams from config file find their way into the PL module?


# a note on the data processing pipeline

TODO: if we ever try to train on a dataset other than crossdocked, then these pickle files would be unncessarily convulted to recreate for a different dataset
this is why I am in favor of the strategy of "data processing" being a separate step in the pipeline from "data loading" that is, for each dataset, there should be a data processing script which writes all the data in a pre-determined format then the actual dataset class (this class) only ever reads from that format and then there is a separate "data processing" script for each dataset that we intend to train on