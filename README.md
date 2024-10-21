# pharmacophore-diffusion

please follow the directions at [getting_the_data.md](getting_the_data.md) to get the data and configure the datasets for your specific work environment.

## Making this repo pip installable

You have to install the dependencies yourself. But, once you do, while you're inside the conda environment, just run `pip install -e ./` from the root of this repo.

# TODO:
- [ ] i have questions about pharm size selection in test.py
- [ ] there should just be a PharmacoForge.sample_random_sizes?
- [ ] implement xhat trajectory saving for diffusion
- [ ] add attention to GVPMultiEdgeConv
- [ ] modality-dependent loss weights!