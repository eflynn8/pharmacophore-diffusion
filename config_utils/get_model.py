from models.pharmacodiff import PharmacophoreDiff
from pathlib import Path

def model_from_config(config: dict):


    # get the number of receptor atom features, ligand atom features, and keypoint features
    n_rec_feat = len(config['dataset']['prot_elements'])
    n_ph_types = len(config['dataset']['ph_type_map'])


    model = PharmacophoreDiff(
        pharm_nf=n_ph_types,
        rec_nf=n_rec_feat,
        processed_data_dir=config['dataset']['processed_data_dir'],
        graph_config=config['graph'],
        dynamics_config=config['dynamics'],
        lr_scheduler_config=config['lr_scheduler'],
        **config['diffusion']
    )

    return model