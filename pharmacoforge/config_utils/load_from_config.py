from pharmacoforge.models.forge import PharmacoForge
from pathlib import Path
from pharmacoforge.dataset.protein_pharm_dataset import ProteinPharmacophoreDataset
from pharmacoforge.dataset.protein_pharmacophore_datamodule import CrossdockedDataModule
import yaml

def model_from_config(config: dict, ckpt=None) -> PharmacoForge:


    # get the number of receptor atom features, ligand atom features, and keypoint features
    n_rec_feat = len(config['dataset']['prot_elements'])
    n_ph_types = len(config['dataset']['ph_type_map'])
    eval_config = config['training']['evaluation']


    model = PharmacoForge(
        pharm_nf=n_ph_types,
        rec_nf=n_rec_feat,
        ph_type_map=config['dataset']['ph_type_map'],
        processed_data_dir=config['dataset']['processed_data_dir'],
        n_pockets_to_sample=eval_config['n_pockets'],
        pharms_per_pocket=eval_config['pharms_per_pocket'],
        sample_interval=eval_config['sample_interval'],
        val_loss_interval=eval_config['val_loss_interval'],
        batch_size=config['training']['batch_size'],
        graph_config=config['graph'],
        dynamics_config=config['dynamics'],
        lr_scheduler_config=config['lr_scheduler'],
        diffusion_config=config['diffusion'],
        fm_config=config['flow-matching'],
        **config['pharmacoforge']
    )

    return model

def data_module_from_config(config: dict) -> CrossdockedDataModule:

    dataset_config = config['dataset']
    model_class = config['pharmacoforge'].get('model_class', 'diffusion')
    
    data_module = CrossdockedDataModule(
        dataset_config=config['dataset'], 
        graph_config=config['graph'],
        batch_size=config['training']['batch_size'], 
        num_workers=config['training']['num_workers'], 
        validation_splits=config['training']['validation_splits'],
        model_class=model_class)
    return data_module

def load_config(config_file: Path) -> dict:
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config