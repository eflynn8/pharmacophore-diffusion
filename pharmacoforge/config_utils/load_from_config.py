from pharmacoforge.models.pharmacodiff import PharmacophoreDiff
from pathlib import Path
from pharmacoforge.dataset.protein_pharm_dataset import ProteinPharmacophoreDataset
from pharmacoforge.dataset.protein_pharmacophore_datamodule import CrossdockedDataModule

def model_from_config(config: dict, ckpt=None) -> PharmacophoreDiff:


    # get the number of receptor atom features, ligand atom features, and keypoint features
    n_rec_feat = len(config['dataset']['prot_elements'])
    n_ph_types = len(config['dataset']['ph_type_map'])

    eval_config = config['training']['evaluation']


    model = PharmacophoreDiff(
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
        **config['diffusion']
    )

    return model

def data_module_from_config(config: dict) -> CrossdockedDataModule:

    dataset_config = config['dataset']
    graph_cutoffs = config['graph']['graph_cutoffs']
    dataset_config['graph_cutoffs'] = graph_cutoffs
    
    data_module = CrossdockedDataModule(dataset_config=config['dataset'], 
        batch_size=config['training']['batch_size'], 
        num_workers=config['training']['num_workers'], 
        validation_splits=config['training']['validation_splits'])
    return data_module