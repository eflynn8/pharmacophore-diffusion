from models.pharmacodiff import PharmacophoreDiff
from pathlib import Path
from dataset.protein_pharm_dataset import ProteinPharmacophoreDataset
from dataset.protein_pharmacophore_datamodule import ProtPharmDataModule

def model_from_config(config: dict) -> PharmacophoreDiff:


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

def data_module_from_config(config: dict) -> ProtPharmDataModule:

    # TODO: create dataset class (or, alternatively, a PL datamodule)
    data_files = config['dataset']['data_files']
    rec_file = config['dataset']['rec_file']
    processed_data_dir = config['dataset']['processed_data_dir']
    prot_elements = config['dataset']['prot_elements']
    # pocket_cutoff = config['dataset']['pocket_cutoff']
    # dataset_size = config['dataset']['dataset_size']
    load_data = config['dataset']['load_data']
    subsample_pharms = config['dataset']['subsample_pharms']
    graph_cutoffs = config['dataset']['graph_cutoffs']
    dataset = ProteinPharmacophoreDataset(name= 'PROTPHARMTRAIN', data_files=data_files, processed_data_dir=processed_data_dir, rec_file=rec_file, 
                prot_elements=prot_elements, load_data=load_data, subsample_pharms=subsample_pharms, graph_cutoffs=graph_cutoffs)
    
    data_module = ProtPharmDataModule(dataset=dataset, batch_size=config['training']['batch_size'], num_workers=config['training']['num_workers'])
    return data_module