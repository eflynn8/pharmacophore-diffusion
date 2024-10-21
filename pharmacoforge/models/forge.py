from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import dgl
import dgl.function as dglfn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
from torch_scatter import segment_coo, segment_csr

from pharmacoforge.losses.dist_hinge_loss import DistanceHingeLoss
from pharmacoforge.analysis.pharm_builder import SampledPharmacophore
from pharmacoforge.analysis.metrics import SampleAnalyzer
from pharmacoforge.models.pharmacodiff import PharmacoDiff
from pharmacoforge.models.pharmacoflow import PharmacoFlow
from pharmacoforge.models.n_nodes_dist import PharmSizeDistribution
from pharmacoforge.utils import get_batch_info, get_nodes_per_batch, copy_graph, get_batch_idxs
from pharmacoforge.utils.graph_ops import remove_com
from pharmacoforge.models.scheduler import LRScheduler

from torch_scatter import segment_csr
import pytorch_lightning as pl


class PharmacoForge(pl.LightningModule):

    def __init__(self, 
        pharm_nf, 
        rec_nf, 
        ph_type_map: List[str], 
        processed_data_dir: Path, 
        graph_config={}, 
        dynamics_config = {}, 
        lr_scheduler_config = {}, 
        sample_interval: float = 1,
        val_loss_interval: float = 1,
        batch_size: int = 64,
        pharms_per_pocket: int = 8,
        n_pockets_to_sample: int = 8,
        diffusion_config = {},
        fm_config = {},
        modality_loss_weights = {},
        model_class: str = 'diffusion' , # can be diffusion or flow-matching
        **kwargs):
        super().__init__()
        self.n_pharm_feats = pharm_nf
        self.n_prot_feats = rec_nf
        self.batch_size = batch_size
        self.ph_type_map = ph_type_map
        self.model_class = model_class
        self.modality_loss_weights = modality_loss_weights

        for loss_name in ['pos loss', 'feat loss']:
            if loss_name not in self.modality_loss_weights:
                self.modality_loss_weights[loss_name] = 1.0

        if model_class == 'diffusion':
        #TODO implement obtaining the pharmacophore size distribution from the dataset
            self.gen_model = PharmacoDiff(pharm_nf, 
                                          rec_nf, 
                                          ph_type_map, 
                                          processed_data_dir, 
                                          graph_config=graph_config, 
                                          dynamics_config=dynamics_config,  
                                          **diffusion_config)
        elif model_class == 'flow-matching':
            self.gen_model = PharmacoFlow(
                pharm_nf,
                rec_nf,
                ph_type_map,
                processed_data_dir,
                graph_config=graph_config,
                **fm_config
            )
            
        
        self.pharm_size_dist = PharmSizeDistribution(processed_data_dir)

        self.lr_scheduler_config = lr_scheduler_config
        
        # set parameters for training-time sampling
        self.sample_interval = sample_interval # how often to sample, in epochs
        self.pharms_per_pocket = pharms_per_pocket # how many pharmacophores to sample per pocket
        self.n_pockets_to_sample = n_pockets_to_sample # how many pockets to sample from
        self.last_sample_marker = 0 # marker for last time we sampled
        self.last_epoch_exact = 0
        self.val_loss_interval = val_loss_interval # how often to calculate val total loss

        self.save_hyperparameters()
        
      
    def forward(self, g: dgl.DGLHeteroGraph):

        outputs = self.gen_model(g)
        
        return outputs
    
    def num_training_batches(self):
        return len(self.trainer.train_dataloader)
    
    def num_training_steps(self):
        return ceil(len(self.trainer.datamodule.train_dataset) / self.batch_size)
    
    def set_lr_scheduler_frequency(self):
        self.lr_scheduler_config['frequency'] = int(self.num_training_steps() * self.val_loss_interval) + 1
    
    def configure_optimizers(self):
        try:
            weight_decay = self.lr_scheduler_config['weight_decay']
        except KeyError:
            weight_decay = 0

        optimizer = optim.Adam(self.parameters(), 
                               lr=self.lr_scheduler_config['base_lr'],
                               weight_decay=weight_decay
        )
        self.lr_scheduler = LRScheduler(model=self, optimizer=optimizer, **self.lr_scheduler_config)
        return {'optimizer': optimizer, }

    def training_step(self, batch, batch_idx):
        protpharm_graphs = batch
        phase='train'

        # compute the epoch as a float
        epoch_exact = self.current_epoch + batch_idx / self.num_training_batches()

        # step the learning rate scheduler
        self.lr_scheduler.step_lr(epoch_exact)
        
        # forward pass, get losses and metrics
        outputs = self.forward(protpharm_graphs)

        # compute total loss
        # TODO: loss weights
        # TODO: change naming convention for losses
        loss_names = [ key for key in outputs if 'loss' in key]
        total_loss = 0
        for loss_name in loss_names:
            total_loss += outputs[loss_name]
        outputs['total loss'] = total_loss

        # sample pharmacopphores and analyze them, if necessary
        if epoch_exact - self.last_sample_marker >= self.sample_interval:
            ph_quality_metrics = self.sample_and_analyze()
            outputs.update(ph_quality_metrics)
            self.last_sample_marker = epoch_exact

        # TODO: this isn't necessary, PL has a built-in way to log the learning rate, LearningRateCheckpoint or something like that
        try:
            outputs['lr'] = self.lr_schedulers().get_last_lr()[0]
        except:
            outputs['lr'] =self.optimizers().param_groups[0]['lr']

        # rename outputs to include phase
        output_keys = list(outputs.keys())
        for key in output_keys:
            val = outputs.pop(key)
            outputs[f'{phase} {key}'] = val

        # log epoch exact
        outputs['epoch_exact'] = epoch_exact

        self.log_dict(outputs, on_step=True, on_epoch=True, prog_bar=True,logger=True, batch_size=protpharm_graphs.batch_size)
        return outputs[phase+' total loss']
    
    def validation_step(self, batch, batch_idx):
        protpharm_graphs = batch
        phase='val'

        # forward pass, get losses and metrics
        outputs = self.forward(protpharm_graphs)

        # compute total loss
        # TODO: change naming convention for losses
        loss_names = [ key for key in outputs if 'loss' in key]
        total_loss = 0
        for loss_name in loss_names:
            try:
                total_loss += outputs[loss_name]*self.modality_loss_weights[loss_name]
            except KeyError:
                raise ValueError(f'Loss name {loss_name} not found in modality loss weights')
        outputs['total loss'] = total_loss

        # rename outputs to include phase
        output_keys = list(outputs.keys())
        for key in output_keys:
            val = outputs.pop(key)
            outputs[f'{phase} {key}'] = val

        # record epochs into training (enables easy alignemnt of training and validation metrics)
        outputs['epoch_exact'] = self.last_epoch_exact
        
        # log loss, metrics
        self.log_dict(outputs, on_step=False, on_epoch=True, prog_bar=True,logger=True, batch_size=batch.batch_size)

        return outputs[phase+' total loss']
    
    @torch.no_grad()
    def sample_and_analyze(self):
        """Samples pharamcophores and computes metrics on them, used during training."""
        val_dataset = self.trainer.datamodule.val_dataset

        # randomly choose self.n_pockets integers between 0 and len(val_dataset) without replacement
        pocket_idxs = torch.randint(low=0, high=len(val_dataset), size=(self.n_pockets_to_sample,))

        # get the actual pockets as dgl graphs
        pockets = [val_dataset[int(i)] for i in pocket_idxs]

        # construct tensor containing the number of pharmacophores to sample for each pocket
        n_pharms = []
        init_pharm_com = []
        for g in pockets:
            n_pharms.append([g.num_nodes('pharm')]*self.pharms_per_pocket)
            init_pharm_com.append(g.nodes['pharm'].data['x_0'].mean(dim=0)) # use reference pharmacophore COM

        init_pharm_com = torch.stack(init_pharm_com, dim=0)

        self.eval()

        # sample pharmacophores
        sampled_pharms = self.sample(pockets, 
                                     n_pharms, 
                                     max_batch_size=64, 
                                     init_pharm_com=init_pharm_com, 
                                     visualize_trajectory=False,
                                     n_timesteps=100, # as of now, this only applies to flow-matching model, diffusoion always users 1000 steps
        )

        self.train()


        # sampled pharms is a list of lists. each sublist contains the sampled pharmacophores for a single receptor. we need to flatten this list
        # for passage to the analysis function
        flat_pharms = []
        for pocket_pharms in sampled_pharms:
            flat_pharms.extend(pocket_pharms)

        # compute metrics on sampled pharmacophores
        metrics = SampleAnalyzer().analyze(flat_pharms)

        return metrics
        
    @torch.no_grad()
    def sample(self,  
               ref_graphs: List[dgl.DGLHeteroGraph], 
               n_pharms: List[List[int]], 
               max_batch_size: int = 32, 
               init_pharm_com: torch.Tensor = None, 
               visualize_trajectory: bool=False,
               n_timesteps: int = None,
               **kwargs
        ):
        """Samples pharmacophores for multiple receptors, allowing complete specification of the number of pharmacophores to sample for each pocket and the number of centers in each pharmacophore.

        Args:
            ref_graphs (List[dgl.DGLHeteroGraph]): List of DGL graphs containing the receptor structures.
            n_pharm (List[List[int]]): List of lists of integers containing the number of pharmacophore centers in each sampled pharmacophore for each receptor.
            max_batch_size (int, optional): Max batch size for sampling. Defaults to 32.
            init_pharm_com (torch.Tensor, optional): Tensor of shape (n_ref_graphs, 3) containing the initial pharmacophore COM for each receptor. Defaults to None. If None, the COM of the receptor will be used.
            visualize_trajectory (bool, optional): Whether to record each frame of the denoising process and return the trajectories. Defaults to False.

        Returns:
            List[List[SampledPharmacophore]]: A list of lists. Each sublist contains the sampled pharmacophores for a single receptor.
        """
        n_receptors = len(ref_graphs)

        if init_pharm_com is None:
            rec_coms = []
            for g in ref_graphs:
                rec_coms.append(g.nodes['prot'].data['x_0'].mean(dim=0))
            init_pharm_com = torch.stack(rec_coms, dim=0)

        # ref_graphs_batched = dgl.batch(ref_graphs)
        graphs = []
        graph_ref_idx = [] # the index of the referece graph that each graph in graphs was built from
        for rec_idx, ref_graph in enumerate(ref_graphs):
            n_pharms_rec = n_pharms[rec_idx]
            g_copies=copy_graph(ref_graph, n_copies=len(n_pharms_rec), pharm_feats_per_copy=torch.tensor(n_pharms_rec))
            graphs.extend(g_copies)
            graph_ref_idx.extend([rec_idx]*len(n_pharms_rec))
        
        #proceed to batched sampling
        n_complexes=len(graphs)
        n_complexes_sampled=0
        sampled_pharms = []
        for batch_idx in range(ceil(n_complexes/max_batch_size)):

            #number of complexes present in batch
            n_samples_batch = min(max_batch_size,n_complexes-n_complexes_sampled)
            start_idx = batch_idx*max_batch_size
            end_idx = start_idx+n_samples_batch
            batch_graphs = dgl.batch(graphs[start_idx:end_idx])

            init_coms = init_pharm_com[graph_ref_idx[start_idx:end_idx]]

            # move data to same device as model
            batch_graphs = batch_graphs.to(self.device)
            init_coms = init_coms.to(self.device)
            
            # sample pharmacophores
            batch_pharms = self.gen_model.sample(batch_graphs, 
                init_pharm_com=init_coms, 
                visualize=visualize_trajectory,
                n_timesteps=n_timesteps,
                **kwargs
            )

            sampled_pharms.extend(batch_pharms)

            n_complexes_sampled += n_samples_batch

        per_pocket_samples = []
        end_idx=0
        for rec_idx in range(n_receptors):
            n_pharmacophores=len(n_pharms[rec_idx])
            start_idx = end_idx
            end_idx = start_idx+n_pharmacophores
            per_pocket_samples.append(sampled_pharms[start_idx:end_idx])
        
        return per_pocket_samples


