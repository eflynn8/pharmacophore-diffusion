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
from pharmacoforge.models.n_nodes_dist import PharmSizeDistribution
from pharmacoforge.utils import get_batch_info, get_nodes_per_batch, copy_graph, get_batch_idxs
# from models.scheduler import LRScheduler

from torch_scatter import segment_csr
import pytorch_lightning as pl


class PharmacoForge(pl.LightningModule):

    def __init__(self, 
        pharm_nf, 
        rec_nf, 
        ph_type_map: List[str], 
        processed_data_dir: Path, 
        n_timesteps: int = 1000, 
        graph_config={}, 
        dynamics_config = {}, 
        lr_scheduler_config = {}, 
        sample_interval: float = 1,
        val_loss_interval: float = 1,
        batch_size: int = 64,
        pharms_per_pocket: int = 8,
        n_pockets_to_sample: int = 8,
        precision=1e-4, 
        pharm_feat_norm_constant=1, 
        endpoint_param_feat: bool = False, 
        endpoint_param_coord: bool = False, 
        weighted_loss: bool = False,
        remove_com: bool = True,
        model_class: str = 'diffusion' , # can be diffusion or flow-matching
        **kwargs):
        super().__init__()
        self.n_pharm_feats = pharm_nf
        self.n_prot_feats = rec_nf
        self.batch_size = batch_size
        self.ph_type_map = ph_type_map
        self.n_timesteps = n_timesteps
        self.remove_com = remove_com
        self.pharm_feat_norm_constant = pharm_feat_norm_constant
        self.endpoint_param_feat = endpoint_param_feat
        self.endpoint_param_coord = endpoint_param_coord
        self.weighted_loss = weighted_loss
        self.model_class = model_class

        if model_class == 'diffusion':
        #TODO implement obtaining the pharmacophore size distribution from the dataset
            self.gen_model = PharmacoDiff(pharm_nf, 
                                          rec_nf, 
                                          ph_type_map, 
                                          processed_data_dir, 
                                          n_timesteps, 
                                          graph_config, 
                                          dynamics_config, 
                                          precision, 
                                          pharm_feat_norm_constant, 
                                          endpoint_param_feat, 
                                          endpoint_param_coord, 
                                          weighted_loss, 
                                          remove_com, 
                                          **kwargs)
        elif model_class == 'flow-matching':
            raise NotImplementedError('Flow-matching model not implemented yet')
            
        
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
    
    def com_removal(self, protpharm_graphs, pharm_batch_idx, prot_batch_idx, com: str = None, pharm_feat='x_t'):
        """Remove center of mass from ligand atom positions and receptor keypoint positions.

        This method can remove either the ligand COM, receptor keypoint COM or the complex COM.
        """               
        if com is None:
            raise NotImplementedError('removing COM of receptor/ligand complex not implemented')
        elif com == 'pharmacophore':
            ntype = 'pharm'
            com_feat = pharm_feat
        elif com == 'protein':
            ntype = 'prot'
            com_feat = 'x_0'
        else:
            raise ValueError(f'invalid value for com: {com=}')
        
        com = dgl.readout_nodes(protpharm_graphs, feat=com_feat, ntype=ntype, op='mean')

        protpharm_graphs.nodes['pharm'].data[pharm_feat] = protpharm_graphs.nodes['pharm'].data[pharm_feat] - com[pharm_batch_idx]
        protpharm_graphs.nodes['prot'].data['x_0'] = protpharm_graphs.nodes['prot'].data['x_0'] - com[prot_batch_idx]
        return protpharm_graphs
    
    def sigma(self, gamma):
        """Computes sigma given gamma."""
        return torch.sqrt(torch.sigmoid(gamma))

    def alpha(self, gamma):
        """Computes alpha given gamma."""
        return torch.sqrt(torch.sigmoid(-gamma))

    def sigma_and_alpha_t_given_s(self, gamma_t, gamma_s):
        # this function is almost entirely copied from DiffSBDD

        sigma2_t_given_s = -torch.expm1(fn.softplus(gamma_s) - fn.softplus(gamma_t))

        log_alpha2_t = fn.logsigmoid(-gamma_t)
        log_alpha2_s = fn.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s
        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_s= torch.exp(0.5 * log_alpha2_s)
        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s, alpha_s
      
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
        optimizer = optim.Adam(self.parameters(), 
                               lr=self.lr_scheduler_config['base_lr'],
                               weight_decay=self.lr_scheduler_config['weight_decay']
        )
        scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,**self.lr_scheduler_config['reducelronplateau'])
        self.set_lr_scheduler_frequency()

        # self.lr_scheduler = LRScheduler(model=self, optimizer=optimizer, **self.lr_scheduler_config)
        return {'optimizer': optimizer, 'lr_scheduler': {"scheduler":scheduler,"monitor": self.lr_scheduler_config['monitor'], "interval": self.lr_scheduler_config['interval'], "frequency": self.lr_scheduler_config['frequency']}}

    def training_step(self, batch, batch_idx):
        protpharm_graphs = batch
        phase='train'

        # compute the epoch as a float
        epoch_exact = self.current_epoch + batch_idx / self.num_training_batches()
        
        # forward pass, get losses and metrics
        outputs = self.forward(protpharm_graphs,phase=phase)

        # compute total loss
        # TODO: loss weights
        # TODO: change naming convention for losses
        loss_names = [ key for key in outputs if 'loss' in key]
        total_loss = 0
        for loss_name in loss_names:
            total_loss += outputs[loss_name]

        # total error
        outputs[phase+' total error']= outputs[phase+' position error'] + 1 - outputs[phase+' accuracy']
        outputs[phase+' weighted total error']= outputs[phase+' weighted position error'] + 1 - outputs[phase+' weighted accuracy']

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
        outputs = self.forward(protpharm_graphs,phase=phase)

        # compute total loss
        # TODO: loss weights
        # TODO: change naming convention for losses
        loss_names = [ key for key in outputs if 'loss' in key]
        total_loss = 0
        for loss_name in loss_names:
            total_loss += outputs[loss_name]

        # total error
        outputs[phase+' total error']= outputs[phase+' position error'] + 1 - outputs[phase+' accuracy']
        outputs[phase+' weighted total error']= outputs[phase+' weighted position error'] + 1 - outputs[phase+' weighted accuracy']

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
        sampled_pharms = self.sample(pockets, n_pharms, max_batch_size=64, init_pharm_com=init_pharm_com, visualize_trajectory=False)

        self.train()


        # sampled pharms is a list of lists. each sublist contains the sampled pharmacophores for a single receptor. we need to flatten this list
        # for passage to the analysis function
        flat_pharms = []
        for pocket_pharms in sampled_pharms:
            flat_pharms.extend(pocket_pharms)

        # compute metrics on sampled pharmacophores
        metrics = SampleAnalyzer().analyze(flat_pharms)

        return metrics
        

    def get_pos_feat_for_visual(self, g:dgl.DGLHeteroGraph, init_prot_com: torch.Tensor, batch_idxs: Dict[str, torch.Tensor]):
        #make a copy of g
        g_frame=copy_graph(g,n_copies=1,batched_graph=True)[0]

        #unnormalize the features
        g_frame = self.unnormalize(g_frame)

        #move features back to initial frame of reference
        prot_com = dgl.readout_nodes(g_frame, feat='x_0', ntype='prot', op='mean')
        delta=init_prot_com-prot_com
        delta = delta[batch_idxs['pharm']]
        g_frame.nodes['pharm'].data['x_t'] = g_frame.nodes['pharm'].data['x_t']+delta

        g_frame=g_frame.to('cpu')
        pharm_pos, pharm_feat =[],[]
        for g_i in dgl.unbatch(g_frame):
            pharm_pos.append(g_i.nodes['pharm'].data['x_t'])
            pharm_feat.append(g_i.nodes['pharm'].data['h_t'])
        return pharm_pos, pharm_feat


    @torch.no_grad()
    def sample_given_receptor(self, g:dgl.DGLHeteroGraph, init_pharm_com: torch.Tensor = None, visualize_trajectory: bool = False):
        #method to sample from one receptor with batch_size being the number of pharmacophores generated for the receptor and n_pharm_feats 
        #being the number of pharmacophore features
        
        device = g.device
        batch_size = g.batch_size

        #get initial protein com 
        init_prot_com = dgl.readout_nodes(g, feat='x_0', ntype='prot', op='mean')

        #get batch indices of every node
        batch_idxs = get_batch_idxs(g)

        #Use the receptor pocket COM if pharmacophore COM not provided
        if init_pharm_com is None:
            init_pharm_com=init_prot_com

        #move the protein to the pharmacophore com
        g.nodes['prot'].data['x_0'] = g.nodes['prot'].data['x_0'] - init_pharm_com[batch_idxs['prot']]

        #sample initial positions/features of pharmacophore features
        g.nodes['pharm'].data['x_t'] = torch.randn(g.num_nodes('pharm'), 3, device=device)
        g.nodes['pharm'].data['h_t'] = torch.randn(g.num_nodes('pharm'), self.n_pharm_feats, device=device)

        if visualize_trajectory:

            pharm_pos_frames,pharm_feat_frames = [],[] 
            pharm_pos,pharm_feats = self.get_pos_feat_for_visual(g, init_prot_com, batch_idxs)
            pharm_pos_frames.append(pharm_pos)
            pharm_feat_frames.append(pharm_feats)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(self.n_timesteps)):
            s_arr=torch.full(size=(batch_size,),fill_value=s,device=device)
            t_arr=s_arr+1
            s_arr=s_arr.float()/self.n_timesteps
            t_arr=t_arr.float()/self.n_timesteps

            g=self.sample_p_zs_given_zt(s_arr, t_arr, g, batch_idxs)

            if visualize_trajectory:
                pharm_pos,pharm_feats = self.get_pos_feat_for_visual(g, init_prot_com, batch_idxs)
                pharm_pos_frames.append(pharm_pos)
                pharm_feat_frames.append(pharm_feats)

        # set the t=t features to t=0 features
        for feat in 'xh':
            g.nodes['pharm'].data[f'{feat}_0'] = g.nodes['pharm'].data[f'{feat}_t']
            
        g = self.com_removal(g, batch_idxs['pharm'], batch_idxs['prot'], com='protein', pharm_feat='x_0')

        for n_type in ['pharm', 'prot']:
            g.nodes[n_type].data['x_0'] = g.nodes[n_type].data['x_0'] + init_prot_com[batch_idxs[n_type]]

        g=self.unnormalize(g)

        if visualize_trajectory:
            # reshape trajectory frames

            pharm_pos_frames = list(zip(*pharm_pos_frames))
            pharm_feat_frames = list(zip(*pharm_feat_frames))

            trajs = []
            for batch_idx in range(len(pharm_pos_frames)):
                pos_frames = torch.stack(pharm_pos_frames[batch_idx], dim=0)
                feat_frames = torch.stack(pharm_feat_frames[batch_idx], dim=0)
                trajs.append((pos_frames, feat_frames))


        g=g.to('cpu')
        sampled_pharms: List[SampledPharmacophore] = []
        for gidx, g_i in enumerate(dgl.unbatch(g)):
            kwargs = {
                'g': g_i, 
                'pharm_type_map': self.ph_type_map,
            }
            if visualize_trajectory:
                kwargs['traj_frames'] = trajs[gidx]
            sampled_pharms.append(SampledPharmacophore(**kwargs))

        return sampled_pharms
    
    def sample(self,  ref_graphs: List[dgl.DGLHeteroGraph], n_pharms: List[List[int]], max_batch_size: int = 32, init_pharm_com: torch.Tensor = None, visualize_trajectory: bool=False):
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
            batch_pharms = self.sample_given_receptor(batch_graphs, init_pharm_com=init_coms, visualize_trajectory=visualize_trajectory)
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


