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

from losses.dist_hinge_loss import DistanceHingeLoss
from models.dynamics_gvp import PharmRecDynamicsGVP
from models.n_nodes_dist import PharmSizeDistribution
# from models.scheduler import LRScheduler
from utils import get_batch_info, get_nodes_per_batch, copy_graph, get_batch_idxs
from torch_scatter import segment_csr
import pytorch_lightning as pl


class PharmacophoreDiff(pl.LightningModule):

    def __init__(self, pharm_nf, rec_nf, processed_data_dir: Path, n_timesteps: int = 1000, graph_config={}, dynamics_config = {}, lr_scheduler_config = {}, precision=1e-4, pharm_feat_norm_constant=1, endpoint_param_feat: bool = False, endpoint_param_coord: bool = False, weighted_loss: bool = False, **kwargs):
        super().__init__()
        self.n_pharm_feats = pharm_nf
        self.n_prot_feats = rec_nf
        self.n_timesteps = n_timesteps
        self.pharm_feat_norm_constant = pharm_feat_norm_constant
        self.endpoint_param_feat = endpoint_param_feat
        self.endpoint_param_coord = endpoint_param_coord
        self.weighted_loss = weighted_loss

        #TODO implement obtaining the pharmacophore size distribution from the dataset
        self.pharm_size_dist = PharmSizeDistribution(processed_data_dir)

        # create noise schedule and dynamics model
        self.gamma = PredefinedNoiseSchedule(noise_schedule='polynomial_2', timesteps=n_timesteps, precision=precision)

        self.dynamics = PharmRecDynamicsGVP(pharm_nf,rec_nf,**graph_config,**dynamics_config)

        self.lr_scheduler_config = lr_scheduler_config
        # self.n_vec_feats = n_vec_feats

        self.save_hyperparameters()
    
    def normalize(self, protpharm_graphs: dgl.DGLHeteroGraph):
        protpharm_graphs.nodes['pharm'].data['h_0'] = protpharm_graphs.nodes['pharm'].data['h_0'] / self.pharm_feat_norm_constant
        return protpharm_graphs
    
    def unnormalize(self, protpharm_graphs: dgl.DGLHeteroGraph):
        protpharm_graphs.nodes['pharm'].data['h_0'] = protpharm_graphs.nodes['pharm'].data['h_0'] * self.pharm_feat_norm_constant
        return protpharm_graphs
    
    def remove_com(self, protpharm_graphs, pharm_batch_idx, prot_batch_idx, com: str = None, pharm_feat='x_t'):
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
    
    def noised_representation(self, g: dgl.DGLHeteroGraph, pharm_batch_idx: torch.Tensor, prot_batch_idx: torch.Tensor,
                              eps: Dict[str, torch.Tensor], gamma_t: torch.Tensor, return_com: bool = False ):
        

        alpha_t = self.alpha(gamma_t)[pharm_batch_idx][:, None]
        sigma_t = self.sigma(gamma_t)[pharm_batch_idx][:, None]

        g.nodes['pharm'].data['x_t'] = alpha_t*g.nodes['pharm'].data['x_0'] + sigma_t*eps['x']
        g.nodes['pharm'].data['h_t'] = alpha_t*g.nodes['pharm'].data['h_0'] + sigma_t*eps['h']
        
        # sampled_com = dgl.readout_nodes(g, feat='x_0', ntype='pharm', op='mean')
        # remove pharmacophore COM from the system
        # g = self.remove_com(g, pharm_batch_idx, prot_batch_idx, com='pharmacophore')
        
        if return_com:
            raise NotImplementedError('i dont think we need to do this so im making it an error for now')
            return g, sampled_com
        else:
            return g
    
    def denoised_representation(self, g: dgl.DGLHeteroGraph, pharm_batch_idx: torch.Tensor, prot_batch_idx: torch.Tensor,
                              eps_x_pred: torch.Tensor, eps_h_pred: torch.Tensor, gamma_t: torch.Tensor):
        # assuming the input ligand COM is zero, we compute the denoised verison of the ligand
        alpha_t = self.alpha(gamma_t)[pharm_batch_idx][:, None]
        sigma_t = self.sigma(gamma_t)[pharm_batch_idx][:, None]

        g.nodes['pharm'].data['x_0'] = (g.nodes['pharm'].data['x_0'] - sigma_t*eps_x_pred)/alpha_t
        g.nodes['pharm'].data['h_0'] = (g.nodes['pharm'].data['h_0'] - sigma_t*eps_h_pred)/alpha_t

        return g
    
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
    
    # def encode_receptors(self, g: dgl.DGLHeteroGraph):
    #     #it seems all I need to do for now is add vector features

    #     device = g.device
    #     # get batch indicies of every ligand and keypoint - useful later
    #     batch_idx = torch.arange(g.batch_size, device=device)
    #     prot_batch_idx = batch_idx.repeat_interleave(g.batch_num_nodes('prot'))
    #     batch_idxs = get_batch_idxs(g)

    #     #add vector features to proteins
    #     g.nodes['prot'].data['v_0'] = torch.zeros((g.num_nodes('prot'), self.vector_size, 3), device=device)

    #     return g
    
    def forward(self, g: dgl.DGLHeteroGraph, phase: str='train'):

        losses = {}
        metrics ={}
        
        #normalize values
        g = self.normalize(g)

        batch_size = g.batch_size
        device = g.device
        
        # # encode receptors
        # g = self.encode_receptors(g)

        batch_idxs = get_batch_idxs(g)
        
        # remove pharmacophore COM from protein-pharmacophore graph
        g = self.remove_com(g, batch_idxs['pharm'], batch_idxs['prot'], com='pharmacophore', pharm_feat='x_0')

        # sample timepoints for each item in the batch
        t = torch.randint(0, self.n_timesteps, size=(batch_size,), device=device).float() # timesteps
        t = t / self.n_timesteps

        # sample epsilon for each ligand
        eps = {
            'h':torch.randn(g.nodes['pharm'].data['h_0'].shape, device=device),
            'x':torch.randn(g.nodes['pharm'].data['x_0'].shape, device=device)
        }

        # construct noisy versions of the ligand
        gamma_t = self.gamma(t).to(device=device)
        g = self.noised_representation(g, batch_idxs['pharm'], batch_idxs['prot'], eps, gamma_t,)

        # predict the noise that was added
        h_dyn, x_dyn = self.dynamics(g, t, batch_idxs)

        alpha_t = self.alpha(gamma_t)[batch_idxs['pharm']][:, None]
        sigma_t = self.sigma(gamma_t)[batch_idxs['pharm']][:, None]

        if self.endpoint_param_feat:
            h_0_pred = h_dyn
            h_loss=fn.cross_entropy(h_0_pred, g_copy.nodes['pharm'].data['h_0'].argmax(dim=1),reduction='none')
        else:
            h_loss = (eps['h'] - h_dyn).square().sum(dim=1)
            h_0_pred = (g.nodes['pharm'].data['h_t'] - sigma_t*h_dyn)/alpha_t
        if self.endpoint_param_coord:
            x_0_pred = x_dyn
            x_loss = ((x_0_pred - g_copy.nodes['pharm'].data['x_0'])).square().sum(dim=1)    
        else:
            x_loss = ((eps['x'] - x_dyn)).square().sum(dim=1)
            #get prediction on original frame of reference
            x_0_pred = (g.nodes['pharm'].data['x_t'] - sigma_t*x_dyn)/alpha_t 
        
        weight_metric=1 - t[batch_idxs['pharm']]
        weight_loss=torch.ones_like(t)[batch_idxs['pharm']]
        if self.weighted_loss:
            weight_loss=weight_metric

        h_loss = (h_loss * weight_loss).sum()
        x_loss = (x_loss * weight_loss).sum()


        #TODO kl divergence loss
        losses[phase + ' pos loss'] = x_loss / eps['x'].numel()
        losses[phase + ' feat loss'] = h_loss / eps['h'].numel()

        with torch.no_grad():
            metrics[phase + ' position error'] = (x_0_pred - g.nodes['pharm'].data['x_0']).square().sum(dim=1).mean()
            metrics[phase + ' weighted position error'] = (weight_metric*((x_0_pred - g.nodes['pharm'].data['x_0'])**2).sum(dim=1)).mean()
            h_0_pred=h_0_pred.argmax(dim=1)
            metrics[phase + ' accuracy'] = (h_0_pred == g.nodes['pharm'].data['h_0'].argmax(dim=1)).float().mean()
            metrics[phase + ' weighted accuracy'] = (weight_metric*(h_0_pred == g.nodes['pharm'].data['h_0'].argmax(dim=1)).float()).mean()
            # if 'prot_ph' in g.ntypes:
            #     prot_ph_batch_idx = batch_idxs['prot_ph']
            #     print(g_copy.nodes['prot_ph'].data['x_0'].shape,g_copy.nodes['prot_ph'].data['h_0'].shape)
        
        return losses,metrics
    
    def num_training_batches(self):
        return len(self.trainer.train_dataloader)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), 
                               lr=self.lr_scheduler_config['base_lr'],
                               weight_decay=self.lr_scheduler_config['weight_decay']
        )
        scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,**self.lr_scheduler_config['reducelronplateau'])

        # self.lr_scheduler = LRScheduler(model=self, optimizer=optimizer, **self.lr_scheduler_config)
        return {'optimizer': optimizer, 'lr_scheduler': {"scheduler":scheduler,"monitor": self.lr_scheduler_config['monitor'], "interval": self.lr_scheduler_config['interval'], "frequency": self.lr_scheduler_config['frequency']}}

    def training_step(self, batch, batch_idx):
        protpharm_graphs = batch
        phase='train'
        #current_epoch = self.current_epoch + batch_idx / self.num_training_batches()
        # self.lr_scheduler.step_lr(current_epoch, self.optimizers())
        loss_dict,metrics_dict = self.forward(protpharm_graphs,phase=phase)
        loss_dict[phase+' total loss'] = torch.sum(torch.stack(list(loss_dict.values()), dim=0))
        metrics_dict[phase+' total error']= metrics_dict[phase+' position error'] + 1 - metrics_dict[phase+' accuracy']
        metrics_dict[phase+' weighted total error']= metrics_dict[phase+' weighted position error'] + 1 - metrics_dict[phase+' weighted accuracy']
        try:
            metrics_dict['lr'] = self.lr_schedulers().get_last_lr()[0]
        except:
            metrics_dict['lr'] =self.optimizers().param_groups[0]['lr']
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True,logger=True, batch_size=protpharm_graphs.batch_size)
        self.log_dict(metrics_dict, on_step=True, on_epoch=True, prog_bar=True,logger=True, batch_size=protpharm_graphs.batch_size)
        return loss_dict[phase+' total loss']
    
    def validation_step(self, batch, batch_idx):
        protpharm_graphs = batch
        phase='val'
        loss_dict, metrics_dict = self.forward(protpharm_graphs,phase=phase)
        loss_dict[phase+' total loss'] = sum(list(loss_dict.values()))
        metrics_dict[phase+' total error']= metrics_dict[phase+' position error'] + 1 - metrics_dict[phase+' accuracy']
        metrics_dict[phase+' weighted total error']= metrics_dict[phase+' weighted position error'] + 1 - metrics_dict[phase+' weighted accuracy']
        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=True,logger=True, batch_size=batch.batch_size)
        self.log_dict(metrics_dict, on_step=False, on_epoch=True, prog_bar=True,logger=True, batch_size=batch.batch_size)
        return loss_dict[phase+' total loss']
    
    
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
    
    def sample_p_zs_given_zt(self, s: torch.Tensor, t: torch.Tensor, g: dgl.heterograph, batch_idxs: Dict[str, torch.Tensor]):

        device=g.device
        pharm_batch_idx=batch_idxs['pharm']
        prot_batch_idx=batch_idxs['prot']

        # compute the alpha and sigma terms that define p(z_s | z_t)
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s,alpha_s = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s)
        sigma_s = self.sigma(gamma_s)
        sigma_t = self.sigma(gamma_t)

        # predict the noise that we should remove from this example, epsilon
        pred_h, pred_x = self.dynamics(g, t, batch_idxs)

        var_terms = sigma2_t_given_s / alpha_t_given_s / sigma_t

        #expand distribution parameters by batch assignment for every pharmacophore feature
        alpha_t_given_s = alpha_t_given_s[pharm_batch_idx].view(-1, 1)
        var_terms = var_terms[pharm_batch_idx].view(-1, 1)

        #TODO: check this math lol
        #compute the mean (mu) for positions/features of the distribution p(z_s | z_t)
        if self.endpoint_param_coord:
            mu_pos = (alpha_t_given_s*(sigma_s**2)/(sigma_t**2))*g.nodes['pharm'].data['x_t'] + (alpha_s*sigma2_t_given_s/(sigma_t**2))*pred_x
        else:
            mu_pos = g.nodes['pharm'].data['x_t'] /alpha_t_given_s - var_terms*pred_x
        if self.endpoint_param_feat:
            mu_feat = (alpha_t_given_s*(sigma_s**2)/(sigma_t**2))*g.nodes['pharm'].data['h_t'] + (alpha_s*sigma2_t_given_s/(sigma_t**2))*pred_h
        else:
            mu_feat = g.nodes['pharm'].data['h_t'] /alpha_t_given_s - var_terms*pred_h

        #compute sigma for p(zs|zt)
        sigma = sigma_t_given_s * sigma_s/sigma_t
        sigma = sigma[pharm_batch_idx].view(-1, 1)

        # sample zs given the mu and sigma we just computed
        pos_noise = torch.randn(g.nodes['pharm'].data['x_t'].shape, device=device)
        feat_noise = torch.randn(g.nodes['pharm'].data['h_t'].shape, device=device)
        g.nodes['pharm'].data['x_t'] = mu_pos + sigma*pos_noise
        g.nodes['pharm'].data['h_t'] = mu_feat + sigma*feat_noise

        #remove pharmacophore COM from system
        g = self.remove_com(g, pharm_batch_idx, prot_batch_idx, com='pharmacophore')

        return g


    
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
            
        g = self.remove_com(g, batch_idxs['pharm'], batch_idxs['prot'], com='protein', pharm_feat='x_0')

        for n_type in ['pharm', 'prot']:
            g.nodes[n_type].data['x_0'] = g.nodes[n_type].data['x_0'] + init_prot_com[batch_idxs[n_type]]

        g=self.unnormalize(g)

        if visualize_trajectory:

            pharm_pos_frames = list(zip(*pharm_pos_frames))
            pharm_feat_frames = list(zip(*pharm_feat_frames))

            return pharm_pos_frames, pharm_feat_frames

        pharm_pos=[]
        pharm_feat=[]

        g=g.to('cpu')
        for g_i in dgl.unbatch(g):
            pharm_pos.append([g_i.nodes['pharm'].data['x_0']])
            pharm_feat.append([g_i.nodes['pharm'].data['h_0']])

        return pharm_pos, pharm_feat
    
    def sample(self,  ref_graphs: List[dgl.DGLHeteroGraph], n_pharm_feats: List[List[int]], diff_batch_size: int = 32, use_ref_pharm_com: bool = False, visualize_trajectory: bool=False):

        device = ref_graphs[0].device
        n_receptors = len(ref_graphs)

        ref_graphs_batched = dgl.batch(ref_graphs)
        graphs=[]

        for rec_idx, ref_graph in enumerate(dgl.unbatch(ref_graphs_batched)):
            n_pharm_feats_rec = n_pharm_feats[rec_idx]
            g_copies=copy_graph(ref_graph,n_copies=len(n_pharm_feats_rec),pharm_feats_per_copy=torch.tensor(n_pharm_feats_rec))
            graphs.extend(g_copies)
        
        #proceed to batched sampling

        n_complexes=len(graphs)
        n_complexes_sampled=0

        pharm_pos, pharm_feat=[],[]

        for batch_idx in range(ceil(n_complexes/diff_batch_size)):

            #number of complexes present in batch
            n_samples_batch=min(diff_batch_size,n_complexes-n_complexes_sampled)
            start_idx=batch_idx*diff_batch_size
            end_idx=start_idx+n_samples_batch
            batch_graphs=dgl.batch(graphs[start_idx:end_idx])
            if use_ref_pharm_com:
                init_pharm_com=dgl.readout_nodes(ref_graphs_batched,feat='x_0',ntype='pharm',op='mean')
            else:
                init_pharm_com=None
            
            batch_pharm_pos, batch_pharm_feat = self.sample_given_receptor(batch_graphs, init_pharm_com=init_pharm_com, visualize_trajectory=visualize_trajectory)
            pharm_pos.extend(batch_pharm_pos)
            pharm_feat.extend(batch_pharm_feat)

            n_complexes_sampled+=n_samples_batch

        samples=[]
        end_idx=0
        for rec_idx in range(n_receptors):
            n_pharmacophores=len(n_pharm_feats[rec_idx])
            start_idx=end_idx
            end_idx=start_idx+n_pharmacophores
            samples.append({'positions': pharm_pos[start_idx:end_idx],'features': pharm_feat[start_idx:end_idx]})
        
        return samples


# noise schedules are taken from DiffSBDD: https://github.com/arneschneuing/DiffSBDD
def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1.
    This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def polynomial_schedule(timesteps: int, s=1e-4, power=3.):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power))**2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


# taken from DiffSBDD: https://github.com/arneschneuing/DiffSBDD
class PredefinedNoiseSchedule(nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined
    (non-learned) noise schedules.
    """
    def __init__(self, noise_schedule, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            alphas2 = cosine_beta_schedule(timesteps)
        elif 'polynomial' in noise_schedule:
            splits = noise_schedule.split('_')
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        else:
            raise ValueError(noise_schedule)

        # print('alphas2', alphas2)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        # print('gamma', -log_alphas2_to_sigmas2)

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]
