from math import ceil
from pathlib import Path
from typing import Dict, List, Tuple

import dgl
import dgl.function as dglfn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch_scatter import segment_coo, segment_csr

from losses.dist_hinge_loss import DistanceHingeLoss
from models.dynamics_gvp import PharmRecDynamicsGVP
from models.n_nodes_dist import PharmSizeDistribution
from utils import get_batch_info, get_nodes_per_batch, copy_graph, get_batch_idxs
from torch_scatter import segment_csr
import lightning as L

class PharmacophoreDiff(L.LightningModule):

    def __init__(self, pharm_nf, rec_nf, processed_dataset_dir: Path, n_timesteps: int = 1000, graph_config={}, dynamics_config = {}, precision=1e-4, pharm_feat_norm_constant=1, pf_dist_threshold=0, use_fake_atoms=False):
        super().__init__()
        self.n_pharm_feats = pharm_nf
        self.n_prot_feats = rec_nf
        self.n_timesteps = n_timesteps
        self.pharm_feat_norm_constant = pharm_feat_norm_constant
        self.use_fake_atoms = use_fake_atoms

        if pf_dist_threshold > 0:
            self.apply_pf_hinge_loss = True
            self.pf_hinge_loss_fn = DistanceHingeLoss(pf_dist_threshold)
        else:
            self.apply_pf_hinge_loss = False

        #TODO implement obtaining the pharmacophore size distribution from the dataset
        self.pharm_size_dist = PharmSizeDistribution(processed_dataset_dir)

        # create noise schedule and dynamics model
        self.gamma = PredefinedNoiseSchedule(noise_schedule='polynomial_2', timesteps=n_timesteps, precision=precision)

        self.dynamics = PharmRecDynamicsGVP(pharm_nf,rec_nf,**graph_config,**dynamics_config)
    
    def normalize(self, protpharm_graphs: dgl.DGLHeteroGraph):
        protpharm_graphs.nodes['pharm'].data['h_0'] = protpharm_graphs.nodes['pharm'].data['h_0'] / self.pharm_feat_norm_constant
        return protpharm_graphs
    
    def unnormalize(self, protpharm_graphs: dgl.DGLHeteroGraph):
        protpharm_graphs.nodes['pharm'].data['h_0'] = protpharm_graphs.nodes['pharm'].data['h_0'] * self.pharm_feat_norm_constant
        return protpharm_graphs
    
    def remove_com(self, protpharm_graphs, pharm_batch_idx, prot_batch_idx, com: str = None):
        """Remove center of mass from ligand atom positions and receptor keypoint positions.

        This method can remove either the ligand COM, receptor keypoint COM or the complex COM.
        """               
        if com is None:
            raise NotImplementedError('removing COM of receptor/ligand complex not implemented')
        elif com == 'pharmacophore':
            ntype = 'pharm'
        elif com == 'protein':
            ntype = 'prot'
        else:
            raise ValueError(f'invalid value for com: {com=}')
        
        com = dgl.readout_nodes(protpharm_graphs, feat='x_0', ntype=ntype, op='mean')

        protpharm_graphs.nodes['pharm'].data['x_0'] = protpharm_graphs.nodes['pharm'].data['x_0'] - com[pharm_batch_idx]
        protpharm_graphs.nodes['prot'].data['x_0'] = protpharm_graphs.nodes['prot'].data['x_0'] - com[prot_batch_idx]
        return protpharm_graphs
    
    def noised_representation(self, g: dgl.DGLHeteroGraph, pharm_batch_idx: torch.Tensor, prot_batch_idx: torch.Tensor,
                              eps: Dict[str, torch.Tensor], gamma_t: torch.Tensor):
        

        alpha_t = self.alpha(gamma_t)[pharm_batch_idx][:, None]
        sigma_t = self.sigma(gamma_t)[pharm_batch_idx][:, None]

        g.nodes['pharm'].data['x_0'] = alpha_t*g.nodes['pharm'].data['x_0'] + sigma_t*eps['x']
        g.nodes['pharm'].data['h_0'] = alpha_t*g.nodes['pharm'].data['h_0'] + sigma_t*eps['h']
        

        # remove ligand COM from the system
        g = self.remove_com(g, pharm_batch_idx, prot_batch_idx, com='pharmacophore')
        
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

        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s
    
    def encode_receptors(self, g: dgl.DGLHeteroGraph):
        #it seems all I need to do for now is add vector features

        device = g.device
        # get batch indicies of every ligand and keypoint - useful later
        batch_idx = torch.arange(g.batch_size, device=device)
        prot_batch_idx = batch_idx.repeat_interleave(g.batch_num_nodes('prot'))
        batch_idxs = get_batch_idxs(g)

        #add vector features to proteins
        g.nodes['prot'].data['v_0'] = torch.zeros((g.num_nodes('prot'), self.n_vec_feats, 3), device=device)

        return g
    
    def forward(self, protpharm_graphs: dgl.DGLHeteroGraph):

        losses = {}
        
        #normalize values
        protpharm_graphs = self.normalize(protpharm_graphs)

        batch_size = protpharm_graphs.batch_size
        device = protpharm_graphs.device
        
        # encode receptors
        protpharm_graphs = self.encode_receptors(protpharm_graphs)

        batch_idxs = get_batch_idxs(protpharm_graphs)

        # if we are applying the RL hinge loss, we will need to be able to put receptor atoms and the pharmacophore into the same
        # referance frame. in order to do this, we need the initial COM of the keypoints - TODO: need to check if this is necessary
        if self.apply_pf_hinge_loss:
            init_prot_com = dgl.readout_nodes(protpharm_graphs, feat='x_0', ntype='prot', op='mean')
        
        # remove pharmacophore COM from receptor/ligand complex
        protpharm_graphs = self.remove_com(protpharm_graphs, batch_idxs['pharm'], batch_idxs['prot'], com='pharmacophore')

        # sample timepoints for each item in the batch
        t = torch.randint(0, self.n_timesteps, size=(batch_size,), device=device).float() # timesteps
        t = t / self.n_timesteps

        # sample epsilon for each ligand
        eps = {
            'h':torch.randn(protpharm_graphs.nodes['pharm'].data['h_0'].shape, device=device),
            'x':torch.randn(protpharm_graphs.nodes['pharm'].data['x_0'].shape, device=device)
        }

        # construct noisy versions of the ligand
        gamma_t = self.gamma(t).to(device=device)
        protpharm_graphs = self.noised_representation(protpharm_graphs, batch_idxs['pharm'], batch_idxs['prot'], eps, gamma_t)

        # predict the noise that was added
        #TODO predict feature class instead of noise, cross entropy loss on feature type
        eps_h_pred, eps_x_pred = self.dynamics(protpharm_graphs, t, batch_idxs)

        # compute hinge loss if necessary
        if self.apply_pf_hinge_loss:

            with protpharm_graphs.local_scope():

                # predict denoised ligand
                g_denoised = self.denoised_representation(protpharm_graphs, batch_idxs['pharm'], batch_idxs['prot'], eps_x_pred, eps_h_pred, gamma_t)

                # translate ligand back to intitial frame of reference
                g_denoised = self.remove_com(g_denoised, batch_idxs['pharm'], batch_idxs['prot'], com='protein')
                g_denoised.nodes['pharm'].data['x_0'] = g_denoised.nodes['pharm'].data['x_0'] + init_prot_com[batch_idxs['pharm']]

                # compute hinge loss between ligand atom position and receptor atom positions
                pf_hinge_loss = 0
                for g in dgl.unbatch(g_denoised):
                    denoised_pharm_pos = g.nodes['pharm'].data['x_0']
                    prot_atom_pos = g.nodes['prot'].data['x_0']
                    pf_hinge_loss += self.pf_hinge_loss_fn(denoised_pharm_pos, prot_atom_pos)

                losses['rl_hinge'] = pf_hinge_loss

        #compute l2 loss on noise
        if self.use_fake_atoms:
            # real_atom_mask = torch.concat([ ~(lig_feat[:, -1].bool()) for lig_feat in lig_atom_features ])[:, None]
            real_atom_mask = protpharm_graphs.nodes['pharm'].data['h_0'][:, -1:].bool()
            n_real_atoms = real_atom_mask.sum()
            n_x_loss_terms = n_real_atoms*3
            x_loss = ((eps['x'] - eps_x_pred)*real_atom_mask).square().sum() # mask out loss on predicted position of fake atoms

        else:
            x_loss = ((eps['x'] - eps_x_pred)).square().sum()
            n_x_loss_terms = eps['x'].numel()

        h_loss = (eps['h'] - eps_h_pred).square().sum()
        losses['l2'] = (x_loss + h_loss) / (n_x_loss_terms + eps['h'].numel())

        losses['pos'] = x_loss / n_x_loss_terms
        losses['feat'] = h_loss / eps['h'].numel()

        return losses
    
    

        


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
