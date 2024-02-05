from torch.optim import Optimizer
import numpy as np
from pathlib import Path
from utils import save_model
from models.pharmacodiff import PharmacophoreDiff

class Scheduler:

    def __init__(self,
                 model: PharmacophoreDiff,
                 base_lr: float,
                 warmup_length: float = 0, 
                 restart_interval: float = 0, 
                 restart_type: str = 'linear'):
        
        self.model = model
        self.base_lr = base_lr
        self.restart_interval = restart_interval
        self.restart_type = restart_type
        self.warmup_length = warmup_length

        self.restart_marker = self.warmup_length

        if self.restart_type == 'linear':
            self.restart_fn = self.linear_restart
        elif self.restart_type == 'cosine':
            self.restart_fn = self.cosine_restart
        else:
            raise NotImplementedError

    def step_lr(self, epoch_exact,optimizer):

        if epoch_exact <= self.warmup_length and self.warmup_length != 0:
            optimizer.param_groups[0]['lr'] = self.base_lr*epoch_exact/self.warmup_length
            return
        
        if self.restart_interval == 0:
            return
        
        # TODO: account for the case where we are not doing restarts but we are doing something to the learning rate, such as an exponential decay

        # assuming we are out of the warmup phase and we are now doing restarts
        epochs_into_interval = epoch_exact - self.restart_marker
        if epochs_into_interval < self.restart_interval: # if we are within a restart interval
            optimizer.param_groups[0]['lr'] = self.restart_fn(epochs_into_interval)
        elif epochs_into_interval >= self.restart_interval:
            self.restart_marker = epoch_exact
            optimizer.param_groups[0]['lr'] = self.restart_fn(0)
            # save model on restart
            # model_file = self.output_dir / f'model_on_restart_{epoch_exact:.0f}.pt'
            # save_model(model_file)
    
    def linear_restart(self, epochs_into_interval):
        new_lr = -1.0*self.base_lr*epochs_into_interval/self.restart_interval + self.base_lr
        return new_lr

    def cosine_restart(self, epochs_into_interval):
        new_lr = 0.5*self.base_lr*(1+np.cos(epochs_into_interval*np.pi/self.restart_interval))
        return new_lr
    
    def get_lr(self,optimizer) -> float:
        return optimizer.param_groups[0]['lr']

        
