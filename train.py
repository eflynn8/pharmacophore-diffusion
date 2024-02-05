import argparse
from collections import defaultdict
from datetime import datetime
from distutils.util import strtobool
from pathlib import Path

import dgl
import numpy as np
import torch
import yaml

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

from pytorch_lightning.loggers import WandbLogger

from config_utils.cmdline import register_hyperparameter_args, merge_config_and_args

def parse_arguments():
    p = argparse.ArgumentParser()

    p.add_argument('--config', type=str, default=None)
    p.add_argument('--resume', default=None)

    # create a boolean argument for whether or not this is a debug run
    p.add_argument('--debug', action='store_true')
    p.add_argument('--seed', type=int, default=None)

    # create command line arguments for model hyperparameters
    p = register_hyperparameter_args(p)

    args = p.parse_args()

    if args.config is not None and args.resume is not None:
        raise ValueError('only specify a config file or a resume file but not both')

    return args
    
def main():

    args = parse_arguments()

    if args.resume is not None:
        # determine if we are resuming from a run directory or a checkpoint file
        if args.resume.is_dir():
            # we are resuming from a run directory
            # get the config file from the run directory
            run_dir = args.resume
            ckpt_file = str(run_dir / 'checkpoints' / 'last.ckpt')
        elif args.resume.is_file():
            run_dir = args.resume.parent.parent
            ckpt_file = str(args.resume)
        else:
            raise ValueError('resume argument must be a run directory or a checkpoint file that must already exist')
        
        config_file = run_dir / 'config.yaml'
    else:
        config_file = args.config
        ckpt_file = None

    # set seed
    if args.seed is not None:
        pl.seed_everything(args.seed)
    
    # process config file into dictionary
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # merge the config file with the command line arguments
    config = merge_config_and_args(config, args)

    # TODO: create dataset class (or, alternatively, a PL datamodule)


    # TODO: instantiate model


