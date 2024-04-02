import argparse
from distutils.util import strtobool

def register_hyperparameter_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register hyperparameter arguments for the model."""

    # p.add_argument('--batch_size', type=int, default=None)
    # p.add_argument('--lr', type=float, default=None)
    # p.add_argument('--warmup_length', type=float, default=None)

    diff_group = p.add_argument_group('diffusion')
    diff_group.add_argument('--precision', type=float, default=None)
    diff_group.add_argument('--feat_norm_constant', type=float, default=None)
    diff_group.add_argument('--pf_dist_threshold', type=float, default=None, help='distance threshold for protein-pharmacophore loss function')

    dynamics_group = p.add_argument_group('dynamics')
    dynamics_group.add_argument('--vector_size', type=int, default=None)
    dynamics_group.add_argument('--n_convs', type=int, default=None, help='number of graph convolutions in the dynamics model')
    dynamics_group.add_argument('--n_hidden_scalars', type=int, default=None)
    dynamics_group.add_argument('--dropout', type=float, default=None)
    dynamics_group.add_argument('--h_skip_connections', type=bool, default=None)
    dynamics_group.add_argument('--agg_across_edge_types', type=bool, default=None)
    dynamics_group.add_argument('--dynamics_rec_enc_multiplier', type=int, default=None) 

    training_group = p.add_argument_group('training')
    training_group.add_argument('--pf_hinge_loss_weight', type=float, default=None, help='weight applied to protein-pharmacophore hinge loss')
    training_group.add_argument('--lr', type=float, default=None, help='base learning rate')
    training_group.add_argument('--weight_decay', type=float, default=None)
    training_group.add_argument('--clip_value', type=float, default=None, help='max gradient value for clipping')
    training_group.add_argument('--batch_size', type=int, default=None)
    training_group.add_argument('--warmup_length', type=float, default=None)
    training_group.add_argument('--restart_interval', type=float, default=None)
    training_group.add_argument('--restart_type', type=str, default=None)

    
    p.add_argument('--feature_norm', type=int, default=None)
    p.add_argument('--ff_cutoff', type=float, default=None)
    p.add_argument('--pf_cutoff', type=float, default=None)
    p.add_argument('--pp_cutoff', type=float, default=None)
    p.add_argument('--ff_k', type=int, default=None)
    p.add_argument('--pf_k', type=int, default=None)
    p.add_argument('--pp_k', type=int, default=None)

    p.add_argument('--max_fake_atom_frac', type=float, default=None)

    p.add_argument('--use_tanh', type=str, default=None)
    p.add_argument('--message_norm', type=str, default=None)

    p.add_argument('--exp_name', type=str, default=None)
    p.add_argument('--architecture', type=str, default=None)

    return p


def merge_config_and_args(config: dict, args: argparse.Namespace) -> dict:
    """Merge the model configuration with the command line arguments."""

    # override config file args with command line args
    args_dict = vars(args)
    dynamics_key = 'dynamics'

    if args.exp_name is not None:
        config['experiment']['name'] = args.exp_name

    if args.dropout is not None:
        config[dynamics_key]['dropout'] = args.dropout

    for arg_name in ['ff_k', 'pf_k']:
        if args_dict[arg_name] is not None:
            config[dynamics_key][arg_name] = args_dict[arg_name]
    
    for etype in ['ff', 'pp', 'pf']:
        if args_dict[f'{etype}_cutoff'] is not None:
            config['graph']['graph_cutoffs'][etype] = args_dict[f'{etype}_cutoff']
    
    if args.feature_norm is not None:
        check_bool_int(args.feature_norm)
    
    scheduler_args = ['warmup_length', 
                      'restart_interval', 
                      'restart_type']
    
    for scheduler_arg in scheduler_args:
        if args_dict[scheduler_arg] is not None:
            config['training']['scheduler'][scheduler_arg] = args_dict[scheduler_arg]

    if args.max_fake_atom_frac is not None:
        config['dataset']['max_fake_atom_frac'] = args.max_fake_atom_frac

    if args.use_tanh is not None:

        if args.use_tanh not in ["True", "False"]:
            raise ValueError()

        config['dynamics']['use_tanh'] = strtobool(args.use_tanh)

    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size

    if args.precision is not None:
        config['diffusion']['precision'] = args.precision

    if args.feat_norm_constant is not None:
        config['diffusion']['pharm_feat_norm_constant'] = args.feat_norm_constant

    if args.pf_dist_threshold is not None:
        config['diffusion']['pf_dist_threshold'] = args.pf_dist_threshold

    if args.message_norm is not None:

        if args.message_norm.isdecimal():
            args.message_norm = float(args.message_norm)
        config[dynamics_key]['message_norm'] = args.message_norm
    
    if args.n_convs is not None:
        config['dynamics']['n_convs'] = args.n_convs

    if args.h_skip_connections is not None:
        config['dynamics']['h_skip_connections'] = args.h_skip_connections

    if args.agg_across_edge_types is not None:
        config['dynamics']['agg_across_edge_types'] = args.agg_across_edge_types

    if args.n_hidden_scalars is not None:
       config[dynamics_key]['n_hidden_scalars'] = args.n_hidden_scalars

    if args.vector_size is not None:
       config[dynamics_key]['vector_size'] = args.vector_size

    if args.pf_hinge_loss_weight is not None:
        config['training']['pf_hinge_loss_weight'] = args.pf_hinge_loss_weight

    if args.lr is not None:
        config['training']['learning_rate'] = args.lr

    if args.weight_decay is not None:
        config['training']['weight_decay'] = args.weight_decay

    if args.clip_value is not None:
        config['training']['clip_value'] = args.clip_value
    

    return config


# TODO: there are built in functions for doing things like this
def check_bool_int(val):
    if val not in [0, 1]:
        raise ValueError