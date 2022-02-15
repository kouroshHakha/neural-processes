
from email.policy import default
import re
import math
from torch.utils.data import ConcatDataset
# Sine function experiments
from npc.modules.dataset import SineData
def get_sine_datasets(pargs):

    # setting out the defaults
    seed = pargs.seed
    dim = pargs.dim
    default_amp = (0.1, 1.)
    default_shift = (-math.pi, math.pi)
    default_freq = (0.5, 0.5)
    default_n_context = (3, 10)
    default_max_n_targets = 100
    default_n_random_functions = 5000 # max(500, 100 * pargs.dim)
    default_n_random_functions_valid = 100
    default_train_dataset = SineData(dim=dim, amplitude_range=default_amp, shift_range=default_shift, 
                                    freq_range=default_freq, num_context_range=default_n_context,
                                    max_num_targets=default_max_n_targets, num_random_functions=default_n_random_functions, seed=seed)

    default_valid_dataset = SineData(dim=dim, amplitude_range=default_amp, shift_range=default_shift, 
                                    freq_range=default_freq, num_context_range=default_n_context,
                                    max_num_targets=default_max_n_targets, num_random_functions=default_n_random_functions_valid, seed=seed)

    multi_fn_multi_pt = re.compile('fns_(\d+)_pts_(\d+)').match(pargs.exp)


    if pargs.exp == 'default':
        train_dataset = default_train_dataset
        valid_dataset = default_valid_dataset
    elif pargs.exp.startswith('amp'):
        if pargs.exp == 'amp_seen':
            train_dataset = SineData(dim=dim, amplitude_range=(0.3, 0.6), shift_range=default_shift, 
                                freq_range=default_freq, num_context_range=default_n_context,
                                max_num_targets=default_max_n_targets, num_random_functions=default_n_random_functions, seed=seed)
        elif pargs.exp == 'amp_unseen':
            # train_dataset = default_train_dataset
            train_dataset = ConcatDataset([
                SineData(dim=dim, amplitude_range=(0.1, 0.3), shift_range=default_shift, 
                        freq_range=default_freq, num_context_range=default_n_context,
                        max_num_targets=default_max_n_targets, num_random_functions=default_n_random_functions // 2, seed=seed),
                SineData(dim=dim, amplitude_range=(0.6, 1.), shift_range=default_shift, 
                        freq_range=default_freq, num_context_range=default_n_context,
                        max_num_targets=default_max_n_targets, num_random_functions=default_n_random_functions // 2, seed=seed),
            ])

        # valid_dataset = default_valid_dataset
        valid_dataset = SineData(dim=dim, amplitude_range=(0.3, 0.6), shift_range=default_shift, 
                                freq_range=default_freq, num_context_range=default_n_context,
                                max_num_targets=default_max_n_targets, num_random_functions=default_n_random_functions_valid, seed=seed)

    elif pargs.exp.startswith('shift'):
        if pargs.exp == 'shift_seen':
            train_dataset = SineData(dim=dim, amplitude_range=default_amp, shift_range=(-0.5*math.pi, 0.5*math.pi), 
                                freq_range=default_freq, num_context_range=default_n_context,
                                max_num_targets=default_max_n_targets, num_random_functions=default_n_random_functions, seed=seed)
        elif pargs.exp == 'shift_unseen':
            train_dataset = ConcatDataset([
                SineData(dim=dim, amplitude_range=default_amp, shift_range=(-math.pi, -0.5*math.pi), 
                        freq_range=default_freq, num_context_range=default_n_context,
                        max_num_targets=default_max_n_targets, num_random_functions=default_n_random_functions // 2, seed=seed),
                SineData(dim=dim, amplitude_range=default_amp, shift_range=(0.5*math.pi, math.pi), 
                        freq_range=default_freq, num_context_range=default_n_context,
                        max_num_targets=default_max_n_targets, num_random_functions=default_n_random_functions // 2, seed=seed),
            ])

        valid_dataset = SineData(dim=dim, amplitude_range=default_amp, shift_range=(-0.5*math.pi, 0.5*math.pi), 
                                freq_range=default_freq, num_context_range=default_n_context,
                                max_num_targets=default_max_n_targets, num_random_functions=default_n_random_functions_valid, seed=seed)

    elif pargs.exp == 'unseen_freq':
        # this doesn't work for now since NPs cannot fit high frequencies for some reason?
        train_dataset = ConcatDataset([
            SineData(dim=dim, amplitude_range=default_amp, shift_range=default_shift, 
                    freq_range=(0.1, 1), num_context_range=default_n_context,
                    max_num_targets=default_max_n_targets, num_random_functions=default_n_random_functions // 2, seed=seed),
            SineData(dim=dim, amplitude_range=default_amp, shift_range=default_shift, 
                    freq_range=(2, 3), num_context_range=default_n_context,
                    max_num_targets=default_max_n_targets, num_random_functions=default_n_random_functions // 2, seed=seed),
        ])

        valid_dataset = SineData(dim=dim, amplitude_range=default_amp, shift_range=default_shift, 
                                freq_range=(1, 2), num_context_range=default_n_context,
                                max_num_targets=default_max_n_targets, num_random_functions=default_n_random_functions_valid, seed=seed)
    ## For context experiments, we change the training context range 
    # but for validation we should sweep the context points from 1 -- n
    # this happens in evaluation mode, and training mode is not concerened with this.
    elif pargs.exp.startswith('context'):
        context_components = pargs.exp.split('_')
        if len(context_components) == 2:
            # context_10
            num_context = int(context_components[-1])
            train_dataset = SineData(dim=dim, amplitude_range=default_amp, shift_range=default_shift, 
                                    freq_range=default_freq, num_context_range=(num_context, num_context),
                                    max_num_targets=default_max_n_targets, num_random_functions=default_n_random_functions, seed=seed)
        elif len(context_components) == 3:
            # context_1_10
            min_num_context = int(context_components[-2])
            max_num_context = int(context_components[-1])
            train_dataset = SineData(dim=dim, amplitude_range=default_amp, shift_range=default_shift, 
                                    freq_range=default_freq, num_context_range=(min_num_context, max_num_context),
                                    max_num_targets=default_max_n_targets, num_random_functions=default_n_random_functions, seed=seed)
        valid_dataset = default_valid_dataset

    elif multi_fn_multi_pt:
        num_fns = int(multi_fn_multi_pt[1])
        num_pts = int(multi_fn_multi_pt[2])
        train_dataset = SineData(dim=dim, amplitude_range=default_amp, shift_range=default_shift, 
                                freq_range=default_freq, num_context_range=default_n_context,
                                max_num_targets=num_pts, num_random_functions=num_fns, seed=seed)

        valid_dataset = default_valid_dataset

    return train_dataset, valid_dataset


