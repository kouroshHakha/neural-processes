from math import pi
import random
import numpy as np

import torch
from torch.utils.data import Dataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class SineData(Dataset):
    """
    Dataset of functions f(x) = a * sin(x - b) where a and b are randomly
    sampled. The function is evaluated from -pi to pi.

    Parameters
    ----------
    amplitude_range : tuple of float
        Defines the range from which the amplitude (i.e. a) of the sine function
        is sampled.

    shift_range : tuple of float
        Defines the range from which the shift (i.e. b) of the sine function is
        sampled.

    num_samples : int
        Number of samples of the function contained in dataset.

    num_points : int
        Number of points at which to evaluate f(x) for x in [-pi, pi].
    """
    def __init__(self, dim=1, amplitude_range=(-1., 1.), shift_range=(-.5, .5), freq_range=(0.1, 10),
                 num_context_range=(3, 47), max_num_targets=100, num_random_functions=100, device=None, seed=0):
        self.amplitude_range = amplitude_range
        self.shift_range = shift_range
        self.num_context_range = num_context_range
        self.max_num_targets = max_num_targets
        self.num_random_functions = num_random_functions
        self.x_dim = dim
        self.y_dim = 1
        self.device = device

        set_seed(seed)

        # Generate data
        self.data = []
        a_min, a_max = amplitude_range
        b_min, b_max = shift_range
        f_min, f_max = freq_range
        for _ in range(num_random_functions):
            # Sample random amplitude
            a = (a_max - a_min) * torch.rand(1) + a_min
            # Sample random shift
            b = (b_max - b_min) * torch.rand(1) + b_min
            # Sample random frequ
            f = (f_max - f_min) * torch.rand(1) + f_min
            # Shape (num_points, x_dim)
            if self.x_dim == 1:
                xsum = x = torch.linspace(-pi, pi, max_num_targets).unsqueeze(1)
            else:
                # all x dims between -pi, pi
                x = 2 * pi * torch.rand(max_num_targets, self.x_dim) - pi
                # random weight matrix
                weight = torch.ones(self.x_dim, 1)
                # rand_inds = torch.randperm(len(weight))[:torch.randint(len(weight), size=())]
                # weight[rand_inds] = -1
                weight = weight / torch.linalg.norm(weight)
                xsum = x @ weight

            # Shape (num_points, y_dim)
            y = a * torch.sin(f * xsum - b)
            self.data.append((x, y))

    def __getitem__(self, index):
        x_lut, y_lut = self.data[index]
        shuffled_inds = torch.randperm(self.max_num_targets)

        amin, amax = self.num_context_range
        num_context = torch.randint(amin, amax + 1, size=())
        num_target = torch.randint(num_context, self.max_num_targets + 1, size=())

        # context is a subset of target (num_context < num_target)
        context_inds = shuffled_inds[:num_context]
        target_inds = shuffled_inds[:num_target]

        context_mask = torch.zeros(self.max_num_targets, dtype=torch.bool)
        context_mask[context_inds] = True

        target_mask = torch.zeros(self.max_num_targets, dtype=torch.bool)
        target_mask[target_inds] = True
        
        if self.device:
            d = self.device
            return x_lut.to(d), y_lut.to(d), context_mask.to(d), target_mask.to(d)

        return x_lut, y_lut, context_mask, target_mask

    def __len__(self):
        return self.num_random_functions