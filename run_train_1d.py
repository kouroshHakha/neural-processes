

import numpy as np
import matplotlib.pyplot as plt
import torch
from math import pi
from torch.utils.data import DataLoader

from utils import context_target_split
from datasets import SineData
from training import NeuralProcessTrainer
from neural_process import NeuralProcess



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create dataset
    dataset = SineData(amplitude_range=(-1., 1.),
                    shift_range=(-.5, .5),
                    num_samples=2000)


    x_dim = 1
    y_dim = 1
    r_dim = 50  # Dimension of representation of context points
    z_dim = 50  # Dimension of sampled latent variable
    h_dim = 50  # Dimension of hidden layers in encoder and decoder

    max_epochs = 30
    batch_size = 16
    num_context_range = (4,4)#(3, 47)
    num_extra_target_range = (4, 4)#(3, 50)

    neuralprocess = NeuralProcess(x_dim, y_dim, r_dim, z_dim, h_dim)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(neuralprocess.parameters(), lr=3e-4)
    np_trainer = NeuralProcessTrainer(device, neuralprocess, optimizer,
                                    num_context_range=num_context_range,
                                    num_extra_target_range=num_extra_target_range, 
                                    print_freq=200)

    neuralprocess.train = True
    np_trainer.train(data_loader, max_epochs)

    # Extract a batch from data_loader
    batch  = next(iter(data_loader))

    # Use batch to create random set of context points
    x, y = batch
    x_context, y_context, _, _ = context_target_split(x[0:1], y[0:1], 4, 4)

    # Create a set of target points corresponding to entire [-pi, pi] range
    x_target = torch.Tensor(np.linspace(-pi, pi, 100))
    x_target = x_target.unsqueeze(1).unsqueeze(0)

    neuralprocess.training = False

    for i in range(64):
        # Neural process returns distribution over y_target
        p_y_pred = neuralprocess(x_context, y_context, x_target)
        # Extract mean of distribution
        mu = p_y_pred.loc.detach()
        plt.plot(x_target.numpy()[0], mu.numpy()[0], 
                alpha=0.05, c='b')

    plt.scatter(x_context[0].numpy(), y_context[0].numpy(), c='k')

if __name__ == '__main__':
    main()
