
from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from npc.modules.dataset import SineData

if __name__ == '__main__':

    num_rows = 2
    num_cols = 3
    n_samples = num_rows * num_cols

    dataset = SineData(num_context_range=(3, 10), freq_range=(0.1, 10))

    # ######## test data loader
    # batch_size = 256
    # T = 1000
    # dloader = DataLoader(dataset, batch_size=batch_size)
    # batch = next(iter(dloader))

    # x, y, cmask, tmask = batch

    # assert x[cmask].shape[0] == y[cmask].shape[0]
    # assert cmask.sum(-1).shape[0] == min(batch_size, dataset.num_random_functions)

    # context_lens = []
    # for epoch in range(T):
    #     for x, y, cmask, tmask in dloader:
    #         context_lens.append(cmask.sum(-1).cpu().detach().numpy())
    # context_lens = np.concatenate(context_lens, 0)
    # assert set(context_lens) == set(np.arange(dataset.num_context_range[0], dataset.num_context_range[1]+1))
    

    ####### test correctness of the data
    _, axes = plt.subplots(num_rows, num_cols, sharex=True, sharey=True)
    axes = axes.flatten()
    for index in range(n_samples):
        x, y, context_mask, target_mask = dataset[index]

        x_context, y_context = x[context_mask], y[context_mask]
        x_target, y_target = x[target_mask], y[target_mask]

        axes[index].plot(x, y, color='b', label='gt')
        axes[index].scatter(x_target, y_target, color='b', s=15, label='tgt')
        axes[index].scatter(x_context, y_context, color='r', marker='x', s=25, label='ctx')

    plt.tight_layout()
    plt.legend()
    plt.savefig('test_sin_data_f0.1-10.png')

        
    # ####### test multi-dim aspect of the data

    # dataset = SineData(dim=2, num_context_range=(3, 10))
    # x, y, context_mask, target_mask = dataset[0]
    # breakpoint()