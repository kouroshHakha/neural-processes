
import argparse
from pprint import pprint
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np

from pytorch_lightning import seed_everything
import scipy
import torch

from npc.modules.exps import get_sine_datasets
from npc.modules.np import LightningNP
from npc.modules.utils import read_yaml, write_yaml


from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF

from debug import register_pdb_hook
register_pdb_hook()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_CONTEXT_LIST = [1, 3, 5, 10, 20, 50, 75, 100]
NEG_INF = -1e5

def parse_args():
    parser = argparse.ArgumentParser()
    # either provide ckpt or config
    # parser.add_argument('--config', type=str)
    # parser.add_argument('--dataset', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dim', type=int, default=1) # dimensionality of the problem
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--exp', type=str, default='default') # choices 'default', 'unseen_amp/shift/freq', 'context_max', 'context_min_max', 'fns_n_pts_p'

    return parser.parse_args()


def plot_prediction_examples(model: LightningNP, dataset, save_path, seed=10):
    ## plot the predictions
    num_rows = 2
    num_cols = 3
    n_samples = num_rows * num_cols
    
    seed_everything(seed)
    _, axes = plt.subplots(num_rows, num_cols, sharex=True, sharey=True)
    axes = axes.flatten()
    for index in range(n_samples):
        x, y, context_mask, _ = dataset[index]

        x_context, y_context = x[context_mask], y[context_mask]
        
        pred_y = model.np.predict(x_context.to(device), y_context.to(device), x.to(device))

        mu = pred_y.loc.detach().cpu().numpy().flatten()
        std = pred_y.scale.detach().cpu().numpy().flatten()
        
        xplot = x.cpu().detach().numpy().flatten()
        yplot = y.cpu().detach().numpy().flatten()
        xcplot = x_context.cpu().detach().numpy().flatten()
        ycplot = y_context.cpu().detach().numpy().flatten()

        axes[index].plot(xplot, yplot, color='k', label='gt')
        axes[index].plot(xplot, mu, color='b', label='pred')
        axes[index].fill_between(xplot, mu - std, mu + std, color='b', alpha=0.3)
        axes[index].scatter(xcplot, ycplot, color='k')

    plt.tight_layout()
    plt.savefig(Path(save_path) / 'pred.png')


def plot_incermental_extra_information(model: LightningNP, dataset, save_path, seed=10):
    ## plot the predictions
    num_rows = 3
    num_cols = 5
    n_samples = num_rows * num_cols
    
    seed_everything(seed)
    _, axes = plt.subplots(num_rows, num_cols, sharex=True, sharey=True)
    for row in range(num_rows):
        x, y, context_mask, _ = dataset[row]
        while context_mask.sum() < 5:
            x, y, context_mask, _ = dataset[row]

        x_context, y_context = x[context_mask], y[context_mask]
        rand_inds = torch.randperm(len(x_context))
        x_context, y_context = x_context[rand_inds], y_context[rand_inds]

        for col in range(num_cols):
            xc = x_context[:col+1]
            yc = y_context[:col+1]
            pred_y = model.np.predict(xc.to(device), yc.to(device), x.to(device))

            mu = pred_y.loc.detach().cpu().numpy().flatten()
            std = pred_y.scale.detach().cpu().numpy().flatten()
            
            xplot = x.cpu().detach().numpy().flatten()
            yplot = y.cpu().detach().numpy().flatten()
            xcplot = xc.cpu().detach().numpy().flatten()
            ycplot = yc.cpu().detach().numpy().flatten()

            axes[row][col].plot(xplot, yplot, color='k', label='gt')
            axes[row][col].plot(xplot, mu, color='b', label='pred')
            axes[row][col].fill_between(xplot, mu - std, mu + std, color='b', alpha=0.3)
            axes[row][col].scatter(xcplot, ycplot, color='k')

    plt.tight_layout()
    plt.savefig(Path(save_path) / 'pred_incremental.png')


def fit_and_predict_gp(x_train, y_train, x_test, y_test):
    kernel = ConstantKernel() * Matern()
    gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
    gpr = gp.fit(x_train, y_train)

    mu, std = gpr.predict(x_test, return_std=True)
    mu, std = mu.flatten(), std.flatten()

    # plt.close()
    # plt.plot(x_test.flatten(), mu, color='b')
    # plt.fill_between(x_test.flatten(), mu - std, mu + std, color='b', alpha=0.3)
    # plt.plot(x_test, y_test.flatten(), color='k')
    # plt.scatter(x_train.flatten(), y_train.flatten(), c='k')
    # plt.title('GP')
    # plt.savefig('debug_gp')

    probs = scipy.stats.norm.pdf(y_test.flatten(), mu, std)

    return np.log(probs), mu, std



def compute_np_ll(model: LightningNP, dataset, seed=0):
    seed_everything(seed)

    ll = []
    for x, y, context_mask, _ in dataset:
        x_context, y_context = x[context_mask], y[context_mask]
        pred_y = model.np.predict(x_context.to(device), y_context.to(device), x.to(device))
        ll.append(pred_y.log_prob(y.to(device)).mean().item())

    return float(np.mean(ll))

def compute_gp_ll(dataset, seed=0):

    ll = []
    for x, y, context_mask, _ in dataset:
        x_context, y_context = x[context_mask].numpy(), y[context_mask].numpy()
        log_p, _, _ = fit_and_predict_gp(x_context, y_context, x.numpy(), y.numpy())
        foo = np.ma.masked_invalid(log_p)
        foo[foo.mask] = NEG_INF
        ll.append(foo.mean(0).item())

    return float(np.mean(ll))


def get_np_ll_list(model, dataset):
    ll_list = []
    for x, y, _, _ in dataset:
        
        # to ensure matching validation between np and gp
        np.random.seed(0)
        inds_shuffeled = np.random.permutation(np.arange(len(x)))

        ll_row = []
        for n_context in N_CONTEXT_LIST:
            inds = inds_shuffeled[:n_context]
            x_context, y_context = x[inds], y[inds]

            pred_y = model.np.predict(x_context.to(device), y_context.to(device), x.to(device))
            ll_row.append(pred_y.log_prob(y.to(device)).mean().item())

        ll_list.append(ll_row)
    
    np_ll_vs_n_context = np.mean(ll_list, 0)
    return np_ll_vs_n_context.tolist()


def get_gp_ll_list(dataset):
    ll_list = []
    for x, y, _, _ in dataset:
        
        # to ensure matching validation between np and gp
        np.random.seed(0)
        inds_shuffeled = np.random.permutation(np.arange(len(x)))

        ll_row = []
        for n_context in N_CONTEXT_LIST:
            inds = inds_shuffeled[:n_context]
            x_context, y_context = x[inds], y[inds]

            log_p, _, _ = fit_and_predict_gp(x_context, y_context, x.numpy(), y.numpy())
            ll_row.append(log_p.mean().item())

        ll_list.append(ll_row)
    
    foo = np.ma.masked_invalid(ll_list)
    foo[foo.mask] = NEG_INF
    gp_ll_vs_n_context = foo.mean(0)
    # gp_ll_vs_n_context = np.mean(ll_list, 0)
    return gp_ll_vs_n_context.tolist()

# def get_dataset(dataset: str):

#     if dataset not in ['sin']:
#         # treat it as a path
#         ...
#     elif dataset == 'sin':
#         from npc.modules.dataset import SineData
#         return SineData(dim=1, amplitude_range=(-1., 1.), shift_range=(-0.5, -0.5), freq_range=(0.1, 1), num_context_range=(3, 10), device=device)

def main():
    pargs = parse_args()
    seed_everything(pargs.seed)

    # checkpoints = []
    # if pargs.config:
    #     config = read_yaml(pargs.config)
    # else:
    #     if not pargs.ckpt or not pargs.dataset:
    #         raise ValueError('Either provide a config file or the evaulation parameters.')
    #     config = {'checkpoints': [pargs.ckpt], 'dataset': pargs.dataset}

    # checkpoints = config['checkpoints']
    # dataset = get_dataset(config['dataset'])
    # if config['dataset'] == 'sine':
    #     _, dataset = get_sine_datasets(pargs)
    # else:
    #     raise NotImplementedError

    iid_dataset, ood_dataset = get_sine_datasets(pargs)
    x, y, _, _ = ood_dataset[0]

    # for name, ckpt_path in checkpoints.items():
    save_path = Path(pargs.ckpt).parent.parent
    print(f'Evaluating {str(save_path)} ...')
    # loading checkpoint
    model = LightningNP.load_from_checkpoint(pargs.ckpt)
    model.to(device)
    if x.ndim == 2 and x.shape[1] == 1:
        assert y.ndim == 2 and y.shape[1] == 1, 'Does not support non scalar output'
        plot_prediction_examples(model, ood_dataset, save_path)
        plot_incermental_extra_information(model, ood_dataset, save_path)

    print('Evaluatiing ll on the entire test (OOD) dataset for NPs ...')
    np_ll_ood = compute_np_ll(model, ood_dataset)
    print('Evaluatiing ll on the entire test (IID) dataset for NPs ...')
    # np_ll_iid = compute_np_ll(model, iid_dataset)
    np_ll_iid = None
    print(f'done -- np_ll_ood = {np_ll_ood}, np_ll_iid = {np_ll_iid}')

    print('Evaluating LL vs. # of Context points on test set (OOD) for NPs ...')
    np_ll_vs_n_context_ood = get_np_ll_list(model, ood_dataset) 
    print('Evaluating LL vs. # of Context points on test set (IID) for NPs ...')
    # np_ll_vs_n_context_iid = get_np_ll_list(model, iid_dataset) 
    np_ll_vs_n_context_iid = None
    print(f'done. \n ll_list_ood = {np_ll_vs_n_context_ood} \n ll_list_iid = {np_ll_vs_n_context_iid}')

    print('Evaluatiing ll on the entire test (OOD) dataset for GPs ...')
    gp_ll = compute_gp_ll(ood_dataset)
    print(f'done. -- gp_ll = {gp_ll}')

    print('Evaluating LL vs. # of Context points on test set (OOD) for GPs ...')
    gp_ll_vs_n_context = get_gp_ll_list(ood_dataset) 
    print(f'done. -- ll_list = {gp_ll_vs_n_context}')


    results = dict(
        np_ll_ood=np_ll_ood,
        # np_ll_iid=np_ll_iid,
        gp_ll=gp_ll,
        np_ll_vs_n_context_ood=np_ll_vs_n_context_ood,
        # np_ll_vs_n_context_iid=np_ll_vs_n_context_iid,
        gp_ll_vs_n_context=gp_ll_vs_n_context,
        n_context_list=N_CONTEXT_LIST,
    )

    write_yaml(save_path / 'summary.yaml', results)

if __name__ == '__main__':
    main()