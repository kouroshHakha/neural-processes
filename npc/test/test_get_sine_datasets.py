import numpy as np
from npc.modules.exps import get_sine_datasets
import matplotlib.pyplot as plt

from argparse import Namespace

# exps = ['default', 'unseen_amp', 'unseen_shift', 'unseen_freq']
# exps = ['default', 'context_10', 'context_2_10', 'context_100_100']
exps = ['default', 'fns_100_pts_20', 'fns_20_pts_100', 'fns_20_pts_20']

plt.close()
_, axes = plt.subplots(nrows=len(exps), ncols=2, squeeze=False, figsize=(15, 8), sharex=True, sharey=True)

for idx, exp in enumerate(exps):
    pargs = Namespace(exp=exp, dim=1, seed=0)
    train_dataset, valid_dataset = get_sine_datasets(pargs)
    
    train_cm, train_tm = [], []
    for x, y, cm, tm in train_dataset:
        axes[idx][0].plot(x.flatten(), y.flatten(), alpha=0.5, color='blue')
        train_cm.append(cm.sum())
        train_tm.append(tm.sum())

    print('-'*20 + f'// exp={exp:10}')
    print(f'train_context_mask -- mean: {np.mean(train_cm):.2f}, min: {np.min(train_cm)}, max: {np.max(train_cm)}')
    print(f'train_target_mask  -- mean: {np.mean(train_tm):.2f}, min: {np.min(train_tm)}, max: {np.max(train_tm)}')
    
    valid_cm, valid_tm = [], []
    for x, y, cm, tm in valid_dataset:
        axes[idx][1].plot(x.flatten(), y.flatten(), alpha=0.5, color='blue')
        valid_cm.append(cm.sum())
        valid_tm.append(tm.sum())

    print(f'valid_context_mask -- mean: {np.mean(valid_cm):.2f}, min: {np.min(valid_cm)}, max: {np.max(valid_cm)}')
    print(f'valid_target_mask  -- mean: {np.mean(valid_tm):.2f}, min: {np.min(valid_tm)}, max: {np.max(valid_tm)}')

    axes[idx][0].set_ylabel(exp)
    if idx == 0:
        axes[idx][0].set_title('train')
        axes[idx][1].set_title('valid')

plt.tight_layout()
plt.savefig('test_get_sin_datasets.png', dpi=250) 

