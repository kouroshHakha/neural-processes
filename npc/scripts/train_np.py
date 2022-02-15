
import argparse
from gc import callbacks
from random import choice

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from npc.modules.exps import get_sine_datasets
try:
    from pytorch_lightning.loggers import WandbLogger
except ModuleNotFoundError:
    pass

from torch.utils.data import DataLoader

from npc.modules.dataset import SineData
from npc.modules.np import LightningNP

from debug import register_pdb_hook
register_pdb_hook()

def parse_args():
    parser = argparse.ArgumentParser()
    # either provide ckpt or config
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--max_epochs', type=int)
    parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--bsize', type=int, default=128)
    parser.add_argument('--dim', type=int, default=1) # dimensionality of the problem
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--r_dim', type=int, default=128)
    parser.add_argument('--h_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--attn_det_path', action='store_true', default=False)
    parser.add_argument('--attn_latent_path', action='store_true', default=False)
    # experimentation
    parser.add_argument('--exp', type=str, default='default') # choices 'default', 'unseen_amp/shift/freq', 'context_max', 'context_min_max', 'fns_n_pts_p'

    #### loggging
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--use_wandb', '-wb', action='store_true', default=False)
    parser.add_argument('--exp_suffix', type=str)
    parser.add_argument('--run_id', type=str)
    parser.add_argument('--validation_var', type=str, default='valid_elbo_epoch')

    return parser.parse_args()

if __name__ == '__main__':

    pargs = parse_args()
    pl.seed_everything(pargs.seed, workers=True)

    run_name = f'attn_det={pargs.attn_det_path},'\
               f'attn_latent={pargs.attn_latent_path},'\
               f'h_dim={pargs.h_dim},'\
               f'z_dim={pargs.z_dim},'\
               f'r_dim={pargs.r_dim},'\
               f'dim={pargs.dim},'\
               f'exp={pargs.exp}'

    if pargs.exp_suffix:
        run_name = f'{run_name}_{pargs.exp_suffix}'

    train_dataset, valid_dataset = get_sine_datasets(pargs)
    tloader = DataLoader(train_dataset, batch_size=pargs.bsize, num_workers=0, shuffle=True)
    vloader = DataLoader(valid_dataset, batch_size=pargs.bsize, num_workers=0)
    
    batch = next(iter(tloader))
    x, y, cmask, tmask = batch

    model = LightningNP(
        x_dim=x.shape[-1], y_dim=y.shape[-1], z_dim=pargs.z_dim, r_dim=pargs.r_dim, 
        h_dim=pargs.h_dim, lr=pargs.lr,
        attn_det_path=pargs.attn_det_path,
        attn_latent_path=pargs.attn_latent_path,
        n_heads=pargs.n_heads,
    )

    monitor_var = pargs.validation_var
    checkpointer = ModelCheckpoint(
        monitor=monitor_var,
        filename='{epoch}-{' + monitor_var + ':.3f}',
        save_last=True,
        mode='max',
        # save_top_k=5,
    )
    
    ####### train 
    ## setup the logger
    if pargs.use_wandb:
        import wandb
        wandb_run = wandb.init(
            project='np',
            name=run_name,
            id=pargs.run_id,
            # dir='./logs',
            config=dict(seed=pargs.seed)
        )
        logger = WandbLogger(experiment=wandb_run, save_dir='./logs')
    else:
        logger = True
    
    trainer = pl.Trainer(
        max_epochs=pargs.max_epochs,
        max_steps=pargs.max_steps,
        logger=logger,
        gpus=[0],
        log_every_n_steps=pargs.log_every,
        callbacks=[checkpointer]
    )
    trainer.fit(model, train_dataloaders=tloader, val_dataloaders=[vloader])
