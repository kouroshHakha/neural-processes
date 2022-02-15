
import torch
from torch.utils.data import DataLoader
import time

from npc.modules.dataset import SineData
from npc.modules.np import AttnREncoder

from debug import register_pdb_hook
register_pdb_hook()

if __name__ == '__main__':

    dataset = SineData(num_context_range=(3, 10))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    ######## test encoder in fwd and bwd pass
    batch_size = 64
    dloader = DataLoader(dataset, batch_size=batch_size)
    batch = next(iter(dloader))

    x, y, cmask, tmask = batch

    encoder = AttnREncoder(x_dim=x.shape[-1], y_dim=y.shape[-1], r_dim=64, num_heads=4)
    encoder.train = True
    encoder.to(device)

    s = time.time()
    for x, y, cmask, tmask in dloader:
        r_c = encoder(x.to(device), y.to(device), cmask.to(device), tmask.to(device))
        r_t = encoder(x.to(device), y.to(device), tmask.to(device), tmask.to(device))
    
    print(f'fwd time: {time.time() - s:.4f} seconds.')

    s = time.time()
    for x, y, cmask, tmask in dloader:
        r_c = encoder(x.to(device), y.to(device), cmask.to(device), tmask.to(device))
        r_t = encoder(x.to(device), y.to(device), tmask.to(device), tmask.to(device))
        m = (r_c.mean() + r_t.mean())
        m.backward()
    
    print(f'bwd time: {time.time() - s:.4f} seconds.')