
import torch
from torch.utils.data import DataLoader
import time

from npc.modules.dataset import SineData
from npc.modules.np import AttnLatentEncoder

if __name__ == '__main__':

    dataset = SineData(num_context_range=(3, 10))

    ######## test encoder in fwd and bwd pass
    batch_size = 64
    dloader = DataLoader(dataset, batch_size=batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = next(iter(dloader))

    x, y, cmask, tmask = batch

    encoder = AttnLatentEncoder(x_dim=x.shape[-1], y_dim=y.shape[-1], z_dim=64, num_heads=4)
    encoder.train = True
    encoder.to(device)

    s = time.time()
    for x, y, cmask, tmask in dloader:
        q_c = encoder(x.to(device), y.to(device), cmask.to(device))
        q_t = encoder(x.to(device), y.to(device), tmask.to(device))
    
    print(f'fwd time: {time.time() - s:.4f} seconds.')

    s = time.time()
    for x, y, cmask, tmask in dloader:
        q_c = encoder(x.to(device), y.to(device), cmask.to(device))
        q_t = encoder(x.to(device), y.to(device), tmask.to(device))

        z_c = q_c.rsample()
        z_t = q_t.rsample()

        m = (z_c.mean() + z_t.mean())
        m.backward()
    
    print(f'bwd time: {time.time() - s:.4f} seconds.')