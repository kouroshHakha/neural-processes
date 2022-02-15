
import torch
from torch.utils.data import DataLoader
import time

from npc.modules.dataset import SineData
from npc.modules.np import LinearLatentEncoder

if __name__ == '__main__':

    dataset = SineData(num_context_range=(3, 10))

    ######## test encoder in fwd and bwd pass
    batch_size = 64
    dloader = DataLoader(dataset, batch_size=batch_size)
    batch = next(iter(dloader))

    x, y, cmask, tmask = batch

    encoder = LinearLatentEncoder(x_dim=x.shape[-1], y_dim=y.shape[-1], h_dim=128, z_dim=64)
    encoder.train = True

    s = time.time()
    for x, y, cmask, tmask in dloader:
        q_c = encoder(x, y, cmask)
        q_t = encoder(x, y, tmask)
    
    print(f'fwd time: {time.time() - s:.4f} seconds.')

    s = time.time()
    for x, y, cmask, tmask in dloader:
        q_c = encoder(x, y, cmask)
        q_t = encoder(x, y, tmask)

        z_c = q_c.rsample()
        z_t = q_t.rsample()

        m = (z_c.mean() + z_t.mean())
        m.backward()
    
    print(f'bwd time: {time.time() - s:.4f} seconds.')