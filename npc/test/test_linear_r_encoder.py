
import torch
from torch.utils.data import DataLoader
import time

from npc.modules.dataset import SineData
from npc.modules.np import LinearREncoder

if __name__ == '__main__':

    dataset = SineData(num_context_range=(3, 10))

    ######## test encoder in fwd and bwd pass
    batch_size = 64
    dloader = DataLoader(dataset, batch_size=batch_size)
    batch = next(iter(dloader))

    x, y, cmask, tmask = batch

    encoder = LinearREncoder(x_dim=x.shape[-1], y_dim=y.shape[-1], h_dim=128, r_dim=64)
    encoder.train = True

    s = time.time()
    for x, y, cmask, tmask in dloader:
        r_c = encoder(x, y, cmask)
        r_t = encoder(x, y, tmask)
    
    print(f'fwd time: {time.time() - s:.4f} seconds.')

    s = time.time()
    for x, y, cmask, tmask in dloader:
        r_c = encoder(x, y, cmask)
        r_t = encoder(x, y, tmask)
        m = (r_c.mean() + r_t.mean())
        m.backward()
    
    print(f'bwd time: {time.time() - s:.4f} seconds.')