
import torch
from torch.utils.data import DataLoader
import time

from npc.modules.dataset import SineData
from npc.modules.np import NPModule

if __name__ == '__main__':

    use_attention = True
    dataset = SineData(num_context_range=(3, 10))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    ######## test encoder in fwd and bwd pass
    batch_size = 1
    dloader = DataLoader(dataset, batch_size=batch_size)
    batch = next(iter(dloader))

    x, y, cmask, tmask = batch

    np = NPModule(
        x_dim=x.shape[-1], y_dim=y.shape[-1], z_dim=64, r_dim=32, h_dim=128, use_attention=use_attention,
        n_heads=1)
    np.train = True
    np.to(device)

    s = time.time()
    for x, y, cmask, tmask in dloader:
        output = np(x.to(device), y.to(device), cmask.to(device), tmask.to(device))
    
    print(f'fwd time: {time.time() - s:.4f} seconds.')

    s = time.time()
    optim = torch.optim.AdamW(np.parameters(), lr=1e-3)
    for x, y, cmask, tmask in dloader:
        optim.zero_grad()
        output = np(x.to(device), y.to(device), cmask.to(device), tmask.to(device))
        (-output['elbo']).backward()
        optim.step()

        for name, param in np.named_parameters():
            if torch.any(torch.isnan(param)):
                print(f'{name} became nan')
                breakpoint()
        print(output['elbo'])

    print(f'bwd time: {time.time() - s:.4f} seconds.')
