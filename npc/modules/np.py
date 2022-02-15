
import torch
import torch.nn as nn
import torch.functional as F
from torch.distributions import Normal

import torch_scatter


import pytorch_lightning as pl


class AttnREncoder(nn.Module):

    def __init__(self, x_dim, y_dim, r_dim, num_heads) -> None:
        super().__init__()

        context_dim = x_dim + y_dim

        self.context_to_query = nn.Linear(context_dim, r_dim)
        self.target_to_query = nn.Linear(x_dim, r_dim)
        self.attn = nn.MultiheadAttention(
            r_dim, num_heads, batch_first=True, kdim=context_dim, vdim=context_dim)
        self.cross_attn = nn.MultiheadAttention(
            r_dim, num_heads, batch_first=True, kdim=x_dim, vdim=r_dim)

    def forward(self, x, y, cmask, tmask):
        
        B, C = cmask.shape

        input_tokens = torch.cat([x, y], -1)
        tokens, _ = self.attn(
            query=self.context_to_query(input_tokens), # (B, C, D)
            key=input_tokens, 
            value=input_tokens, 
            key_padding_mask=~cmask, # (B, C): True means ignore
            need_weights=False
        ) # (B, C, D)

        assert ~torch.any(torch.isnan(tokens))

        """
        ############## creating the attention mask based on (c/t)masks 
        # create an (B, Q, K) mask matrix: for batch b tmask tells what is the query (Q)
        # and cmask tells what are the keys for that query (K)
        attn_mask = tmask.repeat(C, 1, 1).permute(1, 2, 0) & cmask.unsqueeze(1)
        # repeat the batch dimension n_head times
        n_heads = self.cross_attn.num_heads
        attn_mask = (~attn_mask).view(B, 1, C, C).expand(-1, n_heads, -1, -1). \
            reshape(B*n_heads, C, C)

        r, _ = self.cross_attn(
            query=self.target_to_query(x), # (B, C, D)
            key=x, # (B, C, D)
            value=tokens, # (B, C, D)
            attn_mask=attn_mask, # (B*n_head, C, C)
            need_weights=False
        ) # (B, C, D)

        ####### debug 
        target_rs = torch.where(~torch.isnan(r[:, :, 0]))
        tmask_true = torch.where(tmask)
        assert len(target_rs) == len(tmask_true)
        assert all(torch.equal(target_rs[i], tmask_true[i]) for i in range(len(target_rs)))

        return r[tmask] #(targets, D)
        """
        # return tokens[tmask]

        # cross attention
        r, _ = self.cross_attn(
            query=self.target_to_query(x), # (B, C, D)
            key=x, # (B, C, D)
            value=tokens, # (B, C, D)
            key_padding_mask=~cmask,
            need_weights=False
        ) # (B, C, D)

        return r[tmask]


class LinearREncoder(nn.Module):
    def __init__(self, x_dim, y_dim, h_dim, r_dim, depth=1):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim

        layers = [nn.Linear(x_dim + y_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, r_dim)]

        self.net = nn.Sequential(*layers)

    def forward(self, x, y, mask):

        batch_size, _, x_dim = x.size()
        _, _, y_dim = y.size()

        # assert x_dim == self.x_dim
        # assert y_dim == self.y_dim

        context_lengths = mask.sum(-1)
        group_ids = torch.cat([torch.tensor([i] * context_lengths[i]) for i in range(batch_size)]).to(x.device)

        nn_in = torch.cat([x[mask], y[mask]], dim=-1)
        r = self.net(nn_in)

        r_aggr = torch_scatter.scatter_mean(r, group_ids, dim=0)

        return r_aggr


class AttnLatentEncoder(nn.Module):

    def __init__(self, x_dim, y_dim, z_dim, num_heads):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim

        context_dim = x_dim + y_dim
        self.context_to_query = nn.Linear(context_dim, 2*z_dim)
        self.attn = nn.MultiheadAttention(
            2*z_dim, num_heads, batch_first=True, kdim=context_dim, vdim=context_dim)

    def forward(self, x, y, mask):

        batch_size, _, _ = x.size()

        context_lengths = mask.sum(-1)
        group_ids = torch.cat([torch.tensor([i] * context_lengths[i]) for i in range(batch_size)]).to(x.device)

        nn_in = torch.cat([x, y], dim=-1)
        s, _ = self.attn(
            query=self.context_to_query(nn_in),
            key=nn_in,
            value=nn_in,
            key_padding_mask=~mask, # (B, C): True means ignore
            need_weights=False
        ) # (B, C, z_dim)

        s_aggr = torch_scatter.scatter_mean(s[mask], group_ids, dim=0)

        mean, log_scale = s_aggr.chunk(2, dim=-1)

        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        scale = 0.1 + 0.9 * log_scale.sigmoid()

        return Normal(mean, scale)



class LinearLatentEncoder(nn.Module):

    def __init__(self, x_dim, y_dim, h_dim, z_dim, depth=1):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = z_dim

        layers = [nn.Linear(x_dim + y_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, 2*z_dim)]

        self.net = nn.Sequential(*layers)
    
    def forward(self, x, y, mask):

        batch_size, _, x_dim = x.size()
        _, _, y_dim = y.size()

        context_lengths = mask.sum(-1)
        group_ids = torch.cat([torch.tensor([i] * context_lengths[i]) for i in range(batch_size)]).to(x.device)

        nn_in = torch.cat([x[mask], y[mask]], dim=-1)
        s = self.net(nn_in)

        s_aggr = torch_scatter.scatter_mean(s, group_ids, dim=0)

        mean, log_scale = s_aggr.chunk(2, dim=-1)

        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        scale = 0.1 + 0.9 * log_scale.sigmoid()

        return Normal(mean, scale)


class Decoder(nn.Module):

    def __init__(self, x_dim, z_dim, r_dim, h_dim, y_dim):
        super().__init__()

        self.x_dim = x_dim
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.y_dim = y_dim

        layers = [nn.Linear(x_dim + z_dim + r_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True)]

        self.to_hidden = nn.Sequential(*layers)
        self.hidden_to_mu = nn.Linear(h_dim, y_dim)
        self.hidden_to_sigma = nn.Linear(h_dim, y_dim)

    
    def forward(self, x, r, z):

        batch_size, x_dim = x.size()
        _, r_dim = r.size()
        _, z_dim = z.size()

        nn_in = torch.cat([x, r, z], dim=-1)
        hidden = self.to_hidden(nn_in)
        mu = self.hidden_to_mu(hidden)
        log_scale = self.hidden_to_sigma(hidden)

        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        # scale = 0.1 + 0.9 * log_scale.sigmoid()#torch.log(1 + log_scale.exp()) # softplus
        scale = torch.log(1 + log_scale.exp()) # log_scale.sigmoid()

        return Normal(mu, scale)


class NPModule(nn.Module):

    def __init__(self, x_dim, z_dim, r_dim, h_dim, y_dim, attn_det_path=False, attn_latent_path=False, n_heads=4) -> None:
        super().__init__()

        self.attn_det_path = attn_det_path
        self.attn_latent_path = attn_latent_path
        if attn_det_path:
            self.det_encoder = AttnREncoder(x_dim, y_dim, r_dim, num_heads=n_heads)
        else:
            self.det_encoder = LinearREncoder(x_dim, y_dim, h_dim, r_dim)
        
        if attn_latent_path:
            self.latent_encoder = AttnLatentEncoder(x_dim, y_dim, z_dim, num_heads=n_heads)
        else:
            # h_dim can be different than ouput (r/z) dim
            self.latent_encoder = LinearLatentEncoder(x_dim, y_dim, h_dim, z_dim)

        self.decoder = Decoder(x_dim, z_dim, r_dim, h_dim, y_dim)

    def forward(self, x, y, context_mask, target_mask, compute_elbo=False):
        # returns all the components required for training / inference according to [T|C] loss in 
        # "Empirical Evaluation of Neural Process Objectives"

        batch_size, _, _ = x.size()
        # compute elbo is true either when given or during the training loop
        compute_elbo = compute_elbo or self.training

        target_lengths = target_mask.sum(-1)

        # encode context and target to stochastic latent
        q_c = self.latent_encoder(x, y, context_mask)

        if compute_elbo:
            q_t = self.latent_encoder(x, y, target_mask)
            z_s = q_t.rsample()
            kl = torch.distributions.kl_divergence(q_t, q_c).sum(-1).mean()
        else:
            z_s = q_c.rsample()
            q_t, kl = None, None

        # encode context to deterministic r
        if self.attn_det_path:
            r_decoder = self.det_encoder(x, y, context_mask, target_mask)
        else:
            r_c = self.det_encoder(x, y, context_mask) # (B, r_dim)
            # repeat r and z of each batch item to become r_decoder and z_decoder
            r_decoder = torch.cat([torch.repeat_interleave(r_c[i][None], repeats=target_lengths[i], dim=0) for i in range(batch_size)])
        
        z_decoder = torch.cat([torch.repeat_interleave(z_s[i][None], repeats=target_lengths[i], dim=0) for i in range(batch_size)])

        prob_y = self.decoder(x[target_mask], r_decoder, z_decoder)

        if compute_elbo:
            elbo = prob_y.log_prob(y[target_mask]).sum(-1).mean(0) - kl
        else:
            elbo = None

        return dict(
            prob_y=prob_y, 
            y_target=y[target_mask],
            kl=kl,
            z_decoder=z_decoder,
            r_decoder=r_decoder,
            elbo=elbo,
        )

    def predict(self, xc, yc, xt):
        # runs inference on xc, yc, xt returning prob of y at xt
        self.eval()

        context_len, _ = xc.size()
        _, y_dim = yc.size()
        target_len, _ = xt.size()

        # concatenate context and target and add batch dim
        x = torch.cat([xc, xt], 0)[None]
        y = torch.cat([yc, torch.zeros(target_len, y_dim).to(yc)])[None]

        # create context and target masks with a batch dim
        context_mask = torch.zeros(1, context_len + target_len, dtype=torch.bool, device=xc.device)
        target_mask = torch.zeros(1, context_len + target_len, dtype=torch.bool, device=xc.device)

        context_mask[0, :context_len] = True
        target_mask[0, context_len:] = True

        output = self(x, y, context_mask, target_mask, compute_elbo=False)
        prob_y = output['prob_y']
        # print(prob_y.log_prob(output['y_target']).sum(-1))
        # breakpoint()
        return prob_y


class LightningNP(pl.LightningModule):

    def __init__(
        self, 
        x_dim,
        y_dim,
        attn_det_path=False,
        attn_latent_path=False,
        z_dim=64,
        r_dim=64,
        h_dim=128,
        lr=3e-4,
        n_heads=4,
    ) -> None:
        super().__init__()

        self.lr = lr
        self.save_hyperparameters()

        self.np = NPModule(
            x_dim, z_dim, r_dim, h_dim, y_dim,
            attn_det_path=attn_det_path,
            attn_latent_path=attn_latent_path,
            n_heads=n_heads)
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y, cmask, tmask = batch
        output = self.np(x, y, cmask, tmask)
        loss = -output['elbo']
        return dict(loss=loss, kl=output['kl'].detach())

    def training_epoch_end(self, outputs) -> None:
        loss = torch.stack([output['loss'] for output in outputs], 0).mean()
        kl = torch.stack([output['kl'] for output in outputs], 0).mean()

        self.log('loss_epoch', loss)
        self.log('kl_epoch', kl)
    
    def validation_step(self, batch, batch_idx):
        x, y, cmask, tmask = batch
        output = self.np(x, y, cmask, tmask, compute_elbo=True)
        loss = -output['elbo']

        return dict(loss=loss, elbo=output['elbo'].detach(), kl=output['kl'].detach())

    def validation_epoch_end(self, outputs) -> None:
        
        loss = torch.stack([output['loss'] for output in outputs], 0).mean()
        kl = torch.stack([output['kl'] for output in outputs], 0).mean()
        elbo = torch.stack([output['elbo'] for output in outputs], 0).mean()

        self.log('valid_loss_epoch', loss)
        self.log('valid_kl_epoch', kl)
        self.log('valid_elbo_epoch', elbo)
    