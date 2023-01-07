import torch.nn as nn
import torch


class FixedPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(0, max_seq_len, dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer('emb', emb)

    def forward(self, x: torch.Tensor):
        return self.emb[None, :x.shape[1], :].to(x.device)
