import torch.nn as nn
import torch


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emb = nn.Embedding(config.block_size, config.n_embd)

    def forward(self, x):
        # t = torch.arange(x.shape[1], device=x.device)
        return self.emb(torch.arange(x.shape[1], device=x.device))
