import torch.nn as nn


class ClassicWordEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emb = nn.Embedding(config.src_vocab_size, config.n_embd, padding_idx=config.src_pad_idx)

    def forward(self, x):
        return self.emb(x)
