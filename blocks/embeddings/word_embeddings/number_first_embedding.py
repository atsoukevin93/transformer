import torch.nn as nn
from torch import cat


class AlternateNumberWordEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.numb_emb = nn.Linear(1, config.n_embd)
        self.word_emb = nn.Embedding(config.src_vocab_size, config.n_embd, padding_idx=config.src_pad_idx)

    def forward(self, numbers, x):
        return self.dropout(cat((self.numb_emb(numbers)[:, None, :], self.word_emb(x)), 1))
