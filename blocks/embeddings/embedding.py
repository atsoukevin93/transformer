import torch
import torch.nn as nn
from utils.util import exists, default
from numpy import sqrt
from .positional_embeddings.absolute_positional_embedding import AbsolutePositionalEmbedding
from .word_embeddings.classic_word_embedding import ClassicWordEmbedding


class Embedding(nn.Module):
    def __init__(self, config, word_embedding: nn.Embedding = None, position_embedding: nn.Embedding = None):
        super(Embedding, self).__init__()
        relative_position_embedding = default(config.relative_position_embedding, False)

        self.n_embd = config.n_embd

        # blocks
        if exists(word_embedding):
            self.word_embedding = word_embedding
        else:
            self.word_embedding = ClassicWordEmbedding(config)

        if relative_position_embedding:
            self.position_embedding = None
        elif exists(position_embedding):
            self.position_embedding = position_embedding
        else:
            self.position_embedding = AbsolutePositionalEmbedding(config)

        self.dropout = nn.Dropout(config.embd_pdrop)

    def forward(self, x):
        if exists(self.position_embedding):
            return self.dropout(self.word_embedding(x) + self.position_embedding(x)/sqrt(self.n_embd))
        else:
            return self.dropout(self.word_embedding(x))
