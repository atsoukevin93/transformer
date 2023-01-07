import torch
import torch.nn as nn
from numpy import sqrt
from utils.util import exists, default
from .positional_embeddings.absolute_positional_embedding import AbsolutePositionalEmbedding
from .word_embeddings.classic_word_embedding import ClassicWordEmbedding


class TransformerKmer2KmerEmbedding(nn.Module):
    def __init__(self, config, word_embedding: nn.Embedding = None, position_embedding: nn.Embedding = None):
        super(TransformerKmer2KmerEmbedding, self).__init__()
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

        self.kmer_pos_embedding = nn.Linear(1, config.n_embd, bias=False)

        self.dropout = nn.Dropout(config.embd_pdrop)

    def forward(self, x, **kwargs):
        kmer_pos = kwargs["kmer_pos"] if "kmer_pos" in kwargs else None
        # assert exists(kmer_pos), "provides kmer_pos argument to the embedding layer"
        if exists(self.position_embedding):
            tmp_out = self.word_embedding(x) + self.position_embedding(x)/sqrt(self.n_embd)
            # kmer_pos = kmer_pos.to(out.dtype)
            # print(f"type de out {out.dtype}")
            # print(f"type de kmer_pos {kmer_pos.dtype}")
            tmp_out = tmp_out + self.kmer_pos_embedding(kmer_pos.to(tmp_out.dtype))[:, None, :]/sqrt(self.n_embd)
            out = self.dropout(tmp_out)
        else:
            tmp_out = self.word_embedding(x)
            # kmer_pos = kmer_pos.to(out.dtype)
            out = tmp_out + self.kmer_pos_embedding(kmer_pos.to(tmp_out.dtype))[:, None, :]/sqrt(self.n_embd)
            out = self.dropout(tmp_out)
        del tmp_out
        del kmer_pos
        return out
