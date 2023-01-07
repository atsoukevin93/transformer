import torch
import torch.nn as nn
import utils.util as ut
from .encoder_block import EncoderBlock
from .embeddings.embedding import Embedding


class Encoder(nn.Module):
    def __init__(self, config, embedding: Embedding = Embedding, **kwargs):
        super(Encoder, self).__init__()
        # blocks
        self.embedding = embedding(config)
        self.layers = nn.Sequential(*[EncoderBlock(config) for _ in range(config.n_encoder_block)])

        for k, v in kwargs.items():
            setattr(self, k, v)

    def forward(self, x, input_mask=None, **kwargs):
        # n, seq_length = x.shape
        if not ut.exists(input_mask):
            input_mask = x != self.src_pad_idx

        out = self.embedding(x, **kwargs)

        for layer in self.layers:
            out = layer(out, out, out, input_mask, input_mask)

        return out

    def set_embedding_layer(self, new_embedding: nn.Embedding):
        self.embedding = new_embedding

    def set_encoder_layers(self, new_encoder_layers: nn.Sequential):
        self.layers = new_encoder_layers
