import torch
import torch.nn as nn
from utils.util import exists
from .generic import Generator
from .decoder_block import DecoderBlock
from .embeddings.embedding import Embedding


class Decoder(nn.Module):
    def __init__(self, config, embedding: Embedding = Embedding, **kwargs):
        super(Decoder, self).__init__()

        # configs
        self.trg_pad_idx = config.trg_pad_idx
        self.output_softmax_temperature = config.decoder_output_softmax_temperature

        # model
        self.embedding = embedding(config)

        self.layers = nn.Sequential(*[DecoderBlock(config) for _ in range(config.n_decoder_block)])
        # self.fc_out = nn.Linear(config.n_embd, config.trg_vocab_size)
        self.generator = Generator(config.n_embd, config.trg_vocab_size)

        for k, v in kwargs.items():
            setattr(self, k, v)

    def forward(
            self,
            x,
            enc_out: torch.Tensor,
            input_mask: torch.Tensor = None,
            target_mask: torch.Tensor = None,
            shared_embedding: nn.Embedding = None,
            **kwargs
    ):

        # b: batch size, seq_length: sequence length
        # b, seq_length = x.shape
        assert exists(input_mask), 'you should provide the input mask'

        if not exists(target_mask):
            target_mask = x != self.src_pad_idx
        if exists(shared_embedding):
            x = shared_embedding(x, **kwargs)
        else:
            x = self.embedding(x, **kwargs)
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, input_mask, target_mask)

        out = self.generator(x, temperature=self.output_softmax_temperature)
        del x
        return out
