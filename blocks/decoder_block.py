import torch.nn as nn
import utils.util as ut
from .attentions import AttentionFactory
from .encoder_block import EncoderBlock
from .generic import FeedForward


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super(DecoderBlock, self).__init__()
        # model
        self.causal_attention = AttentionFactory.build_attention(config, causal=True)
        self.norm = nn.LayerNorm(config.n_embd)
        if config.decoder_add_cross_attention_layer:
            self.cross_attention_layer = EncoderBlock(config, is_cross_attention=True)
        else:
            self.cross_attention_layer = None
            self.feed_forward = FeedForward(
                n_layer=config.n_decoder_feedforward_layer,
                input_dim=config.n_embd,
                output_dim=config.n_embd,
                forward_expansion=config.decoder_feedforward_forward_expansion,
                resid_pdrop=config.decoder_feedforward_pdrop
            )
            self.norm2 = nn.LayerNorm(config.n_embd)
        self.dropout = nn.Dropout(config.attention_pdrop)

    def forward(self, x, value, key, input_mask, target_mask):
        causal_attention = self.causal_attention(x, x, x, input_mask=target_mask, target_mask=target_mask)
        query = self.dropout(self.norm(causal_attention + x))
        if ut.exists(self.cross_attention_layer):
            out = self.cross_attention_layer(value, key, query, input_mask, target_mask)
        else:
            out = self.norm2(self.feed_forward(x) + x)
        del causal_attention
        del query
        return out
