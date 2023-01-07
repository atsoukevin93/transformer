import torch.nn as nn
from .attentions.attention_factory import AttentionFactory
from .generic import FeedForward


class EncoderBlock(nn.Module):
    def __init__(self, config, causal=False, is_cross_attention=False):
        super(EncoderBlock, self).__init__()
        # config
        # self.is_cross_attention = kwargs["is_cross_attention"] if "is_cross_attention" in kwargs else None

        # model
        self.norm1 = nn.LayerNorm(config.n_embd)
        self.norm2 = nn.LayerNorm(config.n_embd)
        self.attention = AttentionFactory.build_attention(config, causal=causal, is_cross_attention=is_cross_attention)

        self.feed_forward = FeedForward(
            config.n_encoder_feedforward_layer,
            config.n_embd,
            config.n_embd,
            config.encoder_feedforward_forward_expansion,
            config.encoder_feedforward_pdrop
        )

    def forward(self, value, key, query, input_mask, target_mask):
        attention = self.attention(value, key, query, input_mask=input_mask, target_mask=target_mask)
        x = self.norm1(attention + query)
        del attention
        return self.norm2(self.feed_forward(x) + x)
