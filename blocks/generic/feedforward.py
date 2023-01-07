import torch.nn as nn
import utils.util as ut
from .approximated_gelu import NewGELU


class FeedForward(nn.Module):
    def __init__(self,
                 n_layer=1,
                 input_dim: int = None,
                 output_dim: int = None,
                 forward_expansion: int = 1,
                 resid_pdrop: float = 0.1):
        super(FeedForward, self).__init__()
        self.n_layers = n_layer
        self.norm = nn.LayerNorm(input_dim)
        # self.attention = SelfAttention(config, causal=causal)
        assert ut.exists(input_dim), "provide input dimension"
        assert ut.exists(output_dim), "provide output dimension"
        tmp_feed_forward = nn.Sequential(
            nn.Linear(input_dim, forward_expansion * input_dim),
            NewGELU(),
            nn.Linear(forward_expansion * input_dim, output_dim),
            nn.Dropout(resid_pdrop)
        )
        if n_layer == 1:
            self.feed_forward = tmp_feed_forward
        else:
            self.feed_forward = nn.Sequential(*[tmp_feed_forward for _ in range(n_layer)])
        del tmp_feed_forward

    def forward(self, input_tensor):
        forward = self.feed_forward(input_tensor)
        out = self.norm(forward + input_tensor)
        return out
