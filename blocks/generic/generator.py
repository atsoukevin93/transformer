import torch.nn as nn
from torch.nn.functional import log_softmax


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x, temperature=None):
        if temperature:
            return log_softmax(self.proj(x)/temperature, dim=-1)
        else:
            return log_softmax(self.proj(x), dim=-1)