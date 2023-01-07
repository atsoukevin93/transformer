import math
import torch
import utils.util as ut
# from einops import rearrange, repeat
from base.base_attention import BaseAttention
from .relative_positions import RelativePosition


class PerformerAttention(BaseAttention):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def forward(self, values, keys, queries, input_mask=None, target_mask=None):
        # Ongoing
        pass