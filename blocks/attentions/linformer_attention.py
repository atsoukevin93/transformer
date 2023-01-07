import math
import torch
import torch.nn as nn
from base.base_attention import BaseAttention
from utils.util import custom_einsum


class LinformerAttention(BaseAttention):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.dim_k = config.dim_k
        self.E = nn.Linear(config.block_size, self.dim_k)
        self.F = nn.Linear(config.block_size, self.dim_k)
        # torch.nn.init.xavier_normal_(self.E.weight)
        # torch.nn.init.xavier_normal_(self.F.weight)

    def forward(self, values, keys, queries, input_mask=None, target_mask=None):
        b = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        values, keys, queries = self.proj_query_key_value(
            values,
            keys,
            queries,
            input_mask=input_mask,
            target_mask=target_mask
        )

        # project Keys and Values to k x d dimension
        keys = self.E(keys.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        values = self.F(values.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        similarities = custom_einsum("nqhd,nkhd->nhqk", queries, keys)
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape: (N, dim_k, heads, heads_dim)
        # similarities shape: (N, heads, query_len, dim_k)

        causal_mask = self.get_causal_mask(query_len, self.dim_k, queries.device) if self.causal else None

        if causal_mask is not None:
            similarities = similarities.masked_fill(causal_mask == 0, float('-1e20'))

        # pad_mask = similarities.abs() > float('1e-8')
        # similarities = similarities.masked_fill(~pad_mask, float('-1e20'))

        attention = torch.softmax(similarities / math.sqrt(self.head_dim), dim=3)
        out = custom_einsum("nhql,nlhd->nqhd", attention, values)
        out = out.reshape(b, query_len, self.heads*self.head_dim)
        # attention shape : (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # after einsum (N, query_len, heads, head_dim) then flatten last two dimensions

        out = self.fc_out(out)
        del queries, keys, values
        del causal_mask
        del similarities
        del attention
        torch.cuda.empty_cache()
        return out
