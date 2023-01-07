import math
import torch
import utils.util as ut
# from einops import rearrange, repeat
from base.base_attention import BaseAttention
from .relative_positions import RelativePosition
from utils import custom_einsum


class ClassicAttention(BaseAttention):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        # self.add_relative_positions = kwargs["add_relative_positions"] if "add_relative_positions" in kwargs else None

        is_cross_attention = kwargs["is_cross_attention"] if "is_cross_attention" in kwargs else False

        self.relative_position_embedding = config.relative_position_embedding
        self.add_relative_position_to_values = config.add_relative_position_to_values

        if is_cross_attention:
            self.relative_position_embedding = False
        else:
            self.relative_position_embedding = config.relative_position_embedding
            self.add_relative_position_to_values = config.add_relative_position_to_values
        if self.relative_position_embedding:
            self.max_relative_position = ut.default(config.max_relative_position, BaseAttention.DFLT_MAX_REL_POS)
            self.relative_position_k = RelativePosition(config)
            if self.add_relative_position_to_values:
                self.relative_position_v = RelativePosition(config)

    def forward(self, values, keys, queries, input_mask=None, target_mask=None):
        # Batch size
        b = queries.shape[0]

        # Sequence lengths
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        values, keys, queries = self.proj_query_key_value(
            values,
            keys,
            queries,
            input_mask=input_mask,
            target_mask=target_mask
        )

        if self.relative_position_embedding:
            similarities1 = custom_einsum("nqhd,nkhd->nhqk", queries, keys)
            # queries shape: (b, query_len, heads, heads_dim)
            # keys shape: (b, key_len, heads, heads_dim)
            # similarities shape: (b, heads, query_len, key_len)
            rel_pos_k = self.relative_position_k(query_len, key_len, queries.device)
            rel_pos_k_scores = custom_einsum("nqhd,qkd->nhqk", queries, rel_pos_k)

            similarities = (similarities1 + rel_pos_k_scores)
        else:
            similarities = custom_einsum("nqhd,nkhd->nhqk", queries, keys)
        # queries shape: (b, query_len, heads, heads_dim)
        # keys shape: (b, key_len, heads, heads_dim)
        # similarities shape: (b, heads, query_len, key_len)

        causal_mask = self.get_causal_mask(query_len, key_len, queries.device) if self.causal else None

        if causal_mask is not None:
            similarities = similarities.masked_fill(~causal_mask, float('-1e20'))

        attention = torch.softmax(similarities / math.sqrt(self.head_dim), dim=3)
        attention = self.attn_dropout(attention)

        out = custom_einsum("nhql,nlhd->nqhd", attention, values)
        out = out.reshape(b, query_len, self.heads*self.head_dim)

        if self.relative_position_embedding and self.add_relative_position_to_values:
            rel_pos_v = self.relative_position_v(query_len, value_len, queries.device)
            weight_ = custom_einsum("nhql,qld->nqhd", attention, rel_pos_v)
            weight_ = weight_.reshape(b, query_len, self.heads * self.head_dim)

            out = out + weight_

            del similarities1
            del rel_pos_k_scores
            del rel_pos_k
            del weight_

        # attention shape : (b, heads, query_len, key_len)
        # values shape: (b, value_len, heads, heads_dim)
        # after einsum (b, query_len, heads, head_dim) then flatten last two dimensions

        out = self.fc_out(out)
        out = self.proj_dropout(out)

        del queries, keys, values
        del causal_mask
        del similarities
        del attention

        # if self.relative_position_embedding and self.add_relative_position_to_values:
        #     del similarities1
        #     del rel_pos_k_scores
        #     del rel_pos_k
        #     del weight_

        # del pad_mask
        torch.cuda.empty_cache()
        return out
