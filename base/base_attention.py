import torch
import torch.nn as nn
from abc import abstractmethod


class BaseAttention(nn.Module):
    DFLT_MAX_REL_POS: int = 50

    def __init__(self, config, **kwargs):
        super(BaseAttention, self).__init__()
        assert (config.n_embd % config.n_head == 0), "Embed size needs to be divisible by the number heads"
        self.config = config
        self.embed_size = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.heads = config.n_head

        self.causal = kwargs["causal"] if "causal" in kwargs else None
        self.mode = kwargs["mode"] if "mode" in kwargs else None

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.attn_dropout = nn.Dropout(config.attention_pdrop)
        self.proj_dropout = nn.Dropout(config.attention_values_pdrop)
        self.fc_out = nn.Linear(config.n_head*self.head_dim, config.n_embd, bias=False)

    def proj_query_key_value(self, values, keys, queries, input_mask=None, target_mask=None):
        b = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        if input_mask is not None:
            pad_mask = input_mask[:, :, None]
            values = values.masked_fill(~pad_mask, 0.0)
            keys = keys.masked_fill(~pad_mask, 0.0)
            del pad_mask
        if target_mask is not None:
            pad_mask = target_mask[:, :, None]
            queries = queries.masked_fill(~pad_mask, 0.0)
            del pad_mask

        # Split embeddings into self.heads pieces
        values = values.reshape(b, value_len, self.heads, self.head_dim)
        keys = keys.reshape(b, key_len, self.heads, self.head_dim)
        queries = queries.reshape(b, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        return values, keys, queries

    @staticmethod
    def get_causal_mask(dim_a, dim_b, device):
        """
        Generates a causal mask of size (input_size, dim_k) for linformer
        Else, it generates (input_size, input_size) for full attention
        """
        return torch.tril(torch.ones(dim_a, dim_b, device=device)) == 1
        # return torch.eye(dim_a, dim_b) == 1

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError
