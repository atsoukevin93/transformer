import torch
import torch.nn as nn


# Implementation of relative position representations
class RelativePosition(nn.Module):

    def __init__(self, config):
        super().__init__()
        head_dim = config.n_embd // config.n_head
        self.num_units = head_dim
        self.max_relative_position = config.max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(config.max_relative_position * 2 + 1, head_dim))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k, device):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).to(device)
        embeddings = self.embeddings_table[final_mat].to(device)

        del range_vec_k
        del range_vec_q
        del distance_mat
        del distance_mat_clipped
        del final_mat
        torch.cuda.empty_cache()
        return embeddings
