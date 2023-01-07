import math
import os
import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod
from utils import exists
from typing import Optional, Union


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    def _init_weights(self, module):
        # for p in module.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1/math.sqrt(self.n_embd))
            # torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, weight_decay=0.1, learning_rate=1e-4, betas=[0.9, 0.95]):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the blocks into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embeddings weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay

        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.Parameter)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith('embeddings_table'):
                    no_decay.add(fpn)
        # special case the position embeddings parameter in the root GPT module as not decayed
        # no_decay.add('encoder.position_embedding')
        # no_decay.add('decoder.position_embedding')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either " \
                                                           "decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=tuple(betas))
        return optimizer

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def from_pretrained(
            self,
            pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
            device: torch.DeviceObjType = None
    ):
        if not exists(device):
            device = next(self.model.parameters()).device
        if exists(pretrained_model_name_or_path):
            if os.path.isfile(pretrained_model_name_or_path):
                checkpoint = torch.load(pretrained_model_name_or_path, map_location=device)
            elif pretrained_model_name_or_path.endswith(".pth"):
                raise f"provide the full path to {pretrained_model_name_or_path}"
            state_dict = checkpoint['state_dict']
            self.load_state_dict(state_dict)

    @abstractmethod
    def sample(self, *inputs):
        """
        Logic for sampling from the model

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
