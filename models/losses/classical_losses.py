import torch.nn.functional as F
from utils.util import reshape_output_target_tensors


def cross_entropy_loss(output, target, ignore_index: int = 0, reshape=True):
    if reshape:
        output, target = reshape_output_target_tensors(output, target)
    return F.cross_entropy(
                output,
                target,
                ignore_index=ignore_index
                # label_smoothing=0.1
            )


def nll_loss(output, target):
    return F.nll_loss(output, target)
