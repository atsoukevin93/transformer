import torch
import torch.nn.functional as F
from utils.util import reshape_output_target_tensors


def accuracy(output, target, reshape=True):
    if reshape:
        output, target = reshape_output_target_tensors(output, target)
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3, reshape=True):
    if reshape:
        output, target = reshape_output_target_tensors(output, target)
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def ppl(output, target, reshape=True):
    if reshape:
        output, target = reshape_output_target_tensors(output, target)
    with torch.no_grad():
        loss = F.cross_entropy(output, target)
    return float(torch.exp(loss))


def KL_div(output, target, reshape=True):
    if reshape:
        output, target = reshape_output_target_tensors(output, target)
    with torch.no_grad():
        loss = F.kl_div(output, target, reduction='batchmean')
    return float(loss)
