import random
import json
import torch
import numpy as np
import pandas as pd
from torch.nn import functional as F
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from opt_einsum_torch import EinsumPlanner
from functools import wraps
import time


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    if torch.backends.mps.is_available():
        print("Info: MPS found!")
        device = torch.device("mps")
    else:
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def custom_einsum(equation: str, x: torch.Tensor, y: torch.Tensor):
    if torch.has_cuda:
        ee = EinsumPlanner(x.device, cuda_mem_limit=0.9)
        return ee.einsum(equation, x, y)
    else:
        return torch.einsum(equation, x, y)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def empty(tensor: torch.Tensor):
    return tensor.numel() == 0


def reshape_output_target_tensors(output, target):
    return output.contiguous().view(-1, output.size(-1)), target.contiguous().view(-1)