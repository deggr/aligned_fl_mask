import sys
from tqdm import tqdm
from typing import List
from collections import OrderedDict

import torch
import numpy as np

from config import CONFIG_ARGS

def flwr_get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def flwr_set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    try:
        net.load_state_dict(state_dict, strict=True)
    except:
        pass

def print2(*arg, **kwargs):
    print(*arg, **kwargs, file=CONFIG_ARGS.outfile)
    print(*arg, **kwargs)

def dprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)

def print_csv_line(**kwargs):
    print2(','.join(str(x) for x in kwargs.values()))

def nan_to_num(x, nan=0, posinf=0, neginf=0):
    x = x.clone()
    x[x != x] = nan
    x[x == -float('inf')] = neginf
    x[x == float('inf')] = posinf
    return x.clone()

def get_best_anchor_index(self, weights_results):
    best_anchor_index = None
    raise NotImplementedError('Implement!')
    return best_anchor_index

def evaluate_global(clients, global_model, progress=False, n_batches=0):
    mean_acc = 0
    with torch.no_grad():
        accuracies = {}
        sparsities = {}

        if progress:
            enumerator = tqdm(clients.items())
        else:
            enumerator = clients.items()

        for client_id, client in enumerator:
            accuracies[client_id] = client.test(model=global_model).item()
            sparsities[client_id] = client.sparsity()
            mean_acc += accuracies[client_id]
    mean_acc /= len(clients)
    return accuracies, sparsities, mean_acc


def evaluate_local(clients, global_model, progress=False, n_batches=0):

    # we need to perform an update to client's weights.
    with torch.no_grad():
        accuracies = {}
        sparsities = {}

        if progress:
            enumerator = tqdm(clients.items())
        else:
            enumerator = clients.items()

        for client_id, client in enumerator:
            client.reset_weights(global_state=global_model.state_dict(), use_global_mask=True)
            accuracies[client_id] = client.test().item()
            sparsities[client_id] = client.sparsity()

    return accuracies, sparsities