import argparse
import copy
import os
import sys
import time
import random
from copy import deepcopy
from tqdm import tqdm

import numpy as np
import torch
import torch.cuda
from torch import nn
from torch.nn import functional as F

from config import DEVICE, FP16, READJUSTMENT_RATIO, SPARSITY_DISTRIBUTION, ANCHOR_SELECTION_RULE, ALIGNED_DST, TEST, BETA, ROUNDS, LOCAL_EPOCHS, TOTAL_CLIENTS, CLIENTS, SPARSITY, FINAL_SPARSITY, LR
from utils import dprint, print_csv_line, nan_to_num, get_best_anchor_index, evaluate_global, evaluate_local, flwr_set_parameters, flwr_get_parameters
from datasets import get_dataset
from weight_matching import align_models
from models import all_models, needs_mask, initialize_mask
from client import Client
import permutation_specs

SEED = 42

print(f"Using device: {DEVICE}")

random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
rng = np.random.default_rng(seed=SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def device_list(x):
    if x == 'cpu':
        return [x]
    return [int(y) for y in x.split(',')]

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=('mnist', 'fashionmnist', 'emnist', 'cifar10', 'cifar100'),
                    default='fashionmnist', help='Dataset to use')
parser.add_argument('--distribution', type=str, choices=('dirichlet', 'lotteryfl', 'iid'), default='dirichlet',
                    help='how should the dataset be distributed?')

parser.add_argument('--min-samples', type=int, default=0, help='minimum number of samples required to allow a client to participate')
parser.add_argument('--samples-per-client', type=int, default=20, help='samples to allocate to each client (per class, for lotteryfl, or per client, for iid)')

# Pruning and regrowth options
parser.add_argument('--rate-decay-method', default='cosine', choices=('constant', 'cosine'), help='annealing for readjustment ratio')
parser.add_argument('--rate-decay-end', default=None, type=int, help='round to end annealing')
parser.add_argument('--rounds-between-readjustments', type=int, default=10, help='rounds between readjustments')
parser.add_argument('--remember-old', default=False, action='store_true', help="remember client's old weights when aggregating missing ones")

parser.add_argument('--batch-size', type=int, default=32,
                    help='local client batch size')
parser.add_argument('--l2', default=0.001, type=float, help='L2 regularization strength')
parser.add_argument('--momentum', default=0.9, type=float, help='Local client SGD momentum parameter')
parser.add_argument('--cache-test-set', default=False, action='store_true', help='Load test sets into memory')
parser.add_argument('--test-batches', default=0, type=int, help='Number of minibatches to test on, or 0 for all of them')
parser.add_argument('--eval-every', default=1, type=int, help='Evaluate on test set every N rounds')
parser.add_argument('--device', default='0', type=device_list, help='Device to use for compute. Use "cpu" to force CPU. Otherwise, separate with commas to allow multi-GPU.')
parser.add_argument('--min-votes', default=0, type=int, help='Minimum votes required to keep a weight')
parser.add_argument('--no-eval', default=True, action='store_false', dest='eval')
parser.add_argument('--grasp', default=False, action='store_true')
parser.add_argument('-o', '--outfile', default='output_aligned_dst.log', type=argparse.FileType('a', encoding='ascii'))


args = parser.parse_args()
devices = [DEVICE]
args.pid = os.getpid()

if args.dataset == 'emnist':
    permutation_spec = permutation_specs.cnn_permutation_spec()
elif args.dataset == 'cifar10':
    permutation_spec = permutation_specs.cifar10_cnn_permutation_spec()
elif args.dataset == 'fashionmnist':
    permutation_spec = permutation_specs.fashionmnist_cnn_permutation_spec()

if args.rate_decay_end is None:
    args.rate_decay_end = ROUNDS // 2
if FINAL_SPARSITY is None:
    FINAL_SPARSITY = SPARSITY

# Fetch and cache the dataset
dprint('Fetching dataset...')
cache_devices = devices

'''
if os.path.isfile(args.dataset + '.pickle'):
    with open(args.dataset + '.pickle', 'rb') as f:
        loaders = pickle.load(f)
else:
    loaders = get_dataset(args.dataset, clients=args.total_clients,
                          batch_size=args.batch_size, devices=cache_devices,
                          min_samples=args.min_samples)
    with open(args.dataset + '.pickle', 'wb') as f:
        pickle.dump(loaders, f)
'''

loaders = get_dataset(args.dataset, clients=TOTAL_CLIENTS, mode=args.distribution,
                      beta=BETA, batch_size=args.batch_size, devices=cache_devices,
                      min_samples=args.min_samples, samples=args.samples_per_client)

# initialize clients
dprint('Initializing clients...')
clients = {}
client_ids = []

for i, (client_id, client_loaders) in tqdm(enumerate(loaders.items())):
    cl = Client(client_id, *client_loaders, net=all_models[args.dataset],
                learning_rate=LR, local_epochs=LOCAL_EPOCHS,
                target_sparsity=SPARSITY)
    clients[client_id] = cl
    client_ids.append(client_id)
    torch.cuda.empty_cache()

# initialize global model
dprint('Initializing global model...')
global_model = all_models[args.dataset](device=DEVICE)
initialize_mask(global_model)
for name, weights in global_model.state_dict().items():
    if name[-5] == '_':
        print(f"Initial mask {name} is dense: {torch.prod(torch.Tensor(list(weights.shape))) == weights.sum()}") 

# execute grasp on one client if needed
if args.grasp:
    print('GRASP IS SET TO TRUE')
    client = clients[client_ids[0]]
    from grasp import grasp
    pruned_net = grasp(client, sparsity=SPARSITY, dataset=args.dataset)
    pruned_masks = {}
    pruned_params = {}
    for cname, ch in pruned_net.named_children():
        for bname, buf in ch.named_buffers():
            if bname == 'weight_mask':
                pruned_masks[cname] = buf.to(device=DEVICE, dtype=torch.bool)
        for pname, param in ch.named_parameters():
            pruned_params[(cname, pname)] = param.to(device=DEVICE)
    for cname, ch in global_model.named_children():
        for bname, buf in ch.named_buffers():
            if bname == 'weight_mask':
                buf.copy_(pruned_masks[cname])
        for pname, param in ch.named_parameters():
            param.data.copy_(pruned_params[(cname, pname)])
else:
    print("Initial global model pruning ...")
    print("Skipping initial model pruning")
    # global_model.layer_prune(sparsity=SPARSITY, sparsity_distribution=args.sparsity_distribution)
    # for name, weights in global_model.state_dict().items():
    #     if name[-5] == '_':
    #         print(f"Global model mask {name} after 1st layer prune is dense: {torch.prod(torch.Tensor(list(weights.shape))) == weights.sum()}") 

initial_global_params = deepcopy(global_model.state_dict())
for name, weights in global_model.state_dict().items():
        if name[-5] == '_':
            print(f"Global model mask {name} before client training is dense: {torch.prod(torch.Tensor(list(weights.shape))) == weights.sum()}") 

# we need to accumulate compute/DL/UL costs regardless of round number, resetting only
# when we actually report these numbers
compute_times = np.zeros(len(clients)) # time in seconds taken on client-side for round
download_cost = np.zeros(len(clients))
upload_cost = np.zeros(len(clients))

# for each round t = 1, 2, ... do
for server_round in tqdm(range(ROUNDS)):

    if server_round != 0:
        global_model.layer_prune(sparsity=SPARSITY, sparsity_distribution=SPARSITY_DISTRIBUTION)
    
    for name, weights in global_model.state_dict().items():
        if name[-5] == '_':
            print(f"Round {server_round}, global model layer {name} is dense: {torch.prod(torch.Tensor(list(weights.shape))) == weights.sum()}") 

    # sample clients
    client_indices = rng.choice(list(clients.keys()), size=CLIENTS)

    global_params = global_model.state_dict()
    aggregated_params = {}
    aggregated_params_for_mask = {}
    aggregated_masks = {}
    # set server parameters to 0 in preparation for aggregation,
    for name, param in global_params.items():
        if name.endswith('_mask'):
            continue
        aggregated_params[name] = torch.zeros_like(param, dtype=torch.float, device=DEVICE)
        aggregated_params_for_mask[name] = torch.zeros_like(param, dtype=torch.float, device=DEVICE)
        if needs_mask(name):
            aggregated_masks[name] = torch.zeros_like(param, device=DEVICE)

    # for each client k \in S_t in parallel do
    total_sampled = 0
    client_weights = []
    client_masks = []
    full_client_masks = []
    for client_id in client_indices:
        client = clients[client_id]
        i = client_ids.index(client_id)
        # Local client training.
        t0 = time.process_time()
        if args.rate_decay_method == 'cosine':
            READJUSTMENT_RATIO = READJUSTMENT_RATIO * global_model._decay(server_round, alpha=READJUSTMENT_RATIO, t_end=args.rate_decay_end)
        else:
            READJUSTMENT_RATIO = READJUSTMENT_RATIO
        readjust = (server_round - 1) % args.rounds_between_readjustments == 0 and READJUSTMENT_RATIO > 0.
        if readjust:
            dprint('readjusting', READJUSTMENT_RATIO)
        # determine sparsity desired at the end of this round
        # ...via linear interpolation
        if server_round <= args.rate_decay_end:
            round_sparsity = SPARSITY * (args.rate_decay_end - server_round) / args.rate_decay_end + FINAL_SPARSITY * server_round / args.rate_decay_end
        else:
            round_sparsity = FINAL_SPARSITY
        
        print(f"Round {server_round} sparsity: {round_sparsity}")

        # actually perform training
        train_result = client.train(global_params=global_params, initial_global_params=initial_global_params,
                                    readjustment_ratio=READJUSTMENT_RATIO,
                                    readjust=readjust, sparsity=round_sparsity)
        cl_params = train_result['state']
        download_cost[i] = train_result['dl_cost']
        upload_cost[i] = train_result['ul_cost']
            
        t1 = time.process_time()
        compute_times[i] = t1 - t0
        client.net.clear_gradients() # to save memory

        # Check whether client weights are dense
        for name, weights in client.net.state_dict().items():
            if name[-5] == '_':
                print(f"Round {server_round} client {client_id} layer {name} is dense: {torch.prod(torch.Tensor(list(weights.shape))) == weights.sum()}") 

        # add this client's params to the aggregate
        cl_weight_params = {}
        cl_mask_params = {}

        # first deduce masks for the received weights
        for name, cl_param in cl_params.items():
            if name.endswith('_orig'):
                name = name[:-5]
            elif name.endswith('_mask'):
                name = name[:-5]
                cl_mask_params[name] = cl_param.to(device=DEVICE, copy=True)
                continue

            cl_weight_params[name] = cl_param.to(device=DEVICE, copy=True)
            if FP16:
                cl_weight_params[name] = cl_weight_params[name].to(torch.bfloat16).to(torch.float)

        if not ALIGNED_DST:
            # at this point, we have weights and masks (possibly all-ones)
            # for this client. we will proceed by applying the mask and adding
            # the masked received weights to the aggregate, and adding the mask
            # to the aggregate as well.
            for name, cl_param in cl_weight_params.items():
                if name in cl_mask_params:
                    # things like weights have masks
                    cl_mask = cl_mask_params[name]
                    sv_mask = global_params[name + '_mask'].to('cpu', copy=True)
                    # calculate Hamming distance of masks for debugging
                    # if readjust:
                    #     dprint(f'{client.id} {name} d_h=', torch.sum(cl_mask ^ sv_mask).item())
                    if True:
                        dprint(f'{client.id} {name} d_h=', torch.sum(cl_mask ^ sv_mask).item())
                    aggregated_params[name].add_(client.train_size() * cl_param * cl_mask)
                    aggregated_params_for_mask[name].add_(client.train_size() * cl_param * cl_mask)
                    aggregated_masks[name].add_(client.train_size() * cl_mask)
                    if args.remember_old:
                        sv_mask[cl_mask] = 0
                        sv_param = global_params[name].to('cpu', copy=True)
                        aggregated_params_for_mask[name].add_(client.train_size() * sv_param * sv_mask)
                        aggregated_masks[name].add_(client.train_size() * sv_mask)
                else:
                    # things like biases don't have masks
                    aggregated_params[name].add_(client.train_size() * cl_param)
        else:
            client_weights.append(copy.deepcopy(cl_weight_params))

            copied_cl_mask_params = copy.deepcopy(cl_mask_params)
            # HACKY #
            copied_cl_mask_params["conv1.bias"] = torch.zeros_like(client.net.state_dict()["conv1.bias"])
            copied_cl_mask_params["conv2.bias"] = torch.zeros_like(client.net.state_dict()["conv2.bias"])
            copied_cl_mask_params["fc1.bias"] = torch.zeros_like(client.net.state_dict()["fc1.bias"])
            copied_cl_mask_params["fc2.bias"] = torch.zeros_like(client.net.state_dict()["fc2.bias"])
            if args.dataset in ['emnist', 'fashionmnist']:
                copied_cl_mask_params["conv3.bias"] = torch.zeros_like(client.net.state_dict()["conv3.bias"])
            if args.dataset == 'cifar10':
                copied_cl_mask_params["fc3.bias"] = torch.zeros_like(client.net.state_dict()["fc3.bias"])
            
            client_masks.append(copy.deepcopy(cl_mask_params))
            full_client_masks.append(copied_cl_mask_params)

    if ALIGNED_DST:
        if ANCHOR_SELECTION_RULE == 'random':
            anchor_index = np.random.choice(a=np.arange(CLIENTS))
        elif ANCHOR_SELECTION_RULE == 'best_combination':
            anchor_index = get_best_anchor_index(
                client_weights
            )
        else:
            print("ANCHOR_SELECTION_RULE not implemented")

        if TEST:
            aligned_weights = client_weights
            aligned_masks = client_masks
        else:
            aligned_weights, aligned_masks = align_models(
                all_models=all_models,
                dataset=args.dataset,
                clients=CLIENTS,
                anchor_index=anchor_index,
                weights_results=client_weights,
                masks_results=full_client_masks,
                permutation_spec=permutation_spec
            )

        # 2. Aggregate Weights and Masks
        for j, cl_weight_params in enumerate(aligned_weights):
            for name, cl_param in cl_weight_params.items():
                if name in client_masks[j]:
                    # things like weights have masks
                    cl_mask = aligned_masks[j][name].long().to(DEVICE, copy=True)
                    sv_mask = global_params[name + '_mask'].to(DEVICE, copy=True)
                    # calculate Hamming distance of masks for debugging
                    # if readjust:
                    #     dprint(f'{client.id} {name} d_h=', torch.sum(cl_mask ^ sv_mask).item())
                    if True:
                        dprint(f'{client.id} {name} d_h=', torch.sum(cl_mask ^ sv_mask).item())
                    aggregated_params[name].add_(client.train_size() * cl_param.to(DEVICE, copy=True) * cl_mask)
                    aggregated_params_for_mask[name].add_(client.train_size() * cl_param.to(DEVICE, copy=True) * cl_mask)
                    aggregated_masks[name].add_(client.train_size() * cl_mask.to(DEVICE, copy=True))
                    if args.remember_old:
                        sv_mask[cl_mask] = 0
                        sv_param = global_params[name].to(DEVICE, copy=True)
                        aggregated_params_for_mask[name].add_(client.train_size() * sv_param * sv_mask)
                        aggregated_masks[name].add_(client.train_size() * sv_mask.to(DEVICE, copy=True))
                else:
                    # things like biases don't have masks
                    aggregated_params[name].add_(client.train_size() * cl_param.to(DEVICE, copy=True))

    # at this point, we have the sum of client parameters
    # in aggregated_params, and the sum of masks in aggregated_masks. We
    # can take the average now by simply dividing...
    for name, param in aggregated_params.items():
        # if this parameter has no associated mask, simply take the average.
        if name not in aggregated_masks:
            aggregated_params[name] /= sum(clients[i].train_size() for i in client_indices)
            continue
        # drop parameters with not enough votes
        aggregated_masks[name] = F.threshold_(aggregated_masks[name], args.min_votes, 0)
        # otherwise, we are taking the weighted average w.r.t. the number of 
        # samples present on each of the clients.
        aggregated_params[name] /= aggregated_masks[name]
        aggregated_params_for_mask[name] /= aggregated_masks[name]
        aggregated_masks[name] /= aggregated_masks[name]
        # it's possible that some weights were pruned by all clients. In this
        # case, we will have divided by zero. Those values have already been
        # pruned out, so the values here are only placeholders.
        aggregated_params[name] = torch.nan_to_num(aggregated_params[name],
                                                   nan=0.0, posinf=0.0, neginf=0.0)
        aggregated_params_for_mask[name] = torch.nan_to_num(aggregated_params[name],
                                                   nan=0.0, posinf=0.0, neginf=0.0)
        aggregated_masks[name] = torch.nan_to_num(aggregated_masks[name],
                                                  nan=0.0, posinf=0.0, neginf=0.0)

    # masks are parameters too!
    for name, mask in aggregated_masks.items():
        aggregated_params[name + '_mask'] = mask
        aggregated_params_for_mask[name + '_mask'] = mask

    # reset global params to aggregated values
    global_model.load_state_dict(aggregated_params_for_mask)

    if global_model.sparsity() < round_sparsity:
        # we now have denser networks than we started with at the beginning of
        # the round. reprune on the server to get back to the desired sparsity.
        # we use layer-wise magnitude pruning as before.
        global_model.layer_prune(sparsity=round_sparsity, sparsity_distribution=SPARSITY_DISTRIBUTION)

    # discard old weights and apply new mask
    global_params = global_model.state_dict()
    for name, mask in aggregated_masks.items():
        new_mask = global_params[name + '_mask']
        aggregated_params[name + '_mask'] = new_mask
        aggregated_params[name][~new_mask] = 0
    global_model.load_state_dict(aggregated_params)

    # evaluate performance
    torch.cuda.empty_cache()
    if server_round % args.eval_every == 0 and args.eval:
        accuracies, sparsities, mean_acc = evaluate_global(clients, global_model, progress=True,
                                                 n_batches=args.test_batches)

        print(f'Round {server_round}, test. acc.: {mean_acc}')

    for client_id in clients:
        i = client_ids.index(client_id)
        if server_round % args.eval_every == 0 and args.eval:
            print_csv_line(pid=args.pid,
                           dataset=args.dataset,
                           clients=CLIENTS,
                           total_clients=len(clients),
                           round=server_round,
                           batch_size=args.batch_size,
                           epochs=LOCAL_EPOCHS,
                           target_sparsity=round_sparsity,
                           pruning_rate=READJUSTMENT_RATIO,
                           initial_pruning_threshold='',
                           final_pruning_threshold='',
                           pruning_threshold_growth_method='',
                           pruning_method='',
                           lth=False,
                           client_id=client_id,
                           accuracy=accuracies[client_id],
                           sparsity=sparsities[client_id],
                           compute_time=compute_times[i],
                           download_cost=download_cost[i],
                           upload_cost=upload_cost[i],
                           server_round=server_round,
                           mean_acc=mean_acc
                           )

        # if we didn't send initial global params to any clients in the first round, send them now.
        # (in the real world, this could be implemented as the transmission of
        # a random seed, so the time and place for this is not a concern to us)
        if server_round == 0:
            clients[client_id].initial_global_params = initial_global_params

    if server_round % args.eval_every == 0 and args.eval:
        # clear compute, UL, DL costs
        compute_times[:] = 0
        download_cost[:] = 0
        upload_cost[:] = 0