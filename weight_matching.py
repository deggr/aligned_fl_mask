import copy

import torch
from scipy.optimize import linear_sum_assignment

from config import DEVICE
from floco.mode_connectivity.permutation_specs import *

from floco.fl.client import flwr_set_parameters

from permutation_specs import PermutationSpec, mlp_permutation_spec

import numpy as np

LAP_SEED = 42

def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None):
  """Get parameter `k` from `params`, with the permutations applied."""
  w = params[k].to(DEVICE)
  for axis, p in enumerate(ps.axes_to_perm[k]):
    # Skip the axis we're trying to permute.
    if axis == except_axis:
      continue

    # None indicates that there is no permutation relevant to that axis.
    if p is not None:
        w = torch.index_select(w, axis, perm[p].int().to(DEVICE)).to(DEVICE)

  return w

def apply_permutation(ps: PermutationSpec, perm, params):
  """Apply a `perm` to `params`."""
  return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}

def weight_matching(ps: PermutationSpec, params_a, params_b, max_iter=100, init_perm=None):
  """Find a permutation of `params_b` to make them match `params_a`."""
  perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}

  perm = {p: torch.arange(n).to(DEVICE) for p, n in perm_sizes.items()} if init_perm is None else init_perm
  perm_names = list(perm.keys())

  for iteration in range(max_iter):
    progress = False
    for p_ix in torch.randperm(len(perm_names)):
      p = perm_names[p_ix]
      n = perm_sizes[p]
      A = torch.zeros((n, n)).to(DEVICE)
      for wk, axis in ps.perm_to_axes[p]:
        w_a = params_a[wk].to(DEVICE)
        w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis).to(DEVICE)
        w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1)).to(DEVICE)
        w_b = torch.moveaxis(w_b, axis, 0).reshape((n, -1)).to(DEVICE)
        A += w_a @ w_b.T

      np.random.seed(LAP_SEED)
      ri, ci = linear_sum_assignment(A.cpu().detach().numpy(), maximize=True)
      assert (torch.tensor(ri) == torch.arange(len(ri))).all()
      oldL = torch.einsum('ij,ij->i', A, torch.eye(n).to(DEVICE)[perm[p].long().to(DEVICE)]).sum()
      newL = torch.einsum('ij,ij->i', A,torch.eye(n).to(DEVICE)[ci, :]).sum()
      print(f"{iteration}/{p}: {newL - oldL}")
      progress = progress or newL > oldL + 1e-12

      perm[p] = torch.Tensor(ci).to(DEVICE)

    if not progress:
      break

  return perm

def align_models(all_models, dataset, clients, anchor_index, weights_results, masks_results, permutation_spec):

    anchor_weights = weights_results[anchor_index]
    anchor_mask = masks_results[anchor_index]

    anchor_model = all_models[dataset](device=DEVICE)
    masked_anchor_model = all_models[dataset](device=DEVICE)

    anchor_model.load_state_dict(anchor_weights)
    masked_anchor_model.load_state_dict(anchor_mask)

    # flwr_set_parameters(anchor_model, anchor_weights)
    # flwr_set_parameters(masked_anchor_model,  anchor_mask)
    
    anchor_state_dict = copy.deepcopy(anchor_model.state_dict())
    anchor_mask_state_dict = copy.deepcopy(masked_anchor_model.state_dict())
    
    aligned_weights = [anchor_state_dict]
    aligned_masks = [anchor_mask_state_dict]

    target_model_ids = np.delete(np.arange(clients), anchor_index)
    target_weights = [weights_results[idx] for idx in target_model_ids]
    for i, tmp_target_weights in enumerate(target_weights):
        # Set model weights
        tmp_target_model = all_models[dataset](device=DEVICE)
        # flwr_set_parameters(
        #     tmp_target_model,
        #     tmp_target_weights
        #     )
        tmp_target_model.load_state_dict(tmp_target_weights)
        tmp_target_model_state_dict = copy.deepcopy(tmp_target_model.state_dict())
        
        # Permuted weights
        tmp_target_permutation = weight_matching(
            permutation_spec,
            anchor_state_dict, 
            tmp_target_model_state_dict
            )
        tmp_target_permuted_state_dict = apply_permutation(
            permutation_spec, 
            tmp_target_permutation, 
            tmp_target_model_state_dict
            )
        
        # Permute mask
        tmp_target_model_mask = all_models[dataset](device=DEVICE)
        # flwr_set_parameters(
        #     tmp_target_model_mask,
        #     masks_results[target_model_ids[i]]
        #     )
        tmp_target_model.load_state_dict(masks_results[target_model_ids[i]])
        tmp_target_model_mask_state_dict = copy.deepcopy(tmp_target_model.state_dict())
        tmp_target_permuted_mask_state_dict = apply_permutation(
            permutation_spec, 
            tmp_target_permutation, 
            tmp_target_model_mask_state_dict
            )

        aligned_weights.append(
            tmp_target_permuted_state_dict
        )
        aligned_masks.append(
            tmp_target_permuted_mask_state_dict
        )

    return aligned_weights, aligned_masks

# if __name__ == "__main__":
#   test_weight_matching()