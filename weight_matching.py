from collections import defaultdict
from re import L
from typing import NamedTuple

import torch
from scipy.optimize import linear_sum_assignment

from floco.config.config import DEVICE
from floco.mode_connectivity.permutation_specs import *

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

def test_weight_matching():
  """If we just have a single hidden layer then it should converge after just one step."""
  ps = mlp_permutation_spec(num_hidden_layers=3)
  rng = torch.Generator()
  rng.manual_seed(13)
  num_hidden = 10
  shapes = {
      "layer0.weight": (2, num_hidden),
      "layer0.bias": (num_hidden, ),
      "layer1.weight": (num_hidden, 3),
      "layer1.bias": (3, )
  }

  params_a = {k: torch.randn(shape, generator=rng) for k, shape in shapes.items()}
  params_b = {k: torch.randn(shape, generator=rng) for k, shape in shapes.items()}
  perm = weight_matching(rng, ps, params_a, params_b)

# if __name__ == "__main__":
#   test_weight_matching()