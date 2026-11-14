# SPDX-License-Identifier: Apache-2.0
"""Transfer request extras with nested tensors through the normal tensor codec."""

import json

import torch
from torch.utils._pytree import (
    tree_flatten,
    tree_unflatten,
    treespec_dumps,
    treespec_loads,
)


def extract_extra_tensors(extra, tensor_fields, scalar_fields):
    for key, value in extra.items():
        if key.startswith("_"):
            continue
        leaves, spec = tree_flatten(value)
        indices = [i for i, leaf in enumerate(leaves) if isinstance(leaf, torch.Tensor)]
        if not indices:
            try:
                json.dumps(value)
            except (TypeError, ValueError, OverflowError):
                continue
            scalar_fields[f"_extra_{key}"] = value
            continue
        tensors = [leaves[i] for i in indices]
        for i in indices:
            leaves[i] = None
        try:
            metadata = dict(spec=treespec_dumps(spec), leaves=leaves, indices=indices)
            json.dumps(metadata)
        except (TypeError, ValueError, OverflowError, NotImplementedError):
            continue
        name = f"_extra_tensor_tree_{key}"
        tensor_fields[name] = tensors
        scalar_fields[name] = metadata


def restore_extra_tensors(extra, tensor_fields, scalar_fields):
    for name in list(scalar_fields):
        if not name.startswith("_extra_tensor_tree_"):
            continue
        metadata = scalar_fields.pop(name)
        leaves = metadata["leaves"]
        for index, tensor in zip(
            metadata["indices"], tensor_fields.pop(name), strict=True
        ):
            leaves[index] = tensor
        extra[name[len("_extra_tensor_tree_") :]] = tree_unflatten(
            leaves, treespec_loads(metadata["spec"])
        )
