# SPDX-License-Identifier: Apache-2.0
"""Transfer request extras with nested tensors through the normal tensor codec."""

import json

import msgspec
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
        if isinstance(value, msgspec.Struct):
            # Data-only model execution plans (e.g. H3's resolved media
            # geometry) must survive the hop without resolving them again.
            try:
                value = msgspec.to_builtins(value)
            except TypeError:
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
            metadata = dict(
                spec=treespec_dumps(spec),
                leaves=leaves,
                indices=indices,
                cpu_indices=[
                    i
                    for i, tensor in zip(indices, tensors)
                    if tensor.device.type == "cpu"
                ],
            )
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
        cpu_indices = set(metadata.get("cpu_indices", ()))
        for index, tensor in zip(
            metadata["indices"], tensor_fields.pop(name), strict=True
        ):
            # Transport loads tensors on the receiver's compute device.
            # Keep CPU metadata/noise on CPU for stages that combine it with
            # locally constructed masks or perform deterministic CPU math.
            leaves[index] = tensor.cpu() if index in cpu_indices else tensor
        extra[name[len("_extra_tensor_tree_") :]] = tree_unflatten(
            leaves, treespec_loads(metadata["spec"])
        )
