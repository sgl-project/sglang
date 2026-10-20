# SPDX-License-Identifier: Apache-2.0
"""Named stage fields and strict tensor-tree serialization at deployment edges."""

import copy
import json

import torch
from torch.utils._pytree import (
    tree_flatten,
    tree_unflatten,
    treespec_dumps,
    treespec_loads,
)

from sglang.multimodal_gen.runtime.disaggregation.transport.codec import (
    dtype_to_str,
    str_to_dtype,
)


def read_field(req, path):
    parts = path.split(".")
    if parts[0] == "result":
        value = vars(req.extra["_deployment_result"])
        parts = parts[1:]
    elif parts[0] == "extra":
        value = req.extra
        parts = parts[1:]
    else:
        value = vars(req) if parts[0] in vars(req) else vars(req.sampling_params)
    for part in parts:
        value = value[part]
    return value


def write_field(req, path, value):
    parts = path.split(".")
    if parts[0] == "result":
        target = req.extra.setdefault("_deployment_output_fields", {})
        parts = parts[1:]
    elif parts[0] == "extra":
        target = req.extra
        parts = parts[1:]
    elif len(parts) == 1:
        setattr(req, path, value)
        return
    else:
        target = vars(req)
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def fork_request(req):
    fork = copy.copy(req)
    fork.sampling_params = copy.deepcopy(req.sampling_params)
    fork.extra = copy.copy(req.extra)
    return fork


def pack_fields(values):
    leaves, spec = tree_flatten(values)
    tensors, metadata, empty = {}, [], {}
    for index, leaf in enumerate(leaves):
        if isinstance(leaf, torch.Tensor):
            if leaf.numel():
                tensors[str(index)] = leaf
            else:
                empty[str(index)] = (list(leaf.shape), dtype_to_str(leaf.dtype))
            metadata.append(None)
        else:
            metadata.append(leaf)
    # required fields must never silently disappear on a transport boundary
    scalar = {
        "spec": treespec_dumps(spec),
        "leaves": metadata,
        "tensors": list(tensors),
        "empty": empty,
    }
    json.dumps(scalar, allow_nan=False)
    return tensors, scalar


def unpack_fields(tensors, scalar, device="cpu"):
    if set(tensors) != set(scalar["tensors"]):
        raise ValueError("Received tensor fields do not match the payload contract")
    leaves = list(scalar["leaves"])
    for index, tensor in tensors.items():
        leaves[int(index)] = tensor
    for index, (shape, dtype) in scalar["empty"].items():
        leaves[int(index)] = torch.empty(
            shape, dtype=str_to_dtype(dtype), device=device
        )
    return tree_unflatten(leaves, treespec_loads(scalar["spec"]))
