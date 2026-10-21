# SPDX-License-Identifier: Apache-2.0
"""Named stage fields and strict tensor-tree serialization at deployment edges."""

import base64
import copy
import io
import json
import pickle

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
    if path == "result.output":
        return req.output
    if path == "result.audio":
        return req.audio
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
        if path in (
            "result.output",
            "result.audio",
            "result.audio_sample_rate",
            "result.fps",
        ):
            setattr(req, parts[1], value)
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


class _ContextPickler(pickle.Pickler):
    """encode trusted Python state while keeping tensor storage out of control metadata"""

    def __init__(self, stream):
        super().__init__(stream, protocol=pickle.HIGHEST_PROTOCOL)
        self.tensors = {}
        self.tensor_ids = {}
        self.generators = {}

    def _tensor(self, tensor):
        identity = id(tensor)
        if identity not in self.tensor_ids:
            key = str(len(self.tensors))
            self.tensor_ids[identity] = key
            self.tensors[key] = tensor
        return self.tensor_ids[identity]

    def persistent_id(self, value):
        if isinstance(value, torch.Tensor):
            return "tensor", self._tensor(value), value.device.type
        if isinstance(value, torch.Generator):
            # shared generators retain identity and consumed state, not just their seed
            identity = id(value)
            if identity not in self.generators:
                self.generators[identity] = self._tensor(value.get_state())
            return "generator", self.generators[identity], value.device.type
        if isinstance(value, torch.device):
            return "device", value.type
        if isinstance(value, torch.nn.Module):
            raise TypeError("Model modules cannot cross a request-context boundary")
        return None


class _ContextUnpickler(pickle.Unpickler):
    """restore original CPU semantics and remap CUDA state to the receiving worker"""

    def __init__(self, stream, tensors, device):
        super().__init__(stream)
        self.tensors = tensors
        self.device = device
        self.generators = {}

    def persistent_load(self, reference):
        kind, *fields = reference
        if kind == "device":
            return torch.device(self.device if fields[0] == "cuda" else fields[0])
        key, device_type = fields
        tensor = self.tensors[key]
        if kind == "tensor":
            if device_type == "cpu" and tensor.device.type != "cpu":
                tensor = self.tensors[key] = tensor.cpu()
            return tensor
        if kind == "generator":
            if key not in self.generators:
                device = self.device if device_type == "cuda" else device_type
                self.generators[key] = torch.Generator(device=device).set_state(
                    tensor.cpu()
                )
            return self.generators[key]
        raise ValueError(f"Unknown request-context reference: {kind}")


def capture_request(req):
    """snapshot request-owned state, excluding worker-local instrumentation/modules"""
    # tensors, including those nested inside scheduler state, never enter pickle
    state = {
        name: value
        for name, value in vars(req).items()
        if name not in ("metrics", "trace_ctx", "modules")
    }
    state["extra"] = {
        key: value
        for key, value in req.extra.items()
        if key not in ("_deployment_result", "_deployment_peak_memory_mb")
    }
    if "_deployment_result" in req.extra:
        # preserve the result without migrating its process-local metrics
        result = copy.copy(req.extra["_deployment_result"])
        result.metrics = None
        result.metrics_list = None
        state["extra"]["_deployment_result"] = result
    if req.modules:
        raise TypeError("Request modules must remain local to their pipeline worker")
    stream = io.BytesIO()
    encoder = _ContextPickler(stream)
    try:
        encoder.dump(state)
    except (TypeError, pickle.PicklingError) as exc:
        raise TypeError(
            f"Request contains non-transferable runtime state: {exc}"
        ) from exc
    return {
        "structure": base64.b64encode(stream.getvalue()).decode("ascii"),
        "tensors": encoder.tensors,
    }


def restore_request(req, context, device):
    # this RPC is restricted to trusted workers running the same source version
    stream = io.BytesIO(base64.b64decode(context["structure"]))
    state = _ContextUnpickler(stream, context["tensors"], device).load()
    vars(req).update(state)


def pack_fields(values):
    leaves, spec = tree_flatten(values)
    tensors, empty = {}, {}
    for index, leaf in enumerate(leaves):
        if not isinstance(leaf, torch.Tensor):
            continue
        if leaf.numel():
            tensors[str(index)] = leaf
        else:
            empty[str(index)] = (list(leaf.shape), dtype_to_str(leaf.dtype))
        leaves[index] = None
    # required fields must never silently disappear on a transport boundary
    scalar = {
        "spec": treespec_dumps(spec),
        "leaves": leaves,
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
