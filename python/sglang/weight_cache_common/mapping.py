# SPDX-License-Identifier: Apache-2.0
"""Transactional validation and zero-copy registration of finalized state."""

from __future__ import annotations

import torch
from torch import nn

from .descriptors import StateManifest, resolve_dtype
from .traversal import iter_modules, snapshot_module


def register_tensor(
    module: nn.Module,
    name: str,
    tensor: torch.Tensor,
    *,
    is_param: bool,
    persistent: bool = True,
) -> None:
    """Register an already reconstructed object without breaking exact ties.

    Shared with SRT's setter. Finalized state can change registration kind
    (e.g. quantization turns an original parameter into a buffer).
    """
    if is_param:
        setattr(module, name, tensor)
    else:
        if name in module._parameters:
            del module._parameters[name]
        elif hasattr(module, name) and name not in module._buffers:
            delattr(module, name)
        module.register_buffer(name, tensor, persistent=persistent)


def _supported_metadata(value, visited: set[int]) -> bool:
    if (
        value is None
        or isinstance(
            value, (str, int, float, complex, bool, torch.dtype, torch.device)
        )
        or callable(value)
    ):
        return True
    if id(value) in visited:
        return True
    visited.add(id(value))
    if isinstance(value, dict):
        return all(
            _supported_metadata(item, visited)
            for pair in value.items()
            for item in pair
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return all(_supported_metadata(item, visited) for item in value)
    return False


def validate_meta_schema(module: nn.Module, manifest: StateManifest) -> None:
    """Adapters must construct the FINALIZED schema, not run weight transforms."""
    manifest.validate()
    target = snapshot_module(module)
    if any(tensor.device.type != "meta" for tensor in target.tensors.values()):
        raise ValueError("Cache import requires an entirely meta-initialized state")
    expected = {tensor.name: tensor for tensor in manifest.tensors}
    actual = {tensor.name: tensor for tensor in target.manifest.tensors}
    if expected.keys() != actual.keys():
        raise ValueError(
            f"Tensor names differ: missing={sorted(expected.keys() - actual.keys())}, "
            f"unexpected={sorted(actual.keys() - expected.keys())}"
        )
    for name, descriptor in expected.items():
        if actual[name] != descriptor:
            raise ValueError(f"Finalized tensor schema differs at {name!r}")
        template = target.tensors[name]
        if descriptor.kind == "parameter" and type(template) is not nn.Parameter:
            raise ValueError(f"Parameter subclass needs an explicit adapter: {name}")
        if not _supported_metadata(template.__dict__, set()):
            raise ValueError(
                f"Tensor-valued/custom parameter metadata needs an adapter: {name}"
            )
    modules = dict(iter_modules(module))
    if modules.keys() != dict(manifest.training).keys():
        raise ValueError("Module training-state tree differs")
    modes = {}
    for name, mode in manifest.training:
        if modes.setdefault(id(modules[name]), mode) != mode:
            raise ValueError(f"Conflicting training modes for shared module: {name}")


def import_state(
    module: nn.Module, manifest: StateManifest, storage_views: dict[str, torch.Tensor]
) -> None:
    """Validate everything before replacing any registration; never copy weights.

    Callers own the producer-lifetime guard and keep it active for as long as
    this module (including any separately retained tensor views) can be used.
    """
    validate_meta_schema(module, manifest)
    if storage_views.keys() != {storage.group for storage in manifest.storages}:
        raise ValueError("Imported storage groups differ from manifest")
    storage_ids = [view.untyped_storage()._cdata for view in storage_views.values()]
    if len(set(storage_ids)) != len(storage_ids):
        raise ValueError("Distinct storage groups unexpectedly alias")
    for storage in manifest.storages:
        view = storage_views[storage.group]
        if (
            view.device.type == "meta"
            or view.dtype != torch.uint8
            or view.layout != torch.strided
            or tuple(view.shape) != (storage.nbytes,)
            or view.stride() != (1,)
            or view.storage_offset() != 0
            or view.untyped_storage().nbytes() != storage.nbytes
        ):
            raise ValueError(f"Invalid full-storage byte view: {storage.group}")
    old = snapshot_module(module).tensors
    replacements, objects = {}, {}
    for descriptor in manifest.tensors:
        tensor = objects.get(descriptor.tensor_group)
        if tensor is None:
            backing = storage_views[descriptor.storage_group]
            tensor = torch.empty(
                0, dtype=resolve_dtype(descriptor.dtype), device=backing.device
            ).set_(
                backing.untyped_storage(),
                descriptor.storage_offset,
                descriptor.shape,
                descriptor.stride,
            )
            if descriptor.kind == "parameter":
                tensor = nn.Parameter(tensor, requires_grad=descriptor.requires_grad)
                # Local constructor metadata only: no remote Python state or
                # daemon-bound methods. Adapters own post-load derived state.
                tensor.__dict__.update(old[descriptor.name].__dict__)
            else:
                tensor.requires_grad_(descriptor.requires_grad)
            objects[descriptor.tensor_group] = tensor
        replacements[descriptor.name] = tensor

    modules = dict(iter_modules(module))
    for descriptor in manifest.tensors:
        parent, _, name = descriptor.name.rpartition(".")
        child = modules[parent]
        register_tensor(
            child,
            name,
            replacements[descriptor.name],
            is_param=descriptor.kind == "parameter",
            persistent=descriptor.persistent,
        )
    # Do not call train()/eval(): recursive calls would overwrite a child's
    # independently recorded mode and may invoke model-specific side effects.
    for name, mode in manifest.training:
        modules[name].training = mode
