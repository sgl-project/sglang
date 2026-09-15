# SPDX-License-Identifier: Apache-2.0
"""Inspect finalized module state without state_dict hooks or alias loss."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .descriptors import StateManifest, StorageDescriptor, TensorDescriptor


def iter_modules(
    module: nn.Module, prefix: str = "", ancestors: frozenset = frozenset()
):
    if id(module) in ancestors:
        raise ValueError(f"Cyclic module tree at {prefix!r}")
    yield prefix, module
    ancestors = ancestors | {id(module)}
    for name, child in sorted(module._modules.items()):
        if child is not None:
            path = f"{prefix}.{name}" if prefix else name
            yield from iter_modules(child, path, ancestors)


@dataclass(frozen=True)
class StateSnapshot:
    manifest: StateManifest
    tensors: dict[str, torch.Tensor]
    # Strong references to the ORIGINAL storages, not serialized IPC handles.
    storages: dict[str, torch.UntypedStorage]


def snapshot_module(module: nn.Module) -> StateSnapshot:
    entries, training = [], []
    for prefix, child in iter_modules(module):
        training.append((prefix, child.training))
        for kind, iterator in (
            (
                "parameter",
                child.named_parameters(recurse=False, remove_duplicate=False),
            ),
            ("buffer", child.named_buffers(recurse=False, remove_duplicate=False)),
        ):
            for name, tensor in iterator:
                path = f"{prefix}.{name}" if prefix else name
                persistent = (
                    kind == "parameter" or name not in child._non_persistent_buffers_set
                )
                entries.append((path, kind, persistent, tensor))

    tensor_groups, storage_groups = {}, {}
    tensors, storages, descriptions, storage_descriptions = {}, {}, [], []
    for name, kind, persistent, tensor in sorted(entries, key=lambda item: item[0]):
        if tensor.layout != torch.strided or tensor.is_quantized:
            raise ValueError(f"Unsupported tensor layout/quantization: {name}")
        if tensor.is_conj() or tensor.is_neg():
            raise ValueError(f"Unresolved conjugate/negative view: {name}")
        if kind == "buffer" and type(tensor) is not torch.Tensor:
            raise ValueError(
                f"Buffer subclasses/cross-kind ties need an adapter: {name}"
            )
        storage = tensor.untyped_storage()
        # _cdata distinguishes zero-byte storages whose data_ptr() is always 0.
        storage_key = (str(tensor.device), storage._cdata)
        if storage_key not in storage_groups:
            group = f"s{len(storage_groups)}"
            storage_groups[storage_key] = group
            storage_descriptions.append(StorageDescriptor(group, storage.nbytes()))
            storages[group] = storage
        storage_group = storage_groups[storage_key]
        tensor_group = tensor_groups.setdefault(id(tensor), f"t{len(tensor_groups)}")
        tensors[name] = tensor
        descriptions.append(
            TensorDescriptor(
                name=name,
                kind=kind,
                persistent=persistent,
                dtype=str(tensor.dtype).removeprefix("torch."),
                shape=tuple(tensor.shape),
                stride=tuple(tensor.stride()),
                storage_offset=tensor.storage_offset(),
                storage_group=storage_group,
                tensor_group=tensor_group,
                requires_grad=tensor.requires_grad,
            )
        )
    manifest = StateManifest(
        tuple(storage_descriptions), tuple(descriptions), tuple(sorted(training))
    )
    manifest.validate()
    return StateSnapshot(manifest, tensors, storages)


def storage_byte_views(snapshot: StateSnapshot) -> dict[str, torch.Tensor]:
    """One zero-copy, full-storage uint8 view per allocation."""
    return {
        group: torch.empty(0, dtype=torch.uint8, device=storage.device).set_(
            storage, 0, (storage.nbytes(),), (1,)
        )
        for group, storage in snapshot.storages.items()
    }
