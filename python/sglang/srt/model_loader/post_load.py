# SPDX-License-Identifier: Apache-2.0

"""Device staging for post-load weight processing."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import nn

__all__ = ["stage_module_for_post_load"]


@dataclass(slots=True)
class _TensorState:
    tensor: torch.Tensor
    original_data: torch.Tensor
    origin: torch.device
    staged_data: torch.Tensor | None = None


_SlotKey = tuple[int, str, str]


def _iter_registered_tensors(
    module: nn.Module,
) -> Iterator[tuple[nn.Module, str, str, torch.Tensor]]:
    # inspect the registries directly so aliases and non-persistent buffers are
    # retained. named_parameters()/named_buffers() remove duplicate objects.
    for owner in module.modules():
        for registry_name, registry in (
            ("_parameters", owner._parameters),
            ("_buffers", owner._buffers),
        ):
            for name, tensor in registry.items():
                if tensor is not None:
                    yield owner, registry_name, name, tensor


def _slot_key(owner: nn.Module, registry_name: str, name: str) -> _SlotKey:
    return id(owner), registry_name, name


def _same_staged_data(current: torch.Tensor, staged: torch.Tensor) -> bool:
    if current.layout != torch.strided or staged.layout != torch.strided:
        return False
    return (
        current.device == staged.device
        and current.data_ptr() == staged.data_ptr()
        and current.shape == staged.shape
        and current.dtype == staged.dtype
        and current.stride() == staged.stride()
    )


def _copy_data_to_device(
    data: torch.Tensor,
    device: torch.device,
    *,
    pin_memory: bool,
) -> torch.Tensor:
    if data.device == device:
        return data
    if device.type == "cpu" and data.layout == torch.strided and not data.is_quantized:
        result = torch.empty_strided(
            size=data.size(),
            stride=data.stride(),
            dtype=data.dtype,
            layout=data.layout,
            device=device,
            pin_memory=pin_memory,
        )
        result.copy_(data)
        return result
    return data.to(device)


def _restore_tensor(
    tensor: torch.Tensor,
    destination: torch.device,
    original_state: _TensorState | None,
    *,
    pin_memory: bool,
) -> None:
    if tensor.is_meta:
        raise RuntimeError("Post-load processing produced a meta tensor")

    if (
        original_state is not None
        and tensor is original_state.tensor
        and original_state.staged_data is not None
        and _same_staged_data(tensor.data, original_state.staged_data)
    ):
        original_state.original_data.copy_(tensor.data)
        tensor.data = original_state.original_data
        return

    tensor.data = _copy_data_to_device(
        tensor.data,
        destination,
        pin_memory=pin_memory,
    )


@contextmanager
def stage_module_for_post_load(
    module: nn.Module,
    process_device: torch.device,
    *,
    pin_memory: bool = False,
) -> Iterator[nn.Module]:
    """Temporarily stage a module's registered state for a post-load hook.

    Existing parameters and buffers are restored to their per-slot devices.
    Replacements inherit the slot device unless they newly occupy a
    non-persistent buffer slot; those remain on ``process_device`` because
    state-dict based offload cannot move them on demand. Other new tensors use
    their owner's unique original device, then the module's unique original
    device, and otherwise remain on ``process_device``.

    This context only owns tensor residency. Hook selection, invocation count,
    and model/component lifecycle remain the caller's responsibility.
    """

    if process_device.type == "meta":
        raise ValueError("process_device cannot be meta")

    original_slots: dict[_SlotKey, _TensorState] = {}
    original_nonpersistent_buffer_slots: set[_SlotKey] = set()
    original_name_origins: dict[tuple[int, str], torch.device] = {}
    owner_origins: dict[int, set[torch.device]] = {}
    module_origins: set[torch.device] = set()
    tensor_states: dict[int, _TensorState] = {}

    # snapshot and validate all state before moving any tensor
    for owner, registry_name, name, tensor in _iter_registered_tensors(module):
        if tensor.is_meta:
            raise RuntimeError(
                f"Cannot post-process meta tensor {type(owner).__name__}.{name}"
            )
        state = tensor_states.get(id(tensor))
        if state is None:
            state = _TensorState(tensor, tensor.data, tensor.device)
            tensor_states[id(tensor)] = state
        key = _slot_key(owner, registry_name, name)
        original_slots[key] = state
        if registry_name == "_buffers" and name in owner._non_persistent_buffers_set:
            original_nonpersistent_buffer_slots.add(key)
        original_name_origins[(id(owner), name)] = tensor.device
        owner_origins.setdefault(id(owner), set()).add(tensor.device)
        module_origins.add(tensor.device)

    try:
        for state in tensor_states.values():
            if state.origin != process_device:
                state.staged_data = state.tensor.data.to(process_device)
                state.tensor.data = state.staged_data
        yield module
    finally:
        restore_plan: dict[
            int, tuple[torch.Tensor, torch.device, _TensorState | None]
        ] = {}
        unique_module_origin = (
            next(iter(module_origins)) if len(module_origins) == 1 else None
        )
        for owner, registry_name, name, tensor in _iter_registered_tensors(module):
            key = _slot_key(owner, registry_name, name)
            original_state = original_slots.get(key)
            if original_state is None:
                original_state = tensor_states.get(id(tensor))
            original_name_origin = original_name_origins.get((id(owner), name))
            is_new_nonpersistent_buffer = (
                registry_name == "_buffers"
                and name in owner._non_persistent_buffers_set
                and key not in original_nonpersistent_buffer_slots
            )
            if is_new_nonpersistent_buffer:
                destination = process_device
            elif original_state is not None:
                destination = original_state.origin
            else:
                destination = original_name_origin
                if destination is None:
                    origins = owner_origins.get(id(owner), set())
                    destination = next(iter(origins)) if len(origins) == 1 else None
                if destination is None:
                    destination = unique_module_origin or process_device

            if tensor.is_meta:
                raise RuntimeError("Post-load processing produced a meta tensor")
            previous = restore_plan.get(id(tensor))
            if previous is not None:
                if previous[1] != destination:
                    raise RuntimeError(
                        "Aliased post-load tensor has conflicting restore devices"
                    )
                if previous[2] is None and original_state is not None:
                    restore_plan[id(tensor)] = (tensor, destination, original_state)
                continue
            restore_plan[id(tensor)] = (tensor, destination, original_state)

        restore_errors: list[Exception] = []
        for tensor, destination, original_state in restore_plan.values():
            try:
                _restore_tensor(
                    tensor,
                    destination,
                    original_state,
                    pin_memory=pin_memory,
                )
            except Exception as error:
                restore_errors.append(error)
        if len(restore_errors) == 1:
            raise restore_errors[0]
        if restore_errors:
            raise RuntimeError(
                f"Multiple post-load tensor restores failed: {restore_errors!r}"
            ) from restore_errors[0]
