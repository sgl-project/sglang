from __future__ import annotations

from collections.abc import Sequence

import torch


def _cpu_indices(indices: torch.Tensor) -> torch.Tensor:
    return indices.to(device="cpu", dtype=torch.int64)


def _device_indices(indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    return indices.to(device=device, dtype=torch.int64)


def load_to_device(
    *,
    host_tensors: Sequence[torch.Tensor],
    device_tensors: Sequence[torch.Tensor],
    host_indices: torch.Tensor,
    device_indices: torch.Tensor,
) -> None:
    if len(host_tensors) != len(device_tensors):
        raise ValueError("host_tensors and device_tensors must have the same length")
    if host_indices.numel() != device_indices.numel():
        raise ValueError("host_indices and device_indices must have the same length")
    if host_indices.numel() == 0:
        return

    host_indices = _cpu_indices(host_indices)
    for host_tensor, device_tensor in zip(host_tensors, device_tensors):
        selected = host_tensor.index_select(0, host_indices)
        selected = selected.to(
            device_tensor.device, non_blocking=host_tensor.is_pinned()
        )
        device_tensor.index_copy_(
            0, _device_indices(device_indices, device_tensor.device), selected
        )


def backup_to_host(
    *,
    device_tensors: Sequence[torch.Tensor],
    host_tensors: Sequence[torch.Tensor],
    device_indices: torch.Tensor,
    host_indices: torch.Tensor,
) -> None:
    if len(device_tensors) != len(host_tensors):
        raise ValueError("device_tensors and host_tensors must have the same length")
    if device_indices.numel() != host_indices.numel():
        raise ValueError("device_indices and host_indices must have the same length")
    if device_indices.numel() == 0:
        return

    host_indices = _cpu_indices(host_indices)
    for device_tensor, host_tensor in zip(device_tensors, host_tensors):
        selected = device_tensor.index_select(
            0, _device_indices(device_indices, device_tensor.device)
        )
        selected = selected.to(device="cpu")
        host_tensor.index_copy_(0, host_indices, selected)
