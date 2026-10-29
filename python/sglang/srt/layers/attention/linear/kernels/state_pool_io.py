"""Fixed-shape state I/O for kernels without a masked pool-index interface."""

import torch


def state_pool_indices(
    indices: torch.Tensor, *, dtype: torch.dtype = torch.int64
) -> torch.Tensor:
    # Tensor extents may outlive their backing; only slot 0 is the padding sink.
    return indices.clamp_min(0).to(dtype)


def gather_pool_states(
    states: torch.Tensor, indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Slot 0 is the backed padding sink in both static and unified state pools.
    safe_indices = state_pool_indices(indices)
    live_rows = (indices > 0).view(-1, *((1,) * (states.ndim - 1)))
    return safe_indices, live_rows, torch.where(live_rows, states[safe_indices], 0)


def scatter_pool_states(
    states: torch.Tensor,
    indices: torch.Tensor,
    live_rows: torch.Tensor,
    updates: torch.Tensor,
) -> None:
    # Every padded lane writes the same zero row, independent of kernel output.
    states.index_copy_(0, indices, torch.where(live_rows, updates.to(states.dtype), 0))
