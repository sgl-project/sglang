"""Fixed-shape state I/O for kernels without a masked pool-index interface."""

import msgspec
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


class PoolStaging(msgspec.Struct, frozen=True):
    """Rows for kernels that need a contiguous state pool, keeping the pool
    convention that row 0 is the padding sink: contiguous copies hold the sink
    at row 0 and request ``i`` at row ``i + 1``, and a padded request
    (index <= 0) is addressed at the sink. Every layer of a forward stages the
    same requests, so one instance serves them all."""

    # [num_requests + 1] int64: the pool row each copy row is read from.
    gather_rows: torch.Tensor
    # [num_requests]: the copy row of each request, to hand the kernels.
    kernel_rows: torch.Tensor

    @classmethod
    def from_indices(cls, indices: torch.Tensor) -> "PoolStaging":
        pool_rows = torch.where(indices > 0, indices, 0).to(torch.int64)
        kernel_rows = torch.where(
            indices > 0,
            torch.arange(
                1, indices.shape[0] + 1, device=indices.device, dtype=indices.dtype
            ),
            0,
        )
        return cls(
            gather_rows=torch.cat([pool_rows.new_zeros(1), pool_rows]),
            kernel_rows=kernel_rows,
        )


def stage_pool_states(
    states: tuple[torch.Tensor, ...], staging: PoolStaging
) -> tuple[torch.Tensor, ...]:
    """Contiguous copies of the rows ``staging`` names."""
    return tuple(state[staging.gather_rows].contiguous() for state in states)


def unstage_pool_states(
    states: tuple[torch.Tensor, ...],
    copies: tuple[torch.Tensor, ...],
    staging: PoolStaging,
) -> None:
    """Write `stage_pool_states` copies back: request ``i``'s row goes to its
    pool row; a padded request's row, untouched by the kernels, goes back to
    the sink it was read from."""
    pool_rows = staging.gather_rows[1:]
    for state, copy in zip(states, copies):
        state[pool_rows] = copy[1:]
