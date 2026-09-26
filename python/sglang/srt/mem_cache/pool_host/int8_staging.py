"""Device staging buffers for the INT8 HiCache path.

Pure tensor logic, no SGLang imports and no CUDA-only ops, so it can be unit
tested on CPU locally and then exercised unchanged on the pod.

Why staging exists at all
-------------------------
SGLang's CUDA JIT HiCache mover is a byte copier. It accepts separate source and
destination *strides* but only **one** ``element_size``, so a 2048-byte BF16
device row can never be written directly into a 1152-byte encoded host row. The
BF16 KV must be encoded into a same-width staging row first, and only then moved:

    D2H:  device BF16 -> encode -> device staging [N, ROW_BYTES] -> mover -> host arena
    H2D:  host arena -> mover -> device staging [N, ROW_BYTES] -> decode -> device BF16

Buffer geometry
---------------
Each directional buffer is ``[layer_num, capacity, ROW_BYTES]``: layer-major, so
layer ``l`` is one contiguous ``[capacity, ROW_BYTES]`` slice and its per-token
stride is exactly ``ROW_BYTES``. That is what lets the all-layer mover run with a
per-layer pointer table and a single destination stride, instead of a Python loop
over layers issuing 36 small copies.

D2H and H2D own **separate** buffers because the two HiCache transfer streams can
be in flight at the same time. Sharing one buffer would let a load overwrite the
records of a backup that has not been copied out yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

#: Rows of staging allocated up front. Transfers larger than this grow the
#: buffer (grow-only, never shrink) rather than failing.
DEFAULT_STAGING_TOKENS = 2048

#: Growth quantum, so a sequence of slightly-larger transfers does not trigger a
#: reallocation on every call.
STAGING_GROWTH_QUANTUM = 1024


def next_staging_capacity(required: int, current: int) -> int:
    """Capacity to allocate for a transfer needing ``required`` rows.

    Returns ``current`` when it already fits, so the common path allocates
    nothing. Otherwise rounds up to the next quantum above ``required``.
    """
    if required <= current:
        return current
    quantum = STAGING_GROWTH_QUANTUM
    return max(current, -(-required // quantum) * quantum)


@dataclass
class StagingBuffers:
    """One directional pair of encoded staging buffers (K and V)."""

    k: torch.Tensor
    v: torch.Tensor
    layer_num: int

    @property
    def capacity(self) -> int:
        """Rows available per layer."""
        return self.k.shape[1]

    def fits(self, num_tokens: int) -> bool:
        return num_tokens <= self.capacity

    def layer_k(self, layer_id: int, num_tokens: int) -> torch.Tensor:
        """``[num_tokens, ROW_BYTES]`` view of layer ``layer_id``'s K staging."""
        return self.k[layer_id, :num_tokens]

    def layer_v(self, layer_id: int, num_tokens: int) -> torch.Tensor:
        """``[num_tokens, ROW_BYTES]`` view of layer ``layer_id``'s V staging."""
        return self.v[layer_id, :num_tokens]

    def k_layer_views(self, num_tokens: int) -> list[torch.Tensor]:
        """Per-layer K views, for a per-layer pointer table."""
        return [self.k[layer_id, :num_tokens] for layer_id in range(self.layer_num)]

    def v_layer_views(self, num_tokens: int) -> list[torch.Tensor]:
        """Per-layer V views, for a per-layer pointer table."""
        return [self.v[layer_id, :num_tokens] for layer_id in range(self.layer_num)]


def allocate_staging(
    layer_num: int,
    row_bytes: int,
    *,
    device: torch.device | str,
    capacity: int = DEFAULT_STAGING_TOKENS,
) -> StagingBuffers:
    """Allocate a zeroed ``[layer_num, capacity, row_bytes]`` uint8 K/V pair.

    Zeroed rather than empty so record padding is deterministic and a partially
    filled staging row cannot leak stale bytes from a previous layer's transfer
    into the host arena.
    """
    if capacity <= 0:
        raise ValueError(f"capacity must be positive, got {capacity}")
    shape = (layer_num, capacity, row_bytes)
    k = torch.zeros(shape, dtype=torch.uint8, device=device)
    v = torch.zeros(shape, dtype=torch.uint8, device=device)
    return StagingBuffers(k=k, v=v, layer_num=layer_num)


def pointer_table(
    tensors: list[torch.Tensor], *, device: torch.device | str
) -> torch.Tensor:
    """``uint64`` device tensor of ``data_ptr()`` values.

    Used for the all-layer mover, whose ``k_ptr_*`` arguments are plain pointer
    arrays and never inspected for dtype. For *host* (pinned) tensors the raw
    address is only valid if the memory was registered with the CUDA driver, so
    callers must pass registered buffers -- ``MHATokenToKVPoolHost`` guarantees
    this via ``pin_memory=True`` plus ``alloc_with_host_register``.
    """
    return torch.tensor(
        [t.data_ptr() for t in tensors], dtype=torch.uint64, device=device
    )
