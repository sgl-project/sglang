"""Split the fused K3 [latent | shared] all-reduce when it misses quick-reduce.

The fused-front MoE path packs TP-partial latent (3584) and shared-down (7168)
into one buffer so a single collective can cover both. At 16K chunked prefill
that concat is ~336 MiB, which is above the mxmoe recipe's
``ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB=256`` cap (the cap exists so QR does not
allocate a second 336 MiB IPC buffer on top of 16K scratch). HIP custom-AR is
only 16 MiB, so the concat falls through to NCCL Generic (~1.66 ms in the
Quark 8k/1k traces, ~19% of prefill GPU).

Each slice fits the existing QR workspace: latent ~112 MiB, shared ~224 MiB.
Splitting keeps the 256 MiB cap and moves the oversized pair onto INT4
twoshot instead of raising ``ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from sglang.srt.distributed import get_tp_group, tensor_model_parallel_all_reduce
from sglang.srt.environ import envs

# K3 fused-front layout: [N, moe_hidden] | [N, hidden] with hidden = 2 * moe_hidden.
MOE_LATENT_WIDTH = 3584
MOE_SHARED_WIDTH = 7168


def qr_max_size_bytes() -> Optional[int]:
    group = get_tp_group()
    qr = getattr(group, "qr_comm", None)
    if qr is None or getattr(qr, "disabled", True):
        return None
    max_size = getattr(qr, "qr_max_size", None)
    if max_size is None or int(max_size) <= 0:
        return None
    return int(max_size)


def should_split_oversized_moe_pair(
    total_bytes: int,
    qr_max_bytes: Optional[int],
    enabled: bool,
) -> bool:
    """Pure predicate so CPU tests do not need a live TP group."""
    return bool(enabled and qr_max_bytes is not None and total_bytes > qr_max_bytes)


def moe_pair_nbytes(num_tokens: int, dtype: torch.dtype) -> int:
    return num_tokens * (MOE_LATENT_WIDTH + MOE_SHARED_WIDTH) * dtype.itemsize


def all_reduce_moe_latent_shared(
    buf: torch.Tensor,
    *,
    num_tokens: int,
    moe_hidden_size: int,
    hidden_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """All-reduce the packed [latent | shared] buffer, splitting when needed."""
    latent_numel = num_tokens * moe_hidden_size
    total_bytes = buf.numel() * buf.element_size()
    enabled = envs.SGLANG_ROCM_K3_SPLIT_OVERSIZED_MOE_AR.get()
    if should_split_oversized_moe_pair(total_bytes, qr_max_size_bytes(), enabled):
        latent = buf[:latent_numel].view(num_tokens, moe_hidden_size)
        shared = buf[latent_numel:].view(num_tokens, hidden_size)
        return (
            tensor_model_parallel_all_reduce(latent),
            tensor_model_parallel_all_reduce(shared),
        )
    buf = tensor_model_parallel_all_reduce(buf)
    return (
        buf[:latent_numel].view(num_tokens, moe_hidden_size),
        buf[latent_numel:].view(num_tokens, hidden_size),
    )
