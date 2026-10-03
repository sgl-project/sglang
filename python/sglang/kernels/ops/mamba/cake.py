"""Cake (FlashInfer) backends for the ``mamba`` operator group.

Metadata-only at import time: ``KernelSpec`` targets point at the lazy
adapter in :mod:`sglang.kernels.cake_kernels.mamba`, which imports FlashInfer
only when a kernel is actually called. Callers gate on the adapter's
``supports_*`` predicates before using the ``cake_*`` entry points.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

from sglang.kernels.registry import register_kernel
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
    KernelSpec,
)

if TYPE_CHECKING:
    import torch

_MAMBA = "sglang.kernels.cake_kernels.mamba"
_SM100_SM103 = frozenset({CapabilityRequirement.cuda(min_sm=(10, 0), max_sm=(10, 3))})

for _op, _target, _signature, _description in (
    (
        "mamba.ssd_combined",
        f"{_MAMBA}:ssd_combined",
        FormatSignature(
            supported_dtypes=("bfloat16",),
            in_place=True,
            description=(
                "prepared Mamba2 SSD combined prefill runner (chunk 128, headdim 64, "
                "dstate 128, seqlen % 128 == 0): BF16 x/B/C/D/z, BF16 or FP16 state, "
                "caller-owned out [B,nheads,64,nchunks,128], selective checkpoints"
            ),
        ),
        "Cake SSD combined prefill runner (flashinfer.mamba.SSDCombined, "
        "backend='cake') distributed by FlashInfer.",
    ),
    (
        "mamba.ssd_combined_fwd",
        f"{_MAMBA}:ssd_combined_fwd",
        FormatSignature(
            supported_dtypes=("bfloat16",),
            in_place=True,
            description=(
                "functional Mamba2 SSD combined prefill (Cake-only, runner cached per "
                "device/stream/config); returns token-major [B,S,nheads,64] + final "
                "states"
            ),
        ),
        "Cake SSD combined prefill (flashinfer.mamba.ssd_combined_fwd) "
        "distributed by FlashInfer.",
    ),
    (
        "mamba.selective_state_update",
        f"{_MAMBA}:selective_state_update",
        FormatSignature(
            supported_dtypes=("bfloat16", "float32"),
            in_place=True,
            description=(
                "Mamba selective state update on a 4-D state pool (BF16 or FP32) with "
                "BF16 x/B/C and per-head broadcast FP32 dt/A/D/dt_bias; promoted "
                "T=1 / MTP rows run Cake, others fall back inside FlashInfer"
            ),
        ),
        "Cake selective state update (flashinfer.mamba.selective_state_update, "
        "backend='cake') distributed by FlashInfer.",
    ),
):
    register_kernel(
        KernelSpec(
            op=_op,
            backend=KernelBackend.FLASHINFER,
            target=_target,
            capabilities=_SM100_SM103,
            format_signature=_signature,
            description=_description,
        )
    )
del _op, _target, _signature, _description


def cake_ssd_combined(
    chunk_size: int,
    nheads: int,
    headdim: int,
    dstate: int,
    ngroups: int,
    *,
    io_dtype: Optional[torch.dtype] = None,
    state_dtype: Optional[torch.dtype] = None,
    has_d: bool = True,
    d_has_hdim: bool = False,
    has_initial_states: bool = False,
    has_varlen: bool = False,
    has_z: bool = False,
    seq_idx_dtype: Optional[torch.dtype] = None,
):
    """Explicit Cake entry point (prepared runner); gate on ``supports_ssd_combined``."""
    return get_kernel("mamba.ssd_combined", KernelBackend.FLASHINFER)(
        chunk_size,
        nheads,
        headdim,
        dstate,
        ngroups,
        io_dtype=io_dtype,
        state_dtype=state_dtype,
        has_d=has_d,
        d_has_hdim=d_has_hdim,
        has_initial_states=has_initial_states,
        has_varlen=has_varlen,
        has_z=has_z,
        seq_idx_dtype=seq_idx_dtype,
    )


def cake_ssd_combined_fwd(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    dt_limit: Tuple[float, float] = (0.0, float("inf")),
    initial_states: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    chunk_offsets: Optional[torch.Tensor] = None,
    seq_chunk_cumsum: Optional[torch.Tensor] = None,
    update_seq_chunk_cumsum: bool = False,
    checkpoint_token_indices: Optional[torch.Tensor] = None,
    checkpoint_state_slots: Optional[torch.Tensor] = None,
    checkpoint_states: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    return_final_states: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Explicit Cake entry point; gate on ``supports_ssd_combined``."""
    return get_kernel("mamba.ssd_combined_fwd", KernelBackend.FLASHINFER)(
        x,
        dt,
        A,
        B,
        C,
        D=D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
        initial_states=initial_states,
        seq_idx=seq_idx,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        seq_chunk_cumsum=seq_chunk_cumsum,
        update_seq_chunk_cumsum=update_seq_chunk_cumsum,
        checkpoint_token_indices=checkpoint_token_indices,
        checkpoint_state_slots=checkpoint_state_slots,
        checkpoint_states=checkpoint_states,
        out=out,
        return_final_states=return_final_states,
    )


def cake_selective_state_update(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    state_batch_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = -1,
    state_scale: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    disable_state_update: bool = False,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    intermediate_state_scales: Optional[torch.Tensor] = None,
    rand_seed: Optional[torch.Tensor] = None,
    philox_rounds: int = 10,
    cache_steps: int = 0,
    algorithm: str = "auto",
    dst_state_batch_indices: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Explicit Cake entry point; gate on ``supports_selective_state_update``."""
    return get_kernel("mamba.selective_state_update", KernelBackend.FLASHINFER)(
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        state_batch_indices=state_batch_indices,
        pad_slot_id=pad_slot_id,
        state_scale=state_scale,
        out=out,
        disable_state_update=disable_state_update,
        intermediate_states_buffer=intermediate_states_buffer,
        intermediate_state_indices=intermediate_state_indices,
        intermediate_state_scales=intermediate_state_scales,
        rand_seed=rand_seed,
        philox_rounds=philox_rounds,
        cache_steps=cache_steps,
        algorithm=algorithm,
        dst_state_batch_indices=dst_state_batch_indices,
        cu_seqlens=cu_seqlens,
        num_accepted_tokens=num_accepted_tokens,
    )


__all__ = [
    "cake_ssd_combined",
    "cake_ssd_combined_fwd",
    "cake_selective_state_update",
]
