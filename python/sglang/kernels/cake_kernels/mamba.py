"""Cake Mamba2 SSD combined prefill and selective state update via FlashInfer.

FlashInfer entries (contract at FlashInfer ``58171ea83f32`` -- the
``flashinfer/mamba`` sources are unchanged since ``2a57c19bace5``, the
exact-scan revision (PR #6051) of the ``46340689a5ab`` contract: one exact-scan kernel
family with FP16 delta, FP32 cumsum, any ``seqlen``, BF16 / FP16 / FP32
state, token-major ``out``, varlen without ``initial_states``):

* ``flashinfer.mamba.SSDCombined(..., backend="cake")`` (inventory E1-31) ->
  ``flashinfer.mamba.cake_ssd_combined.CakeSSDCombined`` (E1-33), and the
  functional ``flashinfer.mamba.ssd_combined_fwd`` (E1-32, Cake-only, runner
  cached per device / stream / config). Locked domain: ``chunk_size=128``,
  ``headdim=64``, ``dstate=128``, any ``seqlen > 0`` (the last physical chunk
  may be partial); BF16 ``x [B, S, nheads, 64]``, ``B``/``C [B, S, ngroups,
  128]`` (``nheads % ngroups == 0``), ``dt [B, S, nheads]`` BF16 or FP32, FP32
  ``A [nheads]``, optional BF16 ``D [nheads]`` or ``[nheads, 64]``, ``z`` like
  ``x``, ``dt_bias [nheads]`` BF16 or FP32; state dtype BF16, FP16 or FP32
  (``initial_states [num_seqs, nheads, 64, 128]``); varlen needs ``seq_idx``
  (int32/int64 ``[B, S]``, non-decreasing, every id in ``[0, num_seqs)``
  owning a token) and int32 ``chunk_indices`` / ``chunk_offsets``, and takes
  the packed sequence count from ``initial_states``, ``seq_chunk_cumsum`` or
  the plain int ``num_seqs`` (at least one; they must agree -- ``num_seqs``
  costs no device buffer, the preprocess derives ``seq_chunk_cumsum`` itself).
  Only backend that writes selective ``checkpoint_states``
  (``checkpoint_token_indices`` + ``checkpoint_state_slots``, contiguous
  int32 ``[num_seqs]``, all three together; packed-varlen boundaries are
  absolute exclusive token positions that must be logical chunk ends).
  ``out`` is caller-owned contiguous token-major BF16 ``[B, S, nheads, 64]``
  (the engine's own buffer, written directly by the kernels; no chunked
  layout, no copy); ``run`` returns it plus the final states ``[num_seqs,
  nheads, 64, 128]`` in the state dtype. The runner materializes strided
  pointer-indexed inputs into graph-stable storage; ``x``/``B``/``C`` keep
  their strides (TMA descriptors). **State dtype selection**: the functional
  entry infers it from ``initial_states`` / ``checkpoint_states`` and falls
  back to BF16 when neither is given; the prepared runner takes it as a
  constructor argument. :func:`ssd_combined_fwd` therefore accepts
  ``state_dtype=`` and serves an explicit dtype through a cached prepared
  runner, so a varlen call without a prefix still returns final states in
  the engine's pool dtype (``--mamba-ssm-dtype``, FP32 by default).
* ``flashinfer.mamba.selective_state_update(..., backend="cake")`` (E1-29) ->
  ``flashinfer.jit.mamba.cake_selective_state_update.try_cake_selective_state_update``
  and the thin ``flashinfer.mamba.cake_selective_state_update`` (E1-30).
  Source-built with nvcc on first use. **Outside the promoted rows FlashInfer
  silently runs its own kernel**, so :func:`supports_selective_state_update`
  is the only way for SGLang to know Cake ran. Promoted rows (common gate:
  4-D state BF16/FP32, BF16 ``x``/``B``/``C``, ``dt_bias`` and 1-D
  ``state_batch_indices`` present, per-head broadcast ``dt``/``A``/``D``/
  ``dt_bias`` (stride 0 on the dim / dstate axes), one coefficient ABI --
  "canonical" FP32 ``dt``/``D``/``dt_bias`` with int64 index tables, or
  "raw" BF16 ``dt``/``D``/``dt_bias`` with int32 tables (the SGLang engine
  storage) -- and no ``state_scale`` / ``rand_seed`` / ``cu_seqlens`` /
  ``num_accepted_tokens``):

  - T=1 (``x [B, nheads, 128]``, ``cache_steps == 0``, ``(dim, dstate) =
    (128, 128)``, canonical ABI only): BF16 state -> ``stp_bf16_*``; FP32
    state -> ``stp_fp32_identity`` (no ``z``, no softplus, no
    ``disable_state_update``, destination is the source index tensor,
    ``B * nheads >= 8 * SMs``).
  - T=1 headdim-64 decode (``x [B, nheads, 64]``, ``(dim, dstate) = (64,
    128)``, BF16 or FP32 state, no intermediate buffer, both ABIs;
    Nemotron-H / granite-4.0-h): dense ``x``/``B``/``C`` rows at any batch
    stride (the fused-projection views), contiguous state and index tables.
    BF16 state with an even head count and even heads per group at
    ``B * nheads / 2 >= 2 * SMs`` -> paired TMA programs ``stp_paired_*``
    (unit-head-stride ``dt``/``A``/``D``/``dt_bias``, 16-byte aligned
    ``B``/``C`` rows, batch strides multiples of 8 elements; no row-owner
    fallback when these fail), otherwise the row-owner tiles
    ``stp_hd64_rows_*`` (BF16 only below ``8 * SMs`` batch-heads; FP32 at
    any size).
  - MTP ``x [B, T, nheads, 128]`` BF16 state, T in {1, 2}, canonical ABI, no
    ``z`` / destination / intermediate buffer / ``disable_state_update`` ->
    ``mtp_short``.
  - MTP BF16 state ``(dim, dstate, T) = (64, 128, 6)``, softplus,
    ``disable_state_update``, intermediate buffer (state dtype, ``[*, 6,
    nheads, 64, 128]``) + indices, no ``z`` / destination: ``B < 32`` with
    ``(10 * SMs) // (B * nheads) >= 4`` -> ``mtp_cache_c4_t6`` on either ABI
    (dense 16-byte aligned projection rows, unit-head-stride coefficients);
    ``B >= 32`` with ``algorithm="horizontal"`` and the canonical ABI ->
    ``mtp_horizontal``. Batches between the two windows fall back.
  - FP32 state ``(nheads, dim, dstate, ngroups) = (16, 64, 128, 1)``,
    ``1 <= T <= 8``, ``algorithm="simple"``, softplus, canonical ABI, 2-D
    destination table ``[B, T]`` -> ``dynamic`` (device-side table; no host
    read, graph-capturable).

Both families are built for sm_100a / sm_103a only.

Admission rules absorbed from PR #35444: strict opt-in Cake SSD prefill
(selecting Cake never falls back), compact checkpoints emitted in the main
pass (``checkpoint_*``), ``dt_limit[0] == 0`` for the Nemotron-H route, a
caller-owned ``out`` for CUDA-graph stability; decode stays on
``selective_state_update``. The C256 -> C128 metadata projection and the
``--mamba-backend`` plumbing are call-site work, not part of this adapter.

Not supported here (keep the existing SGLang path): SSD ``headdim != 64`` /
``dstate != 128`` / other chunk sizes, FP16 I/O, SSU stochastic rounding
(``rand_seed``), ``state_scale``, varlen SSU (``cu_seqlens``),
``retrieve_parent_token`` tree verify, sm_90a / sm_120a / sm_121a.
"""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

from sglang.kernels.cake_kernels._support import (
    BLACKWELL_DATACENTER,
    cuda_tensor_on,
    flashinfer_module_available,
)

if TYPE_CHECKING:
    import torch

FI_MODULE = "flashinfer.mamba"
FI_SSD_MODULE = "flashinfer.mamba.cake_ssd_combined"
FI_SSU_JIT_MODULE = "flashinfer.jit.mamba.cake_selective_state_update"
ARCHS = BLACKWELL_DATACENTER
SSD_CHUNK_SIZE = 128
SSD_HEADDIM = 64
SSD_DSTATE = 128
SSU_DYNAMIC_MAX_TOKEN_STEPS = 8


def _contiguous(tensor, dtype, device, ndim=None) -> bool:
    return (
        tensor is not None
        and tensor.is_cuda
        and tensor.device == device
        and tensor.dtype == dtype
        and tensor.is_contiguous()
        and (ndim is None or tensor.ndim == ndim)
    )


def supports_ssd_combined(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    dt_limit: Tuple[float, float] = (0.0, float("inf")),
    initial_states: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    chunk_offsets: Optional[torch.Tensor] = None,
    seq_chunk_cumsum: Optional[torch.Tensor] = None,
    num_seqs: Optional[int] = None,
    checkpoint_token_indices: Optional[torch.Tensor] = None,
    checkpoint_state_slots: Optional[torch.Tensor] = None,
    checkpoint_states: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    chunk_size: int = SSD_CHUNK_SIZE,
    state_dtype: Optional[torch.dtype] = None,
) -> bool:
    """Admission mirroring ``CakeSSDCombined`` host validation; never raises.

    The same predicate serves the prepared runner (whose constructor flags
    ``has_d`` / ``has_z`` / ``has_initial_states`` / ``has_varlen`` must match
    the runtime presence of ``D`` / ``z`` / ``initial_states`` / ``seq_idx``)
    and the functional ``ssd_combined_fwd``.

    ``state_dtype`` names the state dtype explicitly (see
    :func:`ssd_combined_fwd`); it must be BF16 / FP16 / FP32 and agree with
    ``initial_states`` and ``checkpoint_states``. Without it the dtype is the
    one FlashInfer's functional entry infers (``initial_states``, else
    ``checkpoint_states``, else BF16).
    """
    import torch

    state_dtypes = (torch.bfloat16, torch.float16, torch.float32)
    if not (
        flashinfer_module_available(FI_MODULE, FI_SSD_MODULE)
        and cuda_tensor_on(x, ARCHS)
        and chunk_size == SSD_CHUNK_SIZE
        and (state_dtype is None or state_dtype in state_dtypes)
        and x.ndim == 4
        and B.ndim == 4
        and len(dt_limit) == 2
        and math.isfinite(float(dt_limit[0]))
    ):
        return False
    device = x.device
    batch, seqlen, nheads, headdim = x.shape
    ngroups = B.shape[2]
    if not (
        batch > 0
        and seqlen > 0
        and headdim == SSD_HEADDIM
        and nheads > 0
        and ngroups > 0
        and nheads % ngroups == 0
    ):
        return False
    for tensor in (x, B, C):
        if not (
            tensor.is_cuda
            and tensor.device == device
            and tensor.dtype == torch.bfloat16
        ):
            return False
    if (
        tuple(B.shape) != (batch, seqlen, ngroups, SSD_DSTATE)
        or tuple(C.shape) != tuple(B.shape)
        or not (dt.is_cuda and dt.device == device)
        or dt.dtype not in (torch.bfloat16, torch.float32)
        or tuple(dt.shape) != (batch, seqlen, nheads)
        or not (A.is_cuda and A.device == device and A.dtype == torch.float32)
        or tuple(A.shape) != (nheads,)
    ):
        return False
    if D is not None and not (
        D.is_cuda
        and D.device == device
        and D.dtype == torch.bfloat16
        and tuple(D.shape) in ((nheads,), (nheads, SSD_HEADDIM))
    ):
        return False
    if z is not None and not (
        z.is_cuda
        and z.device == device
        and z.dtype == torch.bfloat16
        and tuple(z.shape) == tuple(x.shape)
    ):
        return False
    if dt_bias is not None and not (
        dt_bias.is_cuda
        and dt_bias.device == device
        and dt_bias.dtype in (torch.bfloat16, torch.float32)
        and tuple(dt_bias.shape) == (nheads,)
    ):
        return False
    varlen = seq_idx is not None
    metadata = (seq_idx, chunk_indices, chunk_offsets)
    if varlen:
        if any(t is None for t in metadata):
            return False
        if not (
            seq_idx.is_cuda
            and seq_idx.device == device
            and seq_idx.dtype in (torch.int32, torch.int64)
            and tuple(seq_idx.shape) == (batch, seqlen)
            and _contiguous(chunk_indices, torch.int32, device, ndim=1)
            and _contiguous(chunk_offsets, torch.int32, device, ndim=1)
            and tuple(chunk_indices.shape) == tuple(chunk_offsets.shape)
        ):
            return False
        # The packed sequence count comes from whichever of initial_states /
        # seq_chunk_cumsum / num_seqs the caller gives; they must agree.
        counts = set()
        if initial_states is not None:
            counts.add(int(initial_states.shape[0]))
        if seq_chunk_cumsum is not None:
            counts.add(int(seq_chunk_cumsum.numel()) - 1)
        if num_seqs is not None:
            counts.add(int(num_seqs))
        if len(counts) != 1:
            return False
        num_sequences = counts.pop()
        if num_sequences < 1:
            return False
    elif (
        any(t is not None for t in metadata)
        or seq_chunk_cumsum is not None
        or num_seqs is not None
    ):
        return False
    else:
        num_sequences = batch
    if initial_states is not None:
        if not (
            initial_states.is_cuda
            and initial_states.device == device
            and initial_states.dtype in state_dtypes
            and tuple(initial_states.shape)
            == (num_sequences, nheads, SSD_HEADDIM, SSD_DSTATE)
        ):
            return False
        inferred = initial_states.dtype
    else:
        inferred = (
            checkpoint_states.dtype if checkpoint_states is not None else torch.bfloat16
        )
        if inferred not in state_dtypes:
            return False
    if state_dtype is None:
        state_dtype = inferred
    elif (initial_states is not None or checkpoint_states is not None) and (
        state_dtype != inferred
    ):
        # FlashInfer: "initial_states dtype must match state_dtype" /
        # "checkpoint_states must be ... with state dtype".
        return False
    if seq_chunk_cumsum is not None and not (
        seq_chunk_cumsum.is_cuda
        and seq_chunk_cumsum.device == device
        and seq_chunk_cumsum.dtype == torch.int32
        and tuple(seq_chunk_cumsum.shape) == (num_sequences + 1,)
    ):
        return False
    checkpoint_args = (
        checkpoint_token_indices,
        checkpoint_state_slots,
        checkpoint_states,
    )
    if any(t is not None for t in checkpoint_args):
        if any(t is None for t in checkpoint_args):
            return False
        if not (
            _contiguous(checkpoint_token_indices, torch.int32, device, ndim=1)
            and checkpoint_token_indices.numel() == num_sequences
            and _contiguous(checkpoint_state_slots, torch.int32, device, ndim=1)
            and checkpoint_state_slots.numel() == num_sequences
            and _contiguous(checkpoint_states, state_dtype, device, ndim=4)
            and tuple(checkpoint_states.shape[1:]) == (nheads, SSD_HEADDIM, SSD_DSTATE)
        ):
            return False
    if out is not None and not (
        _contiguous(out, torch.bfloat16, device, ndim=4)
        and tuple(out.shape) == tuple(x.shape)
    ):
        return False
    return True


def ssd_combined(
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
    """Construct ``flashinfer.mamba.SSDCombined(..., backend="cake")``.

    Returns the prepared runner; call ``.run(x, dt, A, B, C, D=, z=, dt_bias=,
    dt_softplus=, dt_limit=, initial_states=, seq_idx=, chunk_indices=,
    chunk_offsets=, seq_chunk_cumsum=, update_seq_chunk_cumsum=, num_seqs=,
    checkpoint_token_indices=, checkpoint_state_slots=, checkpoint_states=,
    out=, return_final_states=)`` per batch. One runner per stream (its
    workspaces are mutable). Dtype defaults: BF16 I/O and state, int64
    ``seq_idx``.
    """
    import torch
    from flashinfer.mamba import SSDCombined

    return SSDCombined(
        chunk_size,
        nheads,
        headdim,
        dstate,
        ngroups,
        io_dtype=torch.bfloat16 if io_dtype is None else io_dtype,
        state_dtype=torch.bfloat16 if state_dtype is None else state_dtype,
        has_d=has_d,
        d_has_hdim=d_has_hdim,
        has_initial_states=has_initial_states,
        has_varlen=has_varlen,
        has_z=has_z,
        seq_idx_dtype=torch.int64 if seq_idx_dtype is None else seq_idx_dtype,
        backend="cake",
    )


def ssd_combined_fwd(
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
    num_seqs: Optional[int] = None,
    update_seq_chunk_cumsum: bool = False,
    checkpoint_token_indices: Optional[torch.Tensor] = None,
    checkpoint_state_slots: Optional[torch.Tensor] = None,
    checkpoint_states: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    return_final_states: bool = True,
    state_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run the Cake SSD combined forward.

    Returns ``(token-major output [B, S, nheads, 64], final_states or None)``;
    ``out`` is written in place and returned. A varlen call without
    ``initial_states`` names its packed sequence count through ``num_seqs``.

    ``state_dtype=None`` forwards to ``flashinfer.mamba.ssd_combined_fwd``,
    whose state dtype is inferred from ``initial_states`` /
    ``checkpoint_states`` (BF16 when neither is given). An explicit
    ``state_dtype`` (the engine's SSM pool dtype) runs the prepared
    ``flashinfer.mamba.SSDCombined(..., backend="cake", state_dtype=...)``
    runner instead -- cached per device / stream / configuration exactly like
    FlashInfer's functional entry -- so the final states come back in that
    dtype even when the scan starts from zero state.
    """
    if state_dtype is not None:
        import torch

        device_index = x.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        with torch.cuda.device(device_index):
            runner = _cached_ssd_runner(
                device_index,
                int(torch.cuda.current_stream(device_index).cuda_stream),
                int(x.shape[2]),
                int(B.shape[2]),
                state_dtype,
                D is not None,
                D is not None and D.ndim == 2,
                initial_states is not None,
                seq_idx is not None,
                z is not None,
                seq_idx.dtype if seq_idx is not None else torch.int64,
            )
        return runner.run(
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
            num_seqs=num_seqs,
            checkpoint_token_indices=checkpoint_token_indices,
            checkpoint_state_slots=checkpoint_state_slots,
            checkpoint_states=checkpoint_states,
            out=out,
            return_final_states=return_final_states,
        )

    from flashinfer.mamba import ssd_combined_fwd as fi_ssd_combined_fwd

    return fi_ssd_combined_fwd(
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
        num_seqs=num_seqs,
        checkpoint_token_indices=checkpoint_token_indices,
        checkpoint_state_slots=checkpoint_state_slots,
        checkpoint_states=checkpoint_states,
        out=out,
        return_final_states=return_final_states,
    )


@functools.lru_cache(maxsize=64)
def _cached_ssd_runner(
    device_index: int,
    cuda_stream: int,
    nheads: int,
    ngroups: int,
    state_dtype: torch.dtype,
    has_d: bool,
    d_has_hdim: bool,
    has_initial_states: bool,
    has_varlen: bool,
    has_z: bool,
    seq_idx_dtype: torch.dtype,
):
    """Prepared Cake runner per device / stream / configuration.

    Runner workspaces are mutable, so the stream is part of the key (one
    runner per concurrently active stream); the cache is bounded like
    FlashInfer's own, an evicted runner is rebuilt on its next call. Called
    under ``torch.cuda.device(device_index)``: ``SSDCombined`` probes the
    current device's capability at construction.
    """
    return ssd_combined(
        SSD_CHUNK_SIZE,
        nheads,
        SSD_HEADDIM,
        SSD_DSTATE,
        ngroups,
        state_dtype=state_dtype,
        has_d=has_d,
        d_has_hdim=d_has_hdim,
        has_initial_states=has_initial_states,
        has_varlen=has_varlen,
        has_z=has_z,
        seq_idx_dtype=seq_idx_dtype,
    )


_HD64_PAIRED_MIN_BLOCKS_PER_SM = 2
_HD64_ROWS_MAX_BATCH_HEADS_PER_SM = 8


def _coefficient_abi(dt, D, dt_bias, indices, dst_indices, buffer_indices):
    """``"canonical"`` (FP32 coefficients, int64 tables), ``"raw"`` (BF16
    coefficients, int32 tables: the SGLang engine storage) or ``None``."""
    import torch

    for coefficient, index in (
        (torch.float32, torch.int64),
        (torch.bfloat16, torch.int32),
    ):
        if (
            dt.dtype == coefficient
            and D.dtype == coefficient
            and dt_bias.dtype == coefficient
            and indices.dtype == index
            and (dst_indices is None or dst_indices.dtype == index)
            and (buffer_indices is None or buffer_indices.dtype == index)
        ):
            return "canonical" if coefficient == torch.float32 else "raw"
    return None


def _dense_rows(tensor, batch: int, rows: int, row: int) -> bool:
    """``(batch, rows, row)`` view with dense rows; the batch stride may pad."""
    return (
        tuple(tensor.shape) == (batch, rows, row)
        and tensor.stride(2) == 1
        and tensor.stride(1) == row
        and (batch == 1 or tensor.stride(0) >= rows * row)
    )


def _supports_hd64_decode(
    state,
    x,
    dt,
    A,
    B,
    C,
    D,
    dt_bias,
    z,
    indices,
    dst_indices,
    *,
    cache_steps,
    nheads,
    ngroups,
    device,
) -> bool:
    """Mirror of FlashInfer's ``_plan_stp_hd64`` for the headdim-64 single-token
    row: the paired TMA programs at saturating BF16 work (no row-owner fallback
    when their stricter layout fails), the row-owner tiles otherwise."""
    import torch

    batch = x.shape[0]
    if not (
        cache_steps == 0
        and indices.ndim == 1
        and tuple(indices.shape) == (batch,)
        and indices.is_contiguous()
        and (
            dst_indices is None
            or (
                dst_indices.ndim == 1
                and tuple(dst_indices.shape) == (batch,)
                and dst_indices.is_contiguous()
            )
        )
        and _dense_rows(x, batch, nheads, 64)
        and _dense_rows(B, batch, ngroups, 128)
        and _dense_rows(C, batch, ngroups, 128)
        and state.is_contiguous()
        and (
            z is None
            or (_dense_rows(z, batch, nheads, 64) and z.stride(0) == x.stride(0))
        )
    ):
        return False
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    if (
        state.dtype == torch.bfloat16
        and nheads % 2 == 0
        and (nheads // ngroups) % 2 == 0
        and batch * (nheads // 2) >= _HD64_PAIRED_MIN_BLOCKS_PER_SM * num_sms
    ):
        return (
            dt.stride(1) == 1
            and A.stride(0) == 1
            and D.stride(0) == 1
            and dt_bias.stride(0) == 1
            and (B.data_ptr() | C.data_ptr()) & 15 == 0
            and (B.stride(0) | C.stride(0)) & 7 == 0
        )
    return (
        state.dtype == torch.float32
        or batch * nheads < _HD64_ROWS_MAX_BATCH_HEADS_PER_SM * num_sms
    )


def _dense_step_rows(tensor, batch: int, steps: int, rows: int, row: int) -> bool:
    """``(batch, steps, rows, row)`` view with dense rows and 16-byte aligned
    batch / step strides (the ``mtp_cache_c4_t6`` projection layout)."""
    return (
        tuple(tensor.shape) == (batch, steps, rows, row)
        and tensor.stride(3) == 1
        and tensor.stride(2) == row
        and (tensor.stride(0) | tensor.stride(1)) & 7 == 0
    )


def _per_head_broadcast(tensor, nheads, trailing: int) -> bool:
    """``[nheads, ...]`` view whose trailing axes are stride-0 broadcasts."""
    return (
        tensor is not None
        and tensor.ndim == 1 + trailing
        and tensor.shape[0] == nheads
        and all(tensor.stride(i) == 0 for i in range(1, 1 + trailing))
    )


def supports_selective_state_update(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor],
    *,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    state_batch_indices: Optional[torch.Tensor] = None,
    dst_state_batch_indices: Optional[torch.Tensor] = None,
    disable_state_update: bool = False,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    state_scale: Optional[torch.Tensor] = None,
    intermediate_state_scales: Optional[torch.Tensor] = None,
    rand_seed: Optional[torch.Tensor] = None,
    cache_steps: int = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    algorithm: str = "auto",
    retrieve_parent_token: Optional[torch.Tensor] = None,
    pad_slot_id: int = -1,
) -> bool:
    """``True`` only when FlashInfer's Cake SSU would run a promoted program.

    ``pad_slot_id`` (the slot value of CUDA-graph padding rows) is accepted for
    every row; the programs skip such rows on the device.

    Encodes the promoted rows listed in the module docstring (mirror of
    FlashInfer's ``plan_route``; the ``out`` tensor is not an input here, its
    layout is validated by the binding at launch). ``backend="cake"`` falls
    back silently otherwise, so a ``False`` here means "Cake will not run",
    not "the call would fail".
    """
    import torch

    if not (
        flashinfer_module_available(FI_MODULE, FI_SSU_JIT_MODULE)
        and cuda_tensor_on(state, ARCHS)
        and D is not None
        and dt_bias is not None
        and state_batch_indices is not None
        and state_scale is None
        and intermediate_state_scales is None
        and rand_seed is None
        and cu_seqlens is None
        and num_accepted_tokens is None
        and retrieve_parent_token is None
        and state.ndim == 4
        and state.dtype in (torch.bfloat16, torch.float32)
        and x.dtype == torch.bfloat16
        and B.dtype == torch.bfloat16
        and C.dtype == torch.bfloat16
        and A.dtype == torch.float32
    ):
        return False
    abi = _coefficient_abi(
        dt,
        D,
        dt_bias,
        state_batch_indices,
        dst_state_batch_indices,
        intermediate_state_indices,
    )
    if abi is None:
        return False
    device = state.device
    if any(
        t is not None and (not t.is_cuda or t.device != device)
        for t in (
            x,
            dt,
            A,
            B,
            C,
            D,
            z,
            dt_bias,
            state_batch_indices,
            dst_state_batch_indices,
            intermediate_states_buffer,
            intermediate_state_indices,
        )
    ):
        return False
    _, nheads, dim, dstate = state.shape
    ngroups = B.shape[-2]
    if ngroups <= 0 or nheads % ngroups or B.shape[-1] != dstate or C.shape != B.shape:
        return False
    if not (
        _per_head_broadcast(A, nheads, 2)
        and _per_head_broadcast(D, nheads, 1)
        and _per_head_broadcast(dt_bias, nheads, 1)
        and dt.ndim == x.ndim
        and dt.stride(-1) == 0
        and tuple(dt.shape) == tuple(x.shape)
    ):
        return False
    batch = x.shape[0]
    if x.ndim == 3:
        if (dim, dstate) == (64, 128):
            return intermediate_states_buffer is None and _supports_hd64_decode(
                state,
                x,
                dt,
                A,
                B,
                C,
                D,
                dt_bias,
                z,
                state_batch_indices,
                dst_state_batch_indices,
                cache_steps=cache_steps,
                nheads=nheads,
                ngroups=ngroups,
                device=device,
            )
        if abi == "raw":
            return False
        if not (
            cache_steps == 0
            and tuple(x.shape) == (batch, nheads, dim)
            and tuple(B.shape) == (batch, ngroups, dstate)
            and (dim, dstate) == (128, 128)
            and state_batch_indices.ndim == 1
            and (dst_state_batch_indices is None or dst_state_batch_indices.ndim == 1)
        ):
            return False
        if state.dtype == torch.bfloat16:
            return True
        return (
            z is None
            and not dt_softplus
            and not disable_state_update
            and (
                dst_state_batch_indices is None
                or dst_state_batch_indices.data_ptr() == state_batch_indices.data_ptr()
            )
            and batch * nheads
            >= 8 * torch.cuda.get_device_properties(device).multi_processor_count
        )
    if x.ndim != 4 or state_batch_indices.ndim != 1:
        return False
    token_steps = x.shape[1]
    if (
        tuple(x.shape) != (batch, token_steps, nheads, dim)
        or tuple(B.shape) != (batch, token_steps, ngroups, dstate)
        or tuple(state_batch_indices.shape) != (batch,)
    ):
        return False
    if (
        abi == "canonical"
        and state.dtype == torch.bfloat16
        and (dim, dstate) == (128, 128)
        and token_steps in (1, 2)
        and z is None
        and dst_state_batch_indices is None
        and intermediate_states_buffer is None
        and not disable_state_update
    ):
        return True
    if (
        state.dtype == torch.bfloat16
        and (dim, dstate, token_steps) == (64, 128, 6)
        and z is None
        and dst_state_batch_indices is None
        and dt_softplus
        and disable_state_update
        and intermediate_states_buffer is not None
        and intermediate_state_indices is not None
        and tuple(intermediate_state_indices.shape) == (batch,)
        and intermediate_states_buffer.ndim == 5
        and tuple(intermediate_states_buffer.shape[1:]) == (6, nheads, 64, 128)
    ):
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        if batch < 32 and (num_sms * 10) // (batch * nheads) >= 4:
            # ``mtp_cache_c4_t6`` (either ABI): dense 16-byte aligned
            # projection rows, unit-head-stride coefficients, contiguous
            # state / buffer / tables.
            return (
                intermediate_states_buffer.dtype == state.dtype
                and _dense_step_rows(x, batch, 6, nheads, 64)
                and _dense_step_rows(B, batch, 6, ngroups, 128)
                and _dense_step_rows(C, batch, 6, ngroups, 128)
                and (x.data_ptr() | B.data_ptr() | C.data_ptr()) & 15 == 0
                and dt.stride(2) == 1
                and A.stride(0) == 1
                and D.stride(0) == 1
                and dt_bias.stride(0) == 1
                and state.is_contiguous()
                and intermediate_states_buffer.is_contiguous()
                and state_batch_indices.is_contiguous()
                and intermediate_state_indices.is_contiguous()
            )
        # ``mtp_horizontal``: canonical ABI, BF16 intermediate buffer.
        return (
            batch >= 32
            and algorithm == "horizontal"
            and abi == "canonical"
            and intermediate_states_buffer.dtype == torch.bfloat16
        )
    if (
        abi == "canonical"
        and state.dtype == torch.float32
        and (nheads, dim, dstate, ngroups) == (16, 64, 128, 1)
        and 1 <= token_steps <= SSU_DYNAMIC_MAX_TOKEN_STEPS
        and algorithm == "simple"
        and dt_softplus
        and z is None
        and dst_state_batch_indices is not None
        and tuple(dst_state_batch_indices.shape) == (batch, token_steps)
        and intermediate_states_buffer is None
        and not disable_state_update
    ):
        # ``dynamic``: the destination table is read on the device, so the
        # row is graph-capturable.
        return True
    return False


def selective_state_update(
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
    """Forward to ``flashinfer.mamba.selective_state_update(backend="cake")``.

    ``state`` (and ``intermediate_states_buffer``) are updated in place; the
    result is written to ``out`` when given and returned. FlashInfer falls
    back to its own kernel outside the promoted rows -- gate on
    :func:`supports_selective_state_update` when Cake execution matters.
    """
    from flashinfer.mamba import selective_state_update as fi_selective_state_update

    return fi_selective_state_update(
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
        backend="cake",
    )


__all__: Sequence[str] = (
    "supports_ssd_combined",
    "ssd_combined",
    "ssd_combined_fwd",
    "supports_selective_state_update",
    "selective_state_update",
)
