"""DeepSeek-V4.1 Engram: the table gather and the SP / DP / TP seam kernels.
See jit/csrc/deepseek_v4/engram_fusion.cuh; every mode gathers token-sharded,
each rank reading the symmetric table shards in place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi import Module

K = 24 * 256
SF_TILE_BYTES = 512
SF_K_TILES = K // 32 // 4


@cache_once
def _jit_engram_module(world_size: int) -> Module:
    cls = f"EngramFusion<{world_size}>"
    return load_jit(
        "engram_fusion",
        f"world{world_size}",
        cuda_files=["deepseek_v4/engram_fusion.cuh"],
        cuda_wrappers=[
            ("gather_mxfp8", f"{cls}::gather_mxfp8"),
            ("gather_bf16", f"{cls}::gather_bf16"),
            ("sp_gate_mhc_combine_norm", f"{cls}::sp_gate_mhc_combine_norm"),
            ("tp_gate_mhc_combine_norm", f"{cls}::tp_gate_mhc_combine_norm"),
            ("dp_gate_mhc_combine_norm", f"{cls}::dp_gate_mhc_combine_norm"),
        ],
    )


def sf_bytes(m_pad: int) -> int:
    return m_pad // 128 * SF_K_TILES * SF_TILE_BYTES


def engram_gather_mxfp8(
    world_size: int,
    ids: torch.Tensor,
    out_a: torch.Tensor,
    out_sf: torch.Tensor,
    w_ptrs: Sequence[int],
    s_ptrs: Sequence[int],
    shard_starts: Sequence[int],
) -> None:
    """ids [M, 24] int32 (a row-strided layer slice is fine); out_a [m_pad, 6144] uint8; out_sf
    [sf_bytes(m_pad)] uint8. ``w_ptrs`` / ``s_ptrs`` are every rank's table shard
    bases and ``shard_starts`` each shard's first global row; the shards must cut
    on hash-column boundaries (``shard_range``), 24 % world_size == 0."""
    _jit_engram_module(world_size).gather_mxfp8(
        ids, out_a, out_sf, list(w_ptrs), list(s_ptrs), list(shard_starts)
    )


def engram_gather_bf16(
    world_size: int,
    ids: torch.Tensor,
    out: torch.Tensor,
    w_ptrs: Sequence[int],
    s_ptrs: Sequence[int],
    shard_starts: Sequence[int],
) -> None:
    """The gather dequantized on the way out: ids [M, 24] int32 -> out [M, 6144]
    bf16, exact (e8m0 scales are powers of two). For a wkv without an MXFP8
    view; same shard contract as ``engram_gather_mxfp8``."""
    _jit_engram_module(world_size).gather_bf16(
        ids, out, list(w_ptrs), list(s_ptrs), list(shard_starts)
    )


def sp_engram_gate_mhc_combine_norm(
    world_size: int,
    comm,
    x_ptrs: Sequence[int],
    residual: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    pre: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    skip: Optional[torch.Tensor] = None,
    row_offset: int,
    total_rows: int,
    eps: float,
    clamp: float,
    rms_eps: float,
) -> None:
    """The whole engram seam in one launch: residual [M, 4, 5120] += gate * value
    in place, then x = RMSNorm(sum_h pre[h] * residual[h]) * norm_weight pushed to
    every rank's symmetric buffer (``x_ptrs``) at this rank's rows. ``pre`` is the
    previous boundary's lagged pre. ``skip`` [M] uint8 keeps marked rows'
    residual (image tokens). Barriers on ``comm``'s pull semaphores; the
    grid is derived from the world size on every rank alike."""
    _jit_engram_module(world_size).sp_gate_mhc_combine_norm(
        comm,
        list(x_ptrs),
        residual,
        kv,
        q_weight,
        k_weight,
        pre,
        norm_weight,
        skip,
        row_offset,
        total_rows,
        eps,
        clamp,
        rms_eps,
    )


def dp_engram_gate_mhc_combine_norm(
    world_size: int,
    residual: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    pre: torch.Tensor,
    norm_weight: torch.Tensor,
    x_out: torch.Tensor,
    *,
    skip: Optional[torch.Tensor] = None,
    eps: float,
    clamp: float,
    rms_eps: float,
) -> None:
    """The DP seam: the SP math on this rank's rows only, purely local -- no
    symmetric buffers, no barriers; x lands in ``x_out`` [M, 5120]. ``skip``
    [M] uint8 keeps marked rows' residual (image tokens). ``world_size`` only
    picks the JIT module the gather already loaded."""
    _jit_engram_module(world_size).dp_gate_mhc_combine_norm(
        residual,
        kv,
        q_weight,
        k_weight,
        pre,
        norm_weight,
        skip,
        x_out,
        eps,
        clamp,
        rms_eps,
    )


def tp_engram_staging_bytes(total_rows: int) -> int:
    """One rank's staging plane: [T, 5120] bf16 values then [T, 4] fp32 gates."""
    return total_rows * (5120 * 2 + 16)


def tp_engram_gate_mhc_combine_norm(
    world_size: int,
    comm,
    staging_ptrs: Sequence[int],
    staging: torch.Tensor,
    residual: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    pre: torch.Tensor,
    norm_weight: torch.Tensor,
    x_out: torch.Tensor,
    *,
    skip: Optional[torch.Tensor] = None,
    eps: float,
    clamp: float,
    rms_eps: float,
) -> None:
    """The TP seam: gate this rank's ragged token share (the boundary's row
    split) and push (value, gates) to every rank's staging plane (barriers on
    ``comm``), then locally update every row of the replicated residual, fold
    ``pre``, norm into ``x_out``. ``skip`` [T] uint8 keeps marked rows'
    residual. ``staging`` must be this rank's entry of ``staging_ptrs``."""
    _jit_engram_module(world_size).tp_gate_mhc_combine_norm(
        comm,
        list(staging_ptrs),
        staging,
        residual,
        kv,
        q_weight,
        k_weight,
        pre,
        norm_weight,
        skip,
        x_out,
        eps,
        clamp,
        rms_eps,
    )
