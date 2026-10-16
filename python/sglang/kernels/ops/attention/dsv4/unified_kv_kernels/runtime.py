"""Runtime glue for the unified_kv backend.

Builds unified_kv-style flat ``kv_indices`` / ``kv_indptr`` from SGLang's already-computed
DSV4 metadata and dispatches the vendored paged decode/prefill kernels.

unified_kv[L] layout (page_size 1, row-major):
  - rows ``[0, swa_pages)``    = SWA ring (``state_slot * win + pos % win``);
  - rows ``[swa_pages, ...)``  = compressed K (``swa_pages + page_index``), where
    SGLang metadata already encodes the compressed slot id:
      HCA (ratio 128): ``c128_page_indices``      (== phys_block, k_per_block=1)
      CSA (ratio   4): ``c4_sparse_page_indices``  (== phys_block*32 + slot)

Under SGLANG_DSV4_UNIFIED_KV_FP8 each row is split over two pools (512 B packed fp8
nope + 128 B bf16 rope); row indexing and every index builder below are unchanged.

Index layout: RAGGED-PACKED. Each token's segment is tightly packed
(``kv_indptr`` is a true prefix sum of per-token valid lengths) so the
attention K-loop scans only real entries. The backing buffer is still
allocated at the fixed worst-case capacity ``N * (win + Wc)`` so its shape is
static across CUDA-graph replay; only ``kv_indptr`` values (and the written
prefix) vary per forward. Compressed valid entries are front-packed in the
``*_page_indices`` rows (the same contract the non-unified_kv flashmla path relies
on via ``topk_length``); the per-token compressed count is recovered from the
``kv_indptr`` delta inside the kernel, so no extra length tensor is threaded.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.decode_attention_sm100 import (
    inverse_rope_tail,
)
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.layout import (
    check_two_pool_pair,
)
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode import (
    _cu_count,
    sparse_attn_v4_paged_decode,
)
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode_indices import (
    write_v4_paged_decode_indices,
)
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_prefill import (
    sparse_attn_v4_paged_prefill,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ragged indptr helper (shared by the decode streams + prefill builders)
# ---------------------------------------------------------------------------
def _lengths_to_indptr(lengths: torch.Tensor) -> torch.Tensor:
    """[N] int32 per-token lengths -> [N+1] int32 indptr"""
    return F.pad(torch.cumsum(lengths, dim=0, dtype=torch.int32), (1, 0))


def decode(
    *,
    q: torch.Tensor,  # [T, H, D] (local heads)
    unified_kv: torch.Tensor,  # [pages, D] bf16
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,  # [H] fp32
    softmax_scale: float,
    kv_splits: int | None = None,
) -> torch.Tensor:
    return sparse_attn_v4_paged_decode(
        q,
        unified_kv,
        kv_indices,
        kv_indptr,
        attn_sink,
        softmax_scale,
        kv_splits=kv_splits,
    )


@lru_cache(maxsize=None)
def decode_qo_indptr(num_tokens: int, device: torch.device) -> torch.Tensor:
    """``qo_indptr`` for the two-pool decode: one q token per sequence.

    Not ``cu_seqlens_q`` -- that one is per-request and differs once MTP puts
    several draft tokens in a batch. Cached unbounded like _token_identity_map:
    all 61 layers ask for the same answer each step, and a captured graph holds
    the address it got back.
    """
    return torch.arange(num_tokens + 1, dtype=torch.int32, device=device)


# aiter sizes the split count off CU occupancy and over-splits just past 40
# tokens, where the stage-2 merge starts to dominate. 4 rather than each shape's
# own optimum -- neighbouring split counts swing ~1.5x either way.
_DECODE_SPLIT_TAIL_MIN_TOKENS = 40
_DECODE_SPLIT_TAIL_VALUE = 4


def _decode_fp8_tail_splits(T: int, compress_ratio: Optional[int]) -> int:
    """Split count for T > _DECODE_SPLIT_TAIL_MIN_TOKENS.

    The v4 nm kernel fits one workgroup per CU and launches T x splits of them,
    so a fixed 4 needs a second wave once T > CU / 4 and the call roughly
    doubles (up to 2.4x at T=96..192 on MI355X). Take the largest split count
    up to 4 that still fits one wave. The SWA stream is 128 rows: unsplit, it
    skips the merge.
    """
    if compress_ratio is None:
        return _DECODE_SPLIT_TAIL_VALUE
    if compress_ratio == 0:
        return 1
    return max(1, min(_DECODE_SPLIT_TAIL_VALUE, _cu_count() // T))


# ---------------------------------------------------------------------------
# Split plan from aiter's v4 nm latency model (ROCm/aiter#6126, ATOM#2462)
# ---------------------------------------------------------------------------
# t(i) ~= L + W(i) * (C + A * kv_len / i) + [i > 1] * (S + D * num_seqs * i),
# W(i) = ceil(num_seqs * ceil(H / 64) * i / CU). The gfx950 fit is aiter's
# (gqa=128, qSeqLen=1, graph replay, uniform KV). Used only when the installed
# aiter predates get_mla_v4_nm_num_kv_splits.
_V4_NM_SPLIT_COST = {"gfx950": (12.74, 3.21, 0.0300, 3.38, 0.076)}
V4_NM_MAX_SPLITS = 16
# rows the constant split table covers; the asm wraps byte offsets past 32768
V4_NM_SPLIT_TABLE_ROWS = 32768


def _ported_num_kv_splits(
    num_seqs: int, num_heads: int, kv_len: int, gfx: str, cu_num: int
) -> Optional[int]:
    cost = _V4_NM_SPLIT_COST.get(gfx)
    if cost is None:
        return None
    L, C, A, S, D = cost
    tg_factor = max(1, -(-num_heads // 64))
    kv_len = max(1, kv_len)
    best_t, best_s = None, 1
    for s in range(1, V4_NM_MAX_SPLITS + 1):
        waves = -(-num_seqs * tg_factor * s // cu_num)
        t = L + waves * (C + A * kv_len / s)
        if s > 1:
            t += S + D * num_seqs * s
        if best_t is None or t < best_t:
            best_t, best_s = t, s
    return best_s


@lru_cache(maxsize=None)
def fp8_decode_num_kv_splits(
    num_seqs: int, num_heads: int, kv_len: int
) -> Optional[int]:
    """aiter's latency-model split count for ``mla_decode_fwd_v4_nm``, or None
    (no model for this arch: keep the existing pick). Host only, cached: the
    inputs are fixed per captured graph, so capture and replay agree."""
    try:
        from aiter.mla import get_mla_v4_nm_num_kv_splits
    except ImportError:
        get_mla_v4_nm_num_kv_splits = None
    if get_mla_v4_nm_num_kv_splits is not None:
        return int(get_mla_v4_nm_num_kv_splits(num_seqs, num_heads, kv_len))
    from aiter.jit.utils.chip_info import get_gfx

    return _ported_num_kv_splits(num_seqs, num_heads, kv_len, get_gfx(), _cu_count())


_split_tables: dict = {}


def fp8_decode_split_table(device: torch.device) -> Optional[torch.Tensor]:
    """``[V4_NM_MAX_SPLITS, V4_NM_SPLIT_TABLE_ROWS + 1]`` int32, row ``s - 1``
    the uniform ``split_indptr`` ``[0, s, 2s, ...]``. Built once per device
    outside capture and never written again, so every plan is a slice of
    constant memory (no per-layer arange). None if it would be built under
    capture: that graph keeps the wrapper's own pick."""
    device = torch.device(device)
    key = device.index if device.index is not None else torch.cuda.current_device()
    table = _split_tables.get(key)
    if table is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        splits = torch.arange(1, V4_NM_MAX_SPLITS + 1, dtype=torch.int32, device=device)
        table = splits[:, None] * torch.arange(
            V4_NM_SPLIT_TABLE_ROWS + 1, dtype=torch.int32, device=device
        )
        _split_tables[key] = table
    return table


def fp8_decode_split_plan(
    n_seq: int, num_heads: int, kv_len_hint: int, device: torch.device
) -> Optional[Tuple[int, torch.Tensor]]:
    """(num_kv_splits, split_indptr) from the latency model, or None."""
    if n_seq > V4_NM_SPLIT_TABLE_ROWS:
        return None
    s = fp8_decode_num_kv_splits(n_seq, num_heads, int(kv_len_hint))
    if s is None:
        return None
    table = fp8_decode_split_table(device)
    if table is None:
        return None
    return s, table[s - 1, : n_seq + 1]


# ---------------------------------------------------------------------------
# HCA persistent decode (aiter mla_decode_fwd_v4_nm_ps, ROCm/aiter#6126)
# ---------------------------------------------------------------------------
HCA_PS_HEADS = 128
HCA_PS_MAX_ROWS = 32768
HCA_PS_PARTITIONS = 128
_hca_ps_workspaces: dict = {}


def _hca_ps_api():
    try:
        from aiter.mla import get_mla_v4_nm_ps_workspace, mla_decode_fwd_v4_nm_ps
    except ImportError:
        return None
    return get_mla_v4_nm_ps_workspace, mla_decode_fwd_v4_nm_ps


def _dev_key(device) -> int:
    device = torch.device(device)
    return device.index if device.index is not None else torch.cuda.current_device()


def prepare_fp8_hca_persist(device, num_heads: int) -> bool:
    """Allocate this device's persistent-decode workspace (~64 MiB) when the
    switch is on and the kernel can serve this rank. Call outside capture,
    before warmup; idempotent. One workspace per device is enough because
    every decode layer runs on the forward's compute stream in order."""
    from sglang.srt.environ import envs
    from sglang.srt.utils import is_gfx95_supported

    if not envs.SGLANG_DSV4_FP8_HCA_PERSIST.get() or num_heads != HCA_PS_HEADS:
        return False
    if not is_gfx95_supported():
        return False
    key = _dev_key(device)
    if key in _hca_ps_workspaces:
        return True
    api = _hca_ps_api()
    if api is None or torch.cuda.is_current_stream_capturing():
        return False
    _hca_ps_workspaces[key] = api[0](
        torch.device("cuda", key), num_partitions=HCA_PS_PARTITIONS
    )
    return True


def fp8_hca_persist_wanted(
    *,
    T: int,
    H: int,
    v_head_dim: int,
    compress_ratio: Optional[int],
    max_seqlen_q: int,
    per_token: bool,
    device,
) -> bool:
    from sglang.srt.environ import envs

    if not (
        compress_ratio == 128
        and H == HCA_PS_HEADS
        and v_head_dim == 512
        and max_seqlen_q == 1
        and per_token
        and envs.SGLANG_DSV4_FP8_HCA_PERSIST.get()
        and envs.SGLANG_DSV4_FP8_HCA_PERSIST_MIN_ROWS.get() <= T <= HCA_PS_MAX_ROWS
    ):
        return False
    # eager calls may allocate it late; a graph captured without one keeps asm
    return prepare_fp8_hca_persist(device, H)


def decode_fp8_hca_ps(
    *,
    q: torch.Tensor,  # [T, 128, 512] fp8 packed
    q_rope: torch.Tensor,  # [T, 128, 64] bf16
    unified_kv: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
) -> torch.Tensor:
    """One-launch HCA decode: the kernel plans the split from ``kv_indptr`` on
    the GPU and merges the partials itself. Rows with an empty KV range are
    left unwritten (the builders never emit one; see decode_fp8_2buff)."""
    T = q.shape[0]
    out = q_rope.new_empty((T, HCA_PS_HEADS, 512))
    if T == 0:
        return out
    _hca_ps_api()[1](
        q,
        q_rope,
        unified_kv,
        unified_kv_rope,
        kv_indptr[: T + 1],
        kv_indices,
        attn_sink,
        _hca_ps_workspaces[_dev_key(q.device)],
        out=out,
    )
    return out


# TODO(aiter#5882): remove once ROCm/aiter#5882 is in the aiter pin
# (with it: the in-tree fallback below, flydsl_v4nm/ and
# SGLANG_DSV4_FLYDSL_DECODE_SOURCE; aiter's copy is then the only one).
_FLYDSL_V4_DECODE_SOURCES = ("auto", "aiter", "intree")


@lru_cache(maxsize=1)
def _flydsl_v4_decode():
    """The FlyDSL v4 nm decode API (ROCm/aiter#5882) or None (asm kernel).

    SGLANG_DSV4_FLYDSL_DECODE=0 forces None. SGLANG_DSV4_FLYDSL_DECODE_SOURCE:
    auto (default) takes aiter.ops.flydsl when the installed aiter ships it, else
    sglang's in-tree copy (flydsl_v4nm); aiter / intree take only that one.
    """
    from sglang.srt.environ import envs

    if not envs.SGLANG_DSV4_FLYDSL_DECODE.get():
        return None
    source = envs.SGLANG_DSV4_FLYDSL_DECODE_SOURCE.get()
    if source not in _FLYDSL_V4_DECODE_SOURCES:
        raise ValueError(
            f"SGLANG_DSV4_FLYDSL_DECODE_SOURCE={source!r}, expected one of "
            f"{_FLYDSL_V4_DECODE_SOURCES}"
        )
    fly = None
    if source != "intree":
        try:
            from aiter.ops import flydsl as fly

            fly.flydsl_mla_v4_decode_supported  # resolves the lazy import
            origin = "aiter.ops.flydsl"
        except (ImportError, AttributeError):
            fly = None
    if fly is None and source != "aiter":
        try:
            from sglang.kernels.ops.attention.dsv4 import flydsl_v4nm as fly

            origin = "sglang in-tree copy of ROCm/aiter#5882"
        except ImportError as e:  # no flydsl, or an aiter before ROCm/aiter#6377
            fly = None
            logger.warning("in-tree FlyDSL v4 nm decode unavailable: %s", e)
    if fly is None:
        logger.info("DSv4 fp8 decode: no FlyDSL v4 nm decode (source=%s)", source)
    else:
        logger.info("DSv4 fp8 decode: FlyDSL v4 nm decode from %s", origin)
    return fly


def flydsl_decode_supported(
    num_heads: int, max_seqlen_q: int = 1, grouped: bool = False
) -> bool:
    fly = _flydsl_v4_decode()
    if fly is None:
        return False
    if num_heads == 128:
        from sglang.srt.environ import envs

        if not envs.SGLANG_DSV4_FLYDSL_DECODE_H128.get():
            return False
    return fly.flydsl_mla_v4_decode_supported(num_heads, max_seqlen_q, grouped)


def _stream_hint(compress_ratio: Optional[int]) -> Optional[str]:
    # FlyDSL split plans are keyed by stream kind; DeepSeek-V4.1's ratio-1/2
    # streams are an SWA-ring prefix plus a top-k tail, the CSA shape
    if compress_ratio is None:
        return None
    return {0: "swa", 1: "csa", 2: "csa", 4: "csa", 128: "hca"}[compress_ratio]


def flydsl_grouped_verify_pays(
    num_heads: int, num_draft: int, num_reqs: int, compress_ratio: int, epi: str
) -> bool:
    """aiter's measured policy: whether target verify should read one shared
    stream per request on this layer kind, for output mode ``epi``."""
    return flydsl_decode_supported(
        num_heads, num_draft, grouped=True
    ) and _flydsl_v4_decode().flydsl_mla_v4_grouped_pays(
        num_heads, num_draft, num_reqs, hint=_stream_hint(compress_ratio), epi=epi
    )


def build_grouped_verify_streams(
    state_slot: torch.Tensor, positions: torch.Tensor, *args, **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(kv_indices, kv_indptr, qo_indptr, q_kv_bounds): one stream per request,
    with per-draft key ranges equal to each draft's per-token stream."""
    return _flydsl_v4_decode().flydsl_mla_v4_grouped_verify_meta(
        state_slot.contiguous(), positions.contiguous(), *args, **kwargs
    )


# DeepSeek-V4.1 low-ratio streams (128 SWA rows + up to 512 top-k rows): the
# occupancy heuristic over-splits them and the stage-2 merge dominates. MI355X,
# 16 heads, 640 rows: auto 16.1 us (merge 9.2) vs 8 splits 14.1 at T=1, best at
# T=8 / 16 too (work/tests/bench_unified_decode.py).
_DECODE_LOW_RATIO_SPLITS = 8


def _decode_fp8_head_splits(T: int, compress_ratio: Optional[int]) -> Optional[int]:
    """Split count for T <= _DECODE_SPLIT_TAIL_MIN_TOKENS; None keeps aiter's pick.

    The 128-row SWA stream runs fastest unsplit (7.1 vs 10.4 us at T=1), the
    ratio-1/2 streams at 8 splits; c4 / c128 keep the heuristic."""
    if compress_ratio == 0:
        return 1
    if compress_ratio in (1, 2):
        return max(1, min(_DECODE_LOW_RATIO_SPLITS, _cu_count() // T))
    return None


@triton.jit
def _merge_splits_inv_rope_kernel(
    mid_o_ptr,  # [n_seq, S, H, Lv] fp32 split partials (normalised per split)
    mid_lse_ptr,  # [n_seq, S, H, 1] fp32
    out_ptr,  # [n_seq, H, Lv] bf16
    kv_indptr_ptr,  # [n_seq+1] int32
    split_indptr_ptr,  # [n_seq+1] int32
    valid_split_ptr,  # [n_seq] int32, written by the asm kernel
    fr_ptr,  # [max_pos, ROPE_DIM] fp32, real/imag interleaved
    pos_ptr,  # [n_seq] int
    stride_mid_ob,  # S * H
    stride_mid_os,  # H
    stride_obs,
    stride_oh,
    fr_stride_pos,
    Lv: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    mgc: tl.constexpr,
    MAX_S: tl.constexpr,  # >= the split count, power of two
):
    """aiter's ``_fwd_kernel_stage2_asm`` (decode: one q row per sequence, split
    count > 1) with the model's inverse RoPE on the last ROPE_DIM dims folded into
    the store, as SGLang's gluon split reduce does for the paged V4.1 path.

    Every split's partial is loaded up front, in one batch of loads, and the
    online-softmax recurrence then runs in stage2's order on registers: the same
    arithmetic, without one dependent global load per split."""
    seq = tl.program_id(0)
    head = tl.program_id(1)
    kv_len = tl.load(kv_indptr_ptr + seq + 1) - tl.load(kv_indptr_ptr + seq)
    n_split = tl.load(split_indptr_ptr + seq + 1) - tl.load(split_indptr_ptr + seq)
    n_valid = tl.minimum(n_split, tl.cdiv(kv_len, mgc))
    n_valid = tl.minimum(n_valid, tl.load(valid_split_ptr + seq))
    d = tl.arange(0, Lv)
    offs_logic = seq * stride_mid_ob + head
    s = tl.arange(0, MAX_S)
    live_s = s < n_valid
    tv_all = tl.load(
        mid_o_ptr + (offs_logic + s[:, None] * stride_mid_os) * Lv + d[None, :],
        mask=live_s[:, None],
        other=0.0,
    )
    lse_all = tl.load(
        mid_lse_ptr + offs_logic + s * stride_mid_os, mask=live_s, other=0.0
    )
    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros((Lv,), dtype=tl.float32)
    for split in tl.static_range(MAX_S):
        pick = s == split
        tv = tl.sum(tl.where(pick[:, None], tv_all, 0.0), axis=0)
        tlogic = tl.sum(tl.where(pick, lse_all, 0.0), axis=0)
        live = split < n_valid
        n_e_max = tl.maximum(tlogic, e_max)
        old_scale = tl.exp(e_max - n_e_max)
        new_acc = acc * old_scale
        exp_logic = tl.exp(tlogic - n_e_max)
        new_acc += exp_logic * tv
        acc = tl.where(live, new_acc, acc)
        e_sum = tl.where(live, e_sum * old_scale + exp_logic, e_sum)
        e_max = tl.where(live, n_e_max, e_max)
    out = (acc / e_sum).to(out_ptr.dtype.element_ty)
    pos = tl.load(pos_ptr + seq)
    out = inverse_rope_tail(out, d, fr_ptr, pos, fr_stride_pos, Lv, ROPE_DIM)
    tl.store(out_ptr + seq * stride_obs + head * stride_oh + d, out)


_MERGE_NUM_WARPS = 2  # bench_merge.py: 3.4 us at T=1..16, 8 splits (1 warp 3.8, 4 warps 5.3; stage2 5.2)


def _decode_fp8_2buff_inv_rope(
    q,
    q_rope,
    unified_kv,
    unified_kv_rope,
    kv_indices,
    kv_indptr,
    attn_sink,
    out,
    qo_indptr,
    num_kv_splits,
    inv_rope,
) -> None:
    """mla_decode_fwd_v4_nm's asm call and split bookkeeping, then the split merge
    with the inverse RoPE folded in (one kernel instead of stage2 + a RoPE pass)."""
    import aiter
    from aiter.jit.utils.chip_info import get_gfx
    from aiter.mla import get_meta_param

    n_seq = qo_indptr.shape[0] - 1
    T, H, row_bytes = q.shape
    rows = unified_kv.shape[0]
    v_head_dim = out.shape[-1]
    _, split_indptr = get_meta_param(
        num_kv_splits, n_seq, kv_indices.shape[0], H, 1, q.dtype, max(1, -(-H // 64)), 1
    )
    split_indptr = split_indptr.to(device=q.device, dtype=torch.int32)
    logits = torch.empty(
        (n_seq, num_kv_splits, H, v_head_dim), dtype=torch.float32, device=q.device
    )
    lse = torch.empty(
        (n_seq, num_kv_splits, H, 1), dtype=torch.float32, device=q.device
    )
    valid = torch.empty((n_seq,), dtype=torch.int32, device=q.device)
    aiter.mla_decode_v4_asm(
        q,
        q_rope,
        unified_kv.view(rows, 1, 1, row_bytes),
        unified_kv_rope.view(rows, 1, 1, unified_kv_rope.shape[-1]),
        qo_indptr,
        kv_indptr,
        kv_indices,
        split_indptr,
        attn_sink,
        1,
        0.0,
        0,
        int(num_kv_splits),
        logits,
        lse,
        out,
        valid,
        1,
        None,
    )
    freqs_real, positions = inv_rope
    assert freqs_real.dtype == torch.float32 and freqs_real.stride(1) == 1
    mgc = (
        32 if (H in (8, 16) and get_gfx() == "gfx950") else (64 if H in (8, 16) else 16)
    )
    _merge_splits_inv_rope_kernel[(n_seq, H)](
        logits,
        lse,
        out,
        kv_indptr,
        split_indptr,
        valid,
        freqs_real,
        positions,
        lse.stride(0),
        lse.stride(1),
        out.stride(0),
        out.stride(1),
        freqs_real.stride(0),
        Lv=v_head_dim,
        ROPE_DIM=freqs_real.shape[-1],
        mgc=mgc,
        MAX_S=triton.next_power_of_2(int(num_kv_splits)),
        num_warps=_MERGE_NUM_WARPS,
        num_stages=2,
        waves_per_eu=4,
    )


def decode_fp8_2buff(
    *,
    q: torch.Tensor,  # [T, H, nope_row_bytes] fp8 packed nope + inline e8m0 scale
    q_rope: torch.Tensor,  # [T, H, rope_dim] bf16
    unified_kv: torch.Tensor,  # [rows, nope_row_bytes] fp8
    unified_kv_rope: torch.Tensor,  # [rows, rope_dim] bf16
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,  # [H] fp32
    v_head_dim: int,
    qo_indptr: Optional[torch.Tensor] = None,
    max_seqlen_q: int = 1,
    num_kv_splits: Optional[int] = None,
    compress_ratio: Optional[int] = None,
    q_kv_bounds: Optional[torch.Tensor] = None,
    kv_len_hint: Optional[int] = None,
    low_ratio_model: bool = False,
    inv_rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Decode over the two-pool fp8 unified_kv, through aiter's FlyDSL v4 nm
    decode (split merge in the same launch) where available, else the asm kernel.

    On the asm kernel, with ``SGLANG_DSV4_FP8_HCA_PERSIST`` the HCA stream
    (H128, one q row per sequence) takes the persistent kernel, and with
    ``SGLANG_DSV4_FP8_DECODE_SPLIT_MODEL`` plus ``kv_len_hint`` (the longest
    KV a row of this stream can see, fixed per server) the split count comes
    from aiter's latency model with a constant ``split_indptr``.

    Q arrives in the same packed form as the pool rows (nope fp8 + duplicated
    e8m0 tile scales) with its rope half beside it in bf16, which is why this
    can't share ``decode``'s single bf16 tensor. The kernel takes the row stride
    off ``kv_buffer.size(-1)`` and only requires Q to match it, so the 512 B row
    is not baked into the reader.

    ``v_head_dim`` is an element count (448 nope + 64 rope) that happens to equal
    the row's byte width; it comes from the caller so that nothing here reads one
    as the other.

    ``compress_ratio`` names the stream (0 SWA, 4 CSA, 128 HCA) so the split
    count past 40 tokens can fit the stream; None keeps the fixed 4.
    ``low_ratio_model`` (DeepSeek-V4.1: ratios 0 / 1 / 2) also picks the asm
    split count up to 40 tokens (``_decode_fp8_head_splits``); V4 keeps aiter's.

    ``inv_rope`` = (freqs_real ``[max_pos, rope_dim]`` fp32, positions) applies the
    model's inverse RoPE to the output: folded into the split merge when the asm
    call splits (decode, one q row per sequence), else a RoPE pass on the bf16
    output.

    ``max_seqlen_q`` above 1 puts several q rows in one tile, which the v4
    dispatcher serves for gqa=16 at 1, 2 and 4 off the same 64-row tile.

    ``q_kv_bounds`` (FlyDSL only): the grouped verify streams of
    ``build_grouped_verify_streams``.
    """
    from aiter.mla import mla_decode_fwd_v4_nm

    T, H, row_bytes = q.shape
    check_two_pool_pair(
        unified_kv,
        unified_kv_rope,
        rope_width=q_rope.shape[-1],
        rope_dtype=q_rope.dtype,
    )
    assert row_bytes == unified_kv.shape[-1], (
        f"aiter derives the row stride from the kv pool ({unified_kv.shape[-1]} B) "
        f"and reads Q with that same stride, but the q row is {row_bytes} B"
    )
    assert q_rope.shape[:2] == (T, H), (
        f"q pair disagrees: packed {tuple(q.shape)[:2]} vs rope "
        f"{tuple(q_rope.shape)[:2]}"
    )
    # the asm kernel walks all four as flat buffers, it has no stride arguments
    assert q.is_contiguous(), f"q must be contiguous, strides {q.stride()}"
    assert q_rope.is_contiguous(), (
        f"q_rope must be contiguous, strides {q_rope.stride()}"
    )
    assert attn_sink.dtype == torch.float32 and attn_sink.numel() == H, (
        f"sink must be {H} fp32 values, got {attn_sink.numel()} x {attn_sink.dtype}"
    )

    if qo_indptr is None:
        qo_indptr = decode_qo_indptr(T, q.device)
    n_seq = qo_indptr.shape[0] - 1
    assert kv_indptr.shape[0] >= n_seq + 1, (
        f"kv_indptr holds {kv_indptr.shape[0]} entries, kernel reads {n_seq + 1}"
    )

    rows = unified_kv.shape[0]
    out = q_rope.new_empty((n_seq * max_seqlen_q, H, v_head_dim))
    # FlyDSL is a drop-in for the asm kernel: its own capture-safe split plan,
    # split merge in the same launch
    if flydsl_decode_supported(H, max_seqlen_q, q_kv_bounds is not None):
        decode_fwd = _flydsl_v4_decode().flydsl_mla_decode_fwd_v4_nm
        kwargs = dict(hint=_stream_hint(compress_ratio), q_kv_bounds=q_kv_bounds)
    else:
        assert q_kv_bounds is None, "grouped verify streams need FlyDSL"
        if inv_rope is None and fp8_hca_persist_wanted(
            T=T,
            H=H,
            v_head_dim=v_head_dim,
            compress_ratio=compress_ratio,
            max_seqlen_q=max_seqlen_q,
            per_token=n_seq == T,
            device=q.device,
        ):
            return decode_fp8_hca_ps(
                q=q,
                q_rope=q_rope,
                unified_kv=unified_kv,
                unified_kv_rope=unified_kv_rope,
                kv_indices=kv_indices,
                kv_indptr=kv_indptr,
                attn_sink=attn_sink,
            )
        split_indptr = None
        if num_kv_splits is None and kv_len_hint is not None:
            from sglang.srt.environ import envs

            if envs.SGLANG_DSV4_FP8_DECODE_SPLIT_MODEL.get():
                plan = fp8_decode_split_plan(n_seq, H, kv_len_hint, q.device)
                if plan is not None:
                    num_kv_splits, split_indptr = plan
        # Left None, the wrapper's occupancy heuristic picks it, folds the
        # cross-split merge back into `out`, and leaves the final bf16 there
        # whether or not it split. Pinning it to 1 costs 6.9x at bs=1 kv=2048.
        # Sequences, not tokens: the wave-fitting rule counts the workgroups the
        # kernel launches, which is n_seq x splits. They are the same number on
        # the per-token path and differ once q rows share a tile.
        if num_kv_splits is None and n_seq > _DECODE_SPLIT_TAIL_MIN_TOKENS:
            num_kv_splits = _decode_fp8_tail_splits(n_seq, compress_ratio)
        elif num_kv_splits is None and low_ratio_model:
            num_kv_splits = _decode_fp8_head_splits(n_seq, compress_ratio)
        if (
            inv_rope is not None
            and num_kv_splits is not None
            and num_kv_splits > 1
            and max_seqlen_q == 1
            and n_seq == T
        ):
            _decode_fp8_2buff_inv_rope(
                q,
                q_rope,
                unified_kv,
                unified_kv_rope,
                kv_indices,
                kv_indptr,
                attn_sink,
                out,
                qo_indptr,
                num_kv_splits,
                inv_rope,
            )
            return out[:T]
        decode_fwd, kwargs = (
            mla_decode_fwd_v4_nm,
            dict(num_kv_splits=num_kv_splits, split_indptr=split_indptr),
        )
    decode_fwd(
        q,
        q_rope,
        unified_kv.view(rows, 1, 1, row_bytes),
        unified_kv_rope.view(rows, 1, 1, unified_kv_rope.shape[-1]),
        out,
        qo_indptr,
        kv_indptr,
        kv_indices,
        max_seqlen_q,
        sink=attn_sink,
        **kwargs,
    )
    # No empty-segment mask: a CG-padded row gets seq_len 1 on the ring slot
    # ReqToTokenPool reserves, so the builders can't emit a zero-length one, and
    # the compare + masked_fill_ was costing a launch per layer for it. One would
    # come back NaN now (all-sink denominator); the guard UT pins that.
    if inv_rope is not None:
        apply_inverse_rope(out[:T], inv_rope)
    return out[:T]


def decode_fp8_2buff_wo_a_mxfp8(
    *,
    q: torch.Tensor,  # [T, H, nope_row_bytes] fp8 packed nope + inline e8m0 scale
    q_rope: torch.Tensor,  # [T, H, rope_dim] bf16
    unified_kv: torch.Tensor,  # [rows, nope_row_bytes] fp8
    unified_kv_rope: torch.Tensor,  # [rows, rope_dim] bf16
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,  # [H] fp32
    positions: torch.Tensor,  # [T] int64
    rope_freqs: torch.Tensor,  # [max_pos, rope_dim] fp32, view_as_real(freqs_cis)
    compress_ratio: int,
    qo_indptr: Optional[torch.Tensor] = None,
    max_seqlen_q: int = 1,
    q_kv_bounds: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``decode_fp8_2buff`` + inverse RoPE + wo_a mxfp8 quant in one FlyDSL
    launch. Returns the wo_a operands (fp8 [T, H*512], e8m0 [T, H*4])."""
    return _flydsl_v4_decode().flydsl_mla_v4_decode_fused(
        q,
        q_rope,
        unified_kv,
        unified_kv_rope,
        kv_indptr,
        kv_indices,
        attn_sink,
        positions,
        rope_freqs,
        compress_ratio,
        qo_indptr=qo_indptr,
        max_seqlen_q=max_seqlen_q,
        q_kv_bounds=q_kv_bounds,
    )


def apply_inverse_rope(
    out: torch.Tensor, inv_rope: Tuple[torch.Tensor, torch.Tensor]
) -> None:
    """The model's standalone inverse RoPE on ``out`` ``[T, H, D]`` (last rope_dim
    dims of every head), for the paths that do not fold it into a merge."""
    from sglang.kernels.ops.attention.dsv4.norm_rope import fused_rope_inplace

    freqs_real, positions = inv_rope
    T = out.shape[0]
    freqs_cis = torch.view_as_complex(freqs_real.view(freqs_real.shape[0], -1, 2))
    fused_rope_inplace(
        out[..., -freqs_real.shape[-1] :],
        None,
        freqs_cis,
        positions.view(-1)[:T],
        inverse=True,
    )


@triton.jit
def _fill_compress_tail_kernel(
    indices_ptr,  # [*] int32 (out)
    indptr_ptr,  # [N+1] int32
    prefix_len_ptr,  # [N] int
    page_idx_ptr,  # [N, Wc] int
    valid_len_ptr,  # [N] int
    swa_pages,
    Wc: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Per token: write valid_len compressed slots (swa_pages+page_idx, -1 for empty) into the stream tail at indptr[t]+prefix_len[t]."""
    t = tl.program_id(0)
    cbase = tl.load(indptr_ptr + t) + tl.load(prefix_len_ptr + t).to(tl.int32)
    nc = tl.load(valid_len_ptr + t).to(tl.int32)
    for off in tl.range(0, Wc, BLOCK):
        j = off + tl.arange(0, BLOCK)
        m = j < nc
        j_clamped = tl.minimum(j, Wc - 1)
        pi = tl.load(page_idx_ptr + t * Wc + j_clamped, mask=m, other=-1).to(tl.int32)
        slot = tl.where(pi >= 0, pi + swa_pages, -1)
        tl.store(indices_ptr + cbase + j, slot, mask=m)


def fill_compress_tail(
    *,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    prefix_len: torch.Tensor,
    page_indices: torch.Tensor,  # [N, Wc] int32
    valid_len: torch.Tensor,
    swa_pages: int,
) -> None:
    N, Wc = page_indices.shape
    if N == 0:
        return
    assert prefix_len.is_contiguous() and page_indices.is_contiguous()
    assert valid_len.is_contiguous()
    _fill_compress_tail_kernel[(N,)](
        indices,
        indptr,
        prefix_len,
        page_indices,
        valid_len,
        swa_pages,
        Wc=Wc,
        BLOCK=min(1024, triton.next_power_of_2(max(Wc, 1))),
        num_warps=4,
    )


def build_decode_streams(
    *,
    state_slot: torch.Tensor,  # [N] int
    positions: torch.Tensor,  # [N] int
    swa_len: torch.Tensor,  # [N] int
    hca_len: torch.Tensor,  # [N] int
    csa_len: torch.Tensor,  # [N] int
    hca_page_indices: torch.Tensor,  # [N, hca_width] int32
    csa_width: int,
    win: int,  # SWA attention window length
    ring_stride: int,  # SWA ring per-slot stride
    swa_pages: int,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    device = state_slot.device
    N = state_slot.shape[0]
    assert state_slot.is_contiguous() and positions.is_contiguous()
    state_slot = state_slot.to(torch.int32)
    positions = positions.to(torch.int32)
    hca_width = hca_page_indices.shape[1]

    swa_p = _lengths_to_indptr(swa_len)
    hca_p = _lengths_to_indptr(swa_len + hca_len)
    csa_p = _lengths_to_indptr(swa_len + csa_len)

    swa_i = torch.empty(N * win, dtype=torch.int32, device=device)
    hca_i = torch.empty(N * (win + hca_width), dtype=torch.int32, device=device)
    csa_i = torch.empty(N * (win + csa_width), dtype=torch.int32, device=device)

    if N > 0:
        batch_id = torch.arange(N, dtype=torch.int32, device=device)
        write_v4_paged_decode_indices(
            state_slot_per_seq=state_slot,
            batch_id_per_token=batch_id,
            positions=positions,
            swa_indptr=swa_p,
            csa_indptr=csa_p,
            hca_indptr=hca_p,
            swa_indices=swa_i,
            csa_indices=csa_i,
            hca_indices=hca_i,
            T=N,
            win=win,
            ring_stride=ring_stride,
        )
        fill_compress_tail(
            indices=hca_i,
            indptr=hca_p,
            prefix_len=swa_len,
            page_indices=hca_page_indices[:N],
            valid_len=hca_len,
            swa_pages=swa_pages,
        )
    return swa_i, swa_p, hca_i, hca_p, csa_i, csa_p


@triton.jit
def _swa_prefix_kernel(
    state_slot_ptr,  # [N] int32
    positions_ptr,  # [N] int32
    indptr_ptr,  # [N+1] int32
    indices_ptr,  # [*] int32 (out)
    ring_stride,
    win: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Per token: the SWA-ring prefix ``slot*ring + (pos-n+1+i)%ring``, i < n,
    n = min(pos+1, win), at indptr[t] -- the prefix every decode stream starts with."""
    t = tl.program_id(0)
    slot = tl.load(state_slot_ptr + t)
    pos = tl.load(positions_ptr + t)
    n = tl.minimum(pos + 1, win)
    base = tl.load(indptr_ptr + t)
    i = tl.arange(0, BLOCK_N)
    abs_pos = pos - n + 1 + i
    tl.store(
        indices_ptr + base + i, slot * ring_stride + abs_pos % ring_stride, mask=i < n
    )


def build_low_ratio_decode_stream(
    *,
    state_slot: torch.Tensor,  # [N] int
    positions: torch.Tensor,  # [N] int
    swa_len: torch.Tensor,  # [N] int, == min(pos+1, win)
    compress_len: torch.Tensor,  # [N] int, valid front-packed top-k entries
    width: int,  # top-k columns of the ratio's sparse page indices
    win: int,
    ring_stride: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode stream of a ratio-1/2 (DeepSeek-V4.1) layer: the SWA-ring prefix,
    written here once per step, then ``compress_len`` compressed rows that
    ``fill_compress_tail`` writes per layer (the indexer rewrites the top-k and
    each kv_source group member has its own compressed-row base)."""
    device = state_slot.device
    N = state_slot.shape[0]
    indptr = _lengths_to_indptr(swa_len + compress_len)
    indices = torch.empty(N * (win + width), dtype=torch.int32, device=device)
    if N > 0:
        _swa_prefix_kernel[(N,)](
            state_slot.to(torch.int32),
            positions.to(torch.int32),
            indptr,
            indices,
            ring_stride,
            win=win,
            BLOCK_N=triton.next_power_of_2(win),
        )
    return indices, indptr


@triton.jit
def _v41_decode_lengths_kernel(
    seq_lens_ptr,  # [N] int
    swa_len_ptr,  # [N] int
    swa_indptr_ptr,  # [N+1] int32 (out)
    c1_len_ptr,  # [N] int32 (out)
    c1_indptr_ptr,  # [N+1] int32 (out)
    c2_len_ptr,
    c2_indptr_ptr,
    N,
    topk,
    HAS_C1: tl.constexpr,
    HAS_C2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program: the SWA stream's indptr and, per ratio R in (1, 2), the top-k
    length min(seq_len // R, topk) and its stream's indptr (SWA prefix + tail)."""
    i = tl.arange(0, BLOCK)
    m = i < N
    sl = tl.load(seq_lens_ptr + i, mask=m, other=0).to(tl.int32)
    sw = tl.load(swa_len_ptr + i, mask=m, other=0).to(tl.int32)
    tl.store(swa_indptr_ptr, 0)
    tl.store(swa_indptr_ptr + 1 + i, tl.cumsum(sw, axis=0), mask=m)
    if HAS_C1:
        ln = tl.minimum(sl, topk)
        tl.store(c1_len_ptr + i, ln, mask=m)
        tl.store(c1_indptr_ptr, 0)
        tl.store(c1_indptr_ptr + 1 + i, tl.cumsum(sw + ln, axis=0), mask=m)
    if HAS_C2:
        ln = tl.minimum(sl // 2, topk)
        tl.store(c2_len_ptr + i, ln, mask=m)
        tl.store(c2_indptr_ptr, 0)
        tl.store(c2_indptr_ptr + 1 + i, tl.cumsum(sw + ln, axis=0), mask=m)


@triton.jit
def _v41_decode_swa_kernel(
    state_slot_ptr,  # [N] int
    positions_ptr,  # [N] int
    indptr_ptr,  # [N+1] int32
    indices_ptr,  # [*] int32 (out)
    swa_loc_ptr,  # [N] int32 (out)
    swa_loc_i64_ptr,  # [N] int64 (out)
    ring_stride,
    win: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Per token: _swa_prefix_kernel's SWA-ring prefix, and the ring row this
    step's K goes to, slot*ring + pos%ring."""
    t = tl.program_id(0)
    slot = tl.load(state_slot_ptr + t).to(tl.int64)
    pos = tl.load(positions_ptr + t).to(tl.int64)
    n = tl.minimum(pos + 1, win)
    base = tl.load(indptr_ptr + t)
    i = tl.arange(0, BLOCK_N)
    abs_pos = pos - n + 1 + i
    tl.store(
        indices_ptr + base + i,
        (slot * ring_stride + abs_pos % ring_stride).to(tl.int32),
        mask=i < n,
    )
    loc = slot * ring_stride + pos % ring_stride
    tl.store(swa_loc_ptr + t, loc.to(tl.int32))
    tl.store(swa_loc_i64_ptr + t, loc)


def build_v41_decode_meta(
    *,
    state_slot: torch.Tensor,  # [N] int
    positions: torch.Tensor,  # [N] int
    seq_lens: torch.Tensor,  # [N] int
    swa_len: torch.Tensor,  # [N] int, == min(pos+1, win)
    ratios,  # the model's ratio-1/2 set
    topk: int,
    win: int,
    ring_stride: int,
) -> dict:
    """Decode metadata of a model whose compressed layers are all ratio 1 / 2
    (DeepSeek-V4.1), in two launches: the SWA stream (ratio-0 layers), each
    ratio's top-k length and stream indptr (the streams themselves are built per
    index source, build_low_ratio_group_streams), and the SWA ring write rows."""
    N = state_slot.shape[0]
    dev = state_slot.device
    i32 = dict(dtype=torch.int32, device=dev)
    out = dict(
        swa_indptr=torch.empty(N + 1, **i32),
        swa_indices=torch.empty(N * win, **i32),
        swa_loc=torch.empty(N, **i32),
        swa_loc_i64=torch.empty(N, dtype=torch.int64, device=dev),
    )
    for r in (1, 2):
        if r in ratios:
            out[f"c{r}_len"] = torch.empty(N, **i32)
            out[f"c{r}_indptr"] = torch.empty(N + 1, **i32)
    if N == 0:
        out["swa_indptr"].zero_()
        for r in ratios:
            out[f"c{r}_indptr"].zero_()
        return out
    assert state_slot.is_contiguous() and positions.is_contiguous()
    assert seq_lens.is_contiguous() and swa_len.is_contiguous()
    dummy = out["swa_indptr"]
    _v41_decode_lengths_kernel[(1,)](
        seq_lens,
        swa_len,
        out["swa_indptr"],
        out.get("c1_len", dummy),
        out.get("c1_indptr", dummy),
        out.get("c2_len", dummy),
        out.get("c2_indptr", dummy),
        N,
        topk,
        HAS_C1=1 in ratios,
        HAS_C2=2 in ratios,
        BLOCK=triton.next_power_of_2(N),
        num_warps=4,
    )
    _v41_decode_swa_kernel[(N,)](
        state_slot,
        positions,
        out["swa_indptr"],
        out["swa_indices"],
        out["swa_loc"],
        out["swa_loc_i64"],
        ring_stride,
        win=win,
        BLOCK_N=triton.next_power_of_2(win),
    )
    return out


@triton.jit
def _low_ratio_group_stream_kernel(
    indices_ptr,  # [G, stride_g] int32 (out)
    indptr_ptr,  # [N+1] int32
    state_slot_ptr,  # [N] int
    positions_ptr,  # [N] int
    prefix_len_ptr,  # [N] int
    page_idx_ptr,  # [N, Wc] int
    valid_len_ptr,  # [N] int
    stride_g,
    ring_stride,
    base0,
    base_step,
    win: tl.constexpr,
    Wc: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Per (token t, layer g): _swa_prefix_kernel's SWA-ring prefix, then
    _fill_compress_tail_kernel's tail at compressed-row base base0 + g*base_step."""
    t = tl.program_id(0)
    g = tl.program_id(1)
    out = indices_ptr + g.to(tl.int64) * stride_g + tl.load(indptr_ptr + t)
    slot = tl.load(state_slot_ptr + t).to(tl.int32)
    pos = tl.load(positions_ptr + t).to(tl.int32)
    n = tl.minimum(pos + 1, win)
    i = tl.arange(0, BLOCK_N)
    tl.store(out + i, slot * ring_stride + (pos - n + 1 + i) % ring_stride, mask=i < n)
    tail = out + tl.load(prefix_len_ptr + t).to(tl.int32)
    base = base0 + g * base_step
    nc = tl.load(valid_len_ptr + t).to(tl.int32)
    for off in tl.range(0, Wc, BLOCK):
        j = off + tl.arange(0, BLOCK)
        m = j < nc
        j_clamped = tl.minimum(j, Wc - 1)
        pi = tl.load(page_idx_ptr + t * Wc + j_clamped, mask=m, other=-1).to(tl.int32)
        tl.store(tail + j, tl.where(pi >= 0, pi + base, -1), mask=m)


def low_ratio_decode_indptr(
    swa_len: torch.Tensor, compress_len: torch.Tensor
) -> torch.Tensor:
    """indptr of a ratio-1/2 decode stream (SWA prefix + compressed tail per token)."""
    return _lengths_to_indptr(swa_len + compress_len)


def build_low_ratio_group_streams(
    *,
    state_slot: torch.Tensor,  # [N] int
    positions: torch.Tensor,  # [N] int
    indptr: torch.Tensor,  # [N+1] int32, low_ratio_decode_indptr
    prefix_len: torch.Tensor,  # [N] int, == min(pos+1, win)
    page_indices: torch.Tensor,  # [N, Wc] int32, the top-k these layers read
    valid_len: torch.Tensor,  # [N] int, valid front-packed top-k entries
    num_layers: int,
    base0: int,
    base_step: int,
    win: int,
    ring_stride: int,
) -> torch.Tensor:
    """The decode streams of ``num_layers`` ratio-1/2 layers that read the same
    top-k, in one launch: row g is what build_low_ratio_decode_stream followed by
    fill_compress_tail(swa_pages=base0 + g*base_step) leaves for layer g. The
    layers differ only in their compressed-row base (kv_source group position)."""
    N, Wc = page_indices.shape
    stride_g = N * (win + Wc)
    out = torch.empty((num_layers, stride_g), dtype=torch.int32, device=indptr.device)
    if N == 0:
        return out
    assert state_slot.is_contiguous() and positions.is_contiguous()
    assert prefix_len.is_contiguous() and page_indices.is_contiguous()
    assert valid_len.is_contiguous()
    _low_ratio_group_stream_kernel[(N, num_layers)](
        out,
        indptr,
        state_slot,
        positions,
        prefix_len,
        page_indices,
        valid_len,
        stride_g,
        ring_stride,
        base0,
        base_step,
        win=win,
        Wc=Wc,
        BLOCK_N=triton.next_power_of_2(win),
        BLOCK=min(1024, triton.next_power_of_2(max(Wc, 1))),
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# Prefill index builder (ragged-packed: paged prefix + flat extend)
# ---------------------------------------------------------------------------
@triton.jit
def _prefill_lengths_kernel(
    positions_ptr,  # [T] int
    chunk_start_ptr,  # [T] int
    page_idx_ptr,  # [T, Wc] int (front-packed, -1 padded)
    prefix_len_ptr,  # [T] int32 out
    extend_len_ptr,  # [T] int32 out
    floor_ptr,  # [T] int: lowest SWA position the token may see (HAS_FLOOR)
    win: tl.constexpr,
    Wc: tl.constexpr,
    HAS_COMPRESS: tl.constexpr,
    HAS_FLOOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Per token: write extend/prefix segment lengths"""
    t = tl.program_id(0)
    pos = tl.load(positions_ptr + t).to(tl.int32)
    cstart = tl.load(chunk_start_ptr + t).to(tl.int32)
    tpic = pos - cstart
    swa_low = tl.maximum(pos - win + 1, 0)
    if HAS_FLOOR:
        swa_low = tl.maximum(swa_low, tl.load(floor_ptr + t).to(tl.int32))
    extend_count = tl.maximum(tl.minimum(tpic + 1, pos - swa_low + 1), 0)
    prefix_swa_count = tl.minimum(tl.maximum(cstart - swa_low, 0), win)
    tl.store(extend_len_ptr + t, extend_count)
    if HAS_COMPRESS:
        nc = 0
        for off in tl.range(0, Wc, BLOCK):
            j = off + tl.arange(0, BLOCK)
            m = j < Wc
            j_clamped = tl.minimum(j, Wc - 1)
            pi = tl.load(page_idx_ptr + t * Wc + j_clamped, mask=m, other=-1)
            nc += tl.sum(tl.where(m & (pi >= 0), 1, 0))
        tl.store(prefix_len_ptr + t, prefix_swa_count + nc)
    else:
        tl.store(prefix_len_ptr + t, prefix_swa_count)


@triton.jit
def _build_prefill_indices_kernel(
    positions_ptr,  # [T] int
    chunk_start_ptr,  # [T] int
    cu_q_ptr,  # [T] int
    state_slot_ptr,  # [T] int
    page_idx_ptr,  # [T, Wc] int (front-packed, -1 padded)
    pre_indptr_ptr,  # [T+1] int32 (prefix stream ragged indptr)
    ext_indptr_ptr,  # [T+1] int32 (extend stream ragged indptr)
    pre_out_ptr,
    ext_out_ptr,
    swa_pages,
    ring_stride,  # SWA ring per-slot stride
    floor_ptr,  # [T] int: lowest SWA position the token may see (HAS_FLOOR)
    win: tl.constexpr,
    Wc: tl.constexpr,
    HAS_COMPRESS: tl.constexpr,
    HAS_FLOOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Per token: write extend rows + prefix (SWA ring slots ++ swa_pages+compressed slots) as two ragged segments"""
    t = tl.program_id(0)
    pos = tl.load(positions_ptr + t).to(tl.int32)
    cstart = tl.load(chunk_start_ptr + t).to(tl.int32)
    cuq = tl.load(cu_q_ptr + t).to(tl.int32)
    s = tl.load(state_slot_ptr + t).to(tl.int32)

    tpic = pos - cstart
    swa_low = tl.maximum(pos - win + 1, 0)
    if HAS_FLOOR:
        swa_low = tl.maximum(swa_low, tl.load(floor_ptr + t).to(tl.int32))
    extend_count = tl.maximum(tl.minimum(tpic + 1, pos - swa_low + 1), 0)
    prefix_swa_count = tl.minimum(tl.maximum(cstart - swa_low, 0), win)

    ebase = tl.load(ext_indptr_ptr + t)
    pbase = tl.load(pre_indptr_ptr + t)

    # ---- extend: rows into the current-chunk kv tensor ----
    ext_start = cuq + tpic - extend_count + 1
    for off in tl.range(0, win, BLOCK):
        k = off + tl.arange(0, BLOCK)
        m = k < extend_count
        tl.store(ext_out_ptr + ebase + k, ext_start + k, mask=m)

    # ---- prefix SWA: prior-chunk ring slots (stride = ring_stride) ----
    for off in tl.range(0, win, BLOCK):
        k = off + tl.arange(0, BLOCK)
        m = k < prefix_swa_count
        gp = swa_low + k
        tl.store(pre_out_ptr + pbase + k, s * ring_stride + (gp % ring_stride), mask=m)

    # ---- prefix compressed: swa_pages + front-packed page index ----
    if HAS_COMPRESS:
        nc = tl.load(pre_indptr_ptr + t + 1) - pbase - prefix_swa_count
        cbase = pbase + prefix_swa_count
        for off in tl.range(0, Wc, BLOCK):
            j = off + tl.arange(0, BLOCK)
            m = j < nc
            j_clamped = tl.minimum(j, Wc - 1)
            pi = tl.load(page_idx_ptr + t * Wc + j_clamped, mask=m, other=0).to(
                tl.int32
            )
            tl.store(pre_out_ptr + cbase + j, pi + swa_pages, mask=m)


def build_prefill_indices(
    *,
    compress_ratio: int,
    state_slot: torch.Tensor,  # [T] int (per token)
    positions: torch.Tensor,  # [T] int (per token absolute position)
    chunk_start: torch.Tensor,  # [T] int (absolute start of this chunk for token's seq)
    cu_q: torch.Tensor,  # [T] int (row in extend `kv` of the seq's first chunk token)
    win: int,  # SWA attention window length
    ring_stride: int,  # SWA ring per-slot stride / modulo (win_with_spec)
    swa_pages: int,
    c128_page_indices: Optional[torch.Tensor],
    c4_sparse_page_indices: Optional[torch.Tensor],
    low_ratio_page_indices: Optional[torch.Tensor] = None,
    swa_floor: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build ragged prefill indices: prefix (SWA ring + swa_pages + compressed) into unified_kv + extend into current-chunk kv; returns (prefix_indices, prefix_indptr, extend_indices, extend_indptr).

    ``swa_pages`` is only the compressed-row base: ratios 1 / 2 pass their layer's
    ``get_unified_compress_row_base`` and their top-k in ``low_ratio_page_indices``.
    ``swa_floor`` ([T] absolute positions) floors each token's SWA window, as
    ``swa_replay_start`` does for the paged layout (decoder SWA bounded replay:
    the late layers never wrote ring rows before their tail); compressed rows are
    not floored."""
    device = state_slot.device
    T = state_slot.shape[0]
    assert positions.is_contiguous() and chunk_start.is_contiguous()
    assert cu_q.is_contiguous() and state_slot.is_contiguous()

    if compress_ratio == 0:
        page_idx = None
    elif compress_ratio == 128:
        assert c128_page_indices is not None
        page_idx = c128_page_indices[:T]
    elif compress_ratio == 4:
        assert c4_sparse_page_indices is not None
        page_idx = c4_sparse_page_indices[:T]
    elif compress_ratio in (1, 2):
        assert low_ratio_page_indices is not None
        page_idx = low_ratio_page_indices[:T]
    else:
        raise ValueError(f"bad compress_ratio {compress_ratio}")

    has_compress = page_idx is not None
    if has_compress:
        assert page_idx.is_contiguous()
    Wc = page_idx.shape[1] if has_compress else 0

    block = min(1024, triton.next_power_of_2(max(win, Wc, 1)))
    prefix_len = torch.empty(T, dtype=torch.int32, device=device)
    extend_len = torch.empty(T, dtype=torch.int32, device=device)
    has_floor = swa_floor is not None
    if has_floor:
        assert swa_floor.is_contiguous() and swa_floor.shape[0] >= T
    _prefill_lengths_kernel[(T,)](
        positions,
        chunk_start,
        page_idx if has_compress else positions,  # dummy ptr when no compress
        prefix_len,
        extend_len,
        swa_floor if has_floor else positions,  # dummy ptr when no floor
        win=win,
        Wc=Wc if has_compress else 1,
        HAS_COMPRESS=has_compress,
        HAS_FLOOR=has_floor,
        BLOCK=block,
        num_warps=4,
    )
    kv_indptr_prefix = _lengths_to_indptr(prefix_len)
    kv_indptr_extend = _lengths_to_indptr(extend_len)

    kv_indices_prefix = torch.empty(T * (win + Wc), dtype=torch.int32, device=device)
    kv_indices_extend = torch.empty(T * win, dtype=torch.int32, device=device)

    _build_prefill_indices_kernel[(T,)](
        positions,
        chunk_start,
        cu_q,
        state_slot,
        page_idx if has_compress else state_slot,  # dummy ptr when no compress
        kv_indptr_prefix,
        kv_indptr_extend,
        kv_indices_prefix,
        kv_indices_extend,
        swa_pages,
        ring_stride,
        swa_floor if has_floor else positions,
        win=win,
        Wc=Wc if has_compress else 1,
        HAS_COMPRESS=has_compress,
        HAS_FLOOR=has_floor,
        BLOCK=block,
        num_warps=4,
    )
    return kv_indices_prefix, kv_indptr_prefix, kv_indices_extend, kv_indptr_extend


def prefill_fp8_2buff(
    *,
    q: torch.Tensor,  # [T, H, nope_row_bytes] fp8 packed nope + inline e8m0 scale
    q_rope: torch.Tensor,  # [T, H, rope_dim] bf16
    unified_kv: torch.Tensor,  # [rows, nope_row_bytes] fp8 prefix pool
    unified_kv_rope: torch.Tensor,  # [rows, rope_dim] bf16 prefix pool
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_extend: torch.Tensor,  # [tokens, nope_row_bytes] fp8 packed current chunk
    kv_extend_rope: torch.Tensor,  # [tokens, rope_dim] bf16
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,  # [H] fp32
    softmax_scale: float,
    v_head_dim: int,
) -> torch.Tensor:
    """Prefill over the two-pool fp8 unified_kv, through aiter's opus kernel.

    Same two regions as ``prefill`` -- paged prefix plus this chunk's flat extend
    -- but every latent arrives as a pair, so there are four buffers instead of
    two. The extend pair is the packed K the fused norm+rope store hands back;
    the ring write after attention consumes that same pair, which is why the
    caller materialises it rather than this function quantizing here.

    Unlike ``decode_fp8_2buff`` the scale is a real argument: this kernel takes
    it, so nothing has to match a hardcoded 1/sqrt(512).

    ``v_head_dim`` is an element count (448 nope + 64 rope) that happens to equal
    the packed row's byte width; it comes from the caller so that nothing here
    reads one as the other.

    A token with neither region comes back zero rather than NaN, so unlike
    ``decode_fp8_2buff`` there is nothing to mask off the result afterwards. An
    empty prefix is the live case here, not a guard: chunk 0 has nothing
    committed yet and every token's prefix segment is empty.
    """
    from aiter.ops.pa_sparse_prefill_opus import pa_sparse_prefill_fp8_opus

    T, H, row_bytes = q.shape
    check_two_pool_pair(
        unified_kv,
        unified_kv_rope,
        rope_width=q_rope.shape[-1],
        rope_dtype=q_rope.dtype,
    )
    # The kernel walks the prefix pool and the extend buffer with the same row
    # layout, so a narrower extend row would read the next token's bytes as this
    # one's scales instead of failing.
    assert kv_extend.shape[-1] == row_bytes and kv_extend.dtype == unified_kv.dtype, (
        f"extend nope row is {kv_extend.shape[-1]} x {kv_extend.dtype}, pool is "
        f"{row_bytes} x {unified_kv.dtype}"
    )
    assert (
        kv_extend_rope.shape[-1] == unified_kv_rope.shape[-1]
        and kv_extend_rope.dtype == unified_kv_rope.dtype
    ), (
        f"extend rope row is {kv_extend_rope.shape[-1]} x {kv_extend_rope.dtype}, "
        f"pool is {unified_kv_rope.shape[-1]} x {unified_kv_rope.dtype}"
    )
    assert kv_extend.shape[0] == kv_extend_rope.shape[0], (
        f"extend pair disagrees: nope {kv_extend.shape[0]} rows vs rope "
        f"{kv_extend_rope.shape[0]}"
    )
    assert row_bytes == unified_kv.shape[-1], (
        f"the kernel reads Q with the kv row stride ({unified_kv.shape[-1]} B), "
        f"but the q row is {row_bytes} B"
    )
    assert q_rope.shape[:2] == (T, H), (
        f"q pair disagrees: packed {tuple(q.shape)[:2]} vs rope "
        f"{tuple(q_rope.shape)[:2]}"
    )
    # no stride arguments anywhere in the op. The two pools got their
    # is_contiguous() from check_two_pool_pair above; these are the rest.
    for name, t in (
        ("q", q),
        ("q_rope", q_rope),
        ("kv_extend", kv_extend),
        ("kv_extend_rope", kv_extend_rope),
    ):
        assert t.is_contiguous(), f"{name} must be contiguous, strides {t.stride()}"
    assert attn_sink.dtype == torch.float32 and attn_sink.numel() == H, (
        f"sink must be {H} fp32 values, got {attn_sink.numel()} x {attn_sink.dtype}"
    )
    for name, indptr in (
        ("prefix", kv_indptr_prefix),
        ("extend", kv_indptr_extend),
    ):
        assert indptr.shape[0] >= T + 1, (
            f"{name} indptr holds {indptr.shape[0]} entries, kernel reads {T + 1}"
        )

    out = q_rope.new_empty((T, H, v_head_dim))
    return pa_sparse_prefill_fp8_opus(
        q,
        q_rope,
        unified_kv,
        unified_kv_rope,
        kv_indices_prefix,
        kv_indptr_prefix[: T + 1],
        kv_extend,
        kv_extend_rope,
        kv_indices_extend,
        kv_indptr_extend[: T + 1],
        attn_sink,
        softmax_scale,
        out=out,
    )


def prefill(
    *,
    q: torch.Tensor,  # [T, H, D]
    unified_kv: torch.Tensor,  # [pages, D]
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_extend: torch.Tensor,  # [T, D] current-chunk K (bf16, norm+rope'd)
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    return sparse_attn_v4_paged_prefill(
        q,
        unified_kv,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_extend,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        softmax_scale,
    )
