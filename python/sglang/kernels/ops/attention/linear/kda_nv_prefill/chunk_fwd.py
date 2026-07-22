# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# Vendored from the NVIDIA KDA_prefill package (benchmark/ Blackwell path)
# for the Kimi-K3 chunked prefill forward. Local deltas: fla.* imports
# re-pointed to sglang's vendored fla subset, flat sibling imports made
# package-relative, RCP_LN2 inlined. INTERNAL COLLABORATION ONLY.
# ruff: noqa  -- vendored kernel library, minimal local deltas

"""
KDA Chunk Forward — FLA-Compatible Interface

Optimized implementation of chunk_kda_fwd using CuTe (K1, K2) + Triton (K3) + cuTile (K4).
Signature matches flash-linear-attention/fla/ops/kda/chunk_fwd.py exactly.

Supported:
  - Equal-length sequences (B ≥ 1, T must be multiple of 64)
  - Variable-length sequences (B=1 with cu_seqlens)
  - safe_gate mode (sigmoid + lower_bound) and softplus mode
  - dt_bias
  - use_gate_in_kernel=True with A_log
  - chunk_size=64

Not supported:
  - disable_recompute=True, return_intermediate_states=True
  - cp_context (context parallel)
  - chunk_size != 64
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

RCP_LN2 = 1.4426950408889634  # fla.ops.utils.constant (1/ln2)
from sglang.kernels.ops.attention.fla.index import (
    prepare_chunk_indices,
)

try:
    from .Akk_inverse_lower_triangle_bf16 import akk_inv_host as _akk_inv_host
    from .fuse_k4_only_persistent import BYTES_PER_TENSORMAP as _K4P_BTM
    from .fuse_k4_only_persistent import NUM_TENSORMAPS as _K4P_NTM
    from .fuse_k4_only_persistent import make_host_fn as _k4p_make_host
    from .fuse_kernel123_persistent import make_host_function as _fused_make_host
except ImportError:
    from Akk_inverse_lower_triangle_bf16 import akk_inv_host as _akk_inv_host
    from fuse_k4_only_persistent import BYTES_PER_TENSORMAP as _K4P_BTM
    from fuse_k4_only_persistent import NUM_TENSORMAPS as _K4P_NTM
    from fuse_k4_only_persistent import make_host_fn as _k4p_make_host
    from fuse_kernel123_persistent import make_host_function as _fused_make_host


def _ct(t, etype):
    """Create a CuTe tensor from PyTorch tensor."""
    r = from_dlpack(t, assumed_align=16)
    r.element_type = etype
    return r


# Cached eqlen dummy cu/ci cute wrappers — shared by K123 and akk_inv.
# Avoids per-call torch.empty + from_dlpack overhead (~10-12us each).
_eqlen_dummy_cache = {}


def _get_eqlen_dummies(device, idx_dtype=torch.int64):
    """Returns cached (cu_ct, ci_ct) cute wrappers for eqlen (B+1=2, NT+1=2)."""
    key = (device.index if device.index is not None else 0, idx_dtype)
    if key not in _eqlen_dummy_cache:
        cu_t = torch.empty(2, dtype=idx_dtype, device=device)
        ci_t = torch.empty(1, 2, dtype=idx_dtype, device=device)
        cu_etype = cutlass.Int64 if idx_dtype == torch.int64 else cutlass.Int32
        _eqlen_dummy_cache[key] = (_ct(cu_t, cu_etype), _ct(ci_t, cu_etype))
    return _eqlen_dummy_cache[key]


def _cute_int_type(dtype):
    """Map PyTorch integer dtype to CUTLASS element type."""
    if dtype == torch.int32:
        return cutlass.Int32
    elif dtype == torch.int64:
        return cutlass.Int64
    else:
        raise ValueError(f"Unsupported integer dtype: {dtype}")


# ========== Fused K1+K2+K3 compilation cache ==========
_fused_k123_cache = {}
# id(cu_seqlens) -> bool. Skips per-call GPU->CPU sync on subsequent calls
# when the same cu_seqlens tensor is reused (typical training/inference loop).
_varlen_pure_cache = {}
# id(cu_seqlens) -> int seqlen, populated alongside _varlen_pure_cache for
# single-seq cu_seqlens.
_varlen_single_seqlen_cache = {}

# id(tensor) -> cute_wrapper. The wrappers themselves are stateless views
# over the tensor's storage, so they remain valid as long as the tensor's
# data pointer / shape / strides don't change. Caller is expected to reuse
# the same tensor objects across iterations (typical PyTorch pattern).
_input_wrap_cache = {}


def _tkey(t):
    """Serving-safe cache key for a tensor VIEW: python ids are recycled as
    activation tensors churn every batch, so id-keyed entries go stale and
    hand kernels freed pointers / wrong shapes. (data_ptr, shape, stride,
    dtype) pins exactly what a cute wrapper captures."""
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype)


def _ct_cached(t, etype):
    """`_ct(t, etype)` with id(t)-based cache. Returns the same cute wrapper
    for repeated calls with the same tensor object, avoiding per-call
    `from_dlpack` overhead (~5-10us each)."""
    key = (_tkey(t), etype)
    w = _input_wrap_cache.get(key)
    if w is None:
        w = _ct(t, etype)
        _input_wrap_cache[key] = w
    return w


# Cache for dt_bias `.float().contiguous().view(H, K)` + cute wrapper.
# dt_bias is typically a nn.Parameter — same object across iterations.
_dt_bias_cache = {}


def _get_dt_bias_ct(dt_bias, H, K):
    """Returns cached cute wrapper for dt_bias.float().view(H, K)."""
    key = (_tkey(dt_bias), H, K)
    entry = _dt_bias_cache.get(key)
    if entry is None:
        bias_t = dt_bias.float().contiguous().view(H, K)
        entry = (bias_t, _ct(bias_t, cutlass.Float32))
        _dt_bias_cache[key] = entry
    return entry[1]


# Cache for the empty 1x1 fp32 bias tensor used when dt_bias is None.
_empty_bias_cache = {}


def _get_empty_bias_ct(device):
    idx = device.index if device.index is not None else 0
    if idx not in _empty_bias_cache:
        t = torch.empty(1, 1, dtype=torch.float32, device=device)
        _empty_bias_cache[idx] = _ct(t, cutlass.Float32)
    return _empty_bias_cache[idx]


# K4 varlen cu_seqlens / chunk_offsets cute wrappers. Caches the int32-cast
# tensor and its mark_layout_dynamic wrapper so they survive multiple calls
# with the same input objects.
_k4_varlen_cu_co_cache = {}


def _get_k4_varlen_cu_co(cu_seqlens, chunk_offsets):
    key = (_tkey(cu_seqlens), _tkey(chunk_offsets))
    entry = _k4_varlen_cu_co_cache.get(key)
    if entry is None:
        cu_int32 = (
            cu_seqlens
            if cu_seqlens.dtype == torch.int32
            else cu_seqlens.to(torch.int32)
        )
        cu_int32 = cu_int32.contiguous()
        co_int32 = (
            chunk_offsets
            if chunk_offsets.dtype == torch.int32
            else chunk_offsets.to(torch.int32)
        )
        co_int32 = co_int32.contiguous()
        cu_ct = from_dlpack(cu_int32, assumed_align=4).mark_layout_dynamic()
        cu_ct.element_type = cutlass.Int32
        co_ct = from_dlpack(co_int32, assumed_align=4).mark_layout_dynamic()
        co_ct.element_type = cutlass.Int32
        # Hold refs to the int32 tensors so they don't get GC'd and the
        # underlying storage remain valid as long as cu_ct / co_ct live.
        entry = (cu_int32, co_int32, cu_ct, co_ct)
        _k4_varlen_cu_co_cache[key] = entry
    return entry[2], entry[3]


# Cache K4's reshaped v wrapper keyed by id(v_beta) — the v.reshape(-1, H, V)
# view is a fresh Python object every call but the storage is stable when the
# user reuses the same v tensor (typical benchmark / inference pattern).
_v_ct_cache = {}


# Cache the (cu_for_k4, chunk_offsets_for_k4) pair keyed by id(cu_seqlens).
# Both tensors are computed on-GPU with no host sync — replaces the previous
# `cu_seqlens.cpu().tolist() + Python loop + torch.tensor` chain that forced
# a 50-200us GPU->CPU stall on the K4 prep path every varlen call.
_varlen_k4_input_cache = {}


def _get_varlen_k4_inputs(cu_seqlens, BT):
    # Recomputed every call: the outputs depend on cu_seqlens VALUES, which
    # change batch-to-batch even when the tensor pointer is recycled, so no
    # tensor-identity key is sound. All ops are GPU-side (no host sync).
    cu_int32 = (
        cu_seqlens if cu_seqlens.dtype == torch.int32 else cu_seqlens.to(torch.int32)
    )
    cu_int32 = cu_int32.contiguous()
    seq_lens = cu_int32[1:] - cu_int32[:-1]
    chunk_counts = (seq_lens + (BT - 1)) // BT
    zero = torch.zeros(1, dtype=torch.int32, device=cu_int32.device)
    co_int32 = torch.cat([zero, torch.cumsum(chunk_counts, dim=0).to(torch.int32)])
    co_int32 = co_int32.contiguous()
    return cu_int32, co_int32


# ========== BF16 akk_inv compilation cache ==========
_akk_inv_cache = {}
# ========== K4 persistent (varlen via cu_seqlens) compilation cache ==========
_k4p_cache = {}
_k4p_tm_ws = {}


# Beta absorption (K1 fusion reverted; K123 stays at 375us baseline).
# Two-stage strategy on side stream:
#   1) v*beta runs PARALLEL with K123 (no data dep on K123 outputs)
#   2) k_scaled*beta runs AFTER K123 on side stream, PARALLEL with akk_inv
@torch.compile(fullgraph=True)
def _v_absorb(v, beta):
    return v * beta.unsqueeze(-1).to(v.dtype)


@torch.compile(fullgraph=True)
def _ks_absorb(k_scaled, beta):
    return k_scaled * beta.unsqueeze(-1).to(k_scaled.dtype)


# Side stream cache for v_absorb || K123 overlap.
_side_streams = {}


def _get_side_stream(dev):
    idx = dev.index or 0
    if idx not in _side_streams:
        _side_streams[idx] = torch.cuda.Stream(device=dev)
    return _side_streams[idx]


# ===== Per-section timing (K123 / K4) =====
# Set _TIMING_ENABLED=True to record CUDA-event timings for each call.
# Times accumulate in _TIMING_STATS; use reset_timings() / get_timings() to manage.
_TIMING_ENABLED = False
_TIMING_STATS = {"k123_us": 0.0, "k4_us": 0.0, "count": 0}


def enable_timing(enabled=True):
    """Enable/disable per-section CUDA event timing in chunk_kda_fwd.
    Each call adds CUDA event records around K123 and K4 launches and a sync
    at end of forward, so this slows down execution. Use only for benchmarking."""
    global _TIMING_ENABLED
    _TIMING_ENABLED = enabled


def reset_timings():
    _TIMING_STATS["k123_us"] = 0.0
    _TIMING_STATS["k4_us"] = 0.0
    _TIMING_STATS["count"] = 0


def get_timings():
    """Returns (k123_avg_us, k4_avg_us, n_calls). Avgs are over all calls since last reset."""
    n = max(_TIMING_STATS["count"], 1)
    return (
        _TIMING_STATS["k123_us"] / n,
        _TIMING_STATS["k4_us"] / n,
        _TIMING_STATS["count"],
    )


# Buffer cache: avoid re-allocating ~67us of intermediate tensors per call.
# Also caches cute.Tensor wrappers (saves ~7us each call from from_dlpack).
_buf_cache = {}

# Padded-input scratch cache for the eqlen partial-chunk path. Keyed by
# (B, T_padded, H, K, dtype_qkv, dtype_g, dtype_beta, device, real_T).
# real_T is part of the key so the g sentinel tail [real_T:T_padded] = -1e3
# is set once and reused across calls with the same shape.
_padded_input_cache = {}

# Sentinel-padded g scratch for varlen single-seq Phase 2.1 path. Keyed by
# (B, T_padded, H, K, dtype, device, real_T). The tail [real_T:T_padded] is
# pre-set to -1e3 once at cache init; subsequent calls only overwrite the
# valid prefix [0, real_T).
_g_sentinel_cache = {}


def _get_g_sentinel_buffer(B, T_padded, H, K, dtype_g, device, real_T):
    key = (
        B,
        T_padded,
        H,
        K,
        dtype_g,
        device.index if device.index is not None else 0,
        real_T,
    )
    e = _g_sentinel_cache.get(key)
    if e is None:
        e = torch.zeros(B, T_padded, H, K, dtype=dtype_g, device=device)
        if real_T < T_padded:
            e[:, real_T:] = -1000.0
        _g_sentinel_cache[key] = e
    return e


def _get_padded_input_buffers(
    B, T_padded, H, K, dtype_qkv, dtype_g, dtype_beta, device, real_T
):
    key = (
        B,
        T_padded,
        H,
        K,
        dtype_qkv,
        dtype_g,
        dtype_beta,
        device.index if device.index is not None else 0,
        real_T,
    )
    e = _padded_input_cache.get(key)
    if e is None:
        q_pad = torch.zeros(B, T_padded, H, K, dtype=dtype_qkv, device=device)
        k_pad = torch.zeros(B, T_padded, H, K, dtype=dtype_qkv, device=device)
        v_pad = torch.zeros(B, T_padded, H, K, dtype=dtype_qkv, device=device)
        beta_pad = torch.zeros(B, T_padded, H, dtype=dtype_beta, device=device)
        # g: zero in [0, real_T), sentinel -1e3 in [real_T, T_padded). Caller's
        # data overwrites the prefix each call; the sentinel tail never moves.
        g_pad = torch.zeros(B, T_padded, H, K, dtype=dtype_g, device=device)
        if real_T < T_padded:
            g_pad[:, real_T:] = -1000.0
        e = (q_pad, k_pad, v_pad, g_pad, beta_pad)
        _padded_input_cache[key] = e
    return e


# Multi-seq varlen repack cache for the Phase 2.2 path. Keyed by id(cu_seqlens).
# Stores: (orig_seq_lens, padded_seq_lens, new_cu_seqlens_tensor,
#          new_chunk_indices_tensor, new_T_total, padded_input_buffers).
# All tensors are GPU-side and pre-allocated at cache build time. Per-call the
# kernel reads from / writes to these buffers; we copy caller's input slices in
# and output slices back (only the valid prefix of each seq).
_multiseq_repack_cache = {}

_caller_layout_O_cache = {}


def _get_caller_layout_O_buffer(multiseq_info, dtype, V_dim, device):
    """Per-shape cached output buffer at caller's contiguous layout (sum of
    seq_lens, no padding gaps). Filled by per-seq copies from K4's padded O."""
    caller_T = multiseq_info["caller_T"]
    H_x_V = multiseq_info.get("_H_V")  # not strictly needed since we fix B=1 H known
    key = (caller_T, V_dim, dtype, device.index if device.index is not None else 0)
    e = _caller_layout_O_cache.get(key)
    if e is None:
        # B=1 enforced upstream when multiseq_info is built.
        e = torch.empty(
            1,
            caller_T,
            multiseq_info["q_pad"].shape[2],
            V_dim,
            dtype=dtype,
            device=device,
        )
        _caller_layout_O_cache[key] = e
    return e


def _get_multiseq_repack_info(cu_seqlens, q, k, v, g, beta, BT, device):
    """Build (and cache) the padded layout for multi-seq varlen with non-aligned
    seqs. Returns None if all seqs are already 64-aligned (caller can use the
    existing varlen_pure path)."""
    import weakref

    key = id(cu_seqlens)
    cached = _multiseq_repack_cache.get(key)
    if cached is not None:
        wref, e = cached
        if wref() is cu_seqlens:
            return e
        # id collision after GC: rebuild
        del _multiseq_repack_cache[key]
    cu_cpu = cu_seqlens.cpu().tolist()
    seq_lens = [cu_cpu[i + 1] - cu_cpu[i] for i in range(len(cu_cpu) - 1)]
    if all(sl % BT == 0 for sl in seq_lens):
        _multiseq_repack_cache[key] = (weakref.ref(cu_seqlens), None)
        return None
    padded_lens = [((sl + BT - 1) // BT) * BT for sl in seq_lens]
    new_cu = [0]
    for pl in padded_lens:
        new_cu.append(new_cu[-1] + pl)
    new_T_total = new_cu[-1]
    B = q.shape[0]
    H = q.shape[2]
    K = q.shape[3]
    # Pre-allocated padded input buffers. q/k/v/beta tail = 0 (zero MMA), g
    # tail = -1e3 sentinel (zero gate activation). The PER-SEQ tail regions
    # are between (new_cu[i] + seq_lens[i], new_cu[i+1]) — pre-fill once.
    q_pad = torch.zeros(B, new_T_total, H, K, dtype=q.dtype, device=device)
    k_pad = torch.zeros_like(q_pad)
    v_pad = torch.zeros(B, new_T_total, H, v.shape[3], dtype=v.dtype, device=device)
    beta_pad = torch.zeros(B, new_T_total, H, dtype=beta.dtype, device=device)
    g_pad = torch.zeros(B, new_T_total, H, K, dtype=g.dtype, device=device)
    for i, (sl, pl) in enumerate(zip(seq_lens, padded_lens)):
        if sl < pl:
            tail_start = new_cu[i] + sl
            tail_end = new_cu[i + 1]
            g_pad[:, tail_start:tail_end] = -1000.0
    new_cu_tensor = torch.tensor(new_cu, dtype=cu_seqlens.dtype, device=device)
    new_chunk_indices = prepare_chunk_indices(new_cu_tensor, BT)
    # Build index map: dst_indices[i] = position in padded layout where orig
    # row i lives. Used by index_copy_ to do the scatter in one op (instead of
    # N_seqs × 5 separate slice copies, which cost ~5us each in Python).
    T_total_orig = cu_cpu[-1]
    dst_indices_list = []
    for i, sl in enumerate(seq_lens):
        for j in range(sl):
            dst_indices_list.append(new_cu[i] + j)
    dst_indices = torch.tensor(dst_indices_list, dtype=torch.long, device=device)
    # Mark this cu_seqlens as VARLEN_PURE eligible — every seq in the new
    # layout is 64-aligned by construction.
    _varlen_pure_cache[id(new_cu_tensor)] = True
    e = {
        "seq_lens": seq_lens,
        "padded_lens": padded_lens,
        "new_cu": new_cu,
        "new_T_total": new_T_total,
        "new_cu_tensor": new_cu_tensor,
        "new_chunk_indices": new_chunk_indices,
        "q_pad": q_pad,
        "k_pad": k_pad,
        "v_pad": v_pad,
        "g_pad": g_pad,
        "beta_pad": beta_pad,
        "dst_indices": dst_indices,
        "T_total_orig": T_total_orig,
    }
    _multiseq_repack_cache[key] = (weakref.ref(cu_seqlens), e)
    return e


def _get_buffers(dev, dtype_k, B, T, H, K_dim, V_dim, NT, N_seqs, BT):
    """All beta fusion lives in akk_inv kernel epilogue (post-inv column-scale)."""
    key = (dev.index or 0, B, T, H, K_dim, V_dim, NT, N_seqs)
    if key not in _buf_cache:
        bf16 = cutlass.BFloat16
        fp32 = cutlass.Float32
        k_scaled = torch.empty(
            B, T, H, K_dim, device=dev, dtype=dtype_k
        )  # raw, no beta
        kg = torch.empty(B, T, H, K_dim, device=dev, dtype=dtype_k)
        q_scaled = torch.empty(B, T, H, K_dim, device=dev, dtype=dtype_k)
        gk_last_exp = torch.empty(B, NT, H, K_dim, device=dev, dtype=torch.float32)
        A_qk = torch.zeros(B, T, H, BT, device=dev, dtype=dtype_k)
        A_kk = torch.zeros(B, T, H, BT, device=dev, dtype=dtype_k)
        O_flat = torch.empty(B, T, H, V_dim, device=dev, dtype=dtype_k)
        # K4 reads initial state from S_out (caller copies it in) and writes
        # the final state back into the same buffer — no separate s_4d, no
        # extra D2D memcpy at the end of the K4 launcher. Layout matches
        # caller's [N_seqs, H, K, V] contig fp32 convention.
        S_out = torch.empty(N_seqs, H, K_dim, V_dim, device=dev, dtype=torch.float32)
        cu_eqlen = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=dev)
        co_eqlen = torch.arange(
            0, (B + 1) * (T // BT), T // BT, dtype=torch.int32, device=dev
        )

        T_total = B * T
        A_kk_flat = A_kk.reshape(T_total, H, BT)
        A_qk_flat = A_qk.reshape(T_total, H, BT)
        KS_flat = k_scaled.reshape(T_total, H, K_dim)
        QS_flat = q_scaled.reshape(T_total, H, K_dim)
        KG_flat = kg.reshape(T_total, H, K_dim)
        O_token = O_flat.reshape(T_total, H, V_dim)
        gk_flat = gk_last_exp.reshape(-1, H, K_dim)

        def _wrap(t, etype):
            r = from_dlpack(t, assumed_align=16).mark_layout_dynamic()
            r.element_type = etype
            return r

        a_ct = _wrap(A_kk_flat, bf16)
        aqc_ct = _wrap(A_qk_flat, bf16)
        ks_ct = _wrap(KS_flat, bf16)  # raw k_scaled (beta absorbed in akk_inv)
        qs_ct = _wrap(QS_flat, bf16)
        kg_ct = _wrap(KG_flat, bf16)
        o_ct = _wrap(O_token, bf16)
        gk_ct = _wrap(gk_flat, fp32)
        cu_eqlen_ct = from_dlpack(cu_eqlen, assumed_align=4).mark_layout_dynamic()
        cu_eqlen_ct.element_type = cutlass.Int32
        co_eqlen_ct = from_dlpack(co_eqlen, assumed_align=4).mark_layout_dynamic()
        co_eqlen_ct.element_type = cutlass.Int32

        # akk_inv views: bf16 storage reinterpreted as fp32 (packed 2x bf16 -> 1x fp32).
        # Built without mark_layout_dynamic to match per-call wrapper type signature
        # (akk_inv kernel was compiled against this layout — must stay identical).
        akk_in_view = from_dlpack(A_kk, assumed_align=16)
        akk_in_view.element_type = fp32
        akk_out_view = from_dlpack(A_kk, assumed_align=16)
        akk_out_view.element_type = fp32

        # K4 state wrapper points directly at S_out — K4 reads/writes in place.
        # Plain from_dlpack (no mark_layout_dynamic) keeps the type signature
        # stable so the kernel only compiles once.
        s_ct = from_dlpack(S_out, assumed_align=16)
        s_ct.element_type = fp32

        # Cache PyTorch streams once per buf_cache entry — torch.cuda.current_stream
        # is a torch C call that costs ~2us each invocation.
        main_stream_cached = torch.cuda.current_stream(dev)
        side_stream_cached = _get_side_stream(dev)

        cute_wrappers = dict(
            a_ct=a_ct,
            aqc_ct=aqc_ct,
            ks_ct=ks_ct,
            qs_ct=qs_ct,
            kg_ct=kg_ct,
            o_ct=o_ct,
            gk_ct=gk_ct,
            cu_eqlen_ct=cu_eqlen_ct,
            co_eqlen_ct=co_eqlen_ct,
            akk_in_view=akk_in_view,
            akk_out_view=akk_out_view,
            s_ct=s_ct,
            main_stream=main_stream_cached,
            side_stream=side_stream_cached,
            # Filled lazily on first launch — saves cache_key tuple build +
            # outer dict lookup on subsequent calls.
            _k123_fns={},
            _akk_inv_fn=None,
            _k4_fn=None,
        )

        _buf_cache[key] = (
            k_scaled,
            kg,
            q_scaled,
            gk_last_exp,
            A_qk,
            A_kk,
            O_flat,
            S_out,
            cu_eqlen,
            co_eqlen,
            cute_wrappers,
        )
    return _buf_cache[key]


def _launch_k4_persistent(
    cute_wrappers,
    v_beta,
    S_in,
    S_out,
    cu_seqlens,
    chunk_offsets,
    cu_eqlen_passed=False,
    num_sm=148,
    H=None,
    V_dim=None,
    use_fast_sync=False,
):
    """Persistent K4 with cached cute wrappers.
    k_scaled (cached, beta-absorbed by K1 fusion) reads from cute_wrappers['ks_ct'].
    v_beta: fresh per-call tensor (torch.compile output of v*beta on side stream)."""
    # Launch on the CALLER's current stream (resolved per call, passed as a
    # runtime arg): with sglang's overlap scheduler the forward runs on a
    # non-default stream and a default-stream K4 races the whole pipeline.
    launch_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    # Fast path: cache the pre-built (k4_fn, args_tuple) keyed by
    # (id(v_beta), id(cu_seqlens), id(S_in), id(S_out)). The state-copy is
    # done by the caller (chunk_kda_fwd) on a side stream to overlap with
    # K123 — this launcher just runs K4.
    fast_key = (
        _tkey(v_beta),
        0 if cu_eqlen_passed else _tkey(cu_seqlens),
        _tkey(S_in),
        _tkey(S_out),
    )
    fast_cache = cute_wrappers.get("_k4_fast_cache")
    if fast_cache is None:
        fast_cache = {}
        cute_wrappers["_k4_fast_cache"] = fast_cache
    fast_entry = fast_cache.get(fast_key)
    if fast_entry is not None:
        k4_fn, args = fast_entry
        k4_fn(*args, launch_stream)
        return

    bf16 = cutlass.BFloat16
    fp32 = cutlass.Float32
    N_seqs = cu_seqlens.shape[0] - 1
    BH = N_seqs * H
    nsm = min(BH, num_sm)
    dev = v_beta.device
    dev_idx = dev.index or 0

    s_ct = cute_wrappers["s_ct"]
    a_ct = cute_wrappers["a_ct"]
    b_ct = cute_wrappers["ks_ct"]  # raw k_scaled (beta absorbed in akk_inv)
    q_ct = cute_wrappers["qs_ct"]
    aqc_ct = cute_wrappers["aqc_ct"]
    kg_ct = cute_wrappers["kg_ct"]
    o_ct = cute_wrappers["o_ct"]
    gk_ct = cute_wrappers["gk_ct"]

    v_key = _tkey(v_beta)
    v_entry = _v_ct_cache.get(v_key)
    if v_entry is None:
        v_view = v_beta.reshape(-1, H, V_dim) if v_beta.dim() == 4 else v_beta
        v_ct = from_dlpack(v_view, assumed_align=16).mark_layout_dynamic()
        v_ct.element_type = bf16
        _v_ct_cache[v_key] = (v_view, v_ct)
    else:
        v_ct = v_entry[1]

    if cu_eqlen_passed:
        cu_ct = cute_wrappers["cu_eqlen_ct"]
        co_ct = cute_wrappers["co_eqlen_ct"]
    else:
        cu_ct, co_ct = _get_k4_varlen_cu_co(cu_seqlens, chunk_offsets)

    tm_key = (dev_idx, nsm)
    if tm_key not in _k4p_tm_ws:
        tm_ws_t = torch.zeros(nsm * _K4P_NTM * _K4P_BTM, dtype=torch.uint8, device=dev)
        tm_ct = from_dlpack(tm_ws_t, assumed_align=16)
        tm_ct.element_type = cutlass.Uint8
        _k4p_tm_ws[tm_key] = (tm_ws_t, tm_ct)
    else:
        tm_ws_t, tm_ct = _k4p_tm_ws[tm_key]

    k4_fn = cute_wrappers.get("_k4_fn")
    if k4_fn is None:
        cache_key = (dev_idx, nsm, N_seqs, H)
        k4_fn = _k4p_cache.get(cache_key)
        if k4_fn is None:
            host_fn = _k4p_make_host(num_sm=nsm)
            k4_fn = cute.compile(
                host_fn,
                a_ct,
                b_ct,
                v_ct,
                q_ct,
                aqc_ct,
                kg_ct,
                o_ct,
                gk_ct,
                s_ct,
                cu_ct,
                co_ct,
                tm_ct,
                launch_stream,
            )
            _k4p_cache[cache_key] = k4_fn
        cute_wrappers["_k4_fn"] = k4_fn

    args = (
        a_ct,
        b_ct,
        v_ct,
        q_ct,
        aqc_ct,
        kg_ct,
        o_ct,
        gk_ct,
        s_ct,
        cu_ct,
        co_ct,
        tm_ct,
    )
    fast_cache[fast_key] = (k4_fn, args)
    k4_fn(*args, launch_stream)


def _launch_fused_k123_inv(
    q,
    k,
    g,
    A_log,
    beta,
    scale,
    k_scaled,
    kg,
    q_scaled,
    gk_last_exp,
    A_qk,
    A_kk_inv,
    cu_seqlens,
    chunk_indices,
    is_varlen,
    NT,
    dt_bias=None,
    safe_gate=False,
    lower_bound=None,
    akk_in_view=None,
    akk_out_view=None,
    cute_wrappers=None,
    varlen_pure_override=None,
):
    """Persistent K1+K2 (writes A_kk in I+L format with diag=1) chained with
    BF16 akk_inv (in-place inversion). Final A_kk_inv = (I+L)^-1."""

    # Resolve the CALLER's stream every call: under sglang's overlap scheduler
    # the forward runs on a non-default stream, and launching these kernels on
    # the default stream makes the whole pipeline a cross-stream data race
    # (torn states in the mamba pool, garbage outputs). Streams are passed as
    # runtime launch args, so the compiled fns stay cached.
    launch_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # Fast path: when (q, k, g, beta, A_log, dt_bias, cu_seqlens, chunk_indices)
    # are stable across calls (typical benchmark / inference), skip the per-call
    # wrapper gathering and launch with a pre-built (k123_fn, args) tuple.
    if cute_wrappers is not None:
        fast_key = (
            _tkey(q),
            _tkey(k),
            _tkey(g),
            _tkey(beta),
            _tkey(A_log),
            _tkey(dt_bias) if dt_bias is not None else 0,
            _tkey(cu_seqlens) if cu_seqlens is not None else 0,
            _tkey(chunk_indices) if chunk_indices is not None else 0,
            bool(safe_gate),
            float(lower_bound) if lower_bound is not None else 0.0,
        )
        fast_cache = cute_wrappers.setdefault("_k123_fast_cache", {})
        fast_entry = fast_cache.get(fast_key)
        if fast_entry is not None:
            k123_fn, k123_args, akk_fn, akk_args = fast_entry
            k123_fn(*k123_args, launch_stream)
            akk_fn(*akk_args, launch_stream)
            return

    B, T, H, K = q.shape
    BT = 64
    dev = q.device.index or 0
    T_padded = T if is_varlen else None
    has_bias = dt_bias is not None

    # Auto-detect VARLEN_PURE eligibility, cached by id(cu_seqlens).
    # First call with a given cu_seqlens object pays one GPU->CPU sync; later
    # calls with the same tensor object are a dict lookup (~100 ns).
    varlen_pure = False
    if varlen_pure_override is not None:
        # Serving path: the caller knows the seq lengths (CPU-side, no sync)
        # and the id-keyed detection below is unsafe under tensor-id reuse.
        varlen_pure = bool(varlen_pure_override) if is_varlen else False
    elif is_varlen and cu_seqlens is not None:
        _vp_key = id(cu_seqlens)
        if _vp_key not in _varlen_pure_cache:
            cu_cpu = cu_seqlens.cpu().tolist()
            seq_lens = [cu_cpu[i + 1] - cu_cpu[i] for i in range(len(cu_cpu) - 1)]
            _varlen_pure_cache[_vp_key] = all((sl % BT) == 0 for sl in seq_lens)
        varlen_pure = _varlen_pure_cache[_vp_key]
    cache_key = (B, NT, H, is_varlen, T_padded, dev, has_bias, safe_gate, varlen_pure)

    # Inputs are guaranteed contiguous by upstream linear projections.
    # A_log is fp32 model param; .float() is no-op when dtype already matches.
    q_ct = _ct_cached(q, cutlass.BFloat16)
    k_ct = _ct_cached(k, cutlass.BFloat16)
    g_ct = _ct_cached(g, cutlass.BFloat16)
    alog_ct = _ct_cached(
        A_log if A_log.dtype == torch.float32 else A_log.float(), cutlass.Float32
    )
    beta_ct = _ct_cached(beta, cutlass.BFloat16)

    ks_ct = _ct_cached(k_scaled, cutlass.BFloat16)
    kg_ct = _ct_cached(kg, cutlass.BFloat16)
    qs_ct = _ct_cached(q_scaled, cutlass.BFloat16)
    gk_ct = _ct_cached(gk_last_exp, cutlass.Float32)
    aqk_ct = _ct_cached(A_qk, cutlass.BFloat16)
    akk_ct = _ct_cached(A_kk_inv, cutlass.BFloat16)

    if is_varlen:
        cu_ct = _ct_cached(cu_seqlens, _cute_int_type(cu_seqlens.dtype))
        ci_ct = _ct_cached(chunk_indices, _cute_int_type(chunk_indices.dtype))
    else:
        cu_ct, ci_ct = _get_eqlen_dummies(q.device, torch.int64)

    if dt_bias is not None:
        bias_ct = _get_dt_bias_ct(dt_bias, H, K)
    else:
        bias_ct = _get_empty_bias_ct(q.device)
    lb_val = float(lower_bound) if lower_bound is not None else 0.0

    ct_args = (
        q_ct,
        k_ct,
        g_ct,
        alog_ct,
        beta_ct,
        scale,
        ks_ct,
        kg_ct,
        qs_ct,
        gk_ct,
        aqk_ct,
        akk_ct,
        cu_ct,
        ci_ct,
        bias_ct,
        lb_val,
    )

    if cache_key not in _fused_k123_cache:
        host_fn = _fused_make_host(
            B,
            NT,
            H,
            is_varlen=is_varlen,
            T_padded=T_padded,
            has_bias=has_bias,
            use_safe_gate=safe_gate,
            varlen_pure=varlen_pure,
        )
        _fused_k123_cache[cache_key] = cute.compile(host_fn, *ct_args, launch_stream)
    k123_fn = _fused_k123_cache[cache_key]
    k123_fn(*ct_args, launch_stream)

    # ===== Chained BF16 akk_inv (in-place: A_kk_inv = (I+L)^-1) =====
    if akk_in_view is None:
        akk_in_view = from_dlpack(A_kk_inv, assumed_align=16)
        akk_in_view.element_type = cutlass.Float32
        akk_out_view = from_dlpack(A_kk_inv, assumed_align=16)
        akk_out_view.element_type = cutlass.Float32

    if is_varlen:
        # Reuse the cached cute wrappers from K123 (same tensor objects).
        akk_cu_ct = cu_ct
        akk_ci_ct = ci_ct
        is_varlen_int = 1
        T_val = T
    else:
        akk_cu_ct, akk_ci_ct = cu_ct, ci_ct
        is_varlen_int = 0
        T_val = NT * BT

    akk_cache_key = (B, NT, H, is_varlen, dev, T_val)
    if akk_cache_key not in _akk_inv_cache:
        _akk_inv_cache[akk_cache_key] = cute.compile(
            _akk_inv_host,
            akk_in_view,
            akk_out_view,
            beta_ct,
            B,
            NT,
            H,
            akk_cu_ct,
            akk_ci_ct,
            is_varlen_int,
            T_val,
            launch_stream,
        )
    akk_fn = _akk_inv_cache[akk_cache_key]
    akk_args = (akk_in_view, akk_out_view, beta_ct, akk_cu_ct, akk_ci_ct)
    akk_fn(*akk_args, launch_stream)

    # Populate the fast cache so subsequent calls with the same input ids skip
    # all wrapper gathering above.
    if cute_wrappers is not None:
        cute_wrappers["_k123_fast_cache"][fast_key] = (
            k123_fn,
            ct_args,
            akk_fn,
            akk_args,
        )


def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    cp_context=None,
    varlen_single_real_T: int | None = None,
    varlen_pure: bool | None = None,
):
    """KDA forward — optimized, FLA-compatible interface."""
    if safe_gate and lower_bound is None:
        lower_bound = -5.0

    is_varlen = cu_seqlens is not None

    B, T, H, K = q.shape
    V_dim = v.shape[-1]
    device = q.device
    BT = 64

    # Phase 1: handle eqlen with T % 64 != 0 entirely on the eqlen path —
    # never borrow varlen's mask code. Pad inputs to a CHUNKS_PER_BLOCK*BT
    # = 256 multiple so the persistent scheduler's `cgs_per_head = NT // 4`
    # divides cleanly and every chunk has 64 valid rows of (zero-padded /
    # sentinel-padded) data. K123 eqlen kernel runs unchanged — it doesn't
    # know partial chunks exist, the boundary is handled at the data
    # boundary (K4's bounded-TMA principle, just expressed via host pad
    # since K123 stores via autovec_copy not TMA).
    #
    # Varlen with non-aligned seq lengths is a separate problem (Phase 2):
    # multi-seq varlen can't be host-padded without repacking memory.
    real_T = T
    needs_eqlen_pad = (not is_varlen) and (T % BT != 0)
    if needs_eqlen_pad:
        if B != 1:
            raise NotImplementedError(
                f"eqlen with B>1 and T % {BT} != 0 not supported "
                f"(got B={B}, T={T})."
            )
        CPB_BT = 4 * BT  # CHUNKS_PER_BLOCK * BT, the cgs_per_head divisibility unit
        T_padded = ((T + CPB_BT - 1) // CPB_BT) * CPB_BT
        # Pre-allocated padded scratch buffers (per (B,T_padded,H,K,dtype) cache
        # key). torch.cat would reallocate + copy the full 200MB q tensor every
        # call — caching the destination buffer drops that to a single slice
        # copy of the valid prefix (caller already lives in our buffer for
        # subsequent calls reusing the same id, but we re-copy unconditionally
        # since the caller may have updated the data in-place).
        q_pad, k_pad, v_pad, g_pad, beta_pad = _get_padded_input_buffers(
            B, T_padded, H, K, q.dtype, g.dtype, beta.dtype, q.device, real_T
        )
        # q/k/v/beta zero-padded → K1/K2 MMAs naturally produce 0 for OOB rows.
        # Tail [real_T:] of q_pad/k_pad/v_pad/beta_pad is pre-zeroed at cache
        # init and never written, so we only copy the valid prefix.
        q_pad[:, :real_T].copy_(q)
        k_pad[:, :real_T].copy_(k)
        v_pad[:, :real_T].copy_(v)
        beta_pad[:, :real_T].copy_(beta)
        # g uses a -1e3 sentinel so the gate activation saturates to 0 for OOB
        # rows (both safe_gate sigmoid and softplus paths). Plain g=0 gives
        # nonzero activation that would corrupt the cumsum past seq end. The
        # tail is set to -1e3 once at cache init; we only copy the valid prefix.
        g_pad[:, :real_T].copy_(g)
        q, k, v, g, beta = q_pad, k_pad, v_pad, g_pad, beta_pad
        T = T_padded  # downstream buffer alloc + kernel layout use T_padded

    # Phase 2.1: varlen with a SINGLE non-aligned sequence — caller already
    # zero-padded q/k/v/beta to a 64-multiple (FLA convention), but g's tail
    # is also zero, which causes the gate activation to be non-zero past seq
    # end and corrupts the cumsum / GkLast. We sentinel-pad g (cheap: ~5MB
    # copy) and force VARLEN_PURE=1 so all 4 mask sites compile-elide. Same
    # K4 "boundary at the data" principle as the eqlen path.
    #
    # Multi-seq varlen is NOT handled here — its OOB regions overlap with
    # adjacent seqs' data, so sentinel-pad on g would corrupt the next seq.
    # Multi-seq optimization needs per-seq dynamic tensormap (Phase 2.2).
    needs_varlen_single_pad = False
    if (
        varlen_single_real_T is not None
        and is_varlen
        and cu_seqlens is not None
        and cu_seqlens.shape[0] == 2
        and B == 1
    ):
        # Caller-supplied real length (serving path): skips the id-keyed
        # alignment cache below, which is unsafe under tensor-id reuse when
        # cu_seqlens churns every batch.
        if varlen_single_real_T % BT != 0:
            real_T = varlen_single_real_T
            needs_varlen_single_pad = True
    elif is_varlen and cu_seqlens is not None and cu_seqlens.shape[0] == 2 and B == 1:
        _vl_key = id(cu_seqlens)
        if _vl_key not in _varlen_pure_cache:
            cu_cpu = cu_seqlens.cpu().tolist()
            sl = cu_cpu[1] - cu_cpu[0]
            _varlen_pure_cache[_vl_key] = sl % BT == 0
            _varlen_single_seqlen_cache[_vl_key] = sl
        if not _varlen_pure_cache[_vl_key]:
            real_T = _varlen_single_seqlen_cache[_vl_key]
            needs_varlen_single_pad = True
    if needs_varlen_single_pad:
        # q/k/v/beta already zero-padded by caller (FLA convention). Re-build g
        # with -1000 sentinel in the tail so VARLEN_PURE=1 path is correct.
        # Cache the resulting g buffer so repeated calls with same input ids
        # don't re-allocate.
        cur_T = q.shape[1]  # caller's padded T
        g_pad = _get_g_sentinel_buffer(B, cur_T, H, K, g.dtype, g.device, real_T)
        g_pad[:, :real_T].copy_(g[:, :real_T])
        g = g_pad
        # Force VARLEN_PURE=1 — caller's cu_seqlens is reused as-is.
        _varlen_pure_cache[id(cu_seqlens)] = True

    # Phase 2.2 (multi-seq via host repack) was attempted but isn't net positive:
    # the scatter/gather memcpy cost (~800us GPU bandwidth per call) exceeds
    # the kernel mask-elision savings (~250us). Keeping multi-seq non-pure on
    # the original masked path. The right fix is kernel-level dynamic
    # tensormap (K4-style per-tile bounded TMA) but that requires a major
    # K123 kernel refactor — left as future work.
    multiseq_info = None

    if is_varlen:
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        NT = len(chunk_indices)
        N_seqs = len(cu_seqlens) - 1
    else:
        NT = T // BT
        N_seqs = B

    # ===== Cached buffers + cute wrappers (avoid alloc + from_dlpack overhead per call) =====
    (
        k_scaled,
        kg,
        q_scaled,
        gk_last_exp,
        A_qk,
        A_kk,
        O_flat,
        S_out,
        cu_eqlen,
        co_eqlen,
        cute_wrappers,
    ) = _get_buffers(device, k.dtype, B, T, H, K, V_dim, NT, N_seqs, BT)

    # ===== State copy on side stream, parallel with K123 =====
    # K4 needs S_out populated with initial_state. By doing this copy on a
    # side stream BEFORE K123 launches, the D2D memcpy overlaps with K123's
    # compute. Especially big win for high-N_seqs varlen where state is huge
    # (192MB for N=32, ~64us memcpy) — would otherwise serialize before K4.
    if initial_state is None:
        S_in = torch.zeros(N_seqs, H, K, V_dim, dtype=torch.float32, device=device)
    else:
        S_in = initial_state
    # Serving hardening: the state copy runs on the CALLER's stream, same as
    # every kernel in the pipeline. The old side-stream overlap (copy || K123)
    # only pays off for huge multi-seq varlen states (~192MB at N=32); for
    # serving single-seq engagements the state is <1MB and the side stream
    # was the last remaining cross-stream surface in this path.
    main_stream = torch.cuda.current_stream(device)
    needs_copy = S_in.data_ptr() != S_out.data_ptr()
    if needs_copy:
        S_out.copy_(S_in)

    # Beta is fused entirely in akk_inv kernel epilogue (post-inv column-scale).
    # No host v*beta and no K1 k_scaled*beta any more.
    if _TIMING_ENABLED:
        k123_s = torch.cuda.Event(enable_timing=True)
        k123_e = torch.cuda.Event(enable_timing=True)
        k123_s.record(stream=main_stream)

    _launch_fused_k123_inv(
        q,
        k,
        g,
        A_log,
        beta,
        scale,
        k_scaled,
        kg,
        q_scaled,
        gk_last_exp,
        A_qk,
        A_kk,
        cu_seqlens,
        chunk_indices,
        is_varlen,
        NT,
        dt_bias=dt_bias,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        akk_in_view=cute_wrappers["akk_in_view"],
        akk_out_view=cute_wrappers["akk_out_view"],
        cute_wrappers=cute_wrappers,
        varlen_pure_override=(True if needs_varlen_single_pad else varlen_pure),
    )

    if _TIMING_ENABLED:
        k123_e.record(stream=main_stream)

    # ===== K4: persistent kernel (eqlen + varlen via cu_seqlens) =====
    if is_varlen:
        # GPU-side cumsum + cache by id(cu_seqlens). No host sync.
        cu_for_k4, chunk_offsets_for_k4 = _get_varlen_k4_inputs(cu_seqlens, BT)
    else:
        cu_for_k4 = cu_eqlen
        chunk_offsets_for_k4 = co_eqlen

    # State copy now runs on the caller's stream; K4 is stream-ordered after it.

    if _TIMING_ENABLED:
        k4_s = torch.cuda.Event(enable_timing=True)
        k4_e = torch.cuda.Event(enable_timing=True)
        k4_s.record(stream=main_stream)

    _launch_k4_persistent(
        cute_wrappers,
        v,
        S_in,
        S_out,
        cu_for_k4,
        chunk_offsets_for_k4,
        cu_eqlen_passed=(not is_varlen),
        H=H,
        V_dim=V_dim,
        use_fast_sync=(not is_varlen),
    )

    if _TIMING_ENABLED:
        k4_e.record(stream=cute_wrappers["main_stream"])
        torch.cuda.synchronize(device)
        _TIMING_STATS["k123_us"] += k123_s.elapsed_time(k123_e) * 1000.0
        _TIMING_STATS["k4_us"] += k4_s.elapsed_time(k4_e) * 1000.0
        _TIMING_STATS["count"] += 1

    o = O_flat
    A_qk_out = A_qk
    A_kk_out = A_kk
    if needs_eqlen_pad:
        # Caller called with original T = real_T; their downstream code expects
        # outputs at that shape. Slice the padded scratch tail back off.
        o = o[:, :real_T]
        A_qk_out = A_qk_out[:, :real_T]
        A_kk_out = A_kk_out[:, :real_T]
    # multiseq_info is always None here (Phase 2.2 disabled — see above)
    final_state = S_out if output_final_state else None

    return (
        o,
        final_state,
        None,
        A_qk_out,
        A_kk_out,
        None,
        None,
        None,
        None,
        None,
        None,
        initial_state,
    )
