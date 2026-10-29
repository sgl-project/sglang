# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from TensorRT-LLM 4d09932ca7 (Apache-2.0): tensorrt_llm/_torch/attention_backend/fmha/q35_cp_fused.py;
# switches are constants here (no q35_env), the debug ring is dropped, the kernel import points at this package.
"""Host side of the Qwen3.5 fused CP-lite partial decode attention (CuTe DSL kernel
``cute_dsl_kernels/blackwell/q35_cp_partial_attn.py``), used by q35_cp.q35_cp_decode_ll under
TRTLLM_Q35_ATTN_CP_FUSED=1 (default off).

One launch per call replaces the merge-free path's prep + trtllm-gen FMHA + LSE (+ own-head draft) launches: this CP
rank's partial attention over the pages it owns (page p belongs to rank p % cp_size) for every q row of the KV group,
written straight into alltoall_merge's input layout. The contract is q35_cp.q35_cp_fused_partial_ref.

The module owns the grid-barrier counters and the partial workspace of each device; q35_cp_fused_prepare() allocates
them (the first call does it otherwise, which must then not be inside a CUDA graph capture).
"""

import functools
import logging
from typing import Dict, Optional, Union

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

PAGE = 64
HEAD_DIM = 256
MAX_ROWS = 128


# Largest batch of one call (TRT's final TEP8 recipe: CP_FUSED_MAX_BATCH=32, CP_FUSED_MAX_BATCH_Q1=32).
MAX_BATCH = 32
MAX_BATCH_Q1 = 32
# Balanced split (TRTLLM_Q35_ATTN_CP_BALANCED_SPLIT): SGLANG_PLB_CP_BALANCED, default off.
_BALANCED = envs.SGLANG_PLB_CP_BALANCED.get()


# Mirror of q35_cp_partial_attn.MAX_COMB_ROWS (rows one CTA's combine chunk may touch).
_MAX_COMB_ROWS = 32
LOG2E = 1.4426950408889634

# Prefix pages below the new tokens' first page that are read only after griddepcontrol.wait (MTP passes write the
# positions just below the new tokens earlier in the step): 1, as in TRTLLM_Q35_ATTN_FUSED_FRESH_MARGIN.
_FRESH_MARGIN = 1
# Request the prefix KV below that margin before the dependency wait (overlaps the q all-gather).
_KV_PRE_WAIT = True
# E4M3 P shift (log2 units), as TRTLLM_Q35_ATTN_FUSED_P_SHIFT: P = exp2(x - m + shift) before the FP8 conversion, so
# probabilities down to 2^-(6 + shift) of the row max stay E4M3 normals (0 = the unscaled P that flushes small
# probabilities: 5-13x the error of trtllm-gen with an attention sink). 5: trtllm-gen's error level (within ~5 % of 6)
# at no cost on flat scores and +0.5 us per call with very wide ones (6: +0.9 us).
_P_SHIFT = 5.0
# In-model diagnostics (default off): TRTLLM_Q35_ATTN_CP_FUSED_DBG_SLOTS=<n> records, for each of the last n launches,
# per-CTA %globaltimer stamps (entry, wait done, grid-barrier arrive / release, end) and the SM id; the ring is saved
# after every burst of launches and at exit to TRTLLM_Q35_ATTN_CP_FUSED_DBG_DIR/cpdbg_r<rank>_p<pid>_{d<k>,final}.pt.
_DBG_SLOTS = 0
# SMs left out of the grid (default 0: one CTA per SM). The in-kernel grid barrier needs every CTA resident; a reserve
# keeps the grid co-resident even when other kernels (e.g. a PDL neighbour still polling its peers) hold some SMs.
_SM_RESERVE = envs.SGLANG_PLB_CP_SM_RESERVE.get()
_DBG_RING_CTAS = 256

# Declared alignment (bytes) of each pointer argument, used at compile and call time (make_ptr checks the address):
# q and the KV pool are TMA sources and the workspace takes 16-byte stores / bulk copies; tables, seq_lens and the
# scales are read element by element (views such as scales[2:3] are 4-byte aligned); out_o is 16 when its rows allow
# 8-element vector stores, else 4 (alltoall_merge reads it as int32).
_ALIGN = dict(q=16, kv=16, tab=4, seq=4, bmm1=4, bmm2=4, lse=4, ws=16, cnt=8, dbg=8)


def cp_partial_geometry(rows: int, n_splits: int) -> dict:
    """Mirror of q35_cp_partial_attn.cp_partial_geometry (kept import-free of cutlass; checked at compile time)."""
    total = rows * HEAD_DIM
    chunk = 8 * ((total + 8 * n_splits - 1) // (8 * n_splits))
    chunk_pad = chunk + (8 if (chunk // 8) % 2 == 0 else 0)
    n_chunks = (total + chunk - 1) // chunk
    lse_stride = 4 * ((n_splits + 3) // 4)
    ws_o_elems = n_chunks * n_splits * chunk_pad
    max_cov_rows = (chunk + HEAD_DIM - 2) // HEAD_DIM + 1
    return dict(
        chunk=chunk, ws_req_bytes=ws_o_elems * 2 + rows * lse_stride * 4, max_cov_rows=max_cov_rows
    )


def cp_partial_balanced_ws_bytes(rows: int, n_ctas: int) -> int:
    """Mirror of q35_cp_partial_attn.cp_partial_balanced_geometry()["ws_bytes"]: per CTA one O slot (the largest
    per-split share n_chunks(c) * chunk_pad(c) of a request's partials over c = 1..n_ctas, FP16) and 16 * rows bytes
    of LSE space."""
    total = rows * HEAD_DIM
    slot_o = 0
    for c in range(1, n_ctas + 1):
        chunk = 8 * ((total + 8 * c - 1) // (8 * c))
        chunk_pad = chunk + (8 if (chunk // 8) % 2 == 0 else 0)
        slot_o = max(slot_o, ((total + chunk - 1) // chunk) * chunk_pad * 2)
    return n_ctas * (16 * ((slot_o + 15) // 16) + 16 * rows)


def balanced_split_plan(seq_lens, cp_rank: int, cp_size: int, n_ctas: int):
    """Host mirror of the balanced split's in-kernel plan (q35_cp_partial_attn.Q35CpPartialAttnKernel._plan).

    Returns (T, c, ctas): T = the largest tile count of a CTA's block, c[r] = CTAs of request r, ctas[i] = (request,
    index in the request, first tile, tiles) of CTA i (tile k of a request = its owned pages 2k, 2k + 1 on this rank).
    """
    t = [
        ((((int(L) + PAGE - 1) // PAGE) - cp_rank + cp_size - 1) // cp_size + 1) // 2
        for L in seq_lens
    ]
    b = len(t)
    assert 1 <= b < n_ctas and b <= 32
    total, tmax = sum(t), max(t)
    lo = max(-(-total // n_ctas), 1)
    hi = max(min(tmax, -(-total // (n_ctas - b))), lo)
    while lo < hi:
        mid = (lo + hi) // 2
        if sum(max(-(-x // mid), 1) for x in t) <= n_ctas:
            hi = mid
        else:
            lo = mid + 1
    c = [max(-(-x // lo), 1) for x in t]
    top = max(r for r in range(b) if t[r] == tmax)
    c[top] += n_ctas - sum(c)
    ctas = []
    for r in range(b):
        base, rem = divmod(t[r], c[r])
        for j in range(c[r]):
            ctas.append((r, j, j * base + min(j, rem), base + (1 if j < rem else 0)))
    return lo, c, ctas


def _num_splits(batch: int, sm_count: int) -> int:
    return max((sm_count - _SM_RESERVE) // batch, 1)


def _num_ctas(sm_count: int) -> int:
    """Grid of the balanced split (any batch)."""
    return max(sm_count - _SM_RESERVE, 1)


def q35_cp_fused_supported(q_len: int, cp_size: int, hq: int, batch: int, sm_count: int) -> bool:
    """Shapes the kernel handles (D 256, FP8 q / KV and 64-token pages are checked by the caller)."""
    rows = q_len * cp_size * hq
    max_batch = MAX_BATCH if q_len > 1 else MAX_BATCH_Q1
    if not (1 <= rows <= MAX_ROWS and 1 <= batch <= max_batch and batch <= sm_count):
        return False
    if _BALANCED:
        return batch < _num_ctas(
            sm_count
        )  # the plan gives every request a CTA; one warp plans <= 32 requests
    return cp_partial_geometry(rows, _num_splits(batch, sm_count))["max_cov_rows"] <= _MAX_COMB_ROWS


class _DeviceState:
    """Per-device grid-barrier counters (int64 [MAX_BATCH + 1, MAX_BATCH], row b = batch size b, one epoch per
    request; zero-initialized once, never reset) and the partial workspace (sized for every supported call)."""

    def __init__(self, device: torch.device, sm_count: int):
        self.sm_count = sm_count
        need = 0
        for b in range(1, MAX_BATCH + 1):
            if b <= sm_count:
                need = max(
                    need,
                    b * cp_partial_geometry(MAX_ROWS, _num_splits(b, sm_count))["ws_req_bytes"],
                )
        if _BALANCED:
            need = max(need, cp_partial_balanced_ws_bytes(MAX_ROWS, _num_ctas(sm_count)))
            logger.info(
                f"TRTLLM_Q35_ATTN_CP_BALANCED_SPLIT=1: CP-lite fused partial attention shares "
                f"{_num_ctas(sm_count)} CTAs among the requests by their KV length (device {device})"
            )
        self.workspace = torch.empty(need // 4 + 64, dtype=torch.float32, device=device)
        self.ws_balanced = None  # test calls that override the split mode (_balanced_workspace)
        self.balanced_logged = False  # runtime evidence: one INFO line at the first balanced call
        self.counters = torch.zeros(MAX_BATCH + 1, MAX_BATCH, dtype=torch.int64, device=device)
        self.dbg_ring = None
        if _DBG_SLOTS > 0:
            self.dbg_ring = torch.zeros(
                _DBG_SLOTS * _DBG_RING_CTAS * 8, dtype=torch.int64, device=device
            )
            _start_dbg_dump(self.dbg_ring, device)


def _start_dbg_dump(ring: torch.Tensor, device: torch.device) -> None:
    raise NotImplementedError("the debug ring is not ported (_DBG_SLOTS = 0)")


_STATES: Dict[int, _DeviceState] = {}


def _sm_count(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def q35_cp_fused_prepare(device: Optional[torch.device] = None, sm_count: int = 0) -> None:
    """Allocate this device's counters / workspace (call before CUDA graph capture)."""
    device = torch.device("cuda", torch.cuda.current_device()) if device is None else device
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _STATES:
        _STATES[idx] = _DeviceState(torch.device("cuda", idx), sm_count or _sm_count(device))


def _state(device: torch.device, sm_count: int) -> _DeviceState:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    st = _STATES.get(idx)
    if st is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "q35_cp_fused_partial: call q35_cp_fused_prepare() (or one eager call) before CUDA "
                "graph capture: the grid-barrier counters must be zero-initialized persistent memory"
            )
        q35_cp_fused_prepare(device, sm_count)
        st = _STATES[idx]
    return st


def _ptr(t: torch.Tensor, dtype, align: int):
    import cutlass.cute as cute

    return cute.runtime.make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)


def _balanced_workspace(state: "_DeviceState", need: int) -> torch.Tensor:
    """Workspace of a balanced-split call: the device workspace (sized for it under TRTLLM_Q35_ATTN_CP_BALANCED_SPLIT)
    or, for test calls that override the mode, a separate one allocated on first use."""
    if state.workspace.numel() * 4 >= need:
        return state.workspace
    if state.ws_balanced is None or state.ws_balanced.numel() * 4 < need:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "q35_cp_fused_partial(balanced=True): make one eager call before CUDA graph capture"
            )
        state.ws_balanced = torch.empty(
            need // 4 + 64, dtype=torch.float32, device=state.workspace.device
        )
    return state.ws_balanced


@functools.lru_cache(maxsize=None)
def _compiled(
    q_len: int,
    gh: int,
    hql: int,
    cp_size: int,
    n_splits: int,
    bmm1_from_ptr: bool,
    bmm2_from_ptr: bool,
    o_vec16: bool,
    use_pdl: bool,
    kv_pre_wait: bool,
    fresh_margin: int,
    debug_ts: bool,
    unsafe_skip_wait: bool,
    p_shift: float = 5.0,
    debug_ring: int = 0,
    balanced: bool = False,
    rescale_threshold: Optional[float] = None,
    l_from_p8: bool = False,
    len_sub: int = 0,
    early_trigger: bool = False,
):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from sglang.srt.layers.attention.q35_cp_partial_attn import Q35CpPartialAttnKernel

    if balanced:
        kernel = Q35CpPartialAttnKernel(
            q_len=q_len,
            gh=gh,
            hql=hql,
            cp_size=cp_size,
            n_splits=n_splits,
            bmm1_from_ptr=bmm1_from_ptr,
            bmm2_from_ptr=bmm2_from_ptr,
            o_vec16=o_vec16,
            use_pdl=use_pdl,
            kv_pre_wait=kv_pre_wait,
            fresh_margin_pages=fresh_margin,
            debug_ts=debug_ts,
            unsafe_skip_wait=unsafe_skip_wait,
            p_shift=p_shift,
            debug_ring=debug_ring,
            balanced=True,
            rescale_threshold=rescale_threshold,
            l_from_p8=l_from_p8,
            len_sub=len_sub,
            early_trigger=early_trigger,
        )
        assert kernel.ws_bytes == cp_partial_balanced_ws_bytes(q_len * gh, n_splits)
    else:
        kernel = Q35CpPartialAttnKernel(
            q_len=q_len,
            gh=gh,
            hql=hql,
            cp_size=cp_size,
            n_splits=n_splits,
            bmm1_from_ptr=bmm1_from_ptr,
            bmm2_from_ptr=bmm2_from_ptr,
            o_vec16=o_vec16,
            use_pdl=use_pdl,
            kv_pre_wait=kv_pre_wait,
            fresh_margin_pages=fresh_margin,
            debug_ts=debug_ts,
            unsafe_skip_wait=unsafe_skip_wait,
            p_shift=p_shift,
            debug_ring=debug_ring,
            rescale_threshold=rescale_threshold,
            l_from_p8=l_from_p8,
            len_sub=len_sub,
            early_trigger=early_trigger,
        )
        assert kernel.ws_req_bytes == cp_partial_geometry(q_len * gh, n_splits)["ws_req_bytes"]
    dev = torch.device("cuda")
    f32 = torch.empty(1024, dtype=torch.float32, device=dev)
    i32 = torch.zeros(64, dtype=torch.int32, device=dev)
    u8 = torch.empty(64 * HEAD_DIM, dtype=torch.uint8, device=dev)
    bf = torch.empty(1024, dtype=torch.bfloat16, device=dev)
    i64 = torch.zeros(16, dtype=torch.int64, device=dev)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fp8 = cutlass.Float8E4M3FN
    a = _ALIGN
    I32 = cutlass.Int32
    return cute.compile(
        kernel,
        _ptr(u8, fp8, a["q"]),
        _ptr(u8, fp8, a["kv"]),
        _ptr(u8, fp8, a["kv"]),
        I32(64),
        _ptr(i32, I32, a["tab"]),
        I32(2),
        I32(1),
        _ptr(i32, I32, a["seq"]),
        I32(0),
        _ptr(f32, cutlass.Float32, a["bmm1"]),
        cutlass.Float32(1.0),
        _ptr(f32, cutlass.Float32, a["bmm2"]),
        cutlass.Float32(1.0),
        _ptr(bf, cutlass.BFloat16, 16 if o_vec16 else 4),
        I32(HEAD_DIM),
        I32(HEAD_DIM),
        I32(HEAD_DIM),
        _ptr(f32, cutlass.Float32, a["lse"]),
        I32(1),
        I32(1),
        I32(1),
        _ptr(f32, cutlass.Float32, a["ws"]),
        _ptr(i64, cutlass.Int64, a["cnt"]),
        _ptr(i64, cutlass.Int64, a["dbg"]),
        I32(1),
        stream,
    )


def q35_cp_fused_partial(
    q_all: torch.Tensor,
    kv_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    q_len: int,
    cp_rank: int,
    cp_size: int,
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    out_o: torch.Tensor,
    out_lse: torch.Tensor,
    sm_count: int = 0,
    enable_pdl: bool = False,
    *,
    debug: Optional[torch.Tensor] = None,
    v_pool: Optional[torch.Tensor] = None,
    unsafe_skip_wait: bool = False,
    n_splits: Optional[int] = None,
    fresh_margin: Optional[int] = None,
    counters: Optional[torch.Tensor] = None,
    p_shift: Optional[float] = None,
    balanced: Optional[bool] = None,
    rescale_threshold: Optional[float] = None,
    l_from_p8: bool = False,
    len_sub: int = 0,
    early_trigger: bool = False,
) -> None:
    """This CP rank's partial attention of every group head over the pages it owns (q35_cp_fused_partial_ref).

    q_all [cp, rows, hq, D] FP8 (the all-gather output viewed member-major; the native layout, strides (hq * D,
    cp * hq * D, D, 1) at a 16-byte aligned address, is read in place, any other view through a copy); kv_pool [blocks,
    1, 64, D] FP8; block_tables [>= b, 2, max_pages] int32 (K / V block ids, unit stride on pages); seq_lens [>= b]
    int32 (L incl. the q_len new tokens); bmm1_scale float (natural-log domain) or a [>= 2] FP32 tensor (raw, raw *
    log2e); bmm2_scale float or a [>= 1] FP32 tensor. out_o [cp, rows, hq, D] bf16 (unit stride on D), out_lse [cp,
    rows, hq] FP32 (base 2). Test hooks: debug (int64 [b * splits, 16] %globaltimer stamps; [CTAs, 16] when balanced),
    unsafe_skip_wait (no dependency waits), n_splits (must come with its own ``counters``: every launch on a counter
    row must use the same split count; balanced: the grid's CTAs), fresh_margin, balanced (overrides
    TRTLLM_Q35_ATTN_CP_BALANCED_SPLIT; a mode other than the module's needs its own ``counters``).
    Numerics: p_shift (E4M3 P shift, default _P_SHIFT), rescale_threshold (lazy rescale threshold in log2 units,
    default min(8, 8.75 - p_shift); 0 = exact running max), l_from_p8 (row sums from the E4M3-rounded P).
    """
    import cuda.bindings.driver as cuda
    import cutlass

    cp, rows, hq, D = q_all.shape
    assert cp == cp_size and D == HEAD_DIM and q_all.dtype == torch.float8_e4m3fn
    assert rows % q_len == 0
    b = rows // q_len
    gh = cp * hq
    R = q_len * gh
    assert 1 <= R <= MAX_ROWS, f"q_len x group heads = {R} > {MAX_ROWS}"
    assert 0 <= cp_rank < cp_size
    assert (
        kv_pool.dim() == 4
        and kv_pool.shape[1] == 1
        and kv_pool.shape[2] == PAGE
        and kv_pool.shape[3] == HEAD_DIM
    )
    assert kv_pool.is_contiguous() and kv_pool.dtype in (torch.float8_e4m3fn, torch.uint8)
    # v_pool: V in its own pool of kv_pool's shape; block_tables [b, P] then names the same page of both
    if v_pool is not None:
        assert v_pool.shape == kv_pool.shape and v_pool.is_contiguous() and v_pool.dtype == kv_pool.dtype
    if block_tables.dim() == 2:
        assert v_pool is not None, "a [b, P] page table needs v_pool"
        tab_sb, tab_skv = block_tables.stride(0), 0
    else:
        assert block_tables.dim() == 3 and block_tables.shape[1] == 2
        tab_sb, tab_skv = block_tables.stride(0), block_tables.stride(1)
    assert block_tables.shape[0] >= b
    assert block_tables.stride(-1) == 1 and block_tables.dtype == torch.int32
    assert seq_lens.dtype == torch.int32 and seq_lens.numel() >= b and seq_lens.stride(0) == 1
    assert (
        tuple(out_o.shape) == (cp, rows, hq, D)
        and out_o.dtype == torch.bfloat16
        and out_o.stride(3) == 1
    )
    assert tuple(out_lse.shape) == (cp, rows, hq) and out_lse.dtype == torch.float32
    assert 1 <= b <= MAX_BATCH
    device = q_all.device
    sms = sm_count or _sm_count(device)
    state = _state(device, sms)
    bal = _BALANCED if balanced is None else bool(balanced)
    if bal:
        splits = n_splits if n_splits else _num_ctas(sms)
        assert b < splits <= sms, (
            "balanced split: more CTAs than requests, every CTA co-resident (one per SM)"
        )
    else:
        splits = n_splits if n_splits else _num_splits(b, sms)
        assert splits * b <= sms, (
            "the in-kernel grid barrier needs every CTA co-resident (one per SM)"
        )
    if n_splits or bal != _BALANCED:
        assert counters is not None, (
            "a split-count or split-mode override needs its own counters tensor"
        )
    cnt = state.counters if counters is None else counters
    assert cnt.dtype == torch.int64 and cnt.dim() == 2 and cnt.shape[0] > b and cnt.shape[1] >= b
    assert cnt.is_contiguous()
    if bal:
        ws = _balanced_workspace(state, cp_partial_balanced_ws_bytes(R, splits))
        if balanced is None and not state.balanced_logged:
            state.balanced_logged = True
            logger.info(
                f"TRTLLM_Q35_ATTN_CP_BALANCED_SPLIT=1: first balanced call (q_len {q_len}, batch {b}, cp_rank "
                f"{cp_rank}, {splits} CTAs)"
            )
    else:
        need = b * cp_partial_geometry(R, splits)["ws_req_bytes"]
        ws = state.workspace
        assert ws.numel() * 4 >= need, "CP partial workspace too small"
    # q: the native all-gather layout is one [b * rows_per_request, 256] matrix read by TMA in place
    native = (
        q_all.stride(3) == 1
        and q_all.stride(2) == D
        and q_all.stride(0) == hq * D
        and q_all.stride(1) == cp * hq * D
        and q_all.data_ptr() % 16 == 0
    )
    q_src = q_all if native else q_all.permute(1, 0, 2, 3).contiguous()
    b1_ptr = isinstance(bmm1_scale, torch.Tensor)
    b2_ptr = isinstance(bmm2_scale, torch.Tensor)
    if b1_ptr:
        assert (
            bmm1_scale.numel() >= 2
            and bmm1_scale.dtype == torch.float32
            and bmm1_scale.stride(0) == 1
        )
    if b2_ptr:
        assert bmm2_scale.numel() >= 1 and bmm2_scale.dtype == torch.float32
    o_vec16 = out_o.data_ptr() % 16 == 0 and all(out_o.stride(i) % 8 == 0 for i in range(3))
    assert out_o.data_ptr() % 4 == 0 and all(out_o.stride(i) % 2 == 0 for i in range(3)), (
        "out_o: 4-byte aligned rows (alltoall_merge reads it as int32)"
    )
    ring = state.dbg_ring if debug is None else None
    if bal:
        fn = _compiled(
            q_len,
            gh,
            hq,
            cp,
            splits,
            b1_ptr,
            b2_ptr,
            o_vec16,
            bool(enable_pdl),
            _KV_PRE_WAIT,
            _FRESH_MARGIN if fresh_margin is None else fresh_margin,
            debug is not None,
            unsafe_skip_wait,
            _P_SHIFT if p_shift is None else float(p_shift),
            _DBG_SLOTS if ring is not None else 0,
            True,
            rescale_threshold,
            bool(l_from_p8),
            int(len_sub),
            bool(early_trigger),
        )
    else:
        fn = _compiled(
            q_len,
            gh,
            hq,
            cp,
            splits,
            b1_ptr,
            b2_ptr,
            o_vec16,
            bool(enable_pdl),
            _KV_PRE_WAIT,
            _FRESH_MARGIN if fresh_margin is None else fresh_margin,
            debug is not None,
            unsafe_skip_wait,
            _P_SHIFT if p_shift is None else float(p_shift),
            _DBG_SLOTS if ring is not None else 0,
            False,
            rescale_threshold,
            bool(l_from_p8),
            int(len_sub),
            bool(early_trigger),
        )
    fp8 = cutlass.Float8E4M3FN
    f32 = cutlass.Float32
    I32 = cutlass.Int32
    a = _ALIGN
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    dummy = ws
    fn(
        _ptr(q_src, fp8, a["q"]),
        _ptr(kv_pool, fp8, a["kv"]),
        _ptr(kv_pool if v_pool is None else v_pool, fp8, a["kv"]),
        I32(kv_pool.shape[0] * PAGE),
        _ptr(block_tables, I32, a["tab"]),
        I32(tab_sb),
        I32(tab_skv),
        _ptr(seq_lens, I32, a["seq"]),
        I32(cp_rank),
        _ptr(bmm1_scale if b1_ptr else dummy, f32, a["bmm1"]),
        f32(0.0 if b1_ptr else float(bmm1_scale) * LOG2E),
        _ptr(bmm2_scale if b2_ptr else dummy, f32, a["bmm2"]),
        f32(0.0 if b2_ptr else float(bmm2_scale)),
        _ptr(out_o, cutlass.BFloat16, 16 if o_vec16 else 4),
        I32(out_o.stride(0)),
        I32(out_o.stride(1)),
        I32(out_o.stride(2)),
        _ptr(out_lse, f32, a["lse"]),
        I32(out_lse.stride(0)),
        I32(out_lse.stride(1)),
        I32(out_lse.stride(2)),
        _ptr(ws, f32, a["ws"]),
        _ptr(cnt[b], cutlass.Int64, a["cnt"]),
        _ptr(
            debug if debug is not None else (ring if ring is not None else cnt[b]),
            cutlass.Int64,
            a["dbg"],
        ),
        I32(b),
        stream,
    )
