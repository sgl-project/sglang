"""QSA prefill sparse attention over row-group token unions on tcgen05 (opt-in, default off).

Enabled by SGLANG_ENABLE_QSA_UNION_PREFILL. It computes what ``_sparse_gqa_chunk_prefill`` (and its
TMA twin, qsa/tma_prefill.py) computes -- for every query row and KV head, the softmax over exactly
the row's selected context tokens (index columns [0, row_limit) that are >= 0) with Q scaled as stock,
``bf16(q * scale * log2(e))``, P rounded to BF16 before PV and FP32 accumulation -- with a different
schedule:

* The rows of each request are cut into groups of ``ROWS`` consecutive rows (10: 10 rows x 12 query
  heads fill the 128-row MMA). A JIT CUDA pass (kernels/jit/csrc/elementwise/qsa_union_prefill.cuh)
  lists, per group, the union of the rows' selected tokens, in position order, and per 128-token tile
  a bitmask of the tokens each row selected.
* One CTA per (group, KV head) streams the union in 128-token tiles through warp-specialized
  partitions: a 4-warp loader (TMA gather4 of the tile's packed BF16 K and V rows into a 2-stage
  ring), a 1-warp tcgen05 issuer (S = Q K^T into one of two TMEM buffers, O += P V with P read from
  TMEM in place of S) and a 4-warp softmax (one query row per thread: S with the row's mask, the
  running max, P in TMEM; the output at the end).

Neighbouring rows select mostly the same blocks, so a token gathered once serves all rows of the group
that selected it; the rows that did not select it see -inf. A row's result is mathematically the stock
one, but not bitwise: the online softmax walks 128-token tiles of the union instead of 16-token steps of
the row's own list, and the MMA reduction order differs. By default a row's running max is exact (O and
the row sum are rescaled whenever it grows), so each row's largest P is exactly 1 before the BF16
rounding, as in the stock kernel; SGLANG_QSA_UNION_PREFILL_RESCALE_LOG2 > 0 rescales only when the max
grows by more than that (FlashAttention-4 uses 8), which is cheaper but adds rounding error.

Per call the union pass allocates G * ceil128(min(ROWS * topk, longest context)) int32 token rows plus a
quarter of that in row masks (G = sum of ceil(extend_len / ROWS) over requests): at most about 180 MB
for a 16K-row pass.

Patterns borrowed: the partitions, TMEM operands and P-in-TMEM aliasing of Triton's Gluon attention
example (triton-lang/triton python/examples/gluon/01-attention-forward.py, v3.9.0), TMA gather4 as in
python/tutorials/gluon/09-tma-gather-scatter.py, the mask-by-bit select of that example's causal mask,
and the conditional rescale of FlashAttention-4 (Dao-AILab/flash-attention, flash_attn/cute).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Sequence

import torch
import triton

logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


# Read once at import, like the backend's other QSA prefill switches; environ.py is untouched so this stacks.
ENABLED = _env_flag("SGLANG_ENABLE_QSA_UNION_PREFILL")
# Query rows per group; 0 picks floor(128 / query heads per KV head).
ROWS = _env_int("SGLANG_QSA_UNION_PREFILL_ROWS", 0)
# K and V ring stages (128 tokens x 256 BF16 each); 2 + 2 fill shared memory next to Q (1 + 3 measured slower).
STAGES_K = _env_int("SGLANG_QSA_UNION_PREFILL_STAGES_K", 2)
STAGES_V = _env_int("SGLANG_QSA_UNION_PREFILL_STAGES_V", 2)
# Passes with fewer query rows keep the existing path; measured faster from 1,024 rows up (VR200 engine checks).
MIN_ROWS = _env_int("SGLANG_QSA_UNION_PREFILL_MIN_ROWS", 1024)
# Debug: the first N served calls also run the replaced kernel and log time and error. Not for served runs.
CHECK_CALLS = _env_int("SGLANG_QSA_UNION_PREFILL_CHECK_CALLS", 0)
# Lazy max threshold (log2): 0 rescales O whenever a row's max grows, so the row's largest P is exactly 1
# before the BF16 rounding, as in the stock kernel; FlashAttention-4 uses 8, which is cheaper but less exact.
RESCALE_LOG2 = float(os.environ.get("SGLANG_QSA_UNION_PREFILL_RESCALE_LOG2", "0") or 0)
_HEAD_DIM = 256
_BM = 128
_BN = 128
# The union pass keeps a bitmap of the request's context in shared memory (2 words per 32 positions).
_MAX_KV = 786432
_state = {"logged": False, "checked": 0, "fallback": set()}

try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.nvidia.blackwell import (
        TensorMemoryLayout,
        allocate_tensor_memory,
        fence_async_shared,
        mbarrier,
        tcgen05_mma,
        tensor_memory_descriptor_type,
        tma,
    )
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

    _GLUON = True
except Exception:  # a Triton without Gluon's Blackwell API: the existing path serves
    _GLUON = False


def rows_per_group(num_q_heads: int, num_kv_heads: int) -> int:
    gq = num_q_heads // num_kv_heads
    return ROWS if ROWS > 0 else _BM // gq


def group_count(extend_lens: Sequence[int], rows: int) -> int:
    return sum((int(n) + rows - 1) // rows for n in extend_lens)


def bitmap_words(kv_max: int) -> int:
    """32-bit words of the union pass's position bitmap (one spare word)."""
    return (int(kv_max) + 31) // 32 + 1


def union_capacity(rows: int, topk: int, kv_max: int) -> int:
    """Union tokens a group can hold (whole 128-token tiles): its rows' entries, or the bitmap's positions,
    if fewer (the bound the JIT pass checks)."""
    return (min(rows * topk, 32 * bitmap_words(kv_max)) + _BN - 1) // _BN * _BN


def _fallback(reason: str) -> None:
    if reason not in _state["fallback"]:
        _state["fallback"].add(reason)
        logger.info("QSA union prefill: fallback (%s) (SGLANG_ENABLE_QSA_UNION_PREFILL)", reason)


def applies(q, indices, *, kv_heads, kv_dtype, extend_lens, kv_max, total_k) -> bool:
    """BF16 q and K/V of head dim 256, at most 128 query rows per group, int32 [rows, topk] indices,
    SM100 family with Gluon, a host extend-length list, and contexts the union bitmap holds."""
    if q.shape[0] < MIN_ROWS or extend_lens is None:
        return False
    if not _GLUON:
        _fallback("no gluon")
        return False
    if not (q.is_cuda and q.dtype == kv_dtype == torch.bfloat16):
        _fallback("dtype")
        return False
    if q.ndim != 3 or q.shape[2] != _HEAD_DIM or q.stride(2) != 1:
        _fallback("shape")
        return False
    if kv_heads < 1 or q.shape[1] % kv_heads:
        _fallback("heads")
        return False
    gq = q.shape[1] // kv_heads
    rows = rows_per_group(q.shape[1], kv_heads)
    if rows < 1 or rows * gq > _BM:
        _fallback("group size")
        return False
    if indices.dtype != torch.int32 or indices.ndim != 2 or indices.stride(1) != 1:
        _fallback("indices")
        return False
    if indices.shape[0] != q.shape[0] or sum(int(n) for n in extend_lens) != q.shape[0]:
        _fallback("rows")
        return False
    if kv_max > _MAX_KV or total_k >= 2**31 - 1:
        _fallback("context")
        return False
    if torch.cuda.get_device_capability(q.device)[0] != 10:
        _fallback("device")
        return False
    return True


def kv_ok(packed_k, packed_v) -> bool:
    """The gather descriptors' needs: contiguous [rows, heads, 256] BF16 with 16-byte aligned bases."""
    return (
        packed_k.shape == packed_v.shape
        and packed_k.is_contiguous()
        and packed_v.is_contiguous()
        and packed_k.data_ptr() % 16 == 0
        and packed_v.data_ptr() % 16 == 0
    )


def _build_module(rows: int):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(rows)
    return load_jit(
        "qsa_union_prefill",
        *args,
        cuda_files=["elementwise/qsa_union_prefill.cuh"],
        cuda_wrappers=[("build", f"QsaUnionPrefillBuild<{args}>::run")],
    )


_MODULES: dict = {}


def build_unions(indices, cu_q, kv_lens, cu_k, *, groups, rows, kv_max, oob):
    """The JIT union pass: (tok [G * nt] int32, mask [G * nt / 128 * rows * 4] int32, meta [G * 4] int32, nt)."""
    module = _MODULES.get(rows)
    if module is None:
        module = _MODULES[rows] = _build_module(rows)
    topk = indices.shape[1]
    nt = union_capacity(rows, topk, kv_max)
    words = bitmap_words(kv_max)
    dev = indices.device
    tok = torch.empty(groups * nt, dtype=torch.int32, device=dev)
    mask = torch.empty(groups * (nt // _BN) * rows * 4, dtype=torch.int32, device=dev)
    meta = torch.empty(groups * 4, dtype=torch.int32, device=dev)
    module.build(
        indices,
        cu_q.to(torch.int64),
        kv_lens.to(torch.int32),
        cu_k.to(torch.int64),
        tok,
        mask,
        meta,
        groups,
        nt,
        words,
        int(oob),
    )
    return tok, mask, meta, nt


if _GLUON:

    @gluon.constexpr_function
    def _reg_layout(dtype, shape, tmem_layout, alloc_shape, num_warps):
        ty = tensor_memory_descriptor_type(dtype, shape, tmem_layout, alloc_shape)
        return ty.get_reg_layout(num_warps=num_warps)

    @gluon.jit
    def _p_view(s_buf, BM: gl.constexpr, BN: gl.constexpr):
        # P (BF16) in the first half of its S buffer's 32-bit columns, as the example's _borrow_s_as_p
        p = s_buf.reinterpret(gl.bfloat16, [BM, 2 * BN], TensorMemoryLayout((BM, BN), col_stride=1))
        return p.slice(0, BN)

    @gluon.jit
    def _upa_load(k_desc, v_desc, TOK, k_smem, v_smem, k_ready, v_ready, k_free, v_free, tok_base, head,
                  n_tiles, BN: gl.constexpr, D: gl.constexpr, NSK: gl.constexpr, NSV: gl.constexpr):
        flat: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
        # gather4 takes each 4-row chunk of offsets whole in every lane of one warp
        rows_layout: gl.constexpr = gl.SliceLayout(0, gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0]))
        col = head * D
        lane = gl.arange(0, BN, flat)
        for i in range(n_tiles):
            sk = i % NSK
            sv = i % NSV
            rows = gl.convert_layout(gl.load(TOK + tok_base + i * BN + lane), rows_layout)
            mbarrier.wait(k_free.index(sk), ((i // NSK) & 1) ^ 1)
            mbarrier.expect(k_ready.index(sk), BN * k_desc.block_type.nbytes)
            tma.async_gather(k_desc, rows, col, k_ready.index(sk), k_smem.index(sk))
            mbarrier.wait(v_free.index(sv), ((i // NSV) & 1) ^ 1)
            mbarrier.expect(v_ready.index(sv), BN * v_desc.block_type.nbytes)
            tma.async_gather(v_desc, rows, col, v_ready.index(sv), v_smem.index(sv))

    @gluon.jit
    def _upa_pv(v_smem, s_tmem, o_tmem, v_ready, v_free, p_ready, o_done, j, BM: gl.constexpr,
                BN: gl.constexpr, NSV: gl.constexpr):
        sj = j % NSV
        bj = j % 2
        mbarrier.wait(v_ready.index(sj), (j // NSV) & 1)
        mbarrier.wait(p_ready.index(bj), (j // 2) & 1)
        # O [BM, D] += P [BM, BN] (TMEM) @ V [BN, D]
        tcgen05_mma(_p_view(s_tmem.index(bj), BM, BN), v_smem.index(sj), o_tmem, use_acc=j > 0,
                    mbarriers=[o_done.index(0), v_free.index(sj)])

    @gluon.jit
    def _upa_mma(q_smem, k_smem, v_smem, s_tmem, o_tmem, k_ready, v_ready, k_free, v_free, s_ready, p_ready,
                 o_done, n_tiles, BM: gl.constexpr, BN: gl.constexpr, NSK: gl.constexpr, NSV: gl.constexpr):
        # QK of tile i is issued before PV of tile i - 1, so the softmax of tile i - 1 overlaps it. S buffer
        # i % 2 last held P of tile i - 2, whose PV was issued (in order) before this QK.
        for i in range(n_tiles):
            s = i % NSK
            b = i % 2
            mbarrier.wait(k_ready.index(s), (i // NSK) & 1)
            # S [BM, BN] = Q [BM, D] @ K^T [D, BN]
            tcgen05_mma(q_smem, k_smem.index(s).permute((1, 0)), s_tmem.index(b), use_acc=False,
                        mbarriers=[s_ready.index(b), k_free.index(s)])
            if i > 0:
                _upa_pv(v_smem, s_tmem, o_tmem, v_ready, v_free, p_ready, o_done, i - 1, BM, BN, NSV)
        if n_tiles > 0:
            _upa_pv(v_smem, s_tmem, o_tmem, v_ready, v_free, p_ready, o_done, n_tiles - 1, BM, BN, NSV)

    @gluon.jit
    def _upa_select(x, word, bitval):
        return gl.where((word & bitval) != 0, x, float("-inf"))

    @gluon.jit
    def _upa_masked(s_buf, C: gl.constexpr, word, bitval, SC: gl.constexpr, layout: gl.constexpr):
        # columns [C * SC, C * SC + SC) of S; a column a row did not select is -inf. One element at a time
        # (map_elementwise), as the Gluon example's causal mask, so ptxas can test the bits with R2P.
        x = s_buf.slice(C * SC, SC).load(layout)
        return gl.map_elementwise(_upa_select, x, word[:, None], bitval[None, :])

    @gluon.jit
    def _upa_softmax(s_tmem, o_tmem, s_ready, p_ready, o_done, MASK, mask_base, OUT, row0, rows, head, n_tiles,
                     O_STRIDE_M, O_STRIDE_H, GQ: gl.constexpr, W: gl.constexpr, D: gl.constexpr,
                     BM: gl.constexpr, BN: gl.constexpr, RESCALE: gl.constexpr):
        SC: gl.constexpr = 32  # one mask word per S chunk
        OC: gl.constexpr = 64
        gl.static_assert(BN == 4 * SC)
        s_tl: gl.constexpr = TensorMemoryLayout((BM, BN), col_stride=1)
        sc_layout: gl.constexpr = _reg_layout(gl.float32, [BM, SC], s_tl, [BM, BN], gl.num_warps())
        o_tl: gl.constexpr = TensorMemoryLayout((BM, D), col_stride=1)
        oc_layout: gl.constexpr = _reg_layout(gl.float32, [BM, OC], o_tl, [BM, D], gl.num_warps())
        row_l: gl.constexpr = gl.SliceLayout(1, sc_layout)
        # one query row (group row j, head h) per thread: m = j * GQ + h
        m_idx = gl.arange(0, BM, row_l)
        j_idx = m_idx // GQ
        ok = (m_idx < W * GQ) & (j_idx < rows)
        wptr = MASK + mask_base + gl.minimum(j_idx, W - 1) * 4
        bitval = 1 << gl.arange(0, SC, gl.SliceLayout(0, sc_layout))
        has0 = ok & (n_tiles > 0)
        n0 = gl.load(wptr + 0, mask=has0, other=0)
        n1 = gl.load(wptr + 1, mask=has0, other=0)
        n2 = gl.load(wptr + 2, mask=has0, other=0)
        n3 = gl.load(wptr + 3, mask=has0, other=0)
        m_i = gl.full([BM], float("-inf"), gl.float32, row_l)
        l_i = gl.zeros([BM], gl.float32, row_l)
        for i in range(n_tiles):
            b = i % 2
            # rows outside the group load zero words, so they select nothing
            w0 = n0
            w1 = n1
            w2 = n2
            w3 = n3
            # the next tile's row words are loaded a tile ahead, off the critical path
            hasn = ok & (i + 1 < n_tiles)
            nptr = wptr + (i + 1) * (W * 4)
            n0 = gl.load(nptr + 0, mask=hasn, other=0)
            n1 = gl.load(nptr + 1, mask=hasn, other=0)
            n2 = gl.load(nptr + 2, mask=hasn, other=0)
            n3 = gl.load(nptr + 3, mask=hasn, other=0)
            mbarrier.wait(s_ready.index(b), (i // 2) & 1)
            s_buf = s_tmem.index(b)
            x0 = _upa_masked(s_buf, 0, w0, bitval, SC, sc_layout)
            x1 = _upa_masked(s_buf, 1, w1, bitval, SC, sc_layout)
            x2 = _upa_masked(s_buf, 2, w2, bitval, SC, sc_layout)
            x3 = _upa_masked(s_buf, 3, w3, bitval, SC, sc_layout)
            mx = gl.maximum(gl.maximum(gl.max(x0, 1), gl.max(x1, 1)), gl.maximum(gl.max(x2, 1), gl.max(x3, 1)))
            m_new = gl.maximum(m_i, mx)
            prior = m_i > float("-inf")
            # A row with prior mass moves its max (and rescales O and l) when a score exceeds it by more
            # than RESCALE (log2 units); a row's first finite scores set it.
            need = prior & (m_new > m_i + RESCALE)
            alpha = gl.where(need, gl.exp2(m_i - m_new), 1.0)
            m_i = gl.where(need | (m_i == float("-inf")), m_new, m_i)
            m_use = gl.where(m_i == float("-inf"), 0.0, m_i)
            p0 = gl.exp2(x0 - m_use[:, None])
            p1 = gl.exp2(x1 - m_use[:, None])
            p2 = gl.exp2(x2 - m_use[:, None])
            p3 = gl.exp2(x3 - m_use[:, None])
            l_i = l_i * alpha + ((gl.sum(p0, 1) + gl.sum(p1, 1)) + (gl.sum(p2, 1) + gl.sum(p3, 1)))
            p_buf = _p_view(s_buf, BM, BN)
            p_buf.slice(0 * SC, SC).store(p0.to(gl.bfloat16))
            p_buf.slice(1 * SC, SC).store(p1.to(gl.bfloat16))
            p_buf.slice(2 * SC, SC).store(p2.to(gl.bfloat16))
            p_buf.slice(3 * SC, SC).store(p3.to(gl.bfloat16))
            if i > 0:
                # PV of tile i - 1 has landed in O (waited every tile, so the phase parity stays exact)
                mbarrier.wait(o_done.index(0), (i - 1) & 1)
                if gl.max(need.to(gl.int32), 0) > 0:
                    a_o = gl.convert_layout(alpha, gl.SliceLayout(1, oc_layout))
                    for c in gl.static_range(D // OC):
                        o_ref = o_tmem.slice(c * OC, OC)
                        o_ref.store(o_ref.load(oc_layout) * a_o[:, None])
            mbarrier.arrive(p_ready.index(b), count=1)

        if n_tiles > 0:
            mbarrier.wait(o_done.index(0), (n_tiles - 1) & 1)
        mo = gl.arange(0, BM, gl.SliceLayout(1, oc_layout))
        jo = mo // GQ
        ok_o = (mo < W * GQ) & (jo < rows)
        l_o = gl.convert_layout(l_i, gl.SliceLayout(1, oc_layout))
        do = gl.arange(0, OC, gl.SliceLayout(0, oc_layout))
        obase = OUT + (row0 + jo).to(gl.int64) * O_STRIDE_M + (head * GQ + mo % GQ) * O_STRIDE_H
        for c in gl.static_range(D // OC):
            o = gl.zeros([BM, OC], gl.float32, oc_layout)
            if n_tiles > 0:
                o = o_tmem.slice(c * OC, OC).load(oc_layout)
            # a row that selected nothing ends in 0 / 0, as the stock kernel's does
            o = o / l_o[:, None]
            gl.store(obase[:, None] + (c * OC + do)[None, :], o.to(gl.bfloat16), mask=ok_o[:, None])

    @gluon.jit
    def _qsa_union_prefill(
        Q,
        k_desc,
        v_desc,
        OUT,
        TOK,
        MASK,
        META,
        scale,
        Q_STRIDE_M,
        Q_STRIDE_H,
        O_STRIDE_M,
        O_STRIDE_H,
        NT,
        GQ: gl.constexpr,
        W: gl.constexpr,
        D: gl.constexpr,
        NSK: gl.constexpr,
        NSV: gl.constexpr,
        RESCALE: gl.constexpr,
    ):
        BM: gl.constexpr = 128  # query rows: W group rows x GQ heads, padded
        BN: gl.constexpr = 128  # union tokens per tile
        QC: gl.constexpr = 64
        gl.static_assert(W * GQ <= BM)
        g = gl.program_id(0)
        head = gl.program_id(1)

        qk_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BM, D], gl.bfloat16)
        q_smem = gl.allocate_shared_memory(gl.bfloat16, [BM, D], qk_layout)
        k_smem = gl.allocate_shared_memory(gl.bfloat16, [NSK, BN, D], k_desc.layout)
        v_smem = gl.allocate_shared_memory(gl.bfloat16, [NSV, BN, D], v_desc.layout)
        k_ready = gl.allocate_shared_memory(gl.int64, [NSK, 1], mbarrier.MBarrierLayout())
        k_free = gl.allocate_shared_memory(gl.int64, [NSK, 1], mbarrier.MBarrierLayout())
        v_ready = gl.allocate_shared_memory(gl.int64, [NSV, 1], mbarrier.MBarrierLayout())
        v_free = gl.allocate_shared_memory(gl.int64, [NSV, 1], mbarrier.MBarrierLayout())
        s_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
        p_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
        o_done = gl.allocate_shared_memory(gl.int64, [1, 1], mbarrier.MBarrierLayout())
        for i in gl.static_range(NSK):
            mbarrier.init(k_ready.index(i), count=1)
            mbarrier.init(k_free.index(i), count=1)
        for i in gl.static_range(NSV):
            mbarrier.init(v_ready.index(i), count=1)
            mbarrier.init(v_free.index(i), count=1)
        for i in gl.static_range(2):
            mbarrier.init(s_ready.index(i), count=1)
            mbarrier.init(p_ready.index(i), count=1)
        mbarrier.init(o_done.index(0), count=1)
        # TMEM columns: S 2 x 128 (P aliases each) + O 256
        s_tmem = allocate_tensor_memory(gl.float32, [2, BM, BN], TensorMemoryLayout((BM, BN), col_stride=1))
        o_tmem = allocate_tensor_memory(gl.float32, [BM, D], TensorMemoryLayout((BM, D), col_stride=1))

        row0 = gl.load(META + g * 4)
        rows = gl.load(META + g * 4 + 1)
        n_tiles = gl.load(META + g * 4 + 3)
        tok_base = g * NT
        mask_base = g * (NT // BN) * (W * 4)

        # Q: the group's rows x this KV head's GQ heads, scaled exactly as the stock kernel
        q_ld: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [gl.num_warps(), 1], [1, 0])
        mq = gl.arange(0, BM, gl.SliceLayout(1, q_ld))
        jq = mq // GQ
        okq = (mq < W * GQ) & (jq < rows)
        qrow = Q + (row0 + jq).to(gl.int64) * Q_STRIDE_M + (head * GQ + mq % GQ) * Q_STRIDE_H
        dq = gl.arange(0, QC, gl.SliceLayout(0, q_ld))
        for c in gl.static_range(D // QC):
            qv = gl.load(qrow[:, None] + (c * QC + dq)[None, :], mask=okq[:, None], other=0.0)
            qv = (qv * scale * 1.4426950408).to(gl.bfloat16)
            q_smem.slice(c * QC, QC, dim=1).store(qv)
        fence_async_shared()

        gl.warp_specialize(
            [
                (_upa_softmax, (s_tmem, o_tmem, s_ready, p_ready, o_done, MASK, mask_base, OUT, row0, rows, head,
                                n_tiles, O_STRIDE_M, O_STRIDE_H, GQ, W, D, BM, BN, RESCALE)),
                (_upa_mma, (q_smem, k_smem, v_smem, s_tmem, o_tmem, k_ready, v_ready, k_free, v_free, s_ready,
                            p_ready, o_done, n_tiles, BM, BN, NSK, NSV)),
                (_upa_load, (k_desc, v_desc, TOK, k_smem, v_smem, k_ready, v_ready, k_free, v_free, tok_base, head,
                             n_tiles, BN, D, NSK, NSV)),
            ],
            [1, 4],
            [48, 80],
        )


LAST_KERNEL: dict = {}


def attend(q, packed_k, packed_v, tok, mask, meta, nt, scale, *, groups, rows, stages=None, rescale=None,
           out=None):
    """The tcgen05 kernel over build_unions' lists; writes out [rows, q_heads, 256] (BF16)."""
    kv_heads = packed_k.shape[1]
    gq = q.shape[1] // kv_heads
    if out is None:
        out = torch.empty_like(q)
    if groups == 0:
        return out
    # Built per call: the packed K/V are fresh tensors every pass.
    layout = gl.NVMMASharedLayout.get_default_for([_BN, _HEAD_DIM], gl.bfloat16)
    k_desc = TensorDescriptor.from_tensor(packed_k.view(packed_k.shape[0], -1), [1, _HEAD_DIM], layout)
    v_desc = TensorDescriptor.from_tensor(packed_v.view(packed_v.shape[0], -1), [1, _HEAD_DIM], layout)
    LAST_KERNEL["union_prefill"] = _qsa_union_prefill[(groups, kv_heads)](
        q,
        k_desc,
        v_desc,
        out,
        tok,
        mask,
        meta,
        float(scale),
        q.stride(0),
        q.stride(1),
        out.stride(0),
        out.stride(1),
        nt,
        GQ=gq,
        W=rows,
        D=_HEAD_DIM,
        NSK=(stages or (STAGES_K, STAGES_V))[0],
        NSV=(stages or (STAGES_K, STAGES_V))[1],
        RESCALE=RESCALE_LOG2 if rescale is None else float(rescale),
        num_warps=4,
    )
    return out


def union_prefill(q, packed_k, packed_v, indices, cu_q, cu_k, kv_lens, scale, *, extend_lens, kv_max,
                  rows=None, stages=None, rescale=None, out=None):
    """``sparse_gqa_fwd_interface_triton_ck``'s contract (indices relative to row cu_k[b] of the packed
    K/V) with the host extend lengths; the caller checks ``applies`` first. ``out``, when given, is a
    [rows, q_heads, 256] BF16 buffer with unit last stride that receives the result in place of a new
    tensor (every row of every group is stored, so no row keeps stale bytes)."""
    heads = packed_k.shape[1]
    rows = rows or rows_per_group(q.shape[1], heads)
    groups = group_count(extend_lens, rows)
    if q.shape[0] == 0 or groups == 0:
        return torch.empty_like(q) if out is None else out
    tok, mask, meta, nt = build_unions(
        indices, cu_q, kv_lens, cu_k, groups=groups, rows=rows, kv_max=kv_max, oob=packed_k.shape[0]
    )
    return attend(q, packed_k, packed_v, tok, mask, meta, nt, scale, groups=groups, rows=rows, stages=stages,
                  rescale=rescale, out=out)


def log_once(q, packed_k, indices) -> None:
    if _state["logged"]:
        return
    _state["logged"] = True
    heads = packed_k.shape[1]
    logger.info(
        "QSA union prefill: on (rows=%d heads=%d/%d topk=%d stages=%d,%d rescale_log2=%g "
        "min_rows=%d check_calls=%d) "
        "(SGLANG_ENABLE_QSA_UNION_PREFILL)",
        rows_per_group(q.shape[1], heads),
        q.shape[1],
        heads,
        indices.shape[-1],
        STAGES_K,
        STAGES_V,
        RESCALE_LOG2,
        MIN_ROWS,
        CHECK_CALLS,
    )


def compare(new: torch.Tensor, ref: torch.Tensor) -> dict:
    """Max abs and rel-L2 difference, and argmax agreement per (row, head) over the head dim."""
    d = new.float() - ref.float()
    am = new.float().argmax(-1)
    bm = ref.float().argmax(-1)
    return {
        "bitwise": bool(torch.equal(new, ref)),
        "max_abs": float(d.abs().nan_to_num(nan=float("inf")).max()) if d.numel() else 0.0,
        "rel_l2": float(d.norm() / ref.float().norm().clamp(min=1e-30)) if d.numel() else 0.0,
        "argmax_agree": float((am == bm).float().mean()) if am.numel() else 1.0,
        "nonfinite": int((~torch.isfinite(new)).sum()),
        "ref_nonfinite": int((~torch.isfinite(ref)).sum()),
    }


def checked(run_new, run_ref, rows: int, kv_max: int, ref_name: str):
    """Serve ``run_new``; the first ``CHECK_CALLS`` calls also time ``run_ref`` and log the comparison."""
    if _state["checked"] >= CHECK_CALLS:
        return run_new()
    _state["checked"] += 1
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    new = run_new()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ref = run_ref()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    c = compare(new, ref)
    logger.info(
        "QSA union prefill check %d/%d: rows=%d kv_max=%d union_ms=%.3f %s_ms=%.3f bitwise=%s rel_l2=%.3e "
        "max_abs=%.3e argmax_agree=%.6f nonfinite=%d ref_nonfinite=%d",
        _state["checked"],
        CHECK_CALLS,
        rows,
        kv_max,
        (t1 - t0) * 1e3,
        ref_name,
        (t2 - t1) * 1e3,
        c["bitwise"],
        c["rel_l2"],
        c["max_abs"],
        c["argmax_agree"],
        c["nonfinite"],
        c["ref_nonfinite"],
    )
    return new
