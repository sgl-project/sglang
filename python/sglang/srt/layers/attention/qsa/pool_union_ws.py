"""Warp-specialized FP8 pool-union verify attention (Gluon, tcgen05): kernel 2 of
qsa/pool_union_decode.py as a pipeline of specialized warps.

Opt-in (``SGLANG_QSA_FP8_POOL_VERIFY_WS=1``, on top of ``SGLANG_QSA_FP8_POOL_VERIFY=1``): the union
verify chain keeps kernel 1 (new K/V write), kernel U (each request's token union) and kernel 3 (the
split merge) with their buffers, and replaces kernel 2 (_qsa_pool_union_splitk) with one kernel per
(request, kv head, split) that writes the same per-split partials (m, l, acc per q row) for the merge.

Each CTA streams its share of the request's token list in 128-token tiles through four partitions,
chained by mbarriers:

    load    (4 warps)  TMA gather4 of the tile's FP8 K and V pool rows into a 2-stage FP8 ring;
                       tokens no row attends read a row past the pool, which TMA fills with zeros
                       (the zeros kernel 2's masked loads give);
    convert (4 warps)  E4M3 -> BF16 (exact): K into tensor memory, V into a 2-stage shared ring;
    mma     (1 warp)   S^T = K Q^T (K from tensor memory, M = 128 tokens, N = 64 q rows), then
                       O^T += V^T P^T and the row sums L += 1 P^T, all tcgen05 into tensor memory;
    softmax (4 warps)  masks S^T by each token's row bits, exponentiates, writes P^T (BF16), and
                       after the last tile the split's partials.

Swapping A and B puts the 128 tokens on the MMA's M (the W x GQ = 48 q rows, padded to 64, are N), so
the tensor cores run at full M; the gather and the conversion run ahead of them, so a tile costs its
slowest stage instead of the sum of the stages (kernel 2's serial ~4.2 us per 128-token iteration).

Math is kernel 2's: the same token list and row masks, the same scale, E4M3 -> BF16 exact, P rounded to
BF16 before PV, FP32 accumulation, and the counted-but-invalid entries (score 0, V 0) folded into split
0. What differs is rounding: the running max is applied lazily (O and L are rescaled only when a q row's
max grows by more than 2^RESCALE_LOG2, FlashAttention-4's conditional rescaling, so P <= 2^RESCALE_LOG2),
and the row sum adds the BF16 P that multiplies V. Not bitwise to kernel 2.

Patterns borrowed: the partitions, TMEM operands and mbarrier rings of Triton's Gluon attention example
(triton-lang/triton python/examples/gluon/01-attention-forward.py, v3.9.0), TMA gather4 as in its
python/tutorials/gluon/09-tma-gather-scatter.py, the swap-AB TMEM core of this tree's
kernels/ops/attention/qsa_tcgen05_v1.py, and the rescale threshold of FlashAttention-4
(Dao-AILab/flash-attention, flash_attn/cute).
"""

from __future__ import annotations

import logging
import os

import torch
import triton

from sglang.srt.layers.attention.qsa import pool_union_decode as union
from sglang.srt.layers.attention.qsa.pool_decode import LAST_KERNEL, LOG2E, _num_sms, _qsa_write_new_kv
from sglang.srt.layers.attention.qsa.pool_decode import FUSED_FP8_KV_WRITE, write_new_kv_e4m3

logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _env_num(name: str, default, cast):
    value = os.environ.get(name, "").strip()
    return cast(value) if value else default


# Read once at import, like the backend's other QSA switches; off by default.
ENABLED = _env_flag("SGLANG_QSA_FP8_POOL_VERIFY_WS")
# None: choose_ws_splits; a number pins the split count (bench and tuning only).
SPLITS = _env_num("SGLANG_QSA_FP8_POOL_VERIFY_WS_SPLITS", None, int)
# PDL into this kernel (gdc_wait after its setup); off by default until measured in the engine.
PDL = _env_flag("SGLANG_QSA_FP8_POOL_VERIFY_WS_PDL")
RESCALE_LOG2 = 8.0
_LOGGED_ROWS: set = set()

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
except Exception:  # a Triton without Gluon's Blackwell API: kernel 2 serves
    _GLUON = False


def supported(q, k_cache, width) -> bool:
    """The shapes this kernel serves; anything else keeps kernel 2."""
    if not _GLUON or q.device.type != "cuda" or torch.cuda.get_device_capability(q.device)[0] != 10:
        return False
    kv_heads, dim = k_cache.shape[1], k_cache.shape[2]
    qh = q.numel() // q.shape[0] // dim
    return (
        k_cache.dtype == torch.float8_e4m3fn
        and q.dtype == torch.bfloat16
        and dim == 256
        and qh % kv_heads == 0
        and 32 < width * (qh // kv_heads) <= 64
        and k_cache.is_contiguous()
    )


def choose_ws_splits(requests, kv_heads, sms):
    """Splits of a request's token list: the most that keep the grid within one wave (one CTA per SM),
    at most 4. Measured fastest on VR200 (212 SMs) at 44-64 requests: 2 at 44-48, 1 at 57-64."""
    pairs = max(1, requests * kv_heads)
    return max(1, min(4, sms // pairs))


def _log_engaged(rows: int, width: int, kv_heads: int, device, served: bool) -> None:
    """One line per verify row count, printed at graph capture (the runner's engagement check)."""
    if rows in _LOGGED_ROWS:
        return
    _LOGGED_ROWS.add(rows)
    requests = rows // width
    if served:
        logger.info(
            "SGLANG_QSA_FP8_POOL_VERIFY_WS engaged rows=%d width=%d requests=%d splits=%d pdl=%d",
            rows,
            width,
            requests,
            SPLITS or choose_ws_splits(requests, kv_heads, _num_sms(device)),
            int(PDL),
        )
    else:
        logger.info(
            "SGLANG_QSA_FP8_POOL_VERIFY_WS fallback rows=%d width=%d: shape or Triton unsupported (gluon=%d), "
            "kernel 2 serves",
            rows,
            width,
            int(_GLUON),
        )


if _GLUON:

    @gluon.constexpr_function
    def _tmem_reg_layout(dtype, shape, tmem_layout, alloc_shape, num_warps):
        ty = tensor_memory_descriptor_type(dtype, shape, tmem_layout, alloc_shape)
        return ty.get_reg_layout(num_warps=num_warps)

    @gluon.jit
    def _ws_rows(TS, TM, base, t, end, oob):
        # pool row of each listed token; tokens no row attends (mask 0) and the tile's tail read row oob
        tin = t < end
        tm = gl.load(TM + base + t, mask=tin, other=0)
        slot = gl.load(TS + base + t, mask=tin, other=0)
        return gl.where(tin & (tm != 0), slot, oob)

    @gluon.jit
    def _ws_load(k_desc, v_desc, TS, TM, k8, v8, f8_ready, f8_empty, r, head, start, end, n_tiles,
                 NT: gl.constexpr, D: gl.constexpr, BN: gl.constexpr, NS: gl.constexpr):
        flat: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
        # gather4 takes each 4-row chunk of offsets whole in every lane of one warp
        rows_layout: gl.constexpr = gl.SliceLayout(0, gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0]))
        HB: gl.constexpr = BN // 2
        oob = k_desc.shape[0]
        base = r.to(gl.int64) * NT
        col = head * D
        # the next tile's pool rows are loaded a tile ahead, off the gather's critical path
        t = start + gl.arange(0, HB, flat)
        ra = _ws_rows(TS, TM, base, t, end, oob)
        rb = _ws_rows(TS, TM, base, t + HB, end, oob)
        for i in range(n_tiles):
            s = i % NS
            cur_a = ra
            cur_b = rb
            t = start + (i + 1) * BN + gl.arange(0, HB, flat)
            ra = _ws_rows(TS, TM, base, t, end, oob)
            rb = _ws_rows(TS, TM, base, t + HB, end, oob)
            mbarrier.wait(f8_empty.index(s), ((i // NS) & 1) ^ 1)
            bar = f8_ready.index(s)
            mbarrier.expect(bar, 2 * BN * k_desc.block_type.nbytes)
            # half tiles keep the offsets, replicated in every lane, within the loader's registers
            rows = gl.convert_layout(cur_a, rows_layout)
            tma.async_gather(k_desc, rows, col, bar, k8.index(s).slice(0, HB))
            tma.async_gather(v_desc, rows, col, bar, v8.index(s).slice(0, HB))
            rows = gl.convert_layout(cur_b, rows_layout)
            tma.async_gather(k_desc, rows, col, bar, k8.index(s).slice(HB, HB))
            tma.async_gather(v_desc, rows, col, bar, v8.index(s).slice(HB, HB))

    @gluon.jit
    def _ws_convert(k8, v8, k_tmem, vb, f8_ready, f8_empty, k_ready, k_empty, v_ready, v_empty, n_tiles,
                    BN: gl.constexpr, D: gl.constexpr, NS: gl.constexpr, NBV: gl.constexpr, KC: gl.constexpr):
        # One token row per thread: the layout TMEM stores take, and conflict-free on the swizzled rows.
        k_tl: gl.constexpr = TensorMemoryLayout((BN, KC), col_stride=1)
        row_reg: gl.constexpr = _tmem_reg_layout(gl.bfloat16, [BN, KC], k_tl, [BN, D], gl.num_warps())
        for i in range(n_tiles):
            s = i % NS
            bv = i % NBV
            mbarrier.wait(f8_ready.index(s), (i // NS) & 1)
            # one K buffer: tile i's K overwrites tile i - 1's once its QK completed
            mbarrier.wait(k_empty.index(0), (i & 1) ^ 1)
            for c in gl.static_range(D // KC):
                x = k8.index(s).slice(c * KC, KC, dim=1).load(row_reg)
                k_tmem.slice(c * KC, KC).store(x.to(gl.bfloat16))
            mbarrier.arrive(k_ready.index(0))
            mbarrier.wait(v_empty.index(bv), ((i // NBV) & 1) ^ 1)
            for c in gl.static_range(D // KC):
                y = v8.index(s).slice(c * KC, KC, dim=1).load(row_reg)
                vb.index(bv).slice(c * KC, KC, dim=1).store(y.to(gl.bfloat16))
            # order the generic-proxy shared reads/writes with the TMA refill and the MMA reads
            fence_async_shared()
            mbarrier.arrive(f8_empty.index(s))
            mbarrier.arrive(v_ready.index(bv))

    @gluon.jit
    def _ws_pv(vb, p_smem, ones_smem, o_tmem, l_tmem, v_ready, v_empty, p_ready, p_free, j, NBV: gl.constexpr):
        bv = j % NBV
        mbarrier.wait(v_ready.index(bv), (j // NBV) & 1)
        mbarrier.wait(p_ready.index(0), j & 1)
        # O^T [D, QP] += V^T [D, BN] @ P^T [BN, QP]
        tcgen05_mma(vb.index(bv).permute((1, 0)), p_smem, o_tmem, use_acc=j > 0)
        # row sums on the tensor core: every row of L [LM, QP] += ones [LM, BN] @ P^T, P^T's column sums
        tcgen05_mma(ones_smem, p_smem, l_tmem, use_acc=j > 0, mbarriers=[p_free.index(0), v_empty.index(bv)])

    @gluon.jit
    def _ws_mma(k_tmem, q_smem, vb, p_smem, ones_smem, s_tmem, o_tmem, l_tmem, k_ready, k_empty, v_ready, v_empty,
                s_ready, p_ready, p_free, n_tiles, NBV: gl.constexpr):
        # QK of tile i is issued before PV of tile i - 1, so the softmax of tile i - 1 overlaps QK of tile i.
        # S buffer i % 2 last held tile i - 2, whose P this warp waited for before PV of tile i - 2.
        for i in range(n_tiles):
            b = i % 2
            mbarrier.wait(k_ready.index(0), i & 1)
            # S^T [BN, QP] = K [BN, D] (TMEM) @ Q^T [D, QP]
            tcgen05_mma(k_tmem, q_smem.permute((1, 0)), s_tmem.index(b), use_acc=False,
                        mbarriers=[s_ready.index(b), k_empty.index(0)])
            if i > 0:
                _ws_pv(vb, p_smem, ones_smem, o_tmem, l_tmem, v_ready, v_empty, p_ready, p_free, i - 1, NBV)
        if n_tiles > 0:
            _ws_pv(vb, p_smem, ones_smem, o_tmem, l_tmem, v_ready, v_empty, p_ready, p_free, n_tiles - 1, NBV)

    @gluon.jit
    def _ws_scale_o(o_tmem, a_smem, alpha, QP: gl.constexpr, OC: gl.constexpr, o_layout: gl.constexpr):
        # O^T columns times alpha, OC columns at a time (alpha goes through shared memory to each slice)
        a_smem.store(alpha)
        for c in gl.static_range(QP // OC):
            o_ref = o_tmem.slice(c * OC, OC)
            a_c = a_smem.slice(c * OC, OC).load(gl.SliceLayout(0, o_layout))
            o_ref.store(o_ref.load(o_layout) * a_c[None, :])

    @gluon.jit
    def _ws_softmax(s_tmem, o_tmem, l_tmem, p_smem, a_smem, s_ready, p_ready, p_free, TM, NINV, M_OUT, L_OUT,
                    ACC_OUT, qk_scale, r, head, split, start, end, n_tiles, SPLITS,
                    H: gl.constexpr, D: gl.constexpr, GQ: gl.constexpr, QP: gl.constexpr, W: gl.constexpr,
                    NT: gl.constexpr, BN: gl.constexpr, OC: gl.constexpr, LM: gl.constexpr, RESCALE: gl.constexpr):
        # S^T: one token per TMEM lane, one q row (request row x head) per column
        s_tl: gl.constexpr = TensorMemoryLayout((BN, QP), col_stride=1)
        s_layout: gl.constexpr = _tmem_reg_layout(gl.float32, [BN, QP], s_tl, [BN, QP], gl.num_warps())
        o_layout: gl.constexpr = _tmem_reg_layout(gl.float32, [D, OC], s_tl, [D, QP], gl.num_warps())
        l_layout: gl.constexpr = _tmem_reg_layout(gl.float32, [LM, QP], TensorMemoryLayout((LM, QP), col_stride=1),
                                                  [LM, QP], gl.num_warps())
        tok_l: gl.constexpr = gl.SliceLayout(1, s_layout)
        q_l: gl.constexpr = gl.SliceLayout(0, s_layout)
        g = gl.arange(0, QP, q_l)
        jq = g // GQ
        qv = jq < W
        m_i = gl.full([QP], float("-inf"), gl.float32, q_l)
        base = r.to(gl.int64) * NT
        lane = gl.arange(0, BN, tok_l)
        t = start + lane
        tm_next = gl.load(TM + base + t, mask=t < end, other=0)
        for i in range(n_tiles):
            b = i % 2
            # the next tile's row bits are loaded a tile ahead, off the softmax's critical path
            tm = tm_next
            t = start + (i + 1) * BN + lane
            tm_next = gl.load(TM + base + t, mask=t < end, other=0)
            mbarrier.wait(s_ready.index(b), (i // 2) & 1)
            s = s_tmem.index(b).load(s_layout)
            s = gl.where(((tm[:, None] >> jq[None, :]) & 1) != 0, s * qk_scale, float("-inf"))
            # The running max is stale by design: P = exp2(s - m) <= 2^RESCALE. Only a score above
            # m + RESCALE (or a q row's first finite score) takes the cross-lane max.
            over = s > (m_i + RESCALE)[None, :]
            if gl.max(gl.max(over.to(gl.int32), 1), 0) > 0:
                m_new = gl.maximum(m_i, gl.max(s, 0))
                grow = (m_new > m_i + RESCALE) & (m_i > float("-inf"))
                if gl.max(grow.to(gl.int32), 0) > 0:
                    mbarrier.wait(p_free.index(0), (i - 1) & 1)  # PV of tile i - 1 has landed in O and L
                    alpha = gl.where(grow, gl.exp2(m_i - m_new), 1.0)
                    _ws_scale_o(o_tmem, a_smem, alpha, QP, OC, o_layout)
                    l_tmem.store(l_tmem.load(l_layout) * gl.convert_layout(alpha, gl.SliceLayout(0, l_layout))[None, :])
                m_i = gl.where(grow | (m_i == float("-inf")), m_new, m_i)
            m_safe = gl.where(m_i == float("-inf"), 0.0, m_i)
            p = gl.exp2(s - m_safe[None, :])
            # the single P^T buffer is free once PV of tile i - 1 completed
            mbarrier.wait(p_free.index(0), (i & 1) ^ 1)
            p_smem.store(p.to(gl.bfloat16))
            fence_async_shared()
            mbarrier.arrive(p_ready.index(0))

        if n_tiles > 0:
            mbarrier.wait(p_free.index(0), (n_tiles - 1) & 1)  # the last PV (and row sums) landed in TMEM
            # every row of L holds the same sums
            l_i = gl.convert_layout(gl.max(l_tmem.load(l_layout), 0), q_l)
        else:
            l_i = gl.zeros([QP], gl.float32, q_l)
        # split 0 folds in the counted-but-invalid entries: score q . 0 = 0, V 0 (kernel 2's epilogue)
        ninv = gl.load(NINV + r * W + jq, mask=qv & (split == 0), other=0).to(gl.float32)
        has = ninv > 0
        m_fin = gl.where(has, gl.maximum(m_i, 0.0), m_i)
        m_safe = gl.where(m_fin == float("-inf"), 0.0, m_fin)
        alpha = gl.exp2(m_i - m_safe)
        l_i = l_i * alpha + gl.where(has, ninv * gl.exp2(0.0 - m_safe), 0.0)
        m_i = m_fin

        part = (r * H + head) * SPLITS + split
        gl.store(M_OUT + part * QP + g, m_i, mask=qv)
        gl.store(L_OUT + part * QP + g, l_i, mask=qv)
        # rows the merge reads: valid q rows with a finite max (alpha -1 marks the others)
        a_smem.store(gl.where(qv & (m_i > float("-inf")), alpha, -1.0))
        acc_base = ACC_OUT + part.to(gl.int64) * (QP * D)
        do = gl.arange(0, D, gl.SliceLayout(1, o_layout))
        for c in gl.static_range(QP // OC):
            a_c = a_smem.slice(c * OC, OC).load(gl.SliceLayout(0, o_layout))
            go = c * OC + gl.arange(0, OC, gl.SliceLayout(0, o_layout))
            if n_tiles > 0:
                o = o_tmem.slice(c * OC, OC).load(o_layout) * a_c[None, :]
            else:
                o = gl.zeros([D, OC], gl.float32, o_layout)
            gl.store(acc_base + go[None, :] * D + do[:, None], o, mask=(a_c >= 0.0)[None, :])

    @gluon.jit
    def _qsa_pool_union_ws(
        Q,
        k_desc,
        v_desc,
        TS,
        TM,
        NTOK,
        NINV,
        M_OUT,
        L_OUT,
        ACC_OUT,
        qk_scale,
        Q_STRIDE,
        SPLITS,
        H: gl.constexpr,
        D: gl.constexpr,
        GQ: gl.constexpr,
        QP: gl.constexpr,
        W: gl.constexpr,
        NT: gl.constexpr,
        RESCALE: gl.constexpr,
        USE_PDL: gl.constexpr,
    ):
        BN: gl.constexpr = 128  # tokens per tile: the MMA's M
        NS: gl.constexpr = 2  # FP8 K+V ring stages (64 KB each)
        NBV: gl.constexpr = 2  # BF16 V stages (64 KB each)
        LM: gl.constexpr = 64  # rows of the row-sum MMA
        KC: gl.constexpr = 128  # columns converted per step
        OC: gl.constexpr = 16  # O^T columns per TMEM load in the rescale and epilogue

        # Shared and tensor memory and barriers first: none of it reads the previous kernels' outputs.
        k8 = gl.allocate_shared_memory(gl.float8e4nv, [NS, BN, D], k_desc.layout)
        v8 = gl.allocate_shared_memory(gl.float8e4nv, [NS, BN, D], k_desc.layout)
        vb = gl.allocate_shared_memory(gl.bfloat16, [NBV, BN, D], gl.NVMMASharedLayout.get_default_for([BN, D], gl.bfloat16))
        p_smem = gl.allocate_shared_memory(gl.bfloat16, [BN, QP], gl.NVMMASharedLayout.get_default_for([BN, QP], gl.bfloat16))
        q_smem = gl.allocate_shared_memory(gl.bfloat16, [QP, D], gl.NVMMASharedLayout.get_default_for([QP, D], gl.bfloat16))
        a_smem = gl.allocate_shared_memory(gl.float32, [QP], gl.SwizzledSharedLayout(1, 1, 1, [0]))
        ones_smem = gl.allocate_shared_memory(
            gl.bfloat16,
            [LM, BN],
            gl.NVMMASharedLayout.get_default_for([LM, BN], gl.bfloat16),
            gl.full([LM, BN], 1.0, gl.bfloat16, gl.BlockedLayout([1, 8], [2, 16], [gl.num_warps(), 1], [1, 0])),
        )

        f8_ready = gl.allocate_shared_memory(gl.int64, [NS, 1], mbarrier.MBarrierLayout())
        f8_empty = gl.allocate_shared_memory(gl.int64, [NS, 1], mbarrier.MBarrierLayout())
        v_ready = gl.allocate_shared_memory(gl.int64, [NBV, 1], mbarrier.MBarrierLayout())
        v_empty = gl.allocate_shared_memory(gl.int64, [NBV, 1], mbarrier.MBarrierLayout())
        k_ready = gl.allocate_shared_memory(gl.int64, [1, 1], mbarrier.MBarrierLayout())
        k_empty = gl.allocate_shared_memory(gl.int64, [1, 1], mbarrier.MBarrierLayout())
        p_ready = gl.allocate_shared_memory(gl.int64, [1, 1], mbarrier.MBarrierLayout())
        p_free = gl.allocate_shared_memory(gl.int64, [1, 1], mbarrier.MBarrierLayout())
        s_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
        for i in gl.static_range(NS):
            mbarrier.init(f8_ready.index(i), count=1)
            mbarrier.init(f8_empty.index(i), count=1)
        for i in gl.static_range(NBV):
            mbarrier.init(v_ready.index(i), count=1)
            mbarrier.init(v_empty.index(i), count=1)
        mbarrier.init(k_ready.index(0), count=1)
        mbarrier.init(k_empty.index(0), count=1)
        mbarrier.init(p_ready.index(0), count=1)
        mbarrier.init(p_free.index(0), count=1)
        for i in gl.static_range(2):
            mbarrier.init(s_ready.index(i), count=1)
        # TMEM columns: K 128 (BF16 packed), S^T 2 x 64, O^T 2 x 64, L 64
        k_tmem = allocate_tensor_memory(gl.bfloat16, [BN, D], TensorMemoryLayout((BN, KC), col_stride=1))
        s_tmem = allocate_tensor_memory(gl.float32, [2, BN, QP], TensorMemoryLayout((BN, QP), col_stride=1))
        o_tmem = allocate_tensor_memory(gl.float32, [D, QP], TensorMemoryLayout((BN, QP), col_stride=1))
        l_tmem = allocate_tensor_memory(gl.float32, [LM, QP], TensorMemoryLayout((LM, QP), col_stride=1))

        if USE_PDL:
            triton.language.extra.cuda.gdc_wait()
        r = gl.program_id(0)
        head = gl.program_id(1)
        split = gl.program_id(2)
        ntok = gl.load(NTOK + r)
        chunk = gl.cdiv(gl.cdiv(ntok, SPLITS), BN) * BN
        start = split * chunk
        end = gl.minimum(start + chunk, ntok)
        n_tiles = gl.where(end > start, (end - start + BN - 1) // BN, 0)

        # Q: the request's W rows x GQ heads of this kv head, padded to QP rows, loaded once
        q_ld: gl.constexpr = gl.BlockedLayout([1, 8], [2, 16], [gl.num_warps(), 1], [1, 0])
        gq_ = gl.arange(0, QP, gl.SliceLayout(1, q_ld))
        dq = gl.arange(0, D, gl.SliceLayout(0, q_ld))
        jq = gq_ // GQ
        qrow = r * W + jq
        q = gl.load(
            Q + qrow.to(gl.int64)[:, None] * Q_STRIDE + (head * GQ + gq_ % GQ)[:, None] * D + dq[None, :],
            mask=(jq < W)[:, None],
            other=0.0,
        )
        q_smem.store(q)
        fence_async_shared()

        gl.warp_specialize(
            [
                (_ws_softmax, (s_tmem, o_tmem, l_tmem, p_smem, a_smem, s_ready, p_ready, p_free, TM, NINV, M_OUT,
                               L_OUT, ACC_OUT, qk_scale, r, head, split, start, end, n_tiles, SPLITS, H, D, GQ, QP, W,
                               NT, BN, OC, LM, RESCALE)),
                (_ws_convert, (k8, v8, k_tmem, vb, f8_ready, f8_empty, k_ready, k_empty, v_ready, v_empty, n_tiles,
                               BN, D, NS, NBV, KC)),
                (_ws_load, (k_desc, v_desc, TS, TM, k8, v8, f8_ready, f8_empty, r, head, start, end, n_tiles, NT,
                            D, BN, NS)),
                (_ws_mma, (k_tmem, q_smem, vb, p_smem, ones_smem, s_tmem, o_tmem, l_tmem, k_ready, k_empty, v_ready,
                           v_empty, s_ready, p_ready, p_free, n_tiles, NBV)),
            ],
            [4, 4, 1],
            # registers per thread; the 4 softmax warps take what is left of 16 warps x 128
            [112, 56, 56],
        )


_DESC_CACHE: dict = {}


def _pool_desc(cache):
    """The pool as [slots, H * D] FP8 with gather4 rows of D bytes; built once per pool."""
    key = (cache.data_ptr(), tuple(cache.shape))
    desc = _DESC_CACHE.get(key)
    if desc is None:
        slots, kv_heads, dim = cache.shape
        layout = gl.NVMMASharedLayout.get_default_for([128, dim], gl.float8e4nv)
        desc = TensorDescriptor.from_tensor(cache.view(slots, kv_heads * dim), [1, dim], layout)
        _DESC_CACHE[key] = desc
    return desc


def union_ws_attend(q3, k_cache, v_cache, tok_slots, tok_masks, ntok, ninv, out, requests, width, scale, nt,
                    splits=None, pdl=False):
    """The warp-specialized kernel 2 and kernel 3 (merge) over kernel U's lists; writes out [rows, QH * D]."""
    rows = q3.shape[0]
    kv_heads, dim = k_cache.shape[1], k_cache.shape[2]
    qh = q3.shape[1]
    gq = qh // kv_heads
    qp = 64
    if splits is None:
        splits = choose_ws_splits(requests, kv_heads, _num_sms(q3.device))
    nparts = requests * kv_heads * splits
    dev = q3.device
    m_buf = torch.empty(nparts * qp, dtype=torch.float32, device=dev)
    l_buf = torch.empty(nparts * qp, dtype=torch.float32, device=dev)
    acc_buf = torch.empty(nparts * qp * dim, dtype=torch.float32, device=dev)
    LAST_KERNEL["union_ws"] = _qsa_pool_union_ws[(requests, kv_heads, splits)](
        q3,
        _pool_desc(k_cache),
        _pool_desc(v_cache),
        tok_slots,
        tok_masks,
        ntok,
        ninv,
        m_buf,
        l_buf,
        acc_buf,
        float(scale) * LOG2E,
        q3.stride(0),
        splits,
        kv_heads,
        dim,
        gq,
        qp,
        width,
        nt,
        RESCALE_LOG2,
        bool(pdl),
        num_warps=4,
        maxnreg=128,
        **({"launch_pdl": True} if pdl else {}),
    )
    sb = min(16, triton.next_power_of_2(splits))
    union._qsa_pool_union_merge[(rows * qh,)](
        m_buf, l_buf, acc_buf, out, out.stride(0), kv_heads, dim, gq, qp, width, splits, sb, True,
        num_warps=2, launch_pdl=True,
    )
    return out


def qsa_pool_union_decode_ws(
    q,
    k_cache,
    v_cache,
    k_new,
    v_new,
    loc,
    req_to_token,
    row_requests,
    block_indices,
    sequence_lens,
    query_positions,
    width,
    scale,
    ratio=4,
    out=None,
    pdl=True,
    write_new_kv=True,
    **launch,
):
    """union.qsa_pool_union_decode with kernel 2 warp-specialized; the same arguments and result.

    Shapes the kernel does not serve (see supported) take union.qsa_pool_union_decode unchanged, with
    ``launch`` (union.LAUNCH) as its launch config."""
    served = supported(q, k_cache, width)
    _log_engaged(block_indices.shape[0], width, k_cache.shape[1], q.device, served)
    if not served:
        return union.qsa_pool_union_decode(
            q, k_cache, v_cache, k_new, v_new, loc, req_to_token, row_requests, block_indices, sequence_lens,
            query_positions, width, scale, ratio=ratio, out=out, pdl=pdl, write_new_kv=write_new_kv, **launch,
        )
    rows, ib = block_indices.shape
    kv_heads, dim = k_cache.shape[1], k_cache.shape[2]
    q3 = q.view(rows, -1, dim)
    qh = q3.shape[1]
    assert rows % width == 0
    assert q3.stride(2) == 1 and q3.stride(1) == dim
    assert k_new.stride(-1) == 1 and k_new.shape[-1] * k_new.shape[-2] == kv_heads * dim
    assert k_new.dtype == v_new.dtype == k_cache.dtype == v_cache.dtype or (
        FUSED_FP8_KV_WRITE and k_new.dtype == v_new.dtype == torch.bfloat16 and k_cache.dtype == v_cache.dtype
    )
    if out is None:
        out = torch.empty(rows, qh * dim, dtype=q.dtype, device=q.device)
    if rows == 0:
        return out
    requests = rows // width
    nb, tb, nt = union.union_buffers(rows, width, ib, ratio)
    dev = q.device
    tok_slots = torch.empty(requests * nt, dtype=torch.int32, device=dev)
    tok_masks = torch.empty(requests * nt, dtype=torch.int32, device=dev)
    scratch = torch.empty(requests * nb, dtype=torch.int32, device=dev)
    ntok = torch.empty(requests, dtype=torch.int32, device=dev)
    ninv = torch.empty(rows, dtype=torch.int32, device=dev)
    pdl_kw = {"launch_pdl": True} if pdl else {}
    if write_new_kv:
        nk = k_new.reshape(rows, kv_heads * dim) if k_new.dim() == 3 else k_new
        nv = v_new.reshape(rows, kv_heads * dim) if v_new.dim() == 3 else v_new
        if nk.dtype != k_cache.dtype:  # SGLANG_QSA_FUSED_FP8_KV_WRITE: BF16 rows, cast in the write
            write_new_kv_e4m3(k_cache, v_cache, nk, nv, loc, rows, kv_heads, dim, pdl, pdl_kw)
        else:
            _qsa_write_new_kv[(rows, kv_heads)](
                k_cache, v_cache, nk, nv, loc, nk.stride(0), nv.stride(0), kv_heads, dim, pdl, num_warps=1, **pdl_kw
            )
    union._qsa_union_tokens[(requests,)](
        block_indices,
        block_indices.stride(0),
        sequence_lens,
        query_positions,
        loc,
        req_to_token,
        req_to_token.stride(0),
        row_requests,
        tok_slots,
        tok_masks,
        ntok,
        ninv,
        scratch,
        ib,
        width,
        ratio,
        nb,
        nt,
        tb,
        pdl,
        # kernel U's warps: union.UNION_WARPS where that knob exists (the H452 rider), else kernel U's 8
        num_warps=union.__dict__.get("UNION_WARPS", 8),
        **pdl_kw,
    )
    return union_ws_attend(
        q3, k_cache, v_cache, tok_slots, tok_masks, ntok, ninv, out, requests, width, scale, nt,
        splits=SPLITS, pdl=pdl and PDL,
    )
