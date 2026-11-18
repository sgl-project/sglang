"""Paged causal prefill for gfx1250 on uTLX: GQA 8, head dim 64, BF16 queries.

Ported from the TLX kernel in facebookexperimental/triton#5000. K/V are BF16 or FP8
token-major caches [pages, PAGE, kv_heads, 64] with PAGE 16 or 64; every 64-key tile
gathers 64 // PAGE pages. FP8 values dequantize exactly to BF16 after the LDS load; the
K scale folds into Q once per tile and the V scale into the output.

Normalize the same BF16 probabilities used by PV; both sums accumulate in FP32.
Abnormal ranges replay stable online softmax in smaller query chunks. The interior
omits masking when causal and window validity follow from its bounds, peeling both
possible left-edge window pages. Q descriptor bounds cover ragged tails. Windows count
keys including self; window 0 means global. Cached prefixes use bottom-right causal alignment.
"""

import torch
import triton
import triton.language as tl
import utlx_plugin as tlx
from utlx_plugin.layout_ops import _register_layout

HEAD_DIM = 64
GQA = 8
# Muse Spark's sliding layers see 2047 keys; #5000 serves 2048. Both include self.
WINDOWS = (2047, 2048)
PAGE_SIZES = (16, 64)
KV_DTYPES = (torch.bfloat16, torch.float8_e4m3fn)


class _WmmaLayout(_register_layout):
    """An explicit gfx1250 WMMA accumulator layout; uTLX only provides MFMA layouts."""

    def __init__(self, warp_bases, reg_bases, instr_shape):
        super().__init__()
        self.warp_bases = [list(b) for b in warp_bases]
        self.reg_bases = [list(b) for b in reg_bases]
        self.instr_shape = list(instr_shape)

    # Constexpr arguments key the JIT cache by repr; the uTLX base repr is the class name alone.
    def __repr__(self):
        return f"_WmmaLayout({vars(self)!r})"

    def __eq__(self, other):
        return isinstance(other, _WmmaLayout) and vars(self) == vars(other)

    def __hash__(self):
        return hash(repr(self))

    def to_ir(self, builder, *_):
        return builder.get_amd_wmma_layout(
            3, True, self.warp_bases, self.reg_bases, [], self.instr_shape, 2
        )


# Layout values include every field in the compilation cache key.
WMMA_TWO = tl.constexpr(_WmmaLayout([[1, 0]], [], (16, 16, 32)))
WMMA_FOUR = tl.constexpr(_WmmaLayout([[2, 0], [4, 0]], [[0, 1], [1, 0]], (16, 16, 32)))


@triton.jit
def _dot(a, b, c):
    layout: tl.constexpr = WMMA_TWO if tlx.num_warps() == 2 else WMMA_FOUR
    aa = tlx.require_layout(a, tlx.dot_operand_layout(0, layout, k_width=8))
    bb = tlx.require_layout(b, tlx.dot_operand_layout(1, layout, k_width=8))
    cc = tlx.require_layout(c, layout)
    return tl.dot(aa, bb, cc)


@triton.jit
def _scale_online(scores):
    # The max and exponent must use the same rounded FP32 scores. Prevent
    # fusion into score*scale-center after the maximum has been rounded.
    return tl.inline_asm_elementwise(
        "v_mul_f32 $0, 0x3e38aa3b, $1;",
        constraints="=v,v",
        args=[scores],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mask_v(v, key_start, SK):
    # An inferred tail-mask layout can propagate back to V loads and add
    # LDS conversions. Keep four-wave masking in the dot-operand layout.
    if tlx.num_warps() == 4:
        layout: tl.constexpr = tlx.dot_operand_layout(1, WMMA_FOUR, k_width=8)
        idx = tlx.require_layout(tl.arange(0, 64), tlx.slice_layout(layout, 1))
        v = tlx.require_layout(v, layout)
    else:
        idx = tl.arange(0, 64)
    return tl.where((key_start + idx)[:, None] < SK, v, 0.0)


@triton.jit
def _consume(
    q,
    k,
    v,
    acc,
    maximum,
    denom,
    rows,
    key_start,
    SQ,
    SK,
    WINDOW: tl.constexpr,
    BM: tl.constexpr,
    FIXED: tl.constexpr,
    MASK: tl.constexpr = True,
):
    scores = _dot(q, k, tl.zeros((q.shape[0], k.shape[1]), tl.float32))
    if FIXED:
        scores *= 0.125 * 1.4426950408889634
    else:
        scores = _scale_online(scores)
    if MASK:
        keys = key_start + tl.arange(0, 64)
        valid = (keys[None, :] <= rows[:, None] + SK - SQ) & (keys[None, :] < SK)
        if WINDOW:
            valid &= keys[None, :] > rows[:, None] + SK - SQ - WINDOW
        scores = tl.where(valid, scores, -float("inf"))
    if SK % 64:
        # Zero masked values too: an unwritten final-page slot may hold NaN,
        # and zero probability alone does not prevent 0 * NaN in PV.
        v = _mask_v(v, key_start, SK)
    if FIXED:
        p = tl.exp2(scores - maximum[:, None])
        denom = _dot(p.to(q.dtype), tl.full((64, 16), 1.0, q.dtype), denom)
        acc = _dot(p.to(q.dtype), v, acc)
        return acc, maximum, denom
    else:
        new_max = tl.maximum(maximum, tl.max(scores, 1))
        # Only an all-masked row needs a finite center. Clamping finite
        # negative maxima would underflow every probability during recovery.
        center = tl.where(new_max == -float("inf"), 0.0, new_max)
        p = tl.exp2(scores - center[:, None])
        alpha = tl.exp2(maximum - center)
        denom = denom * alpha + tl.sum(p, 1)
        acc *= alpha[:, None]
        acc = _dot(p.to(q.dtype), v, acc)
        return acc, new_max, denom


@triton.jit
def _scores(
    q,
    k,
    rows,
    page,
    first_q,
    last_q,
    SQ,
    SK,
    WINDOW: tl.constexpr,
    BM: tl.constexpr,
    SPLIT: tl.constexpr,
):
    scores = _dot(q, k, tl.zeros((q.shape[0], k.shape[1]), tl.float32)) * (
        0.125 * 1.4426950408889634
    )
    partial = True
    if SPLIT:
        full = page * 64 + 63 <= first_q
        if WINDOW:
            full &= page * 64 > last_q - 1 - WINDOW
        partial = not full
    if partial:
        keys = page * 64 + tl.arange(0, 64)
        valid = (keys[None, :] <= rows[:, None] + SK - SQ) & (keys[None, :] < SK)
        if WINDOW:
            valid &= keys[None, :] > rows[:, None] + SK - SQ - WINDOW
        scores = tl.where(valid, scores, -float("inf"))
    return scores


@triton.jit
def _finish(scores, denom, maximum, BM: tl.constexpr, DTYPE: tl.constexpr):
    p = tl.exp2(scores - maximum[:, None])
    return p.to(DTYPE), denom


@triton.jit
def _prob(
    q,
    k,
    denom,
    maximum,
    rows,
    page,
    first_q,
    last_q,
    SQ,
    SK,
    WINDOW: tl.constexpr,
    BM: tl.constexpr,
    SPLIT: tl.constexpr,
):
    # This helper consumes only the final two pages. They are entirely
    # above a 2048-token window's lower edge; only causality can mask them.
    scores = _scores(q, k, rows, page, first_q, last_q, SQ, SK, 0, BM, SPLIT)
    return _finish(scores, denom, maximum, BM, q.dtype)


@triton.jit
def _view_k(kb, slot):
    return tlx.local_load(tlx.local_trans(tlx.local_view(kb, slot)))


@triton.jit
def _view_v(vb, slot):
    v = tlx.local_load(tlx.local_view(vb, slot))
    if tlx.num_warps() == 4:
        v = tlx.require_layout(v, tlx.dot_operand_layout(1, WMMA_FOUR, k_width=8))
    return v


@triton.jit
def _dequant(src, dst, slot):
    # All waves convert one FP8 tile into the BF16 stage with 16-byte LDS reads.
    # Dequantizing in the dot-operand layout instead stores every wave's own
    # converted copy back to LDS before its operand loads.
    # Rows of an A operand hold 8 contiguous columns per lane. uTLX ignores
    # linear register layouts here and falls back to one element per lane.
    layout: tl.constexpr = WMMA_TWO if tlx.num_warps() == 2 else WMMA_FOUR
    rows: tl.constexpr = tlx.dot_operand_layout(0, layout, k_width=8)
    x = tlx.require_layout(tlx.local_load(tlx.local_view(src, slot)), rows)
    y = tlx.require_layout(x.to(tl.bfloat16), rows)
    tlx.local_store(tlx.local_view(dst, slot), y)
    tlx.workgroup_barrier()


@triton.jit
def _get_k(kb, k16, slot, wait: tl.constexpr, SUB: tl.constexpr, FP8: tl.constexpr):
    # Each 64-key tile is SUB TDM copies; waits count tiles queued after this one.
    tlx.tdm_async_wait(wait * SUB)
    if FP8:
        _dequant(kb, k16, slot)
    return _view_k(k16, slot)


@triton.jit
def _get_v(vb, v16, slot, wait: tl.constexpr, SUB: tl.constexpr, FP8: tl.constexpr):
    tlx.tdm_async_wait(wait * SUB)
    if FP8:
        _dequant(vb, v16, slot)
    v = _view_v(v16, slot)
    # Finish every wave's LDS reads before reusing a stage.
    tlx.workgroup_barrier()
    return v


@triton.jit
def _put(
    Cache,
    pages_view,
    slot,
    Table,
    tile,
    NPAGES,
    KH: tl.constexpr,
    PAGE: tl.constexpr,
):
    # Queue 64-key tile `tile` into stage `slot`. uTLX drops a descriptor's shared
    # layout when it crosses a call, so the descriptor is built where it is used.
    desc = tlx.make_tensor_descriptor(
        None, Cache, [NPAGES * PAGE, 64], [KH * 64, 1], [PAGE, 64]
    )
    # The table covers whole tiles. IDs past a sequence's pages feed only masked
    # keys, and TDM zero-fills rows outside the cache; masked loads would branch
    # and expose each scalar load's latency.
    for i in tl.static_range(64 // PAGE):
        phys = tl.load(Table + tile * (64 // PAGE) + i)
        tlx.tdm_async_load(
            desc, tlx.local_view(pages_view, slot * (64 // PAGE) + i), [phys * PAGE, 0]
        )


@triton.jit
def _load_q(
    qb, BM: tl.constexpr, GROUP: tl.constexpr, k_scale, HAS_SCALE: tl.constexpr
):
    rows: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
        [(GROUP * 64, 8)], [BM, 64], [1, 0]
    )
    q = tlx.local_load(
        tlx.local_reinterpret(tlx.local_view(qb, 0), tl.bfloat16, [BM, 64], layout=rows)
    )
    if HAS_SCALE:
        q = (q.to(tl.float32) * k_scale).to(tl.bfloat16)
    return q


@triton.jit
def _pass(
    Q,
    K,
    V,
    Tables,
    Out,
    SQ,
    SK,
    NPAGES,
    KScale,
    VScale,
    KH: tl.constexpr,
    TABLE_STRIDE,
    WINDOW: tl.constexpr,
    BM: tl.constexpr,
    GROUP: tl.constexpr,
    SPLIT_MASK: tl.constexpr,
    REVERSE: tl.constexpr,
    FIXED: tl.constexpr,
    DEEP: tl.constexpr,
    PAGE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    FACTOR: tl.constexpr = 1,
    CHUNK=0,
    Q_START=0,
    SAMPLED: tl.constexpr = True,
):
    batch = tl.program_id(2)
    blocks = tl.num_programs(0)
    linear = tl.program_id(0) + blocks * tl.program_id(1)
    block = linear // (KH * 8 // GROUP)
    if REVERSE:
        block = blocks - 1 - block
    block = block * FACTOR + CHUNK
    head = (linear % (KH * 8 // GROUP)) * GROUP
    kv_head = head // 8
    row_ids = block * BM + tl.arange(0, BM)
    rows = row_ids // GROUP
    heads = head + row_ids % GROUP
    dims = tl.arange(0, 64)
    q_start = Q_START.to(tl.int64)
    offsets = ((q_start + rows[:, None]) * (KH * 8) + heads[:, None]) * 64 + dims[
        None, :
    ]
    qd = tlx.make_tensor_descriptor(
        None,
        Q + q_start * KH * 8 * 64 + head * 64,
        [SQ, GROUP * 64],
        [KH * 8 * 64, 1],
        [BM // GROUP, GROUP * 64],
    )
    q_tile: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
        [(GROUP * 64, 8)], [BM // GROUP, GROUP * 64], [1, 0]
    )
    qb = tlx.local_alloc((BM // GROUP, GROUP * 64), tl.bfloat16, 1, layout=q_tile)
    tlx.tdm_async_load(qd, tlx.local_view(qb, 0), [block * (BM // GROUP), 0])

    # Two stages of 64x64 K and V tiles. A page-16 cache fills each stage with four
    # TDM page copies through a page view of the same allocation; indexed views carry
    # physical LDS bases, unlike local_slice, whose offsets this TDM lowering drops.
    SUB: tl.constexpr = 64 // PAGE
    FP8: tl.constexpr = K.dtype.element_ty.is_fp8()
    PAD: tl.constexpr = 16 if FP8 else 8
    tile_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
        [(64, PAD)], [64, 64], [1, 0]
    )
    if PAGE == 64:
        kb = tlx.local_alloc((64, 64), K.dtype.element_ty, 2, layout=tile_layout)
        vb = tlx.local_alloc((64, 64), K.dtype.element_ty, 2, layout=tile_layout)
        kp = kb
        vp = vb
    else:
        whole_layout: tl.constexpr = (
            tlx.padded_shared_layout_encoding.with_identity_for(
                [(64, PAD)], [128, 64], [1, 0]
            )
        )
        page_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
            [(64, PAD)], [PAGE, 64], [1, 0]
        )
        k_whole = tlx.local_view(
            tlx.local_alloc((128, 64), K.dtype.element_ty, 1, layout=whole_layout), 0
        )
        v_whole = tlx.local_view(
            tlx.local_alloc((128, 64), K.dtype.element_ty, 1, layout=whole_layout), 0
        )
        kp = tlx.local_reinterpret(
            k_whole, K.dtype.element_ty, [2 * SUB, PAGE, 64], layout=page_layout
        )
        vp = tlx.local_reinterpret(
            v_whole, K.dtype.element_ty, [2 * SUB, PAGE, 64], layout=page_layout
        )
        kb = tlx.local_reinterpret(
            k_whole, K.dtype.element_ty, [2, 64, 64], layout=tile_layout
        )
        vb = tlx.local_reinterpret(
            v_whole, K.dtype.element_ty, [2, 64, 64], layout=tile_layout
        )
    if FP8:
        stage: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
            [(64, 8)], [64, 64], [1, 0]
        )
        k16 = tlx.local_alloc((64, 64), tl.bfloat16, 2, layout=stage)
        v16 = tlx.local_alloc((64, 64), tl.bfloat16, 2, layout=stage)
    else:
        k16 = kb
        v16 = vb
    Kh = K + kv_head * 64
    Vh = V + kv_head * 64
    Table = Tables + batch * TABLE_STRIDE
    k_scale = tl.load(KScale) if HAS_SCALE else 1.0

    first_q = block * (BM // GROUP) + SK - SQ
    last_q = tl.minimum(first_q + BM // GROUP, SK)
    first_page = 0
    if WINDOW:
        first_page = tl.maximum(0, (first_q - WINDOW + 1) // 64)
    end_page = tl.cdiv(last_q, 64)
    maximum = tl.full([BM], 0.0 if FIXED else -float("inf"), tl.float32)
    if FIXED and DEEP:
        use_deep = end_page - first_page >= (4 if WINDOW else 3)
    else:
        use_deep: tl.constexpr = False
    if FIXED and SAMPLED and not use_deep:
        # Sample both ends of the visible range. A fixed center avoids online
        # rescaling in the main loop while keeping ordinary large logits away
        # from the replay threshold. Unseen extremes still use stable replay.
        reference_page = tl.maximum(first_page, (first_q + 1) // 64 - 1)
        reference_first = first_page
        if WINDOW:
            reference_first = tl.minimum(
                reference_page, tl.cdiv(tl.maximum(0, last_q - WINDOW), 64)
            )
        _put(Kh, kp, 0, Table, reference_first, NPAGES, KH, PAGE)
        _put(Kh, kp, 1, Table, reference_page, NPAGES, KH, PAGE)
        tlx.tdm_async_wait(2 * SUB)
        q = _load_q(qb, BM, GROUP, k_scale, HAS_SCALE)
        k_first = _get_k(kb, k16, 0, 1, SUB, FP8)
        first_scores = _scores(
            q, k_first, rows, reference_first, first_q, last_q, SQ, SK, WINDOW, BM, True
        )
        first_max = tl.max(first_scores, 1)
        k_last = _get_k(kb, k16, 1, 0, SUB, FP8)
        last_scores = _scores(
            q, k_last, rows, reference_page, first_q, last_q, SQ, SK, WINDOW, BM, True
        )
        maximum = tl.maximum(first_max, tl.max(last_scores, 1))
        maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
        # Bound the center before contraction of dot*scale-center into FMA.
        # Large equal logits otherwise acquire different rounding residues
        # in sampled and interior pages. Extreme ranges still trigger replay.
        # Extra exponent headroom covers sparse peaks missed by both samples.
        maximum = tl.minimum(tl.maximum(maximum + 16.0, -64.0), 64.0)
        tlx.workgroup_barrier()
    if FIXED:
        denom = tl.zeros([BM, 16], tl.float32)
    else:
        denom = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, 64], tl.float32)
    if use_deep:
        _put(Kh, kp, 0, Table, first_page, NPAGES, KH, PAGE)
        _put(Kh, kp, 1, Table, first_page + 1, NPAGES, KH, PAGE)
        _put(Vh, vp, 0, Table, first_page, NPAGES, KH, PAGE)
        tlx.tdm_async_wait(3 * SUB)
        q = _load_q(qb, BM, GROUP, k_scale, HAS_SCALE)
        tlx.workgroup_barrier()
        k = _get_k(kb, k16, 0, 2, SUB, FP8)
        tlx.workgroup_barrier()
        if BM // GROUP <= 64:
            first_scores = _dot(q, k, tl.zeros((BM, 64), tl.float32)) * (
                0.125 * 1.4426950408889634
            )
            if WINDOW:
                keys = first_page * 64 + tl.arange(0, 64)
                first_scores = tl.where(
                    keys[None, :] > rows[:, None] + SK - SQ - WINDOW,
                    first_scores,
                    -float("inf"),
                )
        else:
            first_scores = _scores(
                q, k, rows, first_page, first_q, last_q, SQ, SK, WINDOW, BM, SPLIT_MASK
            )
        if SAMPLED:
            first_max = tl.max(first_scores, 1)
            if WINDOW:
                # Both left-edge pages are already queued. Reuse their QK
                # scores for the reference and the two peeled iterations.
                reference_page = first_page + 1
                reference_k = _get_k(kb, k16, 1, 1, SUB, FP8)
            else:
                reference_page = tl.maximum(first_page, (first_q + 1) // 64 - 1)
                _put(Kh, kp, 0, Table, reference_page, NPAGES, KH, PAGE)
                reference_k = _get_k(kb, k16, 0, 0, SUB, FP8)
            if BM // GROUP <= 64:
                reference_scores = _dot(
                    q, reference_k, tl.zeros((BM, 64), tl.float32)
                ) * (0.125 * 1.4426950408889634)
            else:
                reference_scores = _scores(
                    q,
                    reference_k,
                    rows,
                    reference_page,
                    first_q,
                    last_q,
                    SQ,
                    SK,
                    WINDOW,
                    BM,
                    True,
                )
            if WINDOW:
                keys = reference_page * 64 + tl.arange(0, 64)
                reference_scores = tl.where(
                    keys[None, :] > rows[:, None] + SK - SQ - WINDOW,
                    reference_scores,
                    -float("inf"),
                )
                next_scores = reference_scores
            maximum = tl.maximum(first_max, tl.max(reference_scores, 1))
            maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
            maximum = tl.minimum(tl.maximum(maximum + 16.0, -64.0), 64.0)
            tlx.workgroup_barrier()
        p, denom = _finish(first_scores, denom, maximum, BM, q.dtype)
        _put(Kh, kp, 0, Table, first_page + 2, NPAGES, KH, PAGE)
        _put(Vh, vp, 1, Table, first_page + 1, NPAGES, KH, PAGE)
        if not (WINDOW and SAMPLED):
            k = _get_k(kb, k16, 1, 3, SUB, FP8)
        slot = 0
        body_begin = first_page
        v_tile = first_page + 2
        if WINDOW:
            # Peel the second possible lower-edge page. The steady loop
            # then has no masks or per-iteration visibility branches, even
            # when runtime lengths place a query tile across a page boundary.
            if not SAMPLED:
                if BM // GROUP <= 64:
                    next_scores = _dot(q, k, tl.zeros((BM, 64), tl.float32)) * (
                        0.125 * 1.4426950408889634
                    )
                    keys = (first_page + 1) * 64 + tl.arange(0, 64)
                    next_scores = tl.where(
                        keys[None, :] > rows[:, None] + SK - SQ - WINDOW,
                        next_scores,
                        -float("inf"),
                    )
                else:
                    next_scores = _scores(
                        q,
                        k,
                        rows,
                        first_page + 1,
                        first_q,
                        last_q,
                        SQ,
                        SK,
                        WINDOW,
                        BM,
                        True,
                    )
            v = _get_v(vb, v16, 0, 2, SUB, FP8)
            _put(Kh, kp, 1, Table, first_page + 3, NPAGES, KH, PAGE)
            denom = _dot(p, tl.full((64, 16), 1.0, q.dtype), denom)
            acc = _dot(p, v, acc)
            p, denom = _finish(next_scores, denom, maximum, BM, q.dtype)
            k = _get_k(kb, k16, 0, 2, SUB, FP8)
            _put(Vh, vp, 0, Table, v_tile, NPAGES, KH, PAGE)
            v_tile = first_page + 3
            slot = 1
            body_begin += 1
        for page in range(body_begin, end_page - 3):
            # With <=64 distinct query positions, every key in this
            # prologue-excluded iteration is strictly before the earliest query.
            # At most two pages intersect the left window boundary, even
            # when a runtime prefix places the query tile across a page.
            # The prologue and peeled step covered both lower-edge pages.
            if BM // GROUP <= 64:
                next_scores = _dot(
                    q, k, tl.zeros((q.shape[0], k.shape[1]), tl.float32)
                ) * (0.125 * 1.4426950408889634)
            else:
                next_scores = _scores(
                    q,
                    k,
                    rows,
                    page + 1,
                    first_q,
                    last_q,
                    SQ,
                    SK,
                    WINDOW,
                    BM,
                    SPLIT_MASK,
                )
            v = _get_v(vb, v16, slot, 2, SUB, FP8)
            _put(Kh, kp, 1 - slot, Table, page + 3, NPAGES, KH, PAGE)
            denom = _dot(p, tl.full((64, 16), 1.0, q.dtype), denom)
            acc = _dot(p, v, acc)
            p, denom = _finish(next_scores, denom, maximum, BM, q.dtype)
            k = _get_k(kb, k16, slot, 2, SUB, FP8)
            _put(Vh, vp, slot, Table, v_tile, NPAGES, KH, PAGE)
            v_tile = page + 3
            slot = 1 - slot
        v = _get_v(vb, v16, slot, 2, SUB, FP8)
        denom = _dot(p, tl.full((64, 16), 1.0, q.dtype), denom)
        acc = _dot(p, v, acc)
        p, denom = _prob(
            q,
            k,
            denom,
            maximum,
            rows,
            end_page - 2,
            first_q,
            last_q,
            SQ,
            SK,
            WINDOW,
            BM,
            SPLIT_MASK,
        )
        k = _get_k(kb, k16, slot, 1, SUB, FP8)
        _put(Vh, vp, slot, Table, v_tile, NPAGES, KH, PAGE)
        v = _get_v(vb, v16, 1 - slot, 1, SUB, FP8)
        denom = _dot(p, tl.full((64, 16), 1.0, q.dtype), denom)
        acc = _dot(p, v, acc)
        p, denom = _prob(
            q,
            k,
            denom,
            maximum,
            rows,
            end_page - 1,
            first_q,
            last_q,
            SQ,
            SK,
            WINDOW,
            BM,
            SPLIT_MASK,
        )
        v = _get_v(vb, v16, slot, 0, SUB, FP8)
        if SK % 64:
            v = _mask_v(v, (end_page - 1) * 64, SK)
        denom = _dot(p, tl.full((64, 16), 1.0, q.dtype), denom)
        acc = _dot(p, v, acc)
    else:
        _put(Kh, kp, 0, Table, first_page, NPAGES, KH, PAGE)
        _put(Vh, vp, 0, Table, first_page, NPAGES, KH, PAGE)
        tlx.tdm_async_wait(2 * SUB)
        q = _load_q(qb, BM, GROUP, k_scale, HAS_SCALE)
        tlx.workgroup_barrier()
        for page in range(first_page, end_page):
            slot = (page - first_page) % 2
            _put(Kh, kp, 1 - slot, Table, page + 1, NPAGES, KH, PAGE)
            _put(Vh, vp, 1 - slot, Table, page + 1, NPAGES, KH, PAGE)
            tlx.tdm_async_wait(2 * SUB)
            if FP8:
                _dequant(kb, k16, slot)
                _dequant(vb, v16, slot)
            k = _view_k(k16, slot)
            v = _view_v(v16, slot)
            tlx.workgroup_barrier()
            if SPLIT_MASK:
                full = page * 64 + 63 <= first_q
                if WINDOW:
                    full &= page * 64 > last_q - 1 - WINDOW
                if full:
                    acc, maximum, denom = _consume(
                        q,
                        k,
                        v,
                        acc,
                        maximum,
                        denom,
                        rows,
                        page * 64,
                        SQ,
                        SK,
                        WINDOW,
                        BM,
                        FIXED,
                        False,
                    )
                else:
                    acc, maximum, denom = _consume(
                        q,
                        k,
                        v,
                        acc,
                        maximum,
                        denom,
                        rows,
                        page * 64,
                        SQ,
                        SK,
                        WINDOW,
                        BM,
                        FIXED,
                    )
            else:
                acc, maximum, denom = _consume(
                    q,
                    k,
                    v,
                    acc,
                    maximum,
                    denom,
                    rows,
                    page * 64,
                    SQ,
                    SK,
                    WINDOW,
                    BM,
                    FIXED,
                )
        tlx.tdm_async_wait(0)
    if FIXED:
        denom = tl.sum(denom, 1) * 0.0625
    result = acc / tl.maximum(denom[:, None], 1.0e-30)
    if HAS_SCALE:
        result *= tl.load(VScale)
    tl.store(Out + offsets, result.to(Out.dtype.element_ty), rows[:, None] < SQ)
    if FIXED:
        bad = (denom < 1.0e-20) | ~(denom < 1.0e20)
        bad |= tl.sum((~(tl.abs(acc) < 3.402823466e38)).to(tl.int32), 1) > 0
        return tl.sum(((rows < SQ) & bad).to(tl.int32), 0) > 0
    return False


@triton.jit(do_not_specialize=["NPAGES", "TABLE_STRIDE"])
def _prefill_kernel(
    Q,
    K,
    V,
    Tables,
    Out,
    CuQ,
    UsedK,
    KScale,
    VScale,
    NPAGES,
    TABLE_STRIDE,
    KH: tl.constexpr,
    WINDOW: tl.constexpr,
    BM: tl.constexpr,
    PAGE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    batch = tl.program_id(2)
    q_start = tl.load(CuQ + batch)
    sq = tl.load(CuQ + batch + 1) - q_start
    linear = tl.program_id(0) + tl.num_programs(0) * tl.program_id(1)
    block = tl.num_programs(0) - 1 - linear // KH
    if block * (BM // 8) < sq:
        sk = tl.load(UsedK + batch).to(tl.int32)
        retry = _pass(
            Q, K, V, Tables, Out, sq, sk, NPAGES, KScale, VScale, KH, TABLE_STRIDE, WINDOW,
            BM, 8, True, True, True, True, PAGE, HAS_SCALE, Q_START=q_start,
        )  # fmt: skip
        if retry:
            for chunk in tl.range(2, loop_unroll_factor=1):
                _pass(
                    Q, K, V, Tables, Out, sq, sk, NPAGES, KScale, VScale, KH, TABLE_STRIDE, WINDOW,
                    BM // 2, 8, True, True, False, False, PAGE, HAS_SCALE, 2, chunk, q_start,
                )  # fmt: skip


def _block_m(max_q, max_k, window, batch, kv_heads):
    """Query rows per CTA (GQA rows included), frozen from #5000's BF16 sweep."""
    if not (max_q > 768 or (max_q >= 512 and max_k == max_q)):
        return 64
    window = 0 if window and max_k <= window else window
    if max_q <= 4096:
        # Keep enough CTAs for small grids; wider tiles reuse K/V when
        # heads, batches or a cached prefix provide sufficient work.
        groups = batch * kv_heads
        bm = 128 if groups >= 16 or (max_k > max_q and groups >= 4) else 64
        if groups >= 32 and max_q > 768 and not window:
            bm = 256
    else:
        bm = 128 if window or max_q <= 8192 else 256
    if (
        batch == 1
        and kv_heads == 2
        and not window
        and max_q >= 16384
        and max_k - max_q >= 32768
    ):
        bm = 512
    return bm


def unsupported_reason(
    q,
    k,
    v,
    out,
    block_table,
    max_seqlen_k,
    softmax_scale,
    window,
    k_scale=None,
    v_scale=None,
):
    """None when paged_prefill serves these arguments, else why it does not."""
    if q.dtype != torch.bfloat16 or q.ndim != 3 or q.shape[2] != HEAD_DIM:
        return "needs BF16 queries [tokens, heads, 64]"
    if out.shape != q.shape:
        return "needs an output shaped like the queries"
    if k.ndim != 4 or k.shape[1] not in PAGE_SIZES or k.shape[3] != HEAD_DIM:
        return f"needs K/V caches [pages, page size in {PAGE_SIZES}, kv heads, 64]"
    if q.shape[1] != GQA * k.shape[2]:
        return f"needs {GQA} query heads per KV head"
    if k.dtype not in KV_DTYPES or v.dtype != k.dtype or v.shape != k.shape:
        return "needs matching BF16 or FP8 e4m3 K/V caches"
    if (k_scale is None) != (v_scale is None):
        return "needs both K/V scales or neither"
    if k_scale is not None and any(
        s.numel() != 1 or s.dtype != torch.float32 for s in (k_scale, v_scale)
    ):
        return "needs scalar FP32 K/V scales"
    if softmax_scale != HEAD_DIM**-0.5:
        return "needs softmax scale 1/sqrt(64)"
    if window not in (0, *WINDOWS):
        return f"needs a global window or one of {WINDOWS} keys"
    if not all(t.is_contiguous() for t in (q, k, v, out)):
        return "needs contiguous Q/K/V/output"
    if block_table.dtype != torch.int32 or block_table.stride(1) != 1:
        return "needs an int32 page table with contiguous rows"
    if block_table.shape[1] * k.shape[1] < triton.cdiv(max_seqlen_k, 64) * 64:
        return "needs a page table covering whole 64-key tiles"
    if any(t.data_ptr() % 16 for t in (q, k, v)):
        return "needs 16-byte-aligned Q/K/V"
    return None


def paged_prefill(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    seqused_k,
    block_table,
    max_seqlen_q,
    max_seqlen_k,
    window=0,
    k_scale=None,
    v_scale=None,
):
    """Causal paged prefill into out; arguments must pass unsupported_reason.

    q/out are [tokens, 8 * kv_heads, 64]; k/v [pages, page size, kv_heads, 64].
    block_table [batch, pages] holds int32 page IDs covering max_seqlen_k, and
    cu_seqlens_q/seqused_k give each sequence's query span and cached length,
    including its queries. window counts keys including self; 0 means global.
    """
    kv_heads = k.shape[2]
    batch = seqused_k.numel()
    if max_seqlen_q == 0 or batch == 0:
        return out
    bm = _block_m(max_seqlen_q, max_seqlen_k, window, batch, kv_heads)
    has_scale = k_scale is not None
    grid = (triton.cdiv(max_seqlen_q * GQA, bm), kv_heads, batch)
    _prefill_kernel[grid](
        q, k, v, block_table, out, cu_seqlens_q, seqused_k,
        k_scale if has_scale else cu_seqlens_q, v_scale if has_scale else cu_seqlens_q,
        k.shape[0], block_table.stride(0), kv_heads, window, bm, k.shape[1], has_scale,
        num_warps=4 if bm >= 256 else 2, waves_per_eu=1, reverse_local_assignment=True,
        llvm_fn_attrs="amdgpu-sched-strategy=coexec",
    )  # fmt: skip
    return out
