"""Paged decode Indexer logits in Gluon for SM80/SM90 (mma_v2).

Same scores, bit for bit, as _fp4_index_logits_paged_balanced_kernel: the same
work split, the same mma_v2 layout (warps [1, 4], kWidth 4) and so the same MMA
accumulation order, the same BF16 rounding points and the same head reduction.
What differs is how the keys reach the MMA. Each lane loads, as int32 words,
exactly the fp4 bytes its B-operand registers hold, decodes them in registers
and feeds mma_v2 directly; the queries stay in registers for a whole row. The
Triton kernel stages the decoded tile through shared memory for ldmatrix and
re-reads the loop-invariant queries from shared memory every tile.
"""

from __future__ import annotations

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2

from sglang.kernels.arch import current_arch

# Payload bytes per index-K slot; byte i holds elements 2i and 2i + 1.
HALF_D = gl.constexpr(64)
BLOCK_L = 128
NUM_WARPS = 4
PROGRAMS_PER_SM = 2

# One payload word (bytes 4j..4j+3 of a key row: four even and four odd elements)
# -> four bf16x2: $0, $1 = low nibbles of bytes (0, 1), (2, 3); $2, $3 = high
# nibbles. $5 holds the row's four e8m0 exponents and $6 the bit offset of this
# word's block. For each byte pair, prmt places byte b in bits 0-7 of a half and,
# by sign replication, its msb (the high nibble's e2m1 sign) as 0x00 / 0xFF in bits
# 8-15; << 2 and & 0x81C0 then leave the sign at bit 15 and eem at bits 6-8, which
# as a bf16 is e2m1 * 2^-126 (e == 0 lands on subnormals). The low nibble does the
# same on the word << 4. Two fma.rn.bf16x2, by 2^126 and by the block scale
# 2^(e8m0 - 127), are exact for the exponents the quantizer writes (1..254), so
# the keys equal the fp32 decode rounded to bf16, -0.0 included.
_DECODE_WORD_ASM = gl.constexpr(
    """{
.reg .b32 s, a, c, z, t, x;
bfe.u32 s, $5, $6, 8;
mul.lo.u32 a, s, 0x00800080;
mov.b32 c, 0x7E807E80;
mov.b32 z, 0x80008000;
shl.b32 t, $4, 4;
"""
    + "".join(
        f"""
prmt.b32 x, {src}, 0, {sel};
shl.b32 x, x, 2;
and.b32 x, x, 0x81C081C0;
fma.rn.bf16x2 x, x, c, z;
fma.rn.bf16x2 {out}, x, a, z;
"""
        for src, sel, out in (
            ("t", "0x9180", "$0"),
            ("t", "0xB3A2", "$1"),
            ("$4", "0x9180", "$2"),
            ("$4", "0xB3A2", "$3"),
        )
    )
    + "}"
)

# fp32 -> bf16 -> fp32 for two values with one cvt.rn.bf16x2.f32 (the same rounding
# as .to(bfloat16) per element).
_ROUND_BF16X2_ASM = gl.constexpr(
    "{ .reg .b32 r; cvt.rn.bf16x2.f32 r, $3, $2;"
    " shl.b32 $0, r, 16; and.b32 $1, r, 0xFFFF0000; }"
)


def _key_word_layout(block_l: int, num_warps: int) -> gl.DistributedLinearLayout:
    """The mma_v2 B-operand layout (kWidth 4, warps [1, num_warps]) of a [64, block_l]
    key tile, with each lane's 4 consecutive k as one int32: a [16, block_l] layout
    whose words decode in place into the operand registers."""
    warp_bases = [[0, 8 << i] for i in range(num_warps.bit_length() - 1)]
    reg_bases = [[4, 0], [8, 0]]
    n = 8 * num_warps
    while n < block_l:
        reg_bases.append([0, n])
        n *= 2
    return gl.DistributedLinearLayout(
        reg_bases=reg_bases,
        lane_bases=[[1, 0], [2, 0], [0, 1], [0, 2], [0, 4]],
        warp_bases=warp_bases,
        block_bases=[],
        shape=[16, block_l],
    )


@gluon.jit
def _add(a, b):
    return a + b


@gluon.jit
def _round_bf16(x):
    return gl.inline_asm_elementwise(
        _ROUND_BF16X2_ASM, "=r,=r,r,r", [x], dtype=gl.float32, is_pure=True, pack=2
    )


@gluon.jit
def _words_to_operand(w01, w23, KL: gl.constexpr, BLOCK_L: gl.constexpr):
    """bf16x2 words [16, BLOCK_L] for k = 4j + (0, 1) and 4j + (2, 3) -> the
    [64, BLOCK_L] operand in KL; only renames registers."""
    w = gl.join(w01, w23)
    x = gl.join(
        (w & 0xFFFF).to(gl.int16).to(gl.bfloat16, bitcast=True),
        (w >> 16).to(gl.int16).to(gl.bfloat16, bitcast=True),
    )
    x = gl.reshape(gl.permute(x, (0, 2, 3, 1)), [HALF_D, BLOCK_L])
    return gl.convert_layout(x, KL, assert_trivial=True)


@gluon.jit
def _decode_key_words(words, exps, shift, KL: gl.constexpr, BLOCK_L: gl.constexpr):
    lo01, lo23, hi01, hi23 = gl.inline_asm_elementwise(
        _DECODE_WORD_ASM,
        "=r,=r,=r,=r,r,r,r",
        [words, gl.broadcast(gl.expand_dims(exps, 0), words)[0], shift],
        dtype=(gl.int32, gl.int32, gl.int32, gl.int32),
        is_pure=True,
        pack=1,
    )
    k_low = _words_to_operand(lo01, lo23, KL, BLOCK_L)
    k_high = _words_to_operand(hi01, hi23, KL, BLOCK_L)
    return k_low, k_high


@gluon.jit
def _tile_slots(
    req_row,
    mask_ptr,
    mask_off,
    base,
    n_vis,
    stop,
    RATIO: gl.constexpr,
    HAS_MASK: gl.constexpr,
    BLOCK_L: gl.constexpr,
    LAYOUT: gl.constexpr,
):
    offs_l = base + gl.arange(0, BLOCK_L, layout=LAYOUT)
    valid = (offs_l < n_vis) & (offs_l < stop)
    if HAS_MASK:
        valid = valid & gl.load(mask_ptr + mask_off + offs_l, mask=valid, other=False)
    slot = gl.load(req_row + offs_l * RATIO, mask=valid, other=0) // RATIO
    return slot, valid


@gluon.jit
def _tile_key_words(table_ptr, slot, valid, words, PAGE: gl.constexpr, row_stride):
    page = slot // PAGE
    off = slot % PAGE
    row_base = page.to(gl.int64) * row_stride
    exps = gl.load(
        (table_ptr + row_base + PAGE * HALF_D + off * 4).to(gl.pointer_type(gl.int32)),
        mask=valid,
        other=0x7F7F7F7F,
    )
    pay = (table_ptr + gl.multiple_of(row_base + off * HALF_D, 16)).to(
        gl.pointer_type(gl.int32)
    )
    pay = gl.load(
        gl.expand_dims(pay, 0) + gl.expand_dims(words, 1),
        mask=gl.expand_dims(valid, 0),
        other=0,
    )
    return pay, exps


@gluon.jit
def _fp4_index_logits_paged_balanced_gluon_kernel(
    q_ptr,
    w_ptr,
    req_ptr,
    req_table_ptr,
    lens_ptr,
    table_ptr,
    mask_ptr,
    out_ptr,
    B,
    L,
    row_stride,
    req_stride,
    mask_stride,
    out_stride,
    stride_qb,
    stride_qh,
    stride_wb,
    PAGE: gl.constexpr,
    H: gl.constexpr,
    RATIO: gl.constexpr,
    HAS_MASK: gl.constexpr,
    BLOCK_L: gl.constexpr,
    BLOCK_B: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    WL: gl.constexpr,
):
    MMA: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[1, NUM_WARPS], instr_shape=[16, 8]
    )
    QL: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=MMA, k_width=4)
    KL: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MMA, k_width=4)
    QB: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [NUM_WARPS, 1], [1, 0])
    ROWS: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])
    K_COL: gl.constexpr = gl.SliceLayout(0, WL)
    OUT: gl.constexpr = gl.SliceLayout(0, MMA)

    # The work split of _fp4_index_logits_paged_balanced_kernel.
    rows = gl.arange(0, BLOCK_B, layout=ROWS)
    tiles = gl.cdiv(
        gl.minimum(gl.load(lens_ptr + rows, mask=rows < B, other=0), L), BLOCK_L
    ).to(gl.int64)
    total = gl.sum(tiles, 0)
    pid = gl.program_id(0).to(gl.int64)
    start = pid * total // gl.num_programs(0)
    end = (pid + 1) * total // gl.num_programs(0)
    b = gl.sum((gl.associative_scan(tiles, 0, _add) <= start).to(gl.int32), 0)
    row_start = gl.sum(gl.where(rows < b, tiles, 0), 0)

    words = gl.arange(0, HALF_D // 4, layout=gl.SliceLayout(1, WL))
    # Byte i uses e8m0 block i // 16, so word j uses block j // 4.
    shift = gl.expand_dims(8 * (words // 4), 1)
    q_h = gl.arange(0, H, layout=gl.SliceLayout(1, QB))
    q_d = gl.arange(0, 2 * HALF_D, layout=gl.SliceLayout(0, QB))
    w_h = gl.arange(0, H, layout=gl.SliceLayout(1, MMA))
    t = start
    while t < end:
        n_vis = gl.minimum(gl.load(lens_ptr + b), L)
        row_end = row_start + gl.cdiv(n_vis, BLOCK_L)
        stop = gl.minimum(row_end, end)
        req = gl.load(req_ptr + b).to(gl.int64)
        q = gl.load(
            q_ptr
            + b * stride_qb
            + gl.expand_dims(q_h, 1) * stride_qh
            + gl.expand_dims(q_d, 0)
        )
        q_even, q_odd = gl.split(gl.reshape(q, [H, HALF_D, 2]))
        q_even = gl.convert_layout(q_even, QL)
        q_odd = gl.convert_layout(q_odd, QL)
        w = gl.expand_dims(gl.load(w_ptr + b * stride_wb + w_h).to(gl.float32), 1)
        req_row = req_table_ptr + req * req_stride
        mask_off = b * mask_stride
        base = (t - row_start) * BLOCK_L
        share_end = (stop - row_start) * BLOCK_L
        # Registers carry the next tile's keys and the slots of the one after.
        slot, valid = _tile_slots(
            req_row,
            mask_ptr,
            mask_off,
            base,
            n_vis,
            share_end,
            RATIO,
            HAS_MASK,
            BLOCK_L,
            K_COL,
        )
        pay, exps = _tile_key_words(table_ptr, slot, valid, words, PAGE, row_stride)
        slot, valid = _tile_slots(
            req_row,
            mask_ptr,
            mask_off,
            base + BLOCK_L,
            n_vis,
            share_end,
            RATIO,
            HAS_MASK,
            BLOCK_L,
            K_COL,
        )
        for _ in range(t, stop):
            next_pay, next_exps = _tile_key_words(
                table_ptr, slot, valid, words, PAGE, row_stride
            )
            slot, valid = _tile_slots(
                req_row,
                mask_ptr,
                mask_off,
                base + 2 * BLOCK_L,
                n_vis,
                share_end,
                RATIO,
                HAS_MASK,
                BLOCK_L,
                K_COL,
            )
            k_low, k_high = _decode_key_words(pay, exps, shift, KL, BLOCK_L)
            acc = gl.zeros([H, BLOCK_L], gl.float32, layout=MMA)
            acc = mma_v2(q_even, k_low, acc)
            acc = mma_v2(q_odd, k_high, acc)
            # Same rounding points as _fp4_index_logits_from_k.
            s = gl.maximum(_round_bf16(acc), 0.0)
            s = _round_bf16(s * w)
            logit = gl.sum(s, axis=0).to(gl.bfloat16).to(gl.float32)
            offs_l = base + gl.arange(0, BLOCK_L, layout=OUT)
            visible = offs_l < n_vis
            if HAS_MASK:
                visible = visible & gl.load(
                    mask_ptr + mask_off + offs_l, mask=visible, other=False
                )
            logit = gl.where(visible, logit, float("-inf"))
            gl.store(out_ptr + b * out_stride + offs_l, logit, mask=offs_l < L)
            pay, exps = next_pay, next_exps
            base += BLOCK_L
        t = stop
        b += 1
        row_start = row_end


def fp4_index_logits_paged_gluon_supported(heads: int) -> bool:
    """The mma_v2 kernel is built and checked for SM80 / SM90 and 32 or 64 heads."""
    arch = current_arch()
    return (arch.is_sm(8) or arch.is_sm(9)) and heads in (32, 64)


def launch_fp4_index_logits_paged_balanced_gluon(
    q,
    weights,
    req,
    req_table,
    lens,
    table,
    candidate_mask,
    out,
    page_size,
    max_len,
    ratio,
):
    batch, heads, _ = q.shape
    _fp4_index_logits_paged_balanced_gluon_kernel[
        (current_arch().num_sms * PROGRAMS_PER_SM,)
    ](
        q,
        weights,
        req,
        req_table,
        lens,
        table,
        candidate_mask,
        out,
        batch,
        max_len,
        table.stride(0),
        req_table.stride(0),
        candidate_mask.stride(0) if candidate_mask is not None else 0,
        out.stride(0),
        q.stride(0),
        q.stride(1),
        weights.stride(0),
        PAGE=page_size,
        H=heads,
        RATIO=ratio,
        HAS_MASK=candidate_mask is not None,
        BLOCK_L=BLOCK_L,
        BLOCK_B=triton.next_power_of_2(batch),
        NUM_WARPS=NUM_WARPS,
        WL=_key_word_layout(BLOCK_L, NUM_WARPS),
        num_warps=NUM_WARPS,
    )
