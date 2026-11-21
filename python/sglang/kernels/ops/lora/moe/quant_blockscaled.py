"""Quantize and pack rows for the CuTeDSL block-scaled GEMMs.

MX uses E8M0 scales per 32 elements. NVFP4 uses E4M3 scales per 16,
relative to a global or per-token scale. tcgen05 atoms pack 128 rows
by four groups, ordered by slab, row block, then group block.
"""

from __future__ import annotations

import msgspec
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.numerics import div_rn_f32

# Four E2M1 pairs per 32-bit register; the first source lands in the high nibble.
_E2M1X2_ASM = tl.constexpr(
    """{
.reg .b8 b<4>;
cvt.rn.satfinite.e2m1x2.f32 b0, $5, $1;
cvt.rn.satfinite.e2m1x2.f32 b1, $6, $2;
cvt.rn.satfinite.e2m1x2.f32 b2, $7, $3;
cvt.rn.satfinite.e2m1x2.f32 b3, $8, $4;
mov.b32 $0, {b0, b1, b2, b3};
}"""
)

# Pairs per program; decode batches route a few hundred, prefill tens of thousands.
_ROWS_PER_PROGRAM = 16


def quant_column_chunk(k: int) -> int:
    """Choose a power-of-two chunk up to 1024 that divides K without a masked tail."""
    for block_k in (1024, 512, 256, 128):
        if k % block_k == 0:
            return block_k
    raise ValueError(f"block-scaled rows need K to be a multiple of 128, got {k}")


class BlockScaledRows(msgspec.Struct, frozen=True, kw_only=True):
    """A quantized row buffer plus its scale atoms, as the GEMM reads them."""

    data: torch.Tensor  # [rows, K] E4M3, or [rows, K // 2] uint8 E2M1 pairs
    scales: torch.Tensor  # uint8 atoms, scale_atom_bytes(rows_per_slab, K, vec, slabs)
    sf_vec_size: int
    rows_per_slab: int
    global_scale: float | None = None  # NVFP4: the static activation scale
    # NVFP4 per-token: FP32 row_amax / row_denominator, flattened over slabs.
    row_scales: torch.Tensor | None = None
    row_denominator: float = 0.0
    # 4over6 compares the default block scale with a 1.5x candidate.
    use_4over6: bool = False
    err_mode: str = "MAE"
    err_use_fast_math: bool = False

    @property
    def k(self) -> int:
        return self.data.shape[-1] * (2 if self.data.dtype == torch.uint8 else 1)

    @property
    def nvfp4(self) -> bool:
        return self.global_scale is not None or self.row_scales is not None

    @property
    def per_row(self) -> bool:
        return self.row_scales is not None

    def kernel_args(self) -> dict:
        """Runtime and constexpr arguments of the quantizing kernels."""
        pack_e2m1 = self.data.dtype == torch.uint8
        assert pack_e2m1 or self.data.dtype == torch.float8_e4m3fn
        assert not (self.nvfp4 and not pack_e2m1)
        assert self.sf_vec_size == (16 if self.nvfp4 else 32)
        assert self.k % self.sf_vec_size == 0
        if self.per_row:
            assert self.global_scale is None and self.row_denominator > 0
            assert self.row_scales.dtype == torch.float32
            assert self.row_scales.numel() == self.data.shape[0]
        if self.use_4over6:
            assert self.per_row and self.err_mode in ("MAE", "MSE")
        return dict(
            stride_out_row=self.data.stride(-2),
            rows_per_slab=self.rows_per_slab,
            row_atoms=-(-self.rows_per_slab // 128),
            k_atoms=-(-(self.k // self.sf_vec_size) // 4),
            global_scale=float(self.global_scale or 0.0),
            # Unread unless PER_ROW; any tensor stands in for the pointer.
            row_scale_ptr=self.row_scales if self.per_row else self.scales,
            row_denominator=self.row_denominator,
            VEC=self.sf_vec_size,
            PACK_E2M1=pack_e2m1,
            NVFP4=self.nvfp4,
            PER_ROW=self.per_row,
            USE_4OVER6=self.use_4over6,
            ERR_MSE=self.err_mode == "MSE",
            ERR_FAST=self.err_use_fast_math,
        )


# Placeholder block-scaled arguments when an activation kernel does not quantize.
_NO_BLOCKSCALED = dict(
    stride_q_row=0,
    rows_per_slab=1,
    row_atoms=1,
    k_atoms=1,
    global_scale=0.0,
    QUANT_BLOCKSCALED=False,
    VEC=32,
    PACK_E2M1=False,
    NVFP4=False,
)
# The same for the per-token arguments, which only the unfused activation takes.
_NO_PER_ROW = dict(
    row_denominator=0.0, PER_ROW=False, USE_4OVER6=False, ERR_MSE=False, ERR_FAST=False
)


def act_quant_args(
    rows: BlockScaledRows | None, *, fallback: torch.Tensor | None = None
) -> dict:
    """Prepare block-scaled launch arguments. The fused path omits row-scale operands."""
    if rows is None:
        args = dict(_NO_BLOCKSCALED, **_NO_PER_ROW, row_scale_ptr=fallback)
    else:
        args = rows.kernel_args()
        args["stride_q_row"] = args.pop("stride_out_row")
        args["QUANT_BLOCKSCALED"] = True
    if fallback is None:
        for key in (*_NO_PER_ROW, "row_scale_ptr"):
            del args[key]
    return args


@triton.jit
def _pow2(exponent):
    """Exact ``2 ** exponent`` for int32 exponents in the normal range."""
    return ((exponent + 127) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _pack_e2m1_pairs(scaled):
    """[rows, cols] fp32 -> [rows, cols // 2] uint8 with the even column low."""
    rows: tl.constexpr = scaled.shape[0]
    cols: tl.constexpr = scaled.shape[1]
    even, odd = tl.split(tl.reshape(scaled, (rows, cols // 2, 2)))
    return tl.inline_asm_elementwise(
        _E2M1X2_ASM,
        "=r,r,r,r,r,r,r,r,r",
        [even, odd],
        dtype=tl.uint8,
        is_pure=True,
        pack=4,
    )


@triton.jit
def _e2m1_pair_values(packed):
    """[rows, pairs] packed E2M1 pairs -> [rows, 2 * pairs] fp32 values."""
    rows: tl.constexpr = packed.shape[0]
    pairs: tl.constexpr = packed.shape[1]
    nibbles = tl.join(packed & 0x0F, packed >> 4)
    magnitude = (nibbles & 7).to(tl.float32)
    value = tl.where(
        magnitude < 4,
        magnitude * 0.5,
        tl.where(magnitude < 6, magnitude - 2.0, (magnitude - 4.0) * 2.0),
    )
    value = tl.where((nibbles & 8) != 0, -value, value)
    return tl.reshape(value, (rows, 2 * pairs))


@triton.jit
def _per_row_candidate(scaled, block_scale):
    """Pack one E4M3 scale candidate per block and its E2M1 codes."""
    rows: tl.constexpr = scaled.shape[0]
    cols: tl.constexpr = scaled.shape[1] * scaled.shape[2]
    sf = tl.minimum(block_scale, 448.0).to(tl.float8e4nv)
    decoded = sf.to(tl.float32)
    live = decoded > 0
    denom = tl.where(live, decoded, 1.0)[:, :, None]
    codes = div_rn_f32(scaled, tl.broadcast_to(denom, scaled.shape))
    codes = tl.where(live[:, :, None], codes, 0.0)
    codes = _pack_e2m1_pairs(tl.reshape(codes, (rows, cols)))
    return codes, sf.to(tl.uint8, bitcast=True), decoded


@triton.jit
def _add_rn_f32(a, b):
    """fp32 ``a + b`` rounded to nearest, never fused into a multiply-add."""
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=r,r,r",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sub_rn_f32(a, b):
    """fp32 ``a - b`` rounded to nearest, never fused into a multiply-add."""
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        "=r,r,r",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sum_lanes_in_order(term):
    """Sum 16 errors in FlashInfer's order to preserve rounding."""
    rows: tl.constexpr = term.shape[0]
    groups: tl.constexpr = term.shape[1]
    even, odd = tl.split(tl.reshape(term, (rows, groups, 8, 2)))
    e0, e2 = tl.split(tl.reshape(even, (rows, groups, 4, 2)))  # lanes 4j, 4j + 2
    o1, o3 = tl.split(tl.reshape(odd, (rows, groups, 4, 2)))  # lanes 4j + 1, 4j + 3
    l0_8, l4_12 = tl.split(tl.reshape(e0, (rows, groups, 2, 2)))
    l2_10, l6_14 = tl.split(tl.reshape(e2, (rows, groups, 2, 2)))
    l1_9, l5_13 = tl.split(tl.reshape(o1, (rows, groups, 2, 2)))
    l3_11, l7_15 = tl.split(tl.reshape(o3, (rows, groups, 2, 2)))
    t0, t8 = tl.split(l0_8)
    t4, t12 = tl.split(l4_12)
    t2, t10 = tl.split(l2_10)
    t6, t14 = tl.split(l6_14)
    t1, t9 = tl.split(l1_9)
    t5, t13 = tl.split(l5_13)
    t3, t11 = tl.split(l3_11)
    t7, t15 = tl.split(l7_15)
    acc = _add_rn_f32(t0, t1)
    acc = _add_rn_f32(acc, t2)
    acc = _add_rn_f32(acc, t3)
    acc = _add_rn_f32(acc, t4)
    acc = _add_rn_f32(acc, t5)
    acc = _add_rn_f32(acc, t6)
    acc = _add_rn_f32(acc, t7)
    acc = _add_rn_f32(acc, t8)
    acc = _add_rn_f32(acc, t9)
    acc = _add_rn_f32(acc, t10)
    acc = _add_rn_f32(acc, t11)
    acc = _add_rn_f32(acc, t12)
    acc = _add_rn_f32(acc, t13)
    acc = _add_rn_f32(acc, t14)
    acc = _add_rn_f32(acc, t15)
    return acc


@triton.jit
def _candidate_error(
    x,
    codes,
    decoded,
    row_amax,
    encode_scale,
    row_denominator,
    ERR_MSE: tl.constexpr,
    ERR_FAST: tl.constexpr,
):
    """Fast mode scores encoded values. Strict mode preserves FlashInfer FP32 overflow.
    The caller keeps the 6-based scale on ties, including two infinite errors.
    """
    rows: tl.constexpr = x.shape[0]
    groups: tl.constexpr = x.shape[1]
    vec: tl.constexpr = x.shape[2]
    values = tl.reshape(_e2m1_pair_values(codes), (rows, groups, vec))
    candidate = values * decoded[:, :, None]
    if ERR_FAST:
        original = x * tl.broadcast_to(encode_scale[:, :, None], x.shape)
    else:
        candidate = candidate * tl.broadcast_to(row_amax[:, :, None], x.shape)
        candidate = div_rn_f32(candidate, tl.zeros_like(candidate) + row_denominator)
        original = x
    diff = _sub_rn_f32(candidate, original)
    if ERR_MSE:
        term = diff * diff
    else:
        term = tl.abs(diff)
    return _sum_lanes_in_order(term)


@triton.jit
def _quantize_per_row_tile(
    groups,
    amax,
    row_scale,
    row_denominator,
    USE_4OVER6: tl.constexpr,
    ERR_MSE: tl.constexpr,
    ERR_FAST: tl.constexpr,
):
    """Use non-FTZ division for subnormal row scales.
    Divide by the row scale first to avoid forming the product or its reciprocal.
    """
    row = tl.where(row_scale > 0, row_scale, 1.0)[:, None]
    scale6 = div_rn_f32(amax, tl.broadcast_to(6.0 * row, amax.shape))
    scaled = div_rn_f32(groups, tl.broadcast_to(row[:, :, None], groups.shape))
    codes, sf_bytes, decoded = _per_row_candidate(scaled, scale6)
    if USE_4OVER6:
        codes4, sf_bytes4, decoded4 = _per_row_candidate(scaled, scale6 * 1.5)
        row_amax = (row * row_denominator).to(tl.bfloat16).to(tl.float32)
        # flashinfer's encode scale D / row_amax, saturating at the fp32 maximum.
        encode = div_rn_f32(tl.zeros_like(row_amax) + row_denominator, row_amax)
        encode = tl.minimum(encode, 3.4028234663852886e38)
        err6 = _candidate_error(
            groups, codes, decoded, row_amax, encode, row_denominator, ERR_MSE, ERR_FAST
        )
        err4 = _candidate_error(
            groups,
            codes4,
            decoded4,
            row_amax,
            encode,
            row_denominator,
            ERR_MSE,
            ERR_FAST,
        )
        use4 = err4 < err6
        sf_bytes = tl.where(use4, sf_bytes4, sf_bytes)
        rows: tl.constexpr = groups.shape[0]
        pairs_per_group: tl.constexpr = groups.shape[2] // 2
        use4 = tl.broadcast_to(
            use4[:, :, None], (rows, groups.shape[1], pairs_per_group)
        )
        use4 = tl.reshape(use4, (rows, groups.shape[1] * pairs_per_group))
        codes = tl.where(use4, codes4, codes)
    return codes, sf_bytes


@triton.jit
def _sf_atom_offsets(rows, group_idx, rows_per_slab, row_atoms, k_atoms):
    """[rows] permuted rows and [groups] scale groups -> [rows, groups] bytes."""
    slab = rows // rows_per_slab
    slab_row = rows % rows_per_slab
    row_block = (slab * row_atoms + slab_row // 128)[:, None]
    row_byte = ((slab_row % 32) * 16 + (slab_row // 32 % 4) * 4)[:, None]
    return (
        (row_block * k_atoms + (group_idx // 4)[None, :]) * 512
        + row_byte
        + (group_idx % 4)[None, :]
    )


@triton.jit
def quantize_tile(
    x,
    global_scale,
    row_scale,  # [rows] fp32 stored per-row scales; unread unless PER_ROW
    VEC: tl.constexpr,
    PACK_E2M1: tl.constexpr,
    NVFP4: tl.constexpr,
    PER_ROW: tl.constexpr,
    USE_4OVER6: tl.constexpr = False,
    ERR_MSE: tl.constexpr = False,
    ERR_FAST: tl.constexpr = False,
    row_denominator=0.0,  # PER_ROW under USE_4OVER6: D of the stored scales
):
    """Return FP32 values for E4M3 or packed E2M1 pairs, plus uint8 group scales.
    E4M3 codes are cast at the store.
    """
    ELEM_MAX: tl.constexpr = 6.0 if PACK_E2M1 else 448.0
    ELEM_MAX_LOG2: tl.constexpr = 2 if PACK_E2M1 else 8
    rows: tl.constexpr = x.shape[0]
    cols: tl.constexpr = x.shape[1]
    groups = tl.reshape(x, (rows, cols // VEC, VEC))
    amax = tl.max(tl.abs(groups), axis=2)
    if NVFP4 and PER_ROW:
        codes, sf_bytes = _quantize_per_row_tile(
            groups,
            amax,
            row_scale,
            row_denominator,
            USE_4OVER6=USE_4OVER6,
            ERR_MSE=ERR_MSE,
            ERR_FAST=ERR_FAST,
        )
    else:
        if NVFP4:
            # Static NVFP4 keeps the reference's FTZ division and scale clamp.
            block_scale = libdevice.div_rn(amax, 6.0 * global_scale)
            block_scale = tl.minimum(tl.maximum(block_scale, 0.015625), 448.0)
            block_scale = block_scale.to(tl.float8e4nv)
            inv_denom = libdevice.div_rn(1.0, block_scale.to(tl.float32) * global_scale)
            scaled = groups * inv_denom[:, :, None]
            sf_bytes = block_scale.to(tl.uint8, bitcast=True)
        else:
            # Extract floor(log2(amax)) from the bits and round up with the comparison.
            exponent = (amax.to(tl.int32, bitcast=True) >> 23) - 127 - ELEM_MAX_LOG2
            exponent = tl.maximum(exponent, -126)  # keep the FP32 scale normal
            exponent += (amax > ELEM_MAX * _pow2(exponent)).to(tl.int32)
            exponent = tl.where(amax > 0, exponent, 0)
            scaled = groups * _pow2(-exponent)[:, :, None]
            sf_bytes = (exponent + 127).to(tl.uint8)
        codes = tl.reshape(scaled, (rows, cols))
        if PACK_E2M1:
            codes = _pack_e2m1_pairs(codes)
    return codes, sf_bytes


@triton.jit
def store_quantized_rows(
    codes,
    sf_bytes,
    rows,
    live,
    col0,
    valid_cols,
    out_ptr,
    stride_out_row,
    sf_ptr,
    rows_per_slab,
    row_atoms,
    k_atoms,
    VEC: tl.constexpr,
    PACK_E2M1: tl.constexpr,
):
    """Store a quantized tile of ``rows`` starting at column ``col0``."""
    groups: tl.constexpr = sf_bytes.shape[1]
    group_idx = col0 // VEC + tl.arange(0, groups)
    group_mask = live[:, None] & (group_idx < tl.cdiv(valid_cols, VEC))[None, :]
    sf_offs = _sf_atom_offsets(rows, group_idx, rows_per_slab, row_atoms, k_atoms)
    tl.store(sf_ptr + sf_offs, sf_bytes, mask=group_mask)
    cols: tl.constexpr = codes.shape[1]
    if PACK_E2M1:
        pair_cols = col0 // 2 + tl.arange(0, cols)
        tl.store(
            out_ptr + rows[:, None] * stride_out_row + pair_cols[None, :],
            codes,
            mask=live[:, None] & (pair_cols < (valid_cols + 1) // 2)[None, :],
        )
    else:
        out_cols = col0 + tl.arange(0, cols)
        tl.store(
            out_ptr + rows[:, None] * stride_out_row + out_cols[None, :],
            codes.to(out_ptr.dtype.element_ty),
            mask=live[:, None] & (out_cols < valid_cols)[None, :],
        )


@triton.jit
def _row_scale_kernel(
    x_ptr,
    row_scale_ptr,
    pair_to_row_ptr,
    num_pairs,
    stride_x_row,
    valid_cols,
    row_denominator,
    BLOCK_K: tl.constexpr,
    ROWS: tl.constexpr,
):
    """Per-token NVFP4: store each live row's scale, row_amax / row_denominator."""
    pairs = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rows = tl.load(pair_to_row_ptr + pairs, mask=pairs < num_pairs, other=-1)
    live = rows >= 0
    if tl.max(live.to(tl.int32)) == 0:
        return
    rows = tl.where(live, rows, 0).to(tl.int64)
    amax = tl.zeros((ROWS,), tl.float32)
    for col0 in range(0, valid_cols, BLOCK_K):
        cols = col0 + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr + rows[:, None] * stride_x_row + cols[None, :],
            mask=live[:, None] & (cols < valid_cols)[None, :],
            other=0.0,
        )
        amax = tl.maximum(amax, tl.max(tl.abs(x.to(tl.float32)), axis=1))
    denominator = tl.zeros((ROWS,), tl.float32) + row_denominator
    tl.store(row_scale_ptr + rows, div_rn_f32(amax, denominator), mask=live)


@triton.jit
def _quant_rows_kernel(
    x_ptr,
    out_ptr,
    sf_ptr,
    pair_to_row_ptr,
    num_pairs,
    stride_x_row,
    stride_out_row,
    valid_cols,
    rows_per_slab,
    row_atoms,
    k_atoms,
    global_scale,
    row_scale_ptr,
    row_denominator,
    BLOCK_K: tl.constexpr,
    ROWS: tl.constexpr,
    VEC: tl.constexpr,
    PACK_E2M1: tl.constexpr,
    NVFP4: tl.constexpr,
    PER_ROW: tl.constexpr,
    USE_4OVER6: tl.constexpr,
    ERR_MSE: tl.constexpr,
    ERR_FAST: tl.constexpr,
):
    pairs = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rows = tl.load(pair_to_row_ptr + pairs, mask=pairs < num_pairs, other=-1)
    live = rows >= 0
    if tl.max(live.to(tl.int32)) == 0:
        return
    # Slab rows times the row stride pass 2**31 on large masked batches.
    rows = tl.where(live, rows, 0).to(tl.int64)
    col0 = tl.program_id(1) * BLOCK_K
    cols = col0 + tl.arange(0, BLOCK_K)
    x = tl.load(
        x_ptr + rows[:, None] * stride_x_row + cols[None, :],
        mask=live[:, None] & (cols < valid_cols)[None, :],
        other=0.0,
    )
    if PER_ROW:
        row_scale = tl.load(row_scale_ptr + rows, mask=live, other=0.0)
    else:
        row_scale = tl.zeros((ROWS,), tl.float32)
    codes, sf_bytes = quantize_tile(
        x.to(tl.float32),
        global_scale=global_scale,
        row_scale=row_scale,
        VEC=VEC,
        PACK_E2M1=PACK_E2M1,
        NVFP4=NVFP4,
        PER_ROW=PER_ROW,
        USE_4OVER6=USE_4OVER6,
        ERR_MSE=ERR_MSE,
        ERR_FAST=ERR_FAST,
        row_denominator=row_denominator,
    )
    store_quantized_rows(
        codes,
        sf_bytes,
        rows=rows,
        live=live,
        col0=col0,
        valid_cols=valid_cols,
        out_ptr=out_ptr,
        stride_out_row=stride_out_row,
        sf_ptr=sf_ptr,
        rows_per_slab=rows_per_slab,
        row_atoms=row_atoms,
        k_atoms=k_atoms,
        VEC=VEC,
        PACK_E2M1=PACK_E2M1,
    )


def quantize_rows(
    x: torch.Tensor, target: BlockScaledRows, pair_to_row: torch.Tensor
) -> None:
    """Quantize the rows named by pair_to_row in the source and target's shared layout."""
    rows, k = x.shape
    assert k == target.k and rows % target.rows_per_slab == 0
    assert target.data.shape[0] == rows and pair_to_row.dtype == torch.int32
    block_k = 256 if k % 256 == 0 else 128
    assert block_k % target.sf_vec_size == 0
    grid = (
        triton.cdiv(pair_to_row.numel(), _ROWS_PER_PROGRAM),
        triton.cdiv(k, block_k),
    )
    if target.per_row:
        # The stored row scales first; the quantizer reads them per row.
        _row_scale_kernel[grid[:1]](
            x,
            target.row_scales,
            pair_to_row,
            pair_to_row.numel(),
            x.stride(0),
            k,
            target.row_denominator,
            BLOCK_K=block_k,
            ROWS=_ROWS_PER_PROGRAM,
            num_warps=4,
        )
    _quant_rows_kernel[grid](
        x,
        target.data,
        target.scales,
        pair_to_row,
        pair_to_row.numel(),
        x.stride(0),
        valid_cols=k,
        BLOCK_K=block_k,
        ROWS=_ROWS_PER_PROGRAM,
        num_warps=4,
        **target.kernel_args(),
    )


def pack_scale_atoms(scales: torch.Tensor) -> torch.Tensor:
    """[slabs, rows, groups] scale bytes -> flat tcgen05 atoms (host-side packing)."""
    slabs, rows, groups = scales.shape
    row_atoms, k_atoms = -(-rows // 128), -(-groups // 4)
    padded = torch.zeros(
        (slabs, row_atoms * 128, k_atoms * 4), dtype=scales.dtype, device=scales.device
    )
    padded[:, :rows, :groups] = scales
    return (
        padded.view(slabs, row_atoms, 4, 32, k_atoms, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .view(-1)
    )
