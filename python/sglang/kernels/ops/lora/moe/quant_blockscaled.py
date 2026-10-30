"""Row quantization into block-scaled operands for the CuTeDSL base GEMMs.

Every ``sf_vec_size`` consecutive K elements of a row share one scale: the
smallest power of two ``2 ** e`` with ``amax <= elem_max * 2 ** e`` (E8M0, so
no element clips), or for NVFP4 an E4M3 scale relative to a static global
scale. Elements round to nearest even into E4M3 (MXFP8) or E2M1 pairs with
the even index in the low nibble (MXFP4, NVFP4).
Scales land in the tcgen05 atom order the kernel reads: 512-byte blocks of
128 rows by 4 scale groups, ``((row % 32) * 16 + (row // 32 % 4) * 4 + group % 4``,
blocks ordered by (slab, row block, group block).

``quantize_tile`` and ``store_quantized_rows`` are shared with the dispatch
and activation kernels, which quantize rows as they produce them; the row
kernels here quantize an existing buffer through ``pair_to_row`` (routing
pair -> permuted row, -1 for pairs without a row).
"""

from __future__ import annotations

import msgspec
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

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


class BlockScaledRows(msgspec.Struct, frozen=True, kw_only=True):
    """A quantized row buffer plus its scale atoms, as the GEMM reads them."""

    data: torch.Tensor  # [rows, K] E4M3, or [rows, K // 2] uint8 E2M1 pairs
    scales: torch.Tensor  # uint8 atoms, scale_atom_bytes(rows_per_slab, K, vec, slabs)
    sf_vec_size: int
    rows_per_slab: int
    global_scale: float | None = None  # NVFP4: the static activation scale

    @property
    def k(self) -> int:
        return self.data.shape[-1] * (2 if self.data.dtype == torch.uint8 else 1)

    @property
    def nvfp4(self) -> bool:
        return self.global_scale is not None

    def kernel_args(self) -> dict:
        """Runtime and constexpr arguments of the quantizing kernels."""
        pack_e2m1 = self.data.dtype == torch.uint8
        assert pack_e2m1 or self.data.dtype == torch.float8_e4m3fn
        assert not (self.nvfp4 and not pack_e2m1)
        assert self.sf_vec_size == (16 if self.nvfp4 else 32)
        assert self.k % self.sf_vec_size == 0
        return dict(
            stride_out_row=self.data.stride(-2),
            rows_per_slab=self.rows_per_slab,
            row_atoms=-(-self.rows_per_slab // 128),
            k_atoms=-(-(self.k // self.sf_vec_size) // 4),
            global_scale=float(self.global_scale or 0.0),
            VEC=self.sf_vec_size,
            ELEM_MAX=6.0 if pack_e2m1 else 448.0,
            ELEM_MAX_LOG2=2 if pack_e2m1 else 8,
            PACK_E2M1=pack_e2m1,
            NVFP4=self.nvfp4,
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
    ELEM_MAX=448.0,
    ELEM_MAX_LOG2=8,
    PACK_E2M1=False,
    NVFP4=False,
)


def act_quant_args(rows: BlockScaledRows | None) -> dict:
    """Kernel arguments for an activation kernel's optional block-scaled output."""
    if rows is None:
        return dict(_NO_BLOCKSCALED)
    args = rows.kernel_args()
    args["stride_q_row"] = args.pop("stride_out_row")
    args["QUANT_BLOCKSCALED"] = True
    return args


@triton.jit
def _pow2(exponent):
    """Exact ``2 ** exponent`` for int32 exponents in the normal range."""
    return ((exponent + 127) << 23).to(tl.float32, bitcast=True)


@triton.jit
def pack_e2m1_pairs(scaled):
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
def sf_atom_offsets(rows, group_idx, rows_per_slab, row_atoms, k_atoms):
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
    VEC: tl.constexpr,
    ELEM_MAX: tl.constexpr,
    ELEM_MAX_LOG2: tl.constexpr,
    PACK_E2M1: tl.constexpr,
    NVFP4: tl.constexpr,
):
    """[rows, cols] fp32 -> (codes, scale bytes [rows, cols // VEC]).

    Codes are fp32 for E4M3 (converted at the store) or packed E2M1 pairs.
    """
    rows: tl.constexpr = x.shape[0]
    cols: tl.constexpr = x.shape[1]
    groups = tl.reshape(x, (rows, cols // VEC, VEC))
    amax = tl.max(tl.abs(groups), axis=2)
    if NVFP4:
        # E4M3 block scales relative to the global scale; normal range only.
        # The per-group divisions are correctly rounded and elements multiply
        # by the reciprocal, so host references reproduce the codes.
        block_scale = libdevice.div_rn(amax, 6.0 * global_scale)
        block_scale = tl.minimum(tl.maximum(block_scale, 0.015625), 448.0)
        block_scale = block_scale.to(tl.float8e4nv)
        inv_denom = libdevice.div_rn(1.0, block_scale.to(tl.float32) * global_scale)
        scaled = groups * inv_denom[:, :, None]
        sf_bytes = block_scale.to(tl.uint8, bitcast=True)
    else:
        # floor(log2(amax)) from the float bits, then one comparison for the ceiling.
        exponent = (amax.to(tl.int32, bitcast=True) >> 23) - 127 - ELEM_MAX_LOG2
        exponent = tl.maximum(exponent, -126)  # sub-normal scales flush to zero
        exponent += (amax > ELEM_MAX * _pow2(exponent)).to(tl.int32)
        exponent = tl.where(amax > 0, exponent, 0)
        scaled = groups * _pow2(-exponent)[:, :, None]
        sf_bytes = (exponent + 127).to(tl.uint8)
    codes = tl.reshape(scaled, (rows, cols))
    if PACK_E2M1:
        codes = pack_e2m1_pairs(codes)
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
    sf_offs = sf_atom_offsets(rows, group_idx, rows_per_slab, row_atoms, k_atoms)
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
    BLOCK_K: tl.constexpr,
    ROWS: tl.constexpr,
    VEC: tl.constexpr,
    ELEM_MAX: tl.constexpr,
    ELEM_MAX_LOG2: tl.constexpr,
    PACK_E2M1: tl.constexpr,
    NVFP4: tl.constexpr,
):
    pairs = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rows = tl.load(pair_to_row_ptr + pairs, mask=pairs < num_pairs, other=-1)
    live = rows >= 0
    if tl.max(live.to(tl.int32)) == 0:
        return
    rows = tl.where(live, rows, 0)
    col0 = tl.program_id(1) * BLOCK_K
    cols = col0 + tl.arange(0, BLOCK_K)
    x = tl.load(
        x_ptr + rows[:, None] * stride_x_row + cols[None, :],
        mask=live[:, None] & (cols < valid_cols)[None, :],
        other=0.0,
    )
    codes, sf_bytes = quantize_tile(
        x.to(tl.float32), global_scale, VEC, ELEM_MAX, ELEM_MAX_LOG2, PACK_E2M1, NVFP4
    )
    store_quantized_rows(
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
        VEC,
        PACK_E2M1,
    )


def quantize_rows(
    x: torch.Tensor, target: BlockScaledRows, pair_to_row: torch.Tensor
) -> None:
    """Quantize the ``pair_to_row`` rows of ``x`` [rows, K] into ``target``.

    ``x`` and ``target.data`` share their row numbering (a [slabs,
    rows_per_slab, K] view of expert slabs is flattened by the caller).
    """
    rows, k = x.shape
    assert k == target.k and rows % target.rows_per_slab == 0
    assert target.data.shape[0] == rows and pair_to_row.dtype == torch.int32
    block_k = 256 if k % 256 == 0 else 128
    assert block_k % target.sf_vec_size == 0
    grid = (
        triton.cdiv(pair_to_row.numel(), _ROWS_PER_PROGRAM),
        triton.cdiv(k, block_k),
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
