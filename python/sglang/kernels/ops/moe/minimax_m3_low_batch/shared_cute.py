# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM vllm/models/minimax_m3/nvidia/specialized_decode/moe/shared_cute.py @ddd43a54
"""Shared expert of the MiniMax-M3 low-batch FFN in CuTe DSL.

The stock shared expert (an MXFP8 MLP: input quantization, gate_up GEMM,
SwiGLU-OAI, quantization, down GEMM) runs five kernels; its gate_up GEMM puts
12 CTAs on 9.4 MB of weights. Here, on the aux stream:

1. :class:`SharedInputQuant`: the input rows to MXFP8 (per 32 values a
   UE8M0 scale rounded up from amax / 448, E4M3 values; FlashInfer's
   ``mxfp8_quantize``), one CTA per row, into 8-row tiles of 512 columns
   (with ``r`` row tiles of 8 rows, tile ``s r + t``: rows ``8t ..``, columns
   ``512 s ..``) with their scales in the per-tile 128x4 atom layout;
2. :class:`SharedFc1`: ``gate_up = x @ w_gate_up^T`` split over K into
   :data:`SPLITS` slices of 512 columns, so the 1536 weight rows (768 at TP8)
   give 144 work items per row tile (72): the persistent ``tcgen05`` GEMM the routed FC1
   and FC2 build on (DeepSeek-V4.1's ``RoutedExpertGemm``), with the split as
   the A operand's batch mode and the B operand's token tiles; each item
   stores its FP32 partial sums;
3. :class:`SharedAct`: the partial sums added in split order, rounded to
   BF16 as the stock GEMM stores ``gate_up``, SwiGLU-OAI to BF16 ``h`` and
   ``h`` to MXFP8;
4. :class:`SharedFc2`: ``h @ w_down^T`` (48 items per row tile), stored as
   BF16.

Each is built for the rank's intermediate columns: 768 (TP4) or 384 (TP8).

The weights are the MXFP8 linears' tensors as vLLM (and SGLang) prepare them for
FlashInfer (K-major E4M3 values, E8M0 scales in the 128x4 layout), read in
place. All four kernels use programmatic dependent launch: the GEMMs stream
their weights while the previous kernel finishes.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass import Float32, Int32, Uint8, Uint32

from sglang.kernels.ops.moe.minimax_m3_low_batch.front_helpers import (
    bf16x4_to_e4m3x4,
    block_scale,
    inverse_scale,
    max_f32,
)
from sglang.kernels.ops.moe.minimax_m3_low_batch.gemm_base_cute import (
    TILE,
    RoutedExpertGemm,
    tile_scale_bytes,
)
from sglang.kernels.ops.moe.minimax_m3_low_batch.primitives import load_global_u32x4

from .front_cute import HIDDEN, local_amax
from .gemm_cute import swiglu_oai_bf16x2

SPLITS = 12
SPLIT_K = HIDDEN // SPLITS
SF_VEC = 32
ATOM_BYTES = 512  # one 128-row x 4-scale atom


@cute.jit
def mxfp8_x8(words):
    """Eight BF16 values (four words) of a 32-value block spread over four
    consecutive threads: the E4M3 bytes as two words and the block's UE8M0
    code."""
    amax = local_amax(words)
    amax = max_f32(amax, cute.arch.shuffle_sync_bfly(amax, 1))
    amax = max_f32(amax, cute.arch.shuffle_sync_bfly(amax, 2))
    code = block_scale(amax)
    inverse = inverse_scale(code)
    return (
        bf16x4_to_e4m3x4(words[0], words[1], inverse),
        bf16x4_to_e4m3x4(words[2], words[3], inverse),
        code,
    )


@cute.jit
def store_mxfp8(
    quant: cute.Tensor,
    scales: cute.Tensor,
    words,
    row: Int32,
    first: Int32,
    split: cutlass.Constexpr[int],
    row_tiles: Int32,
):
    """Quantize the eight values at column ``first`` of ``row`` and store them
    and their block's scale (:func:`store_quantized`)."""
    lo, hi, code = mxfp8_x8(words)
    store_quantized(quant, scales, lo, hi, code, row, first, split, row_tiles)


@cute.jit
def store_quantized(
    quant: cute.Tensor,
    scales: cute.Tensor,
    lo: Uint32,
    hi: Uint32,
    code,
    row: Int32,
    first: Int32,
    split: cutlass.Constexpr[int],
    row_tiles: Int32,
):
    """Store the E4M3 values (two words) at column ``first`` of ``row`` and,
    at a block's first column, the block's scale. Columns are cut into slices
    of ``split`` and rows into 8-row tiles: operand tile ``s * row_tiles + t``
    (``quant`` as words, ``split`` columns a row) holds rows ``8t ..`` of
    column slice ``s``, its scales one atom block."""
    tile = first // split * row_tiles + row // TILE
    offset = (tile * TILE + row % TILE) * split + first % split
    quant[offset // 4] = lo
    quant[offset // 4 + 1] = hi
    if first % SF_VEC == 0:
        block = (first % split) // SF_VEC
        scales[
            tile * tile_scale_bytes(split)
            + (block // 4) * ATOM_BYTES
            + row % TILE * 16
            + block % 4
        ] = Uint8(code)


class SharedInputQuant:
    """The input rows to MXFP8 (module docstring), one CTA of 768 threads per
    row."""

    @cute.jit
    def __call__(
        self,
        x_ptr: cute.Pointer,
        quant_ptr: cute.Pointer,
        scale_ptr: cute.Pointer,
        rows: Int32,
        row_tiles: Int32,
        stream: cuda.CUstream,
    ):
        """``x``: BF16 [rows, 6144] as words; ``quant``: E4M3 [12 x
        ``row_tiles`` x 8, 512] as words (module docstring); ``scale``: one
        atom block per tile."""
        tiles = SPLITS * row_tiles
        self.kernel(
            cute.make_tensor(x_ptr, cute.make_layout(rows * HIDDEN // 2)),
            cute.make_tensor(quant_ptr, cute.make_layout(tiles * TILE * SPLIT_K // 4)),
            cute.make_tensor(
                scale_ptr, cute.make_layout(tiles * tile_scale_bytes(SPLIT_K))
            ),
            row_tiles,
        ).launch(
            grid=(rows, 1, 1), block=[HIDDEN // 8, 1, 1], stream=stream, use_pdl=True
        )

    @cute.kernel
    def kernel(
        self, x: cute.Tensor, quant: cute.Tensor, scales: cute.Tensor, row_tiles: Int32
    ):
        cute.arch.griddepcontrol_launch_dependents()
        thread, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        cute.arch.griddepcontrol_wait()
        words = load_global_u32x4(x.iterator + row * (HIDDEN // 2) + thread * 4)
        store_mxfp8(quant, scales, words, row, thread * 8, SPLIT_K, row_tiles)


class SharedFc1(RoutedExpertGemm):
    """gate_up, split over K (module docstring). The A operand's batch mode
    and the routing record's tile expert are the split; B's tile ``s r + t``
    (``r`` row tiles) is rows ``8t ..`` of split ``s``. Each item stores
    its FP32 partial sums [split, 8 r rows, 2 x intermediate]."""

    def __init__(
        self, intermediate: int, cta_k: int = 256, num_ab_stage: int = 6
    ) -> None:
        super().__init__(
            2 * intermediate, 2 * intermediate // 128, SPLIT_K, cta_k, num_ab_stage
        )

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        route_ptr: cute.Pointer,
        row_tiles: Int32,
        cta_words: Int32,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        """A: w_gate_up (2 x intermediate, 6144) E4M3, K-major, E8M0 scales in
        the 128x4 layout; B: the quantized input (12 x ``row_tiles`` x 8,
        512); out FP32."""
        tiles = SPLITS * row_tiles
        self.a_dtype = a_ptr.value_type
        self.b_dtype = b_ptr.value_type
        self.sf_dtype = sfa_ptr.value_type
        m = self.m
        a = cute.make_tensor(
            a_ptr,
            cute.make_layout((m, SPLIT_K, SPLITS), stride=(HIDDEN, 1, SPLIT_K)),
        )
        # The 128x4 layout of (m, HIDDEN) with K split: atoms of 4 scales
        # (128 columns) are 512 bytes apart, row tiles HIDDEN / 128 atoms.
        atoms = SPLIT_K // (4 * SF_VEC)
        sfa = cute.make_tensor(
            sfa_ptr,
            cute.make_layout(
                ((32, 4, m // 128), (SF_VEC, 4, atoms), SPLITS),
                stride=(
                    (16, 4, HIDDEN // (4 * SF_VEC) * ATOM_BYTES),
                    (0, 1, ATOM_BYTES),
                    atoms * ATOM_BYTES,
                ),
            ),
        )
        b = cute.make_tensor(
            b_ptr,
            cute.make_ordered_layout((tiles * TILE, SPLIT_K, 1), order=(1, 0, 2)),
        )
        sfb = cute.make_tensor(
            sfb_ptr,
            blockscaled_utils.tile_atom_to_shape_SF(
                cute.make_ordered_layout(
                    (tiles * 128, SPLIT_K, 1), order=(1, 0, 2)
                ).shape,
                SF_VEC,
            ),
        )
        out = cute.make_tensor(out_ptr, cute.make_layout(tiles * TILE * m))
        route = cute.make_tensor(route_ptr, cute.make_layout(4 + 2 * cta_words))
        # A static record: no ready words.
        self._launch(
            a, sfa, b, sfb, route, (out,), (), cta_words, Int32(0), ctas, stream
        )

    @cute.jit
    def store(
        self,
        tDrAcc: cute.Tensor,
        tDcC: cute.Tensor,
        epi: tuple,
        mRoute: cute.Tensor,
        cta_words: Int32,
        n_tile: Int32,
        m_tile: Int32,
        item: Int32,
        exchange: cute.Tensor,
    ):
        """The FP32 partial sums, at [tile * 8 + row, weight row]."""
        (mOut,) = epi
        for i in cutlass.range_constexpr(cute.size(tDrAcc)):
            coord = tDcC[i]
            mOut[coord[1] * self.m + coord[0]] = tDrAcc[i]


class SharedAct:
    """Split sums, BF16 gate_up, SwiGLU-OAI and MXFP8 of ``h`` (module
    docstring), one CTA per row, eight columns of ``h`` per thread: 96 threads
    at 768 columns; at 384, 48 of 64 (whole warps for the block shuffles)."""

    def __init__(self, intermediate: int) -> None:
        assert intermediate % SF_VEC == 0
        self.intermediate = intermediate
        self.threads = -(-intermediate // 8 // 32) * 32

    @cute.jit
    def __call__(
        self,
        partial_ptr: cute.Pointer,
        quant_ptr: cute.Pointer,
        scale_ptr: cute.Pointer,
        rows: Int32,
        row_tiles: Int32,
        stream: cuda.CUstream,
    ):
        """``partial``: FP32 [SPLITS, ``row_tiles`` x 8, 2 x intermediate];
        ``quant``: E4M3 [``row_tiles`` x 8, intermediate] as words; ``scale``:
        one atom block per row tile."""
        inter = self.intermediate
        self.kernel(
            cute.make_tensor(
                partial_ptr,
                cute.make_layout(SPLITS * row_tiles * TILE * 2 * inter),
            ),
            cute.make_tensor(
                quant_ptr, cute.make_layout(row_tiles * TILE * inter // 4)
            ),
            cute.make_tensor(
                scale_ptr, cute.make_layout(row_tiles * tile_scale_bytes(inter))
            ),
            row_tiles,
        ).launch(
            grid=(rows, 1, 1),
            block=[self.threads, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        partial: cute.Tensor,
        quant: cute.Tensor,
        scales: cute.Tensor,
        row_tiles: Int32,
    ):
        cute.arch.griddepcontrol_launch_dependents()
        thread, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        first = thread * 8
        inter = self.intermediate
        gate = cute.make_rmem_tensor((8,), Float32)
        up = cute.make_rmem_tensor((8,), Float32)
        gate.fill(0.0)
        up.fill(0.0)
        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.threads * 8 == inter):
            self.add_splits(partial, row, row_tiles, first, gate, up)
        else:
            # Threads past the row's columns keep zeros, never stored.
            if first < inter:
                self.add_splits(partial, row, row_tiles, first, gate, up)
        # gate_up as the stock GEMM stores it, then h in BF16.
        words = cute.make_rmem_tensor((4,), Uint32)
        for i in cutlass.range_constexpr(4):
            words[i] = swiglu_oai_bf16x2(
                _bf16_round(up[2 * i]),
                _bf16_round(gate[2 * i]),
                Float32(1.0),
                _bf16_round(up[2 * i + 1]),
                _bf16_round(gate[2 * i + 1]),
            )
        if cutlass.const_expr(self.threads * 8 == inter):
            store_mxfp8(quant, scales, words, row, first, inter, row_tiles)
        else:
            # Every thread takes part in its block's amax shuffles.
            lo, hi, code = mxfp8_x8(words)
            if first < inter:
                store_quantized(
                    quant, scales, lo, hi, code, row, first, inter, row_tiles
                )

    @cute.jit
    def add_splits(
        self,
        partial: cute.Tensor,
        row: Int32,
        row_tiles: Int32,
        first: Int32,
        gate: cute.Tensor,
        up: cute.Tensor,
    ):
        """The FP32 partial sums of gate and up columns ``first ..`` of
        ``row``, added in split order."""
        inter = self.intermediate
        for split in cutlass.range_constexpr(SPLITS):
            base = (split * row_tiles * TILE + row) * 2 * inter + first
            for i in cutlass.range_constexpr(8):
                gate[i] = gate[i] + partial[base + i]
                up[i] = up[i] + partial[base + inter + i]


@cute.jit
def _bf16_round(value: Float32) -> Float32:
    """``value`` rounded to BF16 (nearest even), as FP32."""
    return Float32(cutlass.BFloat16(value))


class SharedFc2(RoutedExpertGemm):
    """down: BF16 ``h @ w_down^T`` (module docstring), rows below the
    record's limit stored."""

    def __init__(
        self, intermediate: int, cta_k: int = 256, num_ab_stage: int = 6
    ) -> None:
        super().__init__(HIDDEN, HIDDEN // 128, intermediate, cta_k, num_ab_stage)
        # K tiles cover K (384 columns take 128 or 384).
        assert intermediate % cta_k == 0, (intermediate, cta_k)

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        route_ptr: cute.Pointer,
        rows: Int32,
        row_tiles: Int32,
        cta_words: Int32,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        """A: w_down (6144, intermediate) E4M3, K-major, E8M0 scales in the
        128x4 layout; B: the quantized ``h`` (``row_tiles`` x 8,
        intermediate); out: BF16
        [rows, 6144]."""
        self.a_dtype = a_ptr.value_type
        self.b_dtype = b_ptr.value_type
        self.sf_dtype = sfa_ptr.value_type
        a = cute.make_tensor(
            a_ptr, cute.make_ordered_layout((self.m, self.k, 1), order=(1, 0, 2))
        )
        sfa = cute.make_tensor(
            sfa_ptr, blockscaled_utils.tile_atom_to_shape_SF(a.shape, SF_VEC)
        )
        b = cute.make_tensor(
            b_ptr,
            cute.make_ordered_layout((row_tiles * TILE, self.k, 1), order=(1, 0, 2)),
        )
        sfb = cute.make_tensor(
            sfb_ptr,
            blockscaled_utils.tile_atom_to_shape_SF(
                cute.make_ordered_layout(
                    (row_tiles * 128, self.k, 1), order=(1, 0, 2)
                ).shape,
                SF_VEC,
            ),
        )
        out = cute.make_tensor(
            out_ptr, cute.make_ordered_layout((self.m, rows), order=(0, 1))
        )
        route = cute.make_tensor(route_ptr, cute.make_layout(4 + 2 * cta_words))
        # A static record: no ready words.
        self._launch(
            a, sfa, b, sfb, route, (out,), (), cta_words, Int32(0), ctas, stream
        )

    @cute.jit
    def store(
        self,
        tDrAcc: cute.Tensor,
        tDcC: cute.Tensor,
        epi: tuple,
        mRoute: cute.Tensor,
        cta_words: Int32,
        n_tile: Int32,
        m_tile: Int32,
        item: Int32,
        exchange: cute.Tensor,
    ):
        """Round to BF16 (nearest even) and store the rows below the limit."""
        (mOut,) = epi
        row_limit = mRoute[4 + cta_words + n_tile]
        tDrC = cute.make_rmem_tensor(tDrAcc.shape, cutlass.BFloat16)
        tDrC.store(tDrAcc.load().to(cutlass.BFloat16))
        for i in cutlass.range_constexpr(cute.size(tDrC)):
            coord = tDcC[i]
            if coord[1] < row_limit:
                mOut[coord[0], coord[1]] = tDrC[i]
