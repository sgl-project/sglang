# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM vllm/models/minimax_m3/nvidia/specialized_decode/moe/__init__.py @ddd43a54
# and vllm/models/deepseek_v41/nvidia/specialized_decode/moe/routed_expert_gemm.py @ddd43a54
"""Low-batch FFN of a MiniMax-M3.1-Flash MoE layer (1-32 decode rows).

The two entry points replace the stock TRT-LLM-gen routed chain (routing,
per-token NVFP4 quantization, FC1, FC2 with ``do_finalize=False``) and the
MXFP8 shared MLP of one TP rank::

    x, logits ─► front ─► FC1 ──────────► FC2 ─► (routed, weights, permuted)
                 (top-4, NVFP4 x, record)  (h to NVFP4 per tile)
    x ─► MXFP8 x ─► FC1 (split K) ─► SwiGLU, MXFP8 h ─► FC2 ─► shared

* ``front_cute.py``: the routing (top-4 of ``sigmoid(logit) + bias``, the
  weights from the recovered scores), the routed input's per-token NVFP4 and
  the routing record of ``route_cute.py`` in one launch, built for calls of up
  to 8, 16 or 32 rows;
* ``gemm_cute.py``: FC1 (with SwiGLU-OAI; the last item of each token tile
  quantizes the tile's BF16 output to per-token NVFP4) and FC2, persistent
  ``tcgen05`` NVFP4 GEMMs over the record's (weight tile, token tile) items;
  above 8 rows FC2 starts on each token tile as FC1 completes it;
* ``shared_cute.py``: the MXFP8 shared expert on the same GEMM, up to four
  8-row tiles.

The routed output is the stock deferred-finalize triple: ``routed`` BF16 rows
in expert-major, tile-padded order (padding rows are never written), the
routing weights (FP32 by default, the dtype of the stock top-k), and int32
``permuted[t, k]``, the row of ``routed`` holding token ``t``'s ``k``-th
expert. The top-k weighted sum, the shared add and the all-reduce stay with
the consumer (SGLang's ``moe_finalize_all_reduce`` boundary).

The kernels are built for 768 (TP4) or 384 (TP8) intermediate columns per
rank. They read the weights SGLang prepares for the TRT-LLM-gen NVFP4 experts
and the FlashInfer MXFP8 linears in place, compile once per device and width
(``precompile_low_batch_ffn``), and launch with programmatic dependent launch
on the current stream, so CUDA graphs capture them like any other kernel.
Every PDL consumer of ``routed`` must wait for its grid dependency before
reading it: FC2 triggers its dependents early.
"""

from __future__ import annotations

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from cutlass.cute.runtime import make_fake_stream, make_ptr

from sglang.kernels.ops.moe.minimax_m3_low_batch.front_cute import (
    LowBatchFront,
    h_tile_bytes,
)
from sglang.kernels.ops.moe.minimax_m3_low_batch.gemm_base_cute import TILE
from sglang.kernels.ops.moe.minimax_m3_low_batch.gemm_cute import RoutedFc1, RoutedFc2
from sglang.kernels.ops.moe.minimax_m3_low_batch.primitives import current_cu_stream
from sglang.kernels.ops.moe.minimax_m3_low_batch.shapes import (
    EXPERTS,
    HIDDEN,
    MAX_ROWS,
    TOP_K,
)
from sglang.kernels.ops.moe.minimax_m3_low_batch.shared_cute import (
    SPLIT_K,
    SPLITS,
    SharedAct,
    SharedFc1,
    SharedFc2,
    SharedInputQuant,
)
from sglang.kernels.ops.moe.minimax_m3_low_batch.shared_cute import (
    tile_scale_bytes as shared_tile_scale_bytes,
)

# The row counts the front is built for (the routing record's levels).
WIDTHS = (8, 16, 32)
# K tile and pipeline stages of the routed GEMMs (shared memory per CTA:
# 21 KB per stage), their persistent CTAs (None: one per SM), and the shared
# FC1's stages (36 KB per stage). Tuned on B300 and B200 (four real
# layers, 1, 4 and 8 rows) against FC1 4/6 stages, FC2 3/6/8, shared FC1 3-5
# and 96-128 routed CTAs: FC2 at 4 stages gains 6% at one row; no other
# setting moves a row count by 3% or more, and fewer routed CTAs slow eight
# rows. At 1-32 rows (B300, 12 real layers), FC1 at K tile 512 (4-5 stages)
# is within 1%, FC2 at K tile 384 or 768 is 5-6% slower, and the stage pairs
# that let an FC1 and an FC2 CTA share an SM (6/4, 5/5, 6/3, 8/2) are 1-16%
# slower. Not re-tuned on GB300.
ROUTED_FC1_CTA_K = 256
ROUTED_FC2_CTA_K = 256
ROUTED_FC1_STAGES = 8
ROUTED_FC2_STAGES = 4
ROUTED_CTAS: int | None = None
# FC2's (routed and shared) K tile and stages by intermediate columns per rank:
# 384 columns (TP8) take three K tiles of 128, twice the stages at half the
# stage size.
FC2_CTA_K = {768: ROUTED_FC2_CTA_K, 384: 128}
ROUTED_FC2_STAGES_BY_WIDTH = {768: ROUTED_FC2_STAGES, 384: 2 * ROUTED_FC2_STAGES}
# Above this many rows FC2 starts on each token tile as FC1 completes it;
# up to it, after the FC1 grid. B300, 12 real layers: per tile 1.6-2.2 us
# faster at 12-32 rows and 1.1-1.3 us slower at 1-3 rows (even at 8).
OVERLAP_ROWS = 8
SHARED_FC1_STAGES = 6
SHARED_FC2_STAGES = {768: 6, 384: 8}
# Persistent CTAs of the shared expert's GEMMs (144 and 48 work items per row
# tile), which leave SMs to the routed GEMMs beside them (the critical path).
# On B300 at 1, 4 and 8 rows: within 3% of the best of 144/48, 72/48,
# 48/24 and 24/16; at 1-32 rows within 2% of 48/32 and 72/48 (24/16: 2-5%
# slower). Not re-tuned on GB300.
SHARED_FC1_CTAS = 36
SHARED_FC2_CTAS = 24
# The shared expert's 8-row tiles, and its records' table widths.
SHARED_ROW_TILES = MAX_ROWS // TILE


def width(rows: int) -> int:
    """The front's build for calls of ``rows`` rows: the narrowest that fits."""
    return next(w for w in WIDTHS if rows <= w)


def max_ctas(rows: int, top_k: int, experts: int) -> int:
    """The most 8-row token tiles any routing of ``rows * top_k`` slots needs."""
    slots = rows * top_k
    filled = min(experts, slots)
    return filled + (slots - filled) // TILE


def cta_words(ctas: int) -> int:
    """``ctas`` rounded up to a multiple of four (the record's table width)."""
    return (ctas + 3) // 4 * 4


def route_words(ctas: int) -> int:
    """int32 length of the routing record for ``ctas`` token tiles (its
    layout is in ``route_cute.py``)."""
    return 4 + 3 * cta_words(ctas) + TILE * ctas


SHARED_FC1_WORDS = cta_words(SPLITS * SHARED_ROW_TILES)
SHARED_FC2_WORDS = cta_words(SHARED_ROW_TILES)


def shared_records(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The shared expert's static routing records, row ``rows - 1`` for calls
    of ``rows`` rows: FC1's tile ``s r + t`` is rows ``8t ..`` of K split
    ``s`` (``r`` row tiles); FC2's tile ``t`` rows ``8t ..``. Allocate them
    outside any CUDA graph capture."""
    fc1, fc2 = [], []
    for rows in range(1, MAX_ROWS + 1):
        row_tiles = -(-rows // TILE)
        limits = [min(TILE * (t + 1), rows) for t in range(row_tiles)]
        fc1.append(
            _record(
                [s for s in range(SPLITS) for _ in range(row_tiles)],
                [
                    TILE * s * row_tiles + limit
                    for s in range(SPLITS)
                    for limit in limits
                ],
                SHARED_FC1_WORDS,
            )
        )
        fc2.append(_record([0] * row_tiles, limits, SHARED_FC2_WORDS))
    return (
        torch.tensor(fc1, dtype=torch.int32, device=device),
        torch.tensor(fc2, dtype=torch.int32, device=device),
    )


def precompile_low_batch_ffn(
    device_index: int, columns: int, *, fp32_weights: bool = True
) -> None:
    """Compile every kernel the two entry points launch at ``columns``
    intermediate columns. Must run outside any CUDA graph capture: a first
    launch compiles (tens of seconds in total), and there is no on-disk cache."""
    for built in WIDTHS:
        _front(device_index, built, fp32_weights)
    for overlap in (False, True):
        _fc1(device_index, columns, overlap)
        _fc2(device_index, columns, overlap)
    _shared_input(device_index)
    _shared_fc1(device_index, columns)
    _shared_act(device_index, columns)
    _shared_fc2(device_index, columns)


def low_batch_routed_ffn(
    x: torch.Tensor,
    logits: torch.Tensor,
    *,
    bias: torch.Tensor,
    routed_scaling_factor: float,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    alpha1: torch.Tensor,
    alpha2: torch.Tensor,
    columns: int,
    weights_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The routed experts of one TP rank on 1-32 rows, stopped before the
    top-k sum: ``(routed, weights, permuted)``, launched on the current stream.

    Coverage: SM100 family; ``x`` BF16 [rows, 6144] contiguous, 16-B aligned,
    1 <= rows <= 32; ``logits`` FP32 [rows, 128] contiguous; ``bias`` FP32
    [128]; 128 local experts, top-4, sigmoid scores renormalized and scaled by
    ``routed_scaling_factor``; ``w13`` / ``w2`` and their E4M3 scales in the
    FlashInfer TRT-LLM-gen NVFP4 layout (epilogue tile 128, w13 gated-act
    interleaved, scales 128x4), 32-B aligned; ``alpha1`` / ``alpha2`` FP32
    [128] per-expert dequant multipliers; ``columns`` in ``shapes.COLUMNS``;
    per-token NVFP4 activations (``1 / 2688`` global encode, the stock
    non-4over6 quantization); SwiGLU-OAI with ``shapes.SWIGLU_*``. The caller checks
    all of it; nothing here does.
    """
    rows = x.size(0)
    device = x.device
    device_index = device.index
    fp32_weights = weights_dtype == torch.float32
    tiles = max_ctas(rows, TOP_K, EXPERTS)
    h_rows = TILE * tiles
    words = Int32(cta_words(tiles))
    weights = torch.empty((rows, TOP_K), dtype=weights_dtype, device=device)
    permuted = torch.empty((rows, TOP_K), dtype=torch.int32, device=device)
    # The record, then one FC1 item count per tile (zeroed by the front).
    counts = route_words(tiles)
    record = torch.empty(counts + cta_words(tiles), dtype=torch.int32, device=device)
    x_quant = torch.empty((rows, HIDDEN // 2), dtype=torch.uint8, device=device)
    x_scale = torch.empty((rows, HIDDEN // 16), dtype=torch.uint8, device=device)
    x_pt = torch.empty(rows, dtype=torch.float32, device=device)
    h = torch.empty((h_rows, columns), dtype=torch.bfloat16, device=device)
    h_quant = torch.empty((h_rows, columns // 2), dtype=torch.uint8, device=device)
    h_scale = torch.empty(
        tiles * h_tile_bytes(columns), dtype=torch.uint8, device=device
    )
    h_pt = torch.empty(h_rows, dtype=torch.float32, device=device)
    output = torch.empty((h_rows, HIDDEN), dtype=torch.bfloat16, device=device)
    stream = current_cu_stream()
    ctas = Int32(ROUTED_CTAS or _sm_count(device_index))
    _front(device_index, width(rows), fp32_weights)(
        _ptr(cutlass.Float32, logits),
        _ptr(cutlass.Float32, bias),
        Float32(routed_scaling_factor),
        _ptr(cutlass.Uint32, x),
        _ptr(cutlass.Float32 if fp32_weights else cutlass.Uint16, weights),
        _ptr(cutlass.Uint32, x_quant),
        _ptr(cutlass.Uint8, x_scale),
        _ptr(cutlass.Float32, x_pt),
        _ptr(cutlass.Int32, permuted),
        _ptr(cutlass.Int32, record),
        Int32(rows),
        stream,
    )
    overlap = rows > OVERLAP_ROWS
    _fc1(device_index, columns, overlap)(
        _ptr(cutlass.Float4E2M1FN, w13, 32),
        _ptr(cutlass.Float8E4M3FN, w13_scale),
        _ptr(cutlass.Float4E2M1FN, x_quant, 32),
        _ptr(cutlass.Uint8, x_quant),
        _ptr(cutlass.Uint32, x_scale),
        _ptr(cutlass.Float32, x_pt),
        _ptr(cutlass.Float32, alpha1),
        _ptr(cutlass.Uint32, h),
        _ptr(cutlass.Uint32, h_quant),
        _ptr(cutlass.Uint8, h_scale),
        _ptr(cutlass.Float32, h_pt),
        _ptr(cutlass.Int32, record),
        _ptr(cutlass.Int32, record[counts:]),
        Int32(rows),
        Int32(EXPERTS),
        words,
        Int32(h_rows),
        ctas,
        stream,
    )
    _fc2(device_index, columns, overlap)(
        _ptr(cutlass.Float4E2M1FN, w2, 32),
        _ptr(cutlass.Float8E4M3FN, w2_scale),
        _ptr(cutlass.Float4E2M1FN, h_quant, 32),
        _ptr(cutlass.Float8E4M3FN, h_scale),
        _ptr(cutlass.Float32, h_pt),
        _ptr(cutlass.Float32, alpha2),
        _ptr(cutlass.BFloat16, output),
        _ptr(cutlass.Int32, record),
        Int32(h_rows),
        Int32(EXPERTS),
        words,
        ctas,
        stream,
    )
    return output, weights, permuted


def low_batch_shared_ffn(
    x: torch.Tensor,
    *,
    gate_up: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down: torch.Tensor,
    down_scale: torch.Tensor,
    fc1_records: torch.Tensor,
    fc2_records: torch.Tensor,
    columns: int,
) -> torch.Tensor:
    """The MXFP8 shared expert of one TP rank on 1-32 rows: its BF16
    [rows, 6144] partial output (not reduced over TP), launched on the current
    stream.

    Coverage: SM100 family; ``x`` as for ``low_batch_routed_ffn``;
    ``gate_up`` E4M3 [2 * columns, 6144] row-major with rows ``[gate; up]``,
    ``down`` E4M3 [6144, columns] row-major, their E8M0 scales in FlashInfer's
    128x4 swizzled layout (``block_scale_interleave``); ``fc1_records`` /
    ``fc2_records`` from ``shared_records``; SwiGLU-OAI with
    ``shapes.SWIGLU_*``.
    """
    rows = x.size(0)
    row_tiles = -(-rows // TILE)
    tiles = SPLITS * row_tiles
    device = x.device
    index = device.index
    x8 = torch.empty(row_tiles * TILE * HIDDEN, dtype=torch.uint8, device=device)
    x8_scale = torch.empty(
        tiles * shared_tile_scale_bytes(SPLIT_K), dtype=torch.uint8, device=device
    )
    partial = torch.empty(
        tiles * TILE * 2 * columns, dtype=torch.float32, device=device
    )
    h8 = torch.empty(row_tiles * TILE * columns, dtype=torch.uint8, device=device)
    h8_scale = torch.empty(
        row_tiles * shared_tile_scale_bytes(columns), dtype=torch.uint8, device=device
    )
    output = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device)
    stream = current_cu_stream()
    sms = _sm_count(index)
    _shared_input(index)(
        _ptr(cutlass.Uint32, x),
        _ptr(cutlass.Uint32, x8),
        _ptr(cutlass.Uint8, x8_scale),
        Int32(rows),
        Int32(row_tiles),
        stream,
    )
    _shared_fc1(index, columns)(
        _ptr(cutlass.Float8E4M3FN, gate_up),
        _ptr(cutlass.Float8E8M0FNU, gate_up_scale),
        _ptr(cutlass.Float8E4M3FN, x8),
        _ptr(cutlass.Float8E8M0FNU, x8_scale),
        _ptr(cutlass.Float32, partial),
        _ptr(cutlass.Int32, fc1_records[rows - 1]),
        Int32(row_tiles),
        Int32(SHARED_FC1_WORDS),
        Int32(min(sms, SHARED_FC1_CTAS)),
        stream,
    )
    _shared_act(index, columns)(
        _ptr(cutlass.Float32, partial),
        _ptr(cutlass.Uint32, h8),
        _ptr(cutlass.Uint8, h8_scale),
        Int32(rows),
        Int32(row_tiles),
        stream,
    )
    _shared_fc2(index, columns)(
        _ptr(cutlass.Float8E4M3FN, down),
        _ptr(cutlass.Float8E8M0FNU, down_scale),
        _ptr(cutlass.Float8E4M3FN, h8),
        _ptr(cutlass.Float8E8M0FNU, h8_scale),
        _ptr(cutlass.BFloat16, output),
        _ptr(cutlass.Int32, fc2_records[rows - 1]),
        Int32(rows),
        Int32(row_tiles),
        Int32(SHARED_FC2_WORDS),
        Int32(min(sms, SHARED_FC2_CTAS)),
        stream,
    )
    return output


def _record(experts: list[int], limits: list[int], words: int) -> list[int]:
    """A static routing record (``route_cute.py``'s first words) of
    ``len(experts)`` tiles in a table of ``words``."""
    pad = [0] * (words - len(experts))
    return [TILE * len(experts), len(experts), 0, 0] + experts + pad + limits + pad


def _ptr(dtype: type[cutlass.Numeric], tensor: torch.Tensor, align: int = 16):
    """A CuTe DSL global-memory pointer to ``tensor``'s data."""
    return make_ptr(
        dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=align
    )


def _fake(dtype: type[cutlass.Numeric], align: int = 16):
    """A placeholder pointer for compilation."""
    return make_ptr(dtype, align, cute.AddressSpace.gmem, assumed_align=align)


@cache
def _front(device_index: int, built: int, fp32_weights: bool):
    """``LowBatchFront`` for calls of up to ``built`` rows."""
    with torch.cuda.device(device_index):
        return cute.compile(
            LowBatchFront(built, fp32_weights),
            _fake(cutlass.Float32),
            _fake(cutlass.Float32),
            Float32(1.0),
            _fake(cutlass.Uint32),
            _fake(cutlass.Float32 if fp32_weights else cutlass.Uint16),
            _fake(cutlass.Uint32),
            _fake(cutlass.Uint8),
            _fake(cutlass.Float32),
            _fake(cutlass.Int32),
            _fake(cutlass.Int32),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _fc1(device_index: int, columns: int, overlap: bool):
    with torch.cuda.device(device_index):
        return cute.compile(
            RoutedFc1(HIDDEN, columns, overlap, ROUTED_FC1_CTA_K, ROUTED_FC1_STAGES),
            _fake(cutlass.Float4E2M1FN, 32),
            _fake(cutlass.Float8E4M3FN),
            _fake(cutlass.Float4E2M1FN, 32),
            _fake(cutlass.Uint8),
            _fake(cutlass.Uint32),
            _fake(cutlass.Float32),
            _fake(cutlass.Float32),
            _fake(cutlass.Uint32),
            _fake(cutlass.Uint32),
            _fake(cutlass.Uint8),
            _fake(cutlass.Float32),
            _fake(cutlass.Int32),
            _fake(cutlass.Int32),
            Int32(1),
            Int32(EXPERTS),
            Int32(4),
            Int32(TILE * TOP_K),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _fc2(device_index: int, columns: int, overlap: bool):
    with torch.cuda.device(device_index):
        return cute.compile(
            RoutedFc2(
                HIDDEN,
                columns,
                # FC1's m tiles: a tile's ready count.
                2 * columns // 128 if overlap else 0,
                FC2_CTA_K[columns],
                ROUTED_FC2_STAGES_BY_WIDTH[columns],
            ),
            _fake(cutlass.Float4E2M1FN, 32),
            _fake(cutlass.Float8E4M3FN),
            _fake(cutlass.Float4E2M1FN, 32),
            _fake(cutlass.Float8E4M3FN),
            _fake(cutlass.Float32),
            _fake(cutlass.Float32),
            _fake(cutlass.BFloat16),
            _fake(cutlass.Int32),
            Int32(TILE * TOP_K),
            Int32(EXPERTS),
            Int32(4),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _shared_input(device_index: int):
    with torch.cuda.device(device_index):
        return cute.compile(
            SharedInputQuant(),
            _fake(cutlass.Uint32),
            _fake(cutlass.Uint32),
            _fake(cutlass.Uint8),
            Int32(1),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _shared_fc1(device_index: int, columns: int):
    with torch.cuda.device(device_index):
        return cute.compile(
            SharedFc1(columns, num_ab_stage=SHARED_FC1_STAGES),
            _fake(cutlass.Float8E4M3FN),
            _fake(cutlass.Float8E8M0FNU),
            _fake(cutlass.Float8E4M3FN),
            _fake(cutlass.Float8E8M0FNU),
            _fake(cutlass.Float32),
            _fake(cutlass.Int32),
            Int32(1),
            Int32(SHARED_FC1_WORDS),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _shared_act(device_index: int, columns: int):
    with torch.cuda.device(device_index):
        return cute.compile(
            SharedAct(columns),
            _fake(cutlass.Float32),
            _fake(cutlass.Uint32),
            _fake(cutlass.Uint8),
            Int32(1),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _shared_fc2(device_index: int, columns: int):
    with torch.cuda.device(device_index):
        return cute.compile(
            SharedFc2(columns, FC2_CTA_K[columns], SHARED_FC2_STAGES[columns]),
            _fake(cutlass.Float8E4M3FN),
            _fake(cutlass.Float8E8M0FNU),
            _fake(cutlass.Float8E4M3FN),
            _fake(cutlass.Float8E8M0FNU),
            _fake(cutlass.BFloat16),
            _fake(cutlass.Int32),
            Int32(1),
            Int32(1),
            Int32(SHARED_FC2_WORDS),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _sm_count(device_index: int) -> int:
    """Streaming multiprocessors of a device: the persistent GEMMs' CTAs."""
    return torch.cuda.get_device_properties(device_index).multi_processor_count
