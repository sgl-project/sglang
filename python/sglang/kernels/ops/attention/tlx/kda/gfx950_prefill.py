# SPDX-License-Identifier: MIT
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
#
# Vendored from facebookexperimental/triton#4243
# (https://github.com/facebookexperimental/triton/pull/4243),
# commit 7d38f0a (7d38f0a66999481017732b395d0c607c12a79770),
# file third_party/tlx/ops/kernels/kda/gfx950_prefill.py.
# Local change, to be upstreamed: kda_paged_prefill's cu_seqlens_cpu.

"""Stateful chunk-parallel Kimi Delta Attention prefill for gfx950.

The kernel accepts normalized Q/K, per-token log decays, and sigmoid-applied
beta coefficients, or the raw inputs those come from. Recurrent states use the
physical ``[N, H, V, K]`` V-major layout, dense or as slots of a state pool.

The implementation uses 64-token chunks and 16-token bands, in two steps:

* one program per (chunk, head) forms the band-local gate cumsum, the causal
  QK and beta-scaled KK products with CDNA4 ``tl.dot`` MFMAs, the 64x64 WY
  inverse, and W/U;
* a walk carries the recurrent state across chunks and adds each chunk's
  causal within-chunk output, split in time when too few sequences would idle
  the GPU.

TLX local-memory async copies stage the walk's per-chunk operands.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from itertools import pairwise

import torch
import triton
import triton.language as tl
from triton.language.extra import tlx
from triton.language.extra.libdevice import fast_dividef
from triton.runtime.jit import constexpr_function

CHUNK_SIZE = 64
SUBCHUNK_SIZE = 16
_KEY_DIM = 128
_VALUE_DIM = 128


def _chunk_pairs(boundaries: list[int], chunk_size: int) -> list[tuple[int, int]]:
    return [
        (sequence, local_chunk)
        for sequence, (begin, end) in enumerate(pairwise(boundaries))
        for local_chunk in range((end - begin + chunk_size - 1) // chunk_size)
    ]


def prepare_chunk_indices(
    cu_seqlens: torch.Tensor, chunk_size: int = CHUNK_SIZE
) -> torch.Tensor:
    """Map packed global chunks to ``(sequence, local_chunk)``."""
    if cu_seqlens.ndim != 1:
        raise ValueError("cu_seqlens must be a vector")
    boundaries = cu_seqlens.detach().to(device="cpu", dtype=torch.int64).tolist()
    pairs = _chunk_pairs(boundaries, chunk_size)
    if not pairs:
        return torch.empty((0, 2), dtype=torch.int32, device=cu_seqlens.device)
    return torch.tensor(pairs, dtype=torch.int32, device=cu_seqlens.device)


_PREFILL_METADATA_CACHE = []


def _prepare_prefill_metadata(
    cu_seqlens: torch.Tensor,
    device: torch.device,
    total_tokens: int,
    cu_seqlens_cpu: Sequence[int] | None = None,
):
    """Cache the latest packed schedule by tensor identity and version.

    With ``cu_seqlens_cpu``, the same boundaries on the host, the schedule is
    built and uploaded without synchronizing the device.
    """
    version = cu_seqlens._version
    if _PREFILL_METADATA_CACHE:
        source, cached_version, cached_device, cached_tokens, value = (
            _PREFILL_METADATA_CACHE
        )
        if source is cu_seqlens and (cached_version, cached_device, cached_tokens) == (
            version,
            device,
            total_tokens,
        ):
            return value

    if cu_seqlens_cpu is None:
        boundaries = cu_seqlens.detach().to(device="cpu", dtype=torch.int64).tolist()
    else:
        boundaries = [int(boundary) for boundary in cu_seqlens_cpu]
        if len(boundaries) != cu_seqlens.numel():
            raise ValueError("cu_seqlens_cpu must have as many entries as cu_seqlens")
    if not boundaries or boundaries[0] != 0 or boundaries[-1] != total_tokens:
        raise ValueError(
            "cu_seqlens must start at zero and end at the packed token count"
        )
    if any(end < begin for begin, end in pairwise(boundaries)):
        raise ValueError("cu_seqlens must be nondecreasing")

    cu_device = cu_seqlens
    if (
        cu_device.device != device
        or cu_device.dtype != torch.int32
        or not cu_device.is_contiguous()
    ):
        cu_device = cu_device.to(device=device, dtype=torch.int32).contiguous()
    pairs = _chunk_pairs(boundaries, CHUNK_SIZE)
    if not pairs:
        chunk_indices = torch.empty((0, 2), dtype=torch.int32, device=device)
    elif cu_seqlens_cpu is None:
        chunk_indices = torch.tensor(pairs, dtype=torch.int32, device=device)
    else:
        # A pageable host-to-device copy would synchronize the stream.
        chunk_indices = (
            torch.tensor(pairs, dtype=torch.int32)
            .pin_memory()
            .to(device, non_blocking=True)
        )
    value = (cu_device, chunk_indices)
    _PREFILL_METADATA_CACHE[:] = [cu_seqlens, version, device, total_tokens, value]
    return value


@triton.jit
def _mm_f32(
    lhs, rhs, solve_layout: tl.constexpr, solve_a: tl.constexpr, solve_b: tl.constexpr
):
    zero = tlx.zeros(lhs.shape, tl.float32, layout=solve_layout)
    product = tl.dot(
        tlx.require_layout(lhs, solve_a),
        tlx.require_layout(rhs, solve_b),
        acc=zero,
        input_precision="ieee",
    )
    return tlx.require_layout(product, solve_layout)


@triton.jit
def _along(n: tl.constexpr, axis: tl.constexpr):
    """arange(n) along ``axis`` of a rank-3 tensor.

    Built by reshape: uTLX re-lays index math feeding a dot operand out in the
    operand's layout, and fails on chained expand_dims.
    """
    return tl.reshape(
        tl.arange(0, n),
        [n if axis == 0 else 1, n if axis == 1 else 1, n if axis == 2 else 1],
    )


@triton.jit
def _along2(n: tl.constexpr, axis: tl.constexpr):
    """arange(n) along ``axis`` of a rank-2 tensor, as for ``_along``."""
    return tl.reshape(tl.arange(0, n), [n if axis == 0 else 1, n if axis == 1 else 1])


@constexpr_function
def _row_padded(rows, cols, pad, bands=0):
    """Row-major LDS image with ``pad`` elements after each row, as [bands, rows // bands, cols] if ``bands``.

    Explicit, because the image picked by default for these buffers'
    transposed reads is slower.
    """
    shape = [bands, rows // bands, cols] if bands else [rows, cols]
    # getattr: a plain attribute would make the layout class a global the
    # kernels depend on, and such kernels skip the C launch fast path.
    return getattr(tlx, "padded_shared_layout_encoding").with_identity_for(
        [[cols, pad]], shape
    )


@triton.jit
def _shift_bands(
    diagonal, s: tl.constexpr, NB: tl.constexpr, BC: tl.constexpr, layout: tl.constexpr
):
    """The [NB, BC, BC] blocks in LDS ``diagonal`` with band b - s moved to band b, in ``layout``.

    Read a band at a time, as subslices must be aligned to their size. Bands
    below s are zero.
    """
    band = tlx.require_layout(
        tl.arange(0, NB), tlx.slice_layout(tlx.slice_layout(layout, 2), 1)
    )[:, None, None]
    shifted = tlx.zeros((NB, BC, BC), tl.float32, layout=layout)
    for b in tl.static_range(s, NB):
        part = tlx.local_load(
            tlx.local_slice(diagonal, [b - s, 0, 0], [1, BC, BC]), layout=layout
        )
        shifted = tl.where(band == b, part, shifted)
    return shifted


@triton.jit
def _prepare_chunk_kernel(
    Q,
    K,
    V,
    G,
    Beta,
    Qg,
    W,
    U,
    Kg,
    Aqk,
    Decay,
    CuSeqLens,
    ChunkIndices,
    ALog,
    DtBias,
    lower_bound,
    SCALE: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    NW: tl.constexpr,
    STRIDE_Q: tl.constexpr,
    STRIDE_K: tl.constexpr,
    STRIDE_V: tl.constexpr,
    STRIDE_G: tl.constexpr,
    STRIDE_BETA: tl.constexpr,
    USE_QK_L2NORM_AND_GATE: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    USE_BETA_SIGMOID: tl.constexpr,
):
    """Everything the walk reads for one (chunk, head), in one program.

    The chunk is held as [NB, BC, D] bands, one per warp, so the band-local
    gate cumsum and every [BC, BC] block product run batched over bands. Akk
    never leaves registers: block (b, J) of all bands comes from one batched
    MFMA per column band J, each band's diagonal block is inverted by repeated
    squaring, and the off-diagonal inverse blocks are built a diagonal at a
    time from band-shifted copies read back from LDS.
    """
    NB: tl.constexpr = BT // BC
    LOG2E: tl.constexpr = 1.4426950408889634
    LN2: tl.constexpr = 0.6931471805599453
    chunk = tl.program_id(0)
    head = tl.program_id(1)
    sequence = tl.load(ChunkIndices + chunk * 2).to(tl.int32)
    local_chunk = tl.load(ChunkIndices + chunk * 2 + 1).to(tl.int32)
    begin = tl.load(CuSeqLens + sequence).to(tl.int32)
    token0 = local_chunk * BT
    rows_left = tl.load(CuSeqLens + sequence + 1).to(tl.int32) - begin - token0

    mb: tl.constexpr = tlx.amd_mfma_layout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[NW, 1, 1]
    )
    a3: tl.constexpr = tlx.dot_operand_layout(0, mb, k_width=8)
    b3: tl.constexpr = tlx.dot_operand_layout(1, mb, k_width=8)
    mf: tl.constexpr = tlx.amd_mfma_layout(
        version=4, instr_shape=[16, 16, 4], transposed=True, warps_per_cta=[NW, 1, 1]
    )
    af: tl.constexpr = tlx.dot_operand_layout(0, mf, k_width=1)
    bf: tl.constexpr = tlx.dot_operand_layout(1, mf, k_width=1)
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[NW, 1]
    )
    a_op: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    b_op: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)

    band = _along(NB, 0)
    row = band * BC + _along(BC, 1)
    chan = _along(D, 2)
    # [NB, BC, D] tiles live in the fp32 MFMA result layout, a band per warp:
    # scans only lower for blocked layouts, so the in-band cumsum is a
    # lower-triangular product and only the [NB, D] band scan is released.
    # Buffer ops keep these layouts: pointer-tensor loads get re-coalesced.
    # Masks are rebuilt from offsets in each layout (row < rows_left exactly
    # when offset < rows_left * H * D): uTLX cannot move i1 tensors via LDS.
    # Inputs keep their token strides (split views of a packed qkv); outputs
    # are dense [T, H, D].
    tile_base = ((begin + token0) * H + head) * D
    tile_limit = rows_left * (H * D)
    tile_offsets = tlx.require_layout(row * (H * D) + chan, mf)
    tile_mask = tile_offsets < tile_limit
    start = (begin + token0).to(tl.int64)
    q_offsets = tlx.require_layout(row * STRIDE_Q + chan, mf)
    k_offsets = tlx.require_layout(row * STRIDE_K + chan, mf)
    v_offsets = tlx.require_layout(row * STRIDE_V + chan, mf)
    g_offsets = tlx.require_layout(row * STRIDE_G + chan, bf)
    g_rows = tlx.require_layout(tl.minimum(row, rows_left - 1) * STRIDE_G + chan, bf)
    beta_offsets = tlx.require_layout(row * STRIDE_BETA, mf)
    # Inputs stay bf16 until their turn below, so few fp32 tiles are live at once.
    q = tlx.buffer_load(
        Q + (start * STRIDE_Q + head * D),
        q_offsets,
        mask=q_offsets < rows_left * STRIDE_Q,
        other=0.0,
    )
    k = tlx.buffer_load(
        K + (start * STRIDE_K + head * D),
        k_offsets,
        mask=k_offsets < rows_left * STRIDE_K,
        other=0.0,
    )
    v = tlx.buffer_load(
        V + (start * STRIDE_V + head * D),
        v_offsets,
        mask=v_offsets < rows_left * STRIDE_V,
        other=0.0,
    )
    # Unmasked, with rows past the sequence rereading its last row, then
    # zeroed: uTLX keys buffer_load helpers on dtype and shape but not layout,
    # so a masked bf16 load here would reuse q's layout.
    g_mask = g_offsets < rows_left * STRIDE_G
    g = tlx.buffer_load(G + (start * STRIDE_G + head * D), g_rows).to(tl.float32)
    beta_mask = beta_offsets < rows_left * STRIDE_BETA
    beta = tlx.buffer_load(
        Beta + (start * STRIDE_BETA + head), beta_offsets, mask=beta_mask, other=0.0
    ).to(tl.float32)
    if USE_QK_L2NORM_AND_GATE:
        # What prepare_kda_inputs does, but the activated gate here and the
        # l2-normalized Q/K below stay fp32 until they become MFMA operands.
        # A [D] vector broadcast in registers: a load with the tile's
        # offsets is slower.
        bias = tlx.release_layout(
            tl.load(
                DtBias
                + head * D
                + tlx.require_layout(
                    tl.arange(0, D), tlx.slice_layout(tlx.slice_layout(bf, 1), 0)
                )
            )
        )
        rate = tl.exp(tl.load(ALog + head).to(tl.float32))
        if USE_LOWER_BOUND:
            # exp(-rate (g + bias)) as one exp2 of an FMA.
            slope = -rate * LOG2E
            decay = tl.math.exp2(g * slope + (slope * bias)[None, None, :])
            g = fast_dividef(lower_bound, 1.0 + decay)
        else:
            x = g + bias[None, None, :]
            g = -rate * tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
    if USE_BETA_SIGMOID:
        beta = tl.where(beta_mask, tl.sigmoid(beta), 0.0)
    g = tl.where(g_mask, g, 0.0)

    # G = pivot + in_band, in log2 units: each band's gate relative to the
    # token before it. exp(G) and exp(last - G) are split into a per-band
    # factor and exp(+-in_band), so no fp32 gate tile stays live.
    lower = tlx.require_layout(
        tl.broadcast_to(
            tl.where(_along(BC, 2) <= _along(BC, 1), LOG2E, 0.0), (NB, BC, BC)
        ),
        af,
    )
    # Released: exp(+-in_band) scales the q and k tiles, which keep their own layout.
    in_band = tlx.release_layout(
        tl.dot(
            lower,
            g,
            acc=tlx.zeros((NB, BC, D), tl.float32, layout=mf),
            input_precision="ieee",
        )
    )
    band_total = tlx.release_layout(tl.sum(g, axis=1)) * LOG2E
    pivot = tl.cumsum(band_total, axis=0) - band_total
    last = tl.sum(band_total, axis=0)
    tl.store(
        Decay
        + ((begin // BT + sequence + local_chunk) * H + head) * D
        + tl.arange(0, D),
        last * LN2,
    )
    rise = tl.math.exp2(in_band)
    fall = tl.math.exp2(-in_band)
    to_pivot = tlx.require_layout(tl.math.exp2(pivot)[:, None, :], mf)
    to_last = tlx.require_layout(tl.math.exp2(last[None, :] - pivot)[:, None, :], mf)

    # The W/U right-hand sides wait in LDS, not registers, until the inverse is done.
    key_rhs = tlx.local_alloc((BT, D), tl.bfloat16, 1, layout=_row_padded(BT, D, 8))
    value_rhs = tlx.local_alloc((BT, D), tl.bfloat16, 1, layout=_row_padded(BT, D, 8))
    tlx.local_store(
        tlx.local_view(value_rhs, 0),
        tl.reshape((v.to(tl.float32) * beta).to(tl.bfloat16), [BT, D]),
    )

    # Rows decayed to their band's pivot (lhs) and columns lifted from it; a
    # bridge exp(pivot_b - pivot_J) on row band b's rows meets column band J.
    q = q.to(tl.float32)
    if USE_QK_L2NORM_AND_GATE:
        q = q * tl.math.rsqrt(tl.sum(q * q, axis=2) + 1e-6)[:, :, None]
    q_rows = q * rise * SCALE
    tlx.buffer_store(
        (q_rows * to_pivot).to(Qg.dtype.element_ty),
        Qg + tile_base,
        tile_offsets,
        mask=tile_mask,
    )
    q_lhs = tlx.require_layout(q_rows.to(tl.bfloat16), a3)
    k = k.to(tl.float32)
    if USE_QK_L2NORM_AND_GATE:
        k = k * tl.math.rsqrt(tl.sum(k * k, axis=2) + 1e-6)[:, :, None]
    k_cols = k * fall
    tlx.buffer_store(
        (k_cols * to_last).to(Kg.dtype.element_ty),
        Kg + tile_base,
        tile_offsets,
        mask=tile_mask,
    )
    lifted = tlx.local_alloc(
        (NB, BC, D), tl.bfloat16, 1, layout=_row_padded(NB * BC, D, 8, NB)
    )
    tlx.local_store(tlx.local_view(lifted, 0), k_cols.to(tl.bfloat16))
    k_rows = k * rise
    tlx.local_store(
        tlx.local_view(key_rhs, 0),
        tl.reshape((k_rows * to_pivot * beta).to(tl.bfloat16), [BT, D]),
    )
    k_lhs = tlx.require_layout(k_rows.to(tl.bfloat16), a3)
    band_pivot = tlx.release_layout(pivot)
    # bridges[J, b] = exp(pivot_b - pivot_J), clamped where b < J and the block is masked anyway.
    bridges = tlx.local_alloc((NB * NB, D), tl.float32, 1)
    pivots = tlx.local_view(tlx.local_alloc((NB, D), tl.float32, 1), 0)
    tlx.local_store(pivots, band_pivot)
    tl.debug_barrier()
    for J in tl.static_range(NB):
        pivot_J = tlx.local_load(tlx.local_slice(pivots, [J, 0], [1, D]))
        tlx.local_store(
            tlx.local_slice(tlx.local_view(bridges, 0), [J * NB, 0], [NB, D]),
            tl.math.exp2(tl.minimum(band_pivot - pivot_J, 0.0)),
        )
    tl.debug_barrier()

    # Block products live in the MFMA result layout, not the tiles'.
    block_rows = tlx.release_layout(row)
    block_beta = tlx.release_layout(beta)
    block_cols = _along(BC, 2)
    aqk_base = Aqk + (start * H + head) * BT
    aqk_offsets = block_rows * (H * BT) + block_cols
    aqk_mask = aqk_offsets < rows_left * (H * BT)
    band_mb = tl.broadcast_to(band, (NB, BC, BC))
    n0 = tl.zeros((NB, BC, BC), tl.float32)
    n1 = tl.zeros((NB, BC, BC), tl.float32)
    n2 = tl.zeros((NB, BC, BC), tl.float32)
    n3 = tl.zeros((NB, BC, BC), tl.float32)
    for J in tl.static_range(NB):
        bridge = tlx.release_layout(
            tlx.local_load(
                tlx.local_slice(tlx.local_view(bridges, 0), [J * NB, 0], [NB, D]),
                layout=tlx.slice_layout(a3, 1),
            )
        )[:, None, :]
        column_band = tlx.local_load(
            tlx.local_trans(
                tlx.local_slice(tlx.local_view(lifted, 0), [J, 0, 0], [1, BC, D]),
                (0, 2, 1),
            ),
            layout=b3,
        )
        rhs = tlx.require_layout(tl.broadcast_to(column_band, (NB, D, BC)), b3)
        k_bridged = tlx.require_layout(
            (k_lhs.to(tl.float32) * bridge).to(tl.bfloat16), a3
        )
        q_bridged = tlx.require_layout(
            (q_lhs.to(tl.float32) * bridge).to(tl.bfloat16), a3
        )
        akk = tl.dot(k_bridged, rhs, acc=tlx.zeros((NB, BC, BC), tl.float32, layout=mb))
        aqk = tl.dot(q_bridged, rhs, acc=tlx.zeros((NB, BC, BC), tl.float32, layout=mb))
        column = J * BC + block_cols
        tlx.buffer_store(
            tl.where(block_rows >= column, aqk, 0.0).to(Aqk.dtype.element_ty),
            aqk_base + J * BC,
            aqk_offsets,
            mask=aqk_mask,
        )
        akk = tl.where(block_rows > column, -block_beta * akk, 0.0)
        n0 = tl.where(band_mb == J, akk, n0)
        n1 = tl.where(band_mb == J + 1, akk, n1)
        n2 = tl.where(band_mb == J + 2, akk, n2)
        n3 = tl.where(band_mb == J + 3, akk, n3)
    # Diagonal blocks: (I - N)^-1 = (I + N)(I + N^2)(I + N^4)(I + N^8).
    tl.static_assert(NB == 4, "the inverse is unrolled for four bands")
    sub_r = _along(BC, 1)
    sub_c = _along(BC, 2)
    n0 = tlx.require_layout(n0, mf)
    power = _mm_f32(n0, n0, mf, af, bf)
    x0 = n0 + (sub_r == sub_c).to(tl.float32)
    x0 += _mm_f32(x0, power, mf, af, bf)
    power = _mm_f32(power, power, mf, af, bf)
    x0 += _mm_f32(x0, power, mf, af, bf)
    power = _mm_f32(power, power, mf, af, bf)
    x0 += _mm_f32(x0, power, mf, af, bf)

    # Off-diagonals X_{b, b-d} = X_bb sum_s N_{b, b-s} X_{b-s, b-d}, with
    # X_{b-s, b-d} diagonal d - s shifted down s bands. N_{b, b-s} is zero for
    # b < s, so so is diagonal d below band d.
    n1 = tlx.require_layout(n1, af)
    n2 = tlx.require_layout(n2, af)
    n3 = tlx.require_layout(n3, af)
    diagonal0 = tlx.local_view(tlx.local_alloc((NB, BC, BC), tl.float32, 1), 0)
    diagonal1 = tlx.local_view(tlx.local_alloc((NB, BC, BC), tl.float32, 1), 0)
    diagonal2 = tlx.local_view(tlx.local_alloc((NB, BC, BC), tl.float32, 1), 0)
    tlx.local_store(diagonal0, x0)
    tl.debug_barrier()
    acc = tl.dot(
        n1,
        _shift_bands(diagonal0, 1, NB, BC, bf),
        acc=tlx.zeros((NB, BC, BC), tl.float32, layout=mf),
        input_precision="ieee",
    )
    x1 = _mm_f32(x0, tlx.require_layout(acc, mf), mf, af, bf)
    tlx.local_store(diagonal1, x1)
    tl.debug_barrier()
    acc = tl.dot(
        n1,
        _shift_bands(diagonal1, 1, NB, BC, bf),
        acc=tlx.zeros((NB, BC, BC), tl.float32, layout=mf),
        input_precision="ieee",
    )
    acc = tl.dot(
        n2, _shift_bands(diagonal0, 2, NB, BC, bf), acc=acc, input_precision="ieee"
    )
    x2 = _mm_f32(x0, tlx.require_layout(acc, mf), mf, af, bf)
    tlx.local_store(diagonal2, x2)
    tl.debug_barrier()
    acc = tl.dot(
        n1,
        _shift_bands(diagonal2, 1, NB, BC, bf),
        acc=tlx.zeros((NB, BC, BC), tl.float32, layout=mf),
        input_precision="ieee",
    )
    acc = tl.dot(
        n2, _shift_bands(diagonal1, 2, NB, BC, bf), acc=acc, input_precision="ieee"
    )
    acc = tl.dot(
        n3, _shift_bands(diagonal0, 3, NB, BC, bf), acc=acc, input_precision="ieee"
    )
    x3 = _mm_f32(x0, tlx.require_layout(acc, mf), mf, af, bf)
    # A^-1 by column band: band b of column band J is diagonal b - J. Then W
    # and U are plain [BT, BT] x [BT, D] products.
    inverse = tlx.local_alloc((BT, BT), tl.bfloat16, 1, layout=_row_padded(BT, BT, 8))
    selector = tlx.release_layout(band) + tl.zeros((NB, BC, BC), tl.int32)
    for J in tl.static_range(NB):
        block = tl.where(selector == J, x0, 0.0)
        block = tl.where(selector == J + 1, x1, block)
        block = tl.where(selector == J + 2, x2, block)
        block = tl.where(selector == J + 3, x3, block)
        tlx.local_store(
            tlx.local_slice(tlx.local_view(inverse, 0), [0, J * BC], [BT, BC]),
            tl.reshape(block.to(tl.bfloat16), [BT, BC]),
        )
    tl.debug_barrier()
    inverse = tlx.local_load(tlx.local_view(inverse, 0), layout=a_op)
    # Stored from the A-operand layout: eight contiguous channels per lane.
    flat_offsets = tlx.require_layout(_along2(BT, 0) * (H * D) + _along2(D, 1), a_op)
    w = tl.dot(
        inverse,
        tlx.local_load(tlx.local_view(key_rhs, 0), layout=b_op),
        acc=tlx.zeros((BT, D), tl.float32, layout=mma),
    )
    tlx.buffer_store(
        tlx.require_layout(w.to(W.dtype.element_ty), a_op),
        W + tile_base,
        flat_offsets,
        mask=flat_offsets < tile_limit,
    )
    u = tl.dot(
        inverse,
        tlx.local_load(tlx.local_view(value_rhs, 0), layout=b_op),
        acc=tlx.zeros((BT, D), tl.float32, layout=mma),
    )
    tlx.buffer_store(
        tlx.require_layout(u.to(U.dtype.element_ty), a_op),
        U + tile_base,
        flat_offsets,
        mask=flat_offsets < tile_limit,
    )


@triton.jit
def _decay_row(
    begin, sequence, head, H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr
):
    """Offset of a sequence's first chunk row in the per-chunk decay table.

    begin // BT + sequence + local_chunk is distinct for every chunk of every
    sequence and stays below T // BT + 2 * N, so the table needs no chunk offsets.
    """
    return ((begin // BT + sequence) * H + head) * K


@constexpr_function
def _ring_layout(rows, cols, pad):
    """Padded LDS image for a [rows, cols] bf16 slot that async copies fill and MFMA operands read.

    One wave's direct-to-LDS copy writes 512 contiguous elements, so padding
    goes after each 512. The rows within a 16-row MFMA read sit in different
    512-element blocks, which gives each a distinct pad and so a distinct bank.
    """
    column_bits = cols.bit_length() - 1
    row_bits = rows.bit_length() - 1
    block_row_bits = (512 // cols).bit_length() - 1
    low_row_bits = row_bits - block_row_bits
    bases = [[0, 1 << b] for b in range(column_bits)]
    bases += [[1 << b, 0] for b in range(low_row_bits, row_bits)]
    bases += [[1 << b, 0] for b in range(low_row_bits)]
    # getattr, as in _row_padded.
    return getattr(tlx, "padded_shared_layout_encoding").with_bases(
        [[512, pad]], bases, [rows, cols]
    )


@triton.jit
def _copy_walk_chunk(
    W,
    Kg,
    Qg,
    U,
    Aqk,
    Decay,
    w_buffers,
    kg_buffers,
    q_buffers,
    u_buffers,
    a_buffers,
    decay_buffers,
    slot,
    key_base,
    value_base,
    decay_base,
    token0,
    length,
    copy_rows,
    keys,
    u_offsets,
    u_columns,
    a_offsets,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    WRITE_OUTPUT: tl.constexpr,
):
    """Start one chunk's async copies into a ring slot, as a single commit group.

    One group per chunk keeps the wait a plain "one group pending" that both
    TLX backends turn into a partial vmcnt.
    """
    rows_left = copy_rows[:, None] < length - token0
    key_mask = rows_left & (keys[None, :] < K)
    key_offsets = key_base + (token0 + copy_rows[:, None]) * H * K + keys[None, :]
    w = tlx.async_load(W + key_offsets, tlx.local_view(w_buffers, slot), mask=key_mask)
    kg = tlx.async_load(
        Kg + key_offsets, tlx.local_view(kg_buffers, slot), mask=key_mask
    )
    u = tlx.async_load(
        U + value_base + token0 * H * V + u_offsets,
        tlx.local_view(u_buffers, slot),
        mask=rows_left & (u_columns[None, :] < V),
    )
    decay = tlx.async_load(
        Decay + decay_base + (token0 // BT) * H * K + keys,
        tlx.local_view(decay_buffers, slot),
        mask=keys < K,
    )
    if WRITE_OUTPUT:
        q = tlx.async_load(
            Qg + key_offsets, tlx.local_view(q_buffers, slot), mask=key_mask
        )
        a = tlx.async_load(
            Aqk + token0 * H * BT + a_offsets,
            tlx.local_view(a_buffers, slot),
            mask=rows_left & (copy_rows[None, :] < BT),
        )
        tlx.async_load_commit_group([w, kg, u, decay, q, a])
    else:
        tlx.async_load_commit_group([w, kg, u, decay])


@triton.jit
def _group_chunks(num_chunks, num_groups, group):
    """Chunks [lo, hi) of one time group: near-equal runs, the first ones a chunk longer."""
    base_chunks = num_chunks // num_groups
    extra_chunks = num_chunks - base_chunks * num_groups
    chunk_lo = group * base_chunks + tl.minimum(group, extra_chunks)
    return chunk_lo, chunk_lo + base_chunks + tl.where(group < extra_chunks, 1, 0)


@triton.jit
def _load_state_rows(
    start,
    value_block,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    NW: tl.constexpr,
):
    """Load one [BV, K] row slice of a [V, K] state at ``start``, in the walk's MFMA layout.

    Rows past V (a group summary's transfer rows) load as zero. Returns the
    slice, its offsets and mask, and the identity on the transfer rows, zero
    on the value rows.
    """
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[NW, 1]
    )
    values = value_block * BV + tl.arange(0, BV)
    keys = tl.arange(0, K)
    offsets = tlx.require_layout(
        (values[:, None] * K + keys[None, :]).to(tl.int32), mma, pin=False
    )
    mask = (
        tlx.require_layout(
            ((values[:, None] < V) & (keys[None, :] < K)).to(tl.int8), mma, pin=False
        )
        != 0
    )
    state = tlx.buffer_load(start, offsets, mask=mask, other=0.0).to(tl.float32)
    state = tlx.require_layout(state, mma, pin=False)
    eye = tlx.require_layout(
        (keys[None, :] == values[:, None] - V).to(tl.float32), mma, pin=False
    )
    return state, offsets, mask, eye


@triton.jit
def _walk_chunks(
    state,
    W,
    Kg,
    Qg,
    U,
    Aqk,
    Decay,
    Output,
    begin,
    length,
    sequence,
    head,
    chunk_lo,
    chunk_hi,
    value_block,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    NW: tl.constexpr,
    WRITE_OUTPUT: tl.constexpr,
):
    """Carry one [BV, K] row slice of a state across chunks [chunk_lo, chunk_hi) and return it.

    Every warp owns whole state rows, so the state, v_new and their bf16
    operands stay in registers. Each chunk's W, Kg, U, decay, Qg and Aqk come
    through a two-slot shared-memory ring. With WRITE_OUTPUT each chunk's
    output is formed as o = Qg S^T + Aqk v_new^T in a token-major layout, so
    neither v_new nor a partial output goes back to memory.
    """
    TH: tl.constexpr = BT // 2
    # The state and v_new: warps along value rows. The output: warps along
    # value columns, so its stores run along V.
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[NW, 1]
    )
    a_op: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=4)
    b_op: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=4)
    key_layout: tl.constexpr = tlx.slice_layout(mma, 0)
    out_mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, NW]
    )
    out_a_op: tl.constexpr = tlx.dot_operand_layout(0, out_mma, k_width=4)
    out_b_op: tl.constexpr = tlx.dot_operand_layout(1, out_mma, k_width=4)

    state = tlx.require_layout(state, mma, pin=False)
    values = value_block * BV + tl.arange(0, BV)
    keys = tl.arange(0, K)
    key_base = (begin * H + head) * K
    value_base = (begin * H + head) * V
    decay_base = _decay_row(begin, sequence, head, H, K, BT)
    copy_rows = tl.arange(0, BT)
    u_offsets = copy_rows[:, None] * H * V + values[None, :]
    a_columns = tl.arange(0, BT)
    a_offsets = (
        (copy_rows[:, None] * H + head) * BT + a_columns[None, :] + begin * H * BT
    )
    # Transposed reads (w, u) want a 4-element pad; kg's plain b-operand
    # reads want 16.
    w_buffers = tlx.local_alloc((BT, K), tl.bfloat16, 2, layout=_ring_layout(BT, K, 4))
    kg_buffers = tlx.local_alloc(
        (BT, K), tl.bfloat16, 2, layout=_ring_layout(BT, K, 16)
    )
    u_buffers = tlx.local_alloc(
        (BT, BV), tl.bfloat16, 2, layout=_ring_layout(BT, BV, 4)
    )
    decay_buffers = tlx.local_alloc((K,), tl.float32, 2)
    if WRITE_OUTPUT:
        q_buffers = tlx.local_alloc(
            (BT, K), tl.bfloat16, 2, layout=_ring_layout(BT, K, 4)
        )
        a_buffers = tlx.local_alloc(
            (BT, BT), tl.bfloat16, 2, layout=_ring_layout(BT, BT, 4)
        )
        out_rows = tl.arange(0, BT)
        out_offsets = tlx.require_layout(
            (out_rows[:, None] * H * V + values[None, :]).to(tl.int32), out_mma
        )
    else:
        q_buffers = w_buffers
        a_buffers = w_buffers

    _copy_walk_chunk(
        W,
        Kg,
        Qg,
        U,
        Aqk,
        Decay,
        w_buffers,
        kg_buffers,
        q_buffers,
        u_buffers,
        a_buffers,
        decay_buffers,
        0,
        key_base,
        value_base,
        decay_base,
        chunk_lo * BT,
        length,
        copy_rows,
        keys,
        u_offsets,
        values,
        a_offsets,
        H,
        K,
        V,
        BT,
        WRITE_OUTPUT,
    )
    for local_chunk in tl.range(chunk_lo, chunk_hi, num_stages=1):
        token0 = local_chunk * BT
        slot = (local_chunk - chunk_lo) % 2
        # Wait before issuing the next chunk's copies, so the wait never
        # covers the prefetch, even where it is lowered to vmcnt(0).
        ready = tlx.async_load_wait_group(0)
        _copy_walk_chunk(
            W,
            Kg,
            Qg,
            U,
            Aqk,
            Decay,
            w_buffers,
            kg_buffers,
            q_buffers,
            u_buffers,
            a_buffers,
            decay_buffers,
            1 - slot,
            key_base,
            value_base,
            decay_base,
            token0 + BT,
            length,
            copy_rows,
            keys,
            u_offsets,
            values,
            a_offsets,
            H,
            K,
            V,
            BT,
            WRITE_OUTPUT,
        )
        w_tile = tlx.local_view(w_buffers, slot)
        kg_tile = tlx.local_view(kg_buffers, slot)
        u_tile = tlx.local_view(u_buffers, slot)

        # Every operand change below is register-trivial for this MFMA
        # config; pinning each one keeps layout propagation from undoing that.
        state_bf16 = tlx.require_layout(state.to(tl.bfloat16), mma)
        state_lhs = tlx.require_layout(state_bf16, a_op)
        zero = tlx.zeros((BV, TH), tl.float32, layout=mma)
        w0 = tlx.local_load(
            tlx.local_trans(tlx.local_slice(w_tile, [0, 0], [TH, K])),
            token=ready,
            layout=b_op,
        )
        w1 = tlx.local_load(
            tlx.local_trans(tlx.local_slice(w_tile, [TH, 0], [TH, K])),
            token=ready,
            layout=b_op,
        )
        u0 = tlx.local_load(
            tlx.local_trans(tlx.local_slice(u_tile, [0, 0], [TH, BV])),
            token=ready,
            layout=a_op,
        )
        u1 = tlx.local_load(
            tlx.local_trans(tlx.local_slice(u_tile, [TH, 0], [TH, BV])),
            token=ready,
            layout=a_op,
        )
        u0 = tlx.require_layout(u0, mma).to(tl.float32)
        u1 = tlx.require_layout(u1, mma).to(tl.float32)
        v0 = tlx.require_layout(
            (u0 - tlx.require_layout(tl.dot(state_lhs, w0, acc=zero), mma)).to(
                tl.bfloat16
            ),
            mma,
        )
        v1 = tlx.require_layout(
            (u1 - tlx.require_layout(tl.dot(state_lhs, w1, acc=zero), mma)).to(
                tl.bfloat16
            ),
            mma,
        )
        if WRITE_OUTPUT:
            q = tlx.local_load(
                tlx.local_view(q_buffers, slot), token=ready, layout=out_a_op
            )
            a_tile = tlx.local_view(a_buffers, slot)
            a0 = tlx.local_load(
                tlx.local_slice(a_tile, [0, 0], [BT, TH]), token=ready, layout=out_a_op
            )
            a1 = tlx.local_load(
                tlx.local_slice(a_tile, [0, TH], [BT, TH]), token=ready, layout=out_a_op
            )
            out = tlx.zeros((BT, BV), tl.float32, layout=out_mma)
            out = tl.dot(q, tlx.require_layout(tl.trans(state_bf16), out_b_op), acc=out)
            out = tl.dot(a0, tlx.require_layout(tl.trans(v0), out_b_op), acc=out)
            out = tlx.require_layout(
                tl.dot(a1, tlx.require_layout(tl.trans(v1), out_b_op), acc=out), out_mma
            )
            tlx.buffer_store(
                out.to(Output.dtype.element_ty),
                Output + value_base + token0 * H * V,
                out_offsets,
                mask=out_offsets < (length - token0) * H * V,
            )

        decay = tlx.local_load(tlx.local_view(decay_buffers, slot), token=ready)
        state *= tlx.require_layout(tl.exp(decay), key_layout, pin=False)[None, :]
        kg0 = tlx.local_load(
            tlx.local_slice(kg_tile, [0, 0], [TH, K]), token=ready, layout=b_op
        )
        kg1 = tlx.local_load(
            tlx.local_slice(kg_tile, [TH, 0], [TH, K]), token=ready, layout=b_op
        )
        state = tl.dot(tlx.require_layout(v0, a_op), kg0, acc=state)
        state = tlx.require_layout(
            tl.dot(tlx.require_layout(v1, a_op), kg1, acc=state), mma
        )
    return state


@triton.jit
def _state_walk_kernel(
    W,
    U,
    Kg,
    Qg,
    Decay,
    Aqk,
    InitialState,
    Output,
    FinalState,
    CuSeqLens,
    StateIndices,
    state_stride,
    GroupEntry,
    num_groups,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    NW: tl.constexpr,
    FROM_GROUP_ENTRY: tl.constexpr,
    PAGED: tl.constexpr,
):
    """Walk one (row slice, sequence, head, time group) of the recurrence and write its output.

    Unsplit, one group walks the whole sequence from the initial state. In a
    time split, each group starts from its entry state in GroupEntry, which
    _group_summary_kernel and _chain_groups_kernel produce. Entry 0 is a copy
    of the initial state, so a split walk never reads InitialState and
    FinalState may alias it. The last group writes the final state.
    """
    value_block = tl.program_id(0)
    sequence_head = tl.program_id(1)
    group = tl.program_id(2)
    sequence = sequence_head // H
    head = sequence_head % H
    begin = tl.load(CuSeqLens + sequence).to(tl.int32)
    length = tl.load(CuSeqLens + sequence + 1).to(tl.int32) - begin
    chunk_lo, chunk_hi = _group_chunks(tl.cdiv(length, BT), num_groups, group)
    state_base = _state_base(StateIndices, state_stride, sequence, head, H, K, V, PAGED)
    if FROM_GROUP_ENTRY:
        start = GroupEntry + (group * tl.num_programs(1) + sequence_head).to(
            tl.int64
        ) * (V * K)
    else:
        start = InitialState + state_base
    state, state_offsets, state_mask, _ = _load_state_rows(
        start, value_block, K, V, BV, NW
    )
    state = _walk_chunks(
        state,
        W,
        Kg,
        Qg,
        U,
        Aqk,
        Decay,
        Output,
        begin,
        length,
        sequence,
        head,
        chunk_lo,
        chunk_hi,
        value_block,
        H,
        K,
        V,
        BT,
        BV,
        NW,
        True,
    )
    if group == num_groups - 1:
        tlx.buffer_store(state, FinalState + state_base, state_offsets, mask=state_mask)


@triton.jit
def _group_summary_kernel(
    W,
    U,
    Kg,
    Decay,
    InitialState,
    CuSeqLens,
    StateIndices,
    state_stride,
    GroupEntry,
    GroupB,
    GroupM,
    num_groups,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    NW: tl.constexpr,
    PAGED: tl.constexpr,
):
    """Summarize one (row slice, sequence, head, time group) of a split walk as S_out = S_in M + B.

    A chunk maps the state affinely, S' = S (D - W^T Kg) + U^T Kg, so a run of
    chunks composes to S_out = S_in M + B. The value rows, walked from zero,
    give B; the K transfer rows past V start as the identity and, with no U,
    give M. Group 0 knows its entry, the initial state: it copies it to
    GroupEntry and walks it, without transfer rows, to group 1's entry. The
    last group needs no summary and is not launched.
    """
    value_block = tl.program_id(0)
    sequence_head = tl.program_id(1)
    group = tl.program_id(2)
    sequence = sequence_head // H
    head = sequence_head % H
    begin = tl.load(CuSeqLens + sequence).to(tl.int32)
    length = tl.load(CuSeqLens + sequence + 1).to(tl.int32) - begin
    # Group 0 needs only value rows.
    if value_block * BV >= V and group == 0:
        return
    chunk_lo, chunk_hi = _group_chunks(tl.cdiv(length, BT), num_groups, group)
    group_plane = (group * tl.num_programs(1) + sequence_head).to(tl.int64)
    state_base = _state_base(StateIndices, state_stride, sequence, head, H, K, V, PAGED)
    state, state_offsets, state_mask, eye = _load_state_rows(
        InitialState + state_base, value_block, K, V, BV, NW
    )
    if group == 0:
        tlx.buffer_store(
            state, GroupEntry + group_plane * (V * K), state_offsets, mask=state_mask
        )
    # Transfer rows load as zero and eye is zero on value rows.
    state = state * tl.where(group == 0, 1.0, 0.0) + eye
    # No output: W stands in for the unused Qg, Aqk and Output.
    state = _walk_chunks(
        state,
        W,
        Kg,
        W,
        U,
        W,
        Decay,
        W,
        begin,
        length,
        sequence,
        head,
        chunk_lo,
        chunk_hi,
        value_block,
        H,
        K,
        V,
        BT,
        BV,
        NW,
        False,
    )
    if group == 0:
        dest = GroupEntry + (tl.num_programs(1) + group_plane) * (V * K)
    else:
        dest = GroupB + group_plane * (V * K)
    tlx.buffer_store(state, dest, state_offsets, mask=state_mask)
    if group != 0:
        tlx.buffer_store(
            state.to(GroupM.dtype.element_ty),
            GroupM + group_plane * (K * K),
            state_offsets - V * K,
            mask=state_offsets >= V * K,
        )


@triton.jit
def _state_base(
    StateIndices,
    state_stride,
    sequence,
    head,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    PAGED: tl.constexpr,
):
    """Offset of a sequence's [V, K] state for one head: its pool slot when PAGED."""
    if PAGED:
        return tl.load(StateIndices + sequence).to(tl.int64) * state_stride + head * (
            V * K
        )
    return (sequence * H + head).to(tl.int64) * (V * K)


@triton.jit
def _chain_groups_kernel(
    GroupEntry,
    GroupB,
    GroupM,
    num_groups,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    """Chain the time groups' summaries into entry states: entry[g + 1] = entry[g] M_g + B_g."""
    value_block = tl.program_id(0)
    sequence_head = tl.program_id(1)
    planes = tl.num_programs(1)
    values = value_block * BV + tl.arange(0, BV)
    keys = tl.arange(0, K)
    state_offsets = values[:, None] * K + keys[None, :]
    m_offsets = keys[:, None] * K + keys[None, :]
    state = tl.load(
        GroupEntry + (planes + sequence_head).to(tl.int64) * (V * K) + state_offsets
    )
    for group in tl.range(1, num_groups - 1, num_stages=3):
        plane = (group * planes + sequence_head).to(tl.int64)
        transfer = tl.load(GroupM + plane * (K * K) + m_offsets)
        offset = tl.load(GroupB + plane * (V * K) + state_offsets)
        state = tl.dot(
            state.to(tl.bfloat16), transfer, acc=offset, out_dtype=tl.float32
        )
        tl.store(GroupEntry + (plane + planes) * (V * K) + state_offsets, state)


def _scan_groups(
    num_sequences: int, heads: int, num_chunks: int, block_value: int, value_dim: int
) -> int:
    """Time groups per sequence, so short batches of long sequences fill the GPU.

    The groups' walk programs, ``block_value`` value rows each, fill one wave of CUs.
    """
    chunks_per_sequence = num_chunks // max(num_sequences, 1)
    planes = num_sequences * heads * (value_dim // block_value)
    cus = _num_cus(torch.cuda.current_device())
    groups = min(cus // planes, chunks_per_sequence // 4, 32)
    # Short sequences whose unsplit walk already fills the GPU skip the split.
    unsplit_fills = num_sequences * heads * (value_dim // 32) >= 3 * cus // 4
    if (
        chunks_per_sequence < 24
        or groups < 4
        or (unsplit_fills and chunks_per_sequence < 48)
    ):
        return 1
    return groups


@functools.cache
def _num_cus(device: int) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _launch_state_scan(t, num_sequences, heads, num_chunks, key_dim, value_dim):
    """Walk the chunk recurrence into t["output"] and t["final_state"].

    Splits sequences in time when too few would idle the GPU: every group is
    summarized at once, the summaries are chained into each group's entry
    state, and every group is then walked at once from its entry.
    """
    initial_state = t["initial_state"]
    planes = num_sequences * heads
    cus = _num_cus(torch.cuda.current_device())
    # When the split runs out of chunks before CUs, the split walk takes
    # half-width value slices, which walk a chunk faster.
    chunk_limited = num_chunks // max(num_sequences, 1) // 4 < cus // planes
    split_block = value_dim // 2 if chunk_limited else value_dim
    groups = _scan_groups(num_sequences, heads, num_chunks, split_block, value_dim)
    common = dict(
        H=heads, K=key_dim, V=value_dim, BT=CHUNK_SIZE, num_stages=2, waves_per_eu=1
    )

    paged = t.get("state_indices") is not None
    state_args = [
        t["state_indices"] if paged else t["cu_seqlens"],
        initial_state.stride(0),
    ]
    walk_args = [
        t[n]
        for n in (
            "w",
            "u",
            "gated_key",
            "gated_query",
            "decay",
            "aqk",
            "initial_state",
            "output",
            "final_state",
            "cu_seqlens",
        )
    ] + state_args

    def walk(block, warps, group_entry):
        grid = (value_dim // block, planes, groups)
        _state_walk_kernel[grid](
            *walk_args,
            group_entry,
            groups,
            BV=block,
            NW=warps,
            FROM_GROUP_ENTRY=groups > 1,
            PAGED=paged,
            num_warps=warps,
            **common,
        )

    if groups == 1:
        # An unsplit walk takes the narrowest value slice that still keeps
        # every program on its own CU, with a warp per 16 value rows.
        block_value = value_dim
        for width in (32, 64):
            if planes * (value_dim // width) <= cus:
                block_value = width
                break
        walk(block_value, max(block_value // 16, 2), initial_state)
        return
    group_entry = torch.empty(
        (groups, planes, value_dim, key_dim),
        dtype=torch.float32,
        device=initial_state.device,
    )
    group_b = torch.empty_like(group_entry)
    group_m = torch.empty(
        (groups, planes, key_dim, key_dim),
        dtype=torch.bfloat16,
        device=initial_state.device,
    )
    # A summary walks the value rows and the transfer rows. It halves its row
    # block while that still fits on the CUs.
    summary_block = value_dim + key_dim
    if 2 * planes * (groups - 1) <= cus:
        summary_block //= 2
    summary_args = [
        t[n] for n in ("w", "u", "gated_key", "decay", "initial_state", "cu_seqlens")
    ] + state_args
    grid = ((value_dim + key_dim) // summary_block, planes, groups - 1)
    _group_summary_kernel[grid](
        *summary_args,
        group_entry,
        group_b,
        group_m,
        groups,
        BV=summary_block,
        NW=4,
        PAGED=paged,
        num_warps=4,
        **common,
    )
    if groups > 2:
        _chain_groups_kernel[(value_dim // 16, planes)](
            group_entry,
            group_b,
            group_m,
            groups,
            K=key_dim,
            V=value_dim,
            BV=16,
            num_warps=4,
        )
    walk(split_block, 8 if split_block == value_dim else 4, group_entry)


def _validate_prefill_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    use_qk_l2norm_and_gate: bool = False,
    state_indices: torch.Tensor | None = None,
) -> tuple[int, int, int, int]:
    tensors = (q, k, v, g, beta, initial_state)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("gfx950 TLX KDA prefill requires GPU tensors")
    if any(tensor.device != q.device for tensor in tensors[1:]):
        raise ValueError("KDA data tensors must be on the same GPU")
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("q must be a packed [1, T, H, K] tensor")
    if use_qk_l2norm_and_gate:
        if q.shape != k.shape or g.numel() != q.numel():
            raise ValueError(
                "q and k must have identical shapes and g as many elements"
            )
        if beta.numel() != q[..., 0].numel():
            raise ValueError("beta must have T * H elements")
    elif q.shape != k.shape or q.shape != g.shape:
        raise ValueError("q, k, and g must have identical shapes")
    elif beta.shape != q.shape[:-1]:
        raise ValueError("beta must have shape [1, T, H]")
    if v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError("v must match q through the head dimension")
    _, total_tokens, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if (key_dim, value_dim) != (_KEY_DIM, _VALUE_DIM):
        raise ValueError("gfx950 TLX KDA prefill specializes K=V=128")
    if q.dtype != torch.bfloat16 or any(t.dtype != q.dtype for t in (k, v)):
        raise ValueError("q, k, and v must be BF16")
    if not g.dtype.is_floating_point or not beta.dtype.is_floating_point:
        raise ValueError("g and beta must use floating-point dtypes")
    if initial_state.dtype != torch.float32:
        raise ValueError("initial_state must be FP32")
    if initial_state.ndim != 4 or initial_state.shape[1:] != (
        heads,
        value_dim,
        key_dim,
    ):
        raise ValueError("initial_state must have V-major shape [N, H, V, K]")
    if initial_state.stride()[-2:] != (key_dim, 1):
        raise ValueError("initial_state inner [V, K] dimensions must be contiguous")
    if state_indices is not None:
        if initial_state.stride(1) != value_dim * key_dim:
            raise ValueError("state_pool must be dense within each [H, V, K] slot")
        if (
            state_indices.ndim != 1
            or cu_seqlens.ndim != 1
            or cu_seqlens.numel() != state_indices.numel() + 1
        ):
            raise ValueError(
                "cu_seqlens and state_indices must describe the same batch"
            )
    elif cu_seqlens.ndim != 1 or cu_seqlens.numel() != initial_state.shape[0] + 1:
        raise ValueError("cu_seqlens and initial_state must describe the same batch")
    return total_tokens, heads, key_dim, value_dim


def kda_paged_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    beta_is_raw: bool = False,
    state_pool: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    cu_seqlens_cpu: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run KDA prefill and return output plus final state.

    By default the inputs are prepared: l2-normalized q/k, the per-channel log
    decay g and an activated beta. Passing ``A_log`` and ``dt_bias`` takes raw
    inputs instead, as ``prepare_kda_inputs`` does: q/k are normalized in the
    kernel and g is the gate pre-activation, mapped to
    ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``, or to
    ``-exp(A_log) * softplus(g + dt_bias)`` without a lower bound. Either way
    ``beta_is_raw`` applies the sigmoid to beta. q, k, v and g may be strided
    over tokens (split views of a packed qkv); each head vector must be dense.

    The state is either a dense ``initial_state`` [N, H, V, K], with the final
    state returned as a new tensor, or a ``state_pool`` [slots, H, V, K] with
    ``state_indices`` [N] naming each sequence's slot. A pool is updated in
    place and returned as the final state.

    ``cu_seqlens_cpu``, the values of ``cu_seqlens`` on the host, lets the
    chunk schedule be built without synchronizing the device.
    """
    use_qk_l2norm_and_gate = A_log is not None
    if use_qk_l2norm_and_gate and dt_bias is None:
        raise ValueError("raw inputs need both A_log and dt_bias")
    paged = state_pool is not None
    if paged == (initial_state is not None) or paged != (state_indices is not None):
        raise ValueError("pass either initial_state, or state_pool with state_indices")
    if paged:
        initial_state = state_pool
    total_tokens, heads, key_dim, value_dim = _validate_prefill_inputs(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
        use_qk_l2norm_and_gate,
        state_indices,
    )
    q, k, v, g = (_token_rows(x, total_tokens, heads) for x in (q, k, v, g))
    beta = _token_rows(beta, total_tokens, heads)
    if not paged:
        initial_state = initial_state.contiguous()
    cu_seqlens, chunk_indices = _prepare_prefill_metadata(
        cu_seqlens, q.device, total_tokens, cu_seqlens_cpu
    )
    num_chunks = chunk_indices.shape[0]
    num_sequences = cu_seqlens.numel() - 1
    if num_chunks == 0:
        output = torch.empty(
            (total_tokens, heads, value_dim), dtype=v.dtype, device=v.device
        )
        return output.unsqueeze(0), state_pool if paged else initial_state.clone()

    activation = None
    if use_qk_l2norm_and_gate:
        activation = dict(
            A_log=A_log.reshape(-1).float(),
            dt_bias=dt_bias.reshape(-1).float(),
            lower_bound=lower_bound,
        )
    chunks = _prepare_chunks(
        q,
        k,
        v,
        g,
        beta,
        scale,
        cu_seqlens,
        chunk_indices,
        num_sequences,
        activation,
        beta_is_raw,
    )
    output = torch.empty(
        (total_tokens, heads, value_dim), dtype=v.dtype, device=v.device
    )
    final_state = state_pool if paged else torch.empty_like(initial_state)
    _launch_state_scan(
        dict(
            **chunks,
            initial_state=initial_state,
            output=output,
            final_state=final_state,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices.contiguous() if paged else None,
        ),
        num_sequences,
        heads,
        num_chunks,
        key_dim,
        value_dim,
    )
    return output.unsqueeze(0), final_state


def _token_rows(x, total_tokens, heads):
    """A [T, H, D] (or [T, H]) view of x that keeps its token stride; copies only if a head is not dense."""
    rows = x.reshape(total_tokens, heads, -1)
    if rows.stride(2) != 1 or rows.stride(1) != rows.shape[2]:
        rows = rows.contiguous()
    return rows if x.numel() != total_tokens * heads else rows[..., 0]


def _decay_table(total_tokens, num_sequences, heads, key_dim, device):
    return torch.empty(
        (total_tokens // CHUNK_SIZE + 2 * num_sequences + 1, heads, key_dim),
        dtype=torch.float32,
        device=device,
    )


def _prepare_chunks(
    q,
    k,
    v,
    g,
    beta,
    scale,
    cu_seqlens,
    chunk_indices,
    num_sequences,
    activation=None,
    beta_is_raw=False,
):
    total_tokens, heads, key_dim = q.shape
    dense = dict(dtype=q.dtype, device=q.device)
    chunks = dict(
        gated_query=torch.empty((total_tokens, heads, key_dim), **dense),
        w=torch.empty((total_tokens, heads, key_dim), **dense),
        u=torch.empty((total_tokens, heads, v.shape[2]), **dense),
        gated_key=torch.empty((total_tokens, heads, key_dim), **dense),
        aqk=torch.empty(
            (total_tokens, heads, CHUNK_SIZE), dtype=torch.bfloat16, device=q.device
        ),
        decay=_decay_table(total_tokens, num_sequences, heads, key_dim, q.device),
    )
    _prepare_chunk_kernel[(chunk_indices.shape[0], heads)](
        q,
        k,
        v,
        g,
        beta,
        chunks["gated_query"],
        chunks["w"],
        chunks["u"],
        chunks["gated_key"],
        chunks["aqk"],
        chunks["decay"],
        cu_seqlens,
        chunk_indices,
        activation["A_log"] if activation else beta,
        activation["dt_bias"] if activation else beta,
        (activation or {}).get("lower_bound") or 0.0,
        SCALE=scale,
        H=heads,
        D=key_dim,
        BT=CHUNK_SIZE,
        BC=SUBCHUNK_SIZE,
        NW=4,
        STRIDE_Q=q.stride(0),
        STRIDE_K=k.stride(0),
        STRIDE_V=v.stride(0),
        STRIDE_G=g.stride(0),
        STRIDE_BETA=beta.stride(0),
        USE_QK_L2NORM_AND_GATE=activation is not None,
        USE_LOWER_BOUND=activation is not None
        and activation["lower_bound"] is not None,
        USE_BETA_SIGMOID=beta_is_raw,
        num_warps=4,
        waves_per_eu=2,
    )
    return chunks


__all__ = [
    "CHUNK_SIZE",
    "SUBCHUNK_SIZE",
    "kda_paged_prefill",
    "prepare_chunk_indices",
]
