# SPDX-License-Identifier: Apache-2.0
"""Step5 indexer norm/partial-RoPE arithmetic; no state or allocation."""

import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_bits(value, SATURATE: tl.constexpr):
    f = value.to(tl.float32)
    bits = f.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    if not SATURATE:
        # 464 is the RNE midpoint between max finite 448 and the NaN encoding
        # at 480. Preserve the caller's probed Torch overflow policy.
        nan = tl.where(f < 0, 0xFF, 0x7F).to(tl.uint8)
        bits = tl.where(tl.abs(f) > 464.0, nan, bits)
    return bits


@triton.jit(
    do_not_specialize=("ROWS",),
    do_not_specialize_on_alignment=("ROWS",),
)
def _indexer_norm_rope(
    Q,
    K,
    QW,
    KW,
    KB,
    Cache,
    Positions,
    QOut,
    KOut,
    Requests,
    Weights,
    PackedW,
    ROWS,
    GROUP: tl.constexpr,
    PACKED: tl.constexpr,
    WEIGHT_SCALE: tl.constexpr,
    Q_ROW: tl.constexpr,
    K_ROW: tl.constexpr,
    HEADS: tl.constexpr,
    EPS: tl.constexpr,
    ZERO_CENTERED: tl.constexpr,
    WRITE_Q_E4M3: tl.constexpr,
    FP8_SATURATE: tl.constexpr,
    USE_PDL: tl.constexpr = False,
):
    if USE_PDL:
        # Finish the projections before the ring consumer starts reading Z.
        # Its independent history reads can overlap this normalization.
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    real_row = token < ROWS
    if PACKED:
        if GROUP == 1:
            group = token
            slot = 0
        else:
            group = token // 4
            slot = token % 4
        request = tl.load(Requests + token, real_row, other=0)
        live_q = real_row & (request > 0)
    channel = tl.arange(0, 256)
    if head < HEADS:
        x = tl.load(Q + token * Q_ROW + head * 256 + channel, real_row, other=0).to(
            tl.float32
        )
        w = tl.load(QW + channel).to(tl.float32)
        inverse_rms = tl.rsqrt(tl.sum(x * x, 0) / 256 + EPS)
        y = x * inverse_rms * (w + ZERO_CENTERED)
    else:
        x = tl.load(K + token * K_ROW + channel, real_row, other=0).to(tl.float32)
        w = tl.load(KW + channel).to(tl.float32)
        b = tl.load(KB + channel).to(tl.float32)
        mean = tl.sum(x, 0) / 256
        variance = tl.sum(x * x, 0) / 256 - mean * mean
        y = (x - mean) * tl.rsqrt(variance + EPS) * w + b

    # Normalization has its own BF16 boundary, including the rotated partner.
    normalized = y.to(tl.bfloat16).to(tl.float32)
    partner = tl.gather(normalized, channel ^ 16, axis=0)
    position = tl.load(Positions + token, real_row, other=0).to(tl.int64)
    cosine = tl.load(Cache + position * 32 + channel % 16, channel < 32, other=0).to(
        tl.float32
    )
    sine = tl.load(Cache + position * 32 + 16 + channel % 16, channel < 32, other=0).to(
        tl.float32
    )
    rotated = normalized * cosine + tl.where(channel < 16, -partner, partner) * sine
    result = tl.where(channel < 32, rotated, normalized).to(tl.bfloat16)
    if PACKED:
        if head < HEADS:
            # Match both existing cast paths: fused norm honors the probed Torch
            # policy; BF16 norm -> prepare_scoring uses Triton saturating conversion.
            bits = _fp8_bits(result, not WRITE_Q_E4M3 or FP8_SATURATE)
            bits = tl.where(live_q, bits, 0).to(tl.uint8)
            packed_h = group * 16 + (head // 2) * 8 + slot * 2 + head % 2
            tl.store(QOut + packed_h * 256 + channel, bits)
            weight = tl.load(Weights + token * HEADS + head, live_q, other=0).to(
                tl.float32
            )
            tl.store(PackedW + group * 128 + slot * 4 + head, weight * WEIGHT_SCALE)
            # Padding is part of the scoring ABI. Fill it from the live head
            # CTAs instead of launching norm/RoPE CTAs for nonexistent Q heads
            # and, in GROUP1 decode, three nonexistent token slots per row.
            if HEADS == 2:
                tl.store(QOut + (packed_h + 8) * 256 + channel, 0)
                tl.store(PackedW + group * 128 + slot * 4 + head + 2, 0)
            if GROUP == 1:
                for pad_slot in tl.static_range(1, 4):
                    tl.store(QOut + (packed_h + pad_slot * 2) * 256 + channel, 0)
                    tl.store(PackedW + group * 128 + pad_slot * 4 + head, 0)
                    if HEADS == 2:
                        tl.store(
                            QOut + (packed_h + 8 + pad_slot * 2) * 256 + channel,
                            0,
                        )
                        tl.store(PackedW + group * 128 + pad_slot * 4 + head + 2, 0)
    else:
        if head < HEADS:
            if WRITE_Q_E4M3:
                bits = _fp8_bits(result, FP8_SATURATE)
                tl.store(
                    QOut + (token * HEADS + head) * 256 + channel,
                    bits.to(tl.float8e4nv, bitcast=True),
                )
            else:
                tl.store(QOut + (token * HEADS + head) * 256 + channel, result)
    if head == HEADS:
        tl.store(KOut + token * 256 + channel, result, real_row)


def launch(
    q,
    k,
    qw,
    kw,
    kb,
    cache,
    positions,
    q_out,
    k_out,
    *,
    eps,
    zero_centered,
    q_e4m3=False,
    fp8_saturate=False,
    use_pdl=False,
    packed_workspace=None,
    weights=None,
    requests=None,
    weight_scale=1.0,
):
    packed = packed_workspace is not None
    group = packed_workspace.rows_per_group if packed else 1
    grid_rows = triton.cdiv(q.shape[0], 4) * 4 if packed and group == 4 else q.shape[0]
    _indexer_norm_rope[(grid_rows, q.shape[1] + 1)](
        q,
        k,
        qw,
        kw,
        kb,
        cache,
        positions,
        packed_workspace.q_runtime.view(torch.uint8) if packed else q_out,
        k_out,
        requests,
        weights,
        packed_workspace.kernel_weights if packed else None,
        q.shape[0],
        group,
        packed,
        weight_scale,
        q.stride(0),
        k.stride(0),
        q.shape[1],
        eps,
        int(zero_centered),
        int(q_e4m3),
        int(fp8_saturate),
        USE_PDL=use_pdl,
        launch_pdl=use_pdl,
        num_warps=4,
        enable_fp_fusion=False,
    )
