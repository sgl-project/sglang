# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
# Adapted from Tencent/DSpine and vLLM PR #63704 (Apache-2.0).

import triton
import triton.language as tl


@triton.jit
def _prepare_read_kernel(
    X,
    Residual,
    Raw,
    Normalized,
    RMS,
    H: tl.constexpr,
    EPS: tl.constexpr,
    ADD: tl.constexpr,
    WIDTH: tl.constexpr,
):
    """Save the residual stream and its per-row RMS before a code-bridge write.

    Raw is bf16(X + Residual) when ADD is set, otherwise X. RMS is
    sqrt(max(mean(Raw**2), fp32 tiny)); Normalized is Raw / (RMS + EPS).
    """
    row = tl.program_id(0)
    h = tl.arange(0, WIDTH)
    x = tl.load(X + row * H + h, h < H, 0).to(tl.float32)
    if ADD:
        residual = tl.load(Residual + row * H + h, h < H, 0).to(tl.float32)
        # The original layer first rounds residual+MLP to hidden dtype.
        x = (x + residual).to(X.dtype.element_ty).to(tl.float32)
    rms = tl.sqrt(tl.maximum(tl.sum(x * x, 0) / H, 1.1754943508222875e-38))
    tl.store(Raw + row * H + h, x, h < H)
    tl.store(Normalized + row * H + h, x / (rms + EPS), h < H)
    tl.store(RMS + row, rms)


@triton.jit
def _normalize_codes_kernel(
    Projected,
    Codebook,
    Anchors,
    Codes,
    C: tl.constexpr,
    T: tl.constexpr,
    ANCHOR_ROW: tl.constexpr,
    EPS: tl.constexpr,
    WIDTH: tl.constexpr,
):
    """Scale codes to sqrt(C), using the token codebook on anchor rows.

    When ANCHOR_ROW is set, anchors use sqrt(C) * Codebook[token]. Other rows
    use sqrt(C) * Projected / (norm(Projected) + EPS), from the layer's read
    projection.
    """
    row = tl.program_id(0)
    k = tl.arange(0, WIDTH)
    if ANCHOR_ROW and row % T == 0:
        token = tl.load(Anchors + row // T)
        z = tl.load(Codebook + token * C + k, k < C, 0).to(tl.float32)
        z = z * tl.sqrt(float(C))
    else:
        z = tl.load(Projected + row * C + k, k < C, 0).to(tl.float32)
        norm = tl.sqrt(tl.sum(z * z, 0))
        z = (tl.sqrt(float(C)) * z) / (norm + EPS)
    tl.store(Codes + row * C + k, z, k < C)


@triton.jit
def _feature_message_kernel(
    Packed, Bias, Message, R: tl.constexpr, T: tl.constexpr, WIDTH: tl.constexpr
):
    """Compute sigmoid(Q[r] + S[r-1] + Bias) * V[r-1], zero on anchor rows.

    Packed contains the code projections [Q | S | V], each R columns wide.
    """
    row = tl.program_id(0)
    k = tl.arange(0, WIDTH)
    query = tl.load(Packed + row * (3 * R) + k, k < R, 0).to(tl.float32)
    prev = tl.maximum(row - 1, 0)
    sender = tl.load(Packed + prev * (3 * R) + R + k, k < R, 0).to(tl.float32)
    value = tl.load(Packed + prev * (3 * R) + 2 * R + k, k < R, 0).to(tl.float32)
    bias = tl.load(Bias + k, k < R, 0).to(tl.float32)
    gate = tl.sigmoid((query + sender) + bias)
    message = tl.where(row % T == 0, 0.0, gate * value)
    tl.store(Message + row * R + k, message, k < R)


@triton.jit
def _feature_finish_kernel(
    Raw,
    Delta,
    RMS,
    Valid,
    Weight,
    Output,
    Normed,
    H: tl.constexpr,
    T: tl.constexpr,
    WRITE: tl.constexpr,
    NORM: tl.constexpr,
    NORM_EPS: tl.constexpr,
    WIDTH: tl.constexpr,
):
    """Write bf16(Raw + RMS * Delta) on valid non-anchor edges when WRITE is set.

    Delta is the projected message. Other rows retain Raw. NORM additionally
    applies the supplied RMSNorm, preserving Qwen3's cast-before-scale order.
    """
    row = tl.program_id(0)
    k = tl.arange(0, WIDTH)
    raw = tl.load(Raw + row * H + k, k < H, 0).to(tl.float32)
    delta = tl.load(Delta + row * H + k, k < H, 0).to(tl.float32)
    rms = tl.load(RMS + row)
    previous = tl.maximum(row - 1, 0)
    edge = (row % T != 0) & tl.load(Valid + row) & tl.load(Valid + previous)
    update = (rms * delta) * tl.cast(edge & WRITE, tl.float32)
    result = (raw + update).to(Output.dtype.element_ty)
    tl.store(Output + row * H + k, result, k < H)
    if NORM:
        x = result.to(tl.float32)
        variance = tl.sum(x * x, 0) / H
        y = (x * tl.rsqrt(variance + NORM_EPS)).to(Output.dtype.element_ty)
        weight = tl.load(Weight + k, k < H, 0)
        # Qwen3RMSNorm casts the normalized activation before affine multiply.
        y = (y.to(tl.float32) * weight.to(tl.float32)).to(Output.dtype.element_ty)
        tl.store(Normed + row * H + k, y, k < H)


@triton.jit
def _repair_transition_scores_kernel(
    X,
    Scale,
    Message,
    Write,
    Norm,
    Head,
    Mask,
    Scores,
    S: tl.constexpr,
    K: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    XB: tl.constexpr,
    XS: tl.constexpr,
    XH: tl.constexpr,
    SB: tl.constexpr,
    SS: tl.constexpr,
    EPS: tl.constexpr,
    WR: tl.constexpr,
    WK: tl.constexpr,
    TILE_H: tl.constexpr,
):
    """Recompute cancellation-prone rows from explicit FP32 residuals."""
    slot = tl.program_id(0)
    lanes = tl.arange(0, WK)
    needs_repair = tl.load(Mask + slot * K + lanes, lanes < K, 0)
    if tl.sum(needs_repair.to(tl.int32), axis=0) > 0:
        for sender in range(K):
            row = slot * K + sender
            if tl.load(Mask + row):
                batch = slot // S
                step = slot % S
                rank = tl.arange(0, WR)
                candidate = tl.arange(0, WK)
                message = tl.load(Message + row * R + rank, rank < R, 0)
                scale = tl.load(Scale + batch * SB + step * SS)
                square_sum = tl.full((), 0, tl.float32)
                logits = tl.full((WK,), 0, tl.float32)
                for start in range(tl.cdiv(H, TILE_H)):
                    hidden = start * TILE_H + tl.arange(0, TILE_H)
                    weights = tl.load(
                        Write + hidden[None, :] * R + rank[:, None],
                        (hidden[None, :] < H) & (rank[:, None] < R),
                        0,
                    )
                    delta = tl.sum(weights * message[:, None], axis=0)
                    raw = tl.load(
                        X + batch * XB + step * XS + hidden * XH, hidden < H, 0
                    )
                    refined = raw + scale * delta
                    square_sum += tl.sum(refined * refined, axis=0)
                    norm = tl.load(Norm + hidden, hidden < H, 0)
                    head = tl.load(
                        Head + slot * K * H + candidate[:, None] * H + hidden[None, :],
                        (candidate[:, None] < K) & (hidden[None, :] < H),
                        0,
                    )
                    logits += tl.sum(head * (refined * norm)[None, :], axis=1)
                logits *= tl.rsqrt(square_sum / H + EPS)
                tl.store(Scores + row * K + candidate, logits, candidate < K)
