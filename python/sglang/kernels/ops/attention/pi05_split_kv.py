# SPDX-License-Identifier: Apache-2.0

"""Noncausal split-KV attention without concatenating or repeating K/V."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _partial_attention(
        Q,
        PK,
        PV,
        SK,
        SV,
        Mask,
        Part,
        Lse,
        qb: tl.constexpr,
        qh: tl.constexpr,
        qs: tl.constexpr,
        pkb: tl.constexpr,
        pkh: tl.constexpr,
        pks: tl.constexpr,
        pvb: tl.constexpr,
        pvh: tl.constexpr,
        pvs: tl.constexpr,
        skb: tl.constexpr,
        skh: tl.constexpr,
        sks: tl.constexpr,
        svb: tl.constexpr,
        svh: tl.constexpr,
        svs: tl.constexpr,
        mb: tl.constexpr,
        mn: tl.constexpr,
        P: tl.constexpr,
        S: tl.constexpr,
        H: tl.constexpr,
        HK: tl.constexpr,
        D: tl.constexpr,
        SCALE: tl.constexpr,
        MASKED: tl.constexpr,
        SPLITS: tl.constexpr,
        PART_LENGTH: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
    ):
        bh = tl.program_id(0)
        b, head = bh // H, bh % H
        kv_head = head // (H // HK)
        partition = tl.program_id(2)
        rows = tl.program_id(1) * BM + tl.arange(0, BM)
        dims = tl.arange(0, D)
        q = tl.load(
            Q + b * qb + head * qh + rows[:, None] * qs + dims[None, :],
            rows[:, None] < S,
            other=0,
        )
        maximum = tl.full((BM,), -float("inf"), tl.float32)
        denominator = tl.zeros((BM,), tl.float32)
        acc = tl.zeros((BM, D), tl.float32)
        for offset in range(PART_LENGTH // BN):
            columns = partition * PART_LENGTH + offset * BN + tl.arange(0, BN)
            prefix = columns < P
            suffix = (columns >= P) & (columns < P + S)
            kp = tl.load(
                PK + b * pkb + kv_head * pkh + columns[:, None] * pks + dims[None, :],
                prefix[:, None],
                other=0,
            )
            ks = tl.load(
                SK
                + b * skb
                + kv_head * skh
                + (columns[:, None] - P) * sks
                + dims[None, :],
                suffix[:, None],
                other=0,
            )
            k = tl.where(prefix[:, None], kp, ks)
            valid = columns < P + S
            if MASKED:
                bias = tl.load(Mask + b * mb + columns * mn, valid, other=-float("inf"))
                valid = valid & (bias == 0)
            scores = tl.dot(q, tl.trans(k)) * (SCALE * 1.4426950408889634)
            scores = tl.where(
                valid[None, :] & (rows[:, None] < S), scores, -float("inf")
            )
            next_max = tl.maximum(maximum, tl.max(scores, 1))
            # An entirely padded partition contributes zero mass, never NaNs.
            safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
            correction = tl.exp2(maximum - safe_max)
            prob = tl.exp2(scores - safe_max[:, None])
            vp = tl.load(
                PV + b * pvb + kv_head * pvh + columns[:, None] * pvs + dims[None, :],
                prefix[:, None],
                other=0,
            )
            vs = tl.load(
                SV
                + b * svb
                + kv_head * svh
                + (columns[:, None] - P) * svs
                + dims[None, :],
                suffix[:, None],
                other=0,
            )
            v = tl.where(prefix[:, None], vp, vs)
            acc = acc * correction[:, None] + tl.dot(prob.to(v.dtype), v)
            denominator = denominator * correction + tl.sum(prob, 1)
            maximum = next_max
        safe_denominator = tl.where(denominator > 0, denominator, 1.0)
        output = acc / safe_denominator[:, None]
        logsum = tl.where(
            denominator > 0, maximum + tl.log2(safe_denominator), -float("inf")
        )
        index = (bh * SPLITS + partition) * S + rows
        tl.store(Part + index[:, None] * D + dims[None, :], output, rows[:, None] < S)
        if SPLITS > 1:
            tl.store(Lse + index, logsum, rows < S)

    @triton.jit
    def _merge_attention(
        Part, Lse, Out, S: tl.constexpr, D: tl.constexpr, SPLITS: tl.constexpr
    ):
        row, bh = tl.program_id(0), tl.program_id(1)
        partitions = tl.arange(0, SPLITS)
        dims = tl.arange(0, D)
        indices = (bh * SPLITS + partitions) * S + row
        logsum = tl.load(Lse + indices)
        maximum = tl.max(logsum, 0)
        safe_max = tl.where(maximum == -float("inf"), 0.0, maximum)
        weight = tl.exp2(logsum - safe_max)
        partial = tl.load(Part + indices[:, None] * D + dims[None, :])
        mass = tl.sum(weight, 0)
        value = tl.sum(partial * weight[:, None], 0) / tl.where(mass > 0, mass, 1.0)
        tl.store(Out + (bh * S + row) * D + dims, value)


def split_kv_attention(
    query: torch.Tensor,
    prefix_key: torch.Tensor,
    prefix_value: torch.Tensor,
    suffix_key: torch.Tensor,
    suffix_value: torch.Tensor,
    mask: torch.Tensor | None,
    scaling: float,
    *,
    num_splits: int | None = None,
) -> torch.Tensor:
    """Attend to cached/current BHSD K/V, with one current KV token per query.
    Optional floating mask[B,1,1,P+S] uses 0 to keep keys and nonzero to mask;
    fully masked queries return zero.
    """
    if torch.is_grad_enabled() and any(
        t.requires_grad
        for t in (query, prefix_key, prefix_value, suffix_key, suffix_value)
    ):
        raise RuntimeError("Pi0.5 split-KV attention is inference-only")
    if triton is None or query.device.type != "cuda":
        raise RuntimeError("split-KV attention requires CUDA and Triton")
    if any(
        tensor.ndim != 4
        for tensor in (query, prefix_key, prefix_value, suffix_key, suffix_value)
    ):
        raise ValueError("split-KV attention tensors must have BHSD layout")
    batch, heads, size, width = query.shape
    if min(batch, heads, size, prefix_key.shape[1]) < 1:
        raise ValueError(
            "split-KV attention requires nonempty batch, heads and queries"
        )
    if query.dtype not in (torch.bfloat16, torch.float16) or width not in (
        64,
        128,
        256,
    ):
        raise ValueError(
            "split-KV attention requires FP16/BF16 and head width 64/128/256"
        )
    if prefix_key.shape != prefix_value.shape or suffix_key.shape != suffix_value.shape:
        raise ValueError("split-KV attention K and V shapes must match")
    if prefix_key.shape[:2] != suffix_key.shape[:2] or prefix_key.shape[0] != batch:
        raise ValueError("cached/current KV batch and KV-head counts must agree")
    if suffix_key.shape[2:] != (size, width) or prefix_key.shape[-1] != width:
        raise ValueError("split-KV attention expects one current KV token per query")
    if heads % prefix_key.shape[1]:
        raise ValueError("query heads must be a multiple of KV heads")
    for tensor in (query, prefix_key, prefix_value, suffix_key, suffix_value):
        if (
            tensor.device != query.device
            or tensor.dtype != query.dtype
            or tensor.stride(-1) != 1
        ):
            raise ValueError(
                "split-KV attention tensors must share device/dtype and contiguous head dimensions"
            )
    prefix = prefix_key.shape[2]
    if mask is not None and (
        mask.shape != (batch, 1, 1, prefix + size) or mask.device != query.device
    ):
        raise ValueError(
            "split-KV padding mask must be [batch, 1, 1, cached + current]"
        )
    if mask is not None and not mask.is_floating_point():
        raise TypeError("split-KV padding mask must use floating zero for valid keys")
    if num_splits is None:
        # Short action chunks need KV parallelism to fill the GPU.
        splits, block_m, block_n = (4, 16, 64) if size <= 64 else (1, 32, 64)
    else:
        if num_splits not in (1, 2, 4, 8, 16):
            raise ValueError("num_splits must be 1, 2, 4, 8 or 16")
        splits, block_m, block_n = num_splits, (16 if size <= 64 else 32), 64
    part_length = triton.cdiv(prefix + size, splits * block_n) * block_n
    with torch.cuda.device(query.device):
        partial = torch.empty(
            (batch, heads, splits, size, width),
            device=query.device,
            dtype=query.dtype if splits == 1 else torch.float32,
        )
        logsum = (
            torch.empty(
                (batch, heads, splits, size), device=query.device, dtype=torch.float32
            )
            if splits > 1
            else None
        )
        _partial_attention[(batch * heads, triton.cdiv(size, block_m), splits)](
            query,
            prefix_key,
            prefix_value,
            suffix_key,
            suffix_value,
            mask,
            partial,
            logsum,
            *query.stride()[:3],
            *prefix_key.stride()[:3],
            *prefix_value.stride()[:3],
            *suffix_key.stride()[:3],
            *suffix_value.stride()[:3],
            0 if mask is None else mask.stride(0),
            0 if mask is None else mask.stride(-1),
            prefix,
            size,
            heads,
            prefix_key.shape[1],
            width,
            scaling,
            mask is not None,
            splits,
            part_length,
            block_m,
            block_n,
            num_warps=4,
            num_stages=1,
        )
        if splits == 1:
            return partial.view(query.shape)
        output = torch.empty(query.shape, device=query.device, dtype=query.dtype)
        _merge_attention[(size, batch * heads)](
            partial, logsum, output, size, width, splits, num_warps=4
        )
    return output
