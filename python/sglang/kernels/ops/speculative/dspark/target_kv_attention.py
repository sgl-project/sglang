"""Noncausal KV-draft attention in logical token order, without gathering KV."""

import torch
import triton
import triton.language as tl


@triton.jit
def _target_kv_attention(
    Q,
    K,
    V,
    QO,
    KI,
    INDEX,
    EXTEND_INDEX,
    OUT,
    Q_STRIDE: tl.constexpr,
    Q_HEAD_STRIDE: tl.constexpr,
    K_STRIDE: tl.constexpr,
    K_HEAD_STRIDE: tl.constexpr,
    V_STRIDE: tl.constexpr,
    V_HEAD_STRIDE: tl.constexpr,
    O_STRIDE: tl.constexpr,
    O_HEAD_STRIDE: tl.constexpr,
    GROUP: tl.constexpr,
    DIM: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROWS_PER_HEAD: tl.constexpr,
    HAS_EXTEND: tl.constexpr,
):
    batch, kv_head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    q_start = tl.load(QO + batch)
    q_len = tl.load(QO + batch + 1) - q_start
    kv_start = tl.load(KI + batch)
    prefix_len = tl.load(KI + batch + 1) - kv_start
    kv_len = prefix_len + q_len if HAS_EXTEND else prefix_len

    rows = tl.arange(0, BLOCK_M)
    local_heads = rows // ROWS_PER_HEAD
    heads = kv_head * GROUP + local_heads
    queries = block * ROWS_PER_HEAD + rows % ROWS_PER_HEAD
    dims = tl.arange(0, BLOCK_D)
    valid_q = (local_heads < GROUP) & (queries < q_len)
    q = tl.load(
        Q
        + (q_start + queries[:, None]) * Q_STRIDE
        + heads[:, None] * Q_HEAD_STRIDE
        + dims[None, :],
        mask=valid_q[:, None] & (dims[None, :] < DIM),
        other=0,
    )
    maximum = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    for token_start in range(0, kv_len, BLOCK_N):
        tokens = token_start + tl.arange(0, BLOCK_N)
        slots = tl.load(INDEX + kv_start + tokens, mask=tokens < prefix_len, other=0)
        if HAS_EXTEND:
            extend_slots = tl.load(
                EXTEND_INDEX + q_start + tokens - prefix_len,
                mask=(tokens >= prefix_len) & (tokens < kv_len),
                other=0,
            )
            slots = tl.where(tokens < prefix_len, slots, extend_slots)
        slots = slots.to(tl.int64)
        k = tl.load(
            K + slots[None, :] * K_STRIDE + kv_head * K_HEAD_STRIDE + dims[:, None],
            mask=(tokens[None, :] < kv_len) & (dims[:, None] < DIM),
            other=0,
        )
        scores = tl.dot(q, k, input_precision="ieee") * SCALE
        scores = tl.where(
            valid_q[:, None] & (tokens[None, :] < kv_len), scores, float("-inf")
        )
        scores *= 1.44269504
        updated_max = tl.maximum(maximum, tl.max(scores, axis=1))
        safe_max = tl.where(updated_max == float("-inf"), 0.0, updated_max)
        alpha = tl.exp2(maximum - safe_max)
        probability = tl.exp2(scores - safe_max[:, None])
        denominator = denominator * alpha + tl.sum(probability, axis=1)
        v = tl.load(
            V + slots[:, None] * V_STRIDE + kv_head * V_HEAD_STRIDE + dims[None, :],
            mask=(tokens[:, None] < kv_len) & (dims[None, :] < DIM),
            other=0,
        )
        acc = tl.dot(
            probability.to(v.dtype), v, acc * alpha[:, None], input_precision="ieee"
        )
        maximum = updated_max

    denominator = tl.where(denominator == 0, 1.0, denominator)
    tl.store(
        OUT
        + (q_start + queries[:, None]) * O_STRIDE
        + heads[:, None] * O_HEAD_STRIDE
        + dims[None, :],
        acc / denominator[:, None],
        mask=valid_q[:, None] & (dims[None, :] < DIM),
    )


def target_kv_attention(
    q,
    k,
    v,
    out,
    qo_indptr,
    kv_indptr,
    kv_indices,
    *,
    max_query,
    scale,
    extend_indices=None,
):
    """Write unmasked block attention; indptr values and slots are backend-owned.

    Reads NHD pools using token indices, independent of physical page boundaries.
    When extend_indices is supplied, kv_indptr/kv_indices cover only the prefix;
    the current block's slots follow qo_indptr in extend_indices.
    All shape checks use host metadata and are safe during CUDA graph capture.
    """
    values = (q, k, v, out)
    if any(value.ndim != 3 or value.stride(-1) != 1 for value in values):
        raise ValueError("KV draft attention requires NHD tensors with contiguous D")
    if (
        q.device.type != "cuda"
        or q.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or any(value.device != q.device or value.dtype != q.dtype for value in values)
        or out.shape != q.shape
        or k.shape != v.shape
        or k.shape[2] != q.shape[2]
        or k.shape[1] == 0
        or q.shape[1] % k.shape[1]
        or not 1 <= max_query <= 64
        or not 16 <= q.shape[2] <= 256
    ):
        raise ValueError("unsupported KV draft attention geometry, dtype or device")
    indices = (qo_indptr, kv_indptr, kv_indices)
    if extend_indices is not None:
        indices += (extend_indices,)
    if (
        any(
            value.ndim != 1
            or not value.is_contiguous()
            or value.device != q.device
            or value.dtype not in (torch.int32, torch.int64)
            for value in indices
        )
        or qo_indptr.shape != kv_indptr.shape
        or qo_indptr.numel() < 1
        or (extend_indices is not None and extend_indices.numel() < q.shape[0])
    ):
        raise ValueError("KV draft attention requires matching CUDA index vectors")
    heads, dim = q.shape[1:]
    group = heads // k.shape[1]
    if not 1 <= group <= 64:
        raise ValueError("KV draft attention requires 1..64 query heads per KV head")
    padded_group = triton.next_power_of_2(group)
    block_m = max(
        16, padded_group, min(64, padded_group * triton.next_power_of_2(max_query))
    )
    rows_per_head = block_m // padded_group
    batch = qo_indptr.numel() - 1
    if batch == 0 or q.numel() == 0:
        return out
    # Keep logical 64-token tiles, exp2 softmax and the narrow P cast in the
    # training reference's order. Splitting prefix/block or KV changes rounding.
    _target_kv_attention[(batch, k.shape[1], triton.cdiv(max_query, rows_per_head))](
        q,
        k,
        v,
        qo_indptr,
        kv_indptr,
        kv_indices,
        extend_indices,
        out,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        out.stride(0),
        out.stride(1),
        GROUP=group,
        DIM=dim,
        SCALE=scale,
        BLOCK_M=block_m,
        BLOCK_N=64,
        BLOCK_D=triton.next_power_of_2(dim),
        ROWS_PER_HEAD=rows_per_head,
        HAS_EXTEND=extend_indices is not None,
        num_warps=2,
        num_stages=1,
    )
    return out
