# SPDX-License-Identifier: Apache-2.0
"""SageAttention in Triton: INT8 Q @ K^T and FP16 P @ V, tuned for Intel XPU.

Implements the algorithm of SageAttention (Zhang et al., "SageAttention:
Accurate 8-Bit Attention for Plug-and-play Inference Acceleration", ICLR 2025,
arXiv:2410.02367). K is smoothed by subtracting its per-head mean over tokens,
which shifts every logit of a query row by the same amount and so leaves the
softmax unchanged. Q and K are then quantized to INT8 with one scale per block
of rows, Q @ K^T runs on INT8 and P @ V on FP16 under an online softmax.

Reference: thu-ml SpargeAttn, https://github.com/thu-ml/SpargeAttn (Apache-2.0),
whose Triton kernels implement the same algorithm for CUDA.

Implementation choices, measured on Arc Pro B60 (Xe2):

- Two memory-bound prepasses write head-major INT8 Q, INT8 K and FP16 V, so
  each attention tile is one contiguous 2D block. Reading V in the [B, S, H, D]
  input layout inside the KV loop is 2-3x slower.
- Every operand is read through a tensor descriptor and K is transposed inside
  the dot. A transposed block-pointer load of K, or pointer-arithmetic loads,
  spill registers and run at 3-19 TFLOPs instead of ~80.
- P @ V accumulates in FP32.
- The optional log-sum-exp is in natural log and adds back q . mean(k), the
  logit shift that smoothing removed, so ring attention can merge partials.
"""

from __future__ import annotations

import msgspec
import torch
import triton
import triton.language as tl

_LOG2E = 1.4426950408889634

# Rows that share one quantization scale (the paper's per-block granularity).
_Q_ROWS_PER_SCALE = 128
_K_ROWS_PER_SCALE = 64

# head_dim -> (BLOCK_Q, BLOCK_KV, num_warps, num_stages); measured on Arc Pro B60.
_ATTN_TILES = {
    64: (64, 64, 4, 2),
    128: (256, 32, 16, 2),
}
# head_dim -> num_warps of the Q and the K/V prepass; measured on Arc Pro B60.
_PREPASS_WARPS = {
    64: (8, 8),
    128: (16, 16),
}


@triton.jit
def _round_to_int8(x, inv_scale):
    # round half away from zero; |x * inv_scale| <= 127 by construction
    y = x * inv_scale
    magnitude = tl.floor(tl.abs(y) + 0.5)
    return tl.where(y < 0, -magnitude, magnitude).to(tl.int8)


@triton.jit
def _quantize_q_kernel(
    Q,
    K_mean,
    Q_int8,
    Q_scale,
    Lse,
    q_len,
    num_scales,
    stride_q_batch,
    stride_q_token,
    stride_q_head,
    stride_mean_batch,
    stride_mean_head,
    logit_scale,
    sm_scale,
    NUM_HEADS: tl.constexpr,
    GQA_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROWS: tl.constexpr,
    WRITE_LSE_SHIFT: tl.constexpr,
):
    block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    bh = batch * NUM_HEADS + head
    row_start = block * ROWS

    src = tl.make_tensor_descriptor(
        Q + batch.to(tl.int64) * stride_q_batch + head.to(tl.int64) * stride_q_head,
        shape=[q_len, HEAD_DIM],
        strides=[stride_q_token, 1],
        block_shape=[ROWS, HEAD_DIM],
    )
    q = src.load([row_start, 0]).to(tl.float32)

    if WRITE_LSE_SHIFT:
        dims = tl.arange(0, HEAD_DIM)
        k_mean = tl.load(
            K_mean
            + batch * stride_mean_batch
            + (head // GQA_GROUP) * stride_mean_head
            + dims
        ).to(tl.float32)
        rows = row_start + tl.arange(0, ROWS)
        shift = tl.sum(q * k_mean[None, :], axis=1) * sm_scale
        tl.store(Lse + bh.to(tl.int64) * q_len + rows, shift, mask=rows < q_len)

    # fold the softmax scale and log2(e) in, so the attention kernel uses exp2
    q = q * logit_scale
    amax = tl.maximum(tl.max(tl.abs(q)), 1e-20)
    dst = tl.make_tensor_descriptor(
        Q_int8 + bh.to(tl.int64) * q_len * HEAD_DIM,
        shape=[q_len, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[ROWS, HEAD_DIM],
    )
    dst.store([row_start, 0], _round_to_int8(q, 127.0 / amax))
    tl.store(Q_scale + bh * num_scales + block, amax / 127.0)


@triton.jit
def _prepare_kv_kernel(
    K,
    V,
    K_mean,
    K_int8,
    K_scale,
    V_fp16,
    kv_len,
    num_scales,
    stride_k_batch,
    stride_k_token,
    stride_k_head,
    stride_v_batch,
    stride_v_token,
    stride_v_head,
    stride_mean_batch,
    stride_mean_head,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROWS: tl.constexpr,
    SMOOTH_K: tl.constexpr,
):
    block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    bh = batch * NUM_KV_HEADS + head
    row_start = block * ROWS
    head_major = bh.to(tl.int64) * kv_len * HEAD_DIM

    k_src = tl.make_tensor_descriptor(
        K + batch.to(tl.int64) * stride_k_batch + head.to(tl.int64) * stride_k_head,
        shape=[kv_len, HEAD_DIM],
        strides=[stride_k_token, 1],
        block_shape=[ROWS, HEAD_DIM],
    )
    k = k_src.load([row_start, 0]).to(tl.float32)
    if SMOOTH_K:
        k_mean = tl.load(
            K_mean
            + batch * stride_mean_batch
            + head * stride_mean_head
            + tl.arange(0, HEAD_DIM)
        ).to(tl.float32)
        rows = row_start + tl.arange(0, ROWS)
        # padding rows must stay zero so they cannot raise the block's max
        k = tl.where((rows < kv_len)[:, None], k - k_mean[None, :], 0.0)
    amax = tl.maximum(tl.max(tl.abs(k)), 1e-20)
    k_dst = tl.make_tensor_descriptor(
        K_int8 + head_major,
        shape=[kv_len, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[ROWS, HEAD_DIM],
    )
    k_dst.store([row_start, 0], _round_to_int8(k, 127.0 / amax))
    tl.store(K_scale + bh * num_scales + block, amax / 127.0)

    v_src = tl.make_tensor_descriptor(
        V + batch.to(tl.int64) * stride_v_batch + head.to(tl.int64) * stride_v_head,
        shape=[kv_len, HEAD_DIM],
        strides=[stride_v_token, 1],
        block_shape=[ROWS, HEAD_DIM],
    )
    v_dst = tl.make_tensor_descriptor(
        V_fp16 + head_major,
        shape=[kv_len, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[ROWS, HEAD_DIM],
    )
    v_dst.store([row_start, 0], v_src.load([row_start, 0]).to(tl.float16))


@triton.jit
def _sage_attn_tile(
    acc,
    row_sum,
    row_max,
    q,
    q_scale,
    k_desc,
    v_desc,
    k_scale_row,
    kv_start,
    q_pos,
    kv_offs,
    kv_len,
    MASK_TAIL: tl.constexpr,
    MASK_CAUSAL: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    K_ROWS_PER_SCALE: tl.constexpr,
):
    k = k_desc.load([kv_start, 0])
    if K_ROWS_PER_SCALE % BLOCK_KV == 0:
        k_scale = tl.load(k_scale_row + kv_start // K_ROWS_PER_SCALE)
    else:
        kv_pos = kv_start + kv_offs
        k_scale = tl.load(
            k_scale_row + kv_pos // K_ROWS_PER_SCALE, mask=kv_pos < kv_len, other=0.0
        )[None, :]
    # logits in log2 units
    s = tl.dot(q, k.T).to(tl.float32) * (q_scale * k_scale)
    if MASK_TAIL:
        s = tl.where((kv_start + kv_offs)[None, :] < kv_len, s, float("-inf"))
    if MASK_CAUSAL:
        s = tl.where(q_pos[:, None] >= (kv_start + kv_offs)[None, :], s, float("-inf"))

    new_max = tl.maximum(row_max, tl.max(s, axis=1))
    correction = tl.math.exp2(row_max - new_max)
    p = tl.math.exp2(s - new_max[:, None])
    row_sum = row_sum * correction + tl.sum(p, axis=1)
    v = v_desc.load([kv_start, 0])
    acc = tl.dot(p.to(tl.float16), v, acc * correction[:, None])
    return acc, row_sum, new_max


@triton.jit
def _store_lse(Lse, bh, q_pos, q_len, row_max, row_sum):
    # Lse holds the smoothing shift written by the Q prepass
    lse_ptrs = Lse + bh.to(tl.int64) * q_len + q_pos
    in_range = q_pos < q_len
    shift = tl.load(lse_ptrs, mask=in_range, other=0.0)
    lse = (row_max + tl.math.log2(row_sum)) * 0.6931471805599453 + shift
    tl.store(lse_ptrs, lse, mask=in_range)


@triton.jit
def _sage_attn_fwd_kernel(
    Q_int8,
    K_int8,
    V_fp16,
    Q_scale,
    K_scale,
    Out,
    Lse,
    stride_o_batch,
    stride_o_token,
    stride_o_head,
    q_len,
    kv_len,
    num_q_scales,
    num_k_scales,
    NUM_HEADS: tl.constexpr,
    GQA_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    Q_ROWS_PER_SCALE: tl.constexpr,
    K_ROWS_PER_SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    q_block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    num_kv_heads: tl.constexpr = NUM_HEADS // GQA_GROUP
    bh = batch * NUM_HEADS + head
    bh_kv = batch * num_kv_heads + head // GQA_GROUP

    q_desc = tl.make_tensor_descriptor(
        Q_int8 + bh.to(tl.int64) * q_len * HEAD_DIM,
        shape=[q_len, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_Q, HEAD_DIM],
    )
    k_desc = tl.make_tensor_descriptor(
        K_int8 + bh_kv.to(tl.int64) * kv_len * HEAD_DIM,
        shape=[kv_len, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_KV, HEAD_DIM],
    )
    v_desc = tl.make_tensor_descriptor(
        V_fp16 + bh_kv.to(tl.int64) * kv_len * HEAD_DIM,
        shape=[kv_len, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_KV, HEAD_DIM],
    )

    q_start = q_block * BLOCK_Q
    q_pos = q_start + tl.arange(0, BLOCK_Q)
    kv_offs = tl.arange(0, BLOCK_KV)
    q_scale_row = Q_scale + bh * num_q_scales
    if Q_ROWS_PER_SCALE % BLOCK_Q == 0:
        q_scale = tl.load(q_scale_row + q_start // Q_ROWS_PER_SCALE)
    else:
        q_scale = tl.load(
            q_scale_row + q_pos // Q_ROWS_PER_SCALE, mask=q_pos < q_len, other=0.0
        )[:, None]
    k_scale_row = K_scale + bh_kv * num_k_scales

    q = q_desc.load([q_start, 0])
    acc = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
    row_sum = tl.zeros([BLOCK_Q], dtype=tl.float32)
    row_max = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)

    if CAUSAL:
        # top-left aligned (q_len == kv_len): tiles left of the diagonal are full
        full_end = q_start // BLOCK_KV * BLOCK_KV
        kv_end = tl.minimum(q_start + BLOCK_Q, kv_len)
    else:
        full_end = kv_len // BLOCK_KV * BLOCK_KV
        kv_end = kv_len
    for kv_start in range(0, full_end, BLOCK_KV):
        acc, row_sum, row_max = _sage_attn_tile(
            acc, row_sum, row_max, q, q_scale, k_desc, v_desc, k_scale_row,
            kv_start, q_pos, kv_offs, kv_len,
            False, False, BLOCK_KV, K_ROWS_PER_SCALE,
        )  # fmt: skip
    for kv_start in range(full_end, kv_end, BLOCK_KV):
        acc, row_sum, row_max = _sage_attn_tile(
            acc, row_sum, row_max, q, q_scale, k_desc, v_desc, k_scale_row,
            kv_start, q_pos, kv_offs, kv_len,
            True, CAUSAL, BLOCK_KV, K_ROWS_PER_SCALE,
        )  # fmt: skip

    o_desc = tl.make_tensor_descriptor(
        Out + batch.to(tl.int64) * stride_o_batch + head.to(tl.int64) * stride_o_head,
        shape=[q_len, HEAD_DIM],
        strides=[stride_o_token, 1],
        block_shape=[BLOCK_Q, HEAD_DIM],
    )
    o_desc.store([q_start, 0], (acc / row_sum[:, None]).to(Out.dtype.element_ty))

    if WRITE_LSE:
        _store_lse(Lse, bh, q_pos, q_len, row_max, row_sum)


# head-major [B, H, S, D] INT8 Q/K and FP16 V, with one scale per row block
class _SageOperands(msgspec.Struct, frozen=True):
    q_int8: torch.Tensor
    q_scale: torch.Tensor
    k_int8: torch.Tensor
    k_scale: torch.Tensor
    v_fp16: torch.Tensor
    # natural-log LSE buffer pre-filled with the K-smoothing shift
    lse: torch.Tensor | None


def _validate(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool
) -> None:
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"SageAttention expects FP16/BF16 inputs, got {q.dtype}")
    if not (q.dtype == k.dtype == v.dtype) or not (q.device == k.device == v.device):
        raise ValueError("q, k and v must share dtype and device")
    if q.dim() != 4 or k.shape != v.shape or q.shape[0] != k.shape[0]:
        raise ValueError(
            f"expected [B, S, H, D] q/k/v, got {tuple(q.shape)}, "
            f"{tuple(k.shape)}, {tuple(v.shape)}"
        )
    if q.shape[-1] > 128:
        raise ValueError(f"Unsupported head_dim: {q.shape[-1]}")
    if q.shape[2] % k.shape[2]:
        raise ValueError(
            f"num_qo_heads {q.shape[2]} is not a multiple of num_kv_heads {k.shape[2]}"
        )
    if causal and q.shape[1] != k.shape[1]:
        raise ValueError("causal SageAttention requires qo_len == kv_len")


def _pad_head_dim(x: torch.Tensor, head_dim: int) -> torch.Tensor:
    if x.shape[-1] == head_dim:
        return x
    return torch.nn.functional.pad(x, (0, head_dim - x.shape[-1]))


def _descriptor_source(x: torch.Tensor) -> torch.Tensor:
    # tensor descriptors need a unit last stride and 16-byte aligned base/strides
    nbytes = x.element_size()
    if (
        x.stride(-1) != 1
        or x.data_ptr() % 16
        or any((s * nbytes) % 16 for s in x.stride()[:-1])
    ):
        return x.contiguous()
    return x


def _prepare_operands(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    sm_scale: float,
    smooth_k: bool,
    return_lse: bool,
) -> _SageOperands:
    batch, q_len, num_heads, head_dim = q.shape
    _, kv_len, num_kv_heads, _ = k.shape
    device = q.device
    num_q_scales = triton.cdiv(q_len, _Q_ROWS_PER_SCALE)
    num_k_scales = triton.cdiv(kv_len, _K_ROWS_PER_SCALE)
    q_int8 = torch.empty(
        batch, num_heads, q_len, head_dim, device=device, dtype=torch.int8
    )
    k_int8 = torch.empty(
        batch, num_kv_heads, kv_len, head_dim, device=device, dtype=torch.int8
    )
    v_fp16 = torch.empty(
        batch, num_kv_heads, kv_len, head_dim, device=device, dtype=torch.float16
    )
    q_scale = torch.empty(
        batch, num_heads, num_q_scales, device=device, dtype=torch.float32
    )
    k_scale = torch.empty(
        batch, num_kv_heads, num_k_scales, device=device, dtype=torch.float32
    )
    # [batch, num_kv_heads, head_dim]
    k_mean = k.mean(dim=1) if smooth_k else None
    lse = None
    if return_lse:
        lse = torch.empty(batch, num_heads, q_len, device=device, dtype=torch.float32)
        if not smooth_k:
            lse.zero_()
    write_lse_shift = return_lse and smooth_k
    q_warps, kv_warps = _PREPASS_WARPS[head_dim]

    _quantize_q_kernel[(num_q_scales, num_heads, batch)](
        q, k_mean if write_lse_shift else q_scale, q_int8, q_scale,
        lse if write_lse_shift else q_scale,
        q_len, num_q_scales,
        q.stride(0), q.stride(1), q.stride(2),
        k_mean.stride(0) if write_lse_shift else 0,
        k_mean.stride(1) if write_lse_shift else 0,
        sm_scale * _LOG2E, sm_scale,
        NUM_HEADS=num_heads, GQA_GROUP=num_heads // num_kv_heads,
        HEAD_DIM=head_dim, ROWS=_Q_ROWS_PER_SCALE,
        WRITE_LSE_SHIFT=write_lse_shift, num_warps=q_warps,
    )  # fmt: skip
    _prepare_kv_kernel[(num_k_scales, num_kv_heads, batch)](
        k, v, k_mean if smooth_k else k_scale, k_int8, k_scale, v_fp16,
        kv_len, num_k_scales,
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        k_mean.stride(0) if smooth_k else 0, k_mean.stride(1) if smooth_k else 0,
        NUM_KV_HEADS=num_kv_heads, HEAD_DIM=head_dim, ROWS=_K_ROWS_PER_SCALE,
        SMOOTH_K=smooth_k, num_warps=kv_warps,
    )  # fmt: skip
    return _SageOperands(
        q_int8=q_int8,
        q_scale=q_scale,
        k_int8=k_int8,
        k_scale=k_scale,
        v_fp16=v_fp16,
        lse=lse,
    )


def _run_attention(
    operands: _SageOperands, *, out_dtype: torch.dtype, causal: bool
) -> torch.Tensor:
    batch, num_heads, q_len, head_dim = operands.q_int8.shape
    _, num_kv_heads, kv_len, _ = operands.k_int8.shape
    block_q, block_kv, num_warps, num_stages = _ATTN_TILES[head_dim]
    out = torch.empty(
        batch,
        q_len,
        num_heads,
        head_dim,
        device=operands.q_int8.device,
        dtype=out_dtype,
    )
    _sage_attn_fwd_kernel[(triton.cdiv(q_len, block_q), num_heads, batch)](
        operands.q_int8, operands.k_int8, operands.v_fp16,
        operands.q_scale, operands.k_scale, out,
        operands.lse if operands.lse is not None else operands.q_scale,
        out.stride(0), out.stride(1), out.stride(2),
        q_len, kv_len, operands.q_scale.shape[-1], operands.k_scale.shape[-1],
        NUM_HEADS=num_heads, GQA_GROUP=num_heads // num_kv_heads,
        HEAD_DIM=head_dim, BLOCK_Q=block_q, BLOCK_KV=block_kv,
        Q_ROWS_PER_SCALE=_Q_ROWS_PER_SCALE, K_ROWS_PER_SCALE=_K_ROWS_PER_SCALE,
        CAUSAL=causal, WRITE_LSE=operands.lse is not None,
        num_warps=num_warps, num_stages=num_stages, grf_mode="256",
    )  # fmt: skip
    return out


@torch.compiler.disable
def sage_attn_qk_int8_pv_fp16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    sm_scale: float | None = None,
    smooth_k: bool = True,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """SageAttention with per-block INT8 Q/K and FP16 P @ V.

    Args:
        q: ``[batch, q_len, num_heads, head_dim]``, FP16 or BF16, last dim
            contiguous.
        k, v: ``[batch, kv_len, num_kv_heads, head_dim]``; ``num_heads`` must
            be a multiple of ``num_kv_heads``. ``head_dim <= 128``.
        causal: top-left causal mask; requires ``q_len == kv_len``.
        sm_scale: softmax scale, ``head_dim ** -0.5`` by default.
        smooth_k: subtract the per-head key mean before quantizing K.
        return_lse: also return the natural-log softmax normalizer.

    Returns:
        The output in ``q``'s dtype and ``[batch, q_len, num_heads, head_dim]``
        layout, plus the LSE as ``[batch, num_heads, q_len]`` FP32 when
        ``return_lse`` is set.
    """
    _validate(q, k, v, causal=causal)
    head_dim = q.shape[-1]
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    padded_dim = 64 if head_dim <= 64 else 128
    q, k, v = (
        _descriptor_source(_pad_head_dim(t, head_dim=padded_dim)) for t in (q, k, v)
    )

    operands = _prepare_operands(
        q, k, v, sm_scale=sm_scale, smooth_k=smooth_k, return_lse=return_lse
    )
    out = _run_attention(operands, out_dtype=q.dtype, causal=causal)
    if padded_dim != head_dim:
        out = out[..., :head_dim]
    if return_lse:
        return out, operands.lse
    return out
