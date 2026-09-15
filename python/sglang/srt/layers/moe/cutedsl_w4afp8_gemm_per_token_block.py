# SPDX-License-Identifier: Apache-2.0
"""Per-token-block W4AFP8 grouped GEMM operator.

W4AFP8 = INT4 weights + FP8 activations. DeepEP low-latency dispatch emits an
FP8 activation payload `[E, M, K]` together with a *per-token-block* scale
`[E, M, K // 128]` -- one scale per 128-element K block of every token. The
CUTLASS `cutlass_w4a8_moe_mm` kernel accepts a per-tensor activation scale.
The upstream LL path correctly requantizes the dispatched payload to that
scale. This experimental alternative removes that extra rounding/streaming
pass; it does not imply the upstream bridge discards scales incorrectly.

This module implements the GEMM that consumes the per-token-block scale directly:

    out[e, m, n] = sum_k a[e, m, k] * a_scale[e, m, k // 128]
                         * w_int4[e, n, k] * w_scale[e, n, k // 128]

Because each operand is scaled along K *before* the contraction, a plain
matmul over the pre-scaled operands reproduces the block-wise product exactly --
no per-tensor approximation is involved.

On SM90, contiguous BF16 output uses a fused CuTe DSL WGMMA kernel. Other
configurations use a dtype-faithful Torch reference (FP8 payload, sign-extended
INT4 weights and FP32 accumulation). Compilation and execution failures are
reported to the caller rather than silently switching to the slow reference.
"""

import logging
from functools import lru_cache
from typing import Optional

import torch

__all__ = [
    "cutedsl_w4afp8_gemm_per_token_block",
    "unpack_int4_weight",
    "deinterleave_w_scale",
]

# W4AFP8 quantizes both weights and activations in blocks of 128 along K.
SCALE_BLOCK_SIZE = 128
# INT4 scales are interleaved in groups of 4 (TRT-LLM layout) when K % 512 == 0.
SCALE_INTERLEAVE = 4


def unpack_int4_weight(packed: torch.Tensor) -> torch.Tensor:
    """Unpack INT4 weights packed two-per-byte into sign-extended FP32.

    `packed[..., j]` holds element `2j` in the low nibble and `2j + 1` in the
    high nibble (matches `pack_int4_values_to_int8` in the CUTLASS kernel test).
    Returns a tensor with the last dim doubled.
    """
    p = packed.to(torch.int32)
    low = p & 0xF
    high = (p >> 4) & 0xF
    # Sign-extend the 4-bit values: nibbles in [8, 15] represent [-8, -1].
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    unpacked = torch.stack([low, high], dim=-1)
    return unpacked.reshape(*packed.shape[:-1], packed.shape[-1] * 2).to(torch.float32)


def deinterleave_w_scale(
    w_scale_interleaved: torch.Tensor, n: int, k: int
) -> torch.Tensor:
    """Invert `interleave_scales`, recovering the logical `[E, N, K // 128]` scale.

    On-layer W4AFP8 weight scales are stored interleaved as `[E, K // 512, N * 4]`
    (see `w4afp8.interleave_scales`). This restores the natural per-(N, K-block)
    layout the GEMM consumes.
    """
    num_blocks_k = k // SCALE_BLOCK_SIZE
    alignment = SCALE_INTERLEAVE if num_blocks_k % SCALE_INTERLEAVE == 0 else 1
    e = w_scale_interleaved.shape[0]
    s = w_scale_interleaved.reshape(e, num_blocks_k // alignment, n, alignment)
    s = s.permute(0, 2, 1, 3)
    return s.reshape(e, n, num_blocks_k).contiguous()


def _expand_block_scale(scale: torch.Tensor, k: int) -> torch.Tensor:
    """Expand a per-block scale `[..., K // 128]` to per-element `[..., K]`."""
    num_blocks = scale.shape[-1]
    assert num_blocks * SCALE_BLOCK_SIZE == k, (
        f"scale blocks {num_blocks} * {SCALE_BLOCK_SIZE} != K {k}"
    )
    return scale.repeat_interleave(SCALE_BLOCK_SIZE, dim=-1)


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_hopper_kernel():
    try:
        from sglang.srt.layers.moe.cutedsl_w4afp8_gemm_hopper import (
            hopper_w4afp8_gemm_per_token_block,
        )
    except ImportError as exc:
        logger.warning("Hopper W4AFP8 unavailable; using the Torch reference: %s", exc)
        return None
    return hopper_w4afp8_gemm_per_token_block


def _try_hopper(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    output: torch.Tensor,
    masked_m: Optional[torch.Tensor],
) -> bool:
    """Run the fused Hopper WGMMA kernel in place if available; else return False
    so the caller evaluates the Torch reference. bf16 output only (kernel emits
    bf16); other dtypes fall through to the reference."""
    if not a.is_cuda or output.dtype != torch.bfloat16 or not output.is_contiguous():
        return False
    if torch.cuda.get_device_capability(a.device) != (9, 0):
        return False
    kernel = _get_hopper_kernel()
    if kernel is None:
        return False
    e, m, _ = a.shape
    if masked_m is None:
        masked_m = torch.full((e,), m, dtype=torch.int32, device=a.device)
    kernel(a, a_scale, w, w_scale, output, masked_m)
    return True


def _validate_inputs(a, a_scale, w, w_scale, output, masked_m):
    """Validate tensor metadata without device synchronization (graph-safe).

    Mask counts must lie in [0, M]. This value contract is the caller's
    responsibility; reading the device tensor here would synchronize CUDA.
    Output must be non-overlapping and use storage separate from all inputs.
    """
    if a.ndim != 3 or w.ndim != 3:
        raise ValueError("a and w must be rank-three grouped tensors")
    e, m, k = a.shape
    n = w.shape[1]
    if k % SCALE_BLOCK_SIZE or w.shape != (e, n, k // 2):
        raise ValueError("weight shape must be [E, N, K/2] with K divisible by 128")
    if a.dtype != torch.float8_e4m3fn or w.dtype != torch.int8:
        raise ValueError("expected float8_e4m3fn activations and packed int8 weights")
    if a_scale.shape != (e, m, k // SCALE_BLOCK_SIZE):
        raise ValueError("a_scale must be [E, M, K/128]")
    if w_scale.shape != (e, n, k // SCALE_BLOCK_SIZE):
        raise ValueError("w_scale must be logical [E, N, K/128]; deinterleave it first")
    if not a_scale.is_floating_point() or not w_scale.is_floating_point():
        raise ValueError("activation and weight scales must be floating point")
    if output.shape != (e, m, n) or output.dtype not in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ):
        raise ValueError("output must be [E, M, N] in bfloat16, float16 or float32")
    tensors = [a_scale, w, w_scale, output]
    if masked_m is not None:
        if masked_m.shape != (e,) or masked_m.dtype not in (torch.int32, torch.int64):
            raise ValueError("masked_m must be an int32/int64 tensor of shape [E]")
        tensors.append(masked_m)
    if any(t.device != a.device for t in tensors):
        raise ValueError("all tensors must be on the same device")
    if any(t.layout != torch.strided for t in [a, *tensors]):
        raise ValueError("all tensors must have strided layouts")
    if output.numel():
        # Conservative metadata-only proof of a non-overlapping destination.
        # Supports transposes and stepped slices, rejects expanded/overlapping views.
        span = 1
        for stride, size in sorted(zip(output.stride(), output.shape)):
            if size > 1:
                if stride < span:
                    raise ValueError("output must not have overlapping elements")
                span += (size - 1) * stride
        output_storage = output.untyped_storage().data_ptr()
        inputs = [a, a_scale, w, w_scale]
        if masked_m is not None:
            inputs.append(masked_m)
        if any(
            t.numel() and t.untyped_storage().data_ptr() == output_storage
            for t in inputs
        ):
            raise ValueError("output must use separate storage from inputs")


def cutedsl_w4afp8_gemm_per_token_block(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    output: torch.Tensor,
    masked_m: Optional[torch.Tensor] = None,
) -> None:
    """W4AFP8 grouped GEMM with native per-token-block activation scale.

    Args:
        a: FP8 activations `[E, M, K]` (``float8_e4m3fn``).
        a_scale: Per-token-block activation scale `[E, M, K // 128]` (FP32).
        w: INT4 weights packed two-per-byte `[E, N, K // 2]` (``int8``).
        w_scale: Logical weight scale `[E, N, K // 128]` (per-(N, K-block)). If
            the interleaved on-layer layout `[E, K // 512, N * 4]` is passed,
            convert it first with :func:`deinterleave_w_scale`.
        output: Non-overlapping destination `[E, M, N]`, written in place
            (typically bf16). Must use storage separate from all inputs.
        masked_m: Optional valid row count per expert `[E]` (int32/int64).
            Counts must lie in [0, M]. Invalid rows are explicitly zero-filled;
            their activation payload and scales need not be initialized.

    Computes ``out[e, m, n] = sum_k a[e,m,k] * a_scale[e,m,k//128]
    * w_int4[e,n,k] * w_scale[e,n,k//128]``.
    """
    _validate_inputs(a, a_scale, w, w_scale, output, masked_m)
    e, m, k = a.shape
    if output.numel() == 0:
        return
    if k == 0:
        output.zero_()
        return

    if _try_hopper(a, a_scale, w, w_scale, output, masked_m):
        return

    if a.is_cuda:
        with torch.cuda.device(a.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "The W4AFP8 Torch fallback does not support CUDA graph capture; "
                    "use the warmed-up Hopper kernel with contiguous bfloat16 output"
                )
    out_dtype = output.dtype
    # Per-expert loop bounds peak memory to one dequantized weight [N, K] rather
    # than materializing [E, N, K] at once; E is small for MoE (<= a few hundred).
    for expert in range(e):
        valid = m if masked_m is None else int(masked_m[expert].item())
        if valid == 0:
            output[expert].zero_()
            continue

        a_e = a[expert, :valid].to(torch.float32)  # [valid, K]
        a_e = a_e * _expand_block_scale(a_scale[expert, :valid], k)

        w_e = unpack_int4_weight(w[expert])  # [N, K]
        w_e = w_e * _expand_block_scale(w_scale[expert], k)

        out_e = torch.matmul(a_e, w_e.t())  # [valid, N], FP32 accumulation
        output[expert, :valid] = out_e.to(out_dtype)
        if valid < m:
            output[expert, valid:].zero_()
