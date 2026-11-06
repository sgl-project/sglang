"""DeepSeek-V4.1 vision RoPE via the SGLang JIT kernel framework.

Drop-in for deepseek_v41_vit.apply_rotary / apply_vision_rotary.

Usage in deepseek_v41_vit.py:

    from sglang.kernels.ops.mm.dsv41_vit_rope import (
        apply_rotary,
        apply_vision_rotary,
    )

Or run a quick check:

    python -m sglang.kernels.ops.mm.dsv41_vit_rope
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


@cache_once
def _jit_module(dtype: torch.dtype) -> Module:
    if dtype not in _SUPPORTED_DTYPES:
        raise RuntimeError(
            f"Unsupported dtype {dtype}. Supported: bfloat16, float16, float32"
        )
    args = make_cpp_args(dtype)
    return load_jit(
        "dsv41_vit_rope",
        *args,
        cuda_files=["mm/dsv41_vit_rope.cuh"],
        cuda_wrappers=[("apply_rotary_inplace", f"apply_rotary_inplace<{args}>")],
        extra_cuda_cflags=["--use_fast_math"],
    )


def _prepare_cos_sin(
    cos: torch.Tensor, sin: torch.Tensor, num_tokens: int, half: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    if cos.dtype != torch.float32:
        cos = cos.float()
    if sin.dtype != torch.float32:
        sin = sin.float()
    if not cos.is_contiguous():
        cos = cos.contiguous()
    if not sin.is_contiguous():
        sin = sin.contiguous()
    if cos.shape != sin.shape:
        raise RuntimeError(f"cos/sin shape mismatch: {cos.shape} vs {sin.shape}")
    if cos.dim() == 3:
        if cos.size(0) != num_tokens or cos.size(1) != 1 or cos.size(2) != half:
            raise RuntimeError(
                f"expected cos shape [{num_tokens}, 1, {half}], got {tuple(cos.shape)}"
            )
        cos = cos.reshape(num_tokens, half)
        sin = sin.reshape(num_tokens, half)
    elif cos.dim() == 2:
        if cos.size(0) != num_tokens or cos.size(1) != half:
            raise RuntimeError(
                f"expected cos shape [{num_tokens}, {half}], got {tuple(cos.shape)}"
            )
    else:
        raise RuntimeError(f"cos must be 2D or 3D, got dim={cos.dim()}")
    return cos, sin


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Same contract as deepseek_v41_vit.apply_rotary, but one fused CUDA kernel.

    Args:
        x:   [T, H, D] CUDA contiguous (bf16/fp16/fp32)
        cos: [T, 1, D/2] or [T, D/2] float32 CUDA contiguous
        sin: same as cos
    Returns:
        x (inplace). Caller may treat return as the rotated tensor.
    """
    if not x.is_cuda:
        raise RuntimeError("x must be a CUDA tensor")
    if x.dim() != 3:
        raise RuntimeError(f"x must be [T, H, D], got {tuple(x.shape)}")
    if not x.is_contiguous():
        x = x.contiguous()
    num_tokens, _, head_dim = x.shape
    if head_dim % 2 != 0:
        raise RuntimeError(f"head_dim must be even, got {head_dim}")
    cos, sin = _prepare_cos_sin(cos, sin, num_tokens, head_dim // 2)
    _jit_module(x.dtype).apply_rotary_inplace(x, cos, sin)
    return x


def apply_vision_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    x_shape: Any = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drop-in for VisionAttention.customized_position_embedding_applier."""
    cos, sin = position_embeddings
    apply_rotary(q, cos, sin)
    apply_rotary(k, cos, sin)
    return q, k


# ---------------------------------------------------------------------------
# Reference (eager) + self-check
# ---------------------------------------------------------------------------


def apply_rotary_eager(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


def _self_check(dtype=torch.bfloat16, T=128, H=16, D=64, device="cuda"):
    assert torch.cuda.is_available(), "CUDA required"
    torch.manual_seed(0)
    x = torch.randn(T, H, D, device=device, dtype=dtype)
    cos = torch.randn(T, 1, D // 2, device=device, dtype=torch.float32)
    sin = torch.randn(T, 1, D // 2, device=device, dtype=torch.float32)

    ref = apply_rotary_eager(x, cos, sin)
    out = x.clone()
    apply_rotary(out, cos, sin)

    # FP32 math + bf16 store: allow a bit of rounding slack vs eager .to(bf16)
    atol = 2e-2 if dtype == torch.bfloat16 else 1e-3
    rtol = 2e-2 if dtype == torch.bfloat16 else 1e-3
    max_err = (out.float() - ref.float()).abs().max().item()
    ok = torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol)
    print(
        f"dtype={dtype} shape={[T, H, D]} max_err={max_err:.6g} "
        f"allclose={ok} (atol={atol})"
    )
    if not ok:
        raise AssertionError("vision RoPE kernel mismatch vs eager reference")

    # tiny bench
    warmup, iters = 10, 50
    for _ in range(warmup):
        apply_rotary(out, cos, sin)
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        apply_rotary(out, cos, sin)
    t1.record()
    torch.cuda.synchronize()
    fused_ms = t0.elapsed_time(t1) / iters

    y = x.clone()
    for _ in range(warmup):
        y = apply_rotary_eager(y, cos, sin)
    torch.cuda.synchronize()
    t0.record()
    for _ in range(iters):
        y = apply_rotary_eager(x, cos, sin)
    t1.record()
    torch.cuda.synchronize()
    eager_ms = t0.elapsed_time(t1) / iters
    print(
        f"bench: fused={fused_ms:.3f}ms  eager={eager_ms:.3f}ms  "
        f"speedup={eager_ms / fused_ms:.1f}x"
    )


if __name__ == "__main__":
    _self_check(torch.bfloat16)
    _self_check(torch.float16)
    print("ok")
