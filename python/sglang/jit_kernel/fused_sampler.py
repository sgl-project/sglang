from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_TILE_SIZE = 1024
_MAX_TOP_K = 8
_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


@cache_once
def _jit_fused_sampler_module(
    dtype: torch.dtype, top_k: int, needs_top_p: bool
) -> Module:
    args = make_cpp_args(dtype, top_k, needs_top_p)
    return load_jit(
        "fused_sampler",
        *args,
        cuda_files=["sampling/fused_sampler.cuh"],
        cuda_wrappers=[("fused_topk_sample", f"FusedTopKSampleKernel<{args}>::run")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )


def _as_batch_float_tensor(
    value: float | torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.numel() != batch_size:
            raise RuntimeError(
                f"{name} must have {batch_size} values, got shape {tuple(value.shape)}"
            )
        return (
            value.reshape(batch_size)
            .to(device=device, dtype=torch.float32)
            .contiguous()
        )
    return torch.full((batch_size,), float(value), device=device, dtype=torch.float32)


def fused_topk_sample(
    logits: torch.Tensor,
    temperatures: float | torch.Tensor,
    top_ps: float | torch.Tensor,
    top_k: int,
    uniforms: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    scratch_scores: torch.Tensor | None = None,
    scratch_indices: torch.Tensor | None = None,
    scratch_sums: torch.Tensor | None = None,
    needs_top_p: bool | None = None,
) -> torch.Tensor:
    """Sample token ids from logits with fused top-k/top-p softmax.

    This lightweight JIT kernel is intended for homogeneous small-top-k batches.
    It expects logits after any penalties or vocab masks have already been applied.
    """
    if logits.dim() != 2:
        raise RuntimeError(
            f"logits must be 2D [batch, vocab], got {tuple(logits.shape)}"
        )
    if not logits.is_cuda:
        raise RuntimeError("logits must be a CUDA tensor")
    if logits.dtype not in _SUPPORTED_DTYPES:
        raise RuntimeError(
            f"Unsupported logits dtype {logits.dtype}; supported: {_SUPPORTED_DTYPES}"
        )
    if not logits.is_contiguous():
        logits = logits.contiguous()
    if top_k < 1 or top_k > _MAX_TOP_K:
        raise RuntimeError(f"top_k must be in [1, {_MAX_TOP_K}], got {top_k}")

    batch_size, vocab_size = logits.shape
    if top_k > vocab_size:
        raise RuntimeError(f"top_k ({top_k}) must not exceed vocab_size ({vocab_size})")

    device = logits.device
    temperatures_t = _as_batch_float_tensor(
        temperatures, batch_size=batch_size, device=device, name="temperatures"
    )
    top_ps_t = _as_batch_float_tensor(
        top_ps, batch_size=batch_size, device=device, name="top_ps"
    )
    if needs_top_p is None:
        if isinstance(top_ps, torch.Tensor):
            # Standalone callers may rely on inference. The runtime hot path passes
            # this flag from SamplingBatchInfo to avoid synchronizing here.
            needs_top_p = bool(torch.any(top_ps_t != 1.0).item())
        else:
            needs_top_p = float(top_ps) != 1.0
    if uniforms is None:
        uniforms_t = torch.rand((batch_size,), device=device, dtype=torch.float32)
    else:
        uniforms_t = _as_batch_float_tensor(
            uniforms, batch_size=batch_size, device=device, name="uniforms"
        )

    if out is None:
        out = torch.empty((batch_size,), device=device, dtype=torch.int64)
    elif out.shape != (batch_size,) or out.dtype != torch.int64 or out.device != device:
        raise RuntimeError(
            "out must have shape (batch_size,), dtype torch.int64, and match logits.device"
        )

    if top_k == 1:
        if scratch_scores is None:
            scratch_scores = torch.empty((0,), device=device, dtype=torch.float32)
        if scratch_indices is None:
            scratch_indices = torch.empty((0,), device=device, dtype=torch.int32)
        if scratch_sums is None:
            scratch_sums = torch.empty((0,), device=device, dtype=torch.float32)
    else:
        num_tiles = (vocab_size + _TILE_SIZE - 1) // _TILE_SIZE
        expected_shape = (batch_size, num_tiles, top_k)
        expected_sums_shape = (batch_size, num_tiles)
        if scratch_scores is None:
            scratch_scores = torch.empty(
                expected_shape, device=device, dtype=torch.float32
            )
        if scratch_indices is None:
            scratch_indices = torch.empty(
                expected_shape, device=device, dtype=torch.int32
            )
        if scratch_sums is None:
            scratch_sums = (
                torch.empty(expected_sums_shape, device=device, dtype=torch.float32)
                if needs_top_p
                else torch.empty((0,), device=device, dtype=torch.float32)
            )
        if (
            scratch_scores.shape != expected_shape
            or scratch_scores.dtype != torch.float32
            or scratch_scores.device != device
        ):
            raise RuntimeError(
                "scratch_scores must have shape "
                f"{expected_shape}, dtype torch.float32, and match logits.device"
            )
        if (
            scratch_indices.shape != expected_shape
            or scratch_indices.dtype != torch.int32
            or scratch_indices.device != device
        ):
            raise RuntimeError(
                "scratch_indices must have shape "
                f"{expected_shape}, dtype torch.int32, and match logits.device"
            )
        expected_scratch_sums_shape = expected_sums_shape if needs_top_p else (0,)
        if (
            scratch_sums.shape != expected_scratch_sums_shape
            or scratch_sums.dtype != torch.float32
            or scratch_sums.device != device
        ):
            raise RuntimeError(
                "scratch_sums must have shape "
                f"{expected_scratch_sums_shape}, dtype torch.float32, and match logits.device"
            )

    module = _jit_fused_sampler_module(logits.dtype, int(top_k), bool(needs_top_p))
    module.fused_topk_sample(
        out,
        logits,
        temperatures_t,
        top_ps_t,
        uniforms_t,
        scratch_scores,
        scratch_indices,
        scratch_sums,
    )
    return out
