"""Native SM90 sparse GQA prefill with FP8 E4M3 Q/K/V and FP8 P."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


class SglNativeQ8KV8BuildError(RuntimeError):
    """Raised when the native Q8KV8 JIT module cannot be built or loaded."""


@cache_once
def _jit_native_q8kv8_module() -> Module:
    return load_jit(
        "minimax_sparse_gqa_q8kv8_sm90",
        cuda_files=["attention/minimax_sparse_gqa_q8kv8_sm90.cuh"],
        cuda_wrappers=[("dispatch", "minimax_sparse_gqa_q8kv8_sm90")],
        extra_cuda_cflags=[
            "-O3",
            "-DNDEBUG",
            "-DCUTE_USE_PACKED_TUPLE=1",
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
            "--use_fast_math",
        ],
        extra_dependencies=["cutlass"],
    )


def _unit_scale(value: Optional[float]) -> float:
    return 1.0 if value is None else float(value)


@torch.no_grad()
def sgl_native_q8kv8_sparse_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    topk_idx: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_size_k: int,
    sm_scale: Optional[float] = None,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
) -> torch.Tensor:
    """Run the SGL-native FP8-Q/FP8-KV sparse GQA prefill kernel."""
    tensors = {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "req_to_token": req_to_token,
        "slot_ids": slot_ids,
        "topk_idx": topk_idx,
        "cu_seqlens": cu_seqlens,
        "seq_lens": seq_lens,
        "prefix_lens": prefix_lens,
    }
    for name, tensor in tensors.items():
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if name != "q" and not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    if q.dtype != torch.float8_e4m3fn:
        raise ValueError(f"q must have dtype torch.float8_e4m3fn, got {q.dtype}")
    if q.stride(-1) != 1:
        raise ValueError(
            "q last dimension must be contiguous, "
            f"got shape={tuple(q.shape)}, stride={tuple(q.stride())}"
        )
    if k_cache.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"k_cache must have dtype torch.float8_e4m3fn, got {k_cache.dtype}"
        )
    if v_cache.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"v_cache must have dtype torch.float8_e4m3fn, got {v_cache.dtype}"
        )
    if req_to_token.dtype != torch.int32:
        raise ValueError("req_to_token must have dtype torch.int32")
    if slot_ids.dtype != torch.int64:
        raise ValueError("slot_ids must have dtype torch.int64")
    for name, tensor in (
        ("topk_idx", topk_idx),
        ("cu_seqlens", cu_seqlens),
        ("seq_lens", seq_lens),
        ("prefix_lens", prefix_lens),
    ):
        if tensor.dtype != torch.int32:
            raise ValueError(f"{name} must have dtype torch.int32")

    if q.ndim != 3 or k_cache.ndim != 3 or v_cache.ndim != 3:
        raise ValueError("q, k_cache, and v_cache must be rank-3 tensors")
    total_q, num_q_heads, head_dim = q.shape
    max_slots, num_kv_heads, k_head_dim = k_cache.shape
    if v_cache.shape != k_cache.shape:
        raise ValueError("k_cache and v_cache must have identical shapes")
    if head_dim != 128 or k_head_dim != 128:
        raise ValueError("the native Q8KV8 kernel requires head_dim=128")
    if block_size_k != 128:
        raise ValueError("the native Q8KV8 kernel requires block_size_k=128")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    group_size = num_q_heads // num_kv_heads
    if group_size not in (1, 2, 4, 8, 16):
        raise ValueError(f"unsupported local GQA group size: {group_size}")
    if topk_idx.ndim != 3 or topk_idx.shape[:2] != (num_kv_heads, total_q):
        raise ValueError("topk_idx must have shape [num_kv_heads, total_q, topk]")
    batch_size = cu_seqlens.numel() - 1
    if slot_ids.numel() != batch_size:
        raise ValueError("slot_ids length must match the prefill batch size")
    if seq_lens.numel() != batch_size or prefix_lens.numel() != batch_size:
        raise ValueError("sequence metadata length must match the prefill batch size")
    if torch.cuda.get_device_capability(q.device)[0] != 9:
        raise ValueError("the native Q8KV8 kernel requires SM90")

    if sm_scale is None:
        sm_scale = head_dim**-0.5
    output = torch.empty(
        (total_q, num_q_heads, head_dim), dtype=torch.bfloat16, device=q.device
    )
    if total_q == 0:
        return output
    with torch.cuda.device(q.device):
        stream = torch._C._cuda_getCurrentRawStream(q.device.index)
        try:
            module = _jit_native_q8kv8_module()
        except (ImportError, RuntimeError) as err:
            raise SglNativeQ8KV8BuildError(
                "failed to build or load the native Q8KV8 JIT module"
            ) from err
        module.dispatch(
            output,
            q,
            k_cache,
            v_cache,
            req_to_token,
            slot_ids,
            topk_idx,
            cu_seqlens,
            seq_lens,
            prefix_lens,
            int(total_q),
            int(num_q_heads),
            int(num_kv_heads),
            int(max_slots),
            int(req_to_token.shape[1]),
            int(topk_idx.shape[2]),
            int(batch_size),
            int(q.stride(0)),
            int(q.stride(1)),
            int(q.stride(2)),
            float(sm_scale),
            _unit_scale(q_scale),
            _unit_scale(k_scale),
            _unit_scale(v_scale),
            int(stream),
        )
    return output
