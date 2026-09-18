"""Native SM90 split-K sparse decode for FP8 E4M3 Q/K/V."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


class SglNativeQ8KV8DecodeBuildError(RuntimeError):
    """Raised when the native decode JIT module cannot be built or loaded."""


@cache_once
def _jit_native_q8kv8_decode_module() -> Module:
    return load_jit(
        "minimax_sparse_decode_q8kv8_sm90",
        cuda_files=["attention/minimax_sparse_decode_q8kv8_sm90.cuh"],
        cuda_wrappers=[("dispatch", "minimax_sparse_decode_q8kv8_sm90")],
        extra_cuda_cflags=[
            "-O3",
            "-DNDEBUG",
            "-DCUTE_USE_PACKED_TUPLE=1",
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
            "--use_fast_math",
        ],
        extra_dependencies=["cutlass"],
    )


def _unit_scale(value: float | None) -> float:
    return 1.0 if value is None else float(value)


def _validate_page_contract(block_size_k: int, page_size: int) -> None:
    if block_size_k != 128 or page_size != block_size_k:
        raise ValueError(
            "the native Q8KV8 decode kernel requires page_size=block_size_k=128"
        )


def _choose_num_splits(batch_size: int, num_kv_heads: int, topk: int) -> int:
    """Choose a graph-static power-of-two split count from tensor shapes only."""
    target = max(1, min(topk, 256 // max(1, batch_size * num_kv_heads)))
    return 1 << (target.bit_length() - 1)


def _validate_contract(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    topk_idx: torch.Tensor,
    block_size_k: int,
    page_size: int,
) -> None:
    tensors = {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "req_to_token": req_to_token,
        "slot_ids": slot_ids,
        "seq_lens": seq_lens,
        "topk_idx": topk_idx,
    }
    for name, tensor in tensors.items():
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if name != "q" and not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if q.stride(-1) != 1:
        raise ValueError("q last dimension must be contiguous")
    if q.dtype != torch.float8_e4m3fn:
        raise ValueError(f"q must have dtype torch.float8_e4m3fn, got {q.dtype}")
    if k_cache.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"k_cache must have dtype torch.float8_e4m3fn, got {k_cache.dtype}"
        )
    if v_cache.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"v_cache must have dtype torch.float8_e4m3fn, got {v_cache.dtype}"
        )
    if q.ndim != 3 or k_cache.ndim != 3 or v_cache.ndim != 3:
        raise ValueError("q, k_cache, and v_cache must be rank-3 tensors")
    batch_size, num_q_heads, head_dim = q.shape
    _, num_kv_heads, k_head_dim = k_cache.shape
    if v_cache.shape != k_cache.shape:
        raise ValueError("k_cache and v_cache must have identical shapes")
    if head_dim != 128 or k_head_dim != 128:
        raise ValueError("the native Q8KV8 decode kernel requires head_dim=128")
    _validate_page_contract(block_size_k, page_size)
    if num_q_heads % num_kv_heads:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    group_size = num_q_heads // num_kv_heads
    if group_size not in (1, 2, 4, 8, 16):
        raise ValueError(f"unsupported local GQA group size: {group_size}")
    if req_to_token.dtype != torch.int32:
        raise ValueError("req_to_token must have dtype torch.int32")
    if slot_ids.dtype != torch.int64:
        raise ValueError("slot_ids must have dtype torch.int64")
    if seq_lens.dtype != torch.int32:
        raise ValueError("seq_lens must have dtype torch.int32")
    if topk_idx.dtype != torch.int32:
        raise ValueError("topk_idx must have dtype torch.int32")
    if slot_ids.numel() != batch_size or seq_lens.numel() != batch_size:
        raise ValueError("decode metadata length must match batch size")
    if topk_idx.ndim != 3 or topk_idx.shape[:2] != (
        num_kv_heads,
        batch_size,
    ):
        raise ValueError("topk_idx must have shape [num_kv_heads, batch, topk]")
    if torch.cuda.get_device_capability(q.device)[0] != 9:
        raise ValueError("the native Q8KV8 decode kernel requires SM90")


@torch.no_grad()
def sgl_native_q8kv8_sparse_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    topk_idx: torch.Tensor,
    block_size_k: int,
    page_size: int,
    sm_scale: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
) -> torch.Tensor:
    """Run native split-K Step3; the caller continues to own Triton indexing."""
    _validate_contract(
        q,
        k_cache,
        v_cache,
        req_to_token,
        slot_ids,
        seq_lens,
        topk_idx,
        block_size_k,
        page_size,
    )
    batch_size, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    topk = topk_idx.shape[2]
    output = torch.empty_like(q, dtype=torch.bfloat16)
    if batch_size == 0:
        return output
    if topk == 0:
        return output.zero_()
    num_splits = _choose_num_splits(batch_size, num_kv_heads, topk)
    partial = torch.empty(
        num_splits,
        batch_size,
        num_q_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=q.device,
    )
    lse = torch.empty(
        num_splits,
        batch_size,
        num_q_heads,
        dtype=torch.float32,
        device=q.device,
    )
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    with torch.cuda.device(q.device):
        stream = torch._C._cuda_getCurrentRawStream(q.device.index)
        try:
            module = _jit_native_q8kv8_decode_module()
        except (ImportError, RuntimeError) as err:
            raise SglNativeQ8KV8DecodeBuildError(
                "failed to build or load the native Q8KV8 decode JIT module"
            ) from err
        module.dispatch(
            output,
            partial,
            lse,
            q,
            k_cache,
            v_cache,
            req_to_token,
            slot_ids,
            topk_idx,
            seq_lens,
            int(batch_size),
            int(num_q_heads),
            int(num_kv_heads),
            int(k_cache.shape[0]),
            int(req_to_token.shape[1]),
            int(topk),
            int(num_splits),
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
