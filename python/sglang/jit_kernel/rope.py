from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils import is_xpu
from sglang.srt.utils.custom_op import register_custom_op


if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_rotary_embedding_module() -> Module:
    return load_jit(
        "rotary_embedding",
        cuda_files=["elementwise/pos_enc.cuh"],
        cuda_wrappers=[("rotary_embedding", "RotaryEmbeddingKernel::run")],
    )


@cache_once
def _jit_fused_rope_module(is_neox: bool, rope_dim: int, dtype: torch.dtype) -> Module:
    args = make_cpp_args(is_neox, rope_dim, is_arch_support_pdl(), dtype)
    return load_jit(
        "fused_rope",
        *args,
        cuda_files=["elementwise/rope.cuh"],
        cuda_wrappers=[
            ("run_rope", f"FusedRopeKernel<{args}>::run"),
            ("run_rope_store", f"FusedRopeKernel<{args}>::run_fused"),
        ],
    )


@register_custom_op(
    op_name="rotary_embedding_with_key",
    mutates_args=["query", "key"],
)
def rotary_embedding_with_key(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
) -> None:
    module = _jit_rotary_embedding_module()
    module.rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox)


@register_custom_op(
    op_name="rotary_embedding_without_key",
    mutates_args=["query"],
)
def rotary_embedding_without_key(
    positions: torch.Tensor,
    query: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
) -> None:
    module = _jit_rotary_embedding_module()
    module.rotary_embedding(positions, query, None, head_size, cos_sin_cache, is_neox)


def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: Optional[torch.Tensor],
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
):
    if key is None:
        rotary_embedding_without_key(
            positions, query, head_size, cos_sin_cache, is_neox
        )
    else:
        rotary_embedding_with_key(
            positions, query, key, head_size, cos_sin_cache, is_neox
        )
    return query, key


# XPU/SYCL implementations
if is_xpu():
    _SUPPORTED_XPU_ROPE_DTYPES = {
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }
    _SUPPORTED_XPU_ROPE_DIMS = [64, 80, 96, 128, 256, 512]

    @cache_once
    def _jit_fused_rope_base_module_xpu(rope_dim: int):
        """Compile/load the shared XPU/SYCL fused RoPE module for rope_dim."""
        from sglang.jit_kernel.utils_xpu import load_jit_sycl

        if rope_dim not in _SUPPORTED_XPU_ROPE_DIMS:
            raise ValueError(
                f"Unsupported rope_dim for XPU RoPE: {rope_dim}. "
                f"Supported: {_SUPPORTED_XPU_ROPE_DIMS}"
            )

        return load_jit_sycl(
            "fused_rope",
            str(rope_dim),
            sycl_files=["elementwise/rope.hpp"],
            extra_sycl_cflags=[f"-DSGL_ROPE_DIM={rope_dim}"],
        )

    @cache_once
    def _jit_fused_rope_module_xpu(is_neox: bool, rope_dim: int, dtype: torch.dtype):
        """Return a cached XPUFusedRopeWrapper for the given configuration."""
        if dtype not in _SUPPORTED_XPU_ROPE_DTYPES:
            raise ValueError(
                f"Unsupported dtype for XPU RoPE: {dtype}. Only fp16/bf16 supported."
            )

        dtype_str = _SUPPORTED_XPU_ROPE_DTYPES[dtype]
        module = _jit_fused_rope_base_module_xpu(rope_dim)
        return _XPUFusedRopeWrapper(module, is_neox, rope_dim, dtype_str)

    class _XPUFusedRopeWrapper:
        """Wrapper for XPU fused RoPE kernel matching CUDA API."""

        def __init__(self, module, is_neox: bool, rope_dim: int, dtype_str: str):
            import ctypes

            self._module = module
            self._rope_dim = rope_dim
            self._dtype_str = dtype_str
            self._is_neox_str = "true" if is_neox else "false"

            # Define argtypes for run_rope
            self._rope_argtypes = [
                ctypes.c_void_p,  # queue
                ctypes.c_void_p,  # q_ptr
                ctypes.c_void_p,  # k_ptr
                ctypes.c_void_p,  # cos_sin_cache_ptr
                ctypes.c_void_p,  # positions
                ctypes.c_int64,  # q_stride
                ctypes.c_int64,  # k_stride
                ctypes.c_int64,  # head_stride
                ctypes.c_uint32,  # num_qo_heads
                ctypes.c_uint32,  # num_kv_heads
                ctypes.c_uint32,  # num_tokens
            ]

            # Define argtypes for run_rope_store
            self._rope_store_argtypes = [
                ctypes.c_void_p,  # queue
                ctypes.c_void_p,  # q_ptr
                ctypes.c_void_p,  # k_ptr
                ctypes.c_void_p,  # v_ptr
                ctypes.c_void_p,  # k_cache
                ctypes.c_void_p,  # v_cache
                ctypes.c_void_p,  # cos_sin_cache_ptr
                ctypes.c_void_p,  # positions
                ctypes.c_void_p,  # out_loc
                ctypes.c_int64,  # q_stride
                ctypes.c_int64,  # k_stride
                ctypes.c_int64,  # v_stride
                ctypes.c_int64,  # head_stride
                ctypes.c_int64,  # cache_stride
                ctypes.c_uint32,  # num_qo_heads
                ctypes.c_uint32,  # num_kv_heads
                ctypes.c_uint32,  # num_tokens
            ]

        def run_rope(self, q, k, cos_sin_cache, positions):
            """Apply RoPE inplace to q and k."""
            queue = torch.xpu.current_stream().sycl_queue

            num_tokens = q.shape[0]
            num_qo_heads = q.shape[1]
            num_kv_heads = k.shape[1]
            q_stride = q.stride(0)
            k_stride = k.stride(0)
            head_stride = q.stride(1)

            idtype_str = "i32" if positions.dtype == torch.int32 else "i64"
            func_name = (
                f"fused_rope_{self._is_neox_str}_{self._rope_dim}_"
                f"{self._dtype_str}_{idtype_str}"
            )

            func = self._module.get_function(func_name, self._rope_argtypes)

            func(
                queue,
                q.data_ptr(),
                k.data_ptr(),
                cos_sin_cache.data_ptr(),
                positions.data_ptr(),
                q_stride,
                k_stride,
                head_stride,
                num_qo_heads,
                num_kv_heads,
                num_tokens,
            )

        def run_rope_store(
            self, q, k, v, k_cache, v_cache, cos_sin_cache, positions, out_loc
        ):
            """Apply RoPE inplace to q/k and store rotated k and v to cache."""
            queue = torch.xpu.current_stream().sycl_queue

            num_tokens = q.shape[0]
            num_qo_heads = q.shape[1]
            num_kv_heads = k.shape[1]
            q_stride = q.stride(0)
            k_stride = k.stride(0)
            v_stride = v.stride(0)
            head_stride = q.stride(1)
            cache_stride = k_cache.stride(0)

            idtype_str = "i32" if positions.dtype == torch.int32 else "i64"
            func_name = (
                f"fused_rope_store_{self._is_neox_str}_{self._rope_dim}_"
                f"{self._dtype_str}_{idtype_str}"
            )

            func = self._module.get_function(func_name, self._rope_store_argtypes)

            func(
                queue,
                q.data_ptr(),
                k.data_ptr(),
                v.data_ptr(),
                k_cache.data_ptr(),
                v_cache.data_ptr(),
                cos_sin_cache.data_ptr(),
                positions.data_ptr(),
                out_loc.data_ptr(),
                q_stride,
                k_stride,
                v_stride,
                head_stride,
                cache_stride,
                num_qo_heads,
                num_kv_heads,
                num_tokens,
            )


@dataclass
class FusedSetKVBufferArg:
    """
    value : Optional[torch.Tensor]
        Value tensor, shape: ``(nnz, num_v_heads * head_size)``.
    k_buffer : Optional[torch.Tensor]
        Buffer for keys, shape: ``(nnz, num_k_heads * head_size)``.
    v_buffer : Optional[torch.Tensor]
        Buffer for values, shape: ``(nnz, num_v_heads * head_size)``.
    cache_loc : Optional[torch.Tensor]
        Cache location tensor, used for indexing kv cache.
    """

    value: torch.Tensor
    k_buffer: torch.Tensor
    v_buffer: torch.Tensor
    cache_loc: torch.Tensor


@register_custom_op(mutates_args=["q", "k"])
def apply_rope_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    is_neox: bool,
    rope_dim: int = 0,
) -> None:
    """
    Fused inplace rotary position embedding for query and key tensors.

    Args:
        q: Query tensor of shape [num_tokens, num_qo_heads, rope_dim].
        k: Key tensor of shape [num_tokens, num_kv_heads, rope_dim].
        cos_sin_cache: Cosine/sine cache of shape [max_position, rope_dim],
            where the first half along dim=-1 is cos and the second half is sin.
            Must be float32.
        positions: Position indices of shape [num_tokens], int32 or int64.
        is_neox: Whether to use GPT-NeoX style (True) or GPT-J interleaved style (False).
        rope_dim: Rotary embedding dimension. Defaults to cos_sin_cache.size(-1).
    """
    rope_dim = rope_dim or cos_sin_cache.size(-1)

    # Dispatch to XPU or CUDA based on device type
    if is_xpu() and q.device.type == "xpu":
        module = _jit_fused_rope_module_xpu(is_neox, rope_dim, q.dtype)
    else:
        module = _jit_fused_rope_module(is_neox, rope_dim, q.dtype)

    module.run_rope(q, k, cos_sin_cache, positions)


@register_custom_op(mutates_args=["q", "k", "k_cache", "v_cache"])
def apply_rope_inplace_with_kvcache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    out_loc: torch.Tensor,
    *,
    is_neox: bool,
    rope_dim: int = 0,
) -> None:
    """
    Fused inplace RoPE + KV cache store.

    Applies rotary position embedding to q and k inplace. The rotated k is also
    stored in k_cache. The original v is also stored in v_cache.

    Args:
        q: Query tensor of shape [num_tokens, num_qo_heads, head_dim].
        k: Key tensor of shape [num_tokens, num_kv_heads, head_dim].
        v: Value tensor of shape [num_tokens, num_kv_heads, head_dim].
        k_cache: Key cache of shape [cache_size, num_kv_heads * head_dim].
        v_cache: Value cache of shape [cache_size, num_kv_heads * head_dim].
        cos_sin_cache: Cosine/sine cache of shape [max_position, rope_dim], float32.
        positions: Position indices of shape [num_tokens], int32 or int64.
        out_loc: Cache write locations of shape [num_tokens], same dtype as positions.
        is_neox: Whether to use GPT-NeoX style (True) or GPT-J interleaved (False).
        rope_dim: Rotary embedding dimension. Defaults to cos_sin_cache.size(-1).
    """
    rope_dim = rope_dim or cos_sin_cache.size(-1)
    v = v.view_as(k)

    # Dispatch to XPU or CUDA based on device type
    if is_xpu() and q.device.type == "xpu":
        module = _jit_fused_rope_module_xpu(is_neox, rope_dim, q.dtype)
    else:
        module = _jit_fused_rope_module(is_neox, rope_dim, q.dtype)

    module.run_rope_store(q, k, v, k_cache, v_cache, cos_sin_cache, positions, out_loc)


# NOTE: this name is intentionally set as the old kernel in `sgl_kernel`
def apply_rope_with_cos_sin_cache_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    is_neox: bool,
    rope_dim: int = 0,
    fused_args: Optional[FusedSetKVBufferArg] = None,
) -> None:
    """
    Apply RoPE to q and k inplace, with optional fused kv cache store.

    If `fused_args` is provided, it will perform fused RoPE and KV cache store.
    Otherwise, it will only apply RoPE inplace.

    Args:
        q: Query tensor of shape [num_tokens, num_qo_heads, head_dim].
        k: Key tensor of shape [num_tokens, num_kv_heads, head_dim].
        cos_sin_cache: Cosine/sine cache of shape [max_position, rope_dim], float32.
        positions: Position indices of shape [num_tokens], int32 or int64.
        is_neox: Whether to use GPT-NeoX style (True) or GPT-J interleaved (False).
        rope_dim: Rotary embedding dimension. Defaults to cos_sin_cache.size(-1).
        fused_args: Optional arguments for fused RoPE + KV cache store. If None,
            only RoPE will be applied inplace without touching kv cache.
    """
    if fused_args is not None:
        apply_rope_inplace_with_kvcache(
            q,
            k,
            fused_args.value,
            fused_args.k_buffer,
            fused_args.v_buffer,
            cos_sin_cache,
            positions,
            fused_args.cache_loc,
            is_neox=is_neox,
            rope_dim=rope_dim,
        )
    else:
        apply_rope_inplace(
            q, k, cos_sin_cache, positions, is_neox=is_neox, rope_dim=rope_dim
        )
