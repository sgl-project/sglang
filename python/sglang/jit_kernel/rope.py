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
from sglang.srt.utils.custom_op import register_custom_op

# XPU support is detected lazily at call time to avoid import-time side effects
# (initialising the XPU runtime, failing on partial driver installs, or slowing imports).

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


# XPU/SYCL implementations.  All definitions are unconditional so the class is
# always importable; the XPU runtime is only initialised when an XPU tensor is
# actually passed (lazy detection via `q.device.type == "xpu"`).

_SUPPORTED_XPU_ROPE_DTYPES = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}
_SUPPORTED_XPU_ROPE_DIMS = [64, 80, 96, 128, 256, 512]


@cache_once
def _jit_fused_rope_base_module_xpu(rope_dim: int):
    """Compile/load the shared XPU/SYCL fused RoPE module for *rope_dim*.

    The module exports all is_neox and dtype variants for the given rope_dim,
    so it is cached only by rope_dim to avoid redundant compilations.
    ``load_jit_sycl`` is imported here (not at module level) to keep the XPU
    runtime initialisation truly lazy.
    """
    from sglang.jit_kernel.utils_xpu import load_jit_sycl  # lazy import

    if rope_dim not in _SUPPORTED_XPU_ROPE_DIMS:
        raise ValueError(
            f"Unsupported rope_dim for XPU RoPE: {rope_dim}. "
            f"Supported: {_SUPPORTED_XPU_ROPE_DIMS}"
        )

    return load_jit_sycl(
        "fused_rope",
        str(rope_dim),
        sycl_files=["elementwise/rope.hpp"],
        extra_sycl_cflags=[
            f"-DSGL_ROPE_DIM={rope_dim}",
        ],
    )


@cache_once
def _jit_fused_rope_module_xpu(is_neox: bool, rope_dim: int, dtype: torch.dtype):
    """Return a cached :class:`XPUFusedRopeWrapper` for the given configuration."""
    if dtype not in _SUPPORTED_XPU_ROPE_DTYPES:
        raise ValueError(
            f"Unsupported dtype for XPU RoPE: {dtype}. "
            f"Only fp16/bf16 supported."
        )

    dtype_str = _SUPPORTED_XPU_ROPE_DTYPES[dtype]
    module = _jit_fused_rope_base_module_xpu(rope_dim)
    return XPUFusedRopeWrapper(module, is_neox, rope_dim, dtype_str)


class XPUFusedRopeWrapper:
    """Wrapper for XPU fused RoPE kernel matching CUDA API."""

    def __init__(self, module, is_neox: bool, rope_dim: int, dtype_str: str):
        import ctypes
        self._module = module
        self._is_neox = is_neox
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
            ctypes.c_int64,   # q_stride
            ctypes.c_int64,   # k_stride
            ctypes.c_int64,   # head_stride
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
            ctypes.c_int64,   # q_stride
            ctypes.c_int64,   # k_stride
            ctypes.c_int64,   # v_stride
            ctypes.c_int64,   # head_stride
            ctypes.c_int64,   # cache_stride
            ctypes.c_uint32,  # num_qo_heads
            ctypes.c_uint32,  # num_kv_heads
            ctypes.c_uint32,  # num_tokens
        ]

    def _check_xpu_device(self, *tensors) -> None:
        """Raise if any tensor is not on an XPU device, or tensors span multiple devices."""
        devices = set()
        for t in tensors:
            if t.device.type != "xpu":
                raise ValueError(
                    f"XPU RoPE: all tensors must be on an XPU device, "
                    f"got tensor on {t.device}"
                )
            devices.add(t.device)
        if len(devices) > 1:
            raise ValueError(
                f"XPU RoPE: all tensors must be on the same XPU device, "
                f"got devices: {devices}"
            )

    def _check_last_dim_contiguous(self, *tensors) -> None:
        """Raise if any tensor does not have a contiguous last dimension."""
        for t in tensors:
            if t.stride(-1) != 1:
                raise ValueError(
                    "XPU RoPE requires tensors with a contiguous last dimension "
                    "(stride(-1) == 1). Pass a contiguous view or call "
                    ".contiguous() before invoking this function."
                )

    def _check_alignment(self, *tensors) -> None:
        """Raise if any tensor has misaligned data for vectorized loads.
        
        The SYCL kernel uses aligned 2-element vector loads for fp16/bf16,
        requiring storage_offset() to be even (divisible by 2).
        """
        for t in tensors:
            if t.dtype in (torch.float16, torch.bfloat16):
                if t.storage_offset() % 2 != 0:
                    raise ValueError(
                        f"XPU RoPE requires even storage_offset for {t.dtype} tensors "
                        f"(2-element vector alignment). Got storage_offset={t.storage_offset()}. "
                        f"Use .contiguous() or ensure slicing starts at even indices."
                    )

    def run_rope(self, q, k, cos_sin_cache, positions):
        """Apply RoPE inplace to *q* and *k*."""

        # Validate all tensors are on the same XPU device.
        self._check_xpu_device(q, k, cos_sin_cache, positions)

        # Validate tensor dtypes.
        if positions.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"positions must be int32 or int64, got {positions.dtype}"
            )
        if cos_sin_cache.dtype != torch.float32:
            raise ValueError(
                f"cos_sin_cache must be float32, got {cos_sin_cache.dtype}"
            )
        # Validate q and k have consistent dtypes and match the compiled dtype
        if q.dtype != k.dtype:
            raise ValueError(
                f"q and k must have the same dtype, got q.dtype={q.dtype}, k.dtype={k.dtype}"
            )
        expected_dtype = torch.float16 if self._dtype_str == "fp16" else torch.bfloat16
        if q.dtype != expected_dtype:
            raise ValueError(
                f"q/k dtype must match compiled dtype {expected_dtype} (compiled as {self._dtype_str}), "
                f"got {q.dtype}"
            )
        # Validate shape consistency: all tensors must have the same batch size
        num_tokens = q.shape[0]
        if k.shape[0] != num_tokens:
            raise ValueError(
                f"q and k must have the same batch size (shape[0]), "
                f"got q.shape[0]={num_tokens}, k.shape[0]={k.shape[0]}"
            )
        if positions.numel() != num_tokens:
            raise ValueError(
                f"positions.numel() must equal batch size, "
                f"got positions.numel()={positions.numel()}, expected {num_tokens}"
            )
        # The vectorised kernel loads require elements within each head to be
        # contiguous in memory.  Strides along the batch and head dimensions can
        # be arbitrary (they are read from stride(0)/stride(1)).
        self._check_last_dim_contiguous(q, k)
        # Validate alignment for vectorized loads (2-element vectors for fp16/bf16)
        self._check_alignment(q, k)
        # cos_sin_cache is accessed as base_ptr + pos * rope_dim (row-major);
        # positions is accessed as a flat array — both must be contiguous.
        if not cos_sin_cache.is_contiguous():
            raise ValueError(
                "cos_sin_cache must be contiguous for XPU RoPE"
            )
        if not positions.is_contiguous():
            raise ValueError(
                "positions must be contiguous for XPU RoPE"
            )

        queue = torch.xpu.current_stream().sycl_queue

        # Use actual strides so that partial-RoPE views (e.g. q[..., :rope_dim])
        # are handled correctly without an additional contiguous copy.
        num_qo_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        q_stride = q.stride(0)
        k_stride = k.stride(0)
        # stride(1) is the number of elements between consecutive heads in memory.
        # Slicing only the last dimension changes the visible width, but keeps this
        # head-to-head stride equal to the original layout's spacing.
        head_stride = q.stride(1)
        # Both q and k must share the same inter-head stride so the kernel can use
        # a single head_stride parameter for addressing both buffers.
        if k.stride(1) != head_stride:
            raise ValueError(
                f"q and k must have the same head stride (stride(1)) for XPU RoPE, "
                f"got q.stride(1)={head_stride}, k.stride(1)={k.stride(1)}"
            )
        # The compiled kernel processes exactly self._rope_dim elements per head.
        if q.shape[-1] != self._rope_dim:
            raise ValueError(
                f"q last dim ({q.shape[-1]}) must equal compiled rope_dim ({self._rope_dim})"
            )
        if k.shape[-1] != self._rope_dim:
            raise ValueError(
                f"k last dim ({k.shape[-1]}) must equal compiled rope_dim ({self._rope_dim})"
            )

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

    def run_rope_store(self, q, k, v, k_cache, v_cache, cos_sin_cache, positions, out_loc):
        """Apply RoPE inplace to *q*/*k* and store the rotated *k* and *v* to cache."""

        # Validate all tensors are on the same XPU device.
        self._check_xpu_device(q, k, v, k_cache, v_cache, cos_sin_cache, positions, out_loc)

        # Validate tensor dtypes.
        if positions.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"positions must be int32 or int64, got {positions.dtype}"
            )
        if out_loc.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"out_loc must be int32 or int64, got {out_loc.dtype}"
            )
        if positions.dtype != out_loc.dtype:
            raise ValueError(
                f"positions and out_loc must have the same dtype, "
                f"got {positions.dtype} and {out_loc.dtype}"
            )
        if cos_sin_cache.dtype != torch.float32:
            raise ValueError(
                f"cos_sin_cache must be float32, got {cos_sin_cache.dtype}"
            )
        # Validate q, k, and v have consistent dtypes and match the compiled dtype
        if q.dtype != k.dtype or q.dtype != v.dtype:
            raise ValueError(
                f"q, k, and v must have the same dtype, "
                f"got q.dtype={q.dtype}, k.dtype={k.dtype}, v.dtype={v.dtype}"
            )
        expected_dtype = torch.float16 if self._dtype_str == "fp16" else torch.bfloat16
        if q.dtype != expected_dtype:
            raise ValueError(
                f"q/k/v dtype must match compiled dtype {expected_dtype} (compiled as {self._dtype_str}), "
                f"got {q.dtype}"
            )
        # Validate shape consistency: all tensors must have the same batch size
        num_tokens = q.shape[0]
        if k.shape[0] != num_tokens or v.shape[0] != num_tokens:
            raise ValueError(
                f"q, k, and v must have the same batch size (shape[0]), "
                f"got q.shape[0]={num_tokens}, k.shape[0]={k.shape[0]}, v.shape[0]={v.shape[0]}"
            )
        if positions.numel() != num_tokens:
            raise ValueError(
                f"positions.numel() must equal batch size, "
                f"got positions.numel()={positions.numel()}, expected {num_tokens}"
            )
        if out_loc.numel() != num_tokens:
            raise ValueError(
                f"out_loc.numel() must equal batch size, "
                f"got out_loc.numel()={out_loc.numel()}, expected {num_tokens}"
            )
        # Same last-dim contiguity requirement as run_rope.
        self._check_last_dim_contiguous(q, k, v)
        # Validate alignment for vectorized loads (2-element vectors for fp16/bf16)
        self._check_alignment(q, k, v)
        if not cos_sin_cache.is_contiguous():
            raise ValueError(
                "cos_sin_cache must be contiguous for XPU RoPE store"
            )
        if not positions.is_contiguous():
            raise ValueError(
                "positions must be contiguous for XPU RoPE store"
            )
        if not out_loc.is_contiguous():
            raise ValueError(
                "out_loc must be contiguous for XPU RoPE store"
            )
        # The SYCL kernel uses a single cache_stride for both k_cache and v_cache.
        # Both caches must therefore be fully contiguous and share the same row stride.
        if not k_cache.is_contiguous():
            raise ValueError("k_cache must be contiguous for XPU RoPE store")
        if not v_cache.is_contiguous():
            raise ValueError("v_cache must be contiguous for XPU RoPE store")
        if k_cache.stride(0) != v_cache.stride(0):
            raise ValueError(
                f"k_cache and v_cache must have the same row stride for XPU RoPE store, "
                f"got k_cache.stride(0)={k_cache.stride(0)}, "
                f"v_cache.stride(0)={v_cache.stride(0)}"
            )

        queue = torch.xpu.current_stream().sycl_queue

        num_qo_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        q_stride = q.stride(0)
        k_stride = k.stride(0)
        v_stride = v.stride(0)
        head_stride = q.stride(1)
        # q, k, and v must all share the same inter-head stride; the SYCL kernel
        # uses a single head_stride for all three input buffers.
        if k.stride(1) != head_stride or v.stride(1) != head_stride:
            raise ValueError(
                f"q, k, and v must have the same head stride (stride(1)) for XPU RoPE store, "
                f"got q.stride(1)={head_stride}, k.stride(1)={k.stride(1)}, "
                f"v.stride(1)={v.stride(1)}"
            )
        # The SYCL fused-store kernel writes exactly rope_dim elements per head into
        # the cache at offset kv_head_id * head_stride.  head_stride must therefore
        # equal rope_dim (the compiled constant) so writes land in the right rows.
        if head_stride != self._rope_dim:
            raise ValueError(
                f"head_stride ({head_stride}) must equal rope_dim ({self._rope_dim}) "
                f"for XPU RoPE fused store. Partial-RoPE slicing alone does not change "
                f"stride(1) in PyTorch; pass tensors with stride(1) == rope_dim, e.g. "
                f"use x = x[..., :rope_dim].contiguous(), or allocate tensors with "
                f"head_dim == rope_dim."
            )
        # Each cache row must hold exactly num_kv_heads * rope_dim elements so that
        # the per-head scatter (loc * cache_stride + head_id * head_stride) stays
        # within bounds.
        expected_row_size = num_kv_heads * self._rope_dim
        if k_cache.shape[1] != expected_row_size:
            raise ValueError(
                f"k_cache second dim ({k_cache.shape[1]}) must equal "
                f"num_kv_heads * rope_dim = {expected_row_size}"
            )
        if v_cache.shape[1] != expected_row_size:
            raise ValueError(
                f"v_cache second dim ({v_cache.shape[1]}) must equal "
                f"num_kv_heads * rope_dim = {expected_row_size}"
            )
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
    
    # Dispatch to XPU or CUDA implementation based on the tensor's device.
    if q.device.type == "xpu":
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
    
    # Dispatch to XPU or CUDA implementation based on the tensor's device.
    if q.device.type == "xpu":
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
