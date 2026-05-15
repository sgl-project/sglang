from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# XPU support
_HAS_XPU = hasattr(torch, "xpu") and torch.xpu.is_available()

if _HAS_XPU:
    from sglang.jit_kernel.utils_xpu import load_jit_sycl


_RMSNORM_WARP_SIZES = frozenset({64, 128, 256})
_RMSNORM_MAX_HIDDEN_SIZE = 16384
_RMSNORM_HALF_BLOCK_MIN_SIZE = 2048


def _is_supported_rmsnorm_hidden_size(d: int) -> bool:
    return d in _RMSNORM_WARP_SIZES or (
        (d > 256 and d % 256 == 0 and d <= 8192)
        or (d >= 8192 and d % 512 == 0 and d <= 16384)
    )


def _rmsnorm_kernel_class(hidden_size: int) -> str:
    if hidden_size in _RMSNORM_WARP_SIZES:
        return "RMSNormWarpKernel"
    if hidden_size >= _RMSNORM_HALF_BLOCK_MIN_SIZE:
        if hidden_size % 512 == 0:
            return "RMSNormHalfKernel"
    return "RMSNormKernel"


@cache_once
def _jit_qknorm_module(head_dim: int, dtype: torch.dtype) -> Module:
    args = make_cpp_args(head_dim, is_arch_support_pdl(), dtype)
    return load_jit(
        "qknorm",
        *args,
        cuda_files=["elementwise/qknorm.cuh"],
        cuda_wrappers=[("qknorm", f"QKNormKernel<{args}>::run")],
    )


@cache_once
def _jit_rmsnorm_module(hidden_size: int, dtype: torch.dtype) -> Module:
    args = make_cpp_args(hidden_size, is_arch_support_pdl(), dtype)
    kernel_class = f"{_rmsnorm_kernel_class(hidden_size)}<{args}>"

    return load_jit(
        "rmsnorm",
        *args,
        cuda_files=["elementwise/rmsnorm.cuh"],
        cuda_wrappers=[("rmsnorm", f"{kernel_class}::run")],
    )


if _HAS_XPU:
    @cache_once
    def _jit_rmsnorm_module_xpu(hidden_size: int, dtype: torch.dtype):
        """XPU/SYCL version of RMSNorm JIT compilation"""
        # Map dtype to function suffix
        dtype_map = {
            torch.float32: "fp32",
            torch.float16: "fp16",
            torch.bfloat16: "bf16",
        }
        
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype for XPU RMSNorm: {dtype}")
        
        dtype_str = dtype_map[dtype]
        
        # Supported hidden sizes (must match C API instantiations in rmsnorm.hpp)
        supported_sizes = [64, 128, 256, 512, 1024, 1536, 2048, 2304, 2560, 3072, 4096, 5120, 6144, 7168, 8192, 12288, 16384]
        if hidden_size not in supported_sizes:
            raise ValueError(f"Unsupported hidden_size for XPU RMSNorm: {hidden_size}. Supported: {supported_sizes}")
        
        # Load the SYCL module — compile only the requested hidden_size + dtype
        module = load_jit_sycl(
            "rmsnorm",
            str(hidden_size),
            dtype_str,
            sycl_files=["elementwise/rmsnorm.hpp"],
            extra_sycl_cflags=[
                f"-DSGL_RMSNORM_HIDDEN_SIZE={hidden_size}",
                f"-DSGL_RMSNORM_DTYPE_{dtype_str}",
            ],
        )
        
        # Return a wrapper that matches the CUDA API
        class XPURMSNormWrapper:
            def __init__(self, module, hidden_size, dtype_str):
                import ctypes
                self._module = module
                self._func_name = f"rmsnorm_forward_{dtype_str}_{hidden_size}"
                self._argtypes = [
                    ctypes.c_void_p,  # queue
                    ctypes.c_void_p,  # input
                    ctypes.c_void_p,  # weight
                    ctypes.c_void_p,  # output
                    ctypes.c_int64,   # num_tokens
                    ctypes.c_int64,   # input_stride
                    ctypes.c_int64,   # output_stride
                    ctypes.c_float,   # eps
                ]
                
            def rmsnorm(self, input, weight, output, eps):
                # Validate layout assumptions before calling SYCL kernel
                if input.stride(-1) != 1:
                    raise ValueError(f"XPU RMSNorm requires contiguous last dim on input, got stride={input.stride()}")
                if output.stride(-1) != 1:
                    raise ValueError(f"XPU RMSNorm requires contiguous last dim on output, got stride={output.stride()}")
                if not weight.is_contiguous():
                    raise ValueError("XPU RMSNorm requires contiguous weight tensor")
                
                # Get XPU queue
                queue = torch.xpu.current_stream().sycl_queue
                
                # Get tensor info
                num_tokens = input.shape[0] if input.dim() > 1 else 1
                input_stride = input.stride(0) if input.dim() > 1 else input.numel()
                output_stride = output.stride(0) if output.dim() > 1 else output.numel()
                
                # Call the SYCL kernel using the module's helper method
                func = self._module.get_function(self._func_name, self._argtypes)
                
                func(
                    queue,
                    input.data_ptr(),
                    weight.data_ptr(),
                    output.data_ptr(),
                    num_tokens,
                    input_stride,
                    output_stride,
                    eps,
                )
        
        return XPURMSNormWrapper(module, hidden_size, dtype_str)

    @cache_once
    def _jit_qknorm_module_xpu(head_dim: int, dtype: torch.dtype):
        """XPU/SYCL version of QKNorm JIT compilation"""
        dtype_map = {
            torch.float16: "fp16",
            torch.bfloat16: "bf16",
        }

        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype for XPU QKNorm: {dtype}. Only fp16/bf16 supported.")

        dtype_str = dtype_map[dtype]

        supported_head_dims = [64, 128, 256, 512, 1024]
        if head_dim not in supported_head_dims:
            raise ValueError(f"Unsupported head_dim for XPU QKNorm: {head_dim}. Supported: {supported_head_dims}")

        module = load_jit_sycl(
            "qknorm",
            str(head_dim),
            dtype_str,
            sycl_files=["elementwise/qknorm.hpp"],
            extra_sycl_cflags=[
                f"-DSGL_QKNORM_HEAD_DIM={head_dim}",
                f"-DSGL_QKNORM_DTYPE_{dtype_str}",
            ],
        )

        return XPUQKNormWrapper(module, head_dim, dtype_str)

    class XPUQKNormWrapper:
        def __init__(self, module, head_dim, dtype_str):
            import ctypes
            self._module = module
            self._func_name = f"qknorm_forward_{dtype_str}_{head_dim}"
            self._argtypes = [
                ctypes.c_void_p,   # queue
                ctypes.c_void_p,   # q
                ctypes.c_void_p,   # k
                ctypes.c_void_p,   # q_weight
                ctypes.c_void_p,   # k_weight
                ctypes.c_int64,    # q_stride
                ctypes.c_int64,    # k_stride
                ctypes.c_uint32,   # num_qo_heads
                ctypes.c_uint32,   # num_kv_heads
                ctypes.c_uint32,   # num_tokens
                ctypes.c_float,    # eps
            ]

        def qknorm(self, q, k, q_weight, k_weight, eps):
            queue = torch.xpu.current_stream().sycl_queue

            num_tokens = q.shape[0]
            num_qo_heads = q.shape[1]
            num_kv_heads = k.shape[1]
            q_stride = q.stride(0)
            k_stride = k.stride(0)

            func = self._module.get_function(self._func_name, self._argtypes)

            func(
                queue,
                q.data_ptr(),
                k.data_ptr(),
                q_weight.data_ptr(),
                k_weight.data_ptr(),
                q_stride,
                k_stride,
                num_qo_heads,
                num_kv_heads,
                num_tokens,
                eps,
            )


@cache_once
def _jit_fused_add_rmsnorm_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "fused_add_rmsnorm",
        *args,
        cuda_files=["elementwise/fused_add_rmsnorm.cuh"],
        cuda_wrappers=[("fused_add_rmsnorm", f"FusedAddRMSNormKernel<{args}>::run")],
    )


@cache_once
def _jit_qknorm_across_heads_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "qknorm_across_heads",
        *args,
        cuda_files=["elementwise/qknorm_across_heads.cuh"],
        cuda_wrappers=[
            ("qknorm_across_heads", f"QKNormAcrossHeadsKernel<{args}>::run")
        ],
    )


@torch.compiler.assume_constant_result
@cache_once
def can_use_fused_inplace_qknorm(
    head_dim: int, dtype: torch.dtype, *, device_type: str = "cuda"
) -> bool:
    logger = logging.getLogger(__name__)
    if head_dim not in [64, 128, 256, 512, 1024]:
        logger.warning(f"Unsupported head_dim={head_dim} for JIT QK-Norm kernel")
        return False
    try:
        if device_type == "xpu":
            if not _HAS_XPU:
                return False
            _jit_qknorm_module_xpu(head_dim, dtype)
        elif device_type == "cuda":
            _jit_qknorm_module(head_dim, dtype)
        else:
            logger.warning(f"Unsupported device_type={device_type} for JIT QK-Norm kernel")
            return False
        return True
    except Exception as e:
        logger.warning(f"Failed to load JIT QK-Norm kernel: {e}")
        return False


@debug_kernel_api
def fused_inplace_qknorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    head_dim: int = 0,
) -> None:
    head_dim = head_dim or q.size(-1)
    logger = logging.getLogger(__name__)

    # XPU-specific path
    if _HAS_XPU and q.device.type == "xpu":
        if (
            q.dim() != 3
            or k.dim() != 3
            or q.size(-1) != head_dim
            or k.size(-1) != head_dim
            or not q.is_contiguous()
            or not k.is_contiguous()
            or q.storage_offset() != 0
            or k.storage_offset() != 0
            or q_weight.numel() != head_dim
            or k_weight.numel() != head_dim
            or not q_weight.is_contiguous()
            or not k_weight.is_contiguous()
            or q_weight.storage_offset() != 0
            or k_weight.storage_offset() != 0
        ):
            raise ValueError("Unsupported XPU QKNorm layout for JIT kernel")
        module = _jit_qknorm_module_xpu(head_dim, q.dtype)
        module.qknorm(q, k, q_weight, k_weight, eps)
        return

    # Non-CUDA/non-XPU devices use PyTorch fallback
    if q.device.type != "cuda":
        _qknorm_pytorch_fallback(q, k, q_weight, k_weight, eps)
        return

    # CUDA path
    module = _jit_qknorm_module(head_dim, q.dtype)
    module.qknorm(q, k, q_weight, k_weight, eps)


def _qknorm_pytorch_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
) -> None:
    """Pure PyTorch fallback for QKNorm"""
    q_var = q.float().pow(2).mean(dim=-1, keepdim=True)
    k_var = k.float().pow(2).mean(dim=-1, keepdim=True)
    q.copy_(q.float() * (q_var + eps).rsqrt() * q_weight.float())
    k.copy_(k.float() * (k_var + eps).rsqrt() * k_weight.float())


@debug_kernel_api
def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> None:
    """Apply RMSNorm to input tensor.

    Args:
        input: Input tensor
        weight: Weight tensor
        out: Output tensor (if None, operation is in-place on input)
        eps: Epsilon for numerical stability
    """
    out = out if out is not None else input
    logger = logging.getLogger(__name__)
    
    # XPU-specific path (early return to avoid touching CUDA flow)
    if _HAS_XPU and input.device.type == "xpu":
        hidden_size = input.size(-1)
        if (
            input.dim() != 2
            or out.dim() != 2
            or not input.is_contiguous()
            or not out.is_contiguous()
            or input.storage_offset() != 0
            or out.storage_offset() != 0
            or weight.dim() != 1
            or weight.numel() != hidden_size
            or not weight.is_contiguous()
            or weight.storage_offset() != 0
        ):
            raise ValueError("Unsupported XPU RMSNorm layout for JIT kernel")
        module = _jit_rmsnorm_module_xpu(hidden_size, input.dtype)
        module.rmsnorm(input, weight, out, eps)
        return
    
    # Non-XPU backends use PyTorch fallback.
    if input.device.type != "cuda":
        _rmsnorm_pytorch_fallback(input, weight, out, eps)
        return

    # CUDA path
    hidden_size = input.size(-1)
    
    # Validate hidden size and dtype for CUDA
    if not _is_supported_rmsnorm_hidden_size(hidden_size):
        raise ValueError(
            "Unsupported hidden_size for CUDA RMSNorm: "
            f"{hidden_size}. Supported: 64/128/256, multiples of 256 up to 8192, "
            "and multiples of 512 from 8192 to 16384."
        )
    
    if input.dtype not in [torch.float16, torch.bfloat16]:
        raise ValueError(f"CUDA RMSNorm only supports fp16/bf16, got {input.dtype}")
    
    module = _jit_rmsnorm_module(hidden_size, input.dtype)
    module.rmsnorm(input, weight, out, eps)


def _rmsnorm_pytorch_fallback(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    eps: float,
) -> None:
    """Pure PyTorch fallback implementation of RMSNorm"""
    input_fp32 = input.float()
    variance = input_fp32.pow(2).mean(-1, keepdim=True)
    normalized = input_fp32 * torch.rsqrt(variance + eps)
    output.copy_((normalized * weight.float()).to(input.dtype))


@debug_kernel_api
def fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> None:
    module = _jit_fused_add_rmsnorm_module(input.dtype)
    module.fused_add_rmsnorm(input, residual, weight, eps)


@debug_kernel_api
def fused_inplace_qknorm_across_heads(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float = 1e-6,
) -> None:
    """
    Fused inplace QK normalization across all heads.

    Args:
        q: Query tensor of shape [batch_size, num_heads * head_dim]
        k: Key tensor of shape [batch_size, num_heads * head_dim]
        q_weight: Query weight tensor of shape [num_heads * head_dim]
        k_weight: Key weight tensor of shape [num_heads * head_dim]
        eps: Epsilon for numerical stability
    """
    module = _jit_qknorm_across_heads_module(q.dtype)
    module.qknorm_across_heads(q, k, q_weight, k_weight, eps)
