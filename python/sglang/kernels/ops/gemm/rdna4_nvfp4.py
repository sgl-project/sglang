from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_SUPPORTED_GCN_ARCH = "gfx1201"
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


def normalize_gcn_arch_name(arch_name: str) -> str:
    """Remove optional feature suffixes from a ROCm GCN architecture name."""
    return arch_name.split(":", 1)[0].lower()


def is_rdna4_nvfp4_device(device: torch.device | int | None = None) -> bool:
    """Return whether device is the exact ROCm target supported by this kernel."""
    if torch.version.hip is None or not torch.cuda.is_available():
        return False

    if device is None:
        device_index = torch.cuda.current_device()
    else:
        resolved_device = torch.device(device)
        if resolved_device.type != "cuda":
            return False
        device_index = (
            torch.cuda.current_device()
            if resolved_device.index is None
            else resolved_device.index
        )

    properties = torch.cuda.get_device_properties(device_index)
    arch_name = getattr(properties, "gcnArchName", "")
    return normalize_gcn_arch_name(arch_name) == _SUPPORTED_GCN_ARCH


@cache_once
def _jit_rdna4_nvfp4_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "rdna4_nvfp4",
        *args,
        cuda_files=["gemm/rdna4_nvfp4.cuh"],
        cuda_wrappers=[
            ("run", f"sglang::Rdna4Nvfp4LinearKernel<{args}>::run"),
        ],
        extra_cuda_cflags=["-O3"],
    )


def rdna4_nvfp4_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
) -> torch.Tensor:
    """Apply a canonical packed ModelOpt NVFP4 linear layer on gfx1201."""
    if not is_rdna4_nvfp4_device(input.device):
        raise RuntimeError(
            "The RDNA4 NVFP4 JIT backend requires an exact gfx1201 ROCm device."
        )
    if input.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            "RDNA4 NVFP4 supports BF16 and FP16 activations, "
            f"but received {input.dtype}."
        )
    if input.ndim < 1 or input.shape[-1] == 0:
        raise ValueError("RDNA4 NVFP4 input must have a non-empty K dimension.")
    if input.numel() == 0:
        raise ValueError("RDNA4 NVFP4 input must contain at least one row.")
    if weight.ndim != 2 or weight.dtype != torch.uint8:
        raise TypeError("RDNA4 NVFP4 weight must be a 2D uint8 packed tensor.")
    if weight.shape[0] == 0:
        raise ValueError("RDNA4 NVFP4 weight must contain at least one output row.")
    if weight_scale.ndim != 2 or weight_scale.dtype != torch.float8_e4m3fn:
        raise TypeError("RDNA4 NVFP4 weight_scale must be a 2D float8_e4m3fn tensor.")
    if weight_global_scale.numel() != 1 or weight_global_scale.dtype != torch.float32:
        raise TypeError("RDNA4 NVFP4 global weight scale must be one float32 value.")

    k = input.shape[-1]
    n = weight.shape[0]
    if k % 16 != 0:
        raise ValueError(f"RDNA4 NVFP4 requires K divisible by 16, but got K={k}.")
    if weight.shape[1] * 2 != k:
        raise ValueError(
            "Packed RDNA4 NVFP4 weight shape does not match input K: "
            f"weight={tuple(weight.shape)}, K={k}."
        )
    if tuple(weight_scale.shape) != (n, k // 16):
        raise ValueError(
            "RDNA4 NVFP4 scale shape must be [N, K/16], but got "
            f"{tuple(weight_scale.shape)} for N={n}, K={k}."
        )

    tensors = (input, weight, weight_scale, weight_global_scale)
    if any(t.device != input.device for t in tensors):
        raise ValueError("All RDNA4 NVFP4 tensors must be on the same device.")

    reshaped_input = input.reshape(-1, k).contiguous()
    packed_weight = weight.contiguous()
    block_scale = weight_scale.contiguous()
    global_scale = weight_global_scale.reshape(1).contiguous()
    output = torch.empty(
        (reshaped_input.shape[0], n), dtype=input.dtype, device=input.device
    )

    if reshaped_input.shape[0] == 1:
        module = _jit_rdna4_nvfp4_module(input.dtype)
        module.run(
            reshaped_input,
            packed_weight,
            block_scale,
            global_scale,
            output,
        )
    else:
        from sglang.kernels.ops.gemm.rdna4_nvfp4_triton import (
            rdna4_nvfp4_prefill,
        )

        rdna4_nvfp4_prefill(
            reshaped_input,
            packed_weight,
            block_scale,
            global_scale,
            output,
        )
    return output.reshape(input.shape[:-1] + (n,))
