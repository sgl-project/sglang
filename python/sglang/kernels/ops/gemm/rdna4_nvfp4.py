from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils import is_gfx1201_supported

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


def is_rdna4_nvfp4_device(device: torch.device | int | None = None) -> bool:
    """Return whether device is the exact ROCm target supported by this kernel."""
    if device is None:
        return is_gfx1201_supported()

    resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        return False
    return is_gfx1201_supported(resolved_device.index)


@cache_once
def _jit_rdna4_nvfp4_module(dtype: torch.dtype) -> Module:
    if dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"RDNA4 NVFP4 supports BF16 and FP16 activations, but received {dtype}."
        )
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


@cache_once
def warmup_rdna4_nvfp4(
    dtype: torch.dtype, device: torch.device, size_n: int, size_k: int
) -> None:
    """Compile every specialisation of one layer shape before graph capture.

    Both back ends compile on first use -- the HIP module through hipcc, the
    Triton prefill kernel once per tile shape and per Triton argument
    specialisation. Neither is capturable, so a server that captures CUDA
    graphs before its first real forward would compile inside the capture.
    Loading a layer is the last point where a plain launch is still safe.

    N and K are the layer's own so the Triton argument specialisations match
    what the layer will really launch; in practice every 16-divisible N and K
    collapses onto one specialisation, which makes every shape after the first
    almost free. The M sweep covers the remaining axes: M=1 is the HIP kernel,
    and the four Triton combinations are the two tile shapes (split at M=128)
    times Triton's `M == 1` / `M % 16` argument specialisations.

    Measured on an R9700: about 26 s for the first shape on a cold Triton
    cache, about 2 s warm, and about 10 ms for every shape after it.
    """
    weight = torch.zeros((size_n, size_k // 2), dtype=torch.uint8, device=device)
    weight_scale = torch.zeros(
        (size_n, size_k // 16), dtype=torch.float8_e4m3fn, device=device
    )
    global_scale = torch.ones(1, dtype=torch.float32, device=device)
    for size_m in (1, 2, 16, 128, 129):
        rdna4_nvfp4_linear(
            torch.zeros((size_m, size_k), dtype=dtype, device=device),
            weight,
            weight_scale,
            global_scale,
        )
    torch.cuda.synchronize(device)


def try_warmup_rdna4_nvfp4(
    dtype: torch.dtype, device: torch.device, size_n: int, size_k: int
) -> None:
    """Warm up if we can; a failure only costs a later compile, never the load."""
    try:
        warmup_rdna4_nvfp4(dtype, device, size_n, size_k)
    except Exception as error:  # noqa: BLE001 - warmup is best effort
        logger.warning(
            "RDNA4 NVFP4 warmup failed for N=%d K=%d (%s): %s. The kernels will "
            "compile on first use instead, which is not safe inside a CUDA graph "
            "capture.",
            size_n,
            size_k,
            dtype,
            error,
        )


def rdna4_nvfp4_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
) -> torch.Tensor:
    """Apply a canonical packed ModelOpt NVFP4 linear layer on gfx1201."""
    # The HIP launcher re-verifies shapes and dtypes through TensorMatcher, but
    # the Triton prefill path has no such launcher, so the contract is checked
    # here for both. Every check below is a cached bool or an integer compare.
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

    device = input.device
    if (
        weight.device != device
        or weight_scale.device != device
        or weight_global_scale.device != device
    ):
        raise ValueError("All RDNA4 NVFP4 tensors must be on the same device.")

    reshaped_input = input.reshape(-1, k).contiguous()
    packed_weight = weight.contiguous()
    block_scale = weight_scale.contiguous()
    global_scale = weight_global_scale.reshape(1).contiguous()
    output = torch.empty((reshaped_input.shape[0], n), dtype=input.dtype, device=device)

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
