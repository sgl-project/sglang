from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_ATTENTION_HIDDEN = 6144
_MLP_HIDDEN = 18432
_OUTPUT_HIDDEN = _ATTENTION_HIDDEN + _MLP_HIDDEN
_ALIGNMENT = 32
_MAX_ROWS = 65408  # Largest multiple of 128 accepted by CUDA grid.y.


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() not in {"", "0", "false", "off", "no"}


def _is_dense_bf16(x: torch.Tensor, hidden: int) -> bool:
    return (
        isinstance(x, torch.Tensor)
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.dim() == 3
        and x.shape[0] == 1
        and x.shape[-1] == hidden
        and x.is_contiguous()
        and x.numel() > 0
        and x.data_ptr() % _ALIGNMENT == 0
    )


@cache_once
def _module() -> Module:
    return load_jit(
        "flux2_token_cat_nvfp4",
        cuda_files=["diffusion/flux2_token_cat_nvfp4.cuh"],
        cuda_wrappers=[("run", "flux2_token_cat_nvfp4::Kernel::run")],
        extra_cuda_cflags=["-DENABLE_BF16", "-DENABLE_FP4"],
        extra_dependencies=["flashinfer", "flashinfer_nv_internal"],
    )


def try_flux2_token_cat_nvfp4(
    attention: torch.Tensor,
    mlp: torch.Tensor,
    global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Concatenate FLUX.2 single-block branches directly into NVFP4."""
    if (
        torch.compiler.is_compiling()
        or not _is_dense_bf16(attention, _ATTENTION_HIDDEN)
        or not _is_dense_bf16(mlp, _MLP_HIDDEN)
        or mlp.device != attention.device
        or attention.shape[:-1] != mlp.shape[:-1]
        or torch.cuda.is_current_stream_capturing()
        or _env_enabled("FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH")
        or _env_enabled("TRTLLM_DISABLE_FP4_QUANT_FAST_MATH")
        or _env_enabled("FLASHINFER_NVFP4_4OVER6")
        or torch.cuda.get_device_capability(attention.device) != (10, 3)
    ):
        return None
    if not (
        isinstance(global_scale, torch.Tensor)
        and global_scale.is_cuda
        and global_scale.device == attention.device
        and global_scale.dtype == torch.float32
        and global_scale.numel() == 1
        and global_scale.is_contiguous()
    ):
        return None

    rows = attention.numel() // _ATTENTION_HIDDEN
    if rows > _MAX_ROWS:
        return None
    padded_rows = (rows + 127) // 128 * 128
    quantized = torch.empty(
        (rows, _OUTPUT_HIDDEN // 2), dtype=torch.uint8, device=attention.device
    )
    quant_scales = torch.empty(
        (padded_rows, _OUTPUT_HIDDEN // 16),
        dtype=torch.uint8,
        device=attention.device,
    )
    _module().run(
        quantized,
        quant_scales,
        attention.view(-1, _ATTENTION_HIDDEN),
        mlp.view(-1, _MLP_HIDDEN),
        global_scale.reshape(1),
    )
    return quantized, quant_scales


__all__ = ["try_flux2_token_cat_nvfp4"]
