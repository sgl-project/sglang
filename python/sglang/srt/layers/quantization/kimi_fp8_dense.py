# SPDX-License-Identifier: Apache-2.0
"""Opt-in Kimi K3 FP8 dense dispatch for measured B300 decode shapes."""

from functools import cache

import torch


def use_deepgemm(m: int, n: int, k: int) -> bool:
    # Keep MLA Q-B on FlashInfer through M256; use DeepGEMM above it.
    # DFlash block8 produces aligned rows even for unpadded batch sizes.
    return (
        (
            (m == 8 or (128 <= m <= 384 and m % 8 == 0))
            and (n, k) in ((6144, 7168), (2112, 7168), (1536, 7168))
        )
        or ((128 <= m <= 384 and m % 8 == 0) and (n, k) == (7168, 1536))
        or (264 <= m <= 384 and m % 8 == 0 and (n, k) == (2304, 1536))
    )


def prepare_weights(layer, block_size) -> bool:
    from sglang.srt.layers import deep_gemm_wrapper
    from sglang.srt.layers.quantization.fp8_utils import (
        get_fp8_gemm_runner_backend,
        requant_weight_ue8m0,
    )

    weight, scale = layer.weight, layer.weight_scale_inv
    if (
        block_size != [128, 128]
        or not weight.is_cuda
        or weight.dtype != torch.float8_e4m3fn
        or scale.dtype != torch.float32
        or weight.ndim != 2
        or not any(use_deepgemm(m, *weight.shape) for m in (128, 384))
        or torch.cuda.get_device_capability(weight.device) != (10, 3)
        or not get_fp8_gemm_runner_backend().is_flashinfer_trtllm()
    ):
        return False
    if not (
        deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
    ):
        return False

    # Convert once at load time. Preserve original weights/scales for other M,
    # pre-quantized inputs, and the original prefill/FlashInfer path.
    converted_weight, packed_scale = requant_weight_ue8m0(weight, scale, block_size)
    layer.register_buffer("_k3_dense_weight", converted_weight, persistent=False)
    layer.register_buffer("_k3_dense_scale", packed_scale, persistent=False)
    return True


def try_apply(layer, x, bias):
    if not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16:
        return None
    m = x.numel() // x.shape[-1]
    if not use_deepgemm(m, *layer.weight.shape):
        return None
    from sglang.srt.layers.quantization.fp8_utils import (
        deepgemm_w8a8_block_fp8_linear_with_fallback,
    )

    if 128 < m <= 384:
        q, scales = quantize_rows(x.view(m, x.shape[-1]))
        output = deepgemm_w8a8_block_fp8_linear_with_fallback(
            q,
            layer._k3_dense_weight,
            [128, 128],
            layer._k3_dense_scale,
            input_scale=scales,
            bias=bias,
        )
        return output.view(*x.shape[:-1], layer.weight.shape[0])

    return deepgemm_w8a8_block_fp8_linear_with_fallback(
        x,
        layer._k3_dense_weight,
        [128, 128],
        layer._k3_dense_scale,
        bias=bias,
    )


@cache
def _row_quant_module():
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(
        torch.bfloat16, torch.float8_e4m3fn, 128, True, False, True, False, True
    )
    return load_jit(
        "kimi_fp8_quant_rows",
        *args,
        cuda_files=["gemm/per_token_group_quant.cuh"],
        cuda_wrappers=[("quant", f"PerTokenGroupQuantFlatKernel<{args}>::run_rows")],
        extra_cuda_cflags=["--use_fast_math"],
    )


def quantize_rows(x):
    """BF16 -> FP8 with packed, column-major UE8M0 scales for aligned decode rows."""
    m, k = x.shape
    if not (128 <= m <= 512 and m % 8 == 0) or k % 512 or not x.is_contiguous():
        raise ValueError("Unsupported Kimi row-quantization layout")
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty((k // 512, m), dtype=torch.int32, device=x.device).t()
    _row_quant_module().quant(x, q, scales)
    return q, scales
