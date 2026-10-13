"""Load-time WO-A preparation and projection of fused attention's MXFP8 output."""

from __future__ import annotations

import torch

from sglang.kernels.ops.attention.dsv4.mega_layout import permute_mega_wo_a
from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)


def get_mega_cos_sin_cache(freqs_cis: torch.Tensor, rope_key: tuple) -> torch.Tensor:
    from sglang.srt.runtime_context import get_buffer

    caches = get_buffer("dsv4_mega_cos_sin", dict)
    key = (rope_key, freqs_cis.device, freqs_cis.dtype)
    if key not in caches:
        caches[key] = torch.cat(
            (freqs_cis.real.float(), freqs_cis.imag.float()), dim=-1
        ).contiguous()
    return caches[key]


def prepare_mega_q_projection(linear, num_heads: int) -> torch.nn.Module | None:
    method = linear.quant_method
    backend = getattr(method, "mxfp8_dense_backend", None)
    if (
        num_heads not in (64, 128)
        or not getattr(method, "use_mxfp8", False)
        or getattr(method, "use_marlin", False)
        or backend is None
        or not (backend.is_flashinfer_cutlass() or backend.is_flashinfer_cutedsl())
    ):
        return None

    from flashinfer import block_scale_interleave

    weight, scale = linear.weight, linear.weight_scale_inv
    rows, cols = weight.shape
    if (
        rows != num_heads * 512
        or weight.dtype != torch.float8_e4m3fn
        or scale.dtype != torch.uint8
        or scale.shape != (rows, cols // 32)
    ):
        return None

    def permute_rows(tensor):
        return (
            tensor.view(num_heads, 32, 16, -1)
            .transpose(0, 1)
            .contiguous()
            .view(rows, -1)
        )

    projection = torch.nn.Module()
    projection.register_buffer("weight", permute_rows(weight), persistent=False)
    projection.register_buffer(
        "weight_scale_inv_swizzled",
        block_scale_interleave(permute_rows(scale)).contiguous(),
        persistent=False,
    )
    return projection


def prepare_mega_wo_a(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    permuted = permute_mega_wo_a(weight)
    quantized = [
        sglang_per_token_group_quant_fp8(
            group,
            group_size=32,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        for group in permuted.unbind(0)
    ]
    return (
        torch.stack([data for data, _ in quantized]),
        torch.stack([scale.transpose(0, 1) for _, scale in quantized]).transpose(1, 2),
    )


def project_mega_attention(
    data: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    import deep_gemm

    tokens, groups, rank = data.shape[0], weight.shape[0], weight.shape[1]
    output = torch.empty(
        (tokens, groups, rank), dtype=torch.bfloat16, device=data.device
    )
    if tokens:
        # The fused kernel already applied inverse RoPE and produced packed,
        # TMA-aligned per-32 scales. No cast/quantization/repacking belongs here.
        deep_gemm.fp8_einsum(
            "bhr,hdr->bhd",
            (data[:, :groups], scale[:, :groups]),
            (weight, weight_scale),
            output,
            recipe=(1, 1, 32),
        )
    return output
