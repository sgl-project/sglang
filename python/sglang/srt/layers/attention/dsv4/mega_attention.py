from __future__ import annotations

import torch

from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.runtime_context import get_buffer


def use_mega_paged_prefill(
    num_query_tokens: int,
    seq_lens_cpu: torch.Tensor,
    compress_ratio: int,
    num_heads: int,
) -> bool:
    if num_heads != 64 or compress_ratio not in (1, 2) or num_query_tokens > 1024:
        return False
    # GB300 sweeps: paged reads win once the dequantized history dominates a short extension.
    gather_tokens = int(seq_lens_cpu.sum()) // compress_ratio
    return gather_tokens >= max(32768, 192 * num_query_tokens)


def mega_cos_sin_cache(freqs_cis: torch.Tensor) -> torch.Tensor:
    tables = get_buffer("dsv4_mega_cos_sin", dict)
    key = freqs_cis.data_ptr()
    if key not in tables:
        # The fused kernel reads [positions, cos(32) | sin(32)], not interleaved pairs.
        tables[key] = torch.cat((freqs_cis.real, freqs_cis.imag), dim=-1)
    return tables[key]


def prepare_mega_wq_b(
    linear: torch.nn.Module, num_heads: int
) -> torch.nn.Module | None:
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

    def permute_rows(tensor: torch.Tensor) -> torch.Tensor:
        # (head, chunk, 16) -> (chunk, head, 16): the kernel's Q row order.
        return tensor.view(num_heads, 32, 16, -1).transpose(0, 1).reshape(rows, -1)

    # Only .weight / .weight_scale_inv_swizzled are read by quant_method.apply.
    projection = torch.nn.Module()
    projection.weight = permute_rows(weight)
    projection.weight_scale_inv_swizzled = block_scale_interleave(
        permute_rows(scale)
    ).contiguous()
    return projection


def prepare_mega_wo_a(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    groups, rank, hidden = weight.shape
    # The kernel emits each group's 512 dims as 32-element chunks interleaved across its 8 heads.
    permuted = (
        weight.view(groups, rank, 8, 16, 32).transpose(2, 3).reshape(weight.shape)
    )
    return sglang_per_token_group_quant_fp8(
        permuted,
        group_size=32,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
        # Fully-masked batched quant: the one layout with per-group column-major scales.
        masked_m=torch.full((groups,), rank, dtype=torch.int32, device=weight.device),
    )


def apply_mega_wo_a(
    data: torch.Tensor,
    scale: torch.Tensor,
    wo_a: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    import deep_gemm

    weight, weight_scale = wo_a
    groups, rank = weight.shape[:2]
    out = torch.empty(
        (data.shape[0], groups, rank), dtype=torch.bfloat16, device=data.device
    )
    # Groups past the real heads hold the kernel's zero-padded heads.
    deep_gemm.fp8_einsum(
        "bhr,hdr->bhd",
        (data[:, :groups], scale[:, :groups]),
        (weight, weight_scale),
        out,
        recipe=(1, 1, 32),
    )
    return out
