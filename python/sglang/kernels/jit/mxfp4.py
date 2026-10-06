"""JIT kernel wrappers for MXFP4 paged quantization and dequantization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_mxfp4_dequant_module() -> Module:
    return load_jit(
        "mxfp4_dequant",
        cuda_files=["quantization/mxfp4_dequant.cuh"],
        cuda_wrappers=[
            ("quantize_4w_i32", "mxfp4_quantize_and_store_paged<4, false>"),
            ("quantize_4w_i64", "mxfp4_quantize_and_store_paged<4, true>"),
            ("quantize_8w_i32", "mxfp4_quantize_and_store_paged<8, false>"),
            ("quantize_8w_i64", "mxfp4_quantize_and_store_paged<8, true>"),
            ("quantize_16w_i32", "mxfp4_quantize_and_store_paged<16, false>"),
            ("quantize_16w_i64", "mxfp4_quantize_and_store_paged<16, true>"),
            ("dequantize_4w", "mxfp4_dequantize_paged<4>"),
            ("dequantize_8w", "mxfp4_dequantize_paged<8>"),
            ("dequantize_16w", "mxfp4_dequantize_paged<16>"),
            ("dequantize_32w", "mxfp4_dequantize_paged<32>"),
            ("dequantize_dedup_4w", "mxfp4_dequantize_paged_dedup<4>"),
            ("dequantize_dedup_8w", "mxfp4_dequantize_paged_dedup<8>"),
            (
                "dequantize_dedup_latent_4w",
                "mxfp4_dequantize_paged_dedup<4, false>",
            ),
        ],
    )


def mxfp4_quantize_and_store_paged(
    data: torch.Tensor,
    tail: torch.Tensor,
    locations: torch.Tensor,
    data_output: torch.Tensor,
    scale_output: torch.Tensor,
    tail_output: torch.Tensor,
    warps_per_block: int = 4,
) -> None:
    """Quantize input tensors to MXFP4 and store them into paged KV cache."""
    module = _jit_mxfp4_dequant_module()
    location_kind = "i64" if locations.dtype == torch.int64 else "i32"
    getattr(module, f"quantize_{warps_per_block}w_{location_kind}")(
        data, tail, locations, data_output, scale_output, tail_output
    )


def mxfp4_dequantize_paged(
    data: torch.Tensor,
    scales: torch.Tensor,
    tail: torch.Tensor,
    locations: torch.Tensor,
    output: torch.Tensor,
    compact_page_table: torch.Tensor,
    warps_per_block: int = 4,
) -> None:
    """Dequantize MXFP4 paged KV cache into the output tensor."""
    module = _jit_mxfp4_dequant_module()
    getattr(module, f"dequantize_{warps_per_block}w")(
        data, scales, tail, locations, output, compact_page_table
    )


def mxfp4_dequantize_paged_dedup(
    data: torch.Tensor,
    scales: torch.Tensor,
    tail: torch.Tensor,
    locations: torch.Tensor,
    output: torch.Tensor,
    page_table: torch.Tensor,
    claimed: torch.Tensor,
    warps_per_block: int = 8,
) -> None:
    """Dequantize MXFP4 paged KV cache with page deduplication."""
    module = _jit_mxfp4_dequant_module()
    getattr(module, f"dequantize_dedup_{warps_per_block}w")(
        data, scales, tail, locations, output, page_table, claimed
    )


def mxfp4_dequantize_paged_dedup_latent(
    data: torch.Tensor,
    scales: torch.Tensor,
    tail: torch.Tensor,
    locations: torch.Tensor,
    output: torch.Tensor,
    page_table: torch.Tensor,
    claimed: torch.Tensor,
) -> None:
    """Dequantize MXFP4 paged latent KV cache with page deduplication."""
    module = _jit_mxfp4_dequant_module()
    module.dequantize_dedup_latent_4w(
        data, scales, tail, locations, output, page_table, claimed
    )
