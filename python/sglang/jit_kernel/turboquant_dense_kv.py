from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit
from sglang.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_turboquant_dense_kv_module() -> Module:
    return load_jit(
        "turboquant_dense_kv_dequant_4bit",
        cuda_files=["quantization/turboquant_dense_kv.cuh"],
        cuda_wrappers=[
            (
                "dequantize_selected_4bit",
                "TurboQuantDenseKVDequant4BitKernel::run",
            ),
            (
                "dequantize_selected_2p5",
                "TurboQuantDenseKVDequant2p5BitKernel::run",
            ),
            (
                "dequantize_page_table_selected_2p5",
                "TurboQuantDenseKVDequantPageTable2p5BitKernel::run",
            ),
            (
                "dequantize_page_table_selected_2p5_fp8",
                "TurboQuantDenseKVDequantPageTable2p5BitFP8Kernel::run",
            ),
            (
                "dequantize_page_table_selected_2p5_fp8_reuse",
                "TurboQuantDenseKVDequantPageTable2p5BitFP8ReuseKernel::run",
            ),
            (
                "store_2p5",
                "TurboQuantDenseKVStore2p5BitKernel::run",
            ),
        ],
    )


@debug_kernel_api
def dequantize_selected_4bit(
    compressed: torch.Tensor,
    locs: torch.Tensor,
    out: torch.Tensor,
    centroids: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
) -> None:
    assert compressed.is_cuda
    assert locs.is_cuda
    assert out.is_cuda
    assert compressed.dtype == torch.uint8
    assert locs.dtype == torch.int64
    assert out.dtype == torch.bfloat16
    assert centroids.dtype == torch.float32
    assert signs1.dtype == torch.float32
    assert signs2.dtype == torch.float32

    module = _jit_turboquant_dense_kv_module()
    module.dequantize_selected_4bit(
        compressed.contiguous(),
        locs.contiguous(),
        out,
        centroids.contiguous(),
        signs1.contiguous(),
        signs2.contiguous(),
    )


@debug_kernel_api
def dequantize_selected_2p5(
    compressed: torch.Tensor,
    locs: torch.Tensor,
    out: torch.Tensor,
    centroids_high: torch.Tensor,
    centroids_low: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
) -> None:
    assert compressed.is_cuda
    assert locs.is_cuda
    assert out.is_cuda
    assert compressed.dtype == torch.uint8
    assert locs.dtype == torch.int64
    assert out.dtype == torch.bfloat16
    assert centroids_high.dtype == torch.float32
    assert centroids_low.dtype == torch.float32
    assert signs1.dtype == torch.float32
    assert signs2.dtype == torch.float32

    module = _jit_turboquant_dense_kv_module()
    module.dequantize_selected_2p5(
        compressed.contiguous(),
        locs.contiguous(),
        out,
        centroids_high.contiguous(),
        centroids_low.contiguous(),
        signs1.contiguous(),
        signs2.contiguous(),
    )


@debug_kernel_api
def dequantize_page_table_selected_2p5(
    compressed: torch.Tensor,
    page_table: torch.Tensor,
    out: torch.Tensor,
    compact_page_table: torch.Tensor,
    centroids_high: torch.Tensor,
    centroids_low: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
) -> None:
    assert compressed.is_cuda
    assert page_table.is_cuda
    assert out.is_cuda
    assert compact_page_table.is_cuda
    assert compressed.dtype == torch.uint8
    assert page_table.dtype == torch.int32
    assert compact_page_table.dtype == torch.int32
    assert out.dtype == torch.bfloat16
    assert centroids_high.dtype == torch.float32
    assert centroids_low.dtype == torch.float32
    assert signs1.dtype == torch.float32
    assert signs2.dtype == torch.float32

    module = _jit_turboquant_dense_kv_module()
    module.dequantize_page_table_selected_2p5(
        compressed.contiguous(),
        page_table,
        out,
        compact_page_table,
        centroids_high.contiguous(),
        centroids_low.contiguous(),
        signs1.contiguous(),
        signs2.contiguous(),
    )


@debug_kernel_api
def dequantize_page_table_selected_2p5_fp8(
    compressed: torch.Tensor,
    page_table: torch.Tensor,
    out_u8: torch.Tensor,
    compact_page_table: torch.Tensor,
    centroids_high: torch.Tensor,
    centroids_low: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
) -> None:
    assert compressed.is_cuda
    assert page_table.is_cuda
    assert out_u8.is_cuda
    assert compact_page_table.is_cuda
    assert compressed.dtype == torch.uint8
    assert page_table.dtype == torch.int32
    assert compact_page_table.dtype == torch.int32
    assert out_u8.dtype == torch.uint8
    assert centroids_high.dtype == torch.float32
    assert centroids_low.dtype == torch.float32
    assert signs1.dtype == torch.float32
    assert signs2.dtype == torch.float32

    module = _jit_turboquant_dense_kv_module()
    module.dequantize_page_table_selected_2p5_fp8(
        compressed.contiguous(),
        page_table,
        out_u8,
        compact_page_table,
        centroids_high.contiguous(),
        centroids_low.contiguous(),
        signs1.contiguous(),
        signs2.contiguous(),
    )


@debug_kernel_api
def dequantize_page_table_selected_2p5_fp8_reuse(
    compressed: torch.Tensor,
    page_table: torch.Tensor,
    out_u8: torch.Tensor,
    compact_page_table: torch.Tensor,
    centroids_high: torch.Tensor,
    centroids_low: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
) -> None:
    assert compressed.is_cuda
    assert page_table.is_cuda
    assert out_u8.is_cuda
    assert compact_page_table.is_cuda
    assert compressed.dtype == torch.uint8
    assert page_table.dtype == torch.int32
    assert compact_page_table.dtype == torch.int32
    assert out_u8.dtype == torch.uint8
    assert centroids_high.dtype == torch.float32
    assert centroids_low.dtype == torch.float32
    assert signs1.dtype == torch.float32
    assert signs2.dtype == torch.float32

    module = _jit_turboquant_dense_kv_module()
    module.dequantize_page_table_selected_2p5_fp8_reuse(
        compressed.contiguous(),
        page_table,
        out_u8,
        compact_page_table,
        centroids_high.contiguous(),
        centroids_low.contiguous(),
        signs1.contiguous(),
        signs2.contiguous(),
    )


@debug_kernel_api
def store_2p5(
    compressed: torch.Tensor,
    locs: torch.Tensor,
    latent: torch.Tensor,
    rope: torch.Tensor,
    boundaries_high: torch.Tensor,
    boundaries_low: torch.Tensor,
    centroids_high: torch.Tensor,
    centroids_low: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
) -> None:
    assert compressed.is_cuda
    assert locs.is_cuda
    assert latent.is_cuda
    assert rope.is_cuda
    assert compressed.dtype == torch.uint8
    assert locs.dtype == torch.int64
    assert latent.dtype == torch.bfloat16
    assert rope.dtype == torch.bfloat16
    assert boundaries_high.dtype == torch.float32
    assert boundaries_low.dtype == torch.float32
    assert centroids_high.dtype == torch.float32
    assert centroids_low.dtype == torch.float32
    assert signs1.dtype == torch.float32
    assert signs2.dtype == torch.float32

    module = _jit_turboquant_dense_kv_module()
    module.store_2p5(
        compressed,
        locs.contiguous(),
        latent.contiguous() if not latent.is_contiguous() else latent,
        rope.contiguous() if not rope.is_contiguous() else rope,
        boundaries_high.contiguous(),
        boundaries_low.contiguous(),
        centroids_high.contiguous(),
        centroids_low.contiguous(),
        signs1.contiguous(),
        signs2.contiguous(),
    )
