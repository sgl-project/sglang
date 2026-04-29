from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit
from sglang.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_turboquant_dense_mla_decode_module() -> Module:
    return load_jit(
        "turboquant_dense_mla_decode",
        cuda_files=["quantization/turboquant_dense_mla_decode.cuh"],
        cuda_wrappers=[
            (
                "decode_2p5",
                "TurboQuantDenseMLADecode2p5Kernel::run",
            ),
            (
                "decode_2p5_split",
                "TurboQuantDenseMLADecode2p5SplitKernel::run",
            ),
            (
                "rotate_query",
                "TurboQuantDenseMLARotateQueryKernel::run",
            ),
            (
                "decode_2p5_split_rotated",
                "TurboQuantDenseMLADecode2p5SplitRotatedKernel::run",
            ),
        ],
    )


@debug_kernel_api
def turboquant_dense_mla_decode_2p5(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    compressed: torch.Tensor,
    page_table: torch.Tensor,
    out: torch.Tensor,
    centroids_high: torch.Tensor,
    centroids_low: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
    sm_scale: float,
) -> None:
    assert q_nope.is_cuda
    assert q_rope.is_cuda
    assert compressed.is_cuda
    assert page_table.is_cuda
    assert out.is_cuda
    assert q_nope.dtype == torch.bfloat16
    assert q_rope.dtype == torch.bfloat16
    assert compressed.dtype == torch.uint8
    assert page_table.dtype == torch.int32
    assert out.dtype == torch.bfloat16
    assert centroids_high.dtype == torch.float32
    assert centroids_low.dtype == torch.float32
    assert signs1.dtype == torch.float32
    assert signs2.dtype == torch.float32

    module = _jit_turboquant_dense_mla_decode_module()
    module.decode_2p5(
        q_nope.contiguous(),
        q_rope.contiguous(),
        compressed.contiguous(),
        page_table.contiguous(),
        out,
        centroids_high.contiguous(),
        centroids_low.contiguous(),
        signs1.contiguous(),
        signs2.contiguous(),
        float(sm_scale),
    )


@debug_kernel_api
def turboquant_dense_mla_decode_2p5_split(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    compressed: torch.Tensor,
    page_table: torch.Tensor,
    mid: torch.Tensor,
    out: torch.Tensor,
    centroids_high: torch.Tensor,
    centroids_low: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
    sm_scale: float,
) -> None:
    assert q_nope.is_cuda
    assert q_rope.is_cuda
    assert compressed.is_cuda
    assert page_table.is_cuda
    assert mid.is_cuda
    assert out.is_cuda
    assert q_nope.dtype == torch.bfloat16
    assert q_rope.dtype == torch.bfloat16
    assert compressed.dtype == torch.uint8
    assert page_table.dtype == torch.int32
    assert mid.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert centroids_high.dtype == torch.float32
    assert centroids_low.dtype == torch.float32
    assert signs1.dtype == torch.float32
    assert signs2.dtype == torch.float32

    module = _jit_turboquant_dense_mla_decode_module()
    module.decode_2p5_split(
        q_nope.contiguous(),
        q_rope.contiguous(),
        compressed.contiguous(),
        page_table.contiguous(),
        mid,
        out,
        centroids_high.contiguous(),
        centroids_low.contiguous(),
        signs1.contiguous(),
        signs2.contiguous(),
        float(sm_scale),
    )


@debug_kernel_api
def turboquant_dense_mla_rotate_query(
    q_nope: torch.Tensor,
    q_rotated: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
) -> None:
    assert q_nope.is_cuda
    assert q_rotated.is_cuda
    assert q_nope.dtype == torch.bfloat16
    assert q_rotated.dtype == torch.float32
    assert signs1.dtype == torch.float32
    assert signs2.dtype == torch.float32

    module = _jit_turboquant_dense_mla_decode_module()
    module.rotate_query(
        q_nope.contiguous(),
        q_rotated,
        signs1.contiguous(),
        signs2.contiguous(),
    )


@debug_kernel_api
def turboquant_dense_mla_decode_2p5_split_rotated(
    q_rotated: torch.Tensor,
    q_rope: torch.Tensor,
    compressed: torch.Tensor,
    page_table: torch.Tensor,
    mid: torch.Tensor,
    out: torch.Tensor,
    centroids_high: torch.Tensor,
    centroids_low: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
    sm_scale: float,
) -> None:
    assert q_rotated.is_cuda
    assert q_rope.is_cuda
    assert compressed.is_cuda
    assert page_table.is_cuda
    assert mid.is_cuda
    assert out.is_cuda
    assert q_rotated.dtype == torch.float32
    assert q_rope.dtype == torch.bfloat16
    assert compressed.dtype == torch.uint8
    assert page_table.dtype == torch.int32
    assert mid.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert centroids_high.dtype == torch.float32
    assert centroids_low.dtype == torch.float32
    assert signs1.dtype == torch.float32
    assert signs2.dtype == torch.float32

    module = _jit_turboquant_dense_mla_decode_module()
    module.decode_2p5_split_rotated(
        q_rotated,
        q_rope.contiguous(),
        compressed.contiguous(),
        page_table.contiguous(),
        mid,
        out,
        centroids_high.contiguous(),
        centroids_low.contiguous(),
        signs1.contiguous(),
        signs2.contiguous(),
        float(sm_scale),
    )
