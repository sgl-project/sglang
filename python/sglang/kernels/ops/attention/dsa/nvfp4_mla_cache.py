"""NVFP4 DSA main-cache gather/dequantization.

The persistent cache stores the 576-element MLA row as packed E2M1 data plus
one E4M3 scale per block of 16 values. TRTLLM-GEN sparse MLA consumes FP8, so
the generation path uses a persistent CUDA kernel adapted from TensorRT-LLM to
gather and convert selected rows into a compact FP8 cache. The context path
retains its correctness-first deduplicating implementation for now.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)
_cuda_gather_failed = False


@cache_once
def _jit_nvfp4_mla_gather_module() -> Module:
    if torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError("NVFP4 MLA CUDA gather requires SM100/SM103")
    return load_jit(
        "nvfp4_mla_cache_gather_sm10x",
        cuda_files=["dsa/nvfp4_mla_cache_gather.cuh"],
        cuda_wrappers=[("gather", "nvfp4_mla::GatherKernel::run")],
        extra_cuda_cflags=["-O3", "-DNDEBUG"],
    )


def preload_nvfp4_mla_gather() -> None:
    """Compile/load the CUDA gather before entering CUDA Graph capture."""
    _jit_nvfp4_mla_gather_module()


@triton.jit
def _gather_dequant_nvfp4_mla_generation_kernel(
    data_ptr,
    scale_ptr,
    physical_indices_ptr,
    output_ptr,
    compact_indices_ptr,
    global_scale_ptr,
    data_row_stride: tl.constexpr,
    scale_row_stride: tl.constexpr,
    head_dim: tl.constexpr,
    block_dim: tl.constexpr,
):
    row = tl.program_id(0)
    dim_block = tl.program_id(1)
    physical_row = tl.load(physical_indices_ptr + row)
    row_valid = physical_row >= 0

    offsets = dim_block * block_dim + tl.arange(0, block_dim)
    mask = row_valid & (offsets < head_dim)
    packed = tl.load(
        data_ptr + physical_row * data_row_stride + offsets // 2,
        mask=mask,
        other=0,
    ).to(tl.uint8)
    code = tl.where((offsets & 1) == 0, packed & 0xF, packed >> 4)
    magnitude_code = code & 0x7
    # E2M1 magnitudes for codes 0..7: 0, .5, 1, 1.5, 2, 3, 4, 6.
    value = tl.where(
        magnitude_code == 0,
        0.0,
        tl.where(
            magnitude_code == 1,
            0.5,
            tl.where(
                magnitude_code == 2,
                1.0,
                tl.where(
                    magnitude_code == 3,
                    1.5,
                    tl.where(
                        magnitude_code == 4,
                        2.0,
                        tl.where(
                            magnitude_code == 5,
                            3.0,
                            tl.where(magnitude_code == 6, 4.0, 6.0),
                        ),
                    ),
                ),
            ),
        ),
    )
    value = tl.where((code & 0x8) != 0, -value, value)
    block_scale = tl.load(
        scale_ptr + physical_row * scale_row_stride + offsets // 16,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    global_scale = tl.load(global_scale_ptr).to(tl.float32)
    tl.store(
        output_ptr + row * head_dim + offsets,
        value * block_scale * global_scale,
        mask=offsets < head_dim,
    )

    tl.store(
        compact_indices_ptr + row,
        tl.where(row_valid, row, -1),
        mask=dim_block == 0,
    )


def gather_dequant_nvfp4_mla_cache_generation(
    data_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    physical_indices: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    head_dim: int,
    page_size: int,
    output: torch.Tensor | None = None,
    compact_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather/dequantize generation rows without cross-query deduplication.

    ``output`` and ``compact_indices`` allow the attention backend to reuse
    stable scratch allocations across layers and CUDA Graph replays. The CUDA
    implementation is the optimized persistent, asynchronously prefetched
    TensorRT-LLM algorithm; the Triton kernel remains as a portability fallback.
    """
    global _cuda_gather_failed

    if physical_indices.ndim != 2 or physical_indices.dtype != torch.int32:
        raise ValueError("physical_indices must be contiguous int32 [rows, topk]")
    physical_indices = physical_indices.contiguous()
    if data_cache.shape[-1] * 2 != head_dim:
        raise ValueError("NVFP4 data-cache width does not match head_dim")
    if scale_cache.shape[-1] * 16 != head_dim:
        raise ValueError("NVFP4 scale-cache width does not match head_dim")
    total_rows = physical_indices.numel()
    padded_rows = ((total_rows + page_size - 1) // page_size) * page_size
    if output is None:
        output = torch.empty(
            (padded_rows, 1, head_dim),
            dtype=torch.float8_e4m3fn,
            device=data_cache.device,
        )
    else:
        if (
            output.dtype != torch.float8_e4m3fn
            or output.device != data_cache.device
            or output.ndim != 3
            or output.shape[0] < padded_rows
            or output.shape[1:] != (1, head_dim)
        ):
            raise ValueError("output scratch has incompatible shape, dtype, or device")
        output = output[:padded_rows]
    if compact_indices is None:
        compact_indices = torch.empty_like(physical_indices, dtype=torch.int32)
    else:
        if (
            compact_indices.dtype != torch.int32
            or compact_indices.device != physical_indices.device
            or compact_indices.numel() < total_rows
        ):
            raise ValueError(
                "compact_indices scratch has incompatible dtype or capacity"
            )
        compact_indices = compact_indices.reshape(-1)[:total_rows].view_as(
            physical_indices
        )

    if not _cuda_gather_failed and head_dim == 576:
        try:
            _jit_nvfp4_mla_gather_module().gather(
                data_cache,
                scale_cache,
                physical_indices,
                output,
                compact_indices,
                global_scale,
            )
            return output.view(-1, 1, page_size, head_dim), compact_indices
        # Preserve the Triton implementation as a deployment fallback when a
        # wheel/cache does not contain a usable JIT toolchain for this kernel.
        except Exception as error:  # noqa: BLE001
            _cuda_gather_failed = True
            logger.warning(
                "Falling back to the Triton NVFP4 MLA gather after the optimized "
                "CUDA kernel failed to load or launch: %s",
                error,
            )

    block_dim = 256
    _gather_dequant_nvfp4_mla_generation_kernel[
        (total_rows, triton.cdiv(head_dim, block_dim))
    ](
        data_cache,
        scale_cache.view(torch.float8_e4m3fn),
        physical_indices,
        output,
        compact_indices,
        global_scale,
        data_row_stride=data_cache.stride(0),
        scale_row_stride=scale_cache.stride(0),
        head_dim=head_dim,
        block_dim=block_dim,
        num_warps=4,
    )
    return output.view(-1, 1, page_size, head_dim), compact_indices


def gather_dequant_nvfp4_mla_cache(
    data_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    physical_indices: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    head_dim: int,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather selected NVFP4 rows into a compact FP8 paged cache.

    Args:
        data_cache: ``[num_slots, 1, head_dim / 2]`` packed E2M1 bytes.
        scale_cache: ``[num_slots, 1, head_dim / 16]`` E4M3 scale bytes.
        physical_indices: ``[num_queries, topk]`` physical token slots; ``-1``
            marks padding.
        global_scale: one FP32 dequantization scale for this layer.
        head_dim: logical MLA row width (normally 576).
        page_size: tokens per TRTLLM-GEN KV page (normally 64).

    Returns:
        ``compact_kv`` with shape ``[num_pages, 1, page_size, head_dim]`` and
        dtype FP8 E4M3, plus remapped ``compact_indices`` with the same shape as
        ``physical_indices`` and dtype int32.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "The basic NVFP4 DSA gather uses dynamic eager allocations and cannot "
            "run during CUDA Graph capture yet."
        )
    if physical_indices.ndim != 2:
        raise ValueError(
            f"physical_indices must be 2-D, got shape={tuple(physical_indices.shape)}"
        )
    if data_cache.shape[-1] * 2 != head_dim:
        raise ValueError(
            f"packed NVFP4 row has logical width {data_cache.shape[-1] * 2}, "
            f"expected {head_dim}"
        )
    if scale_cache.shape[-1] * 16 != head_dim:
        raise ValueError(
            f"NVFP4 scale row covers {scale_cache.shape[-1] * 16} values, "
            f"expected {head_dim}"
        )

    flat = physical_indices.reshape(-1)
    valid = flat >= 0
    selected = flat[valid].to(torch.long)
    compact_indices = torch.full_like(physical_indices, -1, dtype=torch.int32)

    if selected.numel() == 0:
        compact_rows = torch.zeros(
            (page_size, 1, head_dim),
            dtype=torch.float8_e4m3fn,
            device=data_cache.device,
        )
        return compact_rows.view(1, 1, page_size, head_dim), compact_indices

    # Context rows frequently share selected tokens.  Deduplicate globally so
    # each physical cache row is dequantized once, then use inverse as the new
    # page-size-1 token index consumed by sparse TRTLLM-GEN MLA.
    unique_slots, inverse = torch.unique(selected, sorted=False, return_inverse=True)
    packed = data_cache.index_select(0, unique_slots).contiguous()
    scales = scale_cache.index_select(0, unique_slots).contiguous()

    from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil

    compact_rows = NVFP4KVQuantizeUtil.dequantize(
        packed,
        scales.view(torch.float8_e4m3fn),
        global_scale,
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)

    num_rows = compact_rows.shape[0]
    padded_rows = ((num_rows + page_size - 1) // page_size) * page_size
    if padded_rows != num_rows:
        padding = torch.zeros(
            (padded_rows - num_rows, 1, head_dim),
            dtype=compact_rows.dtype,
            device=compact_rows.device,
        )
        compact_rows = torch.cat((compact_rows, padding), dim=0)

    compact_indices.view(-1)[valid] = inverse.to(torch.int32)
    compact_kv = compact_rows.view(-1, 1, page_size, head_dim)
    return compact_kv, compact_indices
