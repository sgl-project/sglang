"""Native compact NVFP4-to-FP8 gather/dequant for MSA-v1 decode."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_dequant_module() -> Module:
    major, minor = torch.cuda.get_device_capability()
    if major != 10:
        raise RuntimeError(
            f"MSA-v1 compact NVFP4 dequant requires SM100/SM103, got sm{major}{minor}"
        )
    return load_jit(
        "msa_v1_compact_nvfp4_dequant",
        cuda_files=["minimax/msa_v1/attn/dequant.cuh"],
        cuda_wrappers=[
            (
                "gather_dequant_pair",
                "msa_v1::attn::compact_q8k8::dequant::gather_dequant_pair",
            ),
            (
                "full_page_dequant",
                "msa_v1::attn::compact_q8k8::dequant::full_page_dequant",
            ),
        ],
    )


def gather_dequant_nvfp4_topk_pages_pair(
    packed_k: torch.Tensor,
    packed_v: torch.Tensor,
    scale_k: torch.Tensor,
    scale_v: torch.Tensor,
    global_scale: torch.Tensor,
    page_table: torch.Tensor,
    topk_qht: torch.Tensor,
    query_seq_lens: torch.Tensor,
    output_k: torch.Tensor,
    output_v: torch.Tensor,
    *,
    q_len_per_req: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather K/V TopK pages and dequantize both caches in one CUDA launch."""
    if packed_k.dtype != torch.uint8 or packed_k.ndim != 4:
        raise ValueError("packed K/V must be uint8 [pages, kv_heads, 128, 64]")
    if tuple(packed_k.shape) != tuple(packed_v.shape) or packed_v.dtype != torch.uint8:
        raise ValueError("packed K/V must have identical uint8 shapes")
    pages, num_kv_heads, page_size, packed_dim = packed_k.shape
    if (page_size, packed_dim) != (128, 64):
        raise ValueError("compact NVFP4 dequant requires page_size=head_dim=128")
    expected_scale = (pages, num_kv_heads, 128, 8)
    if (
        scale_k.dtype != torch.float8_e4m3fn
        or scale_v.dtype != torch.float8_e4m3fn
        or tuple(scale_k.shape) != expected_scale
        or tuple(scale_v.shape) != expected_scale
    ):
        raise ValueError(f"K/V scales must be FP8 with shape {expected_scale}")
    if page_table.dtype != torch.int32 or page_table.ndim != 2:
        raise ValueError("page_table must be int32 [batch, max_pages]")
    if topk_qht.dtype != torch.int32 or topk_qht.ndim != 3:
        raise ValueError("topk_qht must be int32 [query_rows, kv_heads, 16]")
    query_rows = topk_qht.shape[0]
    expected_topk = (query_rows, num_kv_heads, 16)
    expected_output = (query_rows, num_kv_heads, 16, 128, 128)
    if tuple(topk_qht.shape) != expected_topk:
        raise ValueError(f"topk_qht must have shape {expected_topk}")
    if (
        output_k.dtype != torch.float8_e4m3fn
        or output_v.dtype != torch.float8_e4m3fn
        or tuple(output_k.shape) != expected_output
        or tuple(output_v.shape) != expected_output
    ):
        raise ValueError(f"output K/V must be FP8 with shape {expected_output}")
    if query_seq_lens.dtype != torch.int32 or tuple(query_seq_lens.shape) != (
        query_rows,
    ):
        raise ValueError("query_seq_lens must be int32 [query_rows]")
    if global_scale.dtype != torch.float32 or tuple(global_scale.shape) != (1,):
        raise ValueError("global_scale must be FP32 [1]")
    if q_len_per_req <= 0 or page_table.shape[0] * q_len_per_req != query_rows:
        raise ValueError("batch * q_len_per_req must equal query_rows")
    tensors = (
        packed_k,
        packed_v,
        scale_k,
        scale_v,
        global_scale,
        page_table,
        topk_qht,
        query_seq_lens,
        output_k,
        output_v,
    )
    if any(t.device != packed_k.device for t in tensors):
        raise ValueError("all compact dequant tensors must share one device")
    # Cache pages may sit inside wider per-page rows; the kernel checks their inner strides.
    if not all(t.is_contiguous() for t in tensors[4:]):
        raise ValueError("compact dequant metadata and outputs must be contiguous")

    _jit_dequant_module().gather_dequant_pair(
        packed_k,
        packed_v,
        scale_k.view(torch.uint8),
        scale_v.view(torch.uint8),
        global_scale,
        page_table,
        topk_qht,
        query_seq_lens,
        output_k.view(torch.uint8),
        output_v.view(torch.uint8),
        q_len_per_req,
    )
    return output_k, output_v


def dequant_nvfp4_page_table(
    packed_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    global_scale: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Dequantize active index pages into a physical-page-aligned FP8 mirror."""
    if packed_cache.dtype != torch.uint8 or packed_cache.ndim != 3:
        raise ValueError("packed index cache must be uint8 [pages, 128, 64]")
    pages, page_size, packed_dim = packed_cache.shape
    if (page_size, packed_dim) != (128, 64):
        raise ValueError("index dequant requires page_size=head_dim=128")
    expected_scale = (pages, 128, 8)
    if (
        scale_cache.dtype != torch.float8_e4m3fn
        or tuple(scale_cache.shape) != expected_scale
    ):
        raise ValueError(f"index scales must be FP8 with shape {expected_scale}")
    if out.dtype != torch.float8_e4m3fn or tuple(out.shape) != (pages, 128, 128):
        raise ValueError(f"index output must be FP8 with shape {(pages, 128, 128)}")
    if page_table.dtype != torch.int32 or page_table.ndim != 2:
        raise ValueError("page_table must be int32 [batch, max_pages]")
    if seq_lens.dtype != torch.int32 or tuple(seq_lens.shape) != (page_table.shape[0],):
        raise ValueError("seq_lens must be int32 [batch]")
    if global_scale.dtype != torch.float32 or tuple(global_scale.shape) != (1,):
        raise ValueError("global_scale must be FP32 [1]")
    tensors = (packed_cache, scale_cache, global_scale, page_table, seq_lens, out)
    if any(t.device != packed_cache.device for t in tensors):
        raise ValueError("all index dequant tensors must share one device")
    # Cache pages may sit inside wider per-page rows; the kernel checks their inner strides.
    if not all(t.is_contiguous() for t in tensors[2:]):
        raise ValueError("index dequant metadata and output must be contiguous")

    _jit_dequant_module().full_page_dequant(
        packed_cache,
        scale_cache.view(torch.uint8),
        global_scale,
        page_table,
        seq_lens,
        out.view(torch.uint8),
    )
    return out


__all__ = [
    "dequant_nvfp4_page_table",
    "gather_dequant_nvfp4_topk_pages_pair",
]
