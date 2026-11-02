"""SHUFFLE 5D KV pool helpers for the AITER attention backend.

This module hosts the attention pathways that are specific to the
``SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d`` (SHUFFLE 5D) physical layout.
They live here rather than inline in
:mod:`sglang.srt.layers.attention.aiter_backend` so the main backend
file keeps focused on the legacy NHD path and on dispatch wiring.

Each entry point takes the :class:`AiterAttnBackend` instance as its
first argument so it can reach the shared per-step metadata
(``forward_metadata``, ``qo_indptr``, ``input_dtype``, …) without
needing to be a method on the class.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Callable

import torch
import triton
import triton.language as tl

from .flydsl_pa import PagedAttention as flypa

try:
    # `mha_batch_prefill_func` is re-exported at the aiter top level via
    # `aiter/__init__.py` (`from .ops.mha import *`). Note: a bare
    # `from aiter.mha import ...` does NOT work — that module path only
    # exists as `aiter.ops.mha`.
    from aiter import (
        flash_attn_varlen_func,
        fmha_v3_fwd,
        fmha_v3_varlen_fwd,
        mha_batch_prefill_func,
    )
    from aiter.ops.triton.gluon.pa_decode_gluon import (
        get_recommended_splits,
        pa_decode_gluon,
    )
except ImportError:  # pragma: no cover - import-time guard mirrors aiter_backend
    flash_attn_varlen_func = None
    fmha_v3_fwd = None
    fmha_v3_varlen_fwd = None
    mha_batch_prefill_func = None
    pa_decode_gluon = None
    get_recommended_splits = None

try:
    from aiter import fmha_fwd_bf16_opus_fwd, fmha_fwd_bf16_opus_varlen_fwd
except ImportError:  # Keep compatibility with AITER builds predating OPUS varlen.
    fmha_fwd_bf16_opus_fwd = None
    fmha_fwd_bf16_opus_varlen_fwd = None

from sglang.srt.layers.attention.utils import launch_gather_shuffle_5d_to_linear
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype, scaled_fp8_quant
from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


FLYDSL_MIMO_QUERY_LENGTHS = (1, 4, 8)
FLYDSL_MIMO_QUERY_HEADS = 16
FLYDSL_MIMO_KV_HEADS = 1
FLYDSL_MIMO_HEAD_DIM = 192
FLYDSL_MIMO_VALUE_HEAD_DIM = 128
FLYDSL_MIMO_PAGE_SIZE = 64
FLYDSL_MIMO_DEFAULT_NUM_PARTITIONS = 8
FLYDSL_MIMO_SUPPORTED_NUM_PARTITIONS = (8, 16, 24, 32)
FLYDSL_MIMO_NUM_PARTITIONS_ENV = "SGLANG_FLYDSL_PA_NUM_PARTITIONS"
FLYDSL_MIMO_PREFILL_ENV = "SGLANG_FLYDSL_MIMO_PREFILL"
FLYPA_MIMO_PREFILL_ENV = "SGLANG_FLYPA_MIMO_PREFILL"

CK_MIMO_PREFILL_QUERY_HEADS = 16
CK_MIMO_PREFILL_KV_HEADS = 1
CK_MIMO_PREFILL_HEAD_DIM = 192
CK_MIMO_PREFILL_VALUE_HEAD_DIM = 128
CK_MIMO_PREFILL_PAGE_SIZE = 64
MIMO_BF16_KV_CACHE_INNER_PACK_ELEMS = 8

MIMO_FRESH_BF16_ASM_ENV = "SGLANG_AITER_MIMO_FRESH_BF16_ASM"
MIMO_FRESH_BF16_ASM_ENABLED = get_bool_env_var(MIMO_FRESH_BF16_ASM_ENV, "false")
MIMO_FRESH_BF16_ASM_VARLEN_ENV = "SGLANG_AITER_MIMO_FRESH_BF16_ASM_VARLEN"
MIMO_FRESH_BF16_ASM_VARLEN_ENABLED = get_bool_env_var(
    MIMO_FRESH_BF16_ASM_VARLEN_ENV, "false"
)
MIMO_FRESH_BF16_ASM_V_HEAD_DIM = 128
MIMO_FRESH_BF16_SWA_VARLEN_ENV = "SGLANG_AITER_MIMO_FRESH_BF16_SWA_VARLEN"
MIMO_FRESH_BF16_SWA_VARLEN_ENABLED = get_bool_env_var(
    MIMO_FRESH_BF16_SWA_VARLEN_ENV, "false"
)
MIMO_FRESH_BF16_SWA_WINDOW_SIZE = 128
MIMO_CHUNK_BF16_OPUS_SPLIT_MIN_Q = 32768
MIMO_CHUNK_BF16_OPUS_SPLIT_MIN_KV = 262144
DFLASH_SWA_IMPL_ENV = "SGLANG_AITER_DFLASH_SWA_IMPL"
TARGET_VERIFY_SWA_IMPL_ENV = "SGLANG_AITER_TARGET_VERIFY_SWA_IMPL"


@triton.jit
def _flatten_page_table_to_csr_kernel(
    page_table_ptr,
    page_indptr_ptr,
    page_indices_ptr,
    page_table_starts_ptr,
    page_table_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """Pack selected rows from a rectangular physical page table into CSR."""
    pid = tl.program_id(0)
    block_id = tl.program_id(1)
    page_start = tl.load(page_indptr_ptr + pid).to(tl.int64)
    num_pages = tl.load(page_indptr_ptr + pid + 1).to(tl.int64) - page_start
    source_start = tl.load(page_table_starts_ptr + pid).to(tl.int64)
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if block_id * BLOCK_SIZE >= num_pages:
        return
    mask = offsets < num_pages
    pages = tl.load(
        page_table_ptr + pid.to(tl.int64) * page_table_stride + source_start + offsets,
        mask=mask,
        other=0,
    )
    tl.store(page_indices_ptr + page_start + offsets, pages, mask=mask)


@lru_cache(maxsize=1)
def is_gfx950() -> bool:
    if not torch.version.hip or not torch.cuda.is_available():
        return False
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    return arch == "gfx950"


@lru_cache(maxsize=1)
def is_gfx942() -> bool:
    if not torch.version.hip or not torch.cuda.is_available():
        return False
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    return arch == "gfx942"


def is_mimo_flypa_arch() -> bool:
    """FlyPA prefill is the gfx942 path; gfx950 is also accepted."""
    return is_gfx942() or is_gfx950()


def flydsl_dflash_swa_enabled() -> bool:
    return os.environ.get(DFLASH_SWA_IMPL_ENV, "gluon").strip().lower() == "flydsl"


def flydsl_target_verify_swa_enabled() -> bool:
    return (
        os.environ.get(TARGET_VERIFY_SWA_IMPL_ENV, "gluon").strip().lower() == "flydsl"
    )


def _prepare_flydsl_swa(
    backend: AiterAttnBackend,
    max_batch_size: int,
    query_length: int,
    window_size: int,
    qk_head_dim: int,
    env_name: str,
) -> None:
    """Allocate graph-stable CSR metadata for a DFlash FlyPA SWA path."""
    if backend.page_size != FLYDSL_MIMO_PAGE_SIZE:
        raise ValueError(
            f"{env_name}=flydsl requires page size "
            f"{FLYDSL_MIMO_PAGE_SIZE}, got {backend.page_size}"
        )
    if backend.kv_cache_dtype != torch.bfloat16:
        raise ValueError(f"{env_name}=flydsl requires a BF16 KV cache")

    shape = (
        backend.num_head,
        backend.num_kv_head,
        backend.head_dim,
        backend.v_head_dim,
    )
    expected = (
        FLYDSL_MIMO_QUERY_HEADS,
        FLYDSL_MIMO_KV_HEADS,
        qk_head_dim,
        FLYDSL_MIMO_VALUE_HEAD_DIM,
    )
    supported_shapes = [expected]
    if getattr(backend, "is_draft_worker", False) and qk_head_dim == 128:
        supported_shapes.append((16, 2, 128, 128))
    if shape not in supported_shapes:
        raise ValueError(
            f"{env_name}=flydsl requires per-rank Q/KV/QK/V in "
            f"{supported_shapes}, got {shape}"
        )
    if query_length not in FLYDSL_MIMO_QUERY_LENGTHS:
        raise ValueError(
            f"{env_name}=flydsl requires query length in "
            f"{FLYDSL_MIMO_QUERY_LENGTHS}, got {query_length}"
        )
    if window_size <= 0:
        raise ValueError(f"{env_name}=flydsl requires a positive sliding-window size")

    # Retain one alignment page before the logical window and the current
    # verification block. This bounds the CSR metadata independently of the
    # full target context length.
    max_kv_len = window_size + backend.page_size - 1 + query_length
    max_pages_per_sequence = (max_kv_len + backend.page_size - 1) // backend.page_size
    device = torch.device(backend.device)
    signature = (
        max_batch_size,
        query_length,
        window_size,
        qk_head_dim,
        backend.page_size,
        device,
    )
    if getattr(backend, "_flydsl_dflash_swa_signature", None) == signature:
        return
    if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("DFlash FlyPA metadata cannot grow during graph capture")

    backend._flydsl_dflash_swa_context_lens = torch.empty(
        max_batch_size, dtype=torch.int32, device=device
    )
    backend._flydsl_dflash_swa_page_lens = torch.empty(
        max_batch_size, dtype=torch.int32, device=device
    )
    backend._flydsl_dflash_swa_page_starts = torch.empty(
        max_batch_size, dtype=torch.int32, device=device
    )
    backend._flydsl_dflash_swa_kv_indptr = torch.empty(
        max_batch_size + 1, dtype=torch.int32, device=device
    )
    backend._flydsl_dflash_swa_kv_indices = torch.empty(
        max_batch_size * max_pages_per_sequence,
        dtype=torch.int32,
        device=device,
    )
    backend._flydsl_dflash_swa_last_page_lens = torch.empty(
        max_batch_size, dtype=torch.int32, device=device
    )
    backend._flydsl_dflash_swa_one = torch.ones(1, dtype=torch.float32, device=device)
    backend._flydsl_dflash_swa_max_pages = max_pages_per_sequence
    backend._flydsl_dflash_swa_max_kv_len = max_kv_len
    backend._flydsl_dflash_swa_signature = signature


def prepare_flydsl_dflash_swa(
    backend: AiterAttnBackend,
    max_batch_size: int,
    query_length: int,
    draft_window_size: int,
) -> None:
    if not flydsl_dflash_swa_enabled() or not backend.is_draft_worker:
        return
    _prepare_flydsl_swa(
        backend,
        max_batch_size,
        query_length,
        draft_window_size,
        128,
        DFLASH_SWA_IMPL_ENV,
    )
    # Flash TP4 benefits from sharing the window across its GQA8 query rows.
    # The one-KV-head draft path keeps its existing single-wave kernel.
    if backend.num_kv_head == 2:
        signature = (max_batch_size, query_length, torch.device(backend.device))
        if getattr(backend, "_flydsl_draft_pa_signature", None) != signature:
            if (
                torch.device(backend.device).type == "cuda"
                and torch.cuda.is_current_stream_capturing()
            ):
                raise RuntimeError(
                    "DFlash PA workspaces cannot grow during graph capture"
                )
            parts = 8
            scalars = max_batch_size * parts * query_length * backend.num_head
            backend._flydsl_draft_pa_context = torch.empty(
                max_batch_size, dtype=torch.int32, device=backend.device
            )
            backend._flydsl_draft_pa_max = torch.empty(
                scalars, dtype=torch.float32, device=backend.device
            )
            backend._flydsl_draft_pa_sum = torch.empty_like(
                backend._flydsl_draft_pa_max
            )
            backend._flydsl_draft_pa_out = torch.empty(
                scalars * 128, dtype=torch.bfloat16, device=backend.device
            )
            backend._flydsl_draft_pa_parts = parts
            backend._flydsl_draft_pa_signature = signature
        backend._flydsl_draft_pa_enabled = True


def prepare_flydsl_target_verify_swa(
    backend: AiterAttnBackend,
    max_batch_size: int,
    query_length: int,
    window_size: int,
) -> None:
    if (
        not flydsl_target_verify_swa_enabled()
        or backend.is_draft_worker
        or not backend.use_sliding_window_kv_pool
    ):
        return
    _prepare_flydsl_swa(
        backend,
        max_batch_size,
        query_length,
        window_size,
        FLYDSL_MIMO_HEAD_DIM,
        TARGET_VERIFY_SWA_IMPL_ENV,
    )


def _use_flydsl_swa(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    layer: RadixAttention,
    sinks,
) -> bool:
    return bool(
        (
            getattr(backend, "_flydsl_dflash_swa_enabled", False)
            or getattr(backend, "_flydsl_target_verify_swa_enabled", False)
        )
        and layer.sliding_window_size is not None
        and layer.sliding_window_size >= 0
        and isinstance(sinks, torch.Tensor)
        and sinks.dtype == torch.float32
        and sinks.device == q.device
        and sinks.is_contiguous()
        and backend.logits_soft_cap == 0.0
        and backend.page_size == FLYDSL_MIMO_PAGE_SIZE
        and backend.kv_cache_dtype == torch.bfloat16
        and q.dtype == torch.bfloat16
        and layer.tp_q_head_num == FLYDSL_MIMO_QUERY_HEADS
        and layer.tp_k_head_num in (FLYDSL_MIMO_KV_HEADS, 2)
        and layer.qk_head_dim in (128, FLYDSL_MIMO_HEAD_DIM)
        and layer.v_head_dim == FLYDSL_MIMO_VALUE_HEAD_DIM
    )


def prepare_flydsl_swa_forward_metadata(
    backend: AiterAttnBackend,
    batch_size: int,
    seq_lens: torch.Tensor,
    *,
    context_includes_query: bool = False,
) -> None:
    """Build the shared FlyPA SWA metadata once for the current forward batch."""
    if not (
        getattr(backend, "_flydsl_dflash_swa_enabled", False)
        or getattr(backend, "_flydsl_target_verify_swa_enabled", False)
    ):
        return

    metadata = backend.forward_metadata
    query_length = int(metadata.max_q_len)
    batch_size = int(batch_size)
    if getattr(backend, "_flydsl_draft_pa_enabled", False):
        torch.add(
            seq_lens[:batch_size],
            0 if context_includes_query else query_length,
            out=backend._flydsl_draft_pa_context[:batch_size],
        )
        return
    block_tables = (
        metadata.swa_page_table
        if metadata.swa_page_table is not None
        else metadata.kv_indices
    )
    if block_tables is None or block_tables.ndim != 2:
        raise ValueError("DFlash FlyPA requires a rectangular physical page table")
    if block_tables.dtype != torch.int32:
        raise ValueError(
            "DFlash FlyPA requires int32 page-table entries; "
            f"got {block_tables.dtype}"
        )
    if seq_lens.shape != (batch_size,):
        raise ValueError(
            "DFlash FlyPA sequence-length shape mismatch: expected "
            f"{(batch_size,)}, got {tuple(seq_lens.shape)}"
        )

    context_lens = backend._flydsl_dflash_swa_context_lens[:batch_size]
    torch.add(
        seq_lens[:batch_size],
        0 if context_includes_query else query_length,
        out=context_lens,
    )

    page_lens = backend._flydsl_dflash_swa_page_lens[:batch_size]
    page_lens.copy_(context_lens)
    page_lens.add_(backend.page_size - 1)
    torch.div(page_lens, backend.page_size, rounding_mode="floor", out=page_lens)
    max_pages = int(backend._flydsl_dflash_swa_max_pages)

    # Target SWA metadata can cover the entire context, whereas the draft
    # table can already be compact. Select only the bounded tail in either case.
    page_starts = backend._flydsl_dflash_swa_page_starts[:batch_size]
    page_starts.copy_(page_lens)
    page_starts.sub_(max_pages)
    page_starts.clamp_(min=0)
    page_starts.masked_fill_(page_lens > block_tables.shape[1], 0)
    page_lens.clamp_(max=min(max_pages, block_tables.shape[1]))

    page_indptr = backend._flydsl_dflash_swa_kv_indptr[: batch_size + 1]
    page_indptr[0].zero_()
    torch.cumsum(page_lens, dim=0, out=page_indptr[1:])

    last_page_lens = backend._flydsl_dflash_swa_last_page_lens[:batch_size]
    last_page_lens.copy_(context_lens)
    last_page_lens.add_(-1)
    torch.remainder(last_page_lens, backend.page_size, out=last_page_lens)
    last_page_lens.add_(1)

    # FlyPA receives lengths relative to the compacted page list.
    context_lens.copy_(page_lens)
    context_lens.add_(-1)
    context_lens.mul_(backend.page_size)
    context_lens.add_(last_page_lens)

    # max_pages is workspace capacity, not a minimum input width. The clamped
    # page counts keep the masked gather within shorter eager-decode tables.
    page_indices = backend._flydsl_dflash_swa_kv_indices
    block_size = 32
    _flatten_page_table_to_csr_kernel[(batch_size, triton.cdiv(max_pages, block_size))](
        block_tables,
        page_indptr,
        page_indices,
        page_starts,
        block_tables.stride(0),
        BLOCK_SIZE=block_size,
    )

    metadata.flydsl_swa_context_lens = context_lens
    metadata.flydsl_swa_kv_indptr = page_indptr
    metadata.flydsl_swa_kv_indices = page_indices
    metadata.flydsl_swa_last_page_lens = last_page_lens


def forward_target_verify_flydsl_swa_5d(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    output: torch.Tensor,
    sinks: torch.Tensor,
) -> None:
    """Run DFlash verification directly on the BF16 SHUFFLE-5D cache."""
    from aiter.ops.flydsl import flydsl_paged_attention_swa_bf16

    query_length = int(backend.forward_metadata.max_q_len)
    batch_size = int(forward_batch.batch_size)
    if q.shape[0] != batch_size * query_length:
        raise ValueError(
            "DFlash FlyPA requires uniform verification blocks; "
            f"got q rows={q.shape[0]}, batch={batch_size}, qlen={query_length}"
        )

    metadata = backend.forward_metadata
    if getattr(backend, "_flydsl_draft_pa_enabled", False):
        from aiter.ops.flydsl.pa_decode import pa_decode_tile

        parts = backend._flydsl_draft_pa_parts
        shape = (
            batch_size,
            layer.tp_k_head_num,
            parts,
            query_length * layer.tp_q_head_num // layer.tp_k_head_num,
        )
        count = batch_size * parts * query_length * layer.tp_q_head_num
        pa_decode_tile(
            output.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k_cache,
            v_cache,
            metadata.kv_indices,
            backend._flydsl_draft_pa_context[:batch_size],
            None,
            None,
            softmax_scale=layer.scaling,
            num_partitions=parts,
            pmax=backend._flydsl_draft_pa_max[:count].view(shape),
            psum=backend._flydsl_draft_pa_sum[:count].view(shape),
            pout=backend._flydsl_draft_pa_out[: count * 128].view(*shape, 128),
            sliding_window=int(layer.sliding_window_size),
            sinks=sinks,
        )
        return
    context_lens = getattr(metadata, "flydsl_swa_context_lens", None)
    page_indptr = getattr(metadata, "flydsl_swa_kv_indptr", None)
    page_indices = getattr(metadata, "flydsl_swa_kv_indices", None)
    last_page_lens = getattr(metadata, "flydsl_swa_last_page_lens", None)
    if any(
        value is None
        for value in (context_lens, page_indptr, page_indices, last_page_lens)
    ):
        raise RuntimeError(
            "DFlash FlyPA SWA metadata was not prepared before the layer forward"
        )

    one = backend._flydsl_dflash_swa_one
    flydsl_paged_attention_swa_bf16(
        q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
        k_cache,
        v_cache,
        metadata.qo_indptr[: batch_size + 1],
        page_indptr,
        page_indices,
        query_length,
        int(backend._flydsl_dflash_swa_max_kv_len),
        # Flash's two-head target/draft fallback keeps WINDOW keys including self.
        # Preserve the existing one-head FlyPA window convention.
        window_left=int(layer.sliding_window_size) - int(layer.tp_k_head_num == 2),
        kv_last_page_lens=last_page_lens,
        q_descale=one,
        k_descale=one,
        v_descale=one,
        out=output.view(-1, layer.tp_q_head_num, layer.v_head_dim),
        sink_ptr=sinks,
        softmax_scale=layer.scaling,
    )


def _is_scalar_f32_scale(scale) -> bool:
    return (
        isinstance(scale, torch.Tensor)
        and scale.dtype == torch.float32
        and scale.numel() == 1
    )


def _mimo_logical_qkv_views(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    forward_batch: ForwardBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Return live-token Q/K/V views and the physical query row count.

    TBO pads token-major tensors to the attention-TP width while keeping
    ``extend_seq_lens_cpu`` logical. MiMo's ASM kernels consume only the live
    prefix; downstream TBO stages still require an output with the original
    physical row count.
    """
    lengths = getattr(forward_batch, "extend_seq_lens_cpu", None)
    physical_tokens = q.shape[0]
    if lengths is None:
        return q, k, v, physical_tokens

    logical_tokens = sum(int(length) for length in lengths)
    if logical_tokens < 0:
        raise ValueError(f"MiMo extend token count must be non-negative: {lengths}")
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.shape[0] < logical_tokens:
            raise ValueError(
                f"MiMo {name} has {tensor.shape[0]} physical rows but metadata "
                f"requires {logical_tokens} live rows"
            )

    return (
        q[:logical_tokens],
        k[:logical_tokens],
        v[:logical_tokens],
        physical_tokens,
    )


def _allocate_mimo_asm_output(
    q: torch.Tensor,
    logical_tokens: int,
    output_tokens: int,
    num_heads: int,
    value_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate a physical TBO output and return its live-token prefix."""
    if output_tokens < logical_tokens:
        raise ValueError(
            f"MiMo ASM output has {output_tokens} rows for {logical_tokens} live tokens"
        )
    output = q.new_empty((output_tokens, num_heads, value_head_dim))
    if output_tokens > logical_tokens:
        output[logical_tokens:].zero_()
    return output, output[:logical_tokens]


def _mimo_effective_prefix_lengths(forward_batch: ForwardBatch) -> list[int] | None:
    """Recover logical prefix lengths when chunk-cache accounting reports zero."""
    reported = getattr(forward_batch, "extend_prefix_lens_cpu", None)
    if reported is None:
        return None
    reported = [int(length) for length in reported]

    extend = getattr(forward_batch, "extend_seq_lens_cpu", None)
    seq_lens = getattr(forward_batch, "seq_lens_cpu", None)
    if extend is None or seq_lens is None or len(extend) != len(seq_lens):
        return reported

    derived = [
        int(seq_len) - int(extend_len) for seq_len, extend_len in zip(seq_lens, extend)
    ]
    return derived if all(length >= 0 for length in derived) else reported


def can_use_mimo_fresh_bf16_asm(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    window_size,
    sinks,
) -> bool:
    """Return whether this extend is the qualified gfx950 MiMo ASM contract."""
    lengths = forward_batch.extend_seq_lens_cpu
    uniform_length = (
        lengths is not None
        and len(lengths) > 0
        and lengths[0] > 128
        and all(length == lengths[0] for length in lengths)
    )
    total_tokens = 0 if not uniform_length else len(lengths) * lengths[0]
    return (
        MIMO_FRESH_BF16_ASM_ENABLED
        and fmha_v3_fwd is not None
        and is_gfx950()
        and uniform_length
        and q.shape[0] == total_tokens
        and k.shape[0] == total_tokens
        and v.shape[0] == total_tokens
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and backend.input_dtype == torch.bfloat16
        and layer.tp_q_head_num == CK_MIMO_PREFILL_QUERY_HEADS
        and layer.tp_k_head_num == CK_MIMO_PREFILL_KV_HEADS
        and layer.tp_v_head_num == CK_MIMO_PREFILL_KV_HEADS
        and layer.qk_head_dim == CK_MIMO_PREFILL_HEAD_DIM
        and layer.head_dim == CK_MIMO_PREFILL_HEAD_DIM
        and layer.v_head_dim == CK_MIMO_PREFILL_VALUE_HEAD_DIM
        and q.shape[-1] == layer.tp_q_head_num * layer.head_dim
        and k.shape[-2:] == (layer.tp_k_head_num, layer.qk_head_dim)
        and v.shape[-2:] == (layer.tp_v_head_num, layer.v_head_dim)
        and q.stride(-1) == 1
        and k.stride(-1) == 1
        and v.stride(-1) == 1
        and (layer.sliding_window_size is None or layer.sliding_window_size < 0)
        and tuple(window_size) == (-1, -1)
        and sinks is None
        and float(backend.logits_soft_cap) == 0.0
    )


def mimo_fresh_bf16_asm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    output_token_count: int | None = None,
) -> torch.Tensor:
    """Run gfx950 D192/V128 ASM with a native V128 model/cache ABI."""
    lengths = forward_batch.extend_seq_lens_cpu
    batch_size = len(lengths)
    sequence_length = lengths[0]
    q_4d = q.view(
        batch_size,
        sequence_length,
        layer.tp_q_head_num,
        layer.qk_head_dim,
    )
    k_4d = k.view(
        batch_size,
        sequence_length,
        layer.tp_k_head_num,
        layer.qk_head_dim,
    )
    v_4d = v.view(
        batch_size,
        sequence_length,
        layer.tp_v_head_num,
        layer.v_head_dim,
    )

    logical_tokens = batch_size * sequence_length
    output_storage, output_live = _allocate_mimo_asm_output(
        q,
        logical_tokens,
        logical_tokens if output_token_count is None else output_token_count,
        layer.tp_q_head_num,
        layer.v_head_dim,
    )
    output = output_live.view(
        batch_size,
        sequence_length,
        layer.tp_q_head_num,
        layer.v_head_dim,
    )
    result = fmha_v3_fwd(
        q_4d,
        k_4d,
        v_4d,
        0.0,
        layer.scaling,
        True,
        -1,
        -1,
        False,
        False,
        0,
        output,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    if result[0].data_ptr() != output.data_ptr():
        raise RuntimeError("gfx950 MiMo fresh BF16 ASM ignored its output buffer")
    return output_storage.view(-1, layer.tp_q_head_num * layer.v_head_dim)


def can_use_mimo_fresh_bf16_varlen_asm(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    window_size,
    sinks,
) -> bool:
    """Return whether this fresh ragged extend matches MiMo varlen ASM."""
    lengths = forward_batch.extend_seq_lens_cpu
    valid_lengths = lengths is not None and len(lengths) > 0 and min(lengths) > 128
    total_tokens = 0 if not valid_lengths else sum(lengths)
    return (
        MIMO_FRESH_BF16_ASM_ENABLED
        and MIMO_FRESH_BF16_ASM_VARLEN_ENABLED
        and fmha_v3_varlen_fwd is not None
        and is_gfx950()
        and valid_lengths
        and q.shape[0] == total_tokens
        and k.shape[0] == total_tokens
        and v.shape[0] == total_tokens
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and backend.input_dtype == torch.bfloat16
        and layer.tp_q_head_num == CK_MIMO_PREFILL_QUERY_HEADS
        and layer.tp_k_head_num == CK_MIMO_PREFILL_KV_HEADS
        and layer.tp_v_head_num == CK_MIMO_PREFILL_KV_HEADS
        and layer.qk_head_dim == CK_MIMO_PREFILL_HEAD_DIM
        and layer.head_dim == CK_MIMO_PREFILL_HEAD_DIM
        and layer.v_head_dim == CK_MIMO_PREFILL_VALUE_HEAD_DIM
        and q.shape[-1] == layer.tp_q_head_num * layer.head_dim
        and k.shape[-2:] == (layer.tp_k_head_num, layer.qk_head_dim)
        and v.shape[-2:] == (layer.tp_v_head_num, layer.v_head_dim)
        and q.stride(-1) == 1
        and k.stride(-1) == 1
        and v.stride(-1) == 1
        and (layer.sliding_window_size is None or layer.sliding_window_size < 0)
        and tuple(window_size) == (-1, -1)
        and sinks is None
        and float(backend.logits_soft_cap) == 0.0
    )


def mimo_fresh_bf16_varlen_asm(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    output_token_count: int | None = None,
) -> torch.Tensor:
    """Run gfx950's grouped D192/V128 ASM on a fresh ragged MiMo batch."""
    lengths = forward_batch.extend_seq_lens_cpu
    batch_size = len(lengths)
    max_length = max(lengths)
    min_length = min(lengths)
    q_varlen = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
    k_varlen = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
    v_varlen = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)

    logical_tokens = q.shape[0]
    output_storage, output = _allocate_mimo_asm_output(
        q,
        logical_tokens,
        logical_tokens if output_token_count is None else output_token_count,
        layer.tp_q_head_num,
        layer.v_head_dim,
    )
    cu_seqlens = backend.qo_indptr[: batch_size + 1]
    result = fmha_v3_varlen_fwd(
        q_varlen,
        k_varlen,
        v_varlen,
        cu_seqlens,
        cu_seqlens,
        max_length,
        max_length,
        min_length,
        0.0,
        layer.scaling,
        0.0,
        False,
        True,
        -1,
        -1,
        False,
        False,
        0,
        output,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    if result[0].data_ptr() != output.data_ptr():
        raise RuntimeError(
            "gfx950 MiMo fresh BF16 varlen ASM ignored its output buffer"
        )
    return output_storage.view(-1, layer.tp_q_head_num * layer.v_head_dim)


def can_use_mimo_chunk_bf16_varlen_asm(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    window_size,
    sinks,
    is_swa_layer: bool,
    sub_pool,
    k_buf: torch.Tensor,
    v_buf: torch.Tensor,
    metadata,
) -> bool:
    """Return whether cached ragged prefill can reuse MiMo BF16 varlen ASM."""
    extend_lengths = getattr(forward_batch, "extend_seq_lens_cpu", None)
    prefix_lengths = _mimo_effective_prefix_lengths(forward_batch)
    seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
    valid_ragged_batch = (
        extend_lengths is not None
        and prefix_lengths is not None
        and seq_lens_cpu is not None
        and len(extend_lengths) > 0
        and len(extend_lengths) == len(prefix_lengths) == len(seq_lens_cpu)
    )
    if not valid_ragged_batch:
        return False

    extend_lengths = [int(length) for length in extend_lengths]
    prefix_lengths = [int(length) for length in prefix_lengths]
    seq_lengths = [int(length) for length in seq_lens_cpu]
    total_q = sum(extend_lengths)
    total_kv = int(forward_batch.seq_lens_sum)
    batch_size = len(extend_lengths)
    expected_k_tail = (
        CK_MIMO_PREFILL_KV_HEADS,
        CK_MIMO_PREFILL_HEAD_DIM // MIMO_BF16_KV_CACHE_INNER_PACK_ELEMS,
        CK_MIMO_PREFILL_PAGE_SIZE,
        MIMO_BF16_KV_CACHE_INNER_PACK_ELEMS,
    )
    expected_v_tail = (
        CK_MIMO_PREFILL_KV_HEADS,
        CK_MIMO_PREFILL_PAGE_SIZE // MIMO_BF16_KV_CACHE_INNER_PACK_ELEMS,
        CK_MIMO_PREFILL_VALUE_HEAD_DIM,
        MIMO_BF16_KV_CACHE_INNER_PACK_ELEMS,
    )
    return (
        MIMO_FRESH_BF16_ASM_ENABLED
        and MIMO_FRESH_BF16_ASM_VARLEN_ENABLED
        and fmha_v3_varlen_fwd is not None
        and is_gfx950()
        and not is_swa_layer
        and sinks is None
        and tuple(window_size) == (-1, -1)
        and any(prefix_len > 0 for prefix_len in prefix_lengths)
        and all(prefix_len >= 0 for prefix_len in prefix_lengths)
        and all(extend_len > 0 for extend_len in extend_lengths)
        and max(extend_lengths) > 128
        and all(
            seq_len == prefix_len + extend_len
            for seq_len, prefix_len, extend_len in zip(
                seq_lengths, prefix_lengths, extend_lengths
            )
        )
        and total_kv == sum(seq_lengths)
        and q.shape[0] == total_q
        and k.shape[0] == total_q
        and v.shape[0] == total_q
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and backend.input_dtype == torch.bfloat16
        and backend.kv_cache_dtype != fp8_dtype
        and sub_pool.dtype == torch.bfloat16
        and sub_pool.store_dtype == torch.bfloat16
        and layer.tp_q_head_num == CK_MIMO_PREFILL_QUERY_HEADS
        and layer.tp_k_head_num == CK_MIMO_PREFILL_KV_HEADS
        and layer.tp_v_head_num == CK_MIMO_PREFILL_KV_HEADS
        and layer.qk_head_dim == CK_MIMO_PREFILL_HEAD_DIM
        and layer.head_dim == CK_MIMO_PREFILL_HEAD_DIM
        and layer.v_head_dim == CK_MIMO_PREFILL_VALUE_HEAD_DIM
        and getattr(layer, "mimo_original_v_head_dim", None)
        == MIMO_FRESH_BF16_ASM_V_HEAD_DIM
        and q.shape[-1] == layer.tp_q_head_num * layer.head_dim
        and k.shape[-2:] == (layer.tp_k_head_num, layer.qk_head_dim)
        and v.shape[-2:] == (layer.tp_v_head_num, layer.v_head_dim)
        and q.stride(-1) == 1
        and k.stride(-1) == 1
        and v.stride(-1) == 1
        and float(backend.logits_soft_cap) == 0.0
        and metadata.kv_indices is not None
        and metadata.kv_indptr is not None
        and metadata.max_kv_len is not None
        and metadata.kv_indices.dtype == torch.int32
        and metadata.kv_indptr.dtype == torch.int32
        and backend.qo_indptr.dtype == torch.int32
        and metadata.kv_indices.device == q.device
        and metadata.kv_indptr.device == q.device
        and backend.qo_indptr.device == q.device
        and metadata.kv_indices.numel() >= total_kv
        and metadata.kv_indptr.numel() >= batch_size + 1
        and backend.qo_indptr.numel() >= batch_size + 1
        and int(metadata.max_q_len) == max(extend_lengths)
        and int(metadata.max_kv_len) == max(seq_lengths)
        and k_buf.ndim == 5
        and v_buf.ndim == 5
        and tuple(k_buf.shape[1:]) == expected_k_tail
        and tuple(v_buf.shape[1:]) == expected_v_tail
        and k_buf.shape[0] == v_buf.shape[0]
        and k_buf.element_size() == 2
        and v_buf.element_size() == 2
    )


def mimo_chunk_bf16_varlen_asm(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    k_buf: torch.Tensor,
    v_buf: torch.Tensor,
    metadata,
    output_token_count: int | None = None,
) -> torch.Tensor:
    """Run gfx950 D192/V128 ASM for a cached ragged MiMo prefill batch."""
    extend_lengths = [int(length) for length in forward_batch.extend_seq_lens_cpu]
    seq_lengths = [int(length) for length in forward_batch.seq_lens_cpu]
    batch_size = len(extend_lengths)
    total_q = sum(extend_lengths)
    total_kv = sum(seq_lengths)
    slot_ids = metadata.kv_indices[:total_kv]
    k_full, v_full = launch_gather_shuffle_5d_to_linear(k_buf, v_buf, slot_ids)
    q_varlen = q.contiguous().view(-1, layer.tp_q_head_num, layer.qk_head_dim)
    k_varlen = k_full.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
    v_varlen = v_full.view(-1, layer.tp_v_head_num, layer.v_head_dim)

    output_storage, output = _allocate_mimo_asm_output(
        q,
        total_q,
        total_q if output_token_count is None else output_token_count,
        layer.tp_q_head_num,
        layer.v_head_dim,
    )
    use_opus = (
        fmha_fwd_bf16_opus_varlen_fwd is not None
        and int(os.environ.get("AITER_DISABLE_FMHA_OPUS", "0")) == 0
    )
    if use_opus:
        # A full 64K scheduler budget can pack one long request tail together
        # with the next request's short head. On gfx950, OPUS group mode sizes
        # both groups from the batch maxima and is slower for this long-KV
        # two-sequence shape than two dense launches. Keep the split narrowly
        # gated so ordinary short/ragged batches retain the grouped path.
        use_split_dense_opus = (
            fmha_fwd_bf16_opus_fwd is not None
            and batch_size == 2
            and total_q >= MIMO_CHUNK_BF16_OPUS_SPLIT_MIN_Q
            and max(seq_lengths) >= MIMO_CHUNK_BF16_OPUS_SPLIT_MIN_KV
        )
        if use_split_dense_opus:
            if not getattr(backend, "_logged_mimo_chunk_bf16_split_opus", False):
                logger.info(
                    "Using split dense OPUS gfx950 ASM for two-sequence cached "
                    "MiMo BF16 full attention."
                )
                backend._logged_mimo_chunk_bf16_split_opus = True
            q_start = 0
            kv_start = 0
            for q_len, kv_len in zip(extend_lengths, seq_lengths):
                fmha_fwd_bf16_opus_fwd(
                    q_varlen[q_start : q_start + q_len].view(
                        1, q_len, layer.tp_q_head_num, layer.qk_head_dim
                    ),
                    k_varlen[kv_start : kv_start + kv_len].view(
                        1, kv_len, layer.tp_k_head_num, layer.qk_head_dim
                    ),
                    v_varlen[kv_start : kv_start + kv_len].view(
                        1, kv_len, layer.tp_v_head_num, layer.v_head_dim
                    ),
                    softmax_scale=layer.scaling,
                    causal=True,
                    out=output[q_start : q_start + q_len].view(
                        1, q_len, layer.tp_q_head_num, layer.v_head_dim
                    ),
                )
                q_start += q_len
                kv_start += kv_len
            assert q_start == total_q and kv_start == total_kv
            result_output = output
        else:
            if not getattr(backend, "_logged_mimo_chunk_bf16_opus", False):
                logger.info(
                    "Using OPUS gfx950 varlen ASM for cached MiMo BF16 full "
                    "attention."
                )
                backend._logged_mimo_chunk_bf16_opus = True
            result_output = fmha_fwd_bf16_opus_varlen_fwd(
                q_varlen,
                k_varlen,
                v_varlen,
                softmax_scale=layer.scaling,
                causal=True,
                seqstart_q=backend.qo_indptr[: batch_size + 1],
                seqstart_k=metadata.kv_indptr[: batch_size + 1],
                max_seqlen_q=max(extend_lengths),
                max_seqlen_k=max(seq_lengths),
                out=output,
            )
    else:
        result = fmha_v3_varlen_fwd(
            q_varlen,
            k_varlen,
            v_varlen,
            backend.qo_indptr[: batch_size + 1],
            metadata.kv_indptr[: batch_size + 1],
            max(extend_lengths),
            max(seq_lengths),
            min(extend_lengths),
            0.0,
            layer.scaling,
            0.0,
            False,
            True,
            -1,
            -1,
            False,
            False,
            0,
            output,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        result_output = result[0]
    if result_output.data_ptr() != output.data_ptr():
        raise RuntimeError(
            "gfx950 MiMo cached BF16 full-attention ASM ignored its output buffer"
        )
    return output_storage.view(-1, layer.tp_q_head_num * layer.v_head_dim)


def can_use_mimo_fresh_bf16_swa_varlen(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    window_size,
    sinks,
) -> bool:
    """Return whether this fresh extend matches MiMo's gfx950 SWA contract."""
    lengths = forward_batch.extend_seq_lens_cpu
    valid_lengths = (
        lengths is not None
        and len(lengths) > 0
        and min(lengths) > MIMO_FRESH_BF16_SWA_WINDOW_SIZE
    )
    total_tokens = 0 if not valid_lengths else sum(lengths)
    valid_sinks = (
        isinstance(sinks, torch.Tensor)
        and sinks.device == q.device
        and sinks.shape == (CK_MIMO_PREFILL_QUERY_HEADS,)
        and sinks.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and sinks.stride(-1) == 1
    )
    return (
        MIMO_FRESH_BF16_SWA_VARLEN_ENABLED
        and flash_attn_varlen_func is not None
        and is_gfx950()
        and valid_lengths
        and q.shape[0] == total_tokens
        and k.shape[0] == total_tokens
        and v.shape[0] == total_tokens
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and backend.input_dtype == torch.bfloat16
        and layer.tp_q_head_num == CK_MIMO_PREFILL_QUERY_HEADS
        and layer.tp_k_head_num == CK_MIMO_PREFILL_KV_HEADS
        and layer.tp_v_head_num == CK_MIMO_PREFILL_KV_HEADS
        and layer.qk_head_dim == CK_MIMO_PREFILL_HEAD_DIM
        and layer.head_dim == CK_MIMO_PREFILL_HEAD_DIM
        and layer.v_head_dim == CK_MIMO_PREFILL_VALUE_HEAD_DIM
        and getattr(layer, "mimo_original_v_head_dim", None)
        == MIMO_FRESH_BF16_ASM_V_HEAD_DIM
        and q.shape[-1] == layer.tp_q_head_num * layer.head_dim
        and k.shape[-2:] == (layer.tp_k_head_num, layer.qk_head_dim)
        and v.shape[-2:] == (layer.tp_v_head_num, layer.v_head_dim)
        and q.stride(-1) == 1
        and k.stride(-1) == 1
        and v.stride(-1) == 1
        and layer.sliding_window_size == MIMO_FRESH_BF16_SWA_WINDOW_SIZE
        and tuple(window_size) == (MIMO_FRESH_BF16_SWA_WINDOW_SIZE, -1)
        and valid_sinks
        and float(backend.logits_soft_cap) == 0.0
    )


def mimo_fresh_bf16_swa_varlen(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    window_size,
    sinks: torch.Tensor,
) -> torch.Tensor:
    """Run native-D128 CK varlen SWA with a native V128 ABI."""
    lengths = forward_batch.extend_seq_lens_cpu
    batch_size = len(lengths)
    q_varlen = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
    k_varlen = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
    v_varlen = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)

    output = q.new_empty((q.shape[0], layer.tp_q_head_num, layer.v_head_dim))
    cu_seqlens = backend.qo_indptr[: batch_size + 1]
    sink_ptr = sinks if sinks.dtype == torch.float32 else sinks.float()
    result = flash_attn_varlen_func(
        q_varlen,
        k_varlen,
        v_varlen,
        cu_seqlens,
        cu_seqlens,
        max(lengths),
        max(lengths),
        # This is a cascade-attention exclusion threshold, not the shortest
        # sequence length.  Zero is required to execute every request.
        min_seqlen_q=0,
        softmax_scale=layer.scaling,
        causal=True,
        window_size=(int(window_size[0]), 0, 0),
        sink_ptr=sink_ptr,
        out=output,
    )
    if result.data_ptr() != output.data_ptr():
        raise RuntimeError("gfx950 MiMo fresh SWA varlen ignored its output buffer")
    return output.view(-1, layer.tp_q_head_num * layer.v_head_dim)


def quantize_query_per_tensor_fp8(
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a BF16/FP16 query and return its independent FP8 descale.

    A raw ``q.to(fp8_dtype)`` has an implicit descale of one.  Reusing the KV
    cache's K descale for that tensor changes Q by an unrelated factor and is
    only hidden when every cache scale happens to be 1.0.  The batch-prefill
    FP8 ABI expects ``dequantized_q = q_fp8 * q_descale``, so use SGLang's
    dynamic per-tensor quantizer and forward the scale it actually produced.

    Prefill queries arrive as a two-dimensional token-major view whose outer
    stride may still reference the fused QKV allocation.  The quantization
    kernels consume a dense matrix, and the old dtype conversion also
    materialized one, so make that requirement explicit here.
    """
    if q.ndim != 2:
        raise ValueError(
            "AITER FP8 batch-prefill query quantization requires a 2D "
            f"[tokens, heads * head_dim] tensor; got shape {tuple(q.shape)}"
        )
    if not q.is_floating_point() or q.dtype == fp8_dtype:
        raise ValueError(
            "AITER FP8 batch-prefill query quantization requires a non-FP8 "
            f"floating-point query; got {q.dtype}"
        )

    q_fp8, q_descale = scaled_fp8_quant(
        q.contiguous(),
        scale=None,
        use_per_token_if_dynamic=False,
    )
    if q_descale.numel() != 1 or q_descale.dtype != torch.float32:
        raise RuntimeError(
            "AITER FP8 batch-prefill requires one FP32 Q descale; got "
            f"shape={tuple(q_descale.shape)}, dtype={q_descale.dtype}"
        )
    return q_fp8, q_descale


@dataclass(frozen=True)
class FlyDSLMiMoPrefillKernel:
    run: Callable
    version: str
    runtime_path: str
    kernel_path: str


@lru_cache(maxsize=1)
def load_flydsl_mimo_prefill_kernel() -> FlyDSLMiMoPrefillKernel:
    """Lazily load the optional gfx950 MiMo D192/V128 paged kernel."""

    try:
        flydsl = importlib.import_module("flydsl")
        module = importlib.import_module(
            "kernels.attention.flash_attn_fp8_mimo_paged_gfx950"
        )
    except Exception as exc:
        raise RuntimeError(
            f"{FLYDSL_MIMO_PREFILL_ENV}=1 requires the compatible FlyDSL "
            "runtime and the local FlyDSL repository-root `kernels` package "
            "on PYTHONPATH"
        ) from exc

    kernel = FlyDSLMiMoPrefillKernel(
        run=module.mimo_paged_flash_attn_fp8,
        version=str(getattr(flydsl, "__version__", "unknown")),
        runtime_path=str(getattr(flydsl, "__file__", "unknown")),
        kernel_path=str(getattr(module, "__file__", "unknown")),
    )
    logger.info(
        "Loaded FlyDSL MiMo cached-prefill kernel: runtime=%s version=%s kernel=%s",
        kernel.runtime_path,
        kernel.version,
        kernel.kernel_path,
    )
    return kernel


def get_flydsl_mimo_num_partitions() -> int:
    raw_value = os.getenv(
        FLYDSL_MIMO_NUM_PARTITIONS_ENV,
        str(FLYDSL_MIMO_DEFAULT_NUM_PARTITIONS),
    )
    try:
        num_partitions = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{FLYDSL_MIMO_NUM_PARTITIONS_ENV} must be one of "
            f"{FLYDSL_MIMO_SUPPORTED_NUM_PARTITIONS}; got {raw_value!r}"
        ) from exc
    if num_partitions not in FLYDSL_MIMO_SUPPORTED_NUM_PARTITIONS:
        raise ValueError(
            f"{FLYDSL_MIMO_NUM_PARTITIONS_ENV} must be one of "
            f"{FLYDSL_MIMO_SUPPORTED_NUM_PARTITIONS}; got {num_partitions}"
        )
    return num_partitions


@dataclass(frozen=True)
class FlyDSLPADecodeKernels:
    pa_decode_tile: Callable
    compile_pa_decode_tile: Callable
    compile_pa_decode_reduce: Callable
    kv_dtype_parameter: str
    reduce_uses_runtime_query_length: bool
    version: str
    runtime_path: str
    kernel_path: str


@lru_cache(maxsize=1)
def load_flydsl_pa_decode_kernels() -> FlyDSLPADecodeKernels:
    """Load the optional local FlyDSL page-64 tile implementation.

    FlyDSL is intentionally not imported at module scope: the normal AITER
    Gluon configuration must continue to work without FlyDSL installed. The
    BF16 integration requires the 0.3.2 native runtime and the
    ``mimo_flydsl_kernels`` 0.1.2 compile API.
    """

    try:
        flydsl = importlib.import_module("flydsl")
        try:
            tile_module = importlib.import_module("aiter.ops.flydsl.pa_decode")
            reduce_module = importlib.import_module(
                "aiter.ops.flydsl.kernels.pa_decode_reduce"
            )
            compile_reduce = getattr(
                reduce_module, "compile_pa_decode_ps_reduce", None
            ) or getattr(reduce_module, "compile_pa_decode_sw_reduce")
        except (ImportError, ModuleNotFoundError):
            tile_module = importlib.import_module("kernels.attention.pa_decode_tile")
            reduce_module = importlib.import_module("kernels.attention.pa_decode_swa")
            # The compatibility wrapper imports these helpers during launch.
            importlib.import_module("kernels.attention.pa_decode_fp8")
            compile_reduce = reduce_module.compile_pa_decode_sw_reduce
    except Exception as exc:
        raise RuntimeError(
            "SGLANG_AITER_PA_DECODE_IMPL=flydsl requires a compatible FlyDSL "
            "native runtime and the local FlyDSL repository-root `kernels` "
            "package on PYTHONPATH. The MiMo setup expects the FlyDSL 0.3.2 "
            "runtime plus mimo_flydsl_kernels 0.1.2 or newer."
        ) from exc

    version = str(getattr(flydsl, "__version__", "unknown"))
    if version != "0.3.2":
        raise RuntimeError(
            "MiMo FlyDSL integration requires the validated 0.3.2 "
            f"native runtime; imported version {version!r} from "
            f"{getattr(flydsl, '__file__', 'unknown')}"
        )

    compile_tile = getattr(tile_module, "compile_pa_decode_tile", None)
    try:
        compile_tile_parameters = (
            inspect.signature(compile_tile).parameters
            if compile_tile is not None
            else {}
        )
        supports_native_v = "v_head_dim" in compile_tile_parameters and bool(
            {"kv_dtype", "bf16_kv"} & compile_tile_parameters.keys()
        )
    except (TypeError, ValueError):
        supports_native_v = False
    if not supports_native_v:
        raise RuntimeError(
            "SGLANG_AITER_PA_DECODE_IMPL=flydsl requires "
            "a compatible FlyDSL source whose compile_pa_decode_tile accepts "
            "`v_head_dim` and either `kv_dtype` or `bf16_kv`; imported "
            f"incompatible kernel from {getattr(tile_module, '__file__', 'unknown')}"
        )

    return FlyDSLPADecodeKernels(
        pa_decode_tile=tile_module.pa_decode_tile,
        compile_pa_decode_tile=compile_tile,
        compile_pa_decode_reduce=compile_reduce,
        kv_dtype_parameter=(
            "kv_dtype" if "kv_dtype" in compile_tile_parameters else "bf16_kv"
        ),
        reduce_uses_runtime_query_length=(
            "query_seq_len" not in inspect.signature(compile_reduce).parameters
        ),
        version=version,
        runtime_path=str(getattr(flydsl, "__file__", "unknown")),
        kernel_path=str(getattr(tile_module, "__file__", "unknown")),
    )


def can_build_mimo_paged_kv_metadata(
    kv_cache_is_vectorized_5d: bool,
    page_size: int,
    kv_cache_dtype,
    q_dtype,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> bool:
    """Whether to scatter ragged slot ids into page-64 tables.

    FlyPA and CK (gfx942/gfx950, BF16 or FP8 SHUFFLE-5D) both consume
    ``paged_kv_indptr`` / ``paged_kv_indices`` /
    ``paged_kv_last_page_len``. Without these tables ``can_use_mimo_flypa_prefill``
    always returns False.
    """
    return (
        kv_cache_is_vectorized_5d
        and page_size == CK_MIMO_PREFILL_PAGE_SIZE
        and kv_cache_dtype in (fp8_dtype, torch.bfloat16)
        and q_dtype == torch.bfloat16
        and num_qo_heads == CK_MIMO_PREFILL_QUERY_HEADS
        and num_kv_heads in (CK_MIMO_PREFILL_KV_HEADS, 2)
        and head_dim == CK_MIMO_PREFILL_HEAD_DIM
    )


def can_use_mimo_flypa_prefill(
    backend: AiterAttnBackend,
    layer: RadixAttention,
    window_size,
    sinks,
    is_swa_layer: bool,
    sub_pool,
    k_buf: torch.Tensor,
    v_buf: torch.Tensor,
    metadata,
) -> bool:
    """Return whether FlyPA can consume this cached MiMo 5D extend.

    Unlike CK paged prefill, FlyPA is not FP8-only. gfx950 FP8 prefers the
    FlyDSL paged kernel when that path qualifies; otherwise FlyPA covers
    gfx942/gfx950 BF16 or FP8 SHUFFLE-5D full-attention extends with
    page-64 metadata.
    """
    if not get_bool_env_var(FLYPA_MIMO_PREFILL_ENV, "false"):
        return False
    if not is_mimo_flypa_arch():
        return False
    # gfx950 uses the native AITER paths: BF16 goes through the ASM paths above
    # (or the AITER fallback below), while eligible FP8 uses PR #5556's paged
    # FlyDSL kernel. The local FlyPA kernel remains a gfx942 compatibility path.
    if is_gfx950():
        return False
    if (
        is_swa_layer
        or sinks is not None
        or tuple(window_size) != (-1, -1)
        or backend.input_dtype != torch.bfloat16
        or backend.page_size != CK_MIMO_PREFILL_PAGE_SIZE
        or float(backend.logits_soft_cap) != 0.0
    ):
        return False
    if sub_pool.dtype not in (fp8_dtype, torch.bfloat16):
        return False
    if (
        layer.tp_q_head_num != CK_MIMO_PREFILL_QUERY_HEADS
        or layer.tp_k_head_num != CK_MIMO_PREFILL_KV_HEADS
        or layer.tp_v_head_num != CK_MIMO_PREFILL_KV_HEADS
        or layer.qk_head_dim != CK_MIMO_PREFILL_HEAD_DIM
        or layer.head_dim != CK_MIMO_PREFILL_HEAD_DIM
        or layer.v_head_dim != CK_MIMO_PREFILL_VALUE_HEAD_DIM
    ):
        return False
    if k_buf.ndim != 5 or v_buf.ndim != 5 or k_buf.shape[0] != v_buf.shape[0]:
        return False
    pack = 16 // k_buf.element_size()
    expected_k_tail = (
        CK_MIMO_PREFILL_KV_HEADS,
        CK_MIMO_PREFILL_HEAD_DIM // pack,
        CK_MIMO_PREFILL_PAGE_SIZE,
        pack,
    )
    expected_v_tail = (
        CK_MIMO_PREFILL_KV_HEADS,
        CK_MIMO_PREFILL_PAGE_SIZE // pack,
        CK_MIMO_PREFILL_VALUE_HEAD_DIM,
        pack,
    )
    if tuple(k_buf.shape[1:]) != expected_k_tail:
        return False
    if tuple(v_buf.shape[1:]) != expected_v_tail:
        return False
    if any(
        getattr(metadata, name, None) is None
        for name in (
            "paged_kv_indptr",
            "paged_kv_indices",
            "paged_kv_last_page_len",
        )
    ):
        return False
    if (
        metadata.paged_kv_indptr.dtype != torch.int32
        or metadata.paged_kv_indices.dtype != torch.int32
    ):
        return False
    if sub_pool.dtype == fp8_dtype:
        k_scale = layer.k_scale if layer.k_scale is not None else backend.k_scale
        v_scale = layer.v_scale if layer.v_scale is not None else backend.v_scale
        if not _is_scalar_f32_scale(k_scale) or not _is_scalar_f32_scale(v_scale):
            return False
    return True


def can_use_mimo_flypa_swa_prefill(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    window_size,
    sinks,
    is_swa_layer: bool,
    sub_pool,
    k_buf: torch.Tensor,
    v_buf: torch.Tensor,
    metadata,
) -> bool:
    """Return whether native FlyPA can replace CK for cached BF16 MiMo SWA."""
    lengths = getattr(forward_batch, "extend_seq_lens_cpu", None)
    logical_tokens = 0 if lengths is None else sum(int(length) for length in lengths)
    expected_k_tail = (
        CK_MIMO_PREFILL_KV_HEADS,
        CK_MIMO_PREFILL_HEAD_DIM // MIMO_BF16_KV_CACHE_INNER_PACK_ELEMS,
        CK_MIMO_PREFILL_PAGE_SIZE,
        MIMO_BF16_KV_CACHE_INNER_PACK_ELEMS,
    )
    expected_v_tail = (
        CK_MIMO_PREFILL_KV_HEADS,
        CK_MIMO_PREFILL_PAGE_SIZE // MIMO_BF16_KV_CACHE_INNER_PACK_ELEMS,
        CK_MIMO_PREFILL_VALUE_HEAD_DIM,
        MIMO_BF16_KV_CACHE_INNER_PACK_ELEMS,
    )
    return (
        get_bool_env_var(FLYPA_MIMO_PREFILL_ENV, "false")
        and is_gfx950()
        and is_swa_layer
        and lengths is not None
        and len(lengths) > 0
        and all(int(length) > 0 for length in lengths)
        and q.shape[0] == logical_tokens
        and isinstance(sinks, torch.Tensor)
        and sinks.device == q.device
        and sinks.dtype == torch.float32
        and sinks.shape == (CK_MIMO_PREFILL_QUERY_HEADS,)
        and sinks.is_contiguous()
        and int(window_size[0]) > 0
        and backend.input_dtype == torch.bfloat16
        and backend.kv_cache_dtype == torch.bfloat16
        and sub_pool.dtype == torch.bfloat16
        and sub_pool.store_dtype == torch.bfloat16
        and backend.page_size == CK_MIMO_PREFILL_PAGE_SIZE
        and layer.tp_q_head_num == CK_MIMO_PREFILL_QUERY_HEADS
        and layer.tp_k_head_num == CK_MIMO_PREFILL_KV_HEADS
        and layer.tp_v_head_num == CK_MIMO_PREFILL_KV_HEADS
        and layer.qk_head_dim == CK_MIMO_PREFILL_HEAD_DIM
        and layer.head_dim == CK_MIMO_PREFILL_HEAD_DIM
        and layer.v_head_dim == CK_MIMO_PREFILL_VALUE_HEAD_DIM
        and getattr(layer, "mimo_original_v_head_dim", None)
        == MIMO_FRESH_BF16_ASM_V_HEAD_DIM
        and q.dtype == torch.bfloat16
        and q.shape[-1] == layer.tp_q_head_num * layer.head_dim
        and q.stride(-1) == 1
        and float(backend.logits_soft_cap) == 0.0
        and k_buf.ndim == 5
        and v_buf.ndim == 5
        and tuple(k_buf.shape[1:]) == expected_k_tail
        and tuple(v_buf.shape[1:]) == expected_v_tail
        and all(
            getattr(metadata, name, None) is not None
            for name in (
                "paged_kv_indptr",
                "paged_kv_indices",
                "paged_kv_last_page_len",
            )
        )
        and hasattr(backend.token_to_kv_pool, "translate_loc_from_full_to_swa")
    )


def run_mimo_flypa_swa_prefill(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    layer: RadixAttention,
    bs0: int,
    k_buf: torch.Tensor,
    v_buf: torch.Tensor,
    metadata,
    sinks: torch.Tensor,
    output_token_count: int,
) -> torch.Tensor:
    """Run cached BF16 MiMo SWA directly on the SHUFFLE-5D cache."""
    from aiter.ops.flydsl import flydsl_paged_attention_swa_bf16

    if not getattr(backend, "_logged_mimo_flypa_swa_prefill", False):
        logger.info(
            "Using FlyPA for cached MiMo BF16 SWA prefill "
            "(window=%d, max_q=%d, max_kv=%d).",
            int(layer.sliding_window_size),
            int(metadata.max_q_len),
            int(metadata.max_kv_len),
        )
        backend._logged_mimo_flypa_swa_prefill = True

    paged_swa_kv_indices = getattr(metadata, "paged_swa_kv_indices", None)
    if paged_swa_kv_indices is None:
        full_page_starts = metadata.paged_kv_indices.to(torch.int64)
        full_page_starts.mul_(backend.page_size)
        paged_swa_kv_indices = (
            backend.token_to_kv_pool.translate_loc_from_full_to_swa(full_page_starts)
            .div(backend.page_size, rounding_mode="floor")
            .to(torch.int32)
        )
        metadata.paged_swa_kv_indices = paged_swa_kv_indices

    output_storage, output = _allocate_mimo_asm_output(
        q,
        q.shape[0],
        output_token_count,
        layer.tp_q_head_num,
        layer.v_head_dim,
    )
    unit_scale = getattr(backend, "_mimo_flypa_prefill_unit_scale", None)
    if unit_scale is None:
        unit_scale = torch.ones(1, dtype=torch.float32, device=q.device)
        backend._mimo_flypa_prefill_unit_scale = unit_scale

    result = flydsl_paged_attention_swa_bf16(
        # The native SWA kernel accepts a strided token axis. Q is a view into
        # fused QKV, so making it contiguous would copy the query at every layer.
        q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
        k_buf,
        v_buf,
        backend.qo_indptr[:bs0],
        metadata.paged_kv_indptr[:bs0],
        paged_swa_kv_indices,
        int(metadata.max_q_len),
        int(metadata.max_kv_len),
        # Match verification: Flash's two-head window includes the current key.
        window_left=int(layer.sliding_window_size) - int(layer.tp_k_head_num == 2),
        kv_last_page_lens=metadata.paged_kv_last_page_len,
        q_descale=unit_scale,
        k_descale=unit_scale,
        v_descale=unit_scale,
        out=output,
        sink_ptr=sinks,
        softmax_scale=layer.scaling,
        stream=(
            torch.cuda.current_stream(q.device) if q.device.type == "cuda" else None
        ),
    )
    if result.data_ptr() != output.data_ptr():
        raise RuntimeError("MiMo cached BF16 SWA FlyPA ignored its output buffer")
    return output_storage.view(-1, layer.tp_q_head_num * layer.v_head_dim)


def run_mimo_flypa_prefill(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    layer: RadixAttention,
    bs0: int,
    sub_pool,
    k_buf: torch.Tensor,
    v_buf: torch.Tensor,
    metadata,
) -> torch.Tensor:
    k_paged = (
        k_buf.view(sub_pool.dtype) if sub_pool.store_dtype != sub_pool.dtype else k_buf
    )
    v_paged = (
        v_buf.view(sub_pool.dtype) if sub_pool.store_dtype != sub_pool.dtype else v_buf
    )
    if sub_pool.dtype == fp8_dtype:
        q_local, q_descale_local = quantize_query_per_tensor_fp8(q)
        k_descale_local = (
            layer.k_scale if layer.k_scale is not None else backend.k_scale
        )
        v_descale_local = (
            layer.v_scale if layer.v_scale is not None else backend.v_scale
        )
        if not _is_scalar_f32_scale(q_descale_local):
            raise RuntimeError("FlyPA FP8 prefill requires a scalar Q descale")
    else:
        q_local = q
        unit = getattr(backend, "_mimo_flypa_prefill_unit_scale", None)
        if unit is None:
            unit = torch.ones(1, dtype=torch.float32, device=q.device)
            backend._mimo_flypa_prefill_unit_scale = unit
        q_descale_local = unit
        k_descale_local = unit
        v_descale_local = unit
    q_paged = q_local.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
    o = flypa(
        num_qo_heads=layer.tp_q_head_num,
        num_kv_heads=layer.tp_k_head_num,
        head_dim_qk=layer.qk_head_dim,
        head_dim_v=layer.v_head_dim,
        page_size=backend.page_size,
        is_causal=True,
        quant_query_mode="per-tensor",
    )(
        q_paged,
        k_paged,
        v_paged,
        backend.qo_indptr[:bs0],
        None,
        metadata.paged_kv_indptr[:bs0],
        metadata.paged_kv_indices,
        int(metadata.max_q_len),
        int(metadata.max_kv_len),
        True,
        q_descale_local,
        k_descale_local,
        v_descale_local,
        metadata.paged_kv_last_page_len,
    )
    if o.dtype != backend.input_dtype:
        o = o.to(backend.input_dtype)
    return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)


def can_use_mimo_flydsl_fp8_prefill(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    layer: RadixAttention,
    window_size,
    sinks,
    is_swa_layer: bool,
    sub_pool,
    k_buf: torch.Tensor,
    v_buf: torch.Tensor,
    metadata,
) -> bool:
    """Return whether gfx950 FlyDSL paged FP8 can consume this cached extend.

    FP8-only and tried before all fallback paths so the local FlyPA kernel
    cannot intercept supported gfx950 shapes.
    """
    if not get_bool_env_var(FLYDSL_MIMO_PREFILL_ENV, "false"):
        return False
    if not is_gfx950() or is_swa_layer or sinks is not None:
        return False
    if tuple(window_size) != (-1, -1):
        return False
    if (
        backend.input_dtype != torch.bfloat16
        or backend.kv_cache_dtype != fp8_dtype
        or sub_pool.dtype != fp8_dtype
        or backend.page_size != CK_MIMO_PREFILL_PAGE_SIZE
        or float(backend.logits_soft_cap) != 0.0
    ):
        return False
    if (
        layer.tp_q_head_num != CK_MIMO_PREFILL_QUERY_HEADS
        or layer.tp_k_head_num != CK_MIMO_PREFILL_KV_HEADS
        or layer.tp_v_head_num != CK_MIMO_PREFILL_KV_HEADS
        or layer.qk_head_dim != CK_MIMO_PREFILL_HEAD_DIM
        or layer.head_dim != CK_MIMO_PREFILL_HEAD_DIM
        or layer.v_head_dim != FLYDSL_MIMO_VALUE_HEAD_DIM
        or getattr(layer, "mimo_original_v_head_dim", None) != 128
    ):
        return False
    if (
        getattr(metadata, "max_q_len", None) is None
        or getattr(metadata, "max_kv_len", None) is None
    ):
        return False
    expected_k_tail = (
        CK_MIMO_PREFILL_KV_HEADS,
        CK_MIMO_PREFILL_HEAD_DIM // 16,
        CK_MIMO_PREFILL_PAGE_SIZE,
        16,
    )
    expected_v_tail = (
        CK_MIMO_PREFILL_KV_HEADS,
        CK_MIMO_PREFILL_PAGE_SIZE // 16,
        CK_MIMO_PREFILL_VALUE_HEAD_DIM,
        16,
    )
    if (
        k_buf.ndim != 5
        or v_buf.ndim != 5
        or k_buf.shape[0] != v_buf.shape[0]
        or tuple(k_buf.shape[1:]) != expected_k_tail
        or tuple(v_buf.shape[1:]) != expected_v_tail
        or k_buf.element_size() != 1
        or v_buf.element_size() != 1
    ):
        return False
    if any(
        getattr(metadata, name, None) is None
        for name in (
            "kv_indptr",
            "paged_kv_indptr",
            "paged_kv_indices",
            "paged_kv_last_page_len",
        )
    ):
        return False
    if (
        metadata.kv_indptr.dtype != torch.int32
        or metadata.paged_kv_indptr.dtype != torch.int32
        or metadata.paged_kv_indices.dtype != torch.int32
    ):
        return False
    k_scale = layer.k_scale if layer.k_scale is not None else backend.k_scale
    v_scale = layer.v_scale if layer.v_scale is not None else backend.v_scale
    return (
        _is_scalar_f32_scale(k_scale)
        and _is_scalar_f32_scale(v_scale)
        and k_scale.device == q.device
        and v_scale.device == q.device
    )


def run_mimo_flydsl_fp8_prefill(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    layer: RadixAttention,
    bs0: int,
    sub_pool,
    k_buf: torch.Tensor,
    v_buf: torch.Tensor,
    metadata,
) -> torch.Tensor:
    k_paged = (
        k_buf.view(sub_pool.dtype) if sub_pool.store_dtype != sub_pool.dtype else k_buf
    )
    v_paged = (
        v_buf.view(sub_pool.dtype) if sub_pool.store_dtype != sub_pool.dtype else v_buf
    )
    q_local, q_descale_local = quantize_query_per_tensor_fp8(q)
    k_descale_local = layer.k_scale if layer.k_scale is not None else backend.k_scale
    v_descale_local = layer.v_scale if layer.v_scale is not None else backend.v_scale
    q_paged = q_local.contiguous().view(-1, layer.tp_q_head_num, layer.qk_head_dim)
    o = load_flydsl_mimo_prefill_kernel().run(
        q_paged,
        k_paged,
        v_paged,
        backend.qo_indptr[:bs0],
        metadata.kv_indptr[:bs0],
        metadata.paged_kv_indptr[:bs0],
        metadata.paged_kv_indices,
        max_seqlen_q=int(metadata.max_q_len),
        max_seqlen_kv=int(metadata.max_kv_len),
        q_descale=q_descale_local,
        k_descale=k_descale_local,
        v_descale=v_descale_local,
        stream=(
            torch.cuda.current_stream(q.device) if q.device.type == "cuda" else None
        ),
    )
    if o.dtype != backend.input_dtype:
        o = o.to(backend.input_dtype)
    return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)


def _mimo_gfx942_paged_swa_prefill(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    bs0: int,
    window_size,
    sinks,
) -> torch.Tensor | None:
    """Use native BF16 FlyPA for gfx942 MiMo SWA after the cache write.

    Full and SWA allocators use different physical page IDs. The existing
    paged runner translates the table once per forward batch and retains
    per-request page counts, last-page lengths, sinks and prefill masking.
    """
    metadata = backend.forward_metadata
    # Pro TP8 has one local KV head; Flash TP4 has two. The native BF16
    # paged kernel handles both GQA ratios and derives the head stride from K/V.
    num_kv_heads = layer.tp_k_head_num
    lengths = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if getattr(metadata, "swa_page_table", None) is None:
        return None
    if (
        backend.page_size != CK_MIMO_PREFILL_PAGE_SIZE
        or backend.input_dtype != torch.bfloat16
        or backend.kv_cache_dtype != torch.bfloat16
        or layer.sliding_window_size is None
        or int(layer.sliding_window_size) <= 0
        or tuple(window_size) != (int(layer.sliding_window_size), -1)
        or float(backend.logits_soft_cap) != 0.0
        or getattr(layer, "mimo_original_v_head_dim", None) != 128
        or (
            layer.tp_q_head_num,
            layer.tp_k_head_num,
            layer.tp_v_head_num,
            layer.head_dim,
            layer.qk_head_dim,
            layer.v_head_dim,
        )
        != (16, num_kv_heads, num_kv_heads, 192, 192, 128)
        or num_kv_heads not in (1, 2)
        or q.dtype != torch.bfloat16
        or q.ndim != 2
        or q.shape[1] != 16 * 192
        or q.stride(-1) != 1
        or lengths is None
        or len(lengths) != bs0 - 1
        or not lengths
        or any(int(length) <= 0 for length in lengths)
    ):
        return None
    if sinks is not None and not (
        isinstance(sinks, torch.Tensor)
        and sinks.dtype == torch.float32
        and sinks.device == q.device
        and sinks.shape == (16,)
        and sinks.is_contiguous()
    ):
        return None
    tables = [
        backend.qo_indptr,
        getattr(metadata, "paged_kv_indptr", None),
        getattr(metadata, "paged_kv_indices", None),
        getattr(metadata, "paged_kv_last_page_len", None),
    ]
    if any(
        not isinstance(table, torch.Tensor)
        or table.dtype != torch.int32
        or table.device != q.device
        or table.ndim != 1
        or not table.is_contiguous()
        for table in tables
    ):
        return None
    if (
        tables[0].numel() < bs0
        or tables[1].numel() < bs0
        or tables[3].numel() < bs0 - 1
    ):
        return None
    pool = backend.token_to_kv_pool
    mapping = getattr(pool, "layers_mapping", {}).get(layer.layer_id)
    if (
        mapping is None
        or not mapping[1]
        or not hasattr(pool, "translate_loc_from_full_to_swa")
    ):
        return None
    sub_pool = pool.swa_kv_pool
    if sub_pool.dtype != torch.bfloat16 or sub_pool.store_dtype != torch.bfloat16:
        return None
    k_buf = sub_pool.k_buffer[mapping[0] - sub_pool.start_layer]
    v_buf = sub_pool.v_buffer[mapping[0] - sub_pool.start_layer]
    if (
        k_buf.ndim != 5
        or v_buf.ndim != 5
        or k_buf.shape[0] != v_buf.shape[0]
        or tuple(k_buf.shape[1:]) != (num_kv_heads, 24, 64, 8)
        or tuple(v_buf.shape[1:]) != (num_kv_heads, 8, 128, 8)
        or k_buf.dtype != torch.bfloat16
        or v_buf.dtype != torch.bfloat16
        or k_buf.device != q.device
        or v_buf.device != q.device
        or not k_buf.is_contiguous()
        or not v_buf.is_contiguous()
    ):
        return None
    logical_tokens = sum(int(length) for length in lengths)
    if logical_tokens > q.shape[0]:
        raise ValueError("MiMo gfx942 paged SWA query lengths exceed the query buffer")
    if not getattr(backend, "_logged_gfx942_mimo_paged_swa_prefill", False):
        logger.info(
            "Using gfx942 FlyPA MiMo BF16 SWA page-64 5D prefill "
            "(kv_heads=%d, window=%s, max_q=%d, max_kv=%d); no LINEAR K/V gather.",
            num_kv_heads,
            window_size,
            int(metadata.max_q_len),
            int(metadata.max_kv_len),
        )
        backend._logged_gfx942_mimo_paged_swa_prefill = True
    return run_mimo_flypa_swa_prefill(
        backend,
        q[:logical_tokens],
        layer,
        bs0,
        k_buf,
        v_buf,
        metadata,
        sinks,
        output_token_count=q.shape[0],
    )


def forward_extend_vectorized_5d(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    bs0: int,
    window_size,
    sinks,
    kv_cache_written: bool = True,
) -> torch.Tensor:
    """``forward_extend`` specialization for the SHUFFLE 5D KV pool.

    Qualified gfx942 BF16 MiMo SWA uses native page-64 FlyPA first, for both
    fresh and cached chunks. Other extends follow these existing sub-paths:

    1. Fresh-prompt shortcut: when every request in the batch has zero
       ``extend_prefix_lens`` (first chunk of a fresh prompt, or any
       path bypassing prefix reuse) the fresh ``(k, v)`` inputs ARE the
       full KV stream — skip pool reads entirely and run on bf16
       ``(k, v)`` directly. No descales needed since no data is read
       from the (possibly fp8) cache.

    2. gfx950 cached BF16 varlen ASM: gather prefix+chunk and reuse the same
       D192/V128 ASM as fresh prefill. This stays ahead of FlyPA on gfx950.

    3. Direct paged FlyDSL / FlyPA / CK: gfx950 FP8 SHUFFLE-5D prefers the
       size-gated FlyDSL paged kernel over FlyPA. FlyPA is the gfx942 (and
       gfx950 fallback) paged kernel for BF16 or FP8. Remaining qualified
       BF16/FP8 pages go directly to CK's MiMo page-64 specialization.

    4. Gather-and-linearize: every unsupported case gathers per-token K/V from the
       SHUFFLE 5D pool via ``launch_gather_shuffle_5d_to_linear``
       (triton inverse of the SHUFFLE writer) into a contiguous
       ``(T, H, D)`` buffer in the cache's ``store_dtype``, then run the
       same LINEAR prefill. fp8-store layers are forwarded to aiter as
       raw fp8 with the per-tensor descales — aiter's LINEAR-mode kernel
       supports fp8 K/V/Q natively, so no host-side dequant is needed.

    The optimized paths are deliberately narrow. Unsupported dtypes, head
    shapes, and missing metadata retain the established fallback.

    Returns the ``(T, H_q * D_v)`` attention output, ready to be
    returned from ``AiterAttnBackend.forward_extend``.
    """
    # The gfx950 routing below stays unchanged. On gfx942, handle both fresh
    # and cached SWA here so the fresh-prompt shortcut cannot select page-1.
    if (
        is_gfx942()
        and not is_gfx950()
        and kv_cache_written
        and layer.sliding_window_size is not None
        and layer.sliding_window_size > 0
        and get_bool_env_var(FLYPA_MIMO_PREFILL_ENV, "false")
    ):
        paged_swa_output = _mimo_gfx942_paged_swa_prefill(
            backend, q, layer, forward_batch, bs0, window_size, sinks
        )
        if paged_swa_output is not None:
            return paged_swa_output

    asm_q, asm_k, asm_v = q, k, v
    physical_q_tokens = q.shape[0]
    asm_output_kwargs = {}
    if MIMO_FRESH_BF16_ASM_ENABLED and is_gfx950():
        asm_q, asm_k, asm_v, physical_q_tokens = _mimo_logical_qkv_views(
            q, k, v, forward_batch
        )
        if asm_q.shape[0] != physical_q_tokens:
            asm_output_kwargs = {"output_token_count": physical_q_tokens}

    # Path 1: fresh-prompt shortcut.
    # With chunked-prefix caching disabled, the scheduler can report a zero
    # cache-match prefix for resumed chunks. The logical prefix is still
    # visible in seq_lens - extend_lens and must be read from the KV pool.
    effective_prefix_lengths = _mimo_effective_prefix_lengths(forward_batch)
    extend_no_prefix = effective_prefix_lengths is not None and not any(
        effective_prefix_lengths
    )
    if (
        get_bool_env_var(FLYPA_MIMO_PREFILL_ENV, "false")
        and is_gfx942()
        and not is_gfx950()
        and (layer.sliding_window_size is None or int(layer.sliding_window_size) < 0)
    ):
        # gfx942 has no D192/V128 ASM. Force the paged path so FlyPA covers
        # both the first chunk and later cached chunks. gfx950 keeps the
        # fresh ASM/CK shortcut (and SWA never uses FlyPA).
        extend_no_prefix = False

    if extend_no_prefix:
        if can_use_mimo_fresh_bf16_asm(
            backend,
            asm_q,
            asm_k,
            asm_v,
            layer,
            forward_batch,
            window_size,
            sinks,
        ):
            return mimo_fresh_bf16_asm(
                asm_q,
                asm_k,
                asm_v,
                layer,
                forward_batch,
                **asm_output_kwargs,
            )

        if can_use_mimo_fresh_bf16_varlen_asm(
            backend,
            asm_q,
            asm_k,
            asm_v,
            layer,
            forward_batch,
            window_size,
            sinks,
        ):
            return mimo_fresh_bf16_varlen_asm(
                backend,
                asm_q,
                asm_k,
                asm_v,
                layer,
                forward_batch,
                **asm_output_kwargs,
            )

        if can_use_mimo_fresh_bf16_swa_varlen(
            backend,
            q,
            k,
            v,
            layer,
            forward_batch,
            window_size,
            sinks,
        ):
            return mimo_fresh_bf16_swa_varlen(
                backend,
                q,
                k,
                v,
                layer,
                forward_batch,
                window_size,
                sinks,
            )

        if (
            is_gfx950()
            and flash_attn_varlen_func is not None
            and q.dtype == torch.bfloat16
            and k.dtype == torch.bfloat16
            and v.dtype == torch.bfloat16
        ):
            # CK batch-prefill has no page-1 kernel for a single BF16 token.
            # The native D192/V128 varlen path covers that final extend and
            # keeps the same bottom-right causal and sink semantics.
            varlen_window = (
                (-1, -1, 0)
                if tuple(window_size) == (-1, -1)
                else (int(window_size[0]), 0, 0)
            )
            o = flash_attn_varlen_func(
                q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                k.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
                backend.qo_indptr[:bs0],
                backend.qo_indptr[:bs0],
                int(backend.forward_metadata.max_q_len),
                int(backend.forward_metadata.max_q_len),
                min_seqlen_q=0,
                softmax_scale=layer.scaling,
                logits_soft_cap=backend.logits_soft_cap,
                causal=True,
                window_size=varlen_window,
                sink_ptr=sinks,
            )
            if o.dtype != backend.input_dtype:
                o = o.to(backend.input_dtype)
            return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

        # Q and K are head-aligned views whose last dimension is contiguous.
        # AITER's LINEAR prefill kernel accepts their token stride directly, so
        # avoid materializing full-token copies for the fresh-prompt chunk.
        k_lin = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v_lin = v.contiguous().view(-1, layer.tp_v_head_num, layer.v_head_dim)
        total_tokens = k_lin.shape[0]
        kv_indices_lin = torch.arange(
            total_tokens, dtype=torch.int32, device=k_lin.device
        )
        kv_indptr_lin = backend.qo_indptr[:bs0]
        max_q = int(backend.forward_metadata.max_q_len)
        o = mha_batch_prefill_func(
            q.view(-1, layer.tp_q_head_num, layer.head_dim),
            k_lin,
            v_lin,
            backend.qo_indptr[:bs0],
            kv_indptr_lin,
            kv_indices_lin,
            max_q,
            max_q,
            causal=True,
            logits_soft_cap=backend.logits_soft_cap,
            alibi_slopes=None,
            return_lse=False,
            return_attn_probs=False,
            window_size=window_size,
            sink_ptr=sinks,
        )
        if o.dtype != backend.input_dtype:
            o = o.to(backend.input_dtype)
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    # Resolve the raw 5D K/V buffer for this layer (going through the
    # SWA->sub-pool mapping when applicable).
    is_swa_layer = (
        layer.sliding_window_size is not None
        and layer.sliding_window_size > -1
        and backend.forward_metadata.swa_page_table is not None
    )
    pool = backend.token_to_kv_pool
    if hasattr(pool, "layers_mapping"):
        sub_layer_id, sub_is_swa = pool.layers_mapping[layer.layer_id]
        sub_pool = pool.swa_kv_pool if sub_is_swa else pool.full_kv_pool
    else:
        sub_pool = pool
        sub_layer_id = layer.layer_id
    k_buf = sub_pool.k_buffer[sub_layer_id - sub_pool.start_layer]
    v_buf = sub_pool.v_buffer[sub_layer_id - sub_pool.start_layer]

    metadata = backend.forward_metadata
    has_paged_metadata = all(
        getattr(metadata, name, None) is not None
        for name in (
            "paged_kv_indptr",
            "paged_kv_indices",
            "paged_kv_last_page_len",
        )
    )
    kv_vector_size = 16 // k_buf.element_size()
    expected_k_tail = (
        CK_MIMO_PREFILL_KV_HEADS,
        CK_MIMO_PREFILL_HEAD_DIM // kv_vector_size,
        CK_MIMO_PREFILL_PAGE_SIZE,
        kv_vector_size,
    )
    expected_v_tail = (
        CK_MIMO_PREFILL_KV_HEADS,
        CK_MIMO_PREFILL_PAGE_SIZE // kv_vector_size,
        CK_MIMO_PREFILL_VALUE_HEAD_DIM,
        kv_vector_size,
    )
    # gfx950 cached BF16 full-attn prefers varlen ASM. Cached BF16 SWA can
    # instead consume the 5D cache directly through the native FlyPA kernel.
    # gfx950 FP8 prefers the dedicated FlyDSL paged kernel when it qualifies.
    if can_use_mimo_flypa_swa_prefill(
        backend,
        asm_q,
        layer,
        forward_batch,
        window_size,
        sinks,
        is_swa_layer,
        sub_pool,
        k_buf,
        v_buf,
        metadata,
    ):
        return run_mimo_flypa_swa_prefill(
            backend,
            asm_q,
            layer,
            bs0,
            k_buf,
            v_buf,
            metadata,
            sinks,
            physical_q_tokens,
        )

    if can_use_mimo_chunk_bf16_varlen_asm(
        backend,
        asm_q,
        asm_k,
        asm_v,
        layer,
        forward_batch,
        window_size,
        sinks,
        is_swa_layer,
        sub_pool,
        k_buf,
        v_buf,
        metadata,
    ):
        return mimo_chunk_bf16_varlen_asm(
            backend,
            asm_q,
            layer,
            forward_batch,
            k_buf,
            v_buf,
            metadata,
            **asm_output_kwargs,
        )

    if can_use_mimo_flydsl_fp8_prefill(
        backend,
        q,
        layer,
        window_size,
        sinks,
        is_swa_layer,
        sub_pool,
        k_buf,
        v_buf,
        metadata,
    ):
        return run_mimo_flydsl_fp8_prefill(
            backend,
            q,
            layer,
            bs0,
            sub_pool,
            k_buf,
            v_buf,
            metadata,
        )

    if can_use_mimo_flypa_prefill(
        backend,
        layer,
        window_size,
        sinks,
        is_swa_layer,
        sub_pool,
        k_buf,
        v_buf,
        metadata,
    ):
        return run_mimo_flypa_prefill(
            backend,
            q,
            layer,
            bs0,
            sub_pool,
            k_buf,
            v_buf,
            metadata,
        )

    use_direct_paged = (
        mha_batch_prefill_func is not None
        and is_mimo_flypa_arch()
        and not is_swa_layer
        and sinks is None
        and tuple(window_size) == (-1, -1)
        and backend.input_dtype == torch.bfloat16
        and backend.kv_cache_dtype in (fp8_dtype, torch.bfloat16)
        and sub_pool.dtype == backend.kv_cache_dtype
        and (sub_pool.dtype == fp8_dtype or sub_pool.store_dtype == torch.bfloat16)
        and backend.page_size == CK_MIMO_PREFILL_PAGE_SIZE
        and layer.tp_q_head_num == CK_MIMO_PREFILL_QUERY_HEADS
        and layer.tp_k_head_num == CK_MIMO_PREFILL_KV_HEADS
        and layer.tp_v_head_num == CK_MIMO_PREFILL_KV_HEADS
        and layer.qk_head_dim == CK_MIMO_PREFILL_HEAD_DIM
        and layer.v_head_dim == CK_MIMO_PREFILL_VALUE_HEAD_DIM
        and layer.head_dim == CK_MIMO_PREFILL_HEAD_DIM
        and float(backend.logits_soft_cap) == 0.0
        and has_paged_metadata
        and k_buf.ndim == 5
        and v_buf.ndim == 5
        and tuple(k_buf.shape[1:]) == expected_k_tail
        and tuple(v_buf.shape[1:]) == expected_v_tail
        and k_buf.shape[0] == v_buf.shape[0]
        and k_buf.element_size() in (1, 2)
        and v_buf.element_size() == k_buf.element_size()
        and backend.qo_indptr.dtype == torch.int32
        and metadata.paged_kv_indptr.dtype == torch.int32
        and metadata.paged_kv_indices.dtype == torch.int32
        and metadata.paged_kv_last_page_len.dtype == torch.int32
        and backend.qo_indptr.numel() >= bs0
        and metadata.paged_kv_indptr.numel() >= bs0
        and metadata.paged_kv_last_page_len.numel() >= bs0 - 1
    )

    if use_direct_paged:
        # FP8 pools may expose uint8 storage because some PyTorch indexing
        # operations do not implement float8. Reinterpret the identical bytes
        # without copying before handing the physical 5D cache to AITER.
        k_paged = (
            k_buf.view(sub_pool.dtype)
            if sub_pool.store_dtype != sub_pool.dtype
            else k_buf
        )
        v_paged = (
            v_buf.view(sub_pool.dtype)
            if sub_pool.store_dtype != sub_pool.dtype
            else v_buf
        )
        extend_lengths = getattr(forward_batch, "extend_seq_lens_cpu", None)
        logical_q_tokens = (
            q.shape[0]
            if extend_lengths is None
            else sum(int(length) for length in extend_lengths)
        )
        if logical_q_tokens < 0 or logical_q_tokens > q.shape[0]:
            raise ValueError(
                "MiMo CK paged prefill query metadata is inconsistent: "
                f"logical_q_tokens={logical_q_tokens}, physical_q_tokens={q.shape[0]}"
            )
        q_live = q[:logical_q_tokens]
        if sub_pool.dtype == fp8_dtype:
            q_local, q_descale_local = quantize_query_per_tensor_fp8(q_live)
            k_descale_local = (
                layer.k_scale if layer.k_scale is not None else backend.k_scale
            )
            v_descale_local = (
                layer.v_scale if layer.v_scale is not None else backend.v_scale
            )
        else:
            q_local = q_live
            q_descale_local = None
            k_descale_local = None
            v_descale_local = None
        max_kv = int(metadata.max_kv_len)
        max_q = int(metadata.max_q_len)
        q_paged = q_local.contiguous().view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        if not getattr(backend, "_logged_mimo_ck_paged_prefill", False):
            logger.info(
                "Using AITER CK MiMo page-64 prefill fallback (kv_dtype=%s, "
                "max_q=%d, max_kv=%d).",
                sub_pool.dtype,
                max_q,
                max_kv,
            )
            backend._logged_mimo_ck_paged_prefill = True
        o = mha_batch_prefill_func(
            q_paged,
            k_paged,
            v_paged,
            backend.qo_indptr[:bs0],
            metadata.paged_kv_indptr[:bs0],
            metadata.paged_kv_indices,
            max_q,
            max_kv,
            causal=True,
            logits_soft_cap=0.0,
            alibi_slopes=None,
            return_lse=False,
            return_attn_probs=False,
            window_size=(-1, -1),
            sink_ptr=None,
            q_descale=q_descale_local,
            k_descale=k_descale_local,
            v_descale=v_descale_local,
            kv_last_page_lens=metadata.paged_kv_last_page_len,
        )
        if o.dtype != backend.input_dtype:
            o = o.to(backend.input_dtype)
        if q_paged.shape[0] != q.shape[0]:
            output_storage, output_live = _allocate_mimo_asm_output(
                q,
                q_paged.shape[0],
                q.shape[0],
                layer.tp_q_head_num,
                layer.v_head_dim,
            )
            output_live.copy_(o)
            o = output_storage
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    # Path 4: gather-and-linearize. SWA layers gather from the SWA sub-pool;
    # full-attention layers gather from the full pool. Both metadata tensors
    # contain per-token absolute slots in request-major order.
    total_kv = int(forward_batch.seq_lens_sum)
    if is_swa_layer:
        slot_ids = metadata.swa_page_table[:total_kv]
    else:
        slot_ids = metadata.kv_indices[:total_kv]

    if not hasattr(backend, "_linear_kv_workspace"):
        backend._linear_kv_workspace = {}
    k_lin, v_lin = launch_gather_shuffle_5d_to_linear(
        k_buf, v_buf, slot_ids, workspace=backend._linear_kv_workspace
    )
    # k_lin / v_lin come out in ``store_dtype`` (uint8 for fp8 pools
    # because ``Tensor.index_put`` isn't implemented for fp8 — see
    # ``MHATokenToKVPool`` ctor). Reinterpret them back to the compute
    # dtype so aiter sees matching q/k/v dtypes. The bytes are
    # identical; this is a zero-copy view.
    if sub_pool.store_dtype != sub_pool.dtype:
        k_lin = k_lin.view(sub_pool.dtype)
        v_lin = v_lin.view(sub_pool.dtype)

    # For fp8 K/V we hand the raw fp8 tensors and the layer's per-tensor
    # descales straight to aiter.
    if sub_pool.dtype == fp8_dtype:
        q_local, q_descale_local = quantize_query_per_tensor_fp8(q)
        k_descale_local = (
            layer.k_scale if layer.k_scale is not None else backend.k_scale
        )
        v_descale_local = (
            layer.v_scale if layer.v_scale is not None else backend.v_scale
        )
    else:
        q_local = q
        q_descale_local = None
        k_descale_local = None
        v_descale_local = None

    kv_indptr_lin = backend.forward_metadata.kv_indptr[:bs0]
    kv_indices_lin = torch.arange(total_kv, dtype=torch.int32, device=k_lin.device)
    max_kv = int(backend.forward_metadata.max_kv_len)
    max_q = int(backend.forward_metadata.max_q_len)

    o = mha_batch_prefill_func(
        q_local.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
        k_lin,
        v_lin,
        backend.qo_indptr[:bs0],
        kv_indptr_lin,
        kv_indices_lin,
        max_q,
        max_kv,
        causal=True,
        logits_soft_cap=backend.logits_soft_cap,
        alibi_slopes=None,
        return_lse=False,
        return_attn_probs=False,
        window_size=window_size,
        sink_ptr=sinks,
        q_descale=q_descale_local,
        k_descale=k_descale_local,
        v_descale=v_descale_local,
    )
    if o.dtype != backend.input_dtype:
        o = o.to(backend.input_dtype)
    return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)


def forward_decode_vectorized_5d(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    o: torch.Tensor,
    sinks,
) -> None:
    """``forward_decode`` specialization for the SHUFFLE 5D KV pool.

    Honors the full-attention FlyDSL selector for ordinary one-token decode.
    SWA retains its independent selector. The Gluon route handles the
    explicitly selected Gluon configuration and remaining SWA cases.

    The choice between the two layer kinds is purely metadata:

    * Full-attn  → ``kv_indices`` page table + ``sliding_window=0`` +
      ``max_part_num`` recommended by aiter heuristics.
    * SWA layer  → ``swa_page_table`` + ``sliding_window=layer.sliding_window_size``
      + ``max_part_num=1`` (SWA windows are small enough that
      splitting does not help).

    fp8 KV requires per-tensor ``key_scale`` / ``value_scale`` to be
    forwarded; without them the kernel reads the fp8 bytes as fp8
    values without any dequant and produces garbage logits.

    Writes the attention output into ``o`` in place (via a stride-0
    safe ``o.view``).
    """
    if _use_flydsl_swa(backend, q, layer, sinks):
        forward_target_verify_flydsl_swa_5d(
            backend, q, layer, forward_batch, k_cache, v_cache, o, sinks
        )
        return

    bs = forward_batch.batch_size
    num_kv_heads = layer.tp_k_head_num
    num_q_heads = layer.tp_q_head_num
    q_group = num_q_heads // num_kv_heads
    is_swa_layer = (
        layer.sliding_window_size is not None and layer.sliding_window_size > -1
    )
    if getattr(backend, "_use_flydsl_pa_decode", False) and not is_swa_layer:
        _forward_full_flydsl_5d(
            backend,
            q,
            layer,
            forward_batch,
            k_cache,
            v_cache,
            o,
            sinks,
            query_length=1,
            context_includes_query=True,
        )
        return

    if is_swa_layer:
        block_tables_pa = (
            backend.forward_metadata.swa_page_table
            if backend.forward_metadata.swa_page_table is not None
            else backend.forward_metadata.kv_indices
        )
        ctx_part = 256
        max_part_num = 1
        sliding_window_arg = int(layer.sliding_window_size)
    else:
        block_tables_pa = backend.forward_metadata.kv_indices
        ctx_part = 256
        max_part_num = get_recommended_splits(bs, num_kv_heads)
        sliding_window_arg = 0

    q_in = q.view(-1, num_q_heads, layer.qk_head_dim)
    # Direct view of o as kernel output — saves a per-layer o.copy_ of
    # bs * H_q * D bf16 elementwise.
    o_view = o.view(-1, num_q_heads, layer.v_head_dim)
    exp_sums = torch.empty(
        (bs, num_kv_heads, max_part_num, q_group),
        dtype=torch.float32,
        device=q_in.device,
    )
    max_logits = torch.empty_like(exp_sums)
    temporary_output = torch.empty(
        (bs, num_kv_heads, max_part_num, q_group, layer.v_head_dim),
        dtype=q_in.dtype,
        device=q_in.device,
    )

    # For fp8 KV cache the kernel needs per-tensor dequant scales
    # (key_scale / value_scale). Without them the fp8 bytes are
    # interpreted as fp8 values with no dequant.
    key_scale = None
    value_scale = None
    if backend.kv_cache_dtype == fp8_dtype:
        key_scale = layer.k_scale if layer.k_scale is not None else backend.k_scale
        value_scale = layer.v_scale if layer.v_scale is not None else backend.v_scale

    pa_decode_gluon(
        output=o_view,
        query=q_in,
        key_cache=k_cache,
        value_cache=v_cache,
        context_lengths=forward_batch.seq_lens,
        block_tables=block_tables_pa,
        softmax_scale=layer.scaling,
        query_length=1,
        max_context_partition_num=max_part_num,
        context_partition_size=ctx_part,
        compute_type=backend.input_dtype,
        key_scale=key_scale,
        value_scale=value_scale,
        exp_sums=exp_sums,
        max_logits=max_logits,
        temporary_output=temporary_output,
        sinks=sinks,
        sliding_window=sliding_window_arg,
        ps=True,
    )


def _get_flydsl_workspace_views(
    backend: AiterAttnBackend,
    batch_size: int,
    query_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_partitions = backend._flydsl_pa_decode_num_partitions
    configured_query_length = int(backend._flydsl_pa_decode_query_length)
    if query_length not in (1, configured_query_length):
        raise RuntimeError(
            "FlyDSL PA decode was prepared for query_length=1 or "
            f"{configured_query_length}, got {query_length}"
        )
    equivalent_group_size = query_length * (
        FLYDSL_MIMO_QUERY_HEADS // FLYDSL_MIMO_KV_HEADS
    )
    scalar_numel = (
        batch_size * FLYDSL_MIMO_KV_HEADS * num_partitions * equivalent_group_size
    )
    output_numel = scalar_numel * FLYDSL_MIMO_VALUE_HEAD_DIM
    buffers = (
        ("pmax", backend._flydsl_pa_decode_pmax, scalar_numel),
        ("psum", backend._flydsl_pa_decode_psum, scalar_numel),
        ("pout", backend._flydsl_pa_decode_pout, output_numel),
        ("context lengths", backend._flydsl_pa_decode_context_lengths, batch_size),
    )
    for name, buffer, required_numel in buffers:
        if buffer is None or buffer.numel() < required_numel:
            available = 0 if buffer is None else buffer.numel()
            raise RuntimeError(
                "FlyDSL PA decode workspace is not initialized for this batch: "
                f"{name} requires {required_numel} elements, has {available}"
            )

    scalar_shape = (
        batch_size,
        FLYDSL_MIMO_KV_HEADS,
        num_partitions,
        equivalent_group_size,
    )
    pmax = backend._flydsl_pa_decode_pmax[:scalar_numel].view(scalar_shape)
    psum = backend._flydsl_pa_decode_psum[:scalar_numel].view(scalar_shape)
    pout = backend._flydsl_pa_decode_pout[:output_numel].view(
        *scalar_shape, FLYDSL_MIMO_VALUE_HEAD_DIM
    )
    context_lengths = backend._flydsl_pa_decode_context_lengths[:batch_size]
    return pmax, psum, pout, context_lengths


def forward_target_verify_flydsl_5d(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    output: torch.Tensor,
    sinks,
) -> None:
    """Run full TARGET_VERIFY; seq_lens excludes the verification queries."""
    _forward_full_flydsl_5d(
        backend,
        q,
        layer,
        forward_batch,
        k_cache,
        v_cache,
        output,
        sinks,
        query_length=int(backend.forward_metadata.max_q_len),
        context_includes_query=False,
    )


def _forward_full_flydsl_5d(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    output: torch.Tensor,
    sinks,
    *,
    query_length: int,
    context_includes_query: bool,
) -> None:
    """Shared full-attention transport for decode and causal verification."""

    if getattr(backend, "_flydsl_pa_decode_tile", None) is None:
        raise RuntimeError("FlyDSL PA decode was selected but is not initialized")

    batch_size = int(forward_batch.batch_size)
    num_q_heads = int(layer.tp_q_head_num)
    num_kv_heads = int(layer.tp_k_head_num)
    head_dim = int(layer.qk_head_dim)
    v_head_dim = int(layer.v_head_dim)

    if query_length not in FLYDSL_MIMO_QUERY_LENGTHS:
        raise ValueError(
            "FlyDSL MiMo full attention requires query_length in "
            f"{FLYDSL_MIMO_QUERY_LENGTHS}; got {query_length}"
        )
    if (num_q_heads, num_kv_heads) != (
        FLYDSL_MIMO_QUERY_HEADS,
        FLYDSL_MIMO_KV_HEADS,
    ):
        raise ValueError(
            "FlyDSL MiMo full attention requires 16 Q heads and 1 KV head; "
            f"got {num_q_heads}/{num_kv_heads}"
        )
    if head_dim != FLYDSL_MIMO_HEAD_DIM or v_head_dim != FLYDSL_MIMO_VALUE_HEAD_DIM:
        raise ValueError(
            "FlyDSL MiMo full attention requires QK/V head dimensions "
            f"192/128; got {head_dim}/{v_head_dim}"
        )
    if backend.page_size != FLYDSL_MIMO_PAGE_SIZE:
        raise ValueError(
            "FlyDSL MiMo full attention requires page_size=64; "
            f"got {backend.page_size}"
        )
    if layer.sliding_window_size is not None and int(layer.sliding_window_size) > -1:
        raise ValueError(
            "FlyDSL MiMo full attention does not support sliding-window layers"
        )
    if sinks is not None:
        raise ValueError("FlyDSL MiMo full attention does not support attention sinks")
    if float(layer.logit_cap or 0.0) != 0.0:
        raise ValueError(
            "FlyDSL MiMo full attention does not support attention logit "
            f"soft-capping; got logit_cap={layer.logit_cap}"
        )
    if backend.kv_cache_dtype not in (fp8_dtype, torch.bfloat16):
        raise ValueError(
            "FlyDSL MiMo full attention requires FP8 E4M3 or BF16 KV cache; "
            f"got {backend.kv_cache_dtype}"
        )
    if q.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
        raise ValueError(
            "FlyDSL MiMo full attention requires BF16 query and output; "
            f"got {q.dtype}/{output.dtype}"
        )
    if q.shape != (batch_size * query_length, num_q_heads * head_dim):
        raise ValueError(
            "FlyDSL MiMo full attention query shape mismatch: expected "
            f"{(batch_size * query_length, num_q_heads * head_dim)}, got "
            f"{tuple(q.shape)}"
        )
    if output.shape != (batch_size * query_length, num_q_heads * v_head_dim):
        raise ValueError(
            "FlyDSL MiMo full attention output shape mismatch: expected "
            f"{(batch_size * query_length, num_q_heads * v_head_dim)}, got "
            f"{tuple(output.shape)}"
        )

    kv_vector_width = 8 if backend.kv_cache_dtype == torch.bfloat16 else 16
    expected_k_tail = (
        num_kv_heads,
        head_dim // kv_vector_width,
        FLYDSL_MIMO_PAGE_SIZE,
        kv_vector_width,
    )
    expected_v_tail = (
        num_kv_heads,
        FLYDSL_MIMO_PAGE_SIZE // kv_vector_width,
        v_head_dim,
        kv_vector_width,
    )
    if k_cache.ndim != 5 or tuple(k_cache.shape[1:]) != expected_k_tail:
        raise ValueError(
            "FlyDSL MiMo full attention K-cache layout mismatch: expected "
            f"[blocks, {', '.join(map(str, expected_k_tail))}], got "
            f"{tuple(k_cache.shape)}"
        )
    if v_cache.ndim != 5 or tuple(v_cache.shape[1:]) != expected_v_tail:
        raise ValueError(
            "FlyDSL MiMo full attention V-cache layout mismatch: expected "
            f"[blocks, {', '.join(map(str, expected_v_tail))}], got "
            f"{tuple(v_cache.shape)}"
        )
    if k_cache.shape[0] != v_cache.shape[0]:
        raise ValueError(
            "FlyDSL MiMo full attention requires K/V caches with the same "
            f"block count; got {k_cache.shape[0]}/{v_cache.shape[0]}"
        )
    expected_element_size = 2 if backend.kv_cache_dtype == torch.bfloat16 else 1
    if (
        k_cache.dtype != backend.kv_cache_dtype
        or v_cache.dtype != backend.kv_cache_dtype
        or k_cache.element_size() != expected_element_size
        or v_cache.element_size() != expected_element_size
    ):
        raise ValueError(
            "FlyDSL MiMo full attention cache storage does not match the "
            f"configured KV dtype; got {k_cache.dtype}/{v_cache.dtype}"
        )

    block_tables = backend.forward_metadata.kv_indices
    # Ordinary decode graph metadata retains the max-BS allocation. A leading
    # view supplies this capture's active rows without changing its device pointer.
    if context_includes_query and block_tables.ndim == 2:
        block_tables = block_tables[:batch_size]
    if block_tables.ndim != 2 or block_tables.dtype != torch.int32:
        raise ValueError(
            "FlyDSL MiMo full attention requires a two-dimensional int32 "
            f"block table; got shape={tuple(block_tables.shape)}, "
            f"dtype={block_tables.dtype}"
        )
    if block_tables.shape[0] != batch_size:
        raise ValueError(
            "FlyDSL MiMo full attention block-table batch mismatch: "
            f"expected {batch_size}, got {block_tables.shape[0]}"
        )
    if forward_batch.seq_lens.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "FlyDSL MiMo full attention requires integral sequence lengths; "
            f"got {forward_batch.seq_lens.dtype}"
        )
    if forward_batch.seq_lens.shape != (batch_size,):
        raise ValueError(
            "FlyDSL MiMo full attention sequence-length shape mismatch: "
            f"expected {(batch_size,)}, got {tuple(forward_batch.seq_lens.shape)}"
        )

    q_view = q.view(-1, num_q_heads, head_dim)
    output_view = output.view(-1, num_q_heads, v_head_dim)
    if q_view.stride(-1) != 1:
        raise ValueError(
            "FlyDSL MiMo full attention requires a contiguous query head axis"
        )

    key_scale = None
    value_scale = None
    if backend.kv_cache_dtype == fp8_dtype:
        key_scale = layer.k_scale if layer.k_scale is not None else backend.k_scale
        value_scale = layer.v_scale if layer.v_scale is not None else backend.v_scale
        scales = []
        for name, scale in (("key_scale", key_scale), ("value_scale", value_scale)):
            if not isinstance(scale, torch.Tensor):
                raise ValueError(f"FlyDSL MiMo {name} must be a persistent tensor")
            if scale.dtype != torch.float32 or scale.numel() != 1 or scale.ndim > 1:
                raise ValueError(
                    f"FlyDSL MiMo {name} must be a scalar/length-1 float32 "
                    f"per-tensor scale; got shape={tuple(scale.shape)}, "
                    f"dtype={scale.dtype}"
                )
            # SGLang's KV-cache quant method owns 0-D Parameter scales. FlyDSL's
            # tensor JIT requires at least one stride-1 axis, so expose a zero-copy
            # [1] view while retaining the Parameter's stable device allocation.
            scales.append(scale.view(1) if scale.ndim == 0 else scale)
        key_scale, value_scale = scales

    device = q.device
    tensors = [
        ("output", output),
        ("K cache", k_cache),
        ("V cache", v_cache),
        ("block table", block_tables),
        ("sequence lengths", forward_batch.seq_lens),
    ]
    if key_scale is not None:
        tensors.extend((("key scale", key_scale), ("value scale", value_scale)))
    for name, tensor in tensors:
        if tensor.device != device:
            raise ValueError(
                f"FlyDSL MiMo {name} must be on {device}; got {tensor.device}"
            )

    pmax, psum, pout, context_lengths = _get_flydsl_workspace_views(
        backend, batch_size, query_length
    )
    for name, workspace, dtype in (
        ("pmax", pmax, torch.float32),
        ("psum", psum, torch.float32),
        ("pout", pout, torch.bfloat16),
        ("context lengths", context_lengths, torch.int32),
    ):
        if workspace.device != device or workspace.dtype != dtype:
            raise ValueError(
                f"FlyDSL MiMo {name} workspace must be {dtype} on {device}; "
                f"got {workspace.dtype} on {workspace.device}"
            )
    # Decode lengths include the query; verification lengths describe its prefix.
    # Convert either convention into the persistent int32 buffer without a
    # temporary allocation or a host/device synchronization.
    if context_includes_query:
        context_lengths.copy_(forward_batch.seq_lens)
    else:
        torch.add(forward_batch.seq_lens, query_length, out=context_lengths)

    backend._flydsl_pa_decode_tile(
        output=output_view,
        query=q_view,
        key_cache=k_cache,
        value_cache=v_cache,
        block_tables=block_tables,
        context_lengths=context_lengths,
        key_scale=key_scale,
        value_scale=value_scale,
        softmax_scale=layer.scaling,
        num_partitions=backend._flydsl_pa_decode_num_partitions,
        pmax=pmax,
        psum=psum,
        pout=pout,
    )


def warmup_flydsl_full_decode(backend: AiterAttnBackend) -> None:
    """Materialize the qlen=1 executable on owned scratch before graph capture.

    AITER's compile_pa_decode_tile returns a JIT launcher; the first call with
    tensor arguments compiles its executable. Preparing only that factory
    leaves ordinary decode cold when startup captures TARGET_VERIFY instead.
    """
    vector = 8 if backend.kv_cache_dtype == torch.bfloat16 else 16
    q = torch.zeros(
        (1, FLYDSL_MIMO_QUERY_HEADS, FLYDSL_MIMO_HEAD_DIM),
        dtype=torch.bfloat16,
        device=backend.device,
    )
    output = torch.empty(
        (1, FLYDSL_MIMO_QUERY_HEADS, FLYDSL_MIMO_VALUE_HEAD_DIM),
        dtype=torch.bfloat16,
        device=backend.device,
    )
    k_cache = torch.zeros(
        (1, 1, FLYDSL_MIMO_HEAD_DIM // vector, FLYDSL_MIMO_PAGE_SIZE, vector),
        dtype=backend.kv_cache_dtype,
        device=backend.device,
    )
    v_cache = torch.zeros(
        (1, 1, FLYDSL_MIMO_PAGE_SIZE // vector, FLYDSL_MIMO_VALUE_HEAD_DIM, vector),
        dtype=backend.kv_cache_dtype,
        device=backend.device,
    )
    table = torch.zeros((1, 1), dtype=torch.int32, device=backend.device)
    pmax, psum, pout, lengths = _get_flydsl_workspace_views(backend, 1, 1)
    lengths.fill_(1)
    scale = (
        torch.ones(1, dtype=torch.float32, device=backend.device)
        if backend.kv_cache_dtype == fp8_dtype
        else None
    )
    backend._flydsl_pa_decode_tile(
        output=output,
        query=q,
        key_cache=k_cache,
        value_cache=v_cache,
        block_tables=table,
        context_lengths=lengths,
        key_scale=scale,
        value_scale=scale,
        softmax_scale=FLYDSL_MIMO_HEAD_DIM**-0.5,
        num_partitions=backend._flydsl_pa_decode_num_partitions,
        pmax=pmax,
        psum=psum,
        pout=pout,
    )
    torch.cuda.current_stream(backend.device).synchronize()


def _split_swa_target_verify_queries() -> bool:
    return os.environ.get(
        "SGLANG_AITER_VEC5D_SWA_VERIFY_QLEN1", "0"
    ).strip().lower() in ("1", "true")


def _split_target_verify_queries(is_swa_layer: bool) -> bool:
    if os.environ.get(
        "SGLANG_AITER_VEC5D_TARGET_VERIFY_QLEN1", "0"
    ).strip().lower() in ("1", "true"):
        return True
    return is_swa_layer and _split_swa_target_verify_queries()


def _ensure_swa_target_verify_workspace(
    backend: AiterAttnBackend,
    *,
    max_batch_size: int,
    query_length: int,
    num_q_heads: int,
    num_kv_heads: int,
    value_head_dim: int,
    max_blocks_per_sequence: int,
    max_partitions: int = 1,
    device: torch.device,
    output_dtype: torch.dtype,
    context_dtype: torch.dtype,
) -> None:
    """Keep expanded qlen-1 verification metadata pointer-stable for replay."""
    if num_kv_heads <= 0 or num_q_heads % num_kv_heads != 0:
        raise ValueError(
            "Target verification requires Q heads to be divisible by KV heads"
        )

    requested_signature = (
        num_q_heads,
        num_kv_heads,
        value_head_dim,
        output_dtype,
        context_dtype,
        device,
    )
    # A shared backend can alternate full and SWA layers with different KV
    # head counts (Flash TP4: one and two). Keep each layout's buffers alive
    # and restore their pointers instead of reallocating during graph capture.
    workspaces = getattr(backend, "_target_verify_workspaces", None)
    if workspaces is None:
        workspaces = backend._target_verify_workspaces = {}
    for name, value in workspaces.get(requested_signature, {}).items():
        setattr(backend, name, value)

    capacity_bs = int(getattr(backend, "_swa_target_verify_workspace_bs", 0))
    capacity_q = int(getattr(backend, "_swa_target_verify_workspace_q", 0))
    capacity_blocks = int(getattr(backend, "_swa_target_verify_workspace_blocks", 0))
    capacity_partitions = int(
        getattr(backend, "_swa_target_verify_workspace_partitions", 0)
    )
    signature = getattr(backend, "_swa_target_verify_workspace_signature", None)
    if signature != requested_signature:
        capacity_bs = capacity_q = capacity_blocks = capacity_partitions = 0
    if (
        signature == requested_signature
        and capacity_bs >= max_batch_size
        and capacity_q >= query_length
        and capacity_blocks >= max_blocks_per_sequence
        and capacity_partitions >= max_partitions
    ):
        return
    if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "split-SWA target-verification workspace cannot grow during capture"
        )
    if requested_signature in workspaces:
        # Earlier graph buckets still hold device pointers into the old
        # allocation when a later warmup needs more partitions or page slots.
        retired = getattr(backend, "_retired_target_verify_workspaces", None)
        if retired is None:
            retired = backend._retired_target_verify_workspaces = []
        retired.append(workspaces[requested_signature])

    capacity_bs = max(max_batch_size, capacity_bs)
    capacity_q = max(query_length, capacity_q)
    capacity_blocks = max(max_blocks_per_sequence, capacity_blocks)
    capacity_partitions = max(max_partitions, capacity_partitions)
    max_rows = capacity_bs * capacity_q
    query_group_size = num_q_heads // num_kv_heads
    scalar_numel = max_rows * num_kv_heads * capacity_partitions * query_group_size

    backend._swa_target_verify_offsets = torch.arange(
        1, capacity_q + 1, dtype=context_dtype, device=device
    )
    backend._swa_target_verify_context_lengths = torch.empty(
        max_rows, dtype=context_dtype, device=device
    )
    backend._swa_target_verify_block_tables = torch.empty(
        (max_rows, capacity_blocks), dtype=torch.int32, device=device
    )
    backend._swa_target_verify_exp_sums = torch.empty(
        scalar_numel, dtype=torch.float32, device=device
    )
    backend._swa_target_verify_max_logits = torch.empty_like(
        backend._swa_target_verify_exp_sums
    )
    backend._swa_target_verify_temporary_output = torch.empty(
        scalar_numel * value_head_dim, dtype=output_dtype, device=device
    )
    backend._swa_target_verify_workspace_bs = capacity_bs
    backend._swa_target_verify_workspace_q = capacity_q
    backend._swa_target_verify_workspace_blocks = capacity_blocks
    backend._swa_target_verify_workspace_partitions = capacity_partitions
    backend._swa_target_verify_workspace_signature = requested_signature
    workspaces[requested_signature] = {
        name: value
        for name, value in vars(backend).items()
        if name.startswith("_swa_target_verify_")
    }


def prepare_swa_target_verify(
    backend: AiterAttnBackend,
    max_batch_size: int,
    query_length: int,
) -> None:
    if not _split_swa_target_verify_queries():
        return
    max_blocks_per_sequence = (
        int(backend.max_context_len) + int(backend.page_size) - 1
    ) // int(backend.page_size)
    _ensure_swa_target_verify_workspace(
        backend,
        max_batch_size=max_batch_size,
        query_length=query_length,
        num_q_heads=FLYDSL_MIMO_QUERY_HEADS,
        num_kv_heads=FLYDSL_MIMO_KV_HEADS,
        value_head_dim=FLYDSL_MIMO_VALUE_HEAD_DIM,
        max_blocks_per_sequence=max_blocks_per_sequence,
        device=torch.device(backend.device),
        output_dtype=backend.input_dtype,
        context_dtype=torch.int64,
    )


def forward_target_verify_vectorized_5d(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    output: torch.Tensor,
    sinks,
) -> None:
    """Run top-k-1 target verification directly on a SHUFFLE 5D KV pool.

    ``TARGET_VERIFY`` has a fixed query length per request. For qlen up to four,
    ``pa_decode_gluon`` consumes the causal multi-token block directly. The
    opt-in qlen-1 fallback exposes each query row as an independent request with
    its own causal context length, keeping the Gluon equivalent group within
    its supported limit for qlen eight.

    This helper deliberately validates the complete Gluon contract before
    allocating workspaces.  Falling through to the legacy unified-attention
    path is not safe: reshaping SHUFFLE 5D storage as NHD changes only the view,
    not the physical cache permutation.
    """
    query_length = int(backend.forward_metadata.max_q_len)
    batch_size = int(forward_batch.batch_size)
    num_q_heads = int(layer.tp_q_head_num)
    num_kv_heads = int(layer.tp_k_head_num)

    is_swa_layer = (
        layer.sliding_window_size is not None and layer.sliding_window_size > -1
    )
    split_verify_queries = query_length > 1 and _split_target_verify_queries(
        is_swa_layer
    )
    use_flydsl_swa = _use_flydsl_swa(backend, q, layer, sinks)
    max_query_length = 8 if split_verify_queries or use_flydsl_swa else 4
    if not 1 <= query_length <= max_query_length:
        raise ValueError(
            "vectorized-5D TARGET_VERIFY query length exceeds the selected path; "
            f"got {query_length}"
        )
    if num_kv_heads <= 0 or num_q_heads % num_kv_heads != 0:
        raise ValueError(
            "vectorized-5D TARGET_VERIFY requires Q heads to be divisible by "
            f"KV heads; got {num_q_heads} Q heads and {num_kv_heads} KV heads"
        )

    query_group_size = num_q_heads // num_kv_heads
    equivalent_group_size = query_length * query_group_size
    if not split_verify_queries and not use_flydsl_swa and equivalent_group_size > 64:
        raise ValueError(
            "vectorized-5D TARGET_VERIFY exceeds pa_decode_gluon's equivalent "
            f"query-group limit: {query_length} * {query_group_size} = "
            f"{equivalent_group_size} > 64"
        )
    asymmetric_mimo = (
        (num_kv_heads == FLYDSL_MIMO_KV_HEADS or use_flydsl_swa)
        and layer.qk_head_dim == FLYDSL_MIMO_HEAD_DIM
        and layer.v_head_dim == FLYDSL_MIMO_VALUE_HEAD_DIM
    )
    if layer.qk_head_dim != layer.v_head_dim and not asymmetric_mimo:
        raise ValueError(
            "vectorized-5D TARGET_VERIFY supports symmetric K/V or the MiMo "
            "one-KV-head QK192/V128 route; got "
            f"{num_kv_heads} KV heads and dimensions "
            f"{layer.qk_head_dim}/{layer.v_head_dim}"
        )
    if backend.page_size not in (16, 64, 1024):
        raise ValueError(
            "vectorized-5D TARGET_VERIFY requires page size 16, 64, or 1024; "
            f"got {backend.page_size}"
        )
    if float(layer.logit_cap or 0.0) != 0.0:
        raise ValueError(
            "vectorized-5D TARGET_VERIFY does not support attention logit "
            f"soft-capping; got logit_cap={layer.logit_cap}"
        )
    if k_cache.ndim != 5 or v_cache.ndim != 5:
        raise ValueError(
            "vectorized-5D TARGET_VERIFY requires 5D K and V caches; got "
            f"K ndim={k_cache.ndim}, V ndim={v_cache.ndim}"
        )
    if k_cache.shape[1] != num_kv_heads or v_cache.shape[1] != num_kv_heads:
        raise ValueError(
            "vectorized-5D TARGET_VERIFY cache/head mismatch: "
            f"layer has {num_kv_heads} KV heads, K/V caches have "
            f"{k_cache.shape[1]}/{v_cache.shape[1]}"
        )
    if k_cache.shape[-2] != backend.page_size:
        raise ValueError(
            "vectorized-5D TARGET_VERIFY cache/page mismatch: "
            f"backend page size is {backend.page_size}, K cache page axis is "
            f"{k_cache.shape[-2]}"
        )
    if q.shape[0] != batch_size * query_length:
        raise ValueError(
            "vectorized-5D TARGET_VERIFY requires a uniform query length; "
            f"got q.shape[0]={q.shape[0]}, batch_size={batch_size}, and "
            f"query_length={query_length}"
        )

    if use_flydsl_swa:
        forward_target_verify_flydsl_swa_5d(
            backend,
            q,
            layer,
            forward_batch,
            k_cache,
            v_cache,
            output,
            sinks,
        )
        return
    if (
        getattr(backend, "_flydsl_dflash_swa_enabled", False)
        or getattr(backend, "_flydsl_target_verify_swa_enabled", False)
    ) and is_swa_layer:
        raise RuntimeError(
            "FlyDSL DFlash SWA was requested but the live attention call does not "
            "match its BF16 H16/HK1-or-HK2/D128-or-D192/V128/page64/sink contract: "
            f"q_dtype={q.dtype}, Q/KV/QK/V="
            f"{num_q_heads}/{num_kv_heads}/{layer.qk_head_dim}/{layer.v_head_dim}, "
            f"page_size={backend.page_size}, sink_dtype="
            f"{getattr(sinks, 'dtype', None)}, sink_device="
            f"{getattr(sinks, 'device', None)}, sink_contiguous="
            f"{sinks.is_contiguous() if isinstance(sinks, torch.Tensor) else False}, "
            f"logit_cap={backend.logits_soft_cap}."
        )

    if pa_decode_gluon is None or get_recommended_splits is None:
        raise RuntimeError(
            "AITER pa_decode_gluon is unavailable for vectorized-5D TARGET_VERIFY"
        )

    if is_swa_layer:
        block_tables = (
            backend.forward_metadata.swa_page_table
            if backend.forward_metadata.swa_page_table is not None
            else backend.forward_metadata.kv_indices
        )
        max_part_num = 1
        sliding_window = int(layer.sliding_window_size)
    else:
        block_tables = backend.forward_metadata.kv_indices
        max_part_num = int(get_recommended_splits(batch_size, num_kv_heads))
        sliding_window = 0

    q_view = q.view(-1, num_q_heads, layer.qk_head_dim)
    output_view = output.view(-1, num_q_heads, layer.v_head_dim)
    context_lengths = forward_batch.seq_lens + query_length
    kernel_batch_size = batch_size
    kernel_query_length = query_length
    kernel_query_group_size = equivalent_group_size
    if split_verify_queries:
        if is_swa_layer and sliding_window > 0:
            # Gluon's qlen-1 mask starts at context_length - window + 1.
            # Our context length already includes the current query token;
            # keep exactly layer.sliding_window_size keys visible.
            sliding_window += 1
        _ensure_swa_target_verify_workspace(
            backend,
            max_batch_size=batch_size,
            query_length=query_length,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            value_head_dim=layer.v_head_dim,
            max_blocks_per_sequence=block_tables.shape[1],
            max_partitions=max_part_num,
            device=q.device,
            output_dtype=q_view.dtype,
            context_dtype=forward_batch.seq_lens.dtype,
        )
        kernel_batch_size = batch_size * query_length
        kernel_query_length = 1
        kernel_query_group_size = query_group_size
        context_lengths = backend._swa_target_verify_context_lengths[
            :kernel_batch_size
        ].view(batch_size, query_length)
        torch.add(
            forward_batch.seq_lens[:batch_size, None],
            backend._swa_target_verify_offsets[None, :query_length],
            out=context_lengths,
        )
        context_lengths = context_lengths.view(-1)
        block_table_rows = backend._swa_target_verify_block_tables[
            :kernel_batch_size, : block_tables.shape[1]
        ].view(batch_size, query_length, block_tables.shape[1])
        block_table_rows.copy_(block_tables[:, None, :].expand_as(block_table_rows))
        block_tables = block_table_rows.view(kernel_batch_size, -1)
    workspace_shape = (
        kernel_batch_size,
        num_kv_heads,
        max_part_num,
        kernel_query_group_size,
    )
    if split_verify_queries:
        scalar_numel = (
            kernel_batch_size * num_kv_heads * max_part_num * kernel_query_group_size
        )
        exp_sums = backend._swa_target_verify_exp_sums[:scalar_numel].view(
            workspace_shape
        )
        max_logits = backend._swa_target_verify_max_logits[:scalar_numel].view(
            workspace_shape
        )
        temporary_output = backend._swa_target_verify_temporary_output[
            : scalar_numel * layer.v_head_dim
        ].view(*workspace_shape, layer.v_head_dim)
    else:
        exp_sums = torch.empty(
            workspace_shape, dtype=torch.float32, device=q_view.device
        )
        max_logits = torch.empty_like(exp_sums)
        temporary_output = torch.empty(
            (*workspace_shape, layer.v_head_dim),
            dtype=q_view.dtype,
            device=q_view.device,
        )

    key_scale = None
    value_scale = None
    if backend.kv_cache_dtype == fp8_dtype:
        key_scale = layer.k_scale if layer.k_scale is not None else backend.k_scale
        value_scale = layer.v_scale if layer.v_scale is not None else backend.v_scale

    pa_decode_gluon(
        output=output_view,
        query=q_view,
        key_cache=k_cache,
        value_cache=v_cache,
        context_lengths=context_lengths,
        block_tables=block_tables,
        softmax_scale=layer.scaling,
        query_length=kernel_query_length,
        max_context_partition_num=max_part_num,
        context_partition_size=256,
        compute_type=backend.input_dtype,
        key_scale=key_scale,
        value_scale=value_scale,
        exp_sums=exp_sums,
        max_logits=max_logits,
        temporary_output=temporary_output,
        sinks=sinks,
        sliding_window=sliding_window,
        ps=True,
    )
