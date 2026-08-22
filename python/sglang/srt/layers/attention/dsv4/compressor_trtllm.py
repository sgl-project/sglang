"""Uniform-FP8 (trtllm backend) DSV4 compressor store path.

The fused JIT compressor epilogue writes the packed FlashMLA cache layout
(448-dim FP8 NoPE + 64-dim BF16 RoPE + block scales); the trtllm backend's
uniform 512-dim FP8 pool needs a different epilogue. This module carries a
standalone unfused pipeline for that pool -- compress, invalid-row masking,
norm + RoPE, then a plain e4m3 cast store -- so the shared
``compressor_v2.py`` (FlashMLA / HIP) stays untouched.

The pipeline intentionally duplicates the compress/norm/RoPE steps of
``CompressorBackendMixin._forward_unified_hip`` rather than refactoring
them out of the shared file. A follow-up fused uniform-FP8 store (see PR
#32975) replaces this module wholesale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.attention.dsv4 import compress_forward

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsv4.compressor import Compressor
    from sglang.srt.layers.attention.dsv4.compressor_v2 import CompressorBackendMixin
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool


def _mask_invalid_prefill_compress_rows(
    kv_compressed: torch.Tensor,
    plan_raw: torch.Tensor,
    out_loc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Make padded prefill-plan rows inert before the unfused store.

    ``PlanC::invalid()`` uses ``seq_len == -1`` and a ``ragged_id`` of
    ``0xffff``.  The fused epilogue checks ``is_invalid()`` before reading
    ``out_loc``; the unfused uniform-FP8 path must provide the same guard
    without introducing a dynamic-shape mask (it is also used by BCG).
    """
    valid = plan_raw[:, 0] != -1
    kv_compressed = torch.where(
        valid.unsqueeze(-1), kv_compressed, torch.zeros_like(kv_compressed)
    )

    ragged_ids = plan_raw[:, 1].to(torch.int32) & 0xFFFF
    safe_ragged_ids = torch.where(valid, ragged_ids, torch.zeros_like(ragged_ids))
    mapped_out_loc = out_loc[safe_ragged_ids.long()]
    # Slot 0 is the allocator's reserved padding sink, so duplicate invalid
    # writes cannot collide with a live cache entry.
    out_loc_to_store = torch.where(
        valid, mapped_out_loc, torch.zeros_like(mapped_out_loc)
    )
    return kv_compressed, out_loc_to_store


def forward_compress_uniform_fp8(
    backend: CompressorBackendMixin,
    *,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    kv_score_input: torch.Tensor,
    state_pool,
    compressor: Compressor,
    layer_id: int,
) -> None:
    """Unfused compress + norm + RoPE + e4m3 store for the uniform-FP8 pool.

    The compression math is the same JIT kernel as the fused path; only the
    epilogue differs (plain FP8 cast into the 512-dim uniform layout).
    """
    from sglang.kernels.ops.attention.deepseek_v4_rope import (
        fused_norm_rope_inplace_triton,
    )
    from sglang.srt.layers.attention.dsv4.compressor_v2 import (
        _extract_positions_from_plan,
        _use_online_compress,
        is_overlap_compress,
    )

    assert not compressor.is_in_indexer
    assert compressor.head_dim == 512, f"{compressor.head_dim=}"
    assert not _use_online_compress(compressor.ratio), (
        "SGLANG_OPT_USE_ONLINE_COMPRESS is not supported with the "
        "uniform-FP8 KV layout yet."
    )

    compress_ratio = compressor.ratio
    head_dim = compressor.head_dim

    plan = backend._get_paged_compress_metadata(compress_ratio)
    out_loc = backend._get_out_loc(compress_ratio)

    coff = 2 if is_overlap_compress(compress_ratio) else 1
    kv_score_buffer = state_pool.kv_score_buffer.kv_score.view(
        -1, compress_ratio, 2 * head_dim * coff
    )

    kv_compressed = compress_forward(
        kv_score_buffer=kv_score_buffer,
        kv_score_input=kv_score_input,
        ape=compressor.ape.view(-1, head_dim),
        plan=plan,
        compress_ratio=compress_ratio,
        head_dim=head_dim,
        is_online=False,
    )
    if kv_compressed.shape[0] == 0:
        return

    plan_raw = plan[1].view(torch.int32)
    if plan.is_decode:
        # Zero out non-boundary tokens to prevent corrupting kvcache loc 0.
        seq_lens_plan = plan_raw[:, 0].to(torch.int32)
        is_boundary = (seq_lens_plan % compress_ratio == 0).unsqueeze(-1)
        kv_compressed = torch.where(
            is_boundary, kv_compressed, torch.zeros_like(kv_compressed)
        )
        out_loc_to_store = out_loc
    else:
        kv_compressed, out_loc_to_store = _mask_invalid_prefill_compress_rows(
            kv_compressed,
            plan_raw,
            out_loc,
        )

    positions = _extract_positions_from_plan(plan, compress_ratio).clamp(min=0)
    fused_norm_rope_inplace_triton(
        kv_compressed,
        compressor.norm.weight,
        compressor.norm.variance_epsilon,
        compressor.freqs_cis,
        positions=positions,
    )

    token_to_kv_pool.set_extra_key_buffer_fused(
        layer_id=layer_id,
        loc=out_loc_to_store,
        cache_k=kv_compressed,
    )
