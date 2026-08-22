"""Uniform-FP8 (trtllm backend) DSV4 compressor store path.

Mirrors the packed FlashMLA pipeline: ``compress_forward`` (softmax pooling)
followed by the fused ``compress_norm_rope_store`` CUDA kernel with the
uniform-FP8 epilogue -- RMSNorm + RoPE + plain e4m3 cast (per-tensor scale
1.0) + paged scatter into the 512-byte-per-token uniform pool, one launch.
The kernel reads positions, destination slots, decode window boundaries and
prefill-row validity from the compress plan, so no Python-side masking or
position extraction is needed (same contract as the packed epilogue).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.attention.dsv4 import compress_forward, compress_norm_rope_store

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsv4.compressor import Compressor
    from sglang.srt.layers.attention.dsv4.compressor_v2 import CompressorBackendMixin
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool


def forward_compress_uniform_fp8(
    backend: CompressorBackendMixin,
    *,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    kv_score_input: torch.Tensor,
    state_pool,
    compressor: Compressor,
    layer_id: int,
) -> None:
    """Compress + fused norm/RoPE/e4m3 store for the uniform-FP8 pool."""
    from sglang.srt.layers.attention.dsv4.compressor_v2 import (
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

    # One fused launch: RMSNorm + RoPE + e4m3 cast + paged scatter. The
    # kernel skips invalid prefill plan rows and non-boundary decode tokens,
    # and resolves each row's position / destination from the plan.
    kv_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
    page_size = token_to_kv_pool.get_extra_key_page_size(layer_id)
    compress_norm_rope_store(
        kv_compressed,
        plan,
        norm_weight=compressor.norm.weight,
        norm_eps=compressor.norm.variance_epsilon,
        freq_cis=compressor.freqs_cis,
        out_loc=out_loc,
        kvcache=kv_cache.view(torch.uint8),
        page_size=page_size,
        uniform_fp8_store=True,
    )
