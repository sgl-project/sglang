"""Compatibility shim for migrated fused compressor kernel entrypoints."""

from sglang.srt.layers.attention.dsv4.fused_compress_triton import (
    fused_ape_pool_norm_rope,
)

__all__ = ["fused_ape_pool_norm_rope"]
