"""
Test the new FP8 workflow: quantize_k_cache_separate + set_mla_kv_buffer_triton
This tests the optimization that avoids concat by quantizing nope/rope separately.
"""

import torch

from sglang.srt.layers.attention.nsa.quant_k_cache import (
    quantize_k_cache,
    quantize_k_cache_separate,
)
from sglang.srt.mem_cache.utils import set_mla_kv_buffer_triton


def test_quantize_separate_matches_concat():
    """Verify that quantize_k_cache_separate produces same output as concat path."""
    device = "cuda"
    num_tokens = 64
    dim_nope = 512
    dim_rope = 64

    # Generate test data
    k_nope = torch.randn(num_tokens, 1, dim_nope, dtype=torch.bfloat16, device=device)
    k_rope = torch.randn(num_tokens, 1, dim_rope, dtype=torch.bfloat16, device=device)

    # Old path: concat then quantize
    k_concat = torch.cat([k_nope, k_rope], dim=-1)
    # quantize_k_cache expects 4D: (num_blocks, block_size, h_k=1, d)
    k_concat_4d = k_concat.unsqueeze(0)  # (1, num_tokens, 1, 576)
    old_output_4d = quantize_k_cache(k_concat_4d)  # (1, num_tokens, 1, 656)
    old_output = old_output_4d.squeeze(0).squeeze(1)  # (num_tokens, 656)

    # New path: quantize separately
    nope_part, rope_part = quantize_k_cache_separate(k_nope, k_rope)
    # Concatenate the two parts to compare
    new_output = torch.cat([nope_part.squeeze(1), rope_part.squeeze(1)], dim=-1)

    # Compare (convert both to uint8 bytes for comparison)
    old_bytes = old_output.view(torch.uint8)
    new_bytes = new_output  # Already uint8

    assert (
        old_bytes.shape == new_bytes.shape
    ), f"Shape mismatch: {old_bytes.shape} vs {new_bytes.shape}"
    assert torch.equal(
        old_bytes, new_bytes
    ), "Separate quantize output doesn't match concat path"


def test_end_to_end_kv_write():
    """Test complete workflow: separate quantize + two-tensor write."""
    device = "cuda"
    num_tokens = 100
    kv_buffer_size = 1000
    dim_nope = 512
    dim_rope = 64

    # FP8 packed layout: 512 (nope_fp8) + 16 (scales) + 128 (rope_bf16) = 656 bytes
    kv_cache_dim_bytes = 656

    # Generate test data
    k_nope = torch.randn(num_tokens, 1, dim_nope, dtype=torch.bfloat16, device=device)
    k_rope = torch.randn(num_tokens, 1, dim_rope, dtype=torch.bfloat16, device=device)
    loc = torch.randperm(kv_buffer_size, dtype=torch.int32, device=device)[:num_tokens]

    # Create kv_buffer (same as what MLA FP8 cache uses)
    kv_buffer_old = torch.zeros(
        kv_buffer_size, 1, kv_cache_dim_bytes, dtype=torch.uint8, device=device
    )
    kv_buffer_new = torch.zeros(
        kv_buffer_size, 1, kv_cache_dim_bytes, dtype=torch.uint8, device=device
    )

    # Old path: concat + quantize + index_put
    k_concat = torch.cat([k_nope, k_rope], dim=-1)
    # quantize_k_cache expects 4D: (num_blocks=1, block_size=num_tokens, h_k=1, d)
    k_concat_4d = k_concat.unsqueeze(0)
    cache_k_old_4d = quantize_k_cache(k_concat_4d)  # (1, num_tokens, 1, 656)
    cache_k_old = cache_k_old_4d.squeeze(0)  # (num_tokens, 1, 656)
    cache_k_old = cache_k_old.view(torch.uint8)
    kv_buffer_old[loc] = cache_k_old

    # New path: separate quantize + two-tensor write
    cache_k_nope_fp8, cache_k_rope_fp8 = quantize_k_cache_separate(k_nope, k_rope)
    set_mla_kv_buffer_triton(
        kv_buffer_new,
        loc,
        cache_k_nope_fp8,
        cache_k_rope_fp8,
    )

    # Compare results
    assert torch.equal(
        kv_buffer_old, kv_buffer_new
    ), "New workflow doesn't match old path"
