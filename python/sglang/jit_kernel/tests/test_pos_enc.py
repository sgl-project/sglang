import time

import pytest
import torch

from sglang.jit_kernel.pos_enc import rotary_embedding


def create_test_inputs(
    head_size, batch_size, seq_len, device, dtype, num_q_heads, num_kv_heads
):
    """Create test inputs."""
    total_tokens = batch_size * seq_len

    query = torch.randn(
        batch_size, seq_len, num_q_heads, head_size, dtype=dtype, device=device
    )
    key = torch.randn(
        batch_size, seq_len, num_kv_heads, head_size, dtype=dtype, device=device
    )

    pos_ids = torch.randint(
        0, min(seq_len * 2, 100), (total_tokens,), dtype=torch.long, device=device
    )

    query = query.view(total_tokens, num_q_heads, head_size)
    key = key.view(total_tokens, num_kv_heads, head_size)

    return query, key, pos_ids


def create_cos_sin_cache(rotary_dim, max_position_embeddings, base, dtype, device):
    """Create cos/sin cache for rotary embedding."""
    max_pos = max_position_embeddings
    extended_max_pos = max(max_pos, 100)
    cos_sin_cache = torch.zeros(
        extended_max_pos, rotary_dim, dtype=dtype, device=device
    )

    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim
        )
    )
    t = torch.arange(extended_max_pos, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    cos_cache = torch.cos(freqs).to(dtype)
    sin_cache = torch.sin(freqs).to(dtype)

    cos_sin_cache[:, : rotary_dim // 2] = cos_cache
    cos_sin_cache[:, rotary_dim // 2 :] = sin_cache

    return cos_sin_cache


def get_sgl_rotary_embedding(
    head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype, device
):
    """Initialize SglKernelRotaryEmbedding."""
    try:
        from sgl_kernel.testing.rotary_embedding import SglKernelRotaryEmbedding
    except ImportError:
        pytest.skip("SglKernelRotaryEmbedding is not available.")

    return SglKernelRotaryEmbedding(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=max_position_embeddings,
        base=base,
        is_neox_style=is_neox_style,
        dtype=dtype,
    ).to(device)


def compare_results(jit_out, sgl_out, dtype):
    """Compare results between JIT and SGL implementations."""
    if jit_out is None:
        assert sgl_out is None
        return

    assert sgl_out is not None

    # Check for NaN values
    assert not torch.isnan(jit_out).any(), "NaN in JIT results"
    assert not torch.isnan(sgl_out).any(), "NaN in SGL results"

    # Compare results
    atol = 1e-2 if dtype != torch.float32 else 1e-5
    rtol = 1e-2 if dtype != torch.float32 else 1e-5

    torch.testing.assert_close(jit_out, sgl_out, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype, device, batch_size, seq_len, num_q_heads, num_kv_heads",
    [
        # GPT-OSS cases
        *[
            (64, 64, 4096, 8000, True, torch.bfloat16, "cuda", bs, sl, 8, 8)
            for bs, sl in [(1, 1), (32, 1), (128, 1), (512, 1), (2, 512), (4, 4096)]
        ],
        # Other cases
        (64, 64, 32, 8000, True, torch.bfloat16, "cuda", 32, 32, 1, 1),
        (256, 128, 4096, 10000, True, torch.bfloat16, "cuda", 2, 512, 4, 2),
        (512, 128, 311, 10000, True, torch.bfloat16, "cuda", 3, 39, 4, 2),
        (128, 128, 2048, 10000, False, torch.bfloat16, "cuda", 2, 512, 32, 8),
        (128, 128, 2048, 10000, False, torch.bfloat16, "cuda", 2, 512, 16, 4),
        (512, 128, 311, 10000, False, torch.bfloat16, "cuda", 3, 39, 4, 2),
        (64, 64, 32, 8000, True, torch.float32, "cuda", 32, 32, 1, 1),
        (256, 128, 4096, 10000, True, torch.float32, "cuda", 2, 512, 4, 2),
        (512, 128, 311, 10000, True, torch.float32, "cuda", 3, 39, 4, 2),
        (128, 128, 2048, 10000, False, torch.float32, "cuda", 2, 512, 32, 8),
        (128, 128, 2048, 10000, False, torch.float32, "cuda", 2, 512, 16, 4),
        (512, 128, 311, 10000, False, torch.float32, "cuda", 3, 39, 4, 2),
        # Additional test cases for different head sizes and dtypes
        (64, 32, 1024, 10000, True, torch.float16, "cuda", 16, 64, 8, 4),
        (128, 64, 2048, 10000, True, torch.float16, "cuda", 8, 128, 16, 8),
        (256, 128, 4096, 10000, True, torch.float16, "cuda", 4, 256, 8, 4),
    ],
)
@pytest.mark.parametrize(
    "key_is_none",
    [True, False],
)
def test_correctness(
    head_size,
    rotary_dim,
    max_position_embeddings,
    base,
    is_neox_style,
    dtype,
    device,
    batch_size,
    seq_len,
    num_q_heads,
    num_kv_heads,
    key_is_none,
):
    """Test correctness of JIT rotary embedding implementation."""
    # Create inputs and caches
    query, key, pos_ids = create_test_inputs(
        head_size, batch_size, seq_len, device, dtype, num_q_heads, num_kv_heads
    )
    cos_sin_cache = create_cos_sin_cache(
        rotary_dim, max_position_embeddings, base, dtype, device
    )

    # Initialize SGL kernel
    sgl_rotary_emb = get_sgl_rotary_embedding(
        head_size,
        rotary_dim,
        max_position_embeddings,
        base,
        is_neox_style,
        dtype,
        device,
    )
    sgl_rotary_emb.cos_sin_cache = cos_sin_cache

    # Apply rotary embeddings
    query_jit, key_jit = query.clone(), key.clone()
    query_sgl, key_sgl = query.clone(), key.clone()

    if key_is_none:
        key_jit = None
        key_sgl = None
    query_jit_out, key_jit_out = rotary_embedding(
        positions=pos_ids,
        query=query_jit,
        key=key_jit,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox_style,
    )

    query_sgl_out, key_sgl_out = sgl_rotary_emb.forward_cuda(
        positions=pos_ids, query=query_sgl, key=key_sgl
    )

    compare_results(query_jit_out, query_sgl_out, dtype)
    compare_results(key_jit_out, key_sgl_out, dtype)


@pytest.mark.parametrize(
    "head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype, device, batch_size, seq_len, num_q_heads, num_kv_heads",
    [
        # Small scale
        (64, 64, 4096, 8000, True, torch.bfloat16, "cuda", 1, 1, 8, 8),
        (64, 64, 4096, 8000, True, torch.bfloat16, "cuda", 4, 16, 8, 8),
        # Medium scale
        (64, 64, 4096, 8000, True, torch.bfloat16, "cuda", 8, 64, 8, 8),
        (64, 64, 4096, 8000, True, torch.bfloat16, "cuda", 16, 128, 8, 8),
        # Large scale
        (64, 64, 4096, 8000, True, torch.bfloat16, "cuda", 32, 512, 8, 8),
        (64, 64, 4096, 8000, True, torch.bfloat16, "cuda", 64, 1024, 8, 8),
    ],
)
def test_performance(
    head_size,
    rotary_dim,
    max_position_embeddings,
    base,
    is_neox_style,
    dtype,
    device,
    batch_size,
    seq_len,
    num_q_heads,
    num_kv_heads,
):
    """Performance test comparing JIT and SGL implementations with accuracy validation."""
    # Create inputs and caches
    query, key, pos_ids = create_test_inputs(
        head_size, batch_size, seq_len, device, dtype, num_q_heads, num_kv_heads
    )
    cos_sin_cache = create_cos_sin_cache(
        rotary_dim, max_position_embeddings, base, dtype, device
    )

    # Initialize SGL kernel
    sgl_rotary_emb = get_sgl_rotary_embedding(
        head_size,
        rotary_dim,
        max_position_embeddings,
        base,
        is_neox_style,
        dtype,
        device,
    )
    sgl_rotary_emb.cos_sin_cache = cos_sin_cache

    warmup = 3

    # Warmup runs
    for _ in range(warmup):
        query_warm, key_warm = query.clone(), key.clone()
        rotary_embedding(
            positions=pos_ids,
            query=query_warm,
            key=key_warm,
            head_size=head_size,
            cos_sin_cache=cos_sin_cache,
            is_neox=is_neox_style,
        )

        query_sgl_warm, key_sgl_warm = query.clone(), key.clone()
        sgl_rotary_emb.forward_cuda(
            positions=pos_ids, query=query_sgl_warm, key=key_sgl_warm
        )

    iteration = 100

    # Time JIT implementation
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(iteration):
        query_jit, key_jit = query.clone(), key.clone()
        rotary_embedding(
            positions=pos_ids,
            query=query_jit,
            key=key_jit,
            head_size=head_size,
            cos_sin_cache=cos_sin_cache,
            is_neox=is_neox_style,
        )
    torch.cuda.synchronize()
    jit_time = (time.time() - start_time) / iteration

    # Time SGL implementation
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(iteration):
        query_sgl, key_sgl = query.clone(), key.clone()
        sgl_rotary_emb.forward_cuda(positions=pos_ids, query=query_sgl, key=key_sgl)
    torch.cuda.synchronize()
    sgl_time = (time.time() - start_time) / iteration

    # Accuracy validation during performance test
    # Run one more time to get outputs for comparison
    query_jit_final, key_jit_final = query.clone(), key.clone()
    query_sgl_final, key_sgl_final = query.clone(), key.clone()

    query_jit_out, key_jit_out = rotary_embedding(
        positions=pos_ids,
        query=query_jit_final,
        key=key_jit_final,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox_style,
    )

    query_sgl_out, key_sgl_out = sgl_rotary_emb.forward_cuda(
        positions=pos_ids, query=query_sgl_final, key=key_sgl_final
    )

    # Validate accuracy
    compare_results(query_jit_out, query_sgl_out, dtype)
    compare_results(key_jit_out, key_sgl_out, dtype)

    # Print results
    total_tokens = batch_size * seq_len
    print(
        f"\nPerformance Test - Batch={batch_size}, SeqLen={seq_len}, Tokens={total_tokens}"
    )
    print(f"JIT: {jit_time*1000:.9f}ms, SGL: {sgl_time*1000:.9f}ms")
    if sgl_time > 0:
        speedup = sgl_time / jit_time if jit_time > 0 else float("inf")
        print(f"Speedup (SGL/JIT): {speedup:.2f}x")

    assert jit_time >= 0 and sgl_time >= 0
