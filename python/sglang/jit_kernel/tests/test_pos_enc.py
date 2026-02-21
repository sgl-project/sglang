import time
from typing import Optional, Tuple, Union

import pytest
import torch
import triton
import triton.language as tl

from sglang.jit_kernel.pos_enc import rotary_embedding


@triton.jit
def burn_kernel(out_ptr, iters: tl.constexpr):
    pid = tl.program_id(0)
    x = tl.full((), pid + 1, dtype=tl.uint32)

    a = tl.full((), 1664525, dtype=tl.uint32)
    c = tl.full((), 1013904223, dtype=tl.uint32)
    sh = tl.full((), 13, dtype=tl.uint32)

    for _ in range(iters):
        x = x * a + c
        x = x ^ (x >> sh)

    if pid == 0:
        tl.store(out_ptr, x)


def triton_burn(ms: float, grid=(256,)):
    iters = int(ms * 20000)
    out = torch.empty((), device="cuda", dtype=torch.uint32)
    burn_kernel[grid](out, iters=iters)
    return out


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


# vLLM torch native
def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


class RotaryEmbedding(torch.nn.Module):
    # Reference: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding.py
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        cache = self._compute_cos_sin_cache()
        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
            )
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A PyTorch-native implementation of forward()."""

        if offsets is not None:
            positions = positions + offsets

        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions)

        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        # Modification: convert to the correct dtype
        query = query.to(self.dtype)

        if key is not None:
            key_shape = key.shape
            key = key.view(num_tokens, -1, self.head_size)
            key_rot = key[..., : self.rotary_dim]
            key_pass = key[..., self.rotary_dim :]
            key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
            key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)

            key = key.to(self.dtype)

        return query, key


def get_torch_rotary_embedding(
    head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype, device
):
    """Initialize Torch Native RotaryEmbedding based on vLLM implementation."""
    return RotaryEmbedding(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=max_position_embeddings,
        base=base,
        is_neox_style=is_neox_style,
        dtype=dtype,
    ).to(device)


def get_sgl_rotary_embedding(
    head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype, device
):
    """Initialize SglKernelRotaryEmbedding."""
    try:
        from sgl_kernel.testing.rotary_embedding import SglKernelRotaryEmbedding
    except ImportError:
        pytest.skip(
            "SglKernelRotaryEmbedding is not available. Test case can be removed."
        )

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

    # Initialize torch kernel
    torch_rotary_emb = get_torch_rotary_embedding(
        head_size,
        rotary_dim,
        max_position_embeddings,
        base,
        is_neox_style,
        dtype,
        device,
    )
    torch_rotary_emb.cos_sin_cache = cos_sin_cache
    r = torch.randn_like(query)

    # Apply rotary embeddings
    query_jit, key_jit = query.clone(), key.clone()
    query_torch, key_torch = query.clone(), key.clone()
    stream_jit = torch.get_device_module("cuda").Stream()
    stream_kernel = torch.get_device_module("cuda").Stream()

    if key_is_none:
        key_jit = None
        key_torch = None
    triton_burn(100.0, grid=(1024,))

    r_jit, r_torch = r.clone(), r.clone()
    torch.cuda.synchronize()

    with torch.cuda.stream(stream_jit):
        # Test if rotary_embedding runs on stream_jit
        triton_burn(100.0, grid=(1024,))
        query_jit = query_jit + r_jit
        query_jit_out, key_jit_out = rotary_embedding(
            positions=pos_ids,
            query=query_jit,
            key=key_jit,
            head_size=head_size,
            cos_sin_cache=cos_sin_cache,
            is_neox=is_neox_style,
        )

    with torch.cuda.stream(stream_kernel):
        triton_burn(100.0, grid=(1024,))
        query_torch = query_torch + r_torch
        query_torch_out, key_torch_out = torch_rotary_emb.forward_native(
            positions=pos_ids, query=query_torch, key=key_torch
        )

    torch.cuda.synchronize()
    compare_results(query_jit_out, query_torch_out, dtype)
    compare_results(key_jit_out, key_torch_out, dtype)


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
