import pytest
import torch
import triton

from sgl_diffusion.runtime.layers.triton_ops import apply_rotary_embedding


def _apply_rotary_emb_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    o1 = (x1.float() * cos - x2.float() * sin).type_as(x)
    o2 = (x2.float() * cos + x1.float() * sin).type_as(x)
    return torch.stack((o1, o2), dim=-1).flatten(-2)


@pytest.mark.parametrize(
    "num_tokens, num_heads, head_size",
    [
        (1, 1, 64),
        (10, 1, 128),
        (128, 4, 64),
        (4096, 8, 128),
        (8192, 16, 256),
    ],
)
def test_rotary_embedding_correctness(num_tokens, num_heads, head_size):
    torch.manual_seed(0)
    x = torch.randn(
        num_tokens, num_heads, head_size, device="cuda", dtype=torch.float16
    )
    cos = torch.randn(num_tokens, head_size // 2, device="cuda", dtype=torch.float16)
    sin = torch.randn(num_tokens, head_size // 2, device="cuda", dtype=torch.float16)

    output_triton = apply_rotary_embedding(x, cos, sin)
    output_torch = _apply_rotary_emb_torch(x, cos, sin)

    assert torch.allclose(output_triton, output_torch, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("num_tokens", [4096, 32760])
@pytest.mark.parametrize("num_heads", [8, 12])
@pytest.mark.parametrize("head_size", [128])
def test_rotary_embedding_performance(num_tokens, num_heads, head_size):
    torch.manual_seed(0)
    x = torch.randn(
        num_tokens, num_heads, head_size, device="cuda", dtype=torch.float16
    )
    cos = torch.randn(num_tokens, head_size // 2, device="cuda", dtype=torch.float16)
    sin = torch.randn(num_tokens, head_size // 2, device="cuda", dtype=torch.float16)

    # Warmup
    apply_rotary_embedding(x, cos, sin)
    _apply_rotary_emb_torch(x, cos, sin)

    triton_ms = triton.testing.do_bench(lambda: apply_rotary_embedding(x, cos, sin))
    torch_ms = triton.testing.do_bench(lambda: _apply_rotary_emb_torch(x, cos, sin))

    print(f"PyTorch implementation: {torch_ms:.4f} ms")
    print(f"Triton implementation:  {triton_ms:.4f} ms")
    print(f"Speedup: {torch_ms / triton_ms:.2f}x")
