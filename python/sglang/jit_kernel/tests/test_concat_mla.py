import itertools

import pytest
import torch
import triton


def torch_concat_mla_k(
    k: torch.Tensor, k_nope: torch.Tensor, k_rope: torch.Tensor
) -> None:
    """Reference PyTorch implementation for concat_mla_k."""
    # k_nope: [num_tokens, num_heads, nope_head_dim]
    # k_rope: [num_tokens, 1, rope_head_dim]
    # k: [num_tokens, num_heads, nope_head_dim + rope_head_dim]
    nope_head_dim = k_nope.shape[-1]
    k[:, :, :nope_head_dim] = k_nope
    # Broadcast k_rope across all heads
    k[:, :, nope_head_dim:] = k_rope.expand(-1, k.shape[1], -1)


def torch_concat_mla_absorb_q(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor
) -> None:
    """Reference PyTorch implementation for concat_mla_absorb_q."""
    # a: [dim_0, dim_1, a_last_dim]
    # b: [dim_0, dim_1, b_last_dim]
    # out: [dim_0, dim_1, a_last_dim + b_last_dim]
    a_last_dim = a.shape[-1]
    out[:, :, :a_last_dim] = a
    out[:, :, a_last_dim:] = b


def sgl_kernel_concat_mla_k(
    k: torch.Tensor, k_nope: torch.Tensor, k_rope: torch.Tensor
) -> None:
    """AOT compiled sgl_kernel implementation."""
    from sgl_kernel import concat_mla_k

    concat_mla_k(k, k_nope, k_rope)


def sgl_kernel_concat_mla_absorb_q(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor
) -> None:
    """AOT compiled sgl_kernel implementation."""
    from sgl_kernel import concat_mla_absorb_q

    result = concat_mla_absorb_q(a, b)  # AOT returns output
    out.copy_(result)  # Copy to provided tensor for comparison


def jit_concat_mla_k(
    k: torch.Tensor, k_nope: torch.Tensor, k_rope: torch.Tensor
) -> None:
    """JIT compiled implementation."""
    from sglang.jit_kernel.concat_mla import concat_mla_k

    concat_mla_k(k, k_nope, k_rope)


def jit_concat_mla_absorb_q(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor
) -> None:
    """JIT compiled implementation - wrapper for test compatibility."""
    from sglang.jit_kernel.concat_mla import concat_mla_absorb_q

    result = concat_mla_absorb_q(a, b)
    out.copy_(result)


# Constants matching the kernel
NUM_LOCAL_HEADS = 128
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
K_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM

A_LAST_DIM = 512
B_LAST_DIM = 64
OUT_LAST_DIM = A_LAST_DIM + B_LAST_DIM

DEVICE = "cuda"
DTYPE = torch.bfloat16

# Test configurations
NUM_TOKENS_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS_LIST)
def test_concat_mla_k_jit_vs_torch(num_tokens: int) -> None:
    """Test JIT kernel against PyTorch reference."""
    k_jit = torch.empty(
        num_tokens, NUM_LOCAL_HEADS, K_HEAD_DIM, device=DEVICE, dtype=DTYPE
    )
    k_torch = torch.empty(
        num_tokens, NUM_LOCAL_HEADS, K_HEAD_DIM, device=DEVICE, dtype=DTYPE
    )

    k_nope = torch.randn(
        num_tokens, NUM_LOCAL_HEADS, QK_NOPE_HEAD_DIM, device=DEVICE, dtype=DTYPE
    )
    k_rope = torch.randn(num_tokens, 1, QK_ROPE_HEAD_DIM, device=DEVICE, dtype=DTYPE)

    torch_concat_mla_k(k_torch, k_nope, k_rope)
    jit_concat_mla_k(k_jit, k_nope, k_rope)

    triton.testing.assert_close(k_jit, k_torch, atol=0, rtol=0)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS_LIST)
def test_concat_mla_k_jit_vs_aot(num_tokens: int) -> None:
    """Test JIT kernel against AOT kernel for bitwise equivalence."""
    k_jit = torch.empty(
        num_tokens, NUM_LOCAL_HEADS, K_HEAD_DIM, device=DEVICE, dtype=DTYPE
    )
    k_aot = torch.empty(
        num_tokens, NUM_LOCAL_HEADS, K_HEAD_DIM, device=DEVICE, dtype=DTYPE
    )

    k_nope = torch.randn(
        num_tokens, NUM_LOCAL_HEADS, QK_NOPE_HEAD_DIM, device=DEVICE, dtype=DTYPE
    )
    k_rope = torch.randn(num_tokens, 1, QK_ROPE_HEAD_DIM, device=DEVICE, dtype=DTYPE)

    sgl_kernel_concat_mla_k(k_aot, k_nope, k_rope)
    jit_concat_mla_k(k_jit, k_nope, k_rope)

    triton.testing.assert_close(k_jit, k_aot, atol=0, rtol=0)


DIM_0_LIST = [1, 2, 4, 8, 16, 32]
DIM_1_LIST = [1, 2, 4, 8, 16, 128]


@pytest.mark.parametrize(
    "dim_0,dim_1",
    list(itertools.product(DIM_0_LIST, DIM_1_LIST)),
)
def test_concat_mla_absorb_q_jit_vs_torch(dim_0: int, dim_1: int) -> None:
    """Test JIT kernel against PyTorch reference."""
    a = torch.randn(dim_0, dim_1, A_LAST_DIM, device=DEVICE, dtype=DTYPE)
    b = torch.randn(dim_0, dim_1, B_LAST_DIM, device=DEVICE, dtype=DTYPE)
    out_jit = torch.empty(dim_0, dim_1, OUT_LAST_DIM, device=DEVICE, dtype=DTYPE)
    out_torch = torch.empty(dim_0, dim_1, OUT_LAST_DIM, device=DEVICE, dtype=DTYPE)

    torch_concat_mla_absorb_q(a, b, out_torch)
    jit_concat_mla_absorb_q(a, b, out_jit)

    triton.testing.assert_close(out_jit, out_torch, atol=0, rtol=0)


@pytest.mark.parametrize(
    "dim_0,dim_1",
    list(itertools.product(DIM_0_LIST, DIM_1_LIST)),
)
def test_concat_mla_absorb_q_jit_vs_aot(dim_0: int, dim_1: int) -> None:
    """Test JIT kernel against AOT kernel for bitwise equivalence."""
    a = torch.randn(dim_0, dim_1, A_LAST_DIM, device=DEVICE, dtype=DTYPE)
    b = torch.randn(dim_0, dim_1, B_LAST_DIM, device=DEVICE, dtype=DTYPE)
    out_jit = torch.empty(dim_0, dim_1, OUT_LAST_DIM, device=DEVICE, dtype=DTYPE)
    out_aot = torch.empty(dim_0, dim_1, OUT_LAST_DIM, device=DEVICE, dtype=DTYPE)

    sgl_kernel_concat_mla_absorb_q(a, b, out_aot)
    jit_concat_mla_absorb_q(a, b, out_jit)

    triton.testing.assert_close(out_jit, out_aot, atol=0, rtol=0)


if __name__ == "__main__":
    pytest.main([__file__])
