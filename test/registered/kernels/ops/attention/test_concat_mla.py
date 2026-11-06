import itertools
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import triton

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=17, stage="base-b-kernel-unit", runner_config="1-gpu-large")


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
    from sglang.kernels.ops.attention.concat_mla import concat_mla_k

    concat_mla_k(k, k_nope, k_rope)


def jit_concat_mla_absorb_q(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor
) -> None:
    """JIT compiled implementation - wrapper for test compatibility."""
    from sglang.kernels.ops.attention.concat_mla import concat_mla_absorb_q

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


@pytest.mark.parametrize(
    "num_tokens,num_heads",
    [(0, 12), (1, 12), (17, 12), (257, 24), (19, 8), (65536, 12)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_model_mha_k_concat(num_tokens, num_heads, dtype):
    """Exercise model dispatch with strided KV projection and latent RoPE views."""
    from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mha

    attn = SimpleNamespace(
        num_local_heads=num_heads,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        qk_head_dim=192,
        current_attention_backend="trtllm_mla",
        kv_cache_dtype="fp8_e4m3",
    )
    kv = torch.randn(num_tokens, num_heads, 256, device="cuda", dtype=dtype)
    latent = torch.randn(num_tokens, 1, 576, device="cuda", dtype=dtype)
    k_nope, k_rope = kv[..., :128], latent[..., 512:]
    expected = torch.empty(num_tokens, num_heads, 192, device="cuda", dtype=dtype)
    torch_concat_mla_k(expected, k_nope, k_rope)
    with patch.object(
        forward_mha,
        "concat_and_cast_mha_k_triton",
        wraps=forward_mha.concat_and_cast_mha_k_triton,
    ) as concat:
        actual = forward_mha.DeepseekMHAForwardMixin._concat_and_cast_mha_k(
            attn, k_nope, k_rope, None
        )
    concat.assert_called_once()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_model_mha_k_concat_graph():
    from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mha

    attn = SimpleNamespace(
        num_local_heads=12,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        qk_head_dim=192,
        current_attention_backend="trtllm_mla",
        kv_cache_dtype="fp8_e4m3",
    )
    kv = torch.randn(17, 12, 256, device="cuda", dtype=torch.bfloat16)
    latent = torch.randn(17, 1, 576, device="cuda", dtype=torch.bfloat16)
    k_nope, k_rope = kv[..., :128], latent[..., 512:]
    concat = forward_mha.DeepseekMHAForwardMixin._concat_and_cast_mha_k
    concat(attn, k_nope, k_rope, None)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = concat(attn, k_nope, k_rope, None)
    for _ in range(2):
        kv.normal_()
        latent.normal_()
        graph.replay()
        expected = torch.empty_like(actual)
        torch_concat_mla_k(expected, k_nope, k_rope)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_model_mha_k_concat_fa3_fp8():
    from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mha

    attn = SimpleNamespace(
        num_local_heads=12,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        qk_head_dim=192,
        current_attention_backend="fa3",
        kv_cache_dtype="fp8_e4m3",
    )
    k_nope = torch.randn(17, 12, 128, device="cuda", dtype=torch.bfloat16)
    k_rope = torch.randn(17, 1, 64, device="cuda", dtype=torch.bfloat16)
    expected = torch.empty(17, 12, 192, device="cuda", dtype=torch.bfloat16)
    torch_concat_mla_k(expected, k_nope, k_rope)
    with patch.object(
        forward_mha,
        "get_token_to_kv_pool",
        return_value=SimpleNamespace(dtype=torch.float8_e4m3fn),
    ):
        actual = forward_mha.DeepseekMHAForwardMixin._concat_and_cast_mha_k(
            attn, k_nope, k_rope, None
        )
    assert actual.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(
        actual.view(torch.uint8),
        expected.to(actual.dtype).view(torch.uint8),
        rtol=0,
        atol=0,
    )


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
    sys.exit(pytest.main([__file__, "-v", "-s"]))
