import itertools
import sys

import pytest
import torch

from sglang.srt.models.utils import fused_qk_gemma_rmsnorm_with_gate
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, suite="jit-kernel-unit-test-amd")


def reference_qk_gemma_rmsnorm_with_gate(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    head_dim: int,
    num_heads: int,
):
    """Pure-PyTorch reference: deinterleave q/gate, GemmaRMSNorm q and k."""
    seq_len = q_gate.shape[0]

    # Deinterleave q and gate from [q_h0, gate_h0, q_h1, gate_h1, ...]
    qg_3d = q_gate.view(seq_len, num_heads, 2 * head_dim)
    q = qg_3d[:, :, :head_dim].contiguous().view(-1, head_dim)
    gate = qg_3d[:, :, head_dim:].contiguous().view(-1, head_dim)

    k_flat = k.reshape(-1, head_dim)

    # GemmaRMSNorm: x * rsqrt(mean(x^2) + eps) * (weight + 1)
    def gemma_rmsnorm(x, w):
        x_fp32 = x.float()
        var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        normed = x_fp32 * (var + eps).rsqrt() * (w.float() + 1.0)
        return normed.to(x.dtype)

    q_out = gemma_rmsnorm(q, q_weight)
    k_out = gemma_rmsnorm(k_flat, k_weight)

    return q_out, k_out, gate


DEVICE = "cuda"
DTYPE = torch.bfloat16

SEQ_LENS = [1, 2, 4, 7, 16, 128]
NUM_HEADS_LIST = [8, 16, 32]
NUM_KV_HEADS_LIST = [2, 4, 8]
HEAD_DIM_LIST = [64, 128]


@pytest.mark.parametrize(
    "seq_len,num_heads,num_kv_heads,head_dim",
    list(itertools.product(SEQ_LENS, NUM_HEADS_LIST, NUM_KV_HEADS_LIST, HEAD_DIM_LIST)),
)
def test_fused_qk_gemma_rmsnorm_with_gate(
    seq_len: int, num_heads: int, num_kv_heads: int, head_dim: int
):
    if num_kv_heads > num_heads:
        pytest.skip("num_kv_heads > num_heads is not a valid config")

    eps = 1e-6
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim

    # Build a full qkv buffer and split — this gives non-contiguous k,
    # which is the real usage pattern
    qkv = torch.randn(
        seq_len, q_size * 2 + kv_size + kv_size, device=DEVICE, dtype=DTYPE
    )
    q_gate, k, v = qkv.split([q_size * 2, kv_size, kv_size], dim=-1)

    q_weight = torch.randn(head_dim, device=DEVICE, dtype=DTYPE)
    k_weight = torch.randn(head_dim, device=DEVICE, dtype=DTYPE)

    # Reference
    q_ref, k_ref, gate_ref = reference_qk_gemma_rmsnorm_with_gate(
        q_gate, k, q_weight, k_weight, eps, head_dim, num_heads
    )

    # Fused kernel
    q_out, k_out, gate_out = fused_qk_gemma_rmsnorm_with_gate(
        q_gate, k, q_weight, k_weight, eps, head_dim, num_heads
    )

    torch.testing.assert_close(q_out, q_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(k_out, k_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(gate_out, gate_ref, atol=0, rtol=0)


@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_gate_is_exact_copy(head_dim: int):
    """Gate output must be a bitwise-exact copy of the input gate data."""
    seq_len = 4
    num_heads = 16
    num_kv_heads = 4
    eps = 1e-6
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim

    qkv = torch.randn(
        seq_len, q_size * 2 + kv_size + kv_size, device=DEVICE, dtype=DTYPE
    )
    q_gate, k, v = qkv.split([q_size * 2, kv_size, kv_size], dim=-1)
    q_weight = torch.randn(head_dim, device=DEVICE, dtype=DTYPE)
    k_weight = torch.randn(head_dim, device=DEVICE, dtype=DTYPE)

    _, _, gate_out = fused_qk_gemma_rmsnorm_with_gate(
        q_gate, k, q_weight, k_weight, eps, head_dim, num_heads
    )

    # Extract gate from interleaved buffer manually
    qg_3d = q_gate.view(seq_len, num_heads, 2 * head_dim)
    gate_expected = qg_3d[:, :, head_dim:].contiguous().view(-1, head_dim)

    assert torch.equal(gate_out, gate_expected), "Gate must be bitwise exact"


@pytest.mark.parametrize("seq_len", [1, 8])
def test_contiguous_k_also_works(seq_len: int):
    """Kernel should work even when k is already contiguous."""
    num_heads = 16
    num_kv_heads = 4
    head_dim = 128
    eps = 1e-6
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim

    q_gate = torch.randn(seq_len, q_size * 2, device=DEVICE, dtype=DTYPE)
    k = torch.randn(seq_len, kv_size, device=DEVICE, dtype=DTYPE)
    assert k.is_contiguous()

    q_weight = torch.randn(head_dim, device=DEVICE, dtype=DTYPE)
    k_weight = torch.randn(head_dim, device=DEVICE, dtype=DTYPE)

    q_ref, k_ref, gate_ref = reference_qk_gemma_rmsnorm_with_gate(
        q_gate, k, q_weight, k_weight, eps, head_dim, num_heads
    )
    q_out, k_out, gate_out = fused_qk_gemma_rmsnorm_with_gate(
        q_gate, k, q_weight, k_weight, eps, head_dim, num_heads
    )

    torch.testing.assert_close(q_out, q_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(k_out, k_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(gate_out, gate_ref, atol=0, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
