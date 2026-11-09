import sys

import pytest
import torch

from sglang.srt.models.deepseek_common.attention_forward_methods.triton_qk_rmsnorm import (
    fused_qk_rmsnorm_torch,
    fused_qk_rmsnorm_triton,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, stage="jit-kernel-unit", runner_config="amd")

# Kimi-K2 / DeepSeek MLA: q_lora_rank, kv_lora_rank, qk_rope_head_dim.
Q_LORA, KV_LORA, ROPE = 1536, 512, 64
EPS = 1e-6


def _fused_qkv_a(m, dtype):
    torch.manual_seed(m)
    x = torch.randn(m, Q_LORA + KV_LORA + ROPE, device="cuda", dtype=dtype)
    q_w = torch.rand(Q_LORA, device="cuda", dtype=dtype) + 0.5
    k_w = torch.rand(KV_LORA, device="cuda", dtype=dtype) + 0.5
    return x, q_w, k_w


def _split(x):
    # forward_mla_rocm: fused_qkv_a(...).split([q_lora_rank, kv_lora_rank + rope])
    q, latent = x.split([Q_LORA, KV_LORA + ROPE], dim=-1)
    return q, latent[..., :KV_LORA]


@pytest.mark.parametrize("m", [0, 1, 4, 8192])
def test_strided_slices_match_torch(m):
    """Rows are addressed through the input row stride and written densely."""
    x, q_w, k_w = _fused_qkv_a(m, torch.bfloat16)
    q, k = _split(x)
    if m > 1:
        assert not q.is_contiguous() and not k.is_contiguous()

    out_q, out_k = fused_qk_rmsnorm_triton(q, q_w, EPS, k, k_w, EPS)
    ref_q, ref_k = fused_qk_rmsnorm_torch(q, q_w, EPS, k, k_w, EPS)

    assert out_q.shape == q.shape and out_k.shape == k.shape
    assert out_q.is_contiguous() and out_k.is_contiguous()
    torch.testing.assert_close(out_q, ref_q, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(out_k, ref_k, atol=2e-2, rtol=2e-2)


def test_3d_and_column_strided_inputs():
    """Inputs the kernel cannot read in place are flattened or copied first."""
    x, q_w, k_w = _fused_qkv_a(48, torch.bfloat16)
    q, k = _split(x)
    q3 = q.reshape(6, 8, Q_LORA)
    k_t = k.contiguous().t().contiguous().t()
    assert k_t.stride(-1) != 1

    out_q, out_k = fused_qk_rmsnorm_triton(q3, q_w, EPS, k_t, k_w, EPS)
    ref_q, ref_k = fused_qk_rmsnorm_torch(q, q_w, EPS, k, k_w, EPS)

    assert out_q.shape == q3.shape and out_k.shape == k_t.shape
    torch.testing.assert_close(out_q.reshape(q.shape), ref_q, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(out_k, ref_k, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
