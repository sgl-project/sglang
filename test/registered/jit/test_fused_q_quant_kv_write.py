import pytest
import torch

from sglang.jit_kernel.fused_q_quant_kv_write import fused_q_quant_kv_write
from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-b200")

FP8_MAX = 448.0


def _ref_kv_quant(x, inv_scale):
    return torch.clamp(x.float() * inv_scale, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


def _ulp(a, b):
    return (a.view(torch.uint8).int() - b.view(torch.uint8).int()).abs().max().item()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens", [1, 4, 16, 20, 64, 256])
@pytest.mark.parametrize("num_q_heads,num_kv_heads", [(16, 4), (32, 8), (64, 1)])
def test_matches_reference(dtype, num_tokens, num_q_heads, num_kv_heads):
    torch.manual_seed(0)
    head_dim, total_slots = 128, 512
    k_scale, v_scale, scaling = 0.3, 0.5, 1.0 / (head_dim**0.5)
    dev = "cuda"

    q = torch.randn(num_tokens, num_q_heads * head_dim, dtype=dtype, device=dev)
    k = torch.randn(num_tokens, num_kv_heads * head_dim, dtype=dtype, device=dev)
    v = torch.randn(num_tokens, num_kv_heads * head_dim, dtype=dtype, device=dev)
    cache_loc = torch.randperm(total_slots, device=dev)[:num_tokens].to(torch.int64)

    k_cache = torch.zeros(
        total_slots, num_kv_heads, head_dim, dtype=torch.float8_e4m3fn, device=dev
    )
    v_cache = torch.zeros_like(k_cache)
    k_ref = torch.zeros_like(k_cache)
    v_ref = torch.zeros_like(v_cache)

    # Reference: the exact ops the fused kernel replaces.
    q_ref, q_scale_ref = scaled_fp8_quant(q.reshape(-1, q.shape[-1]).contiguous(), None)
    q_ref = q_ref.reshape(q.shape)
    bmm1_ref = q_scale_ref * k_scale * scaling
    k_ref[cache_loc] = _ref_kv_quant(k, 1.0 / k_scale).view(
        num_tokens, num_kv_heads, head_dim
    )
    v_ref[cache_loc] = _ref_kv_quant(v, 1.0 / v_scale).view(
        num_tokens, num_kv_heads, head_dim
    )

    q_fp8, bmm1 = fused_q_quant_kv_write(
        q,
        k,
        v,
        k_cache,
        v_cache,
        cache_loc,
        inv_k_scale=1.0 / k_scale,
        inv_v_scale=1.0 / v_scale,
        bmm1_extra=k_scale * scaling,
    )

    # FP8 quant of the same fp32 values; allow 1 ULP for amax reduction order.
    assert _ulp(q_fp8, q_ref) <= 1
    assert _ulp(k_cache, k_ref) <= 1
    assert _ulp(v_cache, v_ref) <= 1
    torch.testing.assert_close(bmm1, bmm1_ref, rtol=1e-3, atol=1e-6)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
