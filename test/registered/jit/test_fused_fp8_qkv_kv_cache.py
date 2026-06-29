import pytest
import torch

from sglang.jit_kernel.fused_fp8_qkv_kv_cache import fused_fp8_qkv_kv_cache
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=40, suite="base-b-kernel-unit-1-gpu-b200")

FP8 = torch.float8_e4m3fn


def _ref_quant(x_f32: torch.Tensor, inv_scale: float) -> torch.Tensor:
    y = (x_f32 * inv_scale).clamp(-448.0, 448.0)
    return y.to(FP8)


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1).view(torch.uint8)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "hq,hkv,head_dim", [(8, 1, 128), (8, 8, 128), (4, 2, 64), (64, 2, 128)]
)
@pytest.mark.parametrize("num_tokens", [1, 7, 16])
@pytest.mark.parametrize("scale", [None, 0.5, 2.0])
@pytest.mark.parametrize("idx_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("fused_qkv", [False, True])
def test_fused_fp8_qkv_kv_cache(
    dtype, hq, hkv, head_dim, num_tokens, scale, idx_dtype, fused_qkv
):
    torch.manual_seed(0)
    device = "cuda"
    q_dim = hq * head_dim
    kv_dim = hkv * head_dim
    total_slots = num_tokens + 4

    if fused_qkv:
        qkv = torch.randn(num_tokens, q_dim + 2 * kv_dim, dtype=dtype, device=device)
        q = qkv[:, :q_dim]
        k = qkv[:, q_dim : q_dim + kv_dim].view(num_tokens, hkv, head_dim)
        v = qkv[:, q_dim + kv_dim :].view(num_tokens, hkv, head_dim)
        if num_tokens > 1:
            assert not q.is_contiguous()
    else:
        q = torch.randn(num_tokens, q_dim, dtype=dtype, device=device)
        k = torch.randn(num_tokens, hkv, head_dim, dtype=dtype, device=device)
        v = torch.randn(num_tokens, hkv, head_dim, dtype=dtype, device=device)
    k_cache = torch.zeros(total_slots, hkv, head_dim, dtype=FP8, device=device)
    v_cache = torch.zeros(total_slots, hkv, head_dim, dtype=FP8, device=device)

    cache_loc = torch.randperm(total_slots, device=device)[:num_tokens].to(idx_dtype)

    if scale is None:
        k_scale = v_scale = None
        inv_k = inv_v = 1.0
    else:
        k_scale = torch.tensor(scale, dtype=torch.float32, device=device)
        v_scale = torch.tensor(scale * 1.5, dtype=torch.float32, device=device)
        inv_k = 1.0 / float(k_scale)
        inv_v = 1.0 / float(v_scale)

    q_out = fused_fp8_qkv_kv_cache(
        q, k, v, k_cache, v_cache, cache_loc, k_scale, v_scale
    )

    q_ref = q.to(FP8)
    torch.testing.assert_close(_bytes(q_out), _bytes(q_ref), rtol=0, atol=0)

    k_ref = _ref_quant(k.reshape(num_tokens, kv_dim).float(), inv_k)
    v_ref = _ref_quant(v.reshape(num_tokens, kv_dim).float(), inv_v)
    loc = cache_loc.long()
    torch.testing.assert_close(
        _bytes(k_cache.reshape(total_slots, kv_dim)[loc]), _bytes(k_ref), rtol=0, atol=0
    )
    torch.testing.assert_close(
        _bytes(v_cache.reshape(total_slots, kv_dim)[loc]), _bytes(v_ref), rtol=0, atol=0
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
