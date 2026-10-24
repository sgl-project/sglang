"""Cake fused QK RMSNorm + NeoX RoPE + paged KV append through sglang.kernels.

Checks three things for the Cake adapter distributed by FlashInfer:
the registry resolves the explicit FlashInfer backend; the facade result is
bitwise identical to calling FlashInfer directly; and the fused result matches
a pure-torch reference within BF16 tolerance. Skips (with the reason) when the
installed FlashInfer lacks the Cake module or the GPU is outside sm_90a /
sm_100a / sm_103a.
"""

import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import kvcache as cake_kvcache
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.kvcache.cake import (
    cake_fused_qk_rmsnorm_rope_append_paged_kv_cache,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

OP = "kvcache.fused_qk_rmsnorm_rope_append_paged_kv_cache"
HEAD_DIM = 128


def test_registry_resolves_flashinfer_backend():
    spec = select_kernel(OP, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.kvcache:")


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(
        cake_kvcache.FI_MODULE, cake_kvcache.FI_JIT_MODULE
    ):
        pytest.skip("installed FlashInfer lacks flashinfer.cake_fused_qk_rope_append")
    cc = torch.cuda.get_device_capability()
    if cc not in cake_kvcache.ARCHS:
        pytest.skip(f"Cake rope-append is built for sm_90a/100a/103a, device is {cc}")


def _neox_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x [T, H, 128] fp32; cos/sin [T, 64]
    half = HEAD_DIM // 2
    x1, x2 = x[..., :half], x[..., half:]
    c = cos[:, None, :]
    s = sin[:, None, :]
    return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)


def _rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    var = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * w


def _reference(qkv, cos_sin, positions, hq, hkv, policy, qw, kw, eps):
    T = qkv.shape[0]
    x = qkv.float().view(T, hq + 2 * hkv, HEAD_DIM)
    q, k, v = x[:, :hq], x[:, hq : hq + hkv], x[:, hq + hkv :]
    cos = cos_sin[positions, :64]
    sin = cos_sin[positions, 64:]
    if policy == 2:
        q = _rmsnorm(q, qw, eps)
        k = _rmsnorm(k, kw, eps)
    q = _neox_rope(q, cos, sin)
    k = _neox_rope(k, cos, sin)
    if policy == 1:
        q = _rmsnorm(q, qw, eps)
        k = _rmsnorm(k, kw, eps)
    return q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)


@pytest.mark.parametrize("hq,hkv", [(8, 1), (64, 8)])
@pytest.mark.parametrize("policy", [0, 1, 2])
def test_matches_flashinfer_and_reference(hq, hkv, policy):
    _skip_unless_supported()
    torch.manual_seed(0)
    device = torch.device("cuda")
    page_size = 16
    # Two requests: prefix lengths 5 and 20, appending 3 and 7 tokens.
    prefix = [5, 20]
    new = [3, 7]
    seq_lens_list = [p + n for p, n in zip(prefix, new)]
    T = sum(new)
    max_pages = max((s + page_size - 1) // page_size for s in seq_lens_list)
    num_pages = 2 * max_pages + 1
    qkv = torch.randn(T, (hq + 2 * hkv) * HEAD_DIM, device=device, dtype=torch.bfloat16)
    max_pos = 64
    inv = 1.0 / (
        10000 ** (torch.arange(0, HEAD_DIM, 2, device=device).float() / HEAD_DIM)
    )
    ang = torch.arange(max_pos, device=device).float()[:, None] * inv[None, :]
    cos_sin = torch.cat((ang.cos(), ang.sin()), dim=-1).contiguous()
    seq_lens = torch.tensor(seq_lens_list, device=device, dtype=torch.int32)
    q_indptr = torch.tensor([0, new[0], T], device=device, dtype=torch.int32)
    page_indices = torch.arange(
        1, 1 + 2 * max_pages, device=device, dtype=torch.int32
    ).view(2, max_pages)
    key_cache = torch.zeros(
        num_pages, page_size, hkv, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    value_cache = torch.zeros_like(key_cache)
    key_cache_fi = key_cache.clone()
    value_cache_fi = value_cache.clone()
    qw = torch.rand(HEAD_DIM, device=device) + 0.5
    kw = torch.rand(HEAD_DIM, device=device) + 0.5
    eps = 1e-6
    assert cake_kvcache.supports_fused_qk_rmsnorm_rope_append(
        qkv, key_cache, num_q_heads=hq, num_kv_heads=hkv
    )
    kwargs = dict(
        num_q_heads=hq,
        num_kv_heads=hkv,
        qk_norm_policy=policy,
        q_norm_weight=qw if policy else None,
        k_norm_weight=kw if policy else None,
        eps=eps,
    )
    out_q = cake_fused_qk_rmsnorm_rope_append_paged_kv_cache(
        qkv, cos_sin, seq_lens, q_indptr, page_indices, key_cache, value_cache, **kwargs
    )
    from flashinfer.cake_fused_qk_rope_append import (
        cake_fused_qk_rmsnorm_rope_append_paged_kv_cache as fi_direct,
    )

    out_q_fi = fi_direct(
        qkv,
        cos_sin,
        seq_lens,
        q_indptr,
        page_indices,
        key_cache_fi,
        value_cache_fi,
        **kwargs,
    )
    torch.cuda.synchronize()
    assert torch.equal(out_q, out_q_fi)
    assert torch.equal(key_cache, key_cache_fi)
    assert torch.equal(value_cache, value_cache_fi)

    positions = torch.cat(
        [torch.arange(p, p + n, device=device) for p, n in zip(prefix, new)]
    )
    ref_q, ref_k, ref_v = _reference(
        qkv, cos_sin, positions, hq, hkv, policy, qw, kw, eps
    )
    torch.testing.assert_close(out_q.float(), ref_q.float(), atol=1e-2, rtol=1e-2)
    row = 0
    for b, (p, n) in enumerate(zip(prefix, new)):
        for t in range(n):
            pos = p + t
            page = page_indices[b, pos // page_size].item()
            slot = pos % page_size
            torch.testing.assert_close(
                key_cache[page, slot].float(), ref_k[row].float(), atol=1e-2, rtol=1e-2
            )
            torch.testing.assert_close(
                value_cache[page, slot].float(),
                ref_v[row].float(),
                atol=1e-2,
                rtol=1e-2,
            )
            row += 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
