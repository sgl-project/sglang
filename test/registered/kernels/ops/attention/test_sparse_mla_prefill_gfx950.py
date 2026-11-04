"""HIP DSA sparse MLA prefill (gfx950, fp8_ds_mla KV) must match an fp32 reference as
closely as the Triton kernel does.

Covers padded (-1) indices, a row without keys, a topk that is not a multiple of the
32-key tile, sharp softmax (large q) and head-major (strided) q."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsa.sparse_mla_prefill_gfx950 import (
    sparse_mla_prefill_gfx950,
    supports_sparse_mla_prefill_gfx950,
)
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci

# backend-specific: a gfx950 HIP kernel with no CUDA counterpart.
register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

_DEVICE = "cuda"
_DV, _DT, _POOL = 512, 64, 8192
_ROW = _DV + 16 + 2 * _DT


def _kv(seed):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)
    nope = (torch.randn(_POOL, _DV, device=_DEVICE, generator=g) * 0.5).to(
        torch.float8_e4m3fn
    )
    scales = torch.rand(_POOL, 4, device=_DEVICE, generator=g) / 32 + 1 / 128
    rope = (torch.randn(_POOL, _DT, device=_DEVICE, generator=g) * 0.5).to(
        torch.bfloat16
    )
    kv = torch.empty(_POOL, 1, _ROW, dtype=torch.uint8, device=_DEVICE)
    kv[:, 0, :_DV] = nope.view(torch.uint8)
    kv[:, 0, _DV : _DV + 16] = scales.view(torch.uint8)
    kv[:, 0, _DV + 16 :] = rope.view(torch.uint8)
    k = torch.cat([nope.float() * scales.repeat_interleave(128, -1), rope.float()], -1)
    return kv, k


def _reference(q_nope, q_rope, k, idx, sm_scale):
    q = torch.cat([q_nope.float(), q_rope.float()], -1)  # [T, H, 576]
    keys = k[idx.clamp(min=0).long()]  # [T, topk, 576]
    s = torch.einsum("thd,tkd->thk", q, keys) * sm_scale
    s = s.masked_fill((idx < 0)[:, None, :], float("-inf"))
    lse = torch.logsumexp(s, -1)
    out = torch.einsum(
        "thk,tkd->thd", torch.softmax(s, -1).nan_to_num(0.0), keys[..., :_DV]
    )
    return out, lse


@pytest.mark.skipif(not is_gfx95_supported(), reason="gfx950 only")
@pytest.mark.parametrize("H", [12, 16, 8])
@pytest.mark.parametrize("topk", [2048, 1000])
@pytest.mark.parametrize("q_mul", [1.0, 30.0])
@pytest.mark.parametrize("head_major", [False, True])
def test_sparse_mla_prefill_gfx950(H, topk, q_mul, head_major):
    T = 96
    kv, k = _kv(seed=H + topk)
    torch.manual_seed(topk + H)
    q_nope = torch.randn(T, H, _DV, device=_DEVICE, dtype=torch.bfloat16) * (
        0.1 * q_mul
    )
    q_rope = torch.randn(T, H, _DT, device=_DEVICE, dtype=torch.bfloat16) * (
        0.1 * q_mul
    )
    if head_major:
        q_nope = q_nope.transpose(0, 1).contiguous().transpose(0, 1)
        q_rope = q_rope.transpose(0, 1).contiguous().transpose(0, 1)
    idx = torch.randint(0, _POOL, (T, topk), device=_DEVICE, dtype=torch.int32)
    idx[:8, -300:] = -1
    idx[8] = -1
    sm_scale = (_DV + _DT) ** -0.5
    assert supports_sparse_mla_prefill_gfx950(q_nope, q_rope, kv, _DV)

    out, lse = sparse_mla_prefill_gfx950(
        q_nope, q_rope, kv, idx.unsqueeze(1), sm_scale, _DV, return_lse=True
    )
    ref_out, ref_lse = _reference(q_nope, q_rope, k, idx, sm_scale)

    assert out.shape == (1, T, H, _DV) and lse.shape == (T, H)
    assert torch.all(out[0, 8] == 0) and torch.all(torch.isneginf(lse[8]))
    keep = torch.arange(T, device=_DEVICE) != 8
    o, r = out[0, keep].float(), ref_out[keep]
    assert ((o - r).norm() / r.norm()).item() < 4e-3
    torch.testing.assert_close(lse[keep], ref_lse[keep], rtol=0, atol=1e-4)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
