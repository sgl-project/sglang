# Adapted from sgl-flash-attn hopper/test_attn_kvcache.py::test_flash_attn_kvcache_only_qv
# Covers the only_qv (NoPE) decode path that FA3 adds for sparse MLA on SM90:
# the QK^T matmul is skipped and attention is computed as softmax(qv * V) over
# a paged V cache (no K cache, no rope).

import math
import sys
import unittest

import pytest
import torch
from einops import rearrange, repeat

from sglang.jit_kernel.flash_attention import flash_attn_with_kvcache
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

# FA3 only_qv path is SM90 (Hopper) only — skip on pre-Hopper and on
# Blackwell+ (sm100+) where FA3 is not built.
skip_condition = not torch.cuda.is_available() or (
    torch.cuda.get_device_capability()[0] != 9
)


def _only_qv_reference(qv, v_cache, page_table, batch_size, nheads_q):
    v_ref = rearrange(
        v_cache.float()[page_table.flatten()],
        "(b s) p h d -> b (s p) h d",
        b=batch_size,
    )
    v_ref = repeat(v_ref, "b s h d -> b s (h g) d", g=nheads_q)
    scores = torch.einsum("bqhd,bkhd->bhqk", qv.float(), v_ref)
    probs = torch.softmax(scores / math.sqrt(qv.shape[-1]), dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probs, v_ref).to(qv.dtype)


@pytest.mark.skipif(
    skip_condition, reason="FA3 only_qv requires compute capability sm90 (Hopper)."
)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seqlen_k", [129, 257])
@pytest.mark.parametrize("nheads_q", [8, 16])
def test_flash_attn_kvcache_only_qv(batch_size, seqlen_k, nheads_q):
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    seqlen_q = 1
    nheads_kv = 1
    v_dim = 512
    page_size = 1
    num_pages = batch_size * seqlen_k

    v_cache = torch.randn(
        num_pages, page_size, nheads_kv, v_dim, device=device, dtype=dtype
    )
    page_table = torch.arange(num_pages, device=device, dtype=torch.int32).view(
        batch_size, seqlen_k
    )
    qv = torch.randn(batch_size, seqlen_q, nheads_q, v_dim, device=device, dtype=dtype)
    cache_seqlens = torch.full(
        (batch_size,), seqlen_k, device=device, dtype=torch.int32
    )

    out_ref = _only_qv_reference(qv, v_cache, page_table, batch_size, nheads_q)

    out = flash_attn_with_kvcache(
        q=None,
        k_cache=None,
        v_cache=v_cache,
        qv=qv,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        only_qv=True,
        num_splits=1,
        ver=3,
    )

    max_err = (out - out_ref).abs().max().item()
    mean_err = (out - out_ref).abs().mean().item()
    print(f"only_qv max diff: {max_err}  mean diff: {mean_err}")

    assert max_err <= 8e-3, f"max abs err {max_err} > 8e-3"
    assert mean_err <= 3e-4, f"mean abs err {mean_err} > 3e-4"


class TestFlashAttentionV3OnlyQV(CustomTestCase):
    """unittest wrapper so `python -m unittest` and direct invocation also work."""

    @unittest.skipIf(
        skip_condition,
        "FA3 only_qv requires compute capability sm90 (Hopper).",
    )
    def test_only_qv_smoke(self):
        test_flash_attn_kvcache_only_qv(batch_size=2, seqlen_k=257, nheads_q=8)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
