"""MHA KV writers on an HND pool ([pages, heads, page_size, head_dim]) write the same bytes as on slot rows."""

from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.muse_spark_qk_norm_rope_store import FusedQKNorm
from sglang.kernels.ops.kvcache.fused_fp8_qkv_kv_cache import fused_fp8_qkv_kv_cache
from sglang.kernels.spec import KernelBackend
from sglang.srt.mem_cache.memory_pool import _set_mha_kv_buffer_prefix_valid
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

FP8 = torch.float8_e4m3fn
NUM_PAGES = 32


def _as_nhd(hnd: torch.Tensor) -> torch.Tensor:
    pages, heads, page, head_dim = hnd.shape
    return hnd.permute(0, 2, 1, 3).reshape(pages * page, heads, head_dim)


def _assert_same_bytes(a: torch.Tensor, b: torch.Tensor):
    assert torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def _caches(heads, head_dim, page, dtype):
    """Zeroed (NHD k, NHD v, HND k, HND v) caches over the same slots."""
    nhd = (NUM_PAGES * page, heads, head_dim)
    hnd = (NUM_PAGES, heads, page, head_dim)
    return tuple(
        torch.zeros(shape, device="cuda", dtype=dtype) for shape in (nhd, nhd, hnd, hnd)
    )


def _randn(*shape):
    return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)


def _locs(num, page, idx_dtype=torch.int64):
    # Page 0 holds the CUDA-graph padding slot.
    locs = torch.randperm((NUM_PAGES - 1) * page, device="cuda")[:num] + page
    return locs.to(idx_dtype)


@pytest.mark.parametrize("heads,head_dim", [(1, 128), (4, 64), (8, 128)])
@pytest.mark.parametrize("page", [64, 128])
@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("scale", [None, 0.5])
def test_fused_fp8_qkv_kv_cache(heads, head_dim, page, idx_dtype, scale):
    torch.manual_seed(0)
    num_tokens = 37
    q = _randn(num_tokens, 8 * heads * head_dim)
    k, v = _randn(num_tokens, heads, head_dim), _randn(num_tokens, heads, head_dim)
    ks = None if scale is None else torch.tensor(scale, device="cuda")
    nk, nv, hk, hv = _caches(heads, head_dim, page, FP8)
    loc = _locs(num_tokens, page, idx_dtype)
    q_nhd = fused_fp8_qkv_kv_cache(q, k, v, nk, nv, loc, ks, ks)
    q_hnd = fused_fp8_qkv_kv_cache(q, k, v, hk, hv, loc, ks, ks)
    _assert_same_bytes(q_nhd, q_hnd)
    _assert_same_bytes(nk, _as_nhd(hk))
    _assert_same_bytes(nv, _as_nhd(hv))
    assert nk.view(torch.uint8).any()


@pytest.mark.parametrize("heads,head_dim", [(2, 64), (4, 128)])
@pytest.mark.parametrize("page", [64, 128])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, FP8])
@pytest.mark.parametrize("backend", ["triton", "native"])
def test_muse_spark_qk_norm_store(heads, head_dim, page, cache_dtype, backend):
    torch.manual_seed(0)
    num_tokens, num_q, max_pos = 129, 8 * heads, 4096
    op = FusedQKNorm(head_dim=head_dim, num_q_heads=num_q, num_kv_heads=heads, eps=1e-6)
    width = (num_q + 2 * heads) * head_dim
    # A row-strided view, as the fused QKV-O projection output is.
    qkv = _randn(num_tokens, width + 64)[:, :width]
    positions = torch.randint(0, max_pos, (num_tokens,), device="cuda")
    cos_sin = torch.randn(max_pos, head_dim, device="cuda")
    nk, nv, hk, hv = _caches(heads, head_dim, page, cache_dtype)
    loc = _locs(num_tokens, page)
    outs = []
    for store in ((nk.flatten(1), nv.flatten(1), loc), (hk, hv, loc)):
        if backend == "triton":
            assert op.backend_eligible(
                KernelBackend.TRITON, qkv, positions, cos_sin, store
            )
            triton = KernelBackend.TRITON
            outs.append(op(qkv, positions, cos_sin, kv_store=store, backend=triton))
        else:
            outs.append(op.forward_native(qkv, positions, cos_sin, kv_store=store))
    for a, b in zip(*outs):
        _assert_same_bytes(a, b)
    _assert_same_bytes(nk, _as_nhd(hk))
    _assert_same_bytes(nv, _as_nhd(hv))
    assert nk.view(torch.uint8).any()


# Fused FP8 kernel, eager FP8 cast + byte kernel, BF16 byte kernel.
@pytest.mark.parametrize(
    "pool_dtype,scale", [(FP8, 0.5), (FP8, None), (torch.bfloat16, None)]
)
@pytest.mark.parametrize("heads,head_dim", [(1, 128), (4, 128), (8, 64)])
@pytest.mark.parametrize("page", [64, 128])
def test_set_kv_buffer_prefix_valid(heads, head_dim, page, pool_dtype, scale):
    """DFlash draft commit: only rows < commit_len land, at the same slots in both layouts."""
    torch.manual_seed(0)
    batch, draft_len = 9, 7
    store_dtype = torch.uint8 if pool_dtype == FP8 else pool_dtype
    loc_2d = _locs(batch * draft_len, page).view(batch, draft_len)
    commit = torch.randint(0, draft_len + 1, (batch,), device="cuda", dtype=torch.int32)
    commit[0], commit[1], commit[-1] = 0, 1, draft_len
    k = _randn(batch * draft_len, heads, head_dim)
    v = _randn(batch * draft_len, heads, head_dim)
    caches = _caches(heads, head_dim, page, store_dtype)
    for use_hnd, (k_cache, v_cache) in ((False, caches[:2]), (True, caches[2:])):
        pool = SimpleNamespace(
            k_buffer=[k_cache],
            v_buffer=[v_cache],
            use_hnd=use_hnd,
            page_size=page,
            head_dim=head_dim,
            v_head_dim=head_dim,
            row_dim=heads * head_dim,
            v_row_dim=heads * head_dim,
            dtype=pool_dtype,
            store_dtype=store_dtype,
            start_layer=0,
        )
        layer = SimpleNamespace(layer_id=0)
        _set_mha_kv_buffer_prefix_valid(
            pool, layer, loc_2d, commit, k.clone(), v.clone(), scale, scale
        )
    nk, nv, hk, hv = caches
    _assert_same_bytes(nk, _as_nhd(hk))
    _assert_same_bytes(nv, _as_nhd(hv))
    written = nk.flatten(1).view(torch.uint8).ne(0).any(-1)
    live = torch.arange(draft_len, device="cuda")[None, :] < commit[:, None]
    assert written[loc_2d[live]].all()
    assert int(written.sum()) == int(live.sum())


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
