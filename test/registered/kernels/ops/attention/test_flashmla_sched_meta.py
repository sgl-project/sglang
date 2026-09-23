import sys

import pytest
import torch

from sglang.srt.runtime_context import get_platform
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.cuda is None
    or not get_platform().is_blackwell,
    reason="the FlashMLA split-KV schedule is Blackwell-only here",
)

H_Q, D_QK, D_V = 64, 512, 512
# The only cache format this build's sparse decode takes for both caches, and a
# page size that keeps page * bytes_per_token a multiple of 576.
BYTES_PER_TOKEN, PAGE = 584, 288
BLOCK_SIZE_N, FIXED_OVERHEAD = 64, 5


def _cache():
    n = 8192 // PAGE + 16
    return torch.randint(
        0, 200, (n, PAGE, 1, BYTES_PER_TOKEN), device="cuda", dtype=torch.uint8
    )


def _call(kv, b, s_q, topk, topk_length, *, extra=None, meta=None):
    import sgl_kernel.flash_mla as flash_mla

    g = torch.Generator(device="cuda").manual_seed(b * 7 + s_q * 13 + topk)
    q = torch.randn(
        (b, s_q, H_Q, D_QK), device="cuda", dtype=torch.bfloat16, generator=g
    )
    indices = torch.randint(
        0, 5120, (b, s_q, topk), device="cuda", dtype=torch.int32, generator=g
    )
    sink = torch.randn((H_Q,), device="cuda", dtype=torch.float32, generator=g)
    kwargs = {}
    if extra is not None:
        extra_kv, extra_indices, extra_topk_length = extra
        kwargs = dict(
            extra_k_cache=extra_kv,
            extra_indices_in_kvcache=extra_indices,
            extra_topk_length=extra_topk_length,
        )
    sched = flash_mla.FlashMLASchedMeta()
    if meta is not None:
        sched.tile_scheduler_metadata, sched.num_splits = meta
    out, lse = flash_mla.flash_mla_with_kvcache(
        q,
        kv,
        None,
        None,
        D_V,
        sched,
        indices=indices,
        is_fp8_kvcache=True,
        softmax_scale=0.1,
        causal=False,
        topk_length=topk_length,
        attn_sink=sink,
        **kwargs,
    )
    torch.cuda.synchronize()
    return out, lse, sched


def _ours(
    like_meta, like_splits, topk_length, topk, *, extra_topk_length=None, extra_topk=0
):
    from sglang.kernels.ops.attention.dsv4.flashmla_sched_meta import (
        flashmla_sched_meta,
    )

    meta = torch.empty_like(like_meta)
    splits = torch.empty_like(like_splits)
    flashmla_sched_meta(
        meta,
        splits,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
        block_size_n=BLOCK_SIZE_N,
        fixed_overhead_num_blocks=FIXED_OVERHEAD,
        topk=topk,
        extra_topk=extra_topk,
    )
    return meta, splits


def _lengths(b, topk, mode, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if mode == "full":
        return torch.full((b,), topk, device="cuda", dtype=torch.int32)
    if mode == "ones":
        return torch.ones((b,), device="cuda", dtype=torch.int32)
    if mode == "zeros":
        return torch.zeros((b,), device="cuda", dtype=torch.int32)
    lengths = torch.randint(
        0, topk + 1, (b,), device="cuda", dtype=torch.int32, generator=g
    )
    if mode == "mixed":
        lengths[0] = 0
        lengths[-1] = topk
    return lengths


# FlashMLA's DecodingSchedMeta ends in a `_pad` word it never writes, so the
# reference carries whatever torch::empty left there.
DEFINED = slice(0, 7)


@pytest.mark.parametrize("b", [1, 6, 64])
@pytest.mark.parametrize("s_q", [1, 6])
@pytest.mark.parametrize("topk", [2048])
@pytest.mark.parametrize("mode", ["full", "random", "zeros", "ones", "mixed"])
def test_matches_flashmla_schedule(b: int, s_q: int, topk: int, mode: str):
    kv = _cache()
    topk_length = _lengths(b, topk, mode, b * 1000 + s_q * 37 + topk + len(mode))
    _, _, sched = _call(kv, b, s_q, topk, topk_length)
    if sched.tile_scheduler_metadata is None:
        pytest.skip("FlashMLA did not split the KV for this shape")
    meta, splits = _ours(
        sched.tile_scheduler_metadata, sched.num_splits, topk_length, topk
    )
    assert torch.equal(meta[:, DEFINED], sched.tile_scheduler_metadata[:, DEFINED])
    assert torch.equal(splits, sched.num_splits)


@pytest.mark.parametrize("b", [1, 8])
@pytest.mark.parametrize("topk,extra_topk", [(512, 512), (2048, 512), (512, 2048)])
def test_extra_cache_schedule(b: int, topk: int, extra_topk: int):
    kv, extra_kv = _cache(), _cache()
    s_q = 1
    g = torch.Generator(device="cuda").manual_seed(b + topk + extra_topk)
    topk_length = _lengths(b, topk, "random", b + topk)
    extra_topk_length = torch.randint(
        1, extra_topk + 1, (b,), device="cuda", dtype=torch.int32, generator=g
    )
    extra_indices = torch.randint(
        0, 4096, (b, s_q, extra_topk), device="cuda", dtype=torch.int32, generator=g
    )
    extra = (extra_kv, extra_indices, extra_topk_length)
    ref_out, ref_lse, sched = _call(kv, b, s_q, topk, topk_length, extra=extra)
    if sched.tile_scheduler_metadata is None:
        pytest.skip("FlashMLA did not split the KV for this shape")
    meta, splits = _ours(
        sched.tile_scheduler_metadata,
        sched.num_splits,
        topk_length,
        topk,
        extra_topk_length=extra_topk_length,
        extra_topk=extra_topk,
    )
    assert torch.equal(meta[:, DEFINED], sched.tile_scheduler_metadata[:, DEFINED])
    assert torch.equal(splits, sched.num_splits)
    out, lse, _ = _call(kv, b, s_q, topk, topk_length, extra=extra, meta=(meta, splits))
    assert torch.equal(out.view(torch.int16), ref_out.view(torch.int16))
    assert torch.equal(lse.view(torch.int32), ref_lse.view(torch.int32))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
