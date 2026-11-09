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
    from sglang.kernels.ops.attention import flash_mla_interface

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
    sched = flash_mla_interface.FlashMLASchedMeta()
    if meta is not None:
        sched.tile_scheduler_metadata, sched.num_splits = meta
    out, lse = flash_mla_interface.flash_mla_with_kvcache(
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


@pytest.mark.parametrize("offset", [0, 1, 2, 3])
def test_schedule_at_shared_memory_boundary(offset):
    parts = torch.cuda.get_device_properties(0).multi_processor_count
    b = (48 * 1024 // 4 - 1 - parts * 8) // 5 - offset
    meta = torch.empty((parts, 8), dtype=torch.int32, device="cuda")
    splits = torch.empty((b + 1,), dtype=torch.int32, device="cuda")
    lengths = torch.full((b,), 128, dtype=torch.int32, device="cuda")
    meta, splits = _ours(meta, splits, lengths, 128)
    torch.cuda.synchronize()
    # Each request costs two KV blocks plus five overhead blocks. At these
    # shapes every partition fits whole requests, so no request is split.
    payload = (7 * b + parts - 1) // parts + 5
    assert payload % 7 <= 5
    per_part = payload // 7
    expected = []
    for i in range(parts):
        begin, end = min(i * per_part, b), min((i + 1) * per_part, b)
        expected.append([begin, end - 1, 0, 2, 0, 0, 0])
    torch.testing.assert_close(
        meta[:, DEFINED].cpu(), torch.tensor(expected, dtype=torch.int32)
    )
    torch.testing.assert_close(splits.cpu(), torch.arange(b + 1, dtype=torch.int32))


def test_precompute_skips_oversized_schedule(monkeypatch):
    from types import SimpleNamespace

    from sglang.srt.layers.attention import deepseek_v4_backend as backend

    monkeypatch.setattr(backend, "_fast_flashmla_sched_shape", lambda q: True)
    monkeypatch.setattr(backend, "_num_sms", lambda _: 148)
    max_batch = (48 * 1024 // 4 - 1 - 148 * 8) // 5
    for b in (max_batch + 1, 5000, 16384):
        meta = SimpleNamespace(tile_scheduler_metadata=None, num_splits=None)
        backend._maybe_precompute_flashmla_sched_meta(
            meta,
            q=SimpleNamespace(shape=(b, 1, 64, 512), device=SimpleNamespace(index=0)),
            indices=None,
            topk_length=None,
            extra_indices=None,
            extra_topk_length=None,
        )
        assert meta.tile_scheduler_metadata is None
        assert meta.num_splits is None


# b = 7 stays below kMinParallelBatch (both walks serial), 8 is the first
# parallel batch; each (parts, topk, extra_topk) case is a different regime.
@pytest.mark.parametrize("b", [7, 8, 64, 200, 512])
@pytest.mark.parametrize(
    "num_sm_parts,topk,extra_topk", [(148, 2048, 0), (74, 512, 0), (9, 2048, 512)]
)
@pytest.mark.parametrize("mode", ["full", "random", "zeros", "ones", "mixed", "skewed"])
def test_parallel_walk_matches_serial(b, num_sm_parts, topk, extra_topk, mode):
    """The parallel partition walk reproduces the serial one bit for bit."""
    from sglang.kernels.ops.attention.dsv4.flashmla_sched_meta import (
        META_INTS,
        flashmla_sched_meta,
    )

    seed = b * 31 + num_sm_parts + topk + extra_topk + len(mode)
    if mode == "skewed":  # a few long requests among short ones
        g = torch.Generator(device="cuda").manual_seed(seed)
        topk_length = torch.randint(
            0, 65, (b,), device="cuda", dtype=torch.int32, generator=g
        )
        topk_length[:: max(b // 4, 1)] = topk
    else:
        topk_length = _lengths(b, topk, mode, seed)
    extra_topk_length = (
        _lengths(b, extra_topk, "random", seed + 1) if extra_topk else None
    )
    out = []
    for parallel in (False, True):
        meta = torch.full(
            (num_sm_parts, META_INTS), -7, dtype=torch.int32, device="cuda"
        )
        splits = torch.full((b + 1,), -7, dtype=torch.int32, device="cuda")
        flashmla_sched_meta(
            meta,
            splits,
            topk_length=topk_length,
            extra_topk_length=extra_topk_length,
            block_size_n=BLOCK_SIZE_N,
            fixed_overhead_num_blocks=FIXED_OVERHEAD,
            topk=topk,
            extra_topk=extra_topk,
            schedule="parallel" if parallel else "serial",
        )
        out.append((meta, splits))
    assert torch.equal(out[0][0], out[1][0])
    assert torch.equal(out[0][1], out[1][1])


def test_parallel_walk_replay_switches_with_lengths():
    """A captured DCP verify schedule can outgrow its parallel workspace."""
    from sglang.kernels.ops.attention.dsv4.flashmla_sched_meta import (
        META_INTS,
        flashmla_sched_meta,
    )

    b, topk, parts = 512, 2048, 148
    lengths = torch.full((b,), 256, device="cuda", dtype=torch.int32)
    outputs = [
        (
            torch.empty((parts, META_INTS), device="cuda", dtype=torch.int32),
            torch.empty((b + 1,), device="cuda", dtype=torch.int32),
        )
        for _ in range(2)
    ]

    def run(parallel):
        flashmla_sched_meta(
            *outputs[int(parallel)],
            topk_length=lengths,
            block_size_n=BLOCK_SIZE_N,
            fixed_overhead_num_blocks=FIXED_OVERHEAD,
            topk=topk,
            schedule="parallel" if parallel else "serial",
        )

    for parallel in (False, True):
        run(parallel)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(True)
    for length in (256, topk, 0, 384, 448, 1, 256):
        lengths.fill_(length)
        graph.replay()
        run(False)
        assert torch.equal(outputs[0][0], outputs[1][0])
        assert torch.equal(outputs[0][1], outputs[1][1])


def _replay_schedule(meta, splits, num_blocks):
    """Replay how the sparse decode kernel and the combine consume a schedule
    and check it covers every request exactly once; return the costliest
    partition (blocks plus an overhead per piece)."""
    meta, splits, num_blocks = meta.tolist(), splits.tolist(), num_blocks.tolist()
    b = len(num_blocks)
    assert splits[0] == 0 and all(splits[r + 1] > splits[r] for r in range(b))
    assert splits[b] <= b + len(meta)
    pieces = [[] for _ in range(b)]
    max_cost = 0
    for part in meta:
        (
            begin_req,
            end_req,
            begin_block,
            end_block,
            begin_split,
            first_split,
            last_split,
        ) = part[:7]
        if begin_req >= b:
            continue
        cost = 0
        for r in range(begin_req, end_req + 1):
            start = begin_block if r == begin_req else 0
            end = end_block if r == end_req else num_blocks[r]
            split_flag = (
                first_split if r == begin_req else (last_split if r == end_req else 0)
            )
            split_idx = splits[r] + begin_split if r == begin_req else splits[r]
            assert start < end <= num_blocks[r]
            pieces[r].append((start, end, split_idx, split_flag))
            cost += end - start + FIXED_OVERHEAD
        max_cost = max(max_cost, cost)
    for r in range(b):
        pieces[r].sort()
        n = len(pieces[r])
        assert n == splits[r + 1] - splits[r]
        assert [p[0] for p in pieces[r]] == [0] + [p[1] for p in pieces[r][:-1]]
        assert pieces[r][-1][1] == num_blocks[r]
        assert sorted(p[2] for p in pieces[r]) == list(range(splits[r], splits[r + 1]))
        assert all(bool(p[3]) == (n > 1) for p in pieces[r])
    return max_cost


@pytest.mark.parametrize("b", [1, 8, 64, 128, 384, 1000])
@pytest.mark.parametrize("num_sm_parts", [148, 74, 3])
@pytest.mark.parametrize("mode", ["full", "random", "zeros", "ones", "mixed", "dcp"])
def test_windowed_schedule_is_valid(b, num_sm_parts, mode):
    """The windowed schedule covers every request's blocks exactly once, with
    the greedy walk's bound on the costliest partition."""
    from sglang.kernels.ops.attention.dsv4.flashmla_sched_meta import (
        META_INTS,
        flashmla_sched_meta,
    )

    topk = 2048
    seed = b * 7 + num_sm_parts + len(mode)
    if mode == "dcp":  # this rank's share of the top-k under DCP8
        g = torch.Generator(device="cuda").manual_seed(seed)
        topk_length = torch.randint(
            200, 320, (b,), device="cuda", dtype=torch.int32, generator=g
        )
    else:
        topk_length = _lengths(b, topk, mode, seed)
    meta = torch.full((num_sm_parts, META_INTS), -7, dtype=torch.int32, device="cuda")
    splits = torch.full((b + 1,), -7, dtype=torch.int32, device="cuda")
    flashmla_sched_meta(
        meta,
        splits,
        topk_length=topk_length,
        block_size_n=BLOCK_SIZE_N,
        fixed_overhead_num_blocks=FIXED_OVERHEAD,
        topk=topk,
        schedule="windowed",
    )
    num_blocks = (topk_length.clamp(min=1) + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    max_cost = _replay_schedule(meta.cpu(), splits.cpu(), num_blocks.cpu())
    total = int(num_blocks.sum()) + FIXED_OVERHEAD * b
    assert max_cost <= (total + num_sm_parts - 1) // num_sm_parts + FIXED_OVERHEAD


@pytest.mark.parametrize("b", [6, 64])
def test_windowed_schedule_output(b):
    """FlashMLA gives the same attention on the windowed schedule as on its own."""
    from sglang.kernels.ops.attention.dsv4.flashmla_sched_meta import (
        flashmla_sched_meta,
    )

    topk = 2048
    # A finite cache (random bytes decode to NaNs). Each page holds its tokens'
    # fp8 nope (448 B) + bf16 rope (128 B), then their ue8m0 scales (8 B).
    kv = _cache()
    pages = kv.view(kv.shape[0], -1)
    tokens = pages[:, : PAGE * 576].view(kv.shape[0], PAGE, 576)
    g = torch.Generator(device="cuda").manual_seed(b)
    tokens[..., :448] = torch.randint(
        0, 0x70, tokens[..., :448].shape, device="cuda", dtype=torch.uint8, generator=g
    )
    tokens[..., 448:].view(torch.bfloat16).copy_(
        torch.randn(tokens.shape[:-1] + (64,), device="cuda", generator=g) * 0.1
    )
    pages[:, PAGE * 576 :] = 127
    topk_length = _lengths(b, topk, "random", b)
    ref_out, ref_lse, sched = _call(kv, b, 1, topk, topk_length)
    if sched.tile_scheduler_metadata is None:
        pytest.skip("FlashMLA did not split the KV for this shape")
    meta = torch.empty_like(sched.tile_scheduler_metadata)
    splits = torch.empty_like(sched.num_splits)
    flashmla_sched_meta(
        meta,
        splits,
        topk_length=topk_length,
        block_size_n=BLOCK_SIZE_N,
        fixed_overhead_num_blocks=FIXED_OVERHEAD,
        topk=topk,
        schedule="windowed",
    )
    out, lse, _ = _call(kv, b, 1, topk, topk_length, meta=(meta, splits))
    assert torch.isfinite(ref_out).all()
    torch.testing.assert_close(out, ref_out, rtol=1e-2, atol=1e-2)
    finite = torch.isfinite(ref_lse)
    assert torch.equal(finite, torch.isfinite(lse))
    torch.testing.assert_close(lse[finite], ref_lse[finite], rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
