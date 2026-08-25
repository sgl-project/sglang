"""Paged slot resolution in the step-3 sparse kernels vs the per-token gather.

``flash_{prefill,decode}_with_gqa_share_sparse`` take a ``page_size``. When it
and the sparse block size divide one another, a selected block covers whole
page-aligned runs of consecutive slots, so the kernel resolves it with one
``req_to_token`` lookup per run instead of a ``block_size``-entry gather. The
math is untouched, so the two paths must agree element for element -- and they
share one autotune cache entry (``page_size`` reaches the kernel as a constexpr,
which is not part of the autotune key), so there is no config race to make the
comparison approximate.

Every fallback condition is covered alongside the fast path: production sends
whatever ``--page-size`` the pool was built with straight through here, and the
kernel decides on its own which way to resolve.
"""

import argparse

import pytest
import torch
from triton.testing import do_bench

from sglang.kernels.ops.attention.minimax_sparse.common.utils import get_cu_seqblocks
from sglang.kernels.ops.attention.minimax_sparse.decode import topk_sparse as decode_mod
from sglang.kernels.ops.attention.minimax_sparse.decode.topk_sparse import (
    flash_decode_with_gqa_share_sparse,
)
from sglang.kernels.ops.attention.minimax_sparse.prefill import (
    topk_sparse as prefill_mod,
)
from sglang.kernels.ops.attention.minimax_sparse.prefill.topk_sparse import (
    flash_prefill_with_gqa_share_sparse,
)

DEVICE = "cuda"
NUM_Q_HEADS = 16
NUM_KV_HEADS = 1
HEAD_DIM = 128
DTYPE = torch.bfloat16
CONTEXT_LEN = 4096

# page_size -> does the fast path engage at block_size 128?
PAGE_CASES = [
    pytest.param(128, True, id="page128_eq_block"),
    pytest.param(256, True, id="page256_gt_block"),
    pytest.param(64, True, id="page64_two_runs"),
    pytest.param(16, True, id="page16_min_span"),
    pytest.param(8, False, id="page8_below_min_span"),
    pytest.param(48, False, id="page48_indivisible"),
    pytest.param(1, False, id="page1_per_token_gather"),
]


def build_page_table(batch_size, context_len, page_size, device):
    """Tokens contiguous within a page, pages scattered across the pool.

    This is how sglang's paged allocator lays a request out, and it is what the
    fast path relies on: consecutive positions inside one page are consecutive
    slots, while nothing is guaranteed across pages.
    """
    pages_per_req = (context_len + page_size - 1) // page_size
    total_pages = batch_size * pages_per_req
    max_slots = total_pages * page_size
    perm = torch.randperm(total_pages, device=device)
    within = torch.arange(context_len, device=device) % page_size
    page_of_tok = torch.arange(context_len, device=device) // page_size
    phys_page = perm.view(batch_size, pages_per_req)[:, page_of_tok]
    return (phys_page * page_size + within).to(torch.int32), max_slots


def _randn(*shape, generator):
    return torch.randn(*shape, dtype=DTYPE, device=DEVICE, generator=generator)


def _topk_idx(rows, num_blocks, topk, generator):
    u = torch.rand(NUM_KV_HEADS, rows, topk, device=DEVICE, generator=generator)
    return (u * num_blocks).to(torch.int32).clamp_(max=num_blocks - 1)


def _causal_topk_idx(abs_pos, topk, block_size, generator):
    """Selections a query may legally attend to: its own block or an earlier one.

    A row whose every selection lay in the future would softmax over an all-masked
    row and produce NaN, which is never ``torch.equal`` to itself and would mask a
    real divergence.
    """
    rows = abs_pos.numel()
    u = torch.rand(NUM_KV_HEADS, rows, topk, device=DEVICE, generator=generator)
    highest = ((abs_pos + block_size) // block_size).to(torch.float32)
    return (u * highest.view(1, rows, 1)).to(torch.int32)


def build_decode_inputs(
    batch_size, page_size, block_size, topk, generator, context_len=CONTEXT_LEN
):
    req_to_token, max_slots = build_page_table(
        batch_size, context_len, page_size, DEVICE
    )
    num_blocks = (context_len + block_size - 1) // block_size
    return dict(
        q=_randn(batch_size, NUM_Q_HEADS, HEAD_DIM, generator=generator),
        sink=None,
        k_cache=_randn(max_slots, NUM_KV_HEADS, HEAD_DIM, generator=generator),
        v_cache=_randn(max_slots, NUM_KV_HEADS, HEAD_DIM, generator=generator),
        req_to_token=req_to_token,
        seq_lens=torch.full(
            (batch_size,), context_len, dtype=torch.int32, device=DEVICE
        ),
        slot_ids=torch.arange(batch_size, dtype=torch.int64, device=DEVICE),
        block_size=block_size,
        topk_idx=_topk_idx(batch_size, num_blocks, topk, generator),
    )


def build_prefill_inputs(
    batch_size,
    page_size,
    block_size,
    topk,
    chunk_len,
    generator,
    context_len=CONTEXT_LEN,
):
    req_to_token, max_slots = build_page_table(
        batch_size, context_len, page_size, DEVICE
    )
    cu_seqlens = torch.arange(
        0, (batch_size + 1) * chunk_len, chunk_len, dtype=torch.int32, device=DEVICE
    )
    cu_seqblocks_q, max_seqblock_q, _, _, _, _ = get_cu_seqblocks(
        cu_seqlens, chunk_len, 1, block_size
    )
    # block_size_q is 1, so one row per query token: the chunk sits at the end of
    # the context, after a prefix of CONTEXT_LEN - chunk_len tokens.
    abs_pos = (torch.arange(chunk_len, device=DEVICE) + context_len - chunk_len).repeat(
        batch_size
    )
    assert abs_pos.numel() == int(cu_seqblocks_q[-1].item())
    return dict(
        q=_randn(batch_size * chunk_len, NUM_Q_HEADS, HEAD_DIM, generator=generator),
        k_cache=_randn(max_slots, NUM_KV_HEADS, HEAD_DIM, generator=generator),
        v_cache=_randn(max_slots, NUM_KV_HEADS, HEAD_DIM, generator=generator),
        sink=None,
        req_to_token=req_to_token,
        slot_ids=torch.arange(batch_size, dtype=torch.int64, device=DEVICE),
        topk_idx=_causal_topk_idx(abs_pos, topk, block_size, generator),
        block_size_q=1,
        block_size_k=block_size,
        cu_seqlens=cu_seqlens,
        seq_lens=torch.full(
            (batch_size,), context_len, dtype=torch.int32, device=DEVICE
        ),
        prefix_lens=torch.full(
            (batch_size,), context_len - chunk_len, dtype=torch.int32, device=DEVICE
        ),
        max_seqlen_q=chunk_len,
        cu_seqblocks_q=cu_seqblocks_q,
        max_seqblock_q=max_seqblock_q,
    )


def _assert_equal(o_gather, o_paged, what, page_size):
    diff = (o_gather.float() - o_paged.float()).abs()
    assert torch.equal(o_gather, o_paged), (
        f"paged {what} diverged at page_size={page_size}: "
        f"{int((diff > 0).sum())} of {diff.numel()} elements, max {diff.max().item()}"
    )


@pytest.mark.parametrize("page_size,fast_path", PAGE_CASES)
@pytest.mark.parametrize("batch_size", [1, 4, 32])
def test_decode_paged_is_bit_identical(page_size, fast_path, batch_size):
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    kwargs = build_decode_inputs(
        batch_size, page_size=page_size, block_size=128, topk=16, generator=gen
    )
    o_gather = flash_decode_with_gqa_share_sparse(**kwargs, page_size=1)
    o_paged = flash_decode_with_gqa_share_sparse(**kwargs, page_size=page_size)
    _assert_equal(o_gather, o_paged, "decode", page_size)


@pytest.mark.parametrize("page_size,fast_path", PAGE_CASES)
@pytest.mark.parametrize("batch_size", [1, 2])
def test_prefill_paged_is_bit_identical(page_size, fast_path, batch_size):
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    kwargs = build_prefill_inputs(
        batch_size,
        page_size=page_size,
        block_size=128,
        topk=16,
        chunk_len=512,
        generator=gen,
    )
    o_gather = flash_prefill_with_gqa_share_sparse(**kwargs, page_size=1)
    o_paged = flash_prefill_with_gqa_share_sparse(**kwargs, page_size=page_size)
    _assert_equal(o_gather, o_paged, "prefill", page_size)


@pytest.mark.parametrize("block_size", [16, 32, 64, 128])
def test_decode_paged_across_block_sizes(block_size):
    """page_size 128 held fixed while block_size varies.

    Every one of these divides 128, so all take the fast path -- block == page
    resolves in one lookup, block < page in one lookup per run.
    """
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    kwargs = build_decode_inputs(
        2, page_size=128, block_size=block_size, topk=16, generator=gen
    )
    o_gather = flash_decode_with_gqa_share_sparse(**kwargs, page_size=1)
    o_paged = flash_decode_with_gqa_share_sparse(**kwargs, page_size=128)
    _assert_equal(o_gather, o_paged, "decode", 128)


def test_min_page_span_agrees_across_phases():
    """Prefill and decode must stop taking the fast path at the same width."""
    assert decode_mod.MIN_PAGE_SPAN == prefill_mod.MIN_PAGE_SPAN


# ===== Benchmark =====
#
# Run with ``python test_sparse_gqa_paged.py --bench``: the same kernel call on a
# page_size 1 pool (per-token gather) and on a page_size 128 pool (fast path),
# each pool built the way the allocator would build it.
#
# The gain is the lookup elimination, not KV locality -- measured on H200, a
# page 128 layout forced through the gather path landed within noise of page 1.
#
# Decode is timed as a cuda graph replay, which is both how production runs it
# and the only way the number means anything: an eager launch costs ~85us on
# H200 (host-side, it moves with machine load) and buries the kernel under
# dispatch. Prefill runs eager, as it does in production.

BENCH_PAGE_SIZE = 128
BENCH_BLOCK_SIZE = 128
# Long enough that a topk still selects distinct blocks: a topk past
# context_len / block_size would clamp into repeats and read a cache-warm block
# over and over, which flatters both paths.
BENCH_CONTEXT_LEN = 65536
# topk counts BLOCKS, not tokens -- each selects BENCH_BLOCK_SIZE tokens. Past
# 32 blocks (4096 tokens) the attention is not sparse enough to be worth
# serving, so the table stops there.
BENCH_TOPKS = (16, 32)


def _bench_eager_ms(fn):
    # First call resolves the autotune config; keep it out of the timed region.
    fn()
    torch.cuda.synchronize()
    return do_bench(fn, warmup=25, rep=100, return_mode="median")


def _bench_graph_ms(fn):
    # Autotune and the TMA descriptor allocation have to settle on a side stream
    # before capture; capturing them would bake a tuning launch into the graph.
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    ms = do_bench(graph.replay, warmup=25, rep=100, return_mode="median")
    del graph
    return ms


def _bench_variants(kernel, build_inputs, bench_ms):
    """Latency at page_size 1 and at BENCH_PAGE_SIZE.

    One pool is built, timed and freed before the next -- at the largest batch
    the two do not fit side by side. Each build reseeds, so the two variants see
    identical q/k/v/topk_idx and differ only in the slot layout.
    """
    times = []
    for page_size in (1, BENCH_PAGE_SIZE):
        gen = torch.Generator(device=DEVICE).manual_seed(0)
        kwargs = build_inputs(page_size=page_size, generator=gen)
        times.append(
            bench_ms(lambda: kernel(**kwargs, page_size=page_size))  # noqa: F821
        )
        del kwargs
        torch.cuda.empty_cache()
    return tuple(times)


def _print_header(what, how, label_width):
    print(
        f"\n{what} [{how}] (context_len={BENCH_CONTEXT_LEN}, "
        f"block_size={BENCH_BLOCK_SIZE}, page_size={BENCH_PAGE_SIZE})"
    )
    print(
        f"{'topk counts BLOCKS of ' + str(BENCH_BLOCK_SIZE) + ' tokens; ':<{label_width}}"
        "kv tok = tokens attended per query row"
    )
    print(
        f"{'':<{label_width}}  {'kv tok':>7}  {'ctx %':>6}  {'page1':>10}"
        f"  {'page128':>10}  {'speedup':>9}"
    )


def _print_row(label, topk, times, label_width):
    t_page1, t_paged = times
    kv_tokens = topk * BENCH_BLOCK_SIZE
    print(
        f"{label:<{label_width}}  {kv_tokens:>7}  "
        f"{100 * kv_tokens / BENCH_CONTEXT_LEN:>5.1f}%  {t_page1 * 1e3:>9.1f}u"
        f"  {t_paged * 1e3:>9.1f}u  {t_page1 / t_paged:>8.2f}x"
    )


def bench_decode():
    label_width = 32
    _print_header("decode", "cuda graph replay", label_width)
    for batch_size in (1, 8, 32, 64, 128):
        for topk in BENCH_TOPKS:
            times = _bench_variants(
                flash_decode_with_gqa_share_sparse,
                lambda page_size, generator, bs=batch_size, tk=topk: build_decode_inputs(
                    bs,
                    page_size=page_size,
                    block_size=BENCH_BLOCK_SIZE,
                    topk=tk,
                    generator=generator,
                    context_len=BENCH_CONTEXT_LEN,
                ),
                _bench_graph_ms,
            )
            _print_row(f"bs={batch_size:<4} topk={topk:<4}", topk, times, label_width)


def bench_prefill():
    label_width = 32
    _print_header("prefill", "eager", label_width)
    for batch_size in (1, 2):
        for chunk_len in (512, 2048):
            for topk in BENCH_TOPKS:
                times = _bench_variants(
                    flash_prefill_with_gqa_share_sparse,
                    lambda page_size, generator, bs=batch_size, cl=chunk_len, tk=topk: build_prefill_inputs(
                        bs,
                        page_size=page_size,
                        block_size=BENCH_BLOCK_SIZE,
                        topk=tk,
                        chunk_len=cl,
                        generator=generator,
                        context_len=BENCH_CONTEXT_LEN,
                    ),
                    _bench_eager_ms,
                )
                _print_row(
                    f"bs={batch_size:<4} chunk={chunk_len:<5} topk={topk:<4}",
                    topk,
                    times,
                    label_width,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench", action="store_true", help="benchmark instead of test"
    )
    args, rest = parser.parse_known_args()
    if not args.bench:
        raise SystemExit(pytest.main([__file__, "-v", *rest]))
    print(torch.cuda.get_device_name(0))
    bench_decode()
    bench_prefill()
