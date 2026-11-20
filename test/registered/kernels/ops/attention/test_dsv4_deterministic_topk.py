"""SGLANG_ENABLE_DETERMINISTIC_SPARSE_TOPK: the DeepSeek-V4 / V4.1 sparse top-k entry
points pick the same indices in the same order on every run.

The default kernels emit in thread-arrival order (top-k v1/v2, the bf16 candidate
consumer, DeepSelect's cluster tier), and on tied scores some of them also change
which positions they keep. Sparse attention sums in index order, so either changes
greedy outputs between identical requests (#46798). With the flag on, every entry
point here must return bitwise-identical outputs across eager runs and CUDA-graph
replays, keep the true top-k scores (as a multiset), and keep the default kernels'
output contract (page transform, offsets, ``-1`` padding). Each case runs on
DeepSelect and on the torch selection used when DeepSelect cannot take the operands.
"""

from __future__ import annotations

import sys

import pytest
import torch

from sglang.kernels.ops.attention import deep_select
from sglang.kernels.ops.attention.dsv4 import candidate_blocks
from sglang.kernels.ops.attention.dsv4 import topk as topk_mod
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not deep_select.is_deep_select_supported(),
    reason="requires CUDA SM90, SM100, or SM103",
)

K = 512
PAGE = 64
RUNS = 8


@pytest.fixture(autouse=True)
def deterministic_topk():
    with envs.SGLANG_ENABLE_DETERMINISTIC_SPARSE_TOPK.override(True):
        yield


@pytest.fixture(params=["deep_select", "torch"])
def backend(request, monkeypatch):
    if request.param == "torch":
        monkeypatch.setattr(topk_mod, "_deep_select_takes", lambda *a, **k: False)
    return request.param


def make_scores(bs: int, n: int, kind: str, seed: int, width: int | None = None):
    """``kind``: "plateau" keeps fewer than K positive scores and the rest exactly 0
    (the default v2 kernel then changes its selected set from run to run); "bf16"
    rounds to bf16 values, which ties the k-th score with dozens of others."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    width = width or (n + 255) // 256 * 256
    s = torch.randn(bs, width, generator=g, device="cuda")
    if kind == "plateau":
        keep = torch.rand(bs, width, generator=g, device="cuda") < 300 / n
        s = torch.where(keep, s.abs() + 0.01, torch.zeros_like(s))
    else:
        s = s.bfloat16().float()
    lens = torch.full((bs,), n, dtype=torch.int32, device="cuda")
    lens[0] = max(1, n // 3)  # a short row: -1 padding when below K
    pages = -(-width // PAGE)
    table = torch.stack(
        [
            torch.randperm(4 * pages, generator=g, device="cuda")[:pages]
            for _ in range(bs)
        ]
    ).to(torch.int32)
    return s, lens, table


def repeat(fn):
    """RUNS eager outputs, then RUNS replays of one captured graph."""
    fn()
    torch.cuda.synchronize()
    outs = [fn().clone() for _ in range(RUNS)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    for _ in range(RUNS):
        graph.replay()
        outs.append(out.clone())
    return outs


def check(outs, scores, lens, k, to_positions):
    first = outs[0]
    for o in outs[1:]:
        assert torch.equal(o, first), "output changes between runs"
    for b in range(scores.shape[0]):
        n = min(k, int(lens[b]))
        pos = to_positions(b, first[b].cpu())
        valid = pos[:n]
        assert bool((valid >= 0).all()) and bool((pos[n:] == -1).all())
        assert len(set(valid.tolist())) == n, "an index repeats"
        assert bool((valid < int(lens[b])).all()), "an index is past the row length"
        row = scores[b, : int(lens[b])].float()
        want = row.topk(n).values.sort().values
        got = scores[b, valid.cuda().long()].float().sort().values
        assert torch.equal(got, want), "a top-k score is missing"


def inverse_page(table_row):
    inv = {phys: p for p, phys in enumerate(table_row.tolist())}

    def to_pos(slots):
        return torch.tensor(
            [inv[v // PAGE] * PAGE + v % PAGE if v >= 0 else -1 for v in slots.tolist()]
        )

    return to_pos


SHAPES = [(1, 300), (1, 8193), (8, 4097), (2, 40000)]


@pytest.mark.parametrize("kind", ["plateau", "bf16"])
@pytest.mark.parametrize("bs,n", SHAPES)
@pytest.mark.parametrize("entry", ["paged_v2", "raw_v2", "paged_v1"])
def test_paged(entry, bs, n, kind, backend):
    s, lens, table = make_scores(bs, n, kind, seed=n + bs)
    plan = topk_mod.plan_topk_v2(lens)

    def run():
        out = torch.empty(bs, K, dtype=torch.int32, device="cuda")
        if entry == "paged_v2":
            topk_mod.topk_transform_paged_v2(s, lens, table, out, PAGE, plan)
        elif entry == "raw_v2":
            topk_mod.topk_transform_paged_v2(s, lens, None, out, 1, plan)
        else:
            topk_mod.topk_transform_paged(s, lens, table, out, PAGE)
        return out

    if entry == "raw_v2":
        to_pos = lambda b, row: row  # noqa: E731
    else:
        inverses = [inverse_page(table[b].cpu()) for b in range(bs)]
        to_pos = lambda b, row: inverses[b](row)  # noqa: E731
    check(repeat(run), s, lens, K, to_pos)


@pytest.mark.parametrize("kind", ["plateau", "bf16"])
@pytest.mark.parametrize("bs,n", SHAPES)
@pytest.mark.parametrize("with_starts", [False, True])
def test_ragged_v2(bs, n, kind, with_starts, backend):
    s, lens, _ = make_scores(bs, n, kind, seed=n + bs)
    offsets = torch.arange(bs, dtype=torch.int32, device="cuda") * 100000
    starts = None
    src = s
    if with_starts:  # the reference sees each row from its start
        starts = torch.arange(bs, dtype=torch.int32, device="cuda") * 7
        src = torch.full_like(s, -torch.inf)
        for b in range(bs):
            st = int(starts[b])
            src[b, : s.shape[1] - st] = s[b, st:]

    def run():
        out = torch.empty(bs, K, dtype=torch.int32, device="cuda")
        topk_mod.topk_transform_ragged_v2(
            s, lens, out_offsets=offsets, out_indices=out, row_starts=starts
        )
        return out

    to_pos = lambda b, row: torch.where(row >= 0, row - b * 100000, -1)  # noqa: E731
    check(repeat(run), src, lens, K, to_pos)


@pytest.mark.parametrize("kind", ["plateau", "bf16"])
@pytest.mark.parametrize("bs,n", [(1, 8193), (8, 4097), (2, 16384)])
def test_sparse_consumer(bs, n, kind, backend):
    """``topk_transform_sparse`` (V4.1 candidate consumers on SM100) over bf16 logits."""
    s, lens, table = make_scores(bs, n, kind, seed=n + bs)
    width = (n + 63) // 64 * 64
    logits = torch.zeros(bs, 2048 * 8, device="cuda", dtype=torch.bfloat16)
    logits[:, :width] = s[:, :width].bfloat16()
    sub = torch.arange(8, device="cuda", dtype=torch.int32)
    blocks = (table[:, :, None] * 8 + sub).flatten(1)[:, :2048]
    pad = torch.full(
        (bs, 2048 - blocks.shape[1]), 2**31 - 1, dtype=torch.int32, device="cuda"
    )
    blocks = torch.cat([blocks, pad], 1).contiguous()

    def run():
        out = torch.empty(bs, K, dtype=torch.int32, device="cuda")
        topk_mod.topk_transform_sparse(logits, lens, blocks, out)
        return out

    inverses = [{v: i for i, v in enumerate(blocks[b].tolist())} for b in range(bs)]

    def to_pos(b, row):
        return torch.tensor(
            [inverses[b][v // 8] * 8 + v % 8 if v >= 0 else -1 for v in row.tolist()]
        )

    check(repeat(run), logits, lens, K, to_pos)


def test_deep_select_skips_clusters():
    """bf16 rows past the cluster threshold stay on the single-CTA kernel."""
    s, lens, table = make_scores(2, 65536, "bf16", seed=7)
    sb = s.bfloat16()

    def run():
        out = torch.empty(2, K, dtype=torch.int32, device="cuda")
        deep_select.topk_page_transform(
            sb, K, page_table=table, page_size=PAGE, end=lens, output_idx=out
        )
        return out

    inverses = [inverse_page(table[b].cpu()) for b in range(2)]
    check(repeat(run), sb, lens, K, lambda b, row: inverses[b](row))


@pytest.mark.parametrize("bs,width", [(1, 1048580), (3, 20000), (1, 32768)])
def test_amax_topk_blocks(bs, width, backend):
    """The V4.1 candidate source's block selection. 1048580 is the ratio-1 score
    width under a 1M-token context; its block keys are not a multiple of 32 floats."""
    # rows padded to 128 bytes, as fp4_index_logits_paged allocates them
    padded, lens, _ = make_scores(
        bs, min(width, 40000), "bf16", seed=bs, width=-(-width // 32) * 32
    )
    s = padded[:, :width]
    nblocks = (lens + 7) // 8
    cols = torch.arange(padded.shape[1], device="cuda")
    keys = padded.masked_fill(cols >= lens[:, None], -torch.inf)
    keys = keys.view(bs, -1, 8).amax(-1)
    keys[torch.arange(bs, device="cuda"), nblocks.long() - 1] = torch.inf

    def run():
        return candidate_blocks.amax_topk_blocks(s, lens, nblocks, 2048)

    check(repeat(run), keys, nblocks, 2048, lambda b, row: row)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
