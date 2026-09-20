"""The DeepGEMM two-level indexer on the dense prefill path: the published
block table must be the torch selection, and a consumer scoring its blocks
sparsely must pick the positions the mask indexer picks on the dense scores."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.fp4_indexer import quantize_fp4_indexer_tensor
from sglang.srt.layers.attention.dsv4.candidate_indexer import (
    PrefillIndexerInputs,
    select_candidate_blocks,
)
from sglang.srt.layers.attention.dsv4.dense_prefill_indexer import (
    DenseCandidateIndexer,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

HEADS, DIM = 32, 128
TOPK_BLOCKS, BLOCK, TOPK = 2048, 8, 512
PAGE = 128  # index-K pool page size under DeepGEMM's paged sparse logits


def _indexer():
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("DeepGEMM paged sparse MQA logits need SM100")
    from sglang.srt.layers.attention.dsv4.candidate_indexer_deep_gemm import (
        DeepGemmCandidateIndexer,
    )

    return DeepGemmCandidateIndexer(TOPK_BLOCKS, BLOCK)


def _case(rows, ctx, seed):
    """One request: `rows` query tokens whose causal lengths end at `ctx`, its
    index K in the pool page layout [pages, PAGE * 64 payload | PAGE * 4 scale]."""
    torch.manual_seed(seed)
    dev = "cuda"
    n_slots = (ctx + PAGE - 1) // PAGE * PAGE
    k_fp4, k_sf = quantize_fp4_indexer_tensor(
        torch.randn(n_slots, DIM, device=dev, dtype=torch.bfloat16), rne=True
    )
    n_pages = n_slots // PAGE
    k_cache = torch.cat(
        [
            k_fp4.view(torch.uint8).reshape(n_pages, PAGE * 64),
            k_sf.view(torch.uint8).reshape(n_pages, PAGE * 4),
        ],
        1,
    ).view(n_pages, PAGE, 1, 68)
    q_fp4, q_sf = quantize_fp4_indexer_tensor(
        torch.randn(rows * HEADS, DIM, device=dev, dtype=torch.bfloat16), rne=True
    )
    q_fp4, q_sf = q_fp4.view(rows, HEADS, 64), q_sf.view(rows, HEADS)
    weights = torch.rand(rows, HEADS, device=dev)
    lens = torch.linspace(ctx - rows + 1, ctx, rows, device=dev).to(torch.int32)
    lens[0] = 0  # a query that sees no compressed position yet
    width = (ctx + 7) // 8 * 8
    from deep_gemm import fp8_fp4_mqa_logits

    ks = torch.zeros(rows, device=dev, dtype=torch.int32)
    dense = fp8_fp4_mqa_logits(
        (q_fp4, q_sf),
        (k_fp4[:width], k_sf[:width]),
        weights,
        ks,
        ks + lens,
        False,
        width,
    )
    page_table = (
        torch.arange(n_pages, device=dev, dtype=torch.int32)[None].expand(rows, -1)
    ).contiguous()

    inputs = PrefillIndexerInputs(
        q_fp4=q_fp4,
        q_sf=q_sf,
        weights=weights,
        compress_lens=lens,
        request_starts=ks,
        lens_per_request=[ctx],
        rows_per_request=[rows],
        kv=(k_fp4[:width], k_sf[:width]),
        k_cache=k_cache,
        page_size=PAGE,
        # KV pages of PAGE tokens at ratio 1: the index page table is the KV one
        kv_page_table=page_table,
        kv_page_size=PAGE,
        compress_ratio=1,
    )
    return dict(dense=dense, lens=lens, inputs=inputs, rows=rows)


def _reference_blocks(dense, lens):
    j = torch.arange(dense.shape[1], device=dense.device)
    scores = dense.masked_fill(j[None, :] >= lens[:, None], -torch.inf)
    keep = select_candidate_blocks(
        scores, lens[:, None], topk_blocks=TOPK_BLOCKS, block_size=BLOCK
    )
    return keep.unflatten(1, (-1, BLOCK)).any(-1)  # [rows, blocks] bool


def _publish(indexer, c):
    """The source layer's own top-k goes to `own`; the table is returned."""
    own = torch.full((c["rows"], TOPK), -1, dtype=torch.int32, device="cuda")
    return indexer.publish_prefill(c["inputs"], own), own


def _select(indexer, published, c, inputs=None, rows=None):
    positions = torch.full(
        (rows or c["rows"], TOPK), -1, dtype=torch.int32, device="cuda"
    )
    indexer.select_prefill(published, inputs or c["inputs"], positions)
    return positions


@pytest.mark.parametrize("rows,ctx", [(256, 40000), (64, 3000)])
@torch.inference_mode()
def test_publish_prefill_blocks_match_torch_selection(rows, ctx):
    indexer = _indexer()
    c = _case(rows, ctx, seed=rows + ctx)
    table, own = _publish(indexer, c)
    expected = _reference_blocks(c["dense"], c["lens"])
    nb = expected.shape[1]
    # INT32_MAX padding lands in a spare column instead of clearing a block
    got = torch.zeros(rows, nb + 1, dtype=torch.bool, device="cuda")
    got.scatter_(1, table.blocks.clamp_max(nb).long(), True)
    assert torch.equal(got[:, :nb], expected)
    assert not got[0, :nb].any(), "an empty row keeps no block"
    # the source's own top-k from the same pass is the dense implementation's
    _, own_dense = _publish(DenseCandidateIndexer(TOPK_BLOCKS, BLOCK), c)
    for r in range(rows):
        assert set(own[r].tolist()) == set(own_dense[r].tolist()), r


@pytest.mark.parametrize("rows,ctx", [(256, 40000), (64, 3000)])
@torch.inference_mode()
def test_select_prefill_matches_dense_indexer(rows, ctx):
    """Both implementations of the protocol select the same positions from the
    same scores, up to the sparse kernel's bf16 rounding at the selection
    boundary."""
    indexer = _indexer()
    c = _case(rows, ctx, seed=rows * 3 + ctx)
    positions = _select(indexer, _publish(indexer, c)[0], c)

    dense = DenseCandidateIndexer(TOPK_BLOCKS, BLOCK)
    expected = _select(dense, _publish(dense, c)[0], c)
    keep = _reference_blocks(c["dense"], c["lens"]).repeat_interleave(BLOCK, dim=1)

    assert (positions[0] == -1).all()
    for r in range(1, rows):
        got = positions[r][positions[r] >= 0].long()
        want = expected[r][expected[r] >= 0].long()
        assert got.numel() == want.numel(), (r, got.numel(), want.numel())
        assert keep[r, got].all(), f"row {r} selected outside its blocks"
        common = len(set(got.tolist()) & set(want.tolist()))
        assert common >= 0.95 * want.numel(), (r, common, want.numel())
        # a disagreement sits at the selection boundary: the sparse kernel's
        # bf16 weights and accumulation move a score by a few bf16 ulps
        floor = c["dense"][r, want].min()
        assert (c["dense"][r, got] >= floor - floor.abs() * 2**-5).all(), r


@torch.inference_mode()
def test_prefill_tail_rebuilds_the_tail():
    """A late-layer tail keeps the last rows of each request: the rebuilt table
    carries those rows' blocks and selects what the full table selects for them
    (up to the sparse kernel's row pairing, which moves boundary scores)."""
    indexer = _indexer()
    rows, tail = 128, 16
    c = _case(rows, 20000, seed=7)
    table, _ = _publish(indexer, c)
    idx = torch.arange(rows - tail, rows, device="cuda")
    sub = indexer.prefill_tail(table, [tail])
    assert torch.equal(sub.blocks, table.blocks[idx])
    assert torch.equal(sub.valid_lens, table.valid_lens[idx])
    assert torch.equal(sub.compress_lens, table.compress_lens[idx])

    full = _select(indexer, table, c)
    inputs = c["inputs"]
    tail_inputs = PrefillIndexerInputs(
        q_fp4=inputs.q_fp4[idx],
        q_sf=inputs.q_sf[idx],
        weights=inputs.weights[idx],
        compress_lens=inputs.compress_lens[idx],
        request_starts=inputs.request_starts[idx],
        lens_per_request=inputs.lens_per_request,
        rows_per_request=[tail],
        kv=inputs.kv,
        k_cache=inputs.k_cache,
        page_size=inputs.page_size,
        kv_page_table=inputs.kv_page_table[idx],
        kv_page_size=inputs.kv_page_size,
        compress_ratio=inputs.compress_ratio,
    )
    part = _select(indexer, sub, c, inputs=tail_inputs, rows=tail)
    for r in range(tail):
        a = set(full[idx[r]].tolist()) - {-1}
        b = set(part[r].tolist()) - {-1}
        assert len(a & b) >= 0.98 * len(a), (r, len(a & b), len(a))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
