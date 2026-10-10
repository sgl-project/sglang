"""Prefill candidate selection: causal tails, strided CP slices, and replay."""

import pytest
import torch
import torch.nn.functional as F
import triton

from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
    candidate_block_mask,
    select_candidate_block_ids,
)
from sglang.kernels.ops.attention.dsv4.prefill_candidates import (
    causal_block_max,
    select_prefill_candidate_block_ids,
    select_prefill_candidate_blocks,
    topk_prefill_candidates,
)
from sglang.kernels.ops.attention.dsv4.topk import topk_transform_ragged_v2
from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import BlockIds
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def select_candidate_blocks(logits, compress_lens, topk_blocks, block_size):
    ids = select_candidate_block_ids(logits, compress_lens, topk_blocks, block_size)
    return candidate_block_mask(ids, logits.shape[-1], block_size)


def mask_topk_scores(scores, indices, offsets):
    columns = indices.to(torch.int64) - offsets[:, None]
    values = scores.gather(1, columns.clamp(0, scores.shape[1] - 1))
    valid = (columns >= 0) & (columns < scores.shape[1]) & (values > -torch.inf)
    return indices.masked_fill(~valid, -1)


def reference_scores(logits, lens, block):
    visible = logits.masked_fill(
        torch.arange(logits.shape[1], device=logits.device)[None, :] >= lens[:, None],
        -torch.inf,
    )
    scores = F.pad(visible, (0, -logits.shape[1] % block), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block)).amax(-1)
    last = (lens[:, None] - 1) // block
    scores = scores.masked_fill(
        torch.arange(scores.shape[1], device=logits.device) == last, torch.inf
    )
    return visible, scores


@pytest.mark.parametrize("width", [1, 7, 8, 9, 513, 8193, 67045])
@pytest.mark.parametrize("block", [3, 8, 16])
@pytest.mark.parametrize("tied", [False, True])
def test_source_matches_reference(width, block, tied):
    torch.manual_seed(42)
    rows = 9
    # Noncontiguous rows and columns exercise request views and CP metadata.
    storage = torch.randn((rows + 2, 2 * (width + 32)), device="cuda")
    logits = storage[1:-1, : 2 * width : 2]
    if tied:
        logits.round_()
    before = storage.clone()
    lens = torch.tensor(
        [
            0,
            1,
            min(width, block - 1),
            min(width, block),
            min(width, block + 1),
            width // 2,
            max(0, width - 1),
            width,
            width,
        ],
        device="cuda",
        dtype=torch.int32,
    )
    visible, ref_scores = reference_scores(logits, lens, block)
    actual = causal_block_max(logits, lens[:, None], block)
    torch.testing.assert_close(actual, ref_scores, rtol=0, atol=0)
    for k in [1, 3, 2048]:
        ref = select_candidate_blocks(visible, lens[:, None], k, block)
        actual = select_prefill_candidate_blocks(logits, lens[:, None], k, block)
        ids = select_prefill_candidate_block_ids(logits, lens[:, None], k, block)
        actual_ids_mask = candidate_block_mask(ids, width, block)
        assert torch.equal(actual, actual_ids_mask)
        if not tied:
            assert torch.equal(actual, ref)
        else:
            # topk does not promise a stable choice among equal scores.
            actual_blocks = actual[:, ::block]
            ref_blocks = ref[:, ::block]
            assert torch.equal(actual_blocks.sum(-1), ref_blocks.sum(-1))
            for row in range(rows):
                torch.testing.assert_close(
                    ref_scores[row][actual_blocks[row]].sort().values,
                    ref_scores[row][ref_blocks[row]].sort().values,
                    rtol=0,
                    atol=0,
                )
                if lens[row] > 0:
                    assert actual_blocks[row, (lens[row] - 1) // block]
    assert torch.equal(storage, before)


def test_source_zero_rows_and_width():
    for rows, width in [(0, 9), (3, 0), (0, 0)]:
        scores = torch.empty((rows, width), device="cuda")
        lens = torch.zeros(rows, dtype=torch.int32, device="cuda")
        out = select_prefill_candidate_blocks(scores, lens, 2, 8)
        assert out.shape == scores.shape


def test_source_nan_and_infinity():
    logits = torch.tensor(
        [
            [float("nan"), 1, 2, 3, 4, 5, 6, 7, 8],
            [float("-inf")] * 9,
            [float("inf")] * 9,
        ],
        device="cuda",
        dtype=torch.float32,
    )
    lens = torch.tensor([9, 0, 7], dtype=torch.int32, device="cuda")
    _, ref = reference_scores(logits, lens, 4)
    torch.testing.assert_close(causal_block_max(logits, lens, 4), ref, equal_nan=True)


def test_source_cuda_graph_replay():
    logits = torch.randn((17, 8193), device="cuda")
    lens = torch.full((17,), 8193, device="cuda", dtype=torch.int32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            select_prefill_candidate_blocks(logits, lens, 32, 8)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = select_prefill_candidate_blocks(logits, lens, 32, 8)
    for length in [1, 513, 8193]:
        logits.normal_()
        lens.fill_(length)
        graph.replay()
        visible, _ = reference_scores(logits, lens, 8)
        ref = select_candidate_blocks(visible, lens[:, None], 32, 8)
        assert torch.equal(actual, ref)


@pytest.mark.parametrize("width", [7, 511, 513, 8192, 8193, 16384, 16385, 67045])
@pytest.mark.parametrize("k", [512, 2048])
@pytest.mark.parametrize("mode", ["none", "all", "sparse", "underfill"])
def test_consumer_matches_masked_topk(width, k, mode):
    torch.manual_seed(43)
    rows, block = 9, 8
    padded_width = triton.cdiv(width, 32) * 32
    storage = torch.randn((rows + 2, padded_width), device="cuda")
    logits = storage[1:-1, :width]
    before = storage.clone()
    lens = torch.tensor(
        [
            0,
            1,
            min(width, 7),
            min(width, 8),
            min(width, 9),
            width // 2,
            max(0, width - 1),
            width,
            width,
        ],
        device="cuda",
        dtype=torch.int32,
    )
    blocks = triton.cdiv(width, block)
    mask_storage = torch.rand((rows, blocks + 3), device="cuda") > 0.75
    mask = mask_storage[:, :blocks]
    if mode == "none":
        mask.fill_(False)
    elif mode == "all":
        mask.fill_(True)
    elif mode == "underfill":
        mask.fill_(False)
        mask[:, 0] = True
    # Vary request-local to flattened-KV offsets; these are not score row starts.
    offsets = torch.arange(rows, device="cuda", dtype=torch.int32) * 100003
    expanded = mask.repeat_interleave(block, dim=-1)[:, :width]
    ref_storage = storage.clone()
    ref_scores = ref_storage[1:-1, :width]
    ref_scores.masked_fill_(~expanded, -torch.inf)
    ref = torch.empty((rows, k), device="cuda", dtype=torch.int32)
    topk_transform_ragged_v2(ref_scores, lens, out_offsets=offsets, out_indices=ref)
    ref = mask_topk_scores(ref_scores, ref, offsets)
    actual = torch.empty_like(ref)
    topk_prefill_candidates(logits, lens, mask, block, offsets, actual)
    assert torch.equal(storage, before)
    assert torch.equal(actual.sort(-1).values, ref.sort(-1).values)


def test_consumer_ties_and_nonfinite():
    torch.manual_seed(44)
    rows, width, k, block = 5, 16387, 512, 8
    storage = torch.randn((rows, triton.cdiv(width, 32) * 32), device="cuda").round_()
    logits = storage[:, :width]
    logits[0].fill_(-torch.inf)
    logits[1].zero_()
    logits[2, 0] = torch.inf
    lens = torch.full((rows,), width, device="cuda", dtype=torch.int32)
    mask = torch.rand((rows, triton.cdiv(width, block)), device="cuda") > 0.9
    mask[2, 0] = True
    offsets = torch.zeros(rows, device="cuda", dtype=torch.int32)
    out = torch.empty((rows, k), device="cuda", dtype=torch.int32)
    topk_prefill_candidates(logits, lens, mask, block, offsets, out)
    expanded = mask.repeat_interleave(block, -1)[:, :width]
    ref = logits.masked_fill(~expanded, -torch.inf).topk(k, dim=-1).values
    for row in range(rows):
        chosen = out[row][out[row] >= 0].long()
        assert chosen.unique().numel() == chosen.numel()
        assert expanded[row, chosen].all()
        expected = ref[row][ref[row] > -torch.inf].sort().values
        torch.testing.assert_close(
            logits[row, chosen].sort().values, expected, rtol=0, atol=0
        )


def test_consumer_cuda_graph_tail_rows():
    torch.manual_seed(45)
    rows, width, k, block = 13, 8193, 512, 8
    logits_storage = torch.randn((rows, triton.cdiv(width, 32) * 32), device="cuda")
    logits = logits_storage[:, :width]
    lens = torch.full((rows,), width, device="cuda", dtype=torch.int32)
    offsets = torch.arange(rows, device="cuda", dtype=torch.int32) * width
    full = BlockIds(
        blocks=torch.full((30, 1), -1, device="cuda", dtype=torch.int32),
        rows_per_request=[3, 3, 4, 3, 5, 4, 3, 3, 2],
        prefill_mask=torch.rand((30, triton.cdiv(width, block)), device="cuda") > 0.6,
    )
    tail_rows = torch.tensor(
        [1, 2, 5, 8, 9, 12, 16, 17, 21, 24, 27, 28, 29], device="cuda"
    )
    candidates = full.tail([2, 1, 2, 1, 2, 1, 1, 1, 2])
    assert torch.equal(candidates.prefill_mask, full.prefill_mask[tail_rows])
    out = torch.empty((rows, k), device="cuda", dtype=torch.int32)
    # Compile/warm up before capture on a separate stream.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            topk_prefill_candidates(
                logits, lens, candidates.prefill_mask, block, offsets, out
            )
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        topk_prefill_candidates(
            logits, lens, candidates.prefill_mask, block, offsets, out
        )
    for length in [0, 7, 8193]:
        lens.fill_(length)
        candidates.prefill_mask.copy_(
            torch.rand_like(candidates.prefill_mask, dtype=torch.float32) > 0.5
        )
        graph.replay()
        ref = logits_storage.clone()[:, :width]
        ref.masked_fill_(
            ~candidates.prefill_mask.repeat_interleave(block, -1)[:, :width], -torch.inf
        )
        expected = torch.empty_like(out)
        topk_transform_ragged_v2(ref, lens, out_offsets=offsets, out_indices=expected)
        expected = mask_topk_scores(ref, expected, offsets)
        assert torch.equal(out.sort(-1).values, expected.sort(-1).values)


@pytest.mark.parametrize("block", [3, 8, 16])
@pytest.mark.parametrize("cached_mask", [False, True])
def test_publish_consume_ragged_cp_rows(cached_mask, block):
    from types import SimpleNamespace

    from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import (
        _consume_tile_blocks,
        _publish_tile_blocks,
    )

    torch.manual_seed(46)
    q_lens, kv_lens = [0, 2, 3, 4], [31, 0, 17, 31]
    logits = torch.randn((9, 32), device="cuda")
    original = logits.clone()
    lens = torch.tensor(
        [0, 0, 1, 8, 17, 7, 16, 23, 31], device="cuda", dtype=torch.int32
    )
    starts = torch.tensor(
        [31, 31, 31, 31, 31, 48, 48, 48, 48], device="cuda", dtype=torch.int32
    )
    data = SimpleNamespace(
        rows_per_request=q_lens,
        lens_per_request=kv_lens,
        compress_lens=lens,
        request_starts=starts,
    )
    blocks = torch.full((9, 2), -1, dtype=torch.int32, device="cuda")
    # Tiles cross request boundaries, including zero-local-row and zero-KV requests.
    tiles = [slice(0, 3), slice(3, 6), slice(6, 9)]
    for tile in tiles:
        _publish_tile_blocks(
            data=data,
            tile=tile,
            logits=logits[tile],
            blocks=blocks[tile],
            topk_blocks=2,
            block_size=block,
        )
    assert torch.equal(logits, original)
    mask = candidate_block_mask(blocks, triton.cdiv(32, block), 1)
    assert not mask[:2].any()
    start = 0
    for count, width in zip(q_lens, kv_lens):
        if count and width:
            visible, _ = reference_scores(
                logits[start : start + count, :width],
                lens[start : start + count],
                block,
            )
            expected = select_candidate_blocks(
                visible, lens[start : start + count, None], 2, block
            )
            expanded = mask[start : start + count].repeat_interleave(block, -1)
            assert torch.equal(expanded[:, :width], expected)
            assert not expanded[:, width + (-width % block) :].any()
        start += count
    out = torch.empty((9, 512), dtype=torch.int32, device="cuda")
    for tile in tiles:
        _consume_tile_blocks(
            data=data,
            tile=tile,
            logits=logits[tile],
            blocks=blocks[tile],
            block_size=block,
            out=out[tile],
            block_mask=mask[tile] if cached_mask else None,
        )
    reference = logits.masked_fill(
        ~mask.repeat_interleave(block, -1)[:, :32], -torch.inf
    )
    expected = torch.empty_like(out)
    topk_transform_ragged_v2(reference, lens, out_offsets=starts, out_indices=expected)
    expected = mask_topk_scores(reference, expected, starts)
    assert torch.equal(out.sort(-1).values, expected.sort(-1).values)
    assert torch.equal(logits, original)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
