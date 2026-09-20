"""Prefill candidate selection: causal tails, strided CP slices, and replay."""

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
    candidate_block_mask,
    select_candidate_block_ids,
)
from sglang.kernels.ops.attention.dsv4.prefill_candidates import (
    causal_block_max,
    select_prefill_candidate_block_ids,
    select_prefill_candidate_blocks,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def select_candidate_blocks(logits, compress_lens, topk_blocks, block_size):
    ids = select_candidate_block_ids(logits, compress_lens, topk_blocks, block_size)
    return candidate_block_mask(ids, logits.shape[-1], block_size)


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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
