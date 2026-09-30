"""Paged candidate FP4 logits, including changed-input graph replay."""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
@pytest.mark.parametrize("replay", [False, True])
def test_paged_fp4_candidate_scores(replay):
    from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
        fp4_index_logits_decode,
        store_fp4_index_k_cache,
    )

    torch.manual_seed(84)
    rows, heads, length, page_size = 4, 32, 137, 64
    q = torch.randn(rows, heads, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(rows, heads, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(192, 128, device="cuda", dtype=torch.bfloat16)
    cache = torch.zeros(3, page_size * 68, device="cuda", dtype=torch.uint8)
    store_fp4_index_k_cache(
        keys, cache, torch.arange(192, device="cuda"), page_size=page_size
    )
    slots = torch.stack(
        [torch.randperm(192, device="cuda")[:length] for _ in range(rows)]
    )
    lens = torch.tensor([0, 17, 130, 137], device="cuda")
    blocks = torch.tensor(
        [[0, -1, 16], [2, 0, -1], [16, 8, 0], [16, 1, 8]],
        device="cuda",
        dtype=torch.int32,
    )

    def sparse():
        return fp4_index_logits_decode(
            q, weights, slots, lens, cache, page_size, candidate_blocks=blocks
        )

    actual = sparse()
    if replay:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                sparse()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = sparse()
        # Captured code must read new device values, not capture-time lengths/IDs.
        blocks.copy_(blocks.flip(1))
        lens.copy_(torch.tensor([8, 9, 65, 136], device="cuda"))
        graph.replay()
    dense = fp4_index_logits_decode(q, weights, slots, lens, cache, page_size)
    positions = (
        blocks.long()[:, :, None] * 8 + torch.arange(8, device="cuda")
    ).flatten(1)
    expected = dense.gather(1, positions.clamp(0, length - 1))
    expected.masked_fill_(
        (positions < 0) | (positions >= lens[:, None]) | (positions >= length),
        -torch.inf,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("topk", [0, 1, 6, 24])
def test_compact_topk_padding(topk):
    from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import compact_topk

    blocks = torch.tensor([[2, -1, 0], [-1, -1, -1]], dtype=torch.int32)
    positions = (blocks.long()[:, :, None] * 8 + torch.arange(8)).flatten(1)
    lengths = torch.tensor([19, 0])
    scores = positions.float().masked_fill(
        (positions < 0) | (positions >= lengths[:, None]), -torch.inf
    )
    actual = compact_topk(scores, blocks, lengths, topk, 23, 8).sort().values
    visible = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18]
    selected = sorted(visible[-topk:]) if topk else []
    expected = torch.tensor(
        [selected + [23] * (topk - len(selected)), [23] * topk], dtype=torch.int64
    )
    torch.testing.assert_close(actual, expected)
    empty = compact_topk(scores[:, :0], blocks[:, :0], lengths, topk, 23, 8)
    assert (empty == 23).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("tail", [False, True])
@pytest.mark.parametrize("budget", [1024, 1 << 20])
def test_prefill_protocol_matches_torch(ratio, tail, budget):
    from types import SimpleNamespace
    from unittest.mock import patch

    from sglang.srt.layers.attention.dsv4.v41_indexer import scoring
    from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import (
        BlockIds,
        DenseBlocksBackend,
    )
    from sglang.srt.layers.attention.dsv4.v41_indexer.types import Selection

    # Exact BF16 scores avoid cutoff ties: the same test checks slot mapping,
    # ragged causal lengths, tile boundaries and per-request bounded replay.
    q = torch.zeros(6, 32, 128, dtype=torch.bfloat16, device="cuda")
    q[:, 0, 0] = 1
    weights = torch.zeros(6, 32, dtype=torch.bfloat16, device="cuda")
    weights[:, 0] = 1
    keys = torch.zeros(128, 128, dtype=torch.bfloat16, device="cuda")
    keys[:, 0] = torch.arange(128, device="cuda")

    def torch_scores(q, k, w):
        return (torch.einsum("rhd,nd->rhn", q, k).relu() * w[..., None]).sum(1).float()

    indexer = SimpleNamespace(
        index_topk=8,
        queries=lambda q, _: q,
        head_weights=lambda w: w,
        scores=torch_scores,
    )
    pool = SimpleNamespace(get_low_ratio_index_k_dequant=lambda _, slots: keys[slots])
    slots = torch.arange(128 * ratio, device="cuda").reshape(2, 64 * ratio)
    lens = torch.tensor([0, 5, 37, 1, 17, 47], device="cuda")
    req = torch.tensor([0, 0, 0, 1, 1, 1], device="cuda")
    inputs = SimpleNamespace(
        indexer=indexer,
        layer_id=20,
        compress_ratio=ratio,
        q_lora=q,
        x=weights,
        positions=(lens * ratio - 1).clamp_min(0),
        req_rows=req,
        freqs_cis=torch.ones(128, device="cuda"),
        rows_per_request=[3, 3],
    )
    published = BlockIds(
        blocks=torch.tensor([[4, 0, -1]] * 6, dtype=torch.int32, device="cuda"),
        rows_per_request=[3, 3],
    )
    if tail:
        published = published.tail([2, 1])
        rows = torch.tensor([1, 2, 5], device="cuda")
        for name in ("q_lora", "x", "positions", "req_rows"):
            setattr(inputs, name, getattr(inputs, name)[rows])
        inputs.rows_per_request = [2, 1]
    rows = inputs.positions.numel()
    outputs = []
    for triton in (False, True):
        backend = DenseBlocksBackend(
            token_to_kv_pool=pool,
            req_to_token=slots,
            candidate_topk_blocks=3,
            candidate_block_size=8,
            use_deep_gemm_prefill=False,
            use_triton_candidates=triton,
        )
        out = Selection(
            torch.empty(rows, 8, dtype=torch.int32, device="cuda"),
            torch.empty(rows, 8, dtype=torch.int32, device="cuda"),
        )
        # A compact consumer must not call the dense Torch scorer.
        with patch.object(scoring, "_TORCH_SCORE_BUDGET_BYTES", budget):
            if triton:
                with patch.object(
                    indexer, "scores", side_effect=AssertionError("dense scoring")
                ):
                    backend.consume_prefill(inputs, published, out)
            else:
                backend.consume_prefill(inputs, published, out)
        outputs.append(out)
    torch.testing.assert_close(outputs[0].page_indices, outputs[1].page_indices)
    torch.testing.assert_close(outputs[0].raw_indices, outputs[1].raw_indices)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
