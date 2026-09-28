"""Paged candidate FP4 logits, including changed-input graph replay."""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu")


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
@pytest.mark.parametrize("write_pages", [False, True])
def test_prefill_protocol_matches_torch(ratio, tail, budget, write_pages):
    from types import SimpleNamespace
    from unittest.mock import patch

    from sglang.srt.layers.attention.dsv4.v41_indexer import scoring
    from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import (
        BlockIds,
        DenseBlocksBackend,
    )
    from sglang.srt.layers.attention.dsv4.v41_indexer.types import PrefillInputs

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
        positions=lens * ratio - 1,
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
        inputs.out_raw_indices = torch.empty(rows, 8, dtype=torch.int32, device="cuda")
        inputs.out_page_indices = (
            torch.empty_like(inputs.out_raw_indices) if write_pages else None
        )
        inputs.reset_outputs = lambda: PrefillInputs.reset_outputs(inputs)

        # A compact consumer must not call the dense Torch scorer.
        with patch.object(scoring, "_TORCH_SCORE_BUDGET_BYTES", budget):
            if triton:
                with patch.object(
                    indexer, "scores", side_effect=AssertionError("dense scoring")
                ):
                    backend.consume_prefill(inputs, published)
            else:
                backend.consume_prefill(inputs, published)
        outputs.append((inputs.out_page_indices, inputs.out_raw_indices))
    if write_pages:
        torch.testing.assert_close(outputs[0][0], outputs[1][0])
    torch.testing.assert_close(outputs[0][1], outputs[1][1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("block_count", [0, 1, 3, 17])
def test_compact_paged_scores_and_finalize_replay(ratio, block_count):
    from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
        finish_paged_indexer_topk,
        fp4_index_logits_candidates,
        fp4_index_logits_paged,
        store_fp4_index_k_cache,
    )

    torch.manual_seed(84)
    rows, capacity, page_size = 4, 137, 64
    q = torch.randn(rows, 32, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(rows, 32, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(192, 128, device="cuda", dtype=torch.bfloat16)
    cache = torch.zeros(3, page_size * 68, device="cuda", dtype=torch.uint8)
    store_fp4_index_k_cache(
        keys, cache, torch.arange(192, device="cuda"), page_size=page_size
    )
    req = torch.tensor([2, 0, 2, 1], device="cuda", dtype=torch.int32)
    req_table = torch.full((3, capacity * ratio), -1, device="cuda", dtype=torch.int32)
    for r in range(3):
        req_table[r, ::ratio] = (
            torch.randperm(192, device="cuda")[:capacity].int() * ratio
        )
    lens = torch.tensor([0, 17, 130, 137], device="cuda", dtype=torch.int32)
    blocks = torch.stack(
        [torch.randperm(18, device="cuda")[:block_count] for _ in range(rows)]
    ).int()
    if block_count:
        blocks[0].fill_(-1)
        blocks[1, block_count // 2] = -1
    width = block_count * 8
    k = min(width, 16)

    def run():
        scores = fp4_index_logits_candidates(
            q, weights, req, req_table, lens, cache, page_size, capacity, ratio, blocks
        )
        indices = scores.topk(k, sorted=False).indices.int()
        # Strided outputs, an extra padding row, and more columns than selections.
        pages = torch.empty((rows + 1, 40), device="cuda", dtype=torch.int32)[:, :20]
        raw = torch.empty_like(pages)
        finish_paged_indexer_topk(
            indices,
            scores,
            lens,
            req,
            req_table,
            pages,
            raw,
            ratio,
            False,
            candidate_blocks=blocks,
        )
        return scores, indices, pages, raw

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, indices, pages, raw = run()
    for step in range(3):
        if step == 1:
            req.copy_(req.roll(1))
            lens.copy_(torch.tensor([137, 65, 9, 0], device="cuda", dtype=torch.int32))
            blocks.copy_(blocks.flip(1))
            q.neg_()
        elif step == 2:
            blocks.fill_(-1)
        pages.fill_(12345)
        raw.fill_(12345)
        graph.replay()
        dense = fp4_index_logits_paged(
            q, weights, req, req_table, lens, cache, page_size, capacity, ratio
        )
        positions = (
            blocks.long()[:, :, None] * 8 + torch.arange(8, device="cuda")
        ).flatten(1)
        valid = (positions >= 0) & (positions < lens[:, None]) & (positions < capacity)
        expected = dense.gather(1, positions.clamp(0, capacity - 1)).masked_fill(
            ~valid, -torch.inf
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert (pages[-1] == -1).all() and (raw[-1] == -1).all()
        for row in range(rows):
            cols = indices[row].long()
            cols = cols[expected[row, cols] > -torch.inf]
            logical = positions[row, cols].sort().values
            want_raw = torch.full_like(raw[row], -1)
            want_pages = torch.full_like(pages[row], -1)
            want_raw[: logical.numel()] = logical.int()
            want_pages[: logical.numel()] = (
                req_table[req[row].long(), logical * ratio] // ratio
            )
            torch.testing.assert_close(raw[row], want_raw, rtol=0, atol=0)
            torch.testing.assert_close(pages[row], want_pages, rtol=0, atol=0)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_paged_selection_uses_score_coordinates(compact, empty):
    """A causal K length is not a compact score-array scan length."""
    from types import SimpleNamespace
    from unittest.mock import patch

    from sglang.srt.layers.attention.dsv4.v41_indexer import scoring

    lens = torch.tensor([0, 17, 4096], dtype=torch.int32)
    width = (0 if empty else 24) if compact else 4096
    blocks = torch.tensor([[2, -1, 0]] * 3, dtype=torch.int32) if compact else None
    scan = torch.full_like(lens, width) if compact else None
    d = scoring.PagedDecodeScores(
        bs=3,
        lmax=4096,
        lens=lens,
        scores=torch.empty(3, width),
        req=torch.arange(3, dtype=torch.int32),
        req_to_token=torch.empty(3, 4096, dtype=torch.int32),
        ratio=1,
        plan=torch.empty(4, 2, dtype=torch.int32),
        has_candidate_mask=False,
        candidate_blocks=blocks,
        score_lens=scan,
    )
    out = torch.full((4, 512), 123, dtype=torch.int32)
    inputs = SimpleNamespace(out_page_indices=out, reset_outputs=lambda: out.fill_(-1))
    with (
        patch.object(scoring, "topk_transform_paged_v2") as topk,
        patch.object(scoring, "finish_paged_indexer_topk") as finish,
    ):
        scoring.select_decode(inputs, d, 512)
    if width == 0:
        topk.assert_not_called()
        finish.assert_not_called()
        assert (out == -1).all()
    else:
        topk.assert_called_once()
        args = topk.call_args.args
        assert args[1] is (scan if compact else lens)
        assert args[3].shape == (3, min(512, width))
        assert args[5] is d.plan
        finish.assert_called_once()
        assert finish.call_args.args[2] is lens
        assert finish.call_args.kwargs["candidate_blocks"] is blocks


@pytest.mark.parametrize("compact", [False, True])
def test_decode_plan_and_scorer_dispatch(compact):
    from types import SimpleNamespace
    from unittest.mock import patch

    from sglang.srt.layers.attention.dsv4.v41_indexer import scoring

    q = torch.zeros(3, 32, 128, dtype=torch.bfloat16)
    weights = torch.zeros(3, 32, dtype=torch.bfloat16)
    lens = torch.tensor([0, 17, 128], dtype=torch.int32)
    blocks = torch.tensor([[2, -1, 0]] * 3, dtype=torch.int32) if compact else None
    inputs = SimpleNamespace(
        indexer=SimpleNamespace(queries=lambda *_: q, head_weights=lambda _: weights),
        compress_ratio=1,
        req_rows=torch.tensor([1, 0, 1]),
        positions=lens - 1,
        paged_metadata=SimpleNamespace(max_compressed_seq_len=128),
        q_lora=q,
        freqs_cis=torch.ones(129),
        x=weights,
        layer_id=20,
    )
    req_table = torch.empty(2, 128, dtype=torch.int32)
    cache = torch.empty(2, 64 * 68, dtype=torch.uint8)
    with (
        patch.object(
            scoring, "get_platform", return_value=SimpleNamespace(is_sm90=True)
        ),
        patch.object(
            scoring, "plan_topk_v2", return_value=torch.empty(4, 2, dtype=torch.int32)
        ) as plan,
        patch.object(
            scoring, "fp4_index_logits_candidates", return_value=torch.empty(3, 24)
        ) as sparse,
        patch.object(
            scoring, "fp4_index_logits_paged", return_value=torch.empty(3, 128)
        ) as dense,
    ):
        d = scoring.decode_scores(
            inputs=inputs,
            req_to_token=req_table,
            token_to_kv_pool=SimpleNamespace(
                get_index_k_with_scale_buffer=lambda _: cache
            ),
            candidate_blocks=blocks,
        )
    torch.testing.assert_close(d.lens, lens)
    assert d.lmax == 128
    if compact:
        dense.assert_not_called()
        sparse.assert_called_once()
        assert sparse.call_args.args[-1] is blocks
        torch.testing.assert_close(plan.call_args.args[0], torch.full_like(lens, 24))
    else:
        sparse.assert_not_called()
        dense.assert_called_once()
        torch.testing.assert_close(plan.call_args.args[0], lens)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
