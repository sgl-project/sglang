"""Paged candidate FP4 logits, including changed-input graph replay."""

import sys

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.v41_indexer.types import Selection
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@pytest.mark.parametrize("width", [0, 24])
@pytest.mark.parametrize("write_raw", [False, True])
def test_compact_prefill_dispatch(width, write_raw):
    from types import SimpleNamespace
    from unittest.mock import patch

    from sglang.srt.layers.attention.dsv4.v41_indexer import scoring

    chunk = SimpleNamespace(
        scores=torch.empty(2, width),
        lens=torch.tensor([3, 19]),
        tok=torch.tensor([3, 1]),
        candidate_blocks=torch.zeros(2, width // 8, dtype=torch.int32),
    )
    inputs = SimpleNamespace(req_rows=torch.tensor([0, 1, 0, 2]), compress_ratio=2)
    out = Selection(
        page_indices=torch.full((4, 8), -1, dtype=torch.int32),
        raw_indices=torch.full((4, 8), -1, dtype=torch.int32) if write_raw else None,
    )
    with (
        patch.object(scoring, "topk_transform_ragged_v2") as topk,
        patch.object(scoring, "finish_paged_indexer_topk") as finish,
    ):
        scoring.select_compact_prefill(
            inputs, out, SimpleNamespace(k=8), chunk, torch.empty(3, 40)
        )
    if not width:
        topk.assert_not_called()
        finish.assert_not_called()
        assert (out.page_indices == -1).all()
        return
    topk.assert_called_once()
    torch.testing.assert_close(
        topk.call_args.args[1], torch.full((2,), width, dtype=torch.int32)
    )
    assert topk.call_args.kwargs["out_indices"].dtype == torch.int32
    assert not topk.call_args.kwargs["out_offsets"].any()
    finish.assert_called_once()
    assert finish.call_args.args[2] is chunk.lens
    assert finish.call_args.kwargs["output_rows"] is chunk.tok
    assert finish.call_args.kwargs["candidate_blocks"] is chunk.candidate_blocks
    torch.testing.assert_close(finish.call_args.args[3], torch.tensor([2, 1]))
    assert finish.call_args.args[5] is out.page_indices
    assert finish.call_args.args[6] is out.raw_indices


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("write_pages", [False, True])
@pytest.mark.parametrize("topk", [1, 6, 24])
@pytest.mark.parametrize("use_v2", [False, True])
def test_compact_prefill_finish(ratio, write_pages, topk, use_v2):
    from sglang.kernels.ops.attention.dsv4.fp4_indexer import finish_paged_indexer_topk
    from sglang.kernels.ops.attention.dsv4.topk import topk_transform_ragged_v2

    device = "cuda"
    blocks = torch.tensor([[2, -1, 0], [-1, -1, -1]], dtype=torch.int32, device=device)
    positions = (
        blocks.long()[:, :, None] * 8 + torch.arange(8, device=device)
    ).flatten(1)
    lengths = torch.tensor([19, 0], device=device)
    scores = positions.float().masked_fill(
        (positions < 0) | (positions >= lengths[:, None]), -torch.inf
    )
    indices = scores.topk(topk, dim=-1, sorted=False).indices
    if use_v2:
        indices = torch.empty((2, topk), dtype=torch.int32, device=device)
        topk_transform_ragged_v2(
            scores,
            torch.full((2,), 24, dtype=torch.int32, device=device),
            out_offsets=torch.zeros(2, dtype=torch.int32, device=device),
            out_indices=indices,
        )
    # Reverse/noncontiguous destinations; untouched rows and output padding matter.
    output_rows = torch.tensor([3, 1], device=device)
    raw = torch.full((5, topk + 3), -77, dtype=torch.int32, device=device)
    pages = torch.full_like(raw, -77) if write_pages else None
    req = torch.tensor([1, 0], device=device)
    table = torch.arange(2 * 32 * ratio, device=device).reshape(2, 32 * ratio)
    finish_paged_indexer_topk(
        indices,
        scores,
        lengths,
        req if write_pages else None,
        table if write_pages else None,
        pages,
        raw,
        ratio,
        True,
        candidate_blocks=blocks,
        output_rows=output_rows,
    )
    visible = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18]
    selected = sorted(visible[-topk:])
    expected = torch.full_like(raw, -77)
    expected[1].fill_(-1)
    expected[3].fill_(-1)
    expected[3, : len(selected)] = torch.tensor(
        selected, device=device, dtype=torch.int32
    )
    torch.testing.assert_close(raw, expected)
    if write_pages:
        expected[3, : len(selected)] += 32
        torch.testing.assert_close(pages, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="SM90 prefill dispatch",
)
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("tail", [False, True])
@pytest.mark.parametrize("budget", [1024, 1 << 20])
@pytest.mark.parametrize("write_raw", [False, True])
def test_prefill_protocol_matches_torch(ratio, tail, budget, write_raw):
    from types import SimpleNamespace
    from unittest.mock import patch

    from sglang.srt.layers.attention.dsv4.v41_indexer import scoring
    from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import (
        BlockIds,
        DenseBlocksBackend,
    )

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
        seq_lens_cpu=[37 * ratio, 47 * ratio],
        req_pool_indices=torch.tensor([0, 1], device="cuda"),
        req_pool_indices_cpu=[0, 1],
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
            page_indices=torch.empty(rows, 8, dtype=torch.int32, device="cuda"),
            raw_indices=(
                torch.empty(rows, 8, dtype=torch.int32, device="cuda")
                if write_raw
                else None
            ),
        )

        # A compact consumer must not call the dense Torch scorer.
        with (
            patch.object(scoring, "_TORCH_SCORE_BUDGET_BYTES", budget),
            envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER.override(False),
            envs.SGLANG_OPT_USE_FUSED_DSV41_INDEXER_SCORES.override(False),
        ):
            if triton:
                with patch.object(
                    indexer, "scores", side_effect=AssertionError("dense scoring")
                ):
                    backend.consume_prefill(inputs, published, out)
                    # Warm compilation before checking the actual lane submission.
                    torch.cuda.synchronize()
                    previous = torch.cuda.get_sync_debug_mode()
                    try:
                        torch.cuda.set_sync_debug_mode("error")
                        backend.consume_prefill(inputs, published, out)
                    finally:
                        torch.cuda.set_sync_debug_mode(previous)
            else:
                backend.consume_prefill(inputs, published, out)
        outputs.append((out.page_indices, out.raw_indices))
    torch.testing.assert_close(outputs[0][0], outputs[1][0])
    if write_raw:
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
    selection = Selection(out, torch.full_like(out, 123))
    with (
        patch.object(scoring, "topk_transform_paged_v2") as topk,
        patch.object(scoring, "finish_paged_indexer_topk") as finish,
    ):
        scoring.select_decode(selection, d, 512)
    if width == 0:
        topk.assert_not_called()
        finish.assert_not_called()
        assert (out == -1).all()
        assert (selection.raw_indices == -1).all()
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
            out=Selection(torch.empty(3, 8, dtype=torch.int32), None),
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


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("sm90", [False, True])
@pytest.mark.parametrize("block_size", [8, 16])
def test_candidate_factory_admission_is_fixed_at_construction(
    enabled, sm90, block_size
):
    from types import SimpleNamespace
    from unittest.mock import patch

    from sglang.srt.layers.attention.dsv4 import v41_indexer

    with (
        patch.object(v41_indexer, "_use_deep_gemm_prefill", return_value=False),
        patch.object(v41_indexer, "is_sm100_or_newer", return_value=False),
        patch(
            "sglang.srt.runtime_context.get_platform",
            return_value=SimpleNamespace(is_sm90=sm90),
        ),
        envs.SGLANG_OPT_USE_DSV41_SM90_CANDIDATES.override(enabled),
    ):
        prefill, decode = v41_indexer.make_candidate_indexer(
            token_to_kv_pool=SimpleNamespace(),
            req_to_token=torch.empty(3, 128),
            page_size=64,
            candidate_topk_blocks=4,
            candidate_block_size=block_size,
        )
    with envs.SGLANG_OPT_USE_DSV41_SM90_CANDIDATES.override(not enabled):
        assert prefill is decode
        assert prefill.use_triton_candidates == (enabled and sm90 and block_size == 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
def test_candidate_graphs_share_only_readonly_cache_across_streams():
    from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
        fp4_index_logits_candidates,
        fp4_index_logits_paged,
        store_fp4_index_k_cache,
    )

    # A shared model cache with distinct query/candidate buffers for each lane.
    keys = torch.randn(192, 128, dtype=torch.bfloat16, device="cuda")
    cache = torch.zeros(3, 64 * 68, dtype=torch.uint8, device="cuda")
    store_fp4_index_k_cache(keys, cache, torch.arange(192, device="cuda"), page_size=64)
    table = torch.arange(192, dtype=torch.int32, device="cuda").reshape(2, 96)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    graphs, outputs, operands = [], [], []
    for i, stream in enumerate(streams):
        q = torch.randn(2, 32, 128, dtype=torch.bfloat16, device="cuda")
        weights = torch.randn(2, 32, dtype=torch.bfloat16, device="cuda")
        req = torch.tensor([i, 1 - i], dtype=torch.int32, device="cuda")
        lens = torch.tensor([73, 9], dtype=torch.int32, device="cuda")
        blocks = torch.tensor(
            [[8, -1, 0], [0, -1, 1]], dtype=torch.int32, device="cuda"
        )
        args = (q, weights, req, table, lens, cache, 64, 96, 1, blocks)
        operands.append(args)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fp4_index_logits_candidates(*args)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs.append(fp4_index_logits_candidates(*args))
        graphs.append(graph)

    operands[0][0].neg_()
    operands[1][-1].fill_(-1)
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        for stream, graph in zip(streams, graphs):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph.replay()
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    for stream in streams:
        stream.synchronize()
    for args, output in zip(operands, outputs):
        dense = fp4_index_logits_paged(*args[:-1])
        logical = (
            args[-1].long()[:, :, None] * 8 + torch.arange(8, device="cuda")
        ).flatten(1)
        valid = (logical >= 0) & (logical < args[4][:, None])
        expected = dense.gather(1, logical.clamp(0, 95)).masked_fill(~valid, -torch.inf)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
