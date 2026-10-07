import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
    select_candidate_block_ids,
)
from sglang.kernels.ops.attention.dsv4.fp4_indexer import quantize_fp4_indexer_tensor
from sglang.srt.layers.attention.dsv4.v41_indexer import dense_blocks, scoring
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=150, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def make_inputs(request_lengths, ratio=1, zero_queries=False, seed=17):
    torch.manual_seed(seed)
    rows = sum(q for q, _ in request_lengths)
    q = torch.randn((rows, 32, 128), dtype=torch.bfloat16, device="cuda")
    if zero_queries:
        q.zero_()
    packed, scales = quantize_fp4_indexer_tensor(q.flatten(0, 1), rne=True)
    kv = quantize_fp4_indexer_tensor(
        torch.randn(
            (sum(n for _, n in request_lengths), 128),
            dtype=torch.bfloat16,
            device="cuda",
        ),
        rne=True,
    )
    starts, lengths = [], []
    start = 0
    for queries, context in request_lengths:
        starts.extend([start] * queries)
        lengths.extend(
            (position + 1) // ratio
            for position in range(context * ratio - queries, context * ratio)
        )
        start += context
    return dict(
        q=(packed.view(rows, 32, 64), scales.view(rows, 32)),
        kv=kv,
        weights=torch.rand((rows, 32), dtype=torch.float32, device="cuda"),
        starts=torch.tensor(starts, dtype=torch.int32, device="cuda"),
        lengths=torch.tensor(lengths, dtype=torch.int32, device="cuda"),
        request_lengths=request_lengths,
        topk=512,
        candidate_topk_blocks=2,
        candidate_block_size=8,
    )


def prefill_data(inputs):
    return scoring.DeepGEMMPrefillData(
        k_slots=torch.arange(inputs["kv"][0].shape[0], device="cuda"),
        request_starts=inputs["starts"],
        lens_per_request=[n for _, n in inputs["request_lengths"]],
        rows_per_request=[q for q, _ in inputs["request_lengths"]],
        compress_lens=inputs["lengths"],
        q_fp4=inputs["q"][0],
        q_sf=inputs["q"][1],
        weights=inputs["weights"],
    )


def run_dense(inputs, *, publish_candidates, candidates):
    """The plain top-k, or the CP source's own top-k plus its candidate blocks
    (`publish_candidates`), or a CP consumer's top-k among `candidates`."""
    data = prefill_data(inputs)
    if publish_candidates:
        selected, blocks = dense_blocks._publish_prefill_blocks(
            data=data,
            kv=inputs["kv"],
            topk=inputs["topk"],
            topk_blocks=inputs["candidate_topk_blocks"],
            block_size=inputs["candidate_block_size"],
        )
        return selected, dense_blocks.BlockIds(
            blocks=blocks, rows_per_request=data.rows_per_request
        )
    if candidates is not None:
        selected = dense_blocks._consume_prefill_blocks(
            data=data,
            kv=inputs["kv"],
            topk=inputs["topk"],
            blocks=candidates.blocks,
            block_size=inputs["candidate_block_size"],
        )
        return selected, None
    # The plain top-k writes request-local positions into a -1 filled buffer;
    # the checks compare flattened-K columns, as the CP paths return.
    local = torch.full(
        (data.compress_lens.shape[0], inputs["topk"]),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    scoring.dense_prefill_topk(data, inputs["kv"], out=local)
    starts = data.request_starts[:, None]
    return torch.where(local >= 0, local + starts, local), None


def request_blocks(inputs, candidates):
    """Each request's rows of the published block ids, cut to its kept count
    ``min(topk_blocks, ceil(context / block))``."""
    out, row = [], 0
    block = inputs["candidate_block_size"]
    for queries, context in inputs["request_lengths"]:
        kept = min(inputs["candidate_topk_blocks"], -(-context // block))
        out.append(candidates.blocks[row : row + queries, :kept])
        row += queries
    return out


def dense_scores(inputs):
    from deep_gemm import fp8_fp4_mqa_logits

    width = (max(n for _, n in inputs["request_lengths"]) + 3) // 4 * 4
    scores = fp8_fp4_mqa_logits(
        inputs["q"],
        inputs["kv"],
        inputs["weights"],
        inputs["starts"],
        inputs["starts"] + inputs["lengths"],
        False,
        width,
    )
    return scores.masked_fill_(
        torch.arange(width, device="cuda")[None, :] >= inputs["lengths"][:, None],
        -torch.inf,
    )


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "requires SM100",
)
class TestDensePrefillIndexer(CustomTestCase):
    def assert_topk(self, inputs, selected, scores):
        columns = (selected - inputs["starts"][:, None]).long()
        valid = selected >= 0
        expected_count = torch.isfinite(scores).sum(-1).clamp_max(inputs["topk"])
        torch.testing.assert_close(valid.sum(-1), expected_count)
        self.assertTrue(
            (~valid | ((columns >= 0) & (columns < inputs["lengths"][:, None]))).all()
        )
        actual = scores.gather(1, columns.clamp(0, scores.shape[1] - 1)).masked_fill(
            ~valid, -torch.inf
        )
        expected = scores.topk(min(inputs["topk"], scores.shape[1]), dim=-1).values
        expected = torch.nn.functional.pad(
            expected, (0, inputs["topk"] - expected.shape[1]), value=-torch.inf
        )
        torch.testing.assert_close(
            actual.sort(descending=True).values, expected, rtol=1e-5, atol=1e-5
        )
        ordered = (
            columns.masked_fill(~valid, torch.iinfo(torch.int64).max).sort().values
        )
        self.assertTrue(
            (
                (ordered[:, 1:] != ordered[:, :-1])
                | (ordered[:, 1:] == torch.iinfo(torch.int64).max)
            ).all()
        )

    def test_ragged_source_consumer_and_replay(self):
        for ratio in (1, 2):
            for zero_queries in (False, True):
                with self.subTest(ratio=ratio, zero_queries=zero_queries):
                    inputs = make_inputs(
                        [(0, 0), (1, 1), (33, 511), (257, 4097)],
                        ratio=ratio,
                        zero_queries=zero_queries,
                    )
                    inputs["candidate_topk_blocks"] = 128
                    scores = dense_scores(inputs)
                    consumer_inputs = make_inputs(
                        inputs["request_lengths"],
                        ratio=ratio,
                        zero_queries=zero_queries,
                        seed=29,
                    )
                    consumer_inputs["kv"] = inputs["kv"]
                    consumer_inputs["candidate_topk_blocks"] = 128
                    consumer_scores = dense_scores(consumer_inputs)
                    with patch.object(
                        scoring, "_DEEP_GEMM_SCORE_BUDGET_BYTES", 128 << 10
                    ):
                        selected, candidates = run_dense(
                            inputs, publish_candidates=True, candidates=None
                        )
                        self.assert_topk(inputs, selected, scores)
                        row = 0
                        for (queries, context), blocks in zip(
                            inputs["request_lengths"],
                            request_blocks(inputs, candidates),
                        ):
                            local = scores[row : row + queries, :context]
                            if queries and context:
                                padded = torch.nn.functional.pad(
                                    local, (0, -context % 8), value=-torch.inf
                                )
                                block_scores = padded.unflatten(-1, (-1, 8)).amax(-1)
                                last = (inputs["lengths"][row : row + queries] - 1) // 8
                                block_scores.masked_fill_(
                                    torch.arange(block_scores.shape[1], device="cuda")[
                                        None, :
                                    ]
                                    == last[:, None],
                                    torch.inf,
                                )
                                chosen_scores = block_scores.gather(
                                    1, blocks.long().clamp_min(0)
                                ).masked_fill(blocks < 0, -torch.inf)
                                torch.testing.assert_close(
                                    chosen_scores.sort(descending=True).values,
                                    block_scores.topk(blocks.shape[1]).values,
                                    rtol=1e-5,
                                    atol=1e-5,
                                )
                                columns = torch.arange(context, device="cuda")
                                member = (
                                    columns[None, :, None] // 8 == blocks[:, None, :]
                                ).any(-1)
                                consumer_scores[
                                    row : row + queries, :context
                                ].masked_fill_(
                                    ~member,
                                    -torch.inf,
                                )
                            row += queries
                        self.assertGreater(
                            torch.isfinite(consumer_scores[-1]).sum().item(),
                            inputs["topk"],
                        )
                        self.assertLess(
                            torch.isfinite(consumer_scores[-1]).sum().item(),
                            inputs["lengths"][-1].item(),
                        )
                        selected, published = run_dense(
                            consumer_inputs,
                            publish_candidates=False,
                            candidates=candidates,
                        )
                        self.assertIsNone(published)
                        self.assert_topk(consumer_inputs, selected, consumer_scores)
                        tail_lengths = [0, 0, 7, 31]
                        rows, row = [], 0
                        for (queries, _), tail in zip(
                            inputs["request_lengths"], tail_lengths
                        ):
                            rows.extend(range(row + queries - tail, row + queries))
                            row += queries
                        rows = torch.tensor(rows, dtype=torch.int64, device="cuda")
                        tail_inputs = dict(
                            consumer_inputs,
                            q=tuple(t[rows] for t in consumer_inputs["q"]),
                            weights=consumer_inputs["weights"][rows],
                            starts=inputs["starts"][rows],
                            lengths=inputs["lengths"][rows],
                            request_lengths=list(
                                zip(
                                    tail_lengths,
                                    [n for _, n in inputs["request_lengths"]],
                                )
                            ),
                        )
                        selected, _ = run_dense(
                            tail_inputs,
                            publish_candidates=False,
                            candidates=candidates.tail(tail_lengths),
                        )
                        self.assert_topk(tail_inputs, selected, consumer_scores[rows])

    def test_unfiltered_and_zero_length_requests(self):
        for request_lengths in ([(257, 8192)], [(1, 0), (1, 1), (0, 7)]):
            for publish in (False, True):
                with self.subTest(request_lengths=request_lengths, publish=publish):
                    inputs = make_inputs(request_lengths)
                    selected, candidates = run_dense(
                        inputs, publish_candidates=publish, candidates=None
                    )
                    self.assert_topk(inputs, selected, dense_scores(inputs))
                    if publish:
                        self.assertEqual(
                            [b.shape[0] for b in request_blocks(inputs, candidates)],
                            [q for q, _ in request_lengths],
                        )

    def test_empty_queries_or_context(self):
        # block ids are [rows, min(topk_blocks, ceil(max context / 8))]
        for request_lengths, shape, block_shape in (
            ([], (0, 512), (0, 0)),
            ([(0, 0)], (0, 512), (0, 0)),
            ([(0, 0), (0, 17)], (0, 512), (0, 2)),
            ([(1, 0)], (1, 512), (1, 0)),
        ):
            with self.subTest(request_lengths=request_lengths):
                inputs = make_inputs(request_lengths)
                selected, candidates = run_dense(
                    inputs, publish_candidates=True, candidates=None
                )
                torch.testing.assert_close(
                    selected, torch.full(shape, -1, dtype=torch.int32, device="cuda")
                )
                self.assertEqual(tuple(candidates.blocks.shape), block_shape)

    def test_score_budget_includes_allocation_padding(self):
        from deep_gemm import fp8_fp4_mqa_logits

        inputs = make_inputs([(13, 257)])
        expected = dense_scores(inputs)
        for budget in (32 << 10, 28 << 10, 8 << 10):
            with self.subTest(budget=budget):
                allocations = []

                def checked_logits(*args, **kwargs):
                    before = torch.cuda.memory_allocated()
                    logits = fp8_fp4_mqa_logits(*args, **kwargs)
                    allocations.append(torch.cuda.memory_allocated() - before)
                    return logits

                with (
                    patch.object(scoring, "_DEEP_GEMM_SCORE_BUDGET_BYTES", budget),
                    patch("deep_gemm.fp8_fp4_mqa_logits", new=checked_logits),
                ):
                    selected, _ = run_dense(
                        inputs, publish_candidates=True, candidates=None
                    )
                self.assertTrue(allocations)
                self.assertLessEqual(max(allocations), budget)
                self.assert_topk(inputs, selected, expected)

    def test_score_memory_is_bounded(self):
        for context, limit_gib in ((65536, 3), (65535, 3)):
            with self.subTest(context=context):
                inputs = make_inputs([(16384, context)])
                inputs["candidate_topk_blocks"] = 2048
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                baseline = torch.cuda.memory_allocated()
                selected, candidates = run_dense(
                    inputs, publish_candidates=True, candidates=None
                )
                torch.cuda.synchronize()
                self.assertLess(
                    torch.cuda.max_memory_allocated() - baseline, limit_gib << 30
                )
                self.assertEqual(tuple(selected.shape), (16384, 512))
                self.assertEqual(tuple(candidates.blocks.shape), (16384, 2048))
                del selected
                selected, published = run_dense(
                    inputs, publish_candidates=False, candidates=candidates
                )
                self.assertIsNone(published)
                del selected
                torch.cuda.synchronize()
                baseline = torch.cuda.memory_allocated()
                for _ in range(3):
                    torch.cuda.reset_peak_memory_stats()
                    selected, published = run_dense(
                        inputs, publish_candidates=False, candidates=candidates
                    )
                    torch.cuda.synchronize()
                    self.assertIsNone(published)
                    self.assertLess(
                        torch.cuda.max_memory_allocated() - baseline, 4 << 30
                    )
                    self.assertEqual(tuple(selected.shape), (16384, 512))
                    del selected
                    torch.cuda.synchronize()
                    self.assertEqual(torch.cuda.memory_allocated(), baseline)
                del inputs, candidates


def tiled_logits(inputs, budget):
    data = prefill_data(inputs)
    parts = []
    with patch.object(scoring, "_DEEP_GEMM_SCORE_BUDGET_BYTES", budget):
        for tile, logits in scoring.score_tiles(data, inputs["kv"], width_align=8):
            parts.append(logits.clone())
            del logits
    return torch.cat(parts), len(parts)


def same_rows(a, b):
    """Row-wise equal ignoring order: the selections are unordered."""
    return torch.equal(a.sort(dim=-1).values, b.sort(dim=-1).values)


def same_selected_scores(inputs, scores, a, b):
    """Row-wise equal selected scores: selections may break ties differently."""

    def picked(selected):
        cols = (selected - inputs["starts"][:, None]).clamp_min(0).long()
        values = scores.gather(1, cols).masked_fill(selected < 0, -torch.inf)
        return values.sort(dim=-1).values

    return torch.equal(picked(a), picked(b))


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] in (10, 12),
    "requires SM100 or SM120",
)
class TestPrefillScoreTileBudget(CustomTestCase):
    request_lengths = [(3, 4093), (2045, 485_123)]

    def test_tiles_score_bitwise_like_one_tile(self):
        inputs = make_inputs(self.request_lengths)
        whole, tiles = tiled_logits(inputs, 1 << 40)
        self.assertEqual(tiles, 1)
        split, tiles = tiled_logits(inputs, 256 << 20)
        self.assertGreater(tiles, 8)
        valid = (
            torch.arange(whole.shape[1], device="cuda")[None, :]
            < inputs["lengths"][:, None]
        )
        self.assertTrue(torch.equal(whole[valid], split[valid]))

    def test_selection_matches_one_tile(self):
        inputs = make_inputs(self.request_lengths)
        inputs["candidate_topk_blocks"] = 2048
        consumer = make_inputs(self.request_lengths, seed=29)
        consumer["kv"] = inputs["kv"]
        results = []
        for budget in (1 << 40, 256 << 20):
            with patch.object(scoring, "_DEEP_GEMM_SCORE_BUDGET_BYTES", budget):
                selected, candidates = run_dense(
                    inputs, publish_candidates=True, candidates=None
                )
                consumed, _ = run_dense(
                    consumer, publish_candidates=False, candidates=candidates
                )
                full, _ = run_dense(inputs, publish_candidates=False, candidates=None)
            results.append((selected, candidates.blocks, consumed, full))
        scores = dense_scores(inputs)
        consumer_scores = dense_scores(consumer)
        (selected, blocks, consumed, full), tiled = results
        self.assertTrue(same_rows(blocks, tiled[1]))
        self.assertTrue(same_selected_scores(inputs, scores, selected, tiled[0]))
        self.assertTrue(
            same_selected_scores(consumer, consumer_scores, consumed, tiled[2])
        )
        self.assertTrue(same_selected_scores(inputs, scores, full, tiled[3]))
        row = 0
        for (queries, context), blocks in zip(
            self.request_lengths, request_blocks(inputs, candidates)
        ):
            reference_blocks = select_candidate_block_ids(
                logits=scores[row : row + queries, :context],
                compress_lens=inputs["lengths"][row : row + queries, None],
                topk_blocks=inputs["candidate_topk_blocks"],
                block_size=inputs["candidate_block_size"],
            )
            self.assertTrue(
                same_rows(blocks[:, : reference_blocks.shape[1]], reference_blocks)
            )
            row += queries

    def test_tile_scratch_within_declared_bytes(self):
        """Requested peak bytes, excluding outputs, fit the declared tile
        allocation plus a 1 KiB allowance."""
        inputs = make_inputs([(3, 4093), (9, 485_123)])
        inputs["candidate_topk_blocks"] = 2048
        kv, block = inputs["kv"], inputs["candidate_block_size"]
        topk_blocks = inputs["candidate_topk_blocks"]
        data = prefill_data(inputs)
        rows = data.num_rows
        group = 128 // inputs["q"][0].shape[1]
        width = -(-485_123 // 8) * 8
        logits = group * 4 * (-(-width // 256) * 256)

        def peak_past(baseline, held):
            torch.cuda.synchronize()
            stats = torch.cuda.memory_stats()
            return stats["requested_bytes.all.peak"] - baseline - held

        def start():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            return torch.cuda.memory_stats()["requested_bytes.all.current"]

        # one tile's row-group-sized offsets; DeepGEMM allocates only the logits
        slack = 1 << 10
        # A budget of one byte leaves every tile at the one-row-group floor.
        with patch.object(scoring, "_DEEP_GEMM_SCORE_BUDGET_BYTES", 1):
            row, tile = dense_blocks._publish_scratch_bytes(width, block, topk_blocks)
            baseline = start()
            selected, blocks = dense_blocks._publish_prefill_blocks(
                data=data,
                kv=kv,
                topk=inputs["topk"],
                topk_blocks=topk_blocks,
                block_size=block,
            )
            held = 4 * (selected.numel() + blocks.numel())
            with self.subTest(path="publish"):
                self.assertLessEqual(
                    peak_past(baseline, held), logits + group * row + tile + slack
                )
            del selected
            candidates = blocks
            for topk in (inputs["topk"], 16):
                row = dense_blocks._consume_scratch_bytes(topk_blocks, block, topk)
                baseline = start()
                selected = dense_blocks._consume_prefill_blocks(
                    data=data, kv=kv, topk=topk, blocks=candidates, block_size=block
                )
                with self.subTest(path="consume", topk=topk):
                    self.assertLessEqual(
                        peak_past(baseline, 4 * selected.numel()),
                        logits + group * row + slack,
                    )
                del selected
            out = torch.full(
                (rows, inputs["topk"]), -1, dtype=torch.int32, device="cuda"
            )
            baseline = start()
            scoring.dense_prefill_topk(data, kv, out=out)
            with self.subTest(path="full"):
                self.assertLessEqual(peak_past(baseline, 0), logits + slack)


if __name__ == "__main__":
    unittest.main()
