import unittest
from unittest.mock import Mock, patch

import torch

from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
    candidate_topk,
    topk_among_blocks,
)
from sglang.kernels.ops.attention.dsv4.candidate_scores import (
    _candidate_matmul,
    can_use_candidate_scores,
    candidate_scores,
)
from sglang.kernels.ops.attention.dsv4.candidate_table import CANDIDATE_BLOCK_SIZE
from sglang.kernels.ops.quantization.fp4_fake_quant import fake_quant_fp4
from sglang.kernels.ops.quantization.fp4_indexer_quant import (
    quantize_fp4_indexer_tensor,
)
from sglang.srt.layers.attention.dsv4.v41_indexer import dense_blocks, scoring
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


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


def run_dense(inputs, *, publish_candidates, candidates):
    """The plain top-k, or the CP source's own top-k plus its candidate blocks
    (`publish_candidates`), or a CP consumer's top-k among `candidates`."""
    data = scoring.DeepGEMMPrefillData(
        k_slots=torch.arange(inputs["kv"][0].shape[0], device="cuda"),
        request_starts=inputs["starts"],
        lens_per_request=[n for _, n in inputs["request_lengths"]],
        rows_per_request=[q for q, _ in inputs["request_lengths"]],
        compress_lens=inputs["lengths"],
        q_fp4=inputs["q"][0],
        q_sf=inputs["q"][1],
        weights=inputs["weights"],
    )
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

    from sglang.kernels.ops.attention.dsv4.index_logits import (
        index_logits_dtype,
        index_weights_dtype,
    )

    width = (max(n for _, n in inputs["request_lengths"]) + 3) // 4 * 4
    # Score in the dtypes the indexer uses on this GPU, so the reference matches.
    scores = fp8_fp4_mqa_logits(
        inputs["q"],
        inputs["kv"],
        inputs["weights"].to(index_weights_dtype()),
        inputs["starts"],
        inputs["starts"] + inputs["lengths"],
        False,
        width,
        logits_dtype=index_logits_dtype(),
    ).float()
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
        for context, limit_gib in ((65536, 3), (65535, 5)):
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


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is None, "requires NVIDIA CUDA"
)
class TestCandidateScores(CustomTestCase):
    def make_case(self, heads=32, visible=129, *, fp4=False, seed=17):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        padded = -(-visible // 64) * 64
        if fp4:
            q = fake_quant_fp4(
                torch.randn(
                    5, heads, 128, device="cuda", dtype=torch.bfloat16, generator=gen
                )
            )
            k = fake_quant_fp4(
                torch.randn(
                    padded, 128, device="cuda", dtype=torch.bfloat16, generator=gen
                )
            )
        else:
            q = torch.randint(
                -2, 3, (5, heads, 128), device="cuda", generator=gen
            ).bfloat16()
            k = torch.randint(
                -2, 3, (padded, 128), device="cuda", generator=gen
            ).bfloat16()
        k[visible:] = 1024
        weights = (
            torch.randint(-2, 3, (5, heads), device="cuda", generator=gen).bfloat16()
            / 4
        )
        newest = (visible - 1) // CANDIDATE_BLOCK_SIZE
        blocks = torch.tensor(
            [
                [-1, -1, -1],
                [0, -1, -1],
                [0, 1, -1],
                [0, 4, newest],
                [newest, max(0, newest - 1), -1],
            ],
            device="cuda",
            dtype=torch.int32,
        )
        lens = torch.tensor(
            [0, min(1, visible), min(11, visible), visible, visible],
            device="cuda",
            dtype=torch.int32,
        )
        return q, k, weights, blocks, lens

    def compact_reference(self, q, k, weights, blocks, lens, visible):
        per_head = torch.einsum("bhd,nd->bhn", q, k)
        full = (per_head.relu() * weights.unsqueeze(-1)).sum(1).float()[:, :visible]
        positions = (
            blocks.long()[:, :, None] * CANDIDATE_BLOCK_SIZE
            + torch.arange(CANDIDATE_BLOCK_SIZE, device="cuda")
        ).flatten(1)
        valid = (positions >= 0) & (positions < lens[:, None]) & (positions < visible)
        if visible:
            compact = full.gather(1, positions.clamp(0, visible - 1))
            compact = compact.masked_fill(~valid, -torch.inf)
        else:
            compact = torch.full(positions.shape, -torch.inf, device="cuda")
        return full, compact

    def check_pipeline(self, inputs, visible, topk):
        q, k, weights, blocks, lens = inputs
        full, expected_scores = self.compact_reference(*inputs, visible)
        with patch.object(_candidate_matmul, "run", wraps=_candidate_matmul.run) as run:
            compact = candidate_scores(q, k, weights, blocks, lens)
        self.assertEqual(run.call_count, int(q.shape[0] > 0 and blocks.shape[1] > 0))
        self.assertEqual(compact.dtype, torch.float32)
        self.assertTrue(torch.equal(compact, expected_scores))
        actual = candidate_topk(compact, blocks, topk)
        expected = topk_among_blocks(full, lens, blocks, topk, CANDIDATE_BLOCK_SIZE)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.shape, (q.shape[0], topk))
        self.assertTrue(
            torch.equal(
                (actual >= 0).sum(1), torch.isfinite(compact).sum(1).clamp_max(topk)
            )
        )
        return compact

    def test_candidate_pipeline_parity(self):
        for heads in (16, 32, 64):
            for visible in (129, 1025):
                inputs = self.make_case(heads, visible)
                for topk in (0, 7, 32):
                    with self.subTest(heads=heads, visible=visible, topk=topk):
                        self.check_pipeline(inputs, visible, topk)

    def test_small_heads(self):
        for heads in (1, 2, 4, 8, 9):
            with self.subTest(heads=heads):
                self.check_pipeline(self.make_case(heads), 129, 32)

    def test_fp4_operands(self):
        for visible in (129, 1025):
            for seed in (0, 42, 123):
                with self.subTest(visible=visible, seed=seed):
                    self.check_pipeline(
                        self.make_case(visible=visible, fp4=True, seed=seed), visible, 7
                    )

    def test_noncontiguous_operands(self):
        inputs = []
        for value in self.make_case():
            if value.ndim == 1:
                inputs.append(value)
            else:
                backing = torch.empty(
                    (*value.shape[:-1], value.shape[-1] * 2),
                    device=value.device,
                    dtype=value.dtype,
                )
                view = backing[..., ::2]
                view.copy_(value)
                self.assertFalse(view.is_contiguous())
                inputs.append(view)
        self.check_pipeline(tuple(inputs), 129, 32)

    def test_empty_inputs(self):
        inputs = self.make_case()
        q, k, weights, blocks, lens = inputs
        cases = [
            (q[:0], k, weights[:0], blocks[:0], lens[:0]),
            (q, k, weights, blocks[:, :0], lens),
            (q, k[:0], weights, blocks, torch.zeros_like(lens)),
        ]
        for case in cases:
            visible = 129 if case[1].shape[0] else 0
            with self.subTest(
                rows=case[0].shape[0], blocks=case[3].shape[1], visible=visible
            ):
                self.check_pipeline(case, visible, 32)

    def test_eligibility(self):
        q, k, weights, _, _ = self.make_case()
        self.assertTrue(can_use_candidate_scores(q, k, weights, 24, 129))
        self.assertFalse(can_use_candidate_scores(q, k, weights, 24, 47))
        self.assertFalse(can_use_candidate_scores(q, k, weights, 0, 129))
        self.assertFalse(can_use_candidate_scores(q.float(), k, weights, 24, 129))
        self.assertFalse(
            can_use_candidate_scores(q.cpu(), k.cpu(), weights.cpu(), 24, 129)
        )
        with patch(
            "sglang.kernels.ops.attention.dsv4.candidate_scores.is_cuda",
            return_value=False,
        ):
            self.assertFalse(can_use_candidate_scores(q, k, weights, 24, 129))

    def test_candidate_chunking(self):
        inputs = self.make_case()
        q, k, weights, blocks, lens = inputs
        full, expected = self.compact_reference(*inputs, 129)
        tok = torch.tensor([4, 0, 3, 1, 2], device="cuda")
        request = scoring.RequestScores(
            lc=129,
            k=7,
            columns=torch.arange(129, device="cuda"),
            tok=tok,
            lens=lens[tok],
            slots=torch.arange(129, device="cuda"),
        )
        indexer = Mock()
        indexer.scores.side_effect = AssertionError("full scoring was used")
        budget = 2 * q.shape[1] * blocks.shape[1] * CANDIDATE_BLOCK_SIZE * 2
        with patch.object(scoring, "_TORCH_SCORE_BUDGET_BYTES", budget):
            chunks = list(
                scoring._score_chunks(
                    indexer=indexer,
                    q=q,
                    weights=weights,
                    index_k=k,
                    request=request,
                    candidate_blocks=blocks,
                )
            )
        self.assertEqual([chunk.tok.numel() for chunk in chunks], [2, 2, 1])
        for chunk in chunks:
            self.assertTrue(chunk.candidate_only)
            self.assertTrue(torch.equal(chunk.scores, expected[chunk.tok]))
            actual = candidate_topk(chunk.scores, blocks[chunk.tok], request.k)
            reference = topk_among_blocks(
                full[chunk.tok],
                lens[chunk.tok],
                blocks[chunk.tok],
                request.k,
                CANDIDATE_BLOCK_SIZE,
            )
            self.assertTrue(torch.equal(actual, reference))

    def test_full_score_fallback(self):
        q, k, weights, blocks, lens = self.make_case()
        request = scoring.RequestScores(
            lc=129,
            k=7,
            columns=torch.arange(129, device="cuda"),
            tok=torch.arange(5, device="cuda"),
            lens=lens,
            slots=torch.arange(129, device="cuda"),
        )

        def full_scores(q, k, weights):
            s = torch.einsum("bhd,nd->bhn", q, k)
            return (s.relu() * weights.unsqueeze(-1)).sum(1).float()

        indexer = Mock()
        indexer.scores.side_effect = full_scores
        too_wide = blocks.repeat(1, 4)
        with patch.object(
            _candidate_matmul, "run", side_effect=AssertionError("candidate launch")
        ):
            chunks = list(
                scoring._score_chunks(
                    indexer=indexer,
                    q=q,
                    weights=weights,
                    index_k=k,
                    request=request,
                    candidate_blocks=too_wide,
                )
            )
        self.assertTrue(indexer.scores.called)
        expected = full_scores(q, k, weights)[:, :129]
        expected = expected.masked_fill(
            torch.arange(129, device="cuda")[None, :] >= lens[:, None], -torch.inf
        )
        for chunk in chunks:
            self.assertFalse(chunk.candidate_only)
            self.assertTrue(torch.equal(chunk.scores, expected[chunk.tok]))


if __name__ == "__main__":
    unittest.main()
