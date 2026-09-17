import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsv4.fp4_indexer import quantize_fp4_indexer_tensor
from sglang.srt.layers.attention.dsv4 import dense_prefill_indexer
from sglang.srt.layers.attention.dsv4.candidate_indexer import candidate_block_mask
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def make_inputs(request_lengths, ratio=1, zero_queries=False):
    torch.manual_seed(17)
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
                    scores = dense_scores(inputs)
                    with patch.object(
                        dense_prefill_indexer, "_SCORE_BUDGET_BYTES", 32768
                    ):
                        selected, candidates = dense_prefill_indexer.dense_prefill_topk(
                            **inputs, publish_candidates=True, candidates=None
                        )
                        self.assert_topk(inputs, selected, scores)
                        row = 0
                        for (queries, context), blocks in zip(
                            inputs["request_lengths"], candidates.request_blocks
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
                                scores[row : row + queries, :context].masked_fill_(
                                    ~candidate_block_mask(
                                        blocks=blocks, width=context, block_size=8
                                    ),
                                    -torch.inf,
                                )
                            row += queries
                        selected, published = dense_prefill_indexer.dense_prefill_topk(
                            **inputs, publish_candidates=False, candidates=candidates
                        )
                        self.assertIsNone(published)
                        self.assert_topk(inputs, selected, scores)
                        tail_lengths = [0, 0, 7, 31]
                        rows, row = [], 0
                        for (queries, _), tail in zip(
                            inputs["request_lengths"], tail_lengths
                        ):
                            rows.extend(range(row + queries - tail, row + queries))
                            row += queries
                        rows = torch.tensor(rows, dtype=torch.int64, device="cuda")
                        tail_inputs = dict(
                            inputs,
                            q=tuple(t[rows] for t in inputs["q"]),
                            weights=inputs["weights"][rows],
                            starts=inputs["starts"][rows],
                            lengths=inputs["lengths"][rows],
                            request_lengths=list(
                                zip(
                                    tail_lengths,
                                    [n for _, n in inputs["request_lengths"]],
                                )
                            ),
                        )
                        selected, _ = dense_prefill_indexer.dense_prefill_topk(
                            **tail_inputs,
                            publish_candidates=False,
                            candidates=candidates.tail(tail_lengths),
                        )
                        self.assert_topk(tail_inputs, selected, scores[rows])

    def test_unfiltered_and_zero_length_requests(self):
        for request_lengths in ([(257, 8192)], [(1, 0), (1, 1), (0, 7)]):
            for publish in (False, True):
                with self.subTest(request_lengths=request_lengths, publish=publish):
                    inputs = make_inputs(request_lengths)
                    selected, candidates = dense_prefill_indexer.dense_prefill_topk(
                        **inputs, publish_candidates=publish, candidates=None
                    )
                    self.assert_topk(inputs, selected, dense_scores(inputs))
                    if publish:
                        self.assertEqual(
                            [b.shape[0] for b in candidates.request_blocks],
                            [q for q, _ in request_lengths],
                        )

    def test_empty_queries(self):
        inputs = make_inputs([(0, 0), (0, 17)])
        selected, candidates = dense_prefill_indexer.dense_prefill_topk(
            **inputs, publish_candidates=True, candidates=None
        )
        self.assertEqual(tuple(selected.shape), (0, 512))
        self.assertEqual(
            [tuple(b.shape) for b in candidates.request_blocks], [(0, 0), (0, 2)]
        )

    def test_score_memory_is_bounded(self):
        for context, limit_gib in ((65536, 3), (65535, 5)):
            with self.subTest(context=context):
                inputs = make_inputs([(16384, context)])
                inputs["candidate_topk_blocks"] = 2048
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                baseline = torch.cuda.memory_allocated()
                selected, candidates = dense_prefill_indexer.dense_prefill_topk(
                    **inputs, publish_candidates=True, candidates=None
                )
                torch.cuda.synchronize()
                self.assertLess(
                    torch.cuda.max_memory_allocated() - baseline, limit_gib << 30
                )
                self.assertEqual(tuple(selected.shape), (16384, 512))
                self.assertEqual(
                    tuple(candidates.request_blocks[0].shape), (16384, 2048)
                )
                del inputs, selected, candidates


if __name__ == "__main__":
    unittest.main()
