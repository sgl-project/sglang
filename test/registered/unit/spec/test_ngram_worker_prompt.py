import unittest
from array import array
from types import SimpleNamespace

import numpy as np
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=18, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.cpp_ngram.ngram_corpus import NgramCorpus
from sglang.srt.speculative.ngram_worker import (
    NGRAM_EXTEND_PROMPT_WINDOW,
    NGRAMWorker,
)


class TestNgramWorkerPrompt(CustomTestCase):
    def test_multimodal_padding_stays_out_of_corpus(self):
        for last_round_pending in [False, True]:
            for multimodal in [False, True]:
                with self.subTest(
                    last_round_pending=last_round_pending, multimodal=multimodal
                ):
                    worker = object.__new__(NGRAMWorker)
                    worker.max_trie_depth = 12
                    worker.draft_token_num = 4
                    worker.ngram_corpus = NgramCorpus(
                        max_trie_depth=12,
                        min_bfs_breadth=1,
                        max_bfs_breadth=1,
                        draft_token_num=4,
                        capacity=1000,
                    )
                    req = Req("prompt", "", array("q", [1, 2, 3]), SamplingParams())
                    if multimodal:
                        # scheduler replaces cache IDs while preserving the original prompt
                        req.origin_input_ids = array("q", [1] + [1 << 60] * 16 + [2, 3])
                    req.output_ids = array("q", [4])
                    # This launch, plus the unprocessed last round when pending.
                    req.inflight_steps = 2 if last_round_pending else 1
                    batch = SimpleNamespace(
                        reqs=[req],
                        spec_info=SimpleNamespace(
                            accept_tokens=torch.tensor([5, 0, 0, 0]),
                            accept_lens=torch.tensor([1]),
                        ),
                    )
                    sequence = [1, 2, 3, 4, 5, 6, 7, 8]
                    worker.ngram_corpus.batch_put([sequence])
                    worker.ngram_corpus.synchronize()
                    reference = NgramCorpus(
                        max_trie_depth=12,
                        min_bfs_breadth=1,
                        max_bfs_breadth=1,
                        draft_token_num=4,
                        capacity=1000,
                    )
                    reference.batch_put([sequence])
                    reference.synchronize()
                    padded = req.origin_input_ids[:]
                    for step in range(2):
                        tokens = (
                            [1, 2, 3]
                            + list(req.output_ids)
                            + ([5] if last_round_pending else [])
                        )
                        expected = reference.batch_get(
                            [req.rid], [tokens], [len(tokens)]
                        )
                        actual = worker._prepare_draft_tokens(batch)
                        np.testing.assert_array_equal(actual[0], expected[0])
                        np.testing.assert_array_equal(actual[1], expected[1])
                        worker._update_ngram_corpus(batch)
                        worker.ngram_corpus.synchronize()
                        inserted = worker.ngram_corpus.batch_get(
                            [f"inserted-{step}"], [[1, 2]], [2]
                        )[0]
                        self.assertIn(3, inserted)
                        req.output_ids.append(6)
                    self.assertEqual(req.origin_input_ids, padded)
                    self.assertEqual(list(req.origin_input_ids_unpadded), [1, 2, 3])

    def test_extend_inserts_unpadded_tail_after_last_chunk(self):
        worker = object.__new__(NGRAMWorker)
        worker.max_trie_depth = 12
        worker.draft_token_num = 4
        worker.ngram_corpus = NgramCorpus(
            max_trie_depth=12,
            min_bfs_breadth=1,
            max_bfs_breadth=1,
            draft_token_num=4,
            capacity=50000,
        )

        # Prompt exceeds NGRAM_EXTEND_PROMPT_WINDOW: early_prefix lies outside
        # the retained window, while prompt_tail + output_ids spans the boundary.
        early_prefix = [10, 11, 12, 13]
        filler = list(range(1000, 1000 + NGRAM_EXTEND_PROMPT_WINDOW - 2))
        prompt_tail = [20, 21]
        unpadded = early_prefix + filler + prompt_tail

        req = Req("mm", "", array("q", unpadded), SamplingParams())
        req.origin_input_ids = array(
            "q", [10, 1 << 60, 11, 12, 13] + filler + prompt_tail
        )
        req.output_ids = array("q", [22, 23, 24])
        other = Req("text", "", array("q", [30, 31, 32, 33]), SamplingParams())

        # Middle chunk: skip batch.chunked_req so only the completed req is inserted.
        worker._insert_extend_prompt_tails(
            SimpleNamespace(reqs=[req, other], chunked_req=req)
        )
        worker.ngram_corpus.synchronize()
        self.assertIn(
            32, worker.ngram_corpus.batch_get(["other-mid"], [[30, 31]], [2])[0]
        )
        self.assertNotIn(
            22, worker.ngram_corpus.batch_get(["req-mid"], [[20, 21]], [2])[0]
        )

        # Final chunk: unpadded prompt tail + output_ids inserted in order, bounded to window.
        worker._insert_extend_prompt_tails(
            SimpleNamespace(reqs=[req, other], chunked_req=None)
        )
        worker.ngram_corpus.synchronize()
        cross_boundary = worker.ngram_corpus.batch_get(
            ["req-boundary"], [[20, 21]], [2]
        )[0]
        self.assertIn(22, cross_boundary)
        self.assertIn(23, cross_boundary)
        self.assertNotIn(
            20, worker.ngram_corpus.batch_get(["req-reversed"], [[23, 24]], [2])[0]
        )
        self.assertNotIn(
            12, worker.ngram_corpus.batch_get(["req-outside"], [[10, 11]], [2])[0]
        )


if __name__ == "__main__":
    unittest.main()
