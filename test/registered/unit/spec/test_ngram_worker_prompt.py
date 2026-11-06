import unittest
from array import array
from collections import deque
from types import SimpleNamespace

import numpy as np
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=18, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.cpp_ngram.external_corpus import SEPARATOR_TOKEN as SEP
from sglang.srt.speculative.cpp_ngram.ngram_corpus import NgramCorpus
from sglang.srt.speculative.ngram_worker import NGRAMWorker


def _make_worker(corpus_trie_token_budget=1000):
    worker = object.__new__(NGRAMWorker)
    worker.max_trie_depth = 12
    worker.ngram_corpus = NgramCorpus(
        max_trie_depth=12,
        min_bfs_breadth=1,
        max_bfs_breadth=1,
        draft_token_num=4,
        capacity=1000,
    )
    worker._prev_decode_rids = set()
    worker._corpus_trie_docs = {}
    worker._staged_corpus_trie_docs = {}
    worker._corpus_trie_tokens = 0
    worker._corpus_trie_token_budget = corpus_trie_token_budget
    worker._corpus_trie_backlog = deque()
    return worker


def _first_draft(worker, context):
    """Most likely next token after context, or 0 when the trie has none."""
    while worker._corpus_trie_backlog:
        worker._feed_corpus_trie()
    worker.ngram_corpus.synchronize()
    drafts = worker.ngram_corpus.batch_get(["probe"], [context], [len(context)])[0]
    worker.ngram_corpus.erase_match_state(["probe"])
    return int(drafts[1])


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
        req = Req("mm", "", array("q", [11, 12, 13]), SamplingParams())
        # Pad sentinels overflow the trie's int32 tokens, so batch_put would raise.
        req.origin_input_ids = array("q", [11] + [1 << 60] * 16 + [12, 13])
        other = Req("text", "", array("q", [21, 22, 23]), SamplingParams())
        for chunked in [True, False]:
            with self.subTest(chunked=chunked):
                worker = _make_worker()
                batch = SimpleNamespace(
                    reqs=[req, other], chunked_req=req if chunked else None
                )
                worker._insert_extend_prompt_tails(batch)
                self.assertEqual(_first_draft(worker, [21, 22]), 23)
                # A middle chunk waits for the last one instead of inserting early.
                self.assertEqual(_first_draft(worker, [11, 12]), 0 if chunked else 13)

    def test_external_corpus_trie_follows_corpus_lifecycle(self):
        worker = _make_worker(corpus_trie_token_budget=8)
        # [7, 8, 9] spans a chunk boundary; [32, 33] is past the token budget.
        chunks = [[1, 2, 3, SEP, 7, 8], [9, SEP, 30, 31, 32, 33]]
        worker.commit_corpus_load("a", worker.add_external_corpus("a", chunks))
        self.assertEqual(_first_draft(worker, [7, 8]), 9)
        self.assertEqual(_first_draft(worker, [8, 9]), 0)  # split at the separator
        self.assertEqual(_first_draft(worker, [30, 31]), 0)
        # Over budget: the corpus still loads into its SAM, not into the trie.
        worker.commit_corpus_load("b", worker.add_external_corpus("b", [[40, 41]]))
        self.assertEqual(worker.list_external_corpora(), {"a": 12, "b": 2})
        self.assertEqual(_first_draft(worker, [40]), 0)

        # A flush drops request n-grams but keeps loaded corpora.
        worker.ngram_corpus.batch_put([[50, 51, 52]])
        worker.clear_cache_pool()
        self.assertEqual(_first_draft(worker, [50, 51]), 0)
        self.assertEqual(_first_draft(worker, [1, 2]), 3)
        # A removed corpus is gone from the trie, also after the next flush.
        worker.remove_external_corpus("a")
        worker.clear_cache_pool()
        self.assertEqual(_first_draft(worker, [1, 2]), 0)
        self.assertEqual(worker._corpus_trie_tokens, 0)


if __name__ == "__main__":
    unittest.main()
