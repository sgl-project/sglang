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
from sglang.srt.mem_cache.utils import storage_namespace_seed
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.cpp_ngram.ngram_corpus import NgramCorpus
from sglang.srt.speculative.ngram_worker import NGRAMWorker


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

    def test_cache_salt_isolates_drafts(self):
        worker = object.__new__(NGRAMWorker)
        worker.max_trie_depth = 18
        worker.draft_token_num = 8
        worker.ngram_corpus = NgramCorpus(
            max_trie_depth=18,
            min_bfs_breadth=1,
            max_bfs_breadth=10,
            draft_token_num=8,
            capacity=10000,
        )
        known = [5001, 5002, 5003, 5004]
        secret = [7101, 7102, 7103, 7104, 7105, 7106]
        req_a = Req(
            "tenant-A",
            "",
            array("q", list(range(100, 140))),
            SamplingParams(),
            cache_salt="tenant-A",
        )
        req_a.output_ids = array("q", known + secret + [9000, 9001])
        req_a.inflight_steps = 1
        batch_a = SimpleNamespace(reqs=[req_a], spec_info=None)
        worker.prev_token_ids = []
        worker.prev_accept_lens = [0]
        worker._update_ngram_corpus(batch_a)
        worker.ngram_corpus.synchronize()

        req_b = Req(
            "tenant-B",
            "",
            array("q", list(range(300, 340))),
            SamplingParams(),
            cache_salt="tenant-B",
        )
        req_b.output_ids = array("q", list(known))
        req_b.inflight_steps = 1
        batch_b = SimpleNamespace(reqs=[req_b], spec_info=None)
        drafts, _ = worker._prepare_draft_tokens(batch_b)
        leaked = [tok for tok in drafts.tolist() if tok in secret]
        self.assertEqual(leaked, [], f"worker leaked tenant A secrets: {drafts}")
        self.assertNotEqual(
            storage_namespace_seed(None, "tenant-A"),
            storage_namespace_seed(None, "tenant-B"),
        )


if __name__ == "__main__":
    unittest.main()
