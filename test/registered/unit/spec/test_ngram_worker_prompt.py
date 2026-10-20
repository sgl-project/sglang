import unittest
from array import array
from types import SimpleNamespace

import numpy as np
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=20, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.cpp_ngram.ngram_corpus import NgramCorpus
from sglang.srt.speculative.ngram_worker import NGRAMWorker


class TestNgramWorkerPrompt(CustomTestCase):
    def test_multimodal_padding_stays_out_of_corpus(self):
        for overlap, grammar in [(False, False), (True, False), (True, True)]:
            for multimodal in [False, True]:
                with self.subTest(
                    overlap=overlap, grammar=grammar, multimodal=multimodal
                ):
                    worker = object.__new__(NGRAMWorker)
                    worker.enable_overlap = overlap
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
                    splice = overlap and not grammar
                    batch = SimpleNamespace(
                        reqs=[req],
                        grammar_needs_sync=lambda: grammar,
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
                            [1, 2, 3] + list(req.output_ids) + ([5] if splice else [])
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


if __name__ == "__main__":
    unittest.main()
