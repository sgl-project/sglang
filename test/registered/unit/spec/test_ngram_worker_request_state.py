import gc
import unittest
import weakref
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=18, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

import sglang.srt.speculative.ngram_worker as ngram_worker
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.cpp_ngram.ngram_corpus import NgramCorpus
from sglang.srt.speculative.ngram_worker import NGRAMWorker


class TestNgramWorkerRequestState(CustomTestCase):
    def setUp(self):
        self.accelerator_mocks = []
        for name in (
            "record_stream_for_v2_verify",
            "set_time_batch",
            "maybe_detect_nan",
            "maybe_detect_inf",
            "commit_mamba_states_after_verify",
            "move_accept_tokens_to_target_kvcache",
        ):
            self.accelerator_mocks.append(
                self.enterContext(patch.object(ngram_worker, name))
            )
        self.accelerator_mocks.append(
            self.enterContext(
                patch.object(
                    ngram_worker,
                    "eagle_sample",
                    return_value=(
                        torch.tensor([50]),
                        torch.tensor([1]),
                        torch.tensor([0]),
                    ),
                )
            )
        )

    def _make_worker(self, **corpus_options):
        worker = object.__new__(NGRAMWorker)
        worker.device = "cpu"
        worker.enable_overlap = False
        worker.max_trie_depth = 12
        worker.draft_token_num = 6
        worker.speculative_num_draft_tokens = 6
        worker._prev_decode_reqs = {}
        worker.token_to_kv_pool_allocator = None
        worker.model_runner = SimpleNamespace(watermark_state=None)
        worker.ngram_corpus = NgramCorpus(
            max_trie_depth=12,
            min_bfs_breadth=1,
            max_bfs_breadth=8,
            draft_token_num=6,
            capacity=10000,
            **corpus_options,
        )
        worker.last_draft = None

        def prepare(batch):
            if batch.forward_mode.is_decode():
                worker.last_draft, _ = worker._prepare_draft_tokens(batch)
                batch.forward_mode = ForwardMode.TARGET_VERIFY

        # Keep the real forward path and corpus operations; replace GPU work.
        worker._prepare_for_speculative_decoding = prepare
        worker._target_worker = SimpleNamespace(
            forward_batch_generation=lambda *args, **kwargs: SimpleNamespace(
                logits_output=SimpleNamespace(next_token_logits=torch.zeros(1, 64)),
                next_token_ids=torch.tensor([3]),
                can_run_cuda_graph=False,
            )
        )
        return worker

    @staticmethod
    def _make_req(rid, prompt, corpus_ids=None, max_new_tokens=2):
        params = SamplingParams(
            max_new_tokens=max_new_tokens, ngram_corpus_ids=corpus_ids
        )
        params.normalize(None)
        return Req(rid, "", array("q", prompt), params)

    @staticmethod
    def _forward(worker, req, mode, append_output=True):
        batch = SimpleNamespace(
            reqs=[req],
            forward_mode=mode,
            spec_info=None,
            seq_lens=torch.tensor(
                [len(req.origin_input_ids) + len(req.output_ids)], dtype=torch.int64
            ),
            has_grammar=False,
            return_logprob=False,
            grammar_needs_sync=lambda: False,
        )
        result = worker.forward_batch_generation(batch)
        if append_output:
            req.output_ids.extend(result.next_token_ids.tolist())
            req.update_finish_state()

    def _finish_request(self, worker, req):
        self._forward(worker, req, ForwardMode.EXTEND)
        self._forward(worker, req, ForwardMode.DECODE)
        self.assertTrue(req.finished())
        return worker.last_draft.tolist()

    def _worker_with_corpora(self):
        worker = self._make_worker(external_sam_budget=4)
        for name, tail in (("a", [10, 11]), ("b", [20, 21])):
            count = worker.ngram_corpus.load_external_corpus_named(
                name, [[1, 2, 3] + tail]
            )
            worker.ngram_corpus.commit_external_corpus_load(name, count)
        return worker

    def test_completed_rid_can_select_another_corpus(self):
        worker = self._worker_with_corpora()
        first = self._make_req("reused", [1, 2], ["a"])
        second = self._make_req("reused", [1, 2], ["b"])
        self.assertIn(10, self._finish_request(worker, first))
        draft = self._finish_request(worker, second)
        self.assertIn(20, draft)
        self.assertNotIn(10, draft)

    def test_same_request_keeps_state_after_decode_and_retract(self):
        worker = self._worker_with_corpora()
        req = self._make_req("active", [1, 2], ["a"], max_new_tokens=10)
        self._forward(worker, req, ForwardMode.EXTEND)
        self._forward(worker, req, ForwardMode.DECODE)
        state_id = worker.ngram_corpus._req_id_to_state_id[req.rid]
        with patch.object(
            worker.ngram_corpus,
            "erase_match_state",
            wraps=worker.ngram_corpus.erase_match_state,
        ) as erase:
            self._forward(worker, req, ForwardMode.DECODE)
            req.reset_for_retract()
            self._forward(worker, req, ForwardMode.EXTEND)
            self._forward(worker, req, ForwardMode.DECODE)
        erase.assert_not_called()
        self.assertEqual(worker.ngram_corpus._req_id_to_state_id[req.rid], state_id)

    def test_chunked_prefill_preserves_the_active_request_state(self):
        worker = self._worker_with_corpora()
        active = self._make_req("active", [1, 2], ["a"], max_new_tokens=10)
        self._forward(worker, active, ForwardMode.EXTEND)
        self._forward(worker, active, ForwardMode.DECODE)
        state_id = worker.ngram_corpus._req_id_to_state_id[active.rid]
        new = self._make_req("new", [1, 2], ["b"])
        with patch.object(
            worker.ngram_corpus,
            "erase_match_state",
            wraps=worker.ngram_corpus.erase_match_state,
        ) as erase:
            self._forward(worker, new, ForwardMode.EXTEND, append_output=False)
            self._forward(worker, new, ForwardMode.EXTEND, append_output=False)
            self._forward(worker, active, ForwardMode.DECODE)
        erase.assert_not_called()
        self.assertEqual(worker.ngram_corpus._req_id_to_state_id[active.rid], state_id)

    def test_departed_request_state_is_erased(self):
        worker = self._worker_with_corpora()
        first = self._make_req("first", [1, 2], ["a"])
        self._finish_request(worker, first)
        first_ref = weakref.ref(first)
        for accelerator_mock in self.accelerator_mocks:
            accelerator_mock.reset_mock()
        del first
        gc.collect()
        self.assertIsNone(first_ref())
        second = self._make_req("second", [1, 2], ["b"])
        self._finish_request(worker, second)
        self.assertNotIn("first", worker.ngram_corpus._req_id_to_state_id)

    def test_clear_cache_drops_request_state(self):
        worker = self._worker_with_corpora()
        self._finish_request(worker, self._make_req("reused", [1, 2], ["a"]))
        worker.clear_cache_pool()
        self.assertFalse(worker.ngram_corpus._req_id_to_state_id)
        draft = self._finish_request(worker, self._make_req("reused", [1, 2], ["b"]))
        self.assertIn(20, draft)
        self.assertNotIn(10, draft)

    def test_completed_rid_uses_the_new_prompt(self):
        worker = self._make_worker(prompt_sam_budget=4)
        first = self._make_req("reused", [1, 2, 3, 10, 11, 1, 2])
        second = self._make_req("reused", [1, 2, 3, 20, 21, 1, 2])
        self.assertIn(10, self._finish_request(worker, first))
        draft = self._finish_request(worker, second)
        self.assertIn(20, draft)
        self.assertIn(21, draft)


if __name__ == "__main__":
    unittest.main()
