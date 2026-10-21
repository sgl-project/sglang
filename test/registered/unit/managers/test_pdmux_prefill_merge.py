"""Exercise PDMux's prefill-to-decode handoff with the real batch API."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req, ScheduleBatch
from sglang.srt.multiplex.multiplexing_mixin import SchedulerMultiplexMixin
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _req(rid, *, finished=False, prefix_len=0, extend_end=3):
    req = Req(rid, "", array("q", [1, 2, 3]), SamplingParams(max_new_tokens=8))
    req.prefix_len = prefix_len
    req.extend_end = extend_end
    if finished:
        req.finished_reason = FINISH_LENGTH(1)
    return req


def _batch(reqs, *, slot_start=0):
    indices = torch.arange(slot_start, slot_start + len(reqs), dtype=torch.int64)
    lens = torch.full((len(reqs),), 3, dtype=torch.int64)
    return ScheduleBatch(
        reqs=reqs,
        device="cpu",
        model_config=SimpleNamespace(is_encoder_decoder=False),
        req_pool_indices=indices,
        req_pool_indices_cpu=indices.clone(),
        seq_lens=lens,
        seq_lens_cpu=lens.clone(),
        orig_seq_lens=lens.clone(),
        sampling_info=Mock(),
        split_prefill_finished=True,
        batch_is_full=True,
    )


class TestPDMuxPrefillMerge(unittest.TestCase):
    def _merge(self, batch, running, chunked_req=None):
        scheduler = SimpleNamespace(
            chunked_req=chunked_req,
            process_batch_result=Mock(),
            stash_chunked_request=Mock(),
            running_batch=running,
        )
        result = SchedulerMultiplexMixin._merge_completed_prefill_batch(
            scheduler,
            batch=batch,
            prefill_result=SimpleNamespace(new_seq_lens_cpu=None),
            running_batch=running,
        )
        return scheduler, result

    def test_idle_dp_prefill_preserves_running_batch(self):
        running = _batch([_req("decode")])
        batch = _batch([])
        _, result = self._merge(batch, running)
        self.assertIs(result, running)
        self.assertEqual([req.rid for req in result.reqs], ["decode"])
        self.assertTrue(running.batch_is_full)

    def test_finished_prefill_request_does_not_enter_decode(self):
        running = _batch([])
        batch = _batch([_req("finished", finished=True)])
        _, result = self._merge(batch, running)
        self.assertIs(result, running)
        self.assertTrue(batch.is_empty())
        self.assertFalse(running.batch_is_full)

    def test_mixed_prefill_filters_chunked_and_finished_requests(self):
        chunked = _req("chunked", prefix_len=1, extend_end=2)
        finished = _req("finished", finished=True)
        ready = _req("ready")
        batch = _batch([chunked, finished, ready], slot_start=10)
        batch.chunked_req = chunked
        running = _batch([])
        scheduler, result = self._merge(batch, running, chunked)
        scheduler.stash_chunked_request.assert_called_once_with(chunked)
        self.assertIs(result, batch)
        self.assertEqual(result.reqs, [ready])
        self.assertEqual(result.req_pool_indices.tolist(), [12])
        self.assertEqual(result.req_pool_indices_cpu.tolist(), [12])
        self.assertEqual(result.seq_lens.tolist(), [3])
        self.assertFalse(result.split_prefill_finished)
        self.assertFalse(running.batch_is_full)

    def test_parked_chunk_is_excluded_without_stashing(self):
        chunked = _req("parked", prefix_len=2, extend_end=2)
        batch = _batch([chunked])
        batch.chunked_req = chunked
        running = _batch([])
        scheduler, result = self._merge(batch, running, chunked)
        scheduler.stash_chunked_request.assert_not_called()
        self.assertIs(result, running)
        self.assertTrue(batch.is_empty())

    def test_ready_prefill_merges_into_existing_decode_batch(self):
        decode, ready = _req("decode"), _req("ready")
        running = _batch([decode])
        batch = _batch([ready], slot_start=10)
        _, result = self._merge(batch, running)
        self.assertIs(result, running)
        self.assertEqual(result.reqs, [decode, ready])
        self.assertEqual(result.req_pool_indices.tolist(), [0, 10])
        self.assertEqual(result.seq_lens.tolist(), [3, 3])


if __name__ == "__main__":
    unittest.main()
