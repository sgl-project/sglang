"""SGLANG_DSPARK_DEFER_CONTEXT_KV on a PD decode server: who may defer the draft KV
store, and the prealloc queue writing a pending staging before it hands out slots."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.speculative.dspark_components.dspark_deferred_draft_kv import (
    defer_context_kv_refusal,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _refusal(**overrides) -> str:
    args = dict(
        static_verify=True,
        draft_is_moe=True,
        unified_kv=False,
        unified_memory=False,
        hicache=False,
        disaggregation_mode="null",
        decode_offload=False,
        hisparse=False,
        attn_dp=False,
    )
    args.update(overrides)
    return defer_context_kv_refusal(**args)


class TestDeferredDraftKVRefusal(CustomTestCase):
    def test_agg_and_pd_decode_defer(self):
        self.assertEqual(_refusal(), "")
        self.assertEqual(_refusal(disaggregation_mode="decode"), "")

    def test_pd_prefill_is_refused(self):
        self.assertIn("PD prefill", _refusal(disaggregation_mode="prefill"))

    def test_pd_decode_offload_and_hisparse_are_refused(self):
        self.assertIn(
            "KV offload",
            _refusal(disaggregation_mode="decode", decode_offload=True),
        )
        self.assertIn("hisparse", _refusal(disaggregation_mode="decode", hisparse=True))


class _Worker:
    def __init__(self):
        self.calls = 0

    def flush_deferred_draft_kv_before_slot_reuse(self) -> bool:
        self.calls += 1
        return True


class TestPreallocFlush(CustomTestCase):
    def _queue(self, draft_worker, enable_overlap=False):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        stream_ctx = MagicMock()
        queue.scheduler = SimpleNamespace(
            draft_worker=draft_worker,
            enable_overlap=enable_overlap,
            forward_stream_ctx=stream_ctx,
        )
        return queue, stream_ctx

    def test_the_dspark_worker_is_flushed_on_the_forward_stream(self):
        worker = _Worker()
        queue, stream_ctx = self._queue(worker, enable_overlap=True)
        queue._flush_deferred_draft_kv_before_slot_reuse()
        self.assertEqual(worker.calls, 1)
        stream_ctx.__enter__.assert_called_once()

    def test_no_draft_worker_or_another_worker_is_a_no_op(self):
        for worker in (None, SimpleNamespace()):
            queue, stream_ctx = self._queue(worker, enable_overlap=True)
            queue._flush_deferred_draft_kv_before_slot_reuse()
            stream_ctx.__enter__.assert_not_called()


if __name__ == "__main__":
    unittest.main()
