"""Unit tests for HiSparse-aware max token pool sizing.

Covers the HiSparse host-backed capacity fix:
- `ModelRunner.max_token_pool_size` returns the allocator's `size_full` when
  `enable_hisparse` is set (host-backed logical pool), otherwise it delegates
  to `effective_max_total_num_tokens`.
- `DecodePreallocQueue._check_if_req_exceed_kv_capacity` uses that ratio-expanded
  capacity for admission when HiSparse is enabled, so long-context inputs are
  not truncated at the device-only `max_total_num_tokens`.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_model_runner(**attrs):
    """Build a bare ModelRunner (bypass __init__) so that property descriptors
    like `max_token_pool_size` and `effective_max_total_num_tokens` resolve via
    normal attribute lookup — a plain SimpleNamespace would bypass them and
    raise AttributeError on the internal `self.effective_max_total_num_tokens`
    read inside `max_token_pool_size`."""
    instance = object.__new__(ModelRunner)
    for name, value in attrs.items():
        setattr(instance, name, value)
    return instance


class TestMaxTokenPoolSize(CustomTestCase):
    def test_hisparse_returns_allocator_size_full(self):
        """When HiSparse is enabled and the allocator exposes `size_full`, the
        host-backed logical capacity (device_pool * host_to_device_ratio) wins
        over `effective_max_total_num_tokens`."""
        instance = _make_model_runner(
            enable_hisparse=True,
            token_to_kv_pool_allocator=SimpleNamespace(size_full=4096),
            is_hybrid_swa=False,
            max_total_num_tokens=1024,
            full_max_total_num_tokens=None,
            swa_max_total_num_tokens=None,
        )
        self.assertEqual(instance.max_token_pool_size, 4096)

    def test_hisparse_falls_back_when_size_full_missing(self):
        """HiSparse-enabled but allocator has no `size_full` attribute
        (e.g. non-HiSparse allocator wired at init time). Fall back to the
        SWA-aware effective capacity so we never crash on `AttributeError`."""
        instance = _make_model_runner(
            enable_hisparse=True,
            token_to_kv_pool_allocator=SimpleNamespace(),  # no size_full
            is_hybrid_swa=False,
            max_total_num_tokens=2048,
            full_max_total_num_tokens=None,
            swa_max_total_num_tokens=None,
        )
        self.assertEqual(instance.max_token_pool_size, 2048)

    def test_non_hisparse_uses_effective_max_total_num_tokens(self):
        """Non-HiSparse path is unchanged: delegates to
        `effective_max_total_num_tokens` (which returns `max_total_num_tokens`
        when SWA is not hybrid)."""
        instance = _make_model_runner(
            enable_hisparse=False,
            token_to_kv_pool_allocator=SimpleNamespace(size_full=99999),  # ignored
            is_hybrid_swa=False,
            max_total_num_tokens=1024,
            full_max_total_num_tokens=None,
            swa_max_total_num_tokens=None,
        )
        self.assertEqual(instance.max_token_pool_size, 1024)

    def test_non_hisparse_hybrid_swa_prefers_full_max(self):
        instance = _make_model_runner(
            enable_hisparse=False,
            token_to_kv_pool_allocator=SimpleNamespace(),
            is_hybrid_swa=True,
            max_total_num_tokens=1024,
            full_max_total_num_tokens=3000,
            swa_max_total_num_tokens=500,
        )
        self.assertEqual(instance.max_token_pool_size, 3000)
        self.assertEqual(instance.effective_max_total_num_tokens, 3000)


def _make_prealloc_queue(
    *,
    enable_hisparse: bool,
    max_token_pool_size: int,
    max_total_num_tokens: int,
):
    """Build a minimal DecodePreallocQueue for _check_if_req_exceed_kv_capacity."""
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue.max_total_num_tokens = max_total_num_tokens
    queue.token_to_kv_pool_allocator = SimpleNamespace(size_swa=10**9)
    # Disable the SWA-tail branch; this test only exercises the pool-length gate.
    queue._uses_swa_tail_prealloc = MagicMock(return_value=False)

    model_runner = SimpleNamespace(max_token_pool_size=max_token_pool_size)
    tp_worker = SimpleNamespace(model_runner=model_runner)
    queue.scheduler = SimpleNamespace(
        enable_hisparse=enable_hisparse,
        tp_worker=tp_worker,
        output_streamer=MagicMock(),
    )
    return queue


def _make_req(rid: str, prompt_len: int):
    return SimpleNamespace(
        rid=rid,
        origin_input_ids=[0] * prompt_len,
        output_ids=[],
        return_logprob=False,
        pd_rebootstrap_in_progress=False,
        finished_reason=None,
    )


class TestCheckIfReqExceedKvCapacity(CustomTestCase):
    def test_hisparse_admits_beyond_device_pool_up_to_host_backed_size(self):
        """Core regression: request longer than device-only
        `max_total_num_tokens` but within HiSparse host-backed
        `max_token_pool_size` must NOT be aborted."""
        queue = _make_prealloc_queue(
            enable_hisparse=True,
            max_token_pool_size=4096,  # host-backed logical capacity
            max_total_num_tokens=1024,  # device pool
        )
        req = _make_req("hisparse-long", prompt_len=2048)

        self.assertFalse(queue._check_if_req_exceed_kv_capacity(req))
        queue.scheduler.output_streamer.stream_output.assert_not_called()

    def test_hisparse_rejects_beyond_host_backed_size(self):
        """Requests longer than host-backed capacity are still aborted."""
        queue = _make_prealloc_queue(
            enable_hisparse=True,
            max_token_pool_size=4096,
            max_total_num_tokens=1024,
        )
        req = _make_req("hisparse-too-long", prompt_len=5000)

        self.assertTrue(queue._check_if_req_exceed_kv_capacity(req))
        queue.scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )
        # prepare_abort sets finished_reason to a BAD_REQUEST FINISH_ABORT.
        self.assertIsNotNone(req.finished_reason)

    def test_non_hisparse_uses_device_pool_capacity(self):
        """Non-HiSparse path must keep using `max_total_num_tokens` — the
        HiSparse branch must not bleed into normal decode admission."""
        queue = _make_prealloc_queue(
            enable_hisparse=False,
            max_token_pool_size=4096,  # ignored on non-HiSparse
            max_total_num_tokens=1024,
        )
        req = _make_req("non-hisparse-too-long", prompt_len=2048)

        self.assertTrue(queue._check_if_req_exceed_kv_capacity(req))
        queue.scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )

    def test_rebootstrap_input_len_used_for_capacity(self):
        """Rebootstrap requests carry both prompt and emitted output_ids; the
        admission gate must use the rebootstrap-aware length (prompt + output)
        rather than just the prompt length."""
        queue = _make_prealloc_queue(
            enable_hisparse=True,
            max_token_pool_size=100,
            max_total_num_tokens=100,
        )
        req = SimpleNamespace(
            rid="rebootstrap",
            origin_input_ids=[0] * 60,
            output_ids=[0] * 60,  # 60 + 60 = 120 > 100
            return_logprob=False,
            pd_rebootstrap_in_progress=True,
            finished_reason=None,
        )
        self.assertTrue(queue._check_if_req_exceed_kv_capacity(req))


if __name__ == "__main__":
    unittest.main()
