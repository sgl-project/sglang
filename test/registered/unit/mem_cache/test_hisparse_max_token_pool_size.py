"""Unit tests for HiSparse-aware max token pool sizing.

Covers the HiSparse host-backed capacity fix:
- `ModelRunner.max_token_pool_size` returns the allocator's `size_full` when
  `enable_hisparse` is set (host-backed logical pool), otherwise it delegates
  to `effective_max_total_num_tokens`.
- `DecodePreallocQueue._check_if_req_exceed_kv_capacity` uses that ratio-expanded
  capacity for admission when HiSparse is enabled, so long-context inputs are
  not truncated at the device-only `max_total_num_tokens`.
- `ModelRunner.request_token_capacity` is that logical capacity only on a PD
  decode, and the worker's `max_req_len`/`max_req_input_len` and the
  scheduler's `max_new_tokens` clip follow it, so a PD HiSparse decode neither
  rejects a prompt longer than its device pool nor cuts its output short.
"""

import logging
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.separate_buffer_allocator_double import (
    separate_buffer_allocator_double,
)
from sglang.test.test_utils import CustomTestCase, enter_override

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _make_model_runner(**attrs):
    """Build a bare ModelRunner (bypass __init__) so that property descriptors
    like `max_token_pool_size` and `effective_max_total_num_tokens` resolve via
    normal attribute lookup — a plain SimpleNamespace would bypass them and
    raise AttributeError on the internal `self.effective_max_total_num_tokens`
    read inside `max_token_pool_size`."""
    instance = object.__new__(ModelRunner)
    instance.kv_cache_configurator = object.__new__(KVCacheConfigurator)
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
        with get_context().override_server_args(enable_unified_memory=False):
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
    queue.num_reserved_decode_tokens = 0
    queue.token_to_kv_pool_allocator = separate_buffer_allocator_double(
        page_size=1, size_swa=10**9
    )
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
    def setUp(self):
        super().setUp()
        enter_override(self, get_context().override_server_args())

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


def _hisparse_runner(*, enable_hisparse: bool, device: int, ratio: int):
    runner = _make_model_runner(
        enable_hisparse=enable_hisparse,
        token_to_kv_pool_allocator=SimpleNamespace(size_full=device * ratio),
        is_hybrid_swa=False,
        max_total_num_tokens=device,
        full_max_total_num_tokens=None,
        swa_max_total_num_tokens=None,
        req_to_token_pool=SimpleNamespace(schedulable_token_capacity=lambda c: c),
    )
    runner.kv_cache_configurator.is_hybrid_swa = False
    runner.kv_cache_configurator.is_draft_worker = False
    return runner


class TestRequestTokenCapacity(CustomTestCase):
    def test_pd_hisparse_decode_uses_logical_pool(self):
        runner = _hisparse_runner(enable_hisparse=True, device=104960, ratio=48)
        with get_context().override_server_args(disaggregation_mode="decode"):
            self.assertEqual(runner.request_token_capacity, 104960 * 48)

    def test_aggregated_and_prefill_hisparse_keep_device_pool(self):
        """Outside a PD decode, extends take a device slot per token."""
        runner = _hisparse_runner(enable_hisparse=True, device=104960, ratio=48)
        for mode in ("null", "prefill"):
            with get_context().override_server_args(disaggregation_mode=mode):
                self.assertEqual(runner.request_token_capacity, 104960)

    def test_non_hisparse_decode_keeps_device_pool(self):
        runner = _hisparse_runner(enable_hisparse=False, device=104960, ratio=48)
        with get_context().override_server_args(disaggregation_mode="decode"):
            self.assertEqual(runner.request_token_capacity, 104960)


def _worker(runner, *, context_len: int):
    worker = TpModelWorker.__new__(TpModelWorker)
    worker._model_runner = runner
    worker.model_config = SimpleNamespace(context_len=context_len)
    worker.random_seed = 0
    worker.dllm_algorithm = None
    worker.device = "cpu"
    runner.req_to_token_pool = SimpleNamespace(
        schedulable_token_capacity=lambda capacity: capacity,
        size=1,
        max_context_len=context_len,
    )
    runner.max_running_requests = 22
    runner.forward_stream = None
    runner.token_to_kv_pool = SimpleNamespace(size=runner.max_total_num_tokens)
    return worker


class TestWorkerRequestLengthLimit(CustomTestCase):
    """max_req_len = min(context_len - 1, request-token capacity - 1); the input
    limit is five tokens below it. A 105,000-token HiSparse device pool floors to
    104,960 at page 64, which is where a PD decode's 104,954-token limit came
    from before it used the logical pool."""

    def setUp(self):
        super().setUp()
        dcp = get_parallel().override(attn_dcp_size=1)
        dcp.__enter__()
        self.addCleanup(dcp.__exit__, None, None, None)

    def _limits(self, mode, *, enable_hisparse=True, context_len=1048576):
        runner = _hisparse_runner(
            enable_hisparse=enable_hisparse, device=104960, ratio=48
        )
        worker = _worker(runner, context_len=context_len)
        with get_context().override_server_args(disaggregation_mode=mode):
            info = worker.get_worker_info()
        return info[0], info[4], info[5]

    def test_pd_hisparse_decode_limit_is_logical_or_context(self):
        max_total, max_req_len, max_req_input_len = self._limits("decode")
        self.assertEqual(max_total, 104960)  # scheduler pool stays the device
        self.assertEqual(max_req_len, 1048575)  # context_len binds, not device
        self.assertEqual(max_req_input_len, 1048570)
        _, max_req_len, _ = self._limits("decode", context_len=8 << 20)
        self.assertEqual(max_req_len, 104960 * 48 - 1)

    def test_other_roles_keep_device_limit(self):
        self.assertEqual(self._limits("null"), (104960, 104959, 104954))
        self.assertEqual(
            self._limits("decode", enable_hisparse=False), (104960, 104959, 104954)
        )


class TestSchedulerRequestTokenCapacity(CustomTestCase):
    """The max_new_tokens clip uses scheduler.request_token_capacity."""

    def setUp(self):
        super().setUp()
        dcp = get_parallel().override(attn_dcp_size=1)
        dcp.__enter__()
        self.addCleanup(dcp.__exit__, None, None, None)
        logger = logging.getLogger("sglang.srt.managers.scheduler")
        level = logger.level
        logger.setLevel(logging.ERROR)
        self.addCleanup(logger.setLevel, level)

    def _clip(self, *, request_token_capacity, max_req_len, input_len=117624):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.max_req_len = max_req_len
        scheduler.max_total_num_tokens = 104960
        scheduler.request_token_capacity = request_token_capacity
        scheduler.page_size = 64
        scheduler.kv_shard_widening = 1
        scheduler.max_new_tokens_limit = None
        scheduler.sliding_window_size = None
        scheduler.chunked_prefill_size = None
        allocator = SimpleNamespace(page_size=64)
        allocator.max_new_tokens_for_memory = partial(
            BaseTokenToKVPoolAllocator.max_new_tokens_for_memory, allocator
        )
        scheduler.token_to_kv_pool_allocator = allocator
        req = SimpleNamespace(
            rid="long",
            origin_input_ids=[0] * input_len,
            sampling_params=SimpleNamespace(max_new_tokens=1024, min_new_tokens=0),
        )
        scheduler.init_req_max_new_tokens(req)
        return req.sampling_params.max_new_tokens

    def test_pd_hisparse_decode_keeps_output_beyond_device_pool(self):
        self.assertEqual(
            self._clip(request_token_capacity=104960 * 48, max_req_len=1048575),
            1024,
        )

    def test_device_bound_scheduler_still_clips(self):
        self.assertEqual(
            self._clip(request_token_capacity=104960, max_req_len=104959), 0
        )


class TestRoleSwitchRefreshesRequestLimits(CustomTestCase):
    """A PD role switch recomputes the request limits cached at startup."""

    def setUp(self):
        super().setUp()
        dcp = get_parallel().override(attn_dcp_size=1)
        dcp.__enter__()
        self.addCleanup(dcp.__exit__, None, None, None)

    def _scheduler(self):
        runner = _hisparse_runner(enable_hisparse=True, device=104960, ratio=48)
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_hisparse = True
        scheduler.tp_worker = _worker(runner, context_len=8 << 20)
        scheduler.max_total_num_tokens = 104960
        scheduler.disaggregation_mode = None
        scheduler.beam_coordinator = SimpleNamespace(max_req_len=0)
        return scheduler

    def _limits(self, scheduler):
        return (
            scheduler.max_req_len,
            scheduler.max_req_input_len,
            scheduler.request_token_capacity,
            scheduler.beam_coordinator.max_req_len,
        )

    def test_both_directions(self):
        scheduler = self._scheduler()
        logical = 104960 * 48
        with get_context().override_server_args(disaggregation_mode="decode"):
            scheduler._sync_disaggregation_mode_to_subcomponents()
        self.assertEqual(
            self._limits(scheduler), (logical - 1, logical - 6, logical, logical - 1)
        )
        with get_context().override_server_args(disaggregation_mode="prefill"):
            scheduler._sync_disaggregation_mode_to_subcomponents()
        self.assertEqual(self._limits(scheduler), (104959, 104954, 104960, 104959))


if __name__ == "__main__":
    unittest.main()
