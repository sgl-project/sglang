"""Native cache readiness/finalize boundaries: policy is not a PP routing choice."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.pp_cache_test_support import functions, prefetch_wait_type, request
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def cache_fixture(policy, clock, *, async_admission=True):
    cache = NS(
        prefetch_stop_policy=policy,
        prefetch_timeout_base=2.0,
        prefetch_timeout_per_page=0.0,
        page_size=64,
        pp_rank=0,
        ongoing_prefetch={},
        _all_reduce_attn_groups=Mock(),
        _all_reduce=Mock(),
        storage_prefetch_retries=NS(poll_miss=Mock(), cancel=Mock()),
        _storage_prefetch_wait={},
        _finish_storage_prefetch=Mock(),
    )
    names = (
        "check_prefetch_progress",
        "prefetch_ready_to_finalize",
        "finalize_prefetch",
        "_can_terminate_prefetch",
        "_prefetch_stop_allowed",
        "_start_storage_prefetch_wait",
        "_prefetch_deadline_expired",
        "_mark_prefetch_deadline_expired",
    )
    native = functions(
        "mem_cache/unified_radix_cache.py",
        names,
        {
            "rank_consensus": lambda **kw: lambda fn: fn,
            "PPPrefetchDecision": NS(TICKETED=object()),
            "pp_cache_admission_enabled": lambda: async_admission,
            "torch": torch,
            "_StoragePrefetchWait": prefetch_wait_type(clock),
        },
        cls="UnifiedRadixCache",
    )
    for name in names:
        setattr(cache, name, lambda *args, fn=native[name]: fn(cache, *args))
    cache.cache_controller = NS(pp_prefetch_decisions={}, terminate_prefetch=Mock())

    def publish(op):
        cache.ongoing_prefetch.pop(op.handle)

    cache._handle_prefetch_result = Mock(side_effect=publish)
    cache.revoke_pending_prefetch = Mock(
        side_effect=lambda handle: cache.ongoing_prefetch.pop(handle)
    )
    return cache


class ReadinessTests(CustomTestCase):
    def operation(self, cache, rid="r", *, terminated=False, host=True):
        handle = request(rid).cache_request_handle
        op = NS(
            handle=handle,
            start_time=0.0,
            token_ids=list(range(1024)),
            hash_value=["hash"] * 16,
            stats_requested_tokens=1024,
            deadline_expired=False,
            host_indices=list(range(1024)) if host else None,
            completed_tokens=512,
            is_terminated=lambda: terminated,
        )
        cache.ongoing_prefetch[handle] = NS(operation=op)
        return handle, op

    def test_best_effort_allows_stop_without_stopping_any_io(self):
        cache = cache_fixture("best_effort", lambda: 0)
        handle, _ = self.operation(cache)
        self.assertEqual(cache.prefetch_ready_to_finalize([handle]), [True])
        cache.cache_controller.terminate_prefetch.assert_not_called()
        cache._handle_prefetch_result.assert_not_called()
        cache._all_reduce_attn_groups.assert_not_called()

    def test_timeout_starts_at_first_blocked_poll_not_submission(self):
        now = [100.0]
        cache = cache_fixture("timeout", lambda: now[0])
        handle, _ = self.operation(cache)
        self.assertEqual(cache.prefetch_ready_to_finalize([handle]), [False])
        now[0] = 103.0
        self.assertEqual(cache.prefetch_ready_to_finalize([handle]), [True])
        cache.cache_controller.terminate_prefetch.assert_not_called()
        self.assertEqual(cache._all_reduce_attn_groups.call_count, 2)

    def test_wait_complete_empty_operation_and_natural_publication(self):
        cache = cache_fixture("wait_complete", lambda: 100)
        handle, op = self.operation(cache)
        absent = request("no_get").cache_request_handle
        self.assertEqual(
            cache.prefetch_ready_to_finalize([handle, absent]), [False, True]
        )
        cache._handle_prefetch_result(op)  # Native completion ACK publishes first.
        self.assertEqual(
            cache.prefetch_ready_to_finalize([handle, absent]), [True, True]
        )
        cache.finalize_prefetch(handle)
        cache.cache_controller.terminate_prefetch.assert_not_called()
        self.assertEqual(cache._handle_prefetch_result.call_count, 1)

    def test_finalize_publishes_only_ack_safe_prefix_and_is_idempotent(self):
        cache = cache_fixture("timeout", lambda: 0)
        handle, op = self.operation(cache)
        self.assertTrue(cache.finalize_prefetch(handle))
        cache.cache_controller.terminate_prefetch.assert_called_once_with(op)
        cache._handle_prefetch_result.assert_called_once_with(op)
        self.assertEqual(op.completed_tokens, 512)
        self.assertTrue(cache.finalize_prefetch(handle))
        self.assertEqual(cache._handle_prefetch_result.call_count, 1)

    def test_finalize_unallocated_query_uses_native_revoke(self):
        cache = cache_fixture("best_effort", lambda: 0)
        handle, op = self.operation(cache, host=False)
        cache.finalize_prefetch(handle)
        cache.storage_prefetch_retries.poll_miss.assert_called_once_with("r")
        cache.revoke_pending_prefetch.assert_called_once_with(handle)
        cache._handle_prefetch_result.assert_not_called()

    def test_timeout_during_query_closes_accounting_without_retry(self):
        now = [0.0]
        cache = cache_fixture("timeout", lambda: now[0])
        handle, op = self.operation(cache, host=False)
        self.assertEqual(cache.prefetch_ready_to_finalize([handle]), [False])
        now[0] = 3.0
        self.assertEqual(cache.prefetch_ready_to_finalize([handle]), [True])
        cache.finalize_prefetch(handle)
        self.assertTrue(op.deadline_expired)
        self.assertNotIn(handle, cache.ongoing_prefetch)
        cache._finish_storage_prefetch.assert_called_once_with(
            handle, 0, reason="deadline"
        )
        cache.storage_prefetch_retries.cancel.assert_called_once_with("r")
        cache.storage_prefetch_retries.poll_miss.assert_not_called()

    def test_kv_completion_does_not_cut_a_pending_sidecar(self):
        now = [0.0]
        cache = cache_fixture("timeout", lambda: now[0])
        handle, op = self.operation(cache)
        self.assertEqual(cache.prefetch_ready_to_finalize([handle]), [False])
        op.completed_tokens = 1024
        now[0] = 3.0
        self.assertEqual(cache.prefetch_ready_to_finalize([handle]), [False])
        op.is_terminated = lambda: True
        self.assertEqual(cache.prefetch_ready_to_finalize([handle]), [True])
        cache.finalize_prefetch(handle)
        self.assertFalse(op.deadline_expired)
        self.assertNotIn(handle, cache.ongoing_prefetch)
        cache.storage_prefetch_retries.cancel.assert_not_called()

    def test_native_progress_still_checks_policy_before_finalizing(self):
        now = [0.0]
        cache = cache_fixture("timeout", lambda: now[0], async_admission=False)
        handle, _ = self.operation(cache)
        self.assertFalse(cache.check_prefetch_progress(handle))
        cache.cache_controller.terminate_prefetch.assert_not_called()
        now[0] = 3.0
        self.assertTrue(cache.check_prefetch_progress(handle))
        self.assertEqual(cache.cache_controller.terminate_prefetch.call_count, 1)

    def test_native_downstream_uses_pp0_timeout_not_its_local_clock(self):
        clock = Mock(return_value=100)
        cache = cache_fixture("timeout", clock, async_admission=False)
        cache.pp_rank = 2
        handle, op = self.operation(cache)
        cache._start_storage_prefetch_wait(op)
        clock.reset_mock()
        self.assertFalse(cache.check_prefetch_progress(handle))
        clock.assert_not_called()
        cache.cache_controller.terminate_prefetch.assert_not_called()
        cache._all_reduce.side_effect = lambda flags, op: flags.fill_(1)
        self.assertTrue(cache.check_prefetch_progress(handle))
        clock.assert_not_called()
        cache._all_reduce_attn_groups.assert_not_called()

    def test_empty_poll_has_no_collective(self):
        cache = cache_fixture("timeout", lambda: 0)
        self.assertEqual(cache.prefetch_ready_to_finalize([]), [])
        cache._all_reduce_attn_groups.assert_not_called()


if __name__ == "__main__":
    unittest.main()
