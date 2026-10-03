"""Real file/controller fixtures: requestless publication and bounded cleanup."""

import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import test_prefetch_finite_io as finite

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.proactive_prefetch import ProactivePrefetch
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


class TestProactivePrefetch(finite.TestFiniteIO):
    def setUp(self):
        super().setUp()
        self.manager = ProactivePrefetch(self.cache)

    def advance(self, predicate):
        def done():
            self.manager.tick()
            return predicate()

        self.pump_until(done)
        self.cache.drain_storage_control_queues()
        self.manager.tick()

    def restore(self, operation_id="p", tokens=None, **kwargs):
        self.manager.submit(operation_id, tokens or list(self.tokens), **kwargs)
        self.advance(lambda: self.manager.status(operation_id)["state"] != "RUNNING")
        return self.manager.status(operation_id)

    def test_requestless_publication_and_accounting(self):
        before = self.allocator.available_size()
        result = self.restore()
        self.assertEqual(result["state"], "SUCCESS")
        self.assertEqual(result["restored_tokens"], 12)
        self.assertGreater(result["restored_bytes"], 0)
        match = self.cache.match_prefix(MatchPrefixParams(key=self.key))
        self.assertEqual(match.host_hit_length, 12)
        self.assertEqual(len(match.device_indices), 0)
        self.assertEqual(self.allocator.available_size(), before)
        self.conservation(self.manager.records["p"].handle, resident=12)

    def test_idempotency_and_conflicting_identity(self):
        result = self.restore()
        with mock.patch.object(
            self.backend, "batch_get", side_effect=AssertionError("duplicate read")
        ):
            self.assertEqual(
                self.manager.submit("p", list(self.tokens))["state"], "SUCCESS"
            )
            self.assertEqual(self.restore("cached")["state"], "CACHED")
        with self.assertRaisesRegex(ValueError, "different restore"):
            self.manager.submit("p", [9] * 12)
        self.assertEqual(result["restored_tokens"], 12)

    def test_nonroot_restore_extends_existing_host_prefix(self):
        self.assertEqual(
            self.restore("first", list(self.tokens[:4]))["restored_tokens"], 4
        )
        self.assertEqual(self.restore("rest")["restored_tokens"], 8)
        self.assertEqual(
            self.cache.match_prefix(MatchPrefixParams(key=self.key)).host_hit_length, 12
        )
        self.conservation(self.manager.records["rest"].handle, resident=12)

    def test_namespace_miss_does_not_publish_or_leak(self):
        result = self.restore(cache_salt="other-session")
        self.assertEqual(result["state"], "MISS")
        self.conservation(self.manager.records["p"].handle)
        self.assertEqual(
            self.cache.match_prefix(MatchPrefixParams(key=self.key)).host_hit_length, 0
        )

    def test_failure_terminal_does_not_leave_control_accounting(self):
        with mock.patch.object(
            self.backend, "batch_get", side_effect=OSError("read failed")
        ):
            self.assertEqual(self.restore()["state"], "FAILURE")
        self.conservation(self.manager.records["p"].handle)
        self.assertEqual(self.restore("next")["state"], "SUCCESS")

    def test_cancel_running_read_keeps_tail_until_ack(self):
        entered, resume = threading.Event(), threading.Event()
        original = self.backend.batch_get

        def read(*args, **kwargs):
            entered.set()
            self.assertTrue(resume.wait(5))
            return original(*args, **kwargs)

        try:
            with mock.patch.object(self.backend, "batch_get", side_effect=read):
                self.manager.submit("p", list(self.tokens))
                self.advance(entered.is_set)
                self.assertTrue(self.manager.cancel("p")["cleanup_pending"])
                with self.assertRaisesRegex(ValueError, "already active"):
                    self.manager.submit("new", list(self.tokens))
                resume.set()
                self.advance(lambda: self.manager.active is None)
        finally:
            resume.set()
        self.assertEqual(self.manager.status("p")["state"], "CANCELLED")
        self.assertEqual(
            self.cache.match_prefix(MatchPrefixParams(key=self.key)).host_hit_length, 0
        )
        self.conservation(self.manager.records["p"].handle)

    def test_expiry_precedes_late_success_publication(self):
        entered, resume = threading.Event(), threading.Event()
        original = self.backend.batch_get

        def read(*args, **kwargs):
            entered.set()
            self.assertTrue(resume.wait(5))
            return original(*args, **kwargs)

        try:
            with mock.patch.object(self.backend, "batch_get", side_effect=read):
                self.manager.submit("p", list(self.tokens))
                self.advance(entered.is_set)
                self.manager.active.deadline = 0
                self.manager.tick()
                self.assertEqual(self.manager.status("p")["state"], "EXPIRED")
                resume.set()
                self.advance(lambda: self.manager.active is None)
        finally:
            resume.set()
        self.conservation(self.manager.records["p"].handle)
        self.assertEqual(
            self.cache.match_prefix(MatchPrefixParams(key=self.key)).host_hit_length, 0
        )

    def test_early_continuation_joins_only_matching_namespace_prefix(self):
        self.manager.submit("p", list(self.tokens))
        req = SimpleNamespace(
            origin_input_ids=list(self.tokens) + [13], extra_key=None, cache_salt=None
        )
        self.assertTrue(self.manager.waits_for(req))
        req.cache_salt = "unrelated"
        self.assertFalse(self.manager.waits_for(req))
        req.cache_salt = None
        req.origin_input_ids[0] = 999
        self.assertFalse(self.manager.waits_for(req))
        self.advance(lambda: self.manager.active is None)
        self.assertFalse(self.manager.waits_for(req))

    def test_join_polling_does_not_rescan_or_join_replacement_request(self):
        self.manager.submit("p", list(self.tokens))
        req = SimpleNamespace(
            origin_input_ids=list(self.tokens) + [13], extra_key=None, cache_salt=None
        )
        with mock.patch.object(
            self.manager, "waits_for", wraps=self.manager.waits_for
        ) as match:
            self.assertTrue(self.manager.blocks(req))
            self.assertTrue(self.manager.blocks(req))
            self.assertEqual(match.call_count, 1)
            replacement = SimpleNamespace(
                origin_input_ids=[999], extra_key=None, cache_salt=None
            )
            self.assertFalse(self.manager.blocks(replacement))
        self.manager.cancel("p")
        self.assertFalse(self.manager.blocks(req))
        self.assertIsNone(self.manager._waiting_req)
        self.advance(lambda: self.manager.active is None)

    def test_control_registry_is_bounded(self):
        self.restore()
        for i in range(50):
            self.assertEqual(self.restore(str(i))["state"], "CACHED")
        self.assertEqual(len(self.manager.records), 32)

    def test_backend_replacement_rejects_new_submit(self):
        self.restore()
        with mock.patch.object(
            self.cache.cache_controller, "storage_backend", object()
        ):
            with self.assertRaisesRegex(ValueError, "backend changed"):
                self.manager.submit("next", list(self.tokens))
        self.assertEqual(self.manager.status("p")["state"], "SUCCESS")

    def test_terminal_join_releases_request_reference(self):
        self.manager.submit("p", list(self.tokens))
        req = SimpleNamespace(
            origin_input_ids=list(self.tokens) + [13], extra_key=None, cache_salt=None
        )
        self.assertTrue(self.manager.blocks(req))
        self.advance(lambda: self.manager.active is None)
        self.assertIsNone(self.manager._waiting_req)

    def test_rejects_buffer_mode_and_non_file_backend(self):
        with mock.patch.object(self.cache, "host_memory_mode", "buffer_only"):
            with self.assertRaisesRegex(ValueError, "resident FULL"):
                ProactivePrefetch(self.cache)
        with mock.patch.object(
            self.cache.cache_controller, "storage_backend", object()
        ):
            with self.assertRaisesRegex(ValueError, "resident FULL"):
                ProactivePrefetch(self.cache)


if __name__ == "__main__":
    unittest.main()
