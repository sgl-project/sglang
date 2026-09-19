"""CPU tests for accounting across lost connections, fanout and lease expiry."""

import asyncio
import importlib.util
import sys
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

# This module has no engine dependencies. Keep these failure/recovery tests
# runnable on control-plane hosts without installing torch or GPU kernels.
_path = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/managers/request_lifecycle.py"
)
_spec = importlib.util.spec_from_file_location("request_lifecycle", _path)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
RequestLifecycle = _module.RequestLifecycle
SchedulerLifecycle = _module.SchedulerLifecycle


class TestRequestLifecycle(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = 0.0
        self.registry = RequestLifecycle(clock=lambda: self.now, retention_seconds=10)
        self.attempt = uuid.uuid4().hex
        self.registry.claim(self.attempt, "decode", lease_seconds=5)

    def child(self, rid="client", kind="sample"):
        child = self.registry.add_child(self.attempt, rid, kind)
        self.registry.dispatched(child)
        return child

    def test_warmup_completion_does_not_complete_future_samples(self):
        warmup = self.child(kind="warmup")
        self.registry.scheduler_event(warmup, 1, "terminal")
        self.assertFalse(self.registry.snapshot(self.attempt)["terminal"])
        children = [self.child("client-1"), self.child("client-10")]
        self.registry.seal(self.attempt)
        self.assertEqual(self.registry.cancel(self.attempt), children)
        # Abort dispatch / HTTP EOF is not an acknowledgement.
        self.assertFalse(self.registry.snapshot(self.attempt)["terminal"])
        for child in children:
            self.registry.scheduler_event(child, 1, "terminal")
        self.assertTrue(self.registry.snapshot(self.attempt)["terminal"])

    async def test_reconnect_snapshot_and_duplicate_or_reordered_events(self):
        child = self.child()
        self.registry.seal(self.attempt)
        before = self.registry.snapshot(self.attempt)["version"]
        waiter = asyncio.create_task(self.registry.wait(self.attempt, before))
        await asyncio.sleep(0)
        self.registry.scheduler_event(child, 3, "prefill")
        snapshot = await waiter
        self.assertTrue(snapshot["children"][0]["prefill_complete"])
        self.registry.scheduler_event(child, 3, "terminal")
        terminal = self.registry.snapshot(self.attempt)
        self.registry.scheduler_event(child, 3, "prefill")
        self.registry.scheduler_event(child, 3, "terminal")
        self.assertEqual(self.registry.snapshot(self.attempt), terminal)
        recovered = await self.registry.wait(self.attempt, before)
        self.assertTrue(recovered["terminal"])
        self.assertEqual(recovered["incarnation"], self.registry.incarnation)

    def test_expiry_cancels_but_never_frees_unacknowledged_work(self):
        child = self.child()
        self.registry.seal(self.attempt)
        self.now = 6
        self.assertEqual(self.registry.expired(), [self.attempt])
        self.assertEqual(self.registry.cancel(self.attempt), [child])
        with self.assertRaises(ValueError):
            self.registry.renew(self.attempt)
        with self.assertRaises(ValueError):
            self.registry.discard(child)
        self.now = 100
        self.registry.prune()
        self.assertFalse(self.registry.snapshot(self.attempt)["terminal"])
        self.registry.scheduler_event(child, 0, "terminal")
        self.now = 111
        self.registry.prune()
        with self.assertRaises(KeyError):
            self.registry.snapshot(self.attempt)
        self.registry.scheduler_event(child, 0, "terminal")

    def test_cancel_during_tokenization_prevents_late_dispatch_and_fanout(self):
        child = self.registry.add_child(self.attempt, "client")
        self.registry.cancel(self.attempt)
        with self.assertRaises(ValueError):
            self.registry.dispatched(child)
        with self.assertRaises(ValueError):
            self.registry.add_child(self.attempt, "new-sample")
        self.registry.discard(child)
        self.registry.seal(self.attempt)
        self.assertTrue(self.registry.snapshot(self.attempt)["terminal"])

    def test_identity_capacity_and_rank_fences(self):
        with self.assertRaises(ValueError):
            self.registry.claim(self.attempt, "decode")
        child = self.child()
        self.registry.scheduler_event(child, 0, "prefill")
        with self.assertRaises(ValueError):
            self.registry.scheduler_event(child, 1, "terminal")
        other = uuid.uuid4().hex
        self.registry.claim(other, "prefill")
        # Reusing a client rid in another attempt never aliases child identity.
        self.assertNotEqual(self.registry.add_child(other, "client"), child)
        registry = RequestLifecycle(max_attempts=1, max_children=1)
        registry.claim(self.attempt, "decode")
        with self.assertRaises(ValueError):
            registry.claim(other, "decode")
        registry.add_child(self.attempt, "a")
        with self.assertRaises(ValueError):
            registry.add_child(self.attempt, "b")

    def test_batch_child_creation_is_atomic(self):
        registry = RequestLifecycle(max_total_children=1)
        registry.claim(self.attempt, "decode")
        with self.assertRaises(ValueError):
            registry.add_children(self.attempt, ["a", "b"])
        self.assertEqual(registry.snapshot(self.attempt)["children"], [])

    def test_acknowledgement_reclaims_capacity_without_allowing_replay(self):
        registry = RequestLifecycle(max_attempts=1, max_total_children=1)
        registry.claim(self.attempt, "decode")
        child = registry.add_child(self.attempt, "rid")
        registry.dispatched(child)
        registry.seal(self.attempt)
        with self.assertRaises(ValueError):
            registry.acknowledge(self.attempt)
        registry.scheduler_event(child, 0, "terminal")
        registry.acknowledge(self.attempt)
        registry.acknowledge(self.attempt)
        with self.assertRaises(ValueError):
            registry.claim(self.attempt, "decode")
        other = uuid.uuid4().hex
        registry.claim(other, "decode")
        registry.add_child(other, "rid")


class TestSchedulerLifecycle(unittest.TestCase):
    def test_unary_prefill_and_deferred_cleanup_without_any_response_output(self):
        events = []
        tracker = SchedulerLifecycle(
            lambda req, phase: events.append((req.lifecycle_id, phase)), interval=0
        )
        req = SimpleNamespace(
            lifecycle_id="child",
            time_stats=SimpleNamespace(prefill_finished_time=0),
            kv=SimpleNamespace(holds_kv=True, holds_mamba=False),
            inflight_middle_chunks=0,
            metadata_buffer_index=-1,
            finished=lambda: False,
        )
        tracker.register(req)
        tracker.poll()
        self.assertEqual(events, [])
        req.time_stats.prefill_finished_time = 1
        tracker.poll()
        self.assertEqual(events, [("child", "prefill")])
        tracker.retire(req)
        tracker.poll()
        self.assertEqual(len(events), 1)  # Abort output precedes actual cleanup.
        req.kv.holds_kv = False
        req.metadata_buffer_index = 4
        tracker.poll()
        self.assertEqual(len(events), 1)  # P/D metadata still owned.
        req.metadata_buffer_index = -1
        req.inflight_middle_chunks = 1
        tracker.poll()
        self.assertEqual(len(events), 1)
        req.inflight_middle_chunks = 0
        tracker.poll()
        tracker.poll()
        self.assertEqual(events, [("child", "prefill"), ("child", "terminal")])


if __name__ == "__main__":
    unittest.main()
