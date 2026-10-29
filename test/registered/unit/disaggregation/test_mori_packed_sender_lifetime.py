"""CPU-only contract tests for pack-buffer ownership and failure handling.

Load the implementation by path and inject its buffer/engine dependencies.
These tests create no GPU tensors or RDMA connections.
"""

import importlib.util
import sys
import threading
import unittest
from pathlib import Path
from types import MethodType
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "_mori_packed_lifetime_under_test",
    ROOT / "python/sglang/srt/disaggregation/mori/packed.py",
)
packed = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = packed
SPEC.loader.exec_module(packed)


class FakeStatus:
    def __init__(self, pending=False, failed=False):
        self.pending, self.failed = pending, failed

    def InProgress(self):
        return self.pending

    def Failed(self):
        return self.failed


class FakeBuffer:
    def __init__(self, size):
        self.size = size

    def get_ptr(self):
        return 4096

    def get_size(self):
        return self.size


class FakeEngine:
    def __init__(self, outcomes=()):
        self.outcomes = list(outcomes)
        self.submissions, self.deregistered = [], []
        self.uid = 0
        self.registered = object()

    def register_memory(self, *args):
        return self.registered

    def deregister_memory(self, desc):
        self.deregistered.append(desc)

    def allocate_transfer_uid(self):
        self.uid += 1
        return self.uid

    def batch_write(self, *args):
        self.submissions.append(args)
        result = self.outcomes.pop(0) if self.outcomes else FakeStatus()
        if isinstance(result, BaseException):
            raise result
        return [result]


class TestMoriPackedSenderLifetime(unittest.TestCase):
    def setUp(self):
        self.gathers, self.recorded, self.waits = [], [], []
        packed._QUARANTINED_SENDERS.clear()

    def tearDown(self):
        packed._QUARANTINED_SENDERS.clear()

    def build(self, *, outcomes=(), capacity=32, wait=None):
        engine = FakeEngine(outcomes)

        def completion(statuses):
            self.waits.append(list(statuses))
            if wait is not None:
                return wait(statuses)
            return "timeout" if any(s.InProgress() for s in statuses) else None

        sender = packed.MoriPackedKVSender(
            engine=engine,
            gpu_id=0,
            size_bytes=capacity,
            wait_completion=completion,
            record_statuses=self.recorded.extend,
            buffer=FakeBuffer(capacity),
            memory_location="GPU",
            gather=lambda pointers, pages, layout, buffer: self.gathers.append(
                (tuple(pointers), tuple(pages))
            ),
        )
        return sender, engine

    def send(self, sender, *, tensors=1, pages=2):
        return sender.send(
            src_ptrs=list(range(100, 100 + tensors)),
            dst_descs=[object() for _ in range(tensors)],
            src_pages=list(range(3, 3 + pages)),
            dst_pages=list(range(8, 8 + pages)),
            src_page_bytes=32,
            dst_page_bytes=16,
            page_size=4,
            head_offset_bytes=4,
        )

    def test_batches_drain_before_buffer_reuse_and_close_is_idempotent(self):
        sender, engine = self.build(capacity=32)
        statuses = self.send(sender, tensors=2, pages=3)
        self.assertEqual(len(self.gathers), 3)
        self.assertEqual(len(self.waits), 3)
        self.assertEqual(len(statuses), 6)
        self.assertEqual(self.recorded, statuses)
        self.assertTrue(sender.close())
        self.assertTrue(sender.close())
        self.assertEqual(engine.deregistered, [engine.registered])

    def test_timeout_keeps_buffer_registered_and_blocks_repack_until_drain(self):
        pending = FakeStatus(pending=True)
        sender, engine = self.build(outcomes=[pending])
        with self.assertRaisesRegex(RuntimeError, "timeout"):
            self.send(sender)
        self.assertEqual(self.recorded, [pending])
        with self.assertRaisesRegex(RuntimeError, "timeout"):
            self.send(sender)
        self.assertEqual(len(self.gathers), 1)
        self.assertEqual(len(engine.submissions), 1)
        self.assertFalse(sender.close())
        self.assertFalse(engine.deregistered)
        self.assertIn(sender, packed._QUARANTINED_SENDERS)
        pending.pending = False
        self.send(sender)
        self.assertEqual(len(self.gathers), 2)
        self.assertTrue(sender.close())
        self.assertNotIn(sender, packed._QUARANTINED_SENDERS)

    def test_partial_submission_exception_records_prior_status_and_poison(self):
        first = FakeStatus(pending=True)
        sender, engine = self.build(outcomes=[first, RuntimeError("submit failed")])
        with self.assertRaisesRegex(RuntimeError, "submit failed"):
            self.send(sender, tensors=2)
        self.assertEqual(self.recorded[0], first)
        self.assertEqual(len(self.recorded), 2)
        unknown = self.recorded[1]
        self.assertTrue(unknown.InProgress())
        self.assertTrue(unknown.Failed())
        with self.assertRaisesRegex(RuntimeError, "without returning"):
            unknown.Wait()
        self.assertEqual(len(self.gathers), 1)
        self.assertEqual(len(engine.submissions), 2)
        first.pending = False
        with self.assertRaisesRegex(RuntimeError, "quarantined"):
            self.send(sender)
        self.assertEqual(len(self.gathers), 1)
        self.assertFalse(sender.close())
        self.assertFalse(engine.deregistered)

    def test_failed_batch_never_overwrites_buffer_for_next_batch(self):
        failed = FakeStatus(failed=True)
        sender, engine = self.build(outcomes=[failed], capacity=16)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            self.send(sender, pages=3)
        self.assertEqual(len(self.gathers), 1)
        self.assertEqual(self.recorded, [failed])
        self.assertEqual(len(engine.submissions), 1)
        self.assertEqual(sender._inflight, [failed])
        with self.assertRaisesRegex(RuntimeError, "failed submission"):
            self.send(sender)
        self.assertEqual(len(self.gathers), 1)
        self.assertEqual(len(engine.submissions), 1)
        self.assertFalse(sender.close())
        self.assertFalse(engine.deregistered)
        self.assertIn(sender, packed._QUARANTINED_SENDERS)

    def test_close_quarantines_failure_arriving_after_wait_exception(self):
        status = FakeStatus(pending=True)

        def broken_wait(statuses):
            raise TimeoutError("completion callback failed")

        sender, engine = self.build(outcomes=[status], wait=broken_wait)
        with self.assertRaises(TimeoutError):
            self.send(sender)
        status.pending, status.failed = False, True
        self.assertFalse(sender.close())
        self.assertFalse(engine.deregistered)
        with self.assertRaisesRegex(RuntimeError, "failed submission"):
            self.send(sender)
        self.assertEqual(len(self.gathers), 1)
        self.assertIn(sender, packed._QUARANTINED_SENDERS)

    def test_wait_exception_preserves_inflight_statuses(self):
        pending = FakeStatus(pending=True)

        def broken_wait(statuses):
            raise TimeoutError("wait callback failed")

        sender, engine = self.build(outcomes=[pending], wait=broken_wait)
        with self.assertRaisesRegex(TimeoutError, "wait callback"):
            self.send(sender)
        self.assertEqual(self.recorded, [pending])
        with self.assertRaises(TimeoutError):
            self.send(sender)
        self.assertEqual(len(self.gathers), 1)
        self.assertFalse(sender.close())
        self.assertFalse(engine.deregistered)

    def test_capacity_rejected_before_gather_or_network_submission(self):
        sender, engine = self.build(capacity=15)
        with self.assertRaisesRegex(ValueError, "one page"):
            self.send(sender)
        self.assertFalse(self.gathers)
        self.assertFalse(engine.submissions)
        self.assertTrue(sender.close())

    def test_use_after_close_rejected_before_gather(self):
        sender, engine = self.build()
        sender.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.send(sender)
        self.assertFalse(self.gathers)
        self.assertFalse(engine.submissions)

    def test_close_during_gather_retains_registration_until_operation_finishes(self):
        entered, release = threading.Event(), threading.Event()
        sender, engine = self.build()
        failures = []

        def blocked_gather(*args):
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test gather was not released")

        sender._gather_override = blocked_gather

        def worker():
            try:
                self.send(sender)
            except BaseException as exc:
                failures.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(entered.wait(timeout=5))
        self.assertFalse(sender.close())
        self.assertFalse(engine.deregistered)
        release.set()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertFalse(failures)
        self.assertTrue(sender.close())
        self.assertEqual(engine.deregistered, [engine.registered])

    def test_manager_full_and_swa_entrypoints_reach_actual_sender(self):
        # Reuse the AST-loaded real manager methods and geometry fixture, but
        # replace its leaf send stub with the actual registered sender class.
        path = Path(__file__).with_name("test_mori_packed_contract.py")
        spec = importlib.util.spec_from_file_location(
            "_mori_manager_integration_fixture", path
        )
        contract = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = contract
        spec.loader.exec_module(contract)
        fixture = contract.TestMoriPackedContract()
        manager, peer, _ = fixture.manager()
        sender, engine = self.build(capacity=32768)
        manager._packed_local.sender = sender
        manager._try_send_packed = MethodType(contract.try_send, manager)
        manager.kv_args.kv_item_lens = [512, 512]
        manager.kv_args.kv_data_ptrs = [1000, 2000]
        manager.kv_args.kv_layer_ids = [0, 0]
        peer.dst_kv_mem_descs = ["dk", "dv"]
        peer.dst_kv_layer_ids = [0, 0]
        peer.dst_kv_item_len = 256
        manager._get_mha_mem_desc_slices = lambda *args: (
            ["sk"],
            ["sv"],
            ["dk"],
            ["dv"],
            1,
        )
        with patch.dict(sys.modules, {"sglang.srt.disaggregation.mori.packed": packed}):
            full = contract.send_full(manager, peer, [7, 1], [2, 9])
            self.assertEqual(len(full), 2)
            self.assertEqual(self.gathers, [((1000, 2000), (7, 1))])
            self.assertEqual(engine.submissions[0][2], ["dk"])
            self.assertEqual(engine.submissions[0][3], [[512, 2304]])
            manager.kv_args.state_types = ["swa"]
            manager.kv_args.state_data_ptrs = [[3000, 4000]]
            manager.kv_args.mori_pack_layout["swa"] = contract.contract(
                4, page_size=16
            )["full"]
            peer.pack_layout["swa"] = contract.contract(2, page_size=16)["full"]
            swa = contract.send_swa(
                manager,
                peer,
                [1, 3],
                [4, 6],
                ["sk", "sv"],
                [8192, 8192],
                ["swa-dk", "swa-dv"],
                [4096, 4096],
                "swa",
                [1, 1],
                [1, 1],
            )
        self.assertEqual(len(swa), 2)
        self.assertEqual(self.gathers[-1], ((3000, 4000), (1, 3)))
        self.assertEqual(engine.submissions[2][2], ["swa-dk"])
        self.assertEqual(engine.submissions[2][3], [[16384, 24576]])
        self.assertEqual(self.recorded, full + swa)
        self.assertTrue(sender.close())


if __name__ == "__main__":
    unittest.main()
