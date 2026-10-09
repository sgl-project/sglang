"""Teacher batching preserves accepted rows through reuse and partial flushes."""

import unittest
from unittest.mock import patch

import torch
from sglang.srt.training_capture.context import RequestCaptureContext
from sglang.srt.training_capture.host_pool import HostBufferPool
from sglang.srt.training_capture.protocol import ContractError
from sglang.srt.training_capture.teacher import capture_teacher
from sglang.srt.training_capture.teacher_staging import TeacherStaging
from sglang.srt.training_capture.topology import plan_capture_layout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.training_capture_utils import Registrar, make_kv_spec

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestTeacherStaging(CustomTestCase):
    def setUp(self):
        self.pool = HostBufferPool(
            kv=make_kv_spec(),
            max_tokens=12,
            slots=1,
            max_bytes=2 << 20,
            registrar=Registrar(),
            pin_memory=False,
            device=torch.device("cpu"),
            teacher_d2h_batch_tokens=3,
            max_device_bytes=1 << 20,
        )
        self.slot = self.pool.acquire()
        self.names = tuple(self.slot.teacher_device_tensors)
        for name in self.names:
            self.slot.tensors[name].fill_(-5)
        self.rows = capture_teacher(
            torch.arange(10 * 256).reshape(10, 256).float(), 256
        )
        self.expected = dict(
            zip(
                self.names, (self.rows.token_ids, self.rows.logits, self.rows.logsumexp)
            )
        )
        self.staging = TeacherStaging(
            self.slot.teacher_device_tensors, self.slot.tensors
        )

    def tearDown(self):
        self.pool.release(self.slot, transfer_complete=True)
        self.pool.close()

    def test_partial_batches_large_range_and_owned_tail(self):
        self.assertIsNone(self.slot.device_tensors)
        self.staging.append(self.rows, row=1, start=0, count=1)
        self.staging.append(self.rows, row=2, start=1, count=4)
        for name, expected in self.expected.items():
            torch.testing.assert_close(self.slot.tensors[name][:3], expected[1:4])
            self.assertTrue((self.slot.tensors[name][3:5] == -5).all())
        self.staging.append(self.rows, row=6, start=5, count=1)
        self.staging.append(self.rows, row=0, start=6, count=4)
        self.staging.append(self.rows, row=8, start=10, count=1)
        saved = {name: source[8].clone() for name, source in self.expected.items()}
        for source in self.expected.values():
            source.zero_()
        self.staging.flush()
        for name, expected in saved.items():
            torch.testing.assert_close(
                self.slot.tensors[name][10], expected, rtol=0, atol=0
            )
            self.assertTrue((self.slot.tensors[name][11:] == -5).all())

    def test_terminal_trim_flushes_exact_committed_teacher_prefix(self):
        context = RequestCaptureContext(
            slot=self.slot, prompt_ids=(3, 4), max_tokens=12, vocab_size=256
        )
        context.kv_end = 4
        context.record_teacher_range(self.rows, row=1, position=2, count=3)
        context.kv_end = 6
        context.record_teacher_range(self.rows, row=4, position=5, count=2)
        for position in (2, 3, 4, 5):
            context.commit_token(position=position, token_id=position)
        context.trim_terminal_prefix()
        context.seal("length")
        context.wait_for_copies()
        self.assertEqual(context.teacher_rows, 4)
        for name, source in self.expected.items():
            torch.testing.assert_close(self.slot.tensors[name][:4], source[1:5])

    def test_abort_discards_pending_rows_before_slot_reuse(self):
        context = RequestCaptureContext(
            slot=self.slot, prompt_ids=(3, 4), max_tokens=12, vocab_size=256
        )
        context.kv_end = 2
        context.record_teacher(self.rows, row=3, position=2)
        context.abort("request_aborted")
        context.wait_for_copies()
        self.pool.release(self.slot, transfer_complete=True)
        self.assertIs(self.slot, self.pool.acquire())
        replacement = RequestCaptureContext(
            slot=self.slot, prompt_ids=(3,), max_tokens=12, vocab_size=256
        )
        replacement.kv_end = 1
        replacement.record_teacher(self.rows, row=7, position=1)
        replacement.commit_token(position=1, token_id=7)
        replacement.seal("length")
        for name, source in self.expected.items():
            torch.testing.assert_close(self.slot.tensors[name][0], source[7])
            self.assertTrue((self.slot.tensors[name][1:] == -5).all())

    def test_invalid_rows_fail_before_any_host_write(self):
        for overrides in (
            {"start": 1},
            {"row": -1},
            {"row": 10},
            {"count": 0},
            {"count": 11},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ContractError):
                self.staging.append(
                    self.rows, **({"row": 0, "start": 0, "count": 1} | overrides)
                )
        for name in self.names:
            self.assertTrue((self.slot.tensors[name] == -5).all())

    def test_partial_flush_failure_never_seals(self):
        context = RequestCaptureContext(
            slot=self.slot, prompt_ids=(3,), max_tokens=12, vocab_size=256
        )
        context.kv_end = 1
        context.record_teacher(self.rows, row=0, position=1)
        context.commit_token(position=1, token_id=8)
        with (
            patch.object(
                context.teacher_staging,
                "flush",
                side_effect=RuntimeError("copy failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "copy failed"),
        ):
            context.seal("length")
        self.assertEqual(context.state, "COLLECTING")
        context.abort("teacher_flush_failed")

    def test_budget_includes_kv_teacher_all_slots_and_aux_only_ownership(self):
        kv = make_kv_spec()
        registrar = Registrar()
        with self.assertRaisesRegex(ValueError, "teacher staging requires"):
            HostBufferPool(
                kv=kv,
                max_tokens=12,
                slots=2,
                max_bytes=4 << 20,
                registrar=registrar,
                pin_memory=False,
                device=torch.device("cpu"),
                kv_d2h_batch_tokens=3,
                teacher_d2h_batch_tokens=3,
                max_device_bytes=self.pool.device_allocated_bytes * 2,
            )
        self.assertFalse(registrar.buffers)
        layout = plan_capture_layout(kv, tp_size=2, pp_layer_ranges=[(0, 4), (4, 6)])
        for partition in layout.partitions:
            if not partition.active:
                continue
            with self.subTest(owner=partition.owner_id):
                pool = HostBufferPool(
                    kv=kv,
                    max_tokens=12,
                    slots=1,
                    max_bytes=2 << 20,
                    registrar=registrar,
                    pin_memory=False,
                    device=torch.device("cpu"),
                    teacher_d2h_batch_tokens=3,
                    max_device_bytes=1 << 20,
                    partition=partition,
                )
                try:
                    self.assertEqual(
                        bool(pool.slots[0].teacher_device_tensors),
                        partition.include_aux,
                    )
                    if not partition.include_aux:
                        self.assertEqual(pool.device_allocated_bytes, 0)
                finally:
                    pool.close()


if __name__ == "__main__":
    unittest.main()
