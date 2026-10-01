"""Staged KV owns exact ranges through tail flush, abort and slot reuse."""

import unittest

import torch
from sglang.srt.training_capture.context import RequestCaptureContext
from sglang.srt.training_capture.host_pool import HostBufferPool
from sglang.srt.training_capture.kv_exporter import SelectedLayerKVExporter
from sglang.srt.training_capture.teacher import capture_teacher
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.training_capture_utils import Registrar, make_kv_spec

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestKVStaging(CustomTestCase):
    def setUp(self):
        self.kv = make_kv_spec()
        self.pool = HostBufferPool(
            kv=self.kv,
            max_tokens=8,
            slots=1,
            max_bytes=2 << 20,
            registrar=Registrar(),
            pin_memory=False,
            device=torch.device("cpu"),
            kv_d2h_batch_tokens=3,
            max_device_bytes=1 << 20,
        )
        self.slot = self.pool.acquire()
        self.sources = {
            name: torch.arange(16 * 8).reshape(16, 2, 4).bfloat16()
            for name in self.slot.device_tensors
        }
        for name in self.sources:
            self.slot.tensors[name].fill_(-5)
        self.exporter = SelectedLayerKVExporter(self.kv, self.sources)
        self.rows = capture_teacher(torch.arange(256).float()[None], 256)

    def tearDown(self):
        self.pool.release(self.slot, transfer_complete=True)
        self.pool.close()

    def context(self, prompt=(3, 4)):
        return RequestCaptureContext(
            slot=self.slot, prompt_ids=prompt, max_tokens=8, vocab_size=256
        )

    def accept(self, context, position):
        context.record_teacher(self.rows, row=0, position=position)
        context.commit_token(position=position, token_id=255)

    def test_large_prefix_bypasses_staging_and_short_tail_owns_source(self):
        context = self.context((3, 4, 5))
        indices = torch.tensor([7, 1, 9, 2, 12], dtype=torch.int32)
        expected = {
            name: value[indices].clone() for name, value in self.sources.items()
        }
        context.export_kv(self.exporter, indices[:3], end=3)
        for name, value in expected.items():
            torch.testing.assert_close(self.slot.tensors[name][:3], value[:3])
        self.accept(context, 3)
        for end in (4, 5):
            context.export_kv(self.exporter, indices[end - 1 : end], end=end)
            self.accept(context, end)
        for name, source in self.sources.items():
            self.assertTrue((self.slot.tensors[name][3:5] == -5).all())
            source.zero_()
        context.seal("length")
        for name, value in expected.items():
            torch.testing.assert_close(
                self.slot.tensors[name][:5], value, rtol=0, atol=0
            )
            self.assertTrue((self.slot.tensors[name][5:] == -5).all())

    def test_full_batch_flushes_but_aborted_tail_does_not_cross_slot_reuse(self):
        context = self.context()
        indices = torch.tensor([7, 1, 9, 2, 12], dtype=torch.int32)
        context.export_kv(self.exporter, indices[:2], end=2)
        context.export_kv(self.exporter, indices[2:3], end=3)
        for name, source in self.sources.items():
            torch.testing.assert_close(self.slot.tensors[name][:3], source[indices[:3]])
        context.export_kv(self.exporter, indices[3:], end=5)
        context.abort("request_aborted")
        context.wait_for_copies()
        self.pool.release(self.slot, transfer_complete=True)
        self.assertIs(self.pool.acquire(), self.slot)
        replacement = self.context((8,))
        for source in self.sources.values():
            source.fill_(99)
        replacement.export_kv(self.exporter, torch.tensor([0]), end=1)
        self.accept(replacement, 1)
        replacement.seal("length")
        for name in self.sources:
            self.assertTrue((self.slot.tensors[name][:1] == 99).all())
            self.assertTrue((self.slot.tensors[name][3:5] == -5).all())

    def test_device_budget_covers_all_slots_before_registration(self):
        one_slot_bytes = self.slot.device_storage.numel()
        registrar = Registrar()
        with self.assertRaisesRegex(ValueError, "KV staging requires"):
            HostBufferPool(
                kv=self.kv,
                max_tokens=8,
                slots=2,
                max_bytes=4 << 20,
                registrar=registrar,
                pin_memory=False,
                device=torch.device("cpu"),
                kv_d2h_batch_tokens=3,
                max_device_bytes=one_slot_bytes,
            )
        self.assertFalse(registrar.buffers)


if __name__ == "__main__":
    unittest.main()
