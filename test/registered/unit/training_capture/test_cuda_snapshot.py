"""CUDA source-reuse checks for independent teacher/KV snapshots."""

import unittest
from unittest.mock import patch

import torch
from sglang.srt.training_capture.context import RequestCaptureContext
from sglang.srt.training_capture.host_pool import HostBufferPool
from sglang.srt.training_capture.kv_exporter import SelectedLayerKVExporter
from sglang.srt.training_capture.protocol import CaptureError, ContractError
from sglang.srt.training_capture.teacher import TeacherRows, capture_teacher
from sglang.srt.training_capture.topology import plan_capture_layout
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.training_capture_utils import Registrar, make_kv_spec

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestCudaSnapshot(CustomTestCase):
    def test_teacher_tail_fence_failure_retains_staging_and_host_storage(self):
        pool = HostBufferPool(
            kv=make_kv_spec(),
            max_tokens=8,
            slots=1,
            max_bytes=2 << 20,
            registrar=Registrar(),
            device=torch.device("cuda"),
            teacher_d2h_batch_tokens=3,
            max_device_bytes=1 << 20,
        )
        slot = pool.acquire()
        context = RequestCaptureContext(
            slot=slot, prompt_ids=(3,), max_tokens=8, vocab_size=256
        )
        context.kv_end = 1
        context.record_teacher(
            capture_teacher(torch.arange(256, device="cuda").float()[None], 256),
            row=0,
            position=1,
        )
        context.commit_token(position=1, token_id=8)
        storage = slot.device_storage
        try:
            with (
                patch(
                    "torch.cuda.Event", side_effect=RuntimeError("teacher fence failed")
                ),
                self.assertRaisesRegex(RuntimeError, "teacher fence failed"),
            ):
                context.seal("length")
            self.assertTrue(context.transfer_uncertain)
            context.abort("teacher_fence_failed")
            with self.assertRaisesRegex(ContractError, "uncertain"):
                context.wait_for_copies()
            pool.release(slot, transfer_complete=False)
            self.assertIsNone(pool.acquire())
            with self.assertRaises(CaptureError):
                pool.close()
            self.assertIs(pool.slots[0].device_storage, storage)
        finally:
            torch.cuda.synchronize()

    def test_teacher_cpu_handoff_cross_stream_reuse_and_tail(self):
        pool = HostBufferPool(
            kv=make_kv_spec(),
            max_tokens=12,
            slots=1,
            max_bytes=2 << 20,
            registrar=Registrar(),
            device=torch.device("cuda"),
            teacher_d2h_batch_tokens=3,
            max_device_bytes=1 << 20,
        )
        slot = pool.acquire()
        context = RequestCaptureContext(
            slot=slot, prompt_ids=(3, 4), max_tokens=12, vocab_size=256
        )
        names = tuple(slot.teacher_device_tensors)
        raw = torch.arange(8 * 256).reshape(8, 256).float()
        reference = capture_teacher(raw, 256)
        expected = (reference.token_ids, reference.logits, reference.logsumexp)
        for name in names:
            slot.tensors[name].fill_(-5)
        first, second = torch.cuda.Stream(), torch.cuda.Stream()
        try:
            context.kv_end = 2
            context.record_teacher(reference, row=0, position=2)
            context.commit_token(position=2, token_id=10)
            for name, source in zip(names, expected):
                torch.testing.assert_close(slot.tensors[name][0], source[0])
            for index in range(1, 8):
                with torch.cuda.stream(first if index % 2 else second):
                    rows = TeacherRows(
                        *(value[index : index + 1].cuda() for value in expected)
                    )
                    torch.cuda._sleep(1000000)
                    context.kv_end = index + 2
                    context.record_teacher(rows, row=0, position=index + 2)
                    context.commit_token(position=index + 2, token_id=10)
                    for value in (rows.token_ids, rows.logits, rows.logsumexp):
                        value.zero_()
            # Seal runs on a different stream from the last append.
            context.seal("length")
            context.wait_for_copies()
            for name, source in zip(names, expected):
                torch.testing.assert_close(
                    slot.tensors[name][:8], source, rtol=0, atol=0
                )
                self.assertTrue((slot.tensors[name][8:] == -5).all())
        finally:
            torch.cuda.synchronize()
            pool.release(slot, transfer_complete=True)
            pool.close()

    def test_partition_staging_survives_canonical_rank_source_slot_reuse(self):
        kv = make_kv_spec()
        partition = plan_capture_layout(
            kv, tp_size=4, pp_layer_ranges=[(0, 4)]
        ).partition("dp0-pp0-tp2")
        pool = HostBufferPool(
            kv=kv,
            max_tokens=8,
            slots=1,
            max_bytes=4096,
            registrar=Registrar(),
            device=torch.device("cuda"),
            kv_d2h_batch_tokens=3,
            max_device_bytes=4096,
            partition=partition,
        )
        slot = pool.acquire()
        sources = {
            f"target_{component}.{layer.layer_id}": (
                torch.arange(16 * 4, device="cuda").reshape(16, 1, 4).bfloat16()
                + layer.layer_id * 100
            )
            for layer in kv.layers
            for component in ("k", "v")
        }
        indices = torch.tensor([7, 1, 15, 3, 8], device="cuda", dtype=torch.int32)
        expected = {name: source[indices].cpu() for name, source in sources.items()}
        exporter = SelectedLayerKVExporter(kv, sources, partition=partition)
        context = RequestCaptureContext(
            slot=slot,
            prompt_ids=(3, 4),
            max_tokens=8,
            vocab_size=256,
            partition=partition,
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        try:
            with torch.cuda.stream(stream):
                for position in range(5):
                    context.export_kv(
                        exporter,
                        indices[position : position + 1],
                        end=position + 1,
                    )
                    if position >= 1:
                        context.commit_token(
                            position=position + 1, token_id=position + 9
                        )
                    for source in sources.values():
                        source[indices[position]] = 77
            context.seal("length")
            context.wait_for_copies()
            self.assertEqual(set(slot.tensors), set(sources))
            self.assertEqual(slot.manifest_buffer.numel(), 0)
            for name, reference in expected.items():
                torch.testing.assert_close(
                    slot.tensors[name][:5], reference, rtol=0, atol=0
                )
        finally:
            stream.synchronize()
            pool.release(slot, transfer_complete=True)
            pool.close()

    def test_finalization_waits_for_last_lookahead_copy_on_forward_stream(self):
        kv = make_kv_spec()
        pool = HostBufferPool(
            kv=kv,
            max_tokens=8,
            slots=1,
            max_bytes=2 << 20,
            registrar=Registrar(),
            device=torch.device("cuda"),
            kv_d2h_batch_tokens=2,
            teacher_d2h_batch_tokens=3,
            max_device_bytes=1 << 20,
        )
        slot = pool.acquire()
        context = RequestCaptureContext(
            slot=slot, prompt_ids=(3, 4), max_tokens=8, vocab_size=256
        )
        stream = torch.cuda.Stream()
        sources = {
            f"target_{component}.{layer.layer_id}": torch.arange(16 * 8, device="cuda")
            .reshape(16, 2, 4)
            .bfloat16()
            for layer in kv.layers
            for component in ("k", "v")
        }
        exporter = SelectedLayerKVExporter(kv, sources)
        slots = torch.tensor([7, 0, 3, 0, 6, 0], dtype=torch.int32, device="cuda")[::2]
        expected = {name: buffer[slots].cpu() for name, buffer in sources.items()}
        logits = torch.arange(256, device="cuda").float()[None]
        stream.wait_stream(torch.cuda.current_stream())
        try:
            with torch.cuda.stream(stream):
                context.export_kv(exporter, slots[:2], end=2)
                context.record_teacher(capture_teacher(logits, 256), row=0, position=2)
                first_event = context.last_event
                torch.cuda._sleep(2000000)
                context.export_kv(exporter, slots[2:], end=3)
                context.record_teacher(
                    capture_teacher(logits + 1, 256), row=0, position=3
                )
                for source in sources.values():
                    source.zero_()
                logits.zero_()
            self.assertIsNot(context.last_event, first_event)
            context.commit_token(position=2, token_id=10)
            context.trim_terminal_prefix()
            context.seal("length")
            context.wait_for_copies()
            self.assertTrue(context.last_event.query())
            self.assertEqual(context.teacher_rows, 1)
            self.assertEqual(context.kv_end, 3)
            for name, reference in expected.items():
                torch.testing.assert_close(
                    slot.tensors[name][:3], reference, rtol=0, atol=0
                )
            torch.testing.assert_close(
                slot.tensors["teacher_topk_logits"][0],
                torch.arange(255, 127, -1).float(),
                rtol=0,
                atol=0,
            )
        finally:
            stream.synchronize()
            pool.release(slot, transfer_complete=True)
            pool.close()

    def test_source_reuse_after_gather_and_raw_logits_mutation(self):
        kv = make_kv_spec()
        pool = HostBufferPool(
            kv=kv, max_tokens=8, slots=1, max_bytes=2 << 20, registrar=Registrar()
        )
        slot = pool.acquire()
        stream = torch.cuda.Stream()
        event = torch.cuda.Event()
        indices = torch.tensor([7, 1, 15, 3, 8], device="cuda")
        buffers = {
            f"target_{c}.{g.layer_id}": torch.arange(16 * 8, device="cuda")
            .reshape(16, 2, 4)
            .bfloat16()
            + g.layer_id
            for g in kv.layers
            for c in ("k", "v")
        }
        expected = {name: value[indices].cpu() for name, value in buffers.items()}
        original_logits = torch.randn(3, 260, device="cuda", dtype=torch.bfloat16)
        original_logits[:, 256:] = 10000
        reference = original_logits[:, :256].float().cpu()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            rows = capture_teacher(original_logits, 256)
            for name, value in (
                ("teacher_topk_ids", rows.token_ids),
                ("teacher_topk_logits", rows.logits),
                ("teacher_logsumexp", rows.logsumexp),
            ):
                slot.tensors[name][:3].copy_(value, non_blocking=True)
                value.record_stream(stream)
            SelectedLayerKVExporter(kv, buffers).export(indices, slot.tensors, 0, 5)
            # Simulate source slot recycling / the next CUDA graph replay.
            for source in buffers.values():
                source.fill_(77)
            original_logits.fill_(-float("inf"))
            event.record(stream)
        event.synchronize()
        for name in buffers:
            torch.testing.assert_close(slot.tensors[name][:5], expected[name])
        ids = slot.tensors["teacher_topk_ids"][:3].long()
        torch.testing.assert_close(
            slot.tensors["teacher_topk_logits"][:3], reference.gather(1, ids)
        )
        torch.testing.assert_close(
            slot.tensors["teacher_logsumexp"][:3],
            reference.double().logsumexp(-1).float(),
        )
        pool.release(slot, transfer_complete=True)
        pool.close()

    def test_tail_fence_failure_quarantines_host_and_device_storage(self):
        kv = make_kv_spec()
        pool = HostBufferPool(
            kv=kv,
            max_tokens=8,
            slots=1,
            max_bytes=2 << 20,
            registrar=Registrar(),
            device=torch.device("cuda"),
            kv_d2h_batch_tokens=3,
            max_device_bytes=1 << 20,
        )
        slot = pool.acquire()
        device_storage = slot.device_storage
        context = RequestCaptureContext(
            slot=slot, prompt_ids=(3,), max_tokens=8, vocab_size=256
        )
        source = {
            name: torch.ones(16, 2, 4, dtype=torch.bfloat16, device="cuda")
            for name in slot.device_tensors
        }
        context.export_kv(
            SelectedLayerKVExporter(kv, source), torch.tensor([7], device="cuda"), end=1
        )
        context.record_teacher(
            capture_teacher(torch.arange(256, device="cuda").float()[None], 256),
            row=0,
            position=1,
        )
        context.commit_token(position=1, token_id=255)
        try:
            with (
                patch("torch.cuda.Event", side_effect=RuntimeError("event failure")),
                self.assertRaisesRegex(RuntimeError, "event failure"),
            ):
                context.seal("length")
            self.assertTrue(context.transfer_uncertain)
            context.abort("copy_completion_uncertain")
            with self.assertRaisesRegex(ContractError, "uncertain"):
                context.wait_for_copies()
            pool.release(slot, transfer_complete=False)
            self.assertIsNone(pool.acquire())
            with self.assertRaises(CaptureError):
                pool.close()
            self.assertIs(pool.slots[0].device_storage, device_storage)
        finally:
            torch.cuda.synchronize()

    def test_full_staging_reuse_and_cross_stream_tail_preserve_exact_kv(self):
        kv = make_kv_spec()
        pool = HostBufferPool(
            kv=kv,
            max_tokens=8,
            slots=1,
            max_bytes=2 << 20,
            registrar=Registrar(),
            device=torch.device("cuda"),
            kv_d2h_batch_tokens=3,
            max_device_bytes=1 << 20,
        )
        slot = pool.acquire()
        context = RequestCaptureContext(
            slot=slot, prompt_ids=(3, 4), max_tokens=8, vocab_size=256
        )
        indices = torch.tensor([7, 1, 12, 3, 9], device="cuda", dtype=torch.int32)
        sources = {
            name: torch.arange(16 * 8, device="cuda").reshape(16, 2, 4).bfloat16()
            for name in slot.device_tensors
        }
        expected = {name: value[indices].cpu() for name, value in sources.items()}
        for name in sources:
            slot.tensors[name].fill_(-5)
        exporter = SelectedLayerKVExporter(kv, sources)
        first, second = torch.cuda.Stream(), torch.cuda.Stream()
        first.wait_stream(torch.cuda.current_stream())
        try:
            with torch.cuda.stream(first):
                torch.cuda._sleep(2000000)
                context.export_kv(exporter, indices[:2], end=2)
                context.record_teacher(
                    capture_teacher(
                        torch.arange(256, device="cuda").float()[None], 256
                    ),
                    row=0,
                    position=2,
                )
                context.commit_token(position=2, token_id=255)
            with torch.cuda.stream(second):
                context.export_kv(exporter, indices[2:3], end=3)
                context.record_teacher(
                    capture_teacher(
                        torch.arange(256, device="cuda").float()[None], 256
                    ),
                    row=0,
                    position=3,
                )
                context.commit_token(position=3, token_id=255)
                context.export_kv(exporter, indices[3:], end=5)
                context.record_teacher_range(
                    capture_teacher(
                        torch.arange(256, device="cuda").float().repeat(2, 1), 256
                    ),
                    row=0,
                    position=4,
                    count=2,
                )
                context.commit_token(position=4, token_id=255)
                context.commit_token(position=5, token_id=255)
                for value in sources.values():
                    value.zero_()
            context.seal("length")
            context.wait_for_copies()
            for name, reference in expected.items():
                torch.testing.assert_close(
                    slot.tensors[name][:5], reference, rtol=0, atol=0
                )
                self.assertTrue((slot.tensors[name][5:] == -5).all())
        finally:
            first.synchronize()
            second.synchronize()
            pool.release(slot, transfer_complete=True)
            pool.close()


if __name__ == "__main__":
    unittest.main()
