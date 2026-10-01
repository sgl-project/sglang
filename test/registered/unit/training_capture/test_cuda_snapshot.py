"""CUDA source-reuse checks for independent teacher/KV snapshots."""

import unittest

import torch
from sglang.srt.training_capture.context import RequestCaptureContext
from sglang.srt.training_capture.host_pool import HostBufferPool
from sglang.srt.training_capture.kv_exporter import SelectedLayerKVExporter
from sglang.srt.training_capture.teacher import capture_teacher
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.training_capture_utils import Registrar, make_kv_spec

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestCudaSnapshot(CustomTestCase):
    def test_finalization_waits_for_last_lookahead_copy_on_forward_stream(self):
        kv = make_kv_spec()
        pool = HostBufferPool(
            kv=kv, max_tokens=8, slots=1, max_bytes=2 << 20, registrar=Registrar()
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


if __name__ == "__main__":
    unittest.main()
