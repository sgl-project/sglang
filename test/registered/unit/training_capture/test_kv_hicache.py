"""Mapped Host KV export, bounded metadata and source lifetime on CUDA."""

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import msgspec
import torch
from sglang.srt.training_capture.context import RequestCaptureContext
from sglang.srt.training_capture.host_pool import HostBufferPool
from sglang.srt.training_capture.kv_exporter import SelectedLayerKVExporter
from sglang.srt.training_capture.protocol import (
    CaptureError,
    ContractError,
    LayerGeometry,
    validate_kv_spec,
)
from sglang.srt.training_capture.teacher import capture_teacher
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.training_capture_utils import Registrar, make_kv_spec

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")


def fixture(*, dtype=torch.bfloat16, staging=1, slots=1, padded=False):
    kv = msgspec.structs.replace(
        make_kv_spec(),
        dtype=str(dtype).removeprefix("torch."),
        codec=f"dense_{'bf16' if dtype == torch.bfloat16 else 'fp16'}_post_rope_v1",
        layers=[
            LayerGeometry(
                layer_id=3, num_kv_heads=2, key_head_dim=64, value_head_dim=128
            ),
            LayerGeometry(
                layer_id=1, num_kv_heads=1, key_head_dim=64, value_head_dim=128
            ),
        ],
    )
    validate_kv_spec(kv)
    source = {}
    for layer in kv.layers:
        for kind, dim in (("k", layer.key_head_dim), ("v", layer.value_head_dim)):
            value = torch.randn(64, layer.num_kv_heads, dim, device="cuda", dtype=dtype)
            source[f"target_{kind}.{layer.layer_id}"] = value[::2] if padded else value
    exporter = SelectedLayerKVExporter(kv, source)
    pool = HostBufferPool(
        kv=kv,
        max_tokens=17,
        slots=slots,
        max_bytes=slots * (2 << 20),
        registrar=Registrar(),
        device=torch.device("cuda"),
        kv_d2h_batch_tokens=staging,
        teacher_d2h_batch_tokens=3,
        kv_export_backend="hicache",
        max_device_bytes=slots * (1 << 20),
    )
    for slot in pool.slots:
        slot.kv_exporter = exporter.bind(slot)
    torch.cuda.synchronize()
    return kv, source, exporter, pool


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestHiCacheKVExport(CustomTestCase):
    def test_source_reuse_strides_dtypes_stream_changes_and_staged_tail(self):
        for dtype in (torch.bfloat16, torch.float16):
            for staging in (1, 3):
                with self.subTest(dtype=dtype, staging=staging):
                    _, source, exporter, pool = fixture(
                        dtype=dtype, staging=staging, padded=True
                    )
                    slot = pool.acquire()
                    indices = torch.tensor(
                        [19, 0, 19, 0, 3, 0, 12, 0, 2, 0, 17, 0, 5, 0],
                        device="cuda",
                        dtype=torch.int32,
                    )[::2]
                    expected = {
                        name: value.index_select(0, indices).cpu()
                        for name, value in source.items()
                    }
                    for name in source:
                        slot.tensors[name].fill_(9)
                    context = RequestCaptureContext(
                        slot=slot, prompt_ids=(2, 3), max_tokens=17, vocab_size=256
                    )
                    streams = (torch.cuda.Stream(), torch.cuda.Stream())
                    for stream in streams:
                        stream.wait_stream(torch.cuda.current_stream())
                    try:
                        for step, (start, end) in enumerate(
                            ((0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7))
                        ):
                            with torch.cuda.stream(streams[step % 2]):
                                torch.cuda._sleep(200000)
                                context.export_kv(exporter, indices[start:end], end=end)
                                for value in source.values():
                                    value[indices[start:end]] = -31
                                teacher = capture_teacher(
                                    torch.randn(1, 256, device="cuda"), 256
                                )
                                context.record_teacher(teacher, row=0, position=end)
                                context.commit_token(position=end, token_id=7)
                        context.seal("length")
                        context.wait_for_copies()
                        for name, value in expected.items():
                            torch.testing.assert_close(
                                slot.tensors[name][:7], value, rtol=0, atol=0
                            )
                            self.assertTrue((slot.tensors[name][7:] == 9).all())
                        stats = pool.stats()
                        self.assertGreater(stats["device_allocated_bytes"], 0)
                        byte_count = (
                            sum(
                                value[0].numel() * value.element_size()
                                for value in source.values()
                            )
                            * 7
                        )
                        if staging == 1:
                            self.assertEqual(
                                stats["kv_export_host_enqueued_bytes"], byte_count
                            )
                            self.assertEqual(
                                stats["kv_export_device_enqueued_bytes"], 0
                            )
                        else:
                            self.assertEqual(
                                stats["kv_export_device_enqueued_bytes"], byte_count
                            )
                    finally:
                        torch.cuda.synchronize()
                        pool.release(slot, transfer_complete=True)
                        pool.close()

    def test_independent_slots_graphs_and_concurrent_streams(self):
        _, source, _, pool = fixture(slots=2)
        slots = [pool.acquire(), pool.acquire()]
        indices = [
            torch.tensor([2, 0, 2, 7], device="cuda", dtype=dtype)
            for dtype in (torch.int32, torch.int64)
        ]
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
        try:
            for slot, index, stream, graph in zip(slots, indices, streams, graphs):
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        slot.kv_exporter.export(index, slot.tensors, 3, 7)
                stream.synchronize()
                with torch.cuda.graph(graph, stream=stream):
                    slot.kv_exporter.export(index, slot.tensors, 3, 7)
            for _ in range(4):
                for value in source.values():
                    value.normal_()
                expected = {
                    name: value.index_select(0, indices[0]).cpu()
                    for name, value in source.items()
                }
                for stream, graph in zip(streams, graphs):
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        torch.cuda._sleep(200000)
                        graph.replay()
                for stream in streams:
                    stream.synchronize()
                for slot in slots:
                    for name, value in expected.items():
                        torch.testing.assert_close(
                            slot.tensors[name][3:7], value, rtol=0, atol=0
                        )
        finally:
            torch.cuda.synchronize()
            for slot in slots:
                pool.release(slot, transfer_complete=True)
            pool.close()

    def test_budget_rejection_precedes_registration(self):
        registrar = Registrar()
        with self.assertRaisesRegex(ValueError, "metadata and staging require"):
            HostBufferPool(
                kv=make_kv_spec(),
                max_tokens=17,
                slots=2,
                max_bytes=4 << 20,
                registrar=registrar,
                device=torch.device("cuda"),
                kv_export_backend="hicache",
                max_device_bytes=1,
            )
        self.assertEqual(registrar.buffers, {})
        self.assertEqual(registrar.registrations, 0)

    def test_unbound_destinations_ranges_and_unaligned_rows_are_rejected(self):
        _, _, _, pool = fixture()
        slot = pool.acquire()
        indices = torch.zeros(2, dtype=torch.int32, device="cuda")
        try:
            for mapping, start, end in (
                (dict(slot.tensors), 0, 2),
                (slot.tensors, 16, 18),
                (slot.tensors, -1, 1),
                (slot.tensors, 0, 1),
            ):
                with (
                    self.subTest(start=start, end=end),
                    self.assertRaises(ContractError),
                ):
                    slot.kv_exporter.export(indices, mapping, start, end)
            tiny = make_kv_spec()
            source = {
                f"target_{kind}.{layer.layer_id}": torch.zeros(
                    8, 2, 4, device="cuda", dtype=torch.bfloat16
                )
                for layer in tiny.layers
                for kind in ("k", "v")
            }
            with self.assertRaisesRegex(ContractError, "multiple of 128"):
                SelectedLayerKVExporter(tiny, source).bind(slot)
        finally:
            torch.cuda.synchronize()
            pool.release(slot, transfer_complete=True)
            pool.close()

    def test_uncertain_completion_quarantines_metadata_and_host_storage(self):
        _, _, exporter, pool = fixture()
        slot = pool.acquire()
        context = RequestCaptureContext(
            slot=slot, prompt_ids=(3,), max_tokens=17, vocab_size=256
        )
        metadata = slot.device_storage
        try:
            with (
                patch("torch.cuda.Event", side_effect=RuntimeError("fence failed")),
                self.assertRaisesRegex(RuntimeError, "fence failed"),
            ):
                context.export_kv(
                    exporter, torch.tensor([0], device="cuda", dtype=torch.int32), end=1
                )
            self.assertTrue(context.transfer_uncertain)
            pool.release(slot, transfer_complete=False)
            self.assertIsNone(pool.acquire())
            self.assertIs(pool.slots[0].device_storage, metadata)
            with self.assertRaises(CaptureError):
                pool.close()
        finally:
            torch.cuda.synchronize()

    def test_out_of_range_gpu_indices_fail_before_raw_pointer_copy(self):
        # Device assertions poison a CUDA context; isolate each malformed input.
        for invalid in (-1, 64):
            script = f"""
import torch
from test_kv_hicache import fixture
_, _, _, pool = fixture()
slot = pool.acquire()
slot.kv_exporter.export(torch.tensor([{invalid}], device="cuda", dtype=torch.int32), slot.tensors, 0, 1)
torch.cuda.synchronize()
"""
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=Path(__file__).resolve().parent,
                check=False,
                capture_output=True,
                text=True,
                timeout=90,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("capture KV source index out of range", result.stderr)


if __name__ == "__main__":
    unittest.main()
