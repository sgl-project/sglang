"""XPU tensors must survive MultiprocessingSerializer without GPU tensor IPC.

PyTorch has no XPU tensor IPC, so ForkingPickler fell through to the CPU storage
path and every producer of a device tensor died with "_share_fd_: only available
on CPU" - or, between processes without a shared authkey, with an
AuthenticationError from the fd-passing fallback. RL weight sync
(Engine.update_weights_from_tensor) serializes device tensors, so it could not
run on XPU at all.
"""

import os
import subprocess
import sys
import unittest
from multiprocessing import shared_memory

import torch

from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import MultiprocessingSerializer, is_xpu
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
from sglang.srt.utils.xpu_tensor_shm import (
    _rebuild_xpu_tensor_from_shm,
    staged_xpu_segments,
)
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=90, suite="stage-b-test-1-gpu-xpu")

_SHM_DIR = "/dev/shm"
_SEGMENT_PREFIX = "sgl_shm_xputensor_"

# Serialize in a fresh interpreter: a process spawned this way shares no
# multiprocessing authkey with the consumer, which is what makes the fd-passing
# fallback unusable under Ray.
_PRODUCER_SOURCE = """
import os, sys
import torch
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

monkey_patch_torch_reductions()
tensor = torch.arange(24, device="xpu", dtype=torch.bfloat16).reshape(4, 6) * 0.25
payload = MultiprocessingSerializer.serialize([("w", tensor)], output_str=True)
with open(sys.argv[1], "w") as f:
    f.write(payload)
print(os.getpid(), flush=True)
"""


def _segments_of(pid: int) -> list:
    prefix = f"{_SEGMENT_PREFIX}{pid}_"
    return sorted(n for n in os.listdir(_SHM_DIR) if n.startswith(prefix))


@unittest.skipUnless(is_xpu(), "requires an Intel XPU device")
class TestXpuTensorShmSerialization(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        monkey_patch_torch_reductions()

    def tearDown(self):
        self.assertEqual(
            _segments_of(os.getpid()),
            [],
            "deserializing must unlink the staged shared-memory segment",
        )

    def test_roundtrip_preserves_values_shape_and_type(self):
        """Staging math (byte count, dtype view, contiguity) must be exact."""
        cases = {
            "bf16_2d": torch.arange(64, device="xpu", dtype=torch.bfloat16).reshape(
                8, 8
            ),
            "fp8": torch.arange(16, device="xpu").to(torch.float8_e4m3fn),
            "int64": torch.arange(10, device="xpu"),
            "zero_dim": torch.tensor(3.5, device="xpu"),
            "empty": torch.empty(0, 5, device="xpu"),
            "transposed": torch.arange(12, device="xpu").reshape(3, 4).t(),
            "parameter": torch.nn.Parameter(torch.ones(4, 4, device="xpu")),
        }
        for name, tensor in cases.items():
            with self.subTest(case=name):
                out = MultiprocessingSerializer.deserialize(
                    MultiprocessingSerializer.serialize(tensor)
                )
                expected = tensor.detach().contiguous()
                self.assertEqual(out.device.type, "xpu")
                self.assertEqual(out.shape, expected.shape)
                self.assertEqual(out.dtype, expected.dtype)
                self.assertEqual(type(out), type(tensor))
                # Compare raw bytes so fp8 (which has no torch.equal) is covered.
                self.assertTrue(
                    torch.equal(
                        out.reshape(-1).view(torch.uint8),
                        expected.reshape(-1).view(torch.uint8),
                    )
                )

    def test_payload_outlives_an_unrelated_producer_process(self):
        """The consumer owns the staged segment, so it survives producer exit.

        A trainer actor serializes and an mp.spawn-ed scheduler deserializes, with
        no common authkey and no lifetime overlap guaranteed between the two.
        """
        payload_path = os.path.join(
            os.environ.get("TMPDIR", "/tmp"), f"xpu_shm_payload_{os.getpid()}.b64"
        )
        self.addCleanup(
            lambda: os.path.exists(payload_path) and os.remove(payload_path)
        )

        producer = subprocess.run(
            [sys.executable, "-c", _PRODUCER_SOURCE, payload_path],
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(
            producer.returncode,
            0,
            f"producer failed:\n{producer.stdout}\n{producer.stderr}",
        )
        producer_pid = int(producer.stdout.strip().splitlines()[-1])
        self.assertEqual(
            len(_segments_of(producer_pid)),
            1,
            "the staged segment must still exist after the producer exits",
        )

        with open(payload_path) as f:
            named_tensors = MultiprocessingSerializer.deserialize(f.read())

        (name, tensor) = named_tensors[0]
        expected = torch.arange(24, device="xpu", dtype=torch.bfloat16).reshape(4, 6)
        self.assertEqual(name, "w")
        self.assertEqual(tensor.device.type, "xpu")
        self.assertTrue(torch.equal(tensor, expected * 0.25))
        self.assertEqual(
            _segments_of(producer_pid),
            [],
            "deserializing must unlink the producer's segment",
        )

    def test_refuses_a_segment_it_did_not_stage(self):
        """The segment name arrives in a pickle from an unauthenticated request.

        Rebuilding reads the named segment into a model parameter and unlinks it,
        so a name outside the staged shape would be a read-and-unlink primitive
        over any /dev/shm segment the scheduler's uid owns.
        """
        victim = shared_memory.SharedMemory(
            create=True, size=8, name=f"not_staged_{os.getpid()}"
        )
        self.addCleanup(victim.unlink)
        self.addCleanup(victim.close)

        for name in (victim.name, "sgl_shm_xputensor_bogus", "sgl_shm_mq_1_abcdef12"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(RuntimeError, "Refusing to map"):
                    _rebuild_xpu_tensor_from_shm(
                        torch.Tensor, (8,), torch.uint8, False, name, 8, True
                    )
        self.assertTrue(os.path.exists(os.path.join(_SHM_DIR, victim.name)))

    def test_scoped_payload_serves_every_reader_from_one_segment(self):
        """Every TP rank's payload points at one segment, read in any order.

        Serializing once per TP rank must not stage one copy per rank, and no
        reader may unlink what another rank has yet to map.
        """
        weight = torch.arange(32, device="xpu", dtype=torch.bfloat16)
        with staged_xpu_segments(pp_size=1):
            payloads = [MultiprocessingSerializer.serialize([("w", weight)])] * 2
            payloads.append(MultiprocessingSerializer.serialize([("w", weight)]))
            self.assertEqual(len(_segments_of(os.getpid())), 1)

            for payload in payloads + payloads:
                ((_, out),) = MultiprocessingSerializer.deserialize(payload)
                self.assertTrue(torch.equal(out, weight))

        self.assertEqual(_segments_of(os.getpid()), [])

    def test_later_pipeline_stages_read_after_the_call_returned(self):
        """With --pp-size 2, stage 1 reads payload[tp_rank] after stage 0 replied.

        Unlinking when the call returned, or on stage 0's read, left stage 1 a
        missing segment, and the error escaped its scheduler and took the server
        down. Only the last stage's reader may unlink.
        """
        weight = torch.arange(16, device="xpu", dtype=torch.float32)
        with staged_xpu_segments(pp_size=2):
            payload = MultiprocessingSerializer.serialize(weight)
        self.assertEqual(len(_segments_of(os.getpid())), 1)

        for pp_rank in (0, 1):
            with get_parallel().override(pp_rank=pp_rank, pp_size=2):
                restored = MultiprocessingSerializer.deserialize(payload)
            self.assertTrue(torch.equal(restored, weight))
            self.assertEqual(len(_segments_of(os.getpid())), 1 - pp_rank)

    def test_scope_reclaims_payloads_no_reader_took(self):
        """A request rejected before deserializing, e.g. outside a weight-update
        session, must not leave its staging in /dev/shm."""
        with staged_xpu_segments(pp_size=1):
            for i in range(3):
                MultiprocessingSerializer.serialize(torch.full((64,), i, device="xpu"))
            self.assertEqual(len(_segments_of(os.getpid())), 3)

        self.assertEqual(_segments_of(os.getpid()), [])

    def test_scope_leaves_payloads_serialized_before_it(self):
        """A flattened_bucket caller serializes its next bucket ahead of the call.

        That payload is unlinked by its reader, so a scope that does not own it
        must leave it alone.
        """
        weight = torch.ones(16, device="xpu")
        next_bucket = MultiprocessingSerializer.serialize(weight)
        with staged_xpu_segments(pp_size=1):
            MultiprocessingSerializer.serialize(torch.zeros(16, device="xpu"))

        self.assertEqual(len(_segments_of(os.getpid())), 1)
        self.assertTrue(
            torch.equal(MultiprocessingSerializer.deserialize(next_bucket), weight)
        )

    def test_scope_restages_a_tensor_written_in_place(self):
        """Dedup keys on the storage address, which an in-place write keeps.

        Serving the earlier segment would hand the scheduler stale weights.
        """
        weight = torch.zeros(8, device="xpu")
        with staged_xpu_segments(pp_size=1):
            before = MultiprocessingSerializer.serialize(weight)
            weight.fill_(1.0)
            after = MultiprocessingSerializer.serialize(weight)

            self.assertEqual(len(_segments_of(os.getpid())), 2)
            restored_before = MultiprocessingSerializer.deserialize(before)
            restored_after = MultiprocessingSerializer.deserialize(after)
            self.assertTrue(torch.equal(restored_before, torch.zeros_like(weight)))
            self.assertTrue(torch.equal(restored_after, weight))


if __name__ == "__main__":
    unittest.main()
