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

from sglang.srt.utils import MultiprocessingSerializer, is_xpu
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
from sglang.srt.utils.xpu_tensor_shm import (
    _rebuild_xpu_tensor_from_shm,
    discard_staged_segments,
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
                        torch.Tensor, (8,), torch.uint8, False, name, 8
                    )
        self.assertTrue(os.path.exists(os.path.join(_SHM_DIR, victim.name)))

    def test_discard_staged_segments_reclaims_an_unconsumed_payload(self):
        """A payload no consumer takes would otherwise hold host memory forever."""
        MultiprocessingSerializer.serialize(torch.ones(1024, device="xpu"))
        self.assertEqual(len(_segments_of(os.getpid())), 1)

        discard_staged_segments()
        self.assertEqual(_segments_of(os.getpid()), [])


if __name__ == "__main__":
    unittest.main()
