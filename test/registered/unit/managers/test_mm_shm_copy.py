"""Round-trip multimodal shared-memory features with exact bytes."""

import pickle
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory
from unittest.mock import patch

import torch

from sglang.srt.managers.mm_utils import ShmPointerMMData
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


class TestMultimodalShmCopy(CustomTestCase):
    def test_round_trip(self):
        for dtype in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
            torch.bool,
        ):
            for noncontiguous in (False, True):
                with self.subTest(dtype=dtype, noncontiguous=noncontiguous):
                    original = torch.arange(128 * 64).reshape(128, 64).to(dtype)
                    if noncontiguous:
                        original = original.t()
                    if dtype == torch.float32:
                        original.requires_grad_()
                    source = original.detach().clone()
                    ptr = ShmPointerMMData(original, precomputed_hash=123)
                    received = pickle.loads(pickle.dumps(ptr))
                    self.assertEqual(received.precomputed_hash, 123)
                    result = received.materialize()
                    self.assertTrue(torch.equal(result, source))
                    self.assertTrue(
                        torch.equal(
                            result.contiguous().view(torch.uint8),
                            source.contiguous().view(torch.uint8),
                        )
                    )
                    result[0, 0] = 1
                    self.assertTrue(torch.equal(original.detach(), source))

    def test_copy_failure_unlinks_segment(self):
        name = "sglang_test_" + uuid.uuid4().hex
        real_shared_memory = shared_memory.SharedMemory

        class FailingBuffer:
            def __setitem__(self, key, value):
                raise RuntimeError("copy failed")

        class FailingSegment:
            def __init__(self, *args, **kwargs):
                self.segment = real_shared_memory(*args, **kwargs)
                self.buf = FailingBuffer()
                self._fd = self.segment._fd

            def close(self):
                self.segment.close()

            def unlink(self):
                self.segment.unlink()

        with patch("sglang.srt.managers.mm_utils.make_shm_name", return_value=name):
            with patch(
                "sglang.srt.managers.mm_utils.shared_memory.SharedMemory",
                side_effect=FailingSegment,
            ):
                with self.assertRaisesRegex(RuntimeError, "copy failed"):
                    ShmPointerMMData(torch.ones(256))
        with self.assertRaises(FileNotFoundError):
            real_shared_memory(name=name)

    def test_concurrent_round_trip(self):
        def run(seed):
            source = torch.full((256, 512), seed, dtype=torch.uint8)
            ptr = ShmPointerMMData(source)
            result = pickle.loads(pickle.dumps(ptr)).materialize()
            return torch.equal(result, source)

        with ThreadPoolExecutor(max_workers=8) as pool:
            self.assertTrue(all(pool.map(run, range(32))))


if __name__ == "__main__":
    unittest.main()
