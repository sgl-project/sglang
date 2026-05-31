from __future__ import annotations

import unittest

import torch

from sglang.jit_kernel.kv_canary.verify import CANARY_SLOT_BYTES
from sglang.srt.kv_canary.pool_patcher.buffer_alloc import alloc_canary_buf
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.fixtures import DEFAULT_DEVICE
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="extra-a", runner_config="1-gpu-small")


class TestAllocCanaryBuf(CustomTestCase):
    def test_alloc_canary_buf_shape_and_dtype(self) -> None:
        """Verify alloc_canary_buf returns a zeroed uint8 buffer of [num_slots, CANARY_SLOT_BYTES]."""
        buf = alloc_canary_buf(num_slots=8, device=DEFAULT_DEVICE)
        self.assertEqual(buf.shape, (8, CANARY_SLOT_BYTES))
        self.assertEqual(buf.dtype, torch.uint8)
        self.assertEqual(buf.device.type, DEFAULT_DEVICE.type)
        self.assertTrue(torch.equal(buf, torch.zeros_like(buf)))


if __name__ == "__main__":
    unittest.main()
