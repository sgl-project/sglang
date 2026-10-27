"""Unit tests for srt/distributed/device_communicators/quick_all_reduce.py"""

import unittest

import torch

from sglang.srt.distributed.device_communicators.quick_all_reduce import (
    MB,
    QuickAllReduce,
    QuickReduceRegime,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeFlyDSLInt4:
    def __init__(self):
        self.calls = 0
        self.closed = False

    def allreduce(self, inp, out):
        self.calls += 1
        out.copy_(inp)

    def close(self):
        self.closed = True


def _make_flydsl_quick_reduce(fake: _FakeFlyDSLInt4) -> QuickAllReduce:
    quick_reduce = QuickAllReduce.__new__(QuickAllReduce)
    quick_reduce.disabled = False
    quick_reduce._flydsl_int4 = fake
    quick_reduce.world_size = 4
    quick_reduce.qr_quant_level = QuickReduceRegime.INT4
    quick_reduce.use_fp16_kernels = 1
    quick_reduce.qr_max_size = 2048 * MB
    return quick_reduce


class TestQuickAllReduceFlyDSLInt4(CustomTestCase):
    def test_flydsl_int4_takes_only_contiguous_aligned_bf16(self):
        """Input the FlyDSL INT4 kernel cannot take must not reach it."""
        numel = MB  # 2 MiB of 16-bit data, the INT4 minimum size at TP4
        fake = _FakeFlyDSLInt4()
        quick_reduce = _make_flydsl_quick_reduce(fake)

        inp = torch.ones(numel, dtype=torch.bfloat16)
        self.assertTrue(quick_reduce.should_quick_allreduce(inp))
        out = quick_reduce.quick_all_reduce(inp)
        self.assertEqual(fake.calls, 1)
        self.assertTrue(torch.equal(out, inp))

        unaligned = torch.empty(numel + 4, dtype=torch.bfloat16)[4:]
        self.assertEqual(unaligned.data_ptr() % 16, 8)
        rejected = {
            "fp16": torch.empty(numel, dtype=torch.float16),
            "weakly_contiguous": torch.empty(2, numel // 2, dtype=torch.bfloat16).t(),
            "8_byte_offset": unaligned,
        }
        for name, tensor in rejected.items():
            with self.subTest(name):
                self.assertFalse(quick_reduce.should_quick_allreduce(tensor))

    def test_close_releases_flydsl_int4(self):
        fake = _FakeFlyDSLInt4()
        quick_reduce = _make_flydsl_quick_reduce(fake)

        quick_reduce.close()

        self.assertTrue(fake.closed)
        self.assertIsNone(quick_reduce._flydsl_int4)
        self.assertTrue(quick_reduce.disabled)


if __name__ == "__main__":
    unittest.main()
