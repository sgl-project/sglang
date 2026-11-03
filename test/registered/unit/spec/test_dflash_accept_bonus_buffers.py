"""DFlash accept/bonus output buffers: slot alternation and fixed addresses."""

import unittest

import torch

from sglang.srt.speculative.dflash_worker_v2 import _AcceptBonusBuffers
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_KW = dict(device=torch.device("cpu"), block_size=8)


def _ptrs(bufs):
    return [t.data_ptr() for t in bufs]


class TestAcceptBonusBuffers(CustomTestCase):
    def test_slots_alternate_and_keep_addresses(self):
        buffers = _AcceptBonusBuffers()
        first, second, third = (buffers.take(4, **_KW) for _ in range(3))
        # accept_len is step-local and shared; the handed-off outputs alternate.
        self.assertEqual(first[0].data_ptr(), second[0].data_ptr())
        for a, b in zip(first[1:], second[1:]):
            self.assertNotEqual(a.data_ptr(), b.data_ptr())
        self.assertEqual(_ptrs(first), _ptrs(third))
        self.assertEqual(
            [tuple(t.shape) for t in first], [(4,), (4,), (4,), (4, 8), (4,)]
        )
        self.assertEqual(
            [t.dtype for t in first],
            [torch.int32, torch.int32, torch.int64, torch.int64, torch.int64],
        )

    def test_smaller_batch_reuses_and_larger_batch_doubles(self):
        buffers = _AcceptBonusBuffers()
        buffers.take(4, **_KW)
        accept_len = buffers.accept_len
        buffers.take(3, **_KW)
        self.assertIs(buffers.accept_len, accept_len)
        self.assertEqual(buffers.cap, 4)
        buffers.take(5, **_KW)
        self.assertEqual(buffers.cap, 8)
        buffers.take(20, **_KW)
        self.assertEqual(buffers.cap, 20)


if __name__ == "__main__":
    unittest.main()
