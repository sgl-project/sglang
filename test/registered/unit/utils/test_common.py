import unittest
from array import array

import torch

from sglang.srt.utils.common import flatten_arrays_to_int64_tensor
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFlattenArraysToInt64Tensor(CustomTestCase):
    """`flatten_arrays_to_int64_tensor` is invoked by `prepare_for_extend`
    to build the per-batch input_ids tensor (pinned, async H2D) from a
    list of array.array('q') per-req fill_ids slices. Tests the full
    matrix of (device, pin) the production code paths through.
    """

    DEVICES = ("cpu", "cuda")
    PIN_OPTIONS = (False, True)

    def _check(self, parts: list, expected: list[int]) -> None:
        for device in self.DEVICES:
            for pin in self.PIN_OPTIONS:
                with self.subTest(device=device, pin=pin):
                    out = flatten_arrays_to_int64_tensor(parts, device, pin)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    self.assertEqual(out.dtype, torch.int64)
                    self.assertEqual(out.device.type, device)
                    self.assertEqual(out.shape, (len(expected),))
                    self.assertEqual(out.cpu().tolist(), expected)

    def test_single_part(self):
        parts = [array("q", [1, 2, 3, 4, 5])]
        self._check(parts, [1, 2, 3, 4, 5])

    def test_multiple_parts(self):
        parts = [
            array("q", [10, 20, 30]),
            array("q", [100, 200]),
            array("q", [1000]),
        ]
        self._check(parts, [10, 20, 30, 100, 200, 1000])


if __name__ == "__main__":
    unittest.main()
