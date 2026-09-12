"""CPU tensors must survive the reduce_tensor monkey patch.

monkey_patch_torch_reductions replaces the reducer for every tensor, not
only the CUDA ones, so anything that ships a CPU tensor through torch
multiprocessing in a patched process goes through _reduce_tensor_modified.
"""

import unittest
from unittest import mock

import torch
from torch.multiprocessing import reductions

from sglang.srt.utils import MultiprocessingSerializer, patch_torch
from sglang.srt.utils.patch_torch import (
    _REDUCE_TENSOR_ARG_DEVICE_INDEX,
    monkey_patch_torch_reductions,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestReduceTensorCpuGuard(unittest.TestCase):
    def test_cpu_tensor_round_trips_after_patching(self):
        """The reduced form of a CPU tensor has no device slot to rewrite."""
        monkey_patch_torch_reductions()
        expected = torch.arange(8, dtype=torch.float32)

        payload = MultiprocessingSerializer.serialize(
            {"weight": expected}, output_str=True
        )
        restored = MultiprocessingSerializer.deserialize(payload)

        self.assertTrue(torch.equal(restored["weight"], expected))

    def test_device_index_still_rewritten_for_cuda_shaped_args(self):
        """The guard must not disarm the patch on the form it targets."""
        cuda_shaped = ("cls", "storage", 0, 8, 0, "handle", 3, "event", False)
        self.assertGreater(len(cuda_shaped), _REDUCE_TENSOR_ARG_DEVICE_INDEX)

        with mock.patch.object(
            reductions,
            "_reduce_tensor_original",
            create=True,
            return_value=("rebuild", cuda_shaped),
        ), mock.patch.object(
            patch_torch, "_device_to_uuid", side_effect=lambda device: f"uuid-{device}"
        ):
            _, rewritten = patch_torch._reduce_tensor_modified(object())

        self.assertEqual(rewritten[_REDUCE_TENSOR_ARG_DEVICE_INDEX], "uuid-3")
        self.assertEqual(
            rewritten[:_REDUCE_TENSOR_ARG_DEVICE_INDEX],
            cuda_shaped[:_REDUCE_TENSOR_ARG_DEVICE_INDEX],
        )
        self.assertEqual(
            rewritten[_REDUCE_TENSOR_ARG_DEVICE_INDEX + 1 :],
            cuda_shaped[_REDUCE_TENSOR_ARG_DEVICE_INDEX + 1 :],
        )


if __name__ == "__main__":
    unittest.main()
