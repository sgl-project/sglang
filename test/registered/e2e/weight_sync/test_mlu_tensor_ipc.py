"""An MLU tensor shares storage across processes with different device indices."""

import os
import subprocess
import sys
import unittest

import torch

from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
from sglang.srt.utils.serialization import MultiprocessingSerializer
from sglang.test.ci.ci_register import register_mlu_ci
from sglang.test.test_utils import CustomTestCase

register_mlu_ci(est_time=30, suite="pr-test-2-mlu")


def _consume():
    tensor = MultiprocessingSerializer.deserialize(sys.stdin.buffer.read())
    assert tensor.device == torch.device("mlu:0")
    torch.testing.assert_close(tensor.cpu(), torch.tensor([1.0, 2.0]))
    tensor.add_(1)
    torch.mlu.synchronize(0)


class TestMLUTensorIPC(CustomTestCase):
    def test_tensor_ipc(self):
        monkey_patch_torch_reductions()
        tensor = torch.tensor([1.0, 2.0], device="mlu:1")
        torch.mlu.synchronize(1)

        # Keep the exporting tensor alive until the consumer exits.
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import runpy, sys; runpy.run_path(sys.argv[1])['_consume']()",
                __file__,
            ],
            input=MultiprocessingSerializer.serialize(tensor),
            env=dict(os.environ, CN_VISIBLE_DEVICES="1,0"),
            check=True,
            timeout=90,
        )

        # The consumer's in-place update must be visible in the original tensor.
        torch.testing.assert_close(tensor.cpu(), torch.tensor([2.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
