"""WeightChecker checksum on a non-CUDA device (CPU stands in for TPU)."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.utils.weight_checker import ChecksumInfo, WeightChecker
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_scope, published_topology

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestChecksumWithoutCuda(CustomTestCase):
    def test_checksum_does_not_require_cuda(self):
        """The checksum action must work on hosts without CUDA, e.g. TPU engines
        checked by an RL trainer after every weight update."""
        model = nn.Linear(4, 4)
        enter_scope(self, published_topology())
        checker = WeightChecker(get_model=lambda: model)
        with patch.object(
            torch.cuda, "synchronize", side_effect=RuntimeError("no CUDA device")
        ):
            out = checker.handle("checksum", role="target")
        info = ChecksumInfo.model_validate(out)
        self.assertEqual(set(info.checksums), {"weight", "bias"})


if __name__ == "__main__":
    unittest.main()
