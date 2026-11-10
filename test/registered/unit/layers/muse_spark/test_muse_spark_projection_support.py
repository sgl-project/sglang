"""Keep automatic B200 projection dispatch inside the validated TP8 envelope."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.kernels.ops.communication import muse_spark_all_gather as gather
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, stage="base-a", runner_config="cpu")


class TestMuseSparkProjectionSupport(CustomTestCase):
    def setUp(self):
        self.comm = Mock(spec=CustomAllReduceV2)
        self.comm.disabled = False
        self.comm.has_multicast = True
        self.comm.world_size = 8
        self.comm.max_push_size = 512 * 8192 * 2
        self.comm.config = SimpleNamespace(num_push_blocks=2)

    def supported(self, m=16, n=4096, capability=(10, 0), *, x=None, weight=None):
        x = torch.empty(m, 64, dtype=torch.bfloat16) if x is None else x
        weight = torch.empty(n, 64, dtype=torch.bfloat16) if weight is None else weight
        with (
            patch.object(torch.cuda, "get_device_capability", return_value=capability),
            patch.object(gather, "current_arch") as arch,
        ):
            arch.return_value.is_sm.side_effect = lambda *sm: sm == capability
            return gather.gemm_ag_supported(self.comm, x, weight)

    def test_b200_tp8_token_and_output_boundaries(self):
        for n in (4096, 8192):
            for m in (9, 16, 128):
                with self.subTest(m=m, n=n):
                    self.assertTrue(self.supported(m, n))
            self.assertFalse(self.supported(129, n))
        self.assertFalse(self.supported(16, 2048))
        self.assertFalse(self.supported(0))

    def test_existing_hardware_and_small_projection_paths(self):
        for tp in (4, 8):
            self.comm.world_size = tp
            self.assertTrue(self.supported(512, capability=(10, 3)))
            self.assertFalse(self.supported(513, capability=(10, 3)))
        for tp in (2, 4):
            self.comm.world_size = tp
            self.assertFalse(self.supported(9))
            self.assertTrue(self.supported(8))
        self.comm.world_size = 8
        for capability in ((9, 0), (10, 1), (12, 0)):
            self.assertFalse(self.supported(9, capability=capability))

    def test_dtype_shape_and_layout_fallbacks(self):
        self.assertFalse(self.supported(x=torch.empty(16, 64, dtype=torch.float16)))
        self.assertFalse(self.supported(weight=torch.empty(4096, 64)))
        self.assertFalse(self.supported(x=torch.empty(1, 16, 64, dtype=torch.bfloat16)))
        self.assertFalse(
            self.supported(weight=torch.empty(4096, 32, dtype=torch.bfloat16))
        )
        weight = torch.empty(64, 4096, dtype=torch.bfloat16).T
        self.assertFalse(self.supported(weight=weight))

    def test_communicator_and_workspace_fallbacks(self):
        for field in ("disabled", "has_multicast"):
            with patch.object(self.comm, field, field == "disabled"):
                self.assertFalse(self.supported())
        with patch.object(self.comm, "world_size", 1):
            self.assertFalse(self.supported())
        with patch.object(self.comm, "max_push_size", 16 * 4096 * 2 - 1):
            self.assertFalse(self.supported())
        with patch.object(self.comm.config, "num_push_blocks", 1):
            self.assertFalse(self.supported())


if __name__ == "__main__":
    unittest.main()
