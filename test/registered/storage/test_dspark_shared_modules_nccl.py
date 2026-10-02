"""Native embedding/head replication over two actual NCCL devices."""

import unittest

import torch
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.dspark_shared_modules_utils import launch_shared_module_test
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="2-gpu")


class TestDSparkSharedModulesNCCL(CustomTestCase):
    @unittest.skipUnless(torch.cuda.device_count() >= 2, "two CUDA devices required")
    def test_native_shards_and_startup_failure(self):
        launch_shared_module_test(self, tp_size=1, pp_size=2, cuda=True)


if __name__ == "__main__":
    unittest.main()
