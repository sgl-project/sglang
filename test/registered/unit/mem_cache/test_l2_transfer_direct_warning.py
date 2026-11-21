"""The direct-backend metadata warning is raised on CUDA and skipped on ROCm."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache import l2_transfer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDirectBackendWarning(CustomTestCase):
    def _construct(self, *, hip, fill=False, io_backend="direct"):
        l2_transfer._warn_direct_without_metadata_kernels.cache_clear()
        with (
            patch.object(l2_transfer, "device_module", torch.cuda),
            patch.object(torch.cuda, "Stream"),
            patch.object(l2_transfer, "is_hip", return_value=hip),
            envs.SGLANG_ENABLE_MLP_SYNC_METADATA_FILL.override(fill),
        ):
            l2_transfer.L2TransferEngine(io_backend)

    def test_warns_on_cuda(self):
        with self.assertLogs(l2_transfer.logger, level="WARNING"):
            self._construct(hip=False)

    def test_silent_on_rocm(self):
        with self.assertNoLogs(l2_transfer.logger, level="WARNING"):
            self._construct(hip=True)

    def test_silent_with_metadata_kernels_or_other_backend(self):
        with self.assertNoLogs(l2_transfer.logger, level="WARNING"):
            self._construct(hip=False, fill=True)
            self._construct(hip=False, io_backend="kernel")


if __name__ == "__main__":
    unittest.main()
