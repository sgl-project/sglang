"""CPU tests for DeepEP NVSHMEM QP depth configuration."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.layers.moe.token_dispatcher import deepep
from sglang.srt.layers.moe.utils import DeepEPMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDeepEPNVSHMEMQPDepth(CustomTestCase):
    def test_low_latency_buffer_configures_required_depth(self):
        state = SimpleNamespace(buffer=None)
        group = Mock()
        group.size.return_value = 4
        buffer = Mock()
        buffer.get_low_latency_rdma_size_hint.return_value = 1

        with (
            patch.dict(os.environ, {}, clear=False),
            patch.object(deepep.DeepEPBuffer, "_state", return_value=state),
            patch.object(deepep, "Buffer", buffer, create=True),
            patch.object(
                deepep.torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(multi_processor_count=160),
            ),
            patch.object(deepep, "_is_mnnvl_fabric_supported", return_value=False),
            patch.object(deepep, "get_cuda_version", return_value=(13, 0)),
        ):
            os.environ.pop("NVSHMEM_QP_DEPTH", None)

            deepep.DeepEPBuffer.get_deepep_buffer(
                group=group,
                hidden_size=7168,
                param_bytes=1,
                deepep_mode=DeepEPMode.LOW_LATENCY,
                num_max_dispatch_tokens_per_rank=1024,
                num_experts=256,
            )

            self.assertEqual(os.environ["NVSHMEM_QP_DEPTH"], "2050")

    def test_uses_default_depth_when_it_satisfies_dispatch_requirement(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NVSHMEM_QP_DEPTH", None)

            deepep._set_nvshmem_qp_depth(128)

            self.assertEqual(os.environ["NVSHMEM_QP_DEPTH"], "1024")

    def test_increases_depth_for_large_dispatch_capacity(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NVSHMEM_QP_DEPTH", None)

            deepep._set_nvshmem_qp_depth(1024)

            self.assertEqual(os.environ["NVSHMEM_QP_DEPTH"], "2050")

    def test_increases_user_depth_when_it_is_too_small(self):
        with patch.dict(os.environ, {"NVSHMEM_QP_DEPTH": "1024"}):
            deepep._set_nvshmem_qp_depth(1024)

            self.assertEqual(os.environ["NVSHMEM_QP_DEPTH"], "2050")

    def test_increases_user_depth_to_default_floor(self):
        with patch.dict(os.environ, {"NVSHMEM_QP_DEPTH": "512"}):
            deepep._set_nvshmem_qp_depth(128)

            self.assertEqual(os.environ["NVSHMEM_QP_DEPTH"], "1024")

    def test_preserves_larger_user_depth(self):
        with patch.dict(os.environ, {"NVSHMEM_QP_DEPTH": "4096"}):
            deepep._set_nvshmem_qp_depth(1024)

            self.assertEqual(os.environ["NVSHMEM_QP_DEPTH"], "4096")


if __name__ == "__main__":
    unittest.main()
