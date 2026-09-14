import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.distributed import parallel_state
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")


class _FakeHcclOptions:
    def __init__(self):
        self.hccl_config = None


_FAKE_TORCH_NPU = SimpleNamespace(
    _C=SimpleNamespace(
        _distributed_c10d=SimpleNamespace(
            ProcessGroupHCCL=SimpleNamespace(Options=_FakeHcclOptions)
        )
    )
)


class TestNpuDcpHcclOptions(CustomTestCase):
    @patch.object(parallel_state, "_is_npu", True)
    def test_dcp_uses_group_local_buffer_size(self):
        with (
            patch.dict(sys.modules, {"torch_npu": _FAKE_TORCH_NPU}),
            patch.dict(
                os.environ,
                {
                    "HCCL_BUFFSIZE": "2000",
                    "DEEPEP_HCCL_BUFFSIZE": "2000",
                    "DCP_HCCL_BUFFSIZE": "200",
                },
                clear=True,
            ),
        ):
            options = parallel_state.get_torch_distributed_pg_options("dcp")
        self.assertEqual(options.hccl_config, {"hccl_buffer_size": 200})

    @patch.object(parallel_state, "_is_npu", True)
    def test_dcp_unset_preserves_default_group_behavior(self):
        with patch.dict(os.environ, {"HCCL_BUFFSIZE": "2000"}, clear=True):
            self.assertIsNone(parallel_state.get_torch_distributed_pg_options("dcp"))

    @patch.object(parallel_state, "_is_npu", True)
    def test_moe_keeps_deepep_buffer_size(self):
        with (
            patch.dict(sys.modules, {"torch_npu": _FAKE_TORCH_NPU}),
            patch.dict(
                os.environ,
                {
                    "HCCL_BUFFSIZE": "200",
                    "DEEPEP_HCCL_BUFFSIZE": "2000",
                    "DCP_HCCL_BUFFSIZE": "128",
                },
                clear=True,
            ),
        ):
            options = parallel_state.get_torch_distributed_pg_options("moe_ep")
        self.assertEqual(options.hccl_config, {"hccl_buffer_size": 2000})

    @patch.object(parallel_state, "_is_npu", True)
    def test_rejects_non_positive_dcp_buffer_size(self):
        with (
            patch.dict(sys.modules, {"torch_npu": _FAKE_TORCH_NPU}),
            patch.dict(os.environ, {"DCP_HCCL_BUFFSIZE": "0"}, clear=True),
            self.assertRaisesRegex(ValueError, "must be positive"),
        ):
            parallel_state.get_torch_distributed_pg_options("dcp")


if __name__ == "__main__":
    unittest.main()
