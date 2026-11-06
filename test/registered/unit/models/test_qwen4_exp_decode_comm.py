"""Qwen4-exp layers import the CUDA-only qwen4_decode module on CUDA only.

qwen4_decode pulls in custom_all_reduce_v2, which needs tvm_ffi; the XPU and
CPU installs do not ship it, so importing it there failed layer construction.
"""

import sys
import unittest
from unittest import mock

from sglang.srt.models import qwen4_exp as qwen4_exp_module
from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock
from sglang.test.ci.ci_register import register_cpu_ci, register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_xpu_ci(est_time=5, suite="stage-b-test-1-gpu-xpu")

_DECODE_MODULE = "sglang.srt.layers.moe.qwen4_decode"


class _BlockDecodeImport:
    """A meta path finder that fails any import of qwen4_decode."""

    def find_spec(self, name, path=None, target=None):
        if name == _DECODE_MODULE:
            raise AssertionError(f"{_DECODE_MODULE} imported off CUDA")
        return None


class TestQwen4ExpDecodeComm(CustomTestCase):
    def setUp(self):
        sys.modules.pop(_DECODE_MODULE, None)
        finder = _BlockDecodeImport()
        sys.meta_path.insert(0, finder)
        self.addCleanup(sys.meta_path.remove, finder)

    def test_moe_layer_off_cuda_skips_qwen4_decode(self):
        mlp = Qwen2MoeSparseMoeBlock.__new__(Qwen2MoeSparseMoeBlock)
        with mock.patch.object(qwen4_exp_module, "_is_cuda", False):
            self.assertIsNone(qwen4_exp_module._prepare_decode_moe_comm(mlp))
        self.assertNotIn(_DECODE_MODULE, sys.modules)

    def test_dense_layer_skips_qwen4_decode(self):
        with mock.patch.object(qwen4_exp_module, "_is_cuda", True):
            self.assertIsNone(qwen4_exp_module._prepare_decode_moe_comm(object()))
        self.assertNotIn(_DECODE_MODULE, sys.modules)


if __name__ == "__main__":
    unittest.main()
