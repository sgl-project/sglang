import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")

from sglang.srt.hardware_backend.npu.utils import has_npu_a5_support


class TestNPUA5Capability(unittest.TestCase):
    def tearDown(self):
        has_npu_a5_support.cache_clear()

    def test_target_wheel_with_a5_sparse_attention_op_is_supported(self):
        custom_ops = SimpleNamespace(
            npu_kv_quant_sparse_attn_sharedkv=object(),
        )
        with patch.object(torch.ops, "custom", custom_ops, create=True):
            self.assertTrue(has_npu_a5_support())

    def test_target_wheel_without_a5_sparse_attention_op_is_not_supported(self):
        with patch.object(torch.ops, "custom", SimpleNamespace(), create=True):
            self.assertFalse(has_npu_a5_support())


if __name__ == "__main__":
    unittest.main()
