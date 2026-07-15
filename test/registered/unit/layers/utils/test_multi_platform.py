# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC

"""Unit tests for srt/layers/utils/multi_platform.py — CPU-only, no server."""

from sglang.test.ci.ci_register import register_cpu_ci

# Register CPU CI with estimated time and suite
register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.layers.utils.multi_platform import MultiPlatformOp
from sglang.test.test_utils import CustomTestCase


# Concrete subclass for testing
class DummyOp(MultiPlatformOp):
    def forward_native(self, x: str) -> str:
        return x + "native"

    def forward_cuda(self, x: str) -> str:
        return x + "cuda"

    def forward_npu(self, x: str) -> str:
        return x + "npu"

    def forward_cpu(self, x: str) -> str:
        return x + "cpu"

    def forward_custom_gpu(self, x: str) -> str:
        return x + "custom"

    def forward_xpu(self, x: str) -> str:
        return x + "xpu"


class TestMultiPlatformOp(CustomTestCase):

    def setUp(self):
        # Clear OOT registry before each test to ensure isolation
        MultiPlatformOp._oot_forward_registry.clear()

    @patch("sglang.srt.layers.utils.multi_platform._is_cpu", False)
    @patch("sglang.srt.layers.utils.multi_platform._is_cpu_amx_available", False)
    @patch("sglang.srt.layers.utils.multi_platform._is_cuda", True)
    @patch("sglang.srt.layers.utils.multi_platform._is_hip", False)
    @patch("sglang.srt.layers.utils.multi_platform._is_npu", False)
    @patch("sglang.srt.layers.utils.multi_platform._is_xpu", False)
    def test_dispatch_cuda(self):
        # Should dispatch to forward_cuda when _is_cuda is True
        op = DummyOp()
        self.assertEqual(op("input_"), "input_cuda")

    @patch("sglang.srt.layers.utils.multi_platform._is_cpu", False)
    @patch("sglang.srt.layers.utils.multi_platform._is_cpu_amx_available", False)
    @patch("sglang.srt.layers.utils.multi_platform._is_cuda", False)
    @patch("sglang.srt.layers.utils.multi_platform._is_hip", True)
    @patch("sglang.srt.layers.utils.multi_platform._is_npu", False)
    @patch("sglang.srt.layers.utils.multi_platform._is_xpu", False)
    def test_dispatch_hip(self) -> None:
        # Should dispatch to forward_cuda when _is_hip is True (ROCm)
        op = DummyOp()
        self.assertEqual(op("input_"), "input_cuda")

    @patch("sglang.srt.layers.utils.multi_platform._is_cpu", False)
    @patch("sglang.srt.layers.utils.multi_platform._is_cpu_amx_available", False)
    @patch("sglang.srt.layers.utils.multi_platform._is_cuda", False)
    @patch("sglang.srt.layers.utils.multi_platform._is_hip", False)
    @patch("sglang.srt.layers.utils.multi_platform._is_npu", True)
    @patch("sglang.srt.layers.utils.multi_platform._is_xpu", False)
    def test_dispatch_npu(self):
        # Should dispatch to forward_npu when _is_npu is True
        op = DummyOp()
        self.assertEqual(op("input_"), "input_npu")

    @patch("sglang.srt.layers.utils.multi_platform._is_cpu", False)
    @patch("sglang.srt.layers.utils.multi_platform._is_cpu_amx_available", False)
    @patch("sglang.srt.layers.utils.multi_platform._is_cuda", False)
    @patch("sglang.srt.layers.utils.multi_platform._is_hip", False)
    @patch("sglang.srt.layers.utils.multi_platform._is_npu", False)
    @patch("sglang.srt.layers.utils.multi_platform._is_xpu", True)
    def test_dispatch_xpu(self) -> None:
        # Should dispatch to forward_xpu when _is_xpu is True (Intel GPU)
        op = DummyOp()
        self.assertEqual(op("input_"), "input_xpu")

    @patch("sglang.srt.layers.utils.multi_platform._is_cuda", False)
    @patch("sglang.srt.layers.utils.multi_platform._is_hip", False)
    @patch("sglang.srt.layers.utils.multi_platform._is_npu", False)
    @patch("sglang.srt.layers.utils.multi_platform._is_xpu", False)
    @patch("sglang.srt.layers.utils.multi_platform._is_cpu", True)
    @patch("sglang.srt.layers.utils.multi_platform._is_cpu_amx_available", True)
    def test_dispatch_cpu_amx(self):
        # Should dispatch to forward_cpu when _is_cpu and amx available
        op = DummyOp()
        self.assertEqual(op("input_"), "input_cpu")

    @patch("sglang.srt.layers.utils.multi_platform._is_cuda", False)
    @patch("sglang.srt.layers.utils.multi_platform._is_hip", False)
    @patch("sglang.srt.layers.utils.multi_platform._is_npu", False)
    @patch("sglang.srt.layers.utils.multi_platform._is_xpu", False)
    @patch("sglang.srt.layers.utils.multi_platform._is_cpu", True)
    @patch("sglang.srt.layers.utils.multi_platform._is_cpu_amx_available", False)
    def test_dispatch_cpu_no_amx_fallback(self):
        # Should fallback to forward_native when AMX is not available on CPU
        op = DummyOp()
        self.assertEqual(op("input_"), "input_native")

    @patch("sglang.srt.layers.utils.multi_platform.current_platform")
    def test_dispatch_oot_method_lookup(self, mock_platform):
        # Mock Out-of-Tree platform with custom dispatch key
        mock_platform.is_out_of_tree.return_value = True
        mock_platform.get_dispatch_key_name.return_value = "custom_gpu"

        op = DummyOp()
        # Should look up and bind 'forward_custom_gpu' method
        self.assertEqual(op("input_"), "input_custom")

    @patch("sglang.srt.layers.utils.multi_platform.current_platform")
    def test_dispatch_oot_registry(self, mock_platform):
        # Mock Out-of-Tree platform with custom dispatch key
        mock_platform.is_out_of_tree.return_value = True
        mock_platform.get_dispatch_key_name.return_value = "custom_gpu"

        # Define a custom OOT forward function
        def oot_forward_fn(self: DummyOp, x: str) -> str:
            return x + "registered_oot"

        # Register it for DummyOp on 'custom_gpu'
        MultiPlatformOp.register_oot_forward(DummyOp, oot_forward_fn, "custom_gpu")

        op = DummyOp()
        # Registry should take precedence over method lookup
        self.assertEqual(op("input_"), "input_registered_oot")

    @patch("sglang.srt.layers.utils.multi_platform._is_cuda", True)
    def test_torch_compile_state_machine(self):
        op = DummyOp()
        # Initially dispatched to CUDA
        self.assertEqual(op("input_"), "input_cuda")
        self.assertFalse(op.is_torch_compile)

        # Enter compile mode
        op.enter_torch_compile(num_tokens=1)
        self.assertTrue(op.is_torch_compile)
        # Should temporarily switch to native forward
        self.assertEqual(op("input_"), "input_native")

        # Exit compile mode
        op.leave_torch_compile()
        self.assertFalse(op.is_torch_compile)
        # Should restore back to CUDA
        self.assertEqual(op("input_"), "input_cuda")

    @patch("sglang.srt.layers.utils.multi_platform._is_cuda", True)
    def test_torch_compile_idempotency(self):
        op = DummyOp()
        
        # Enter multiple times
        op.enter_torch_compile(num_tokens=1)
        orig_backup = op._original_forward_method
        op.enter_torch_compile(num_tokens=1)
        # Second enter should be a noop and not corrupt backup
        self.assertEqual(op._original_forward_method, orig_backup)

        # Leave multiple times
        op.leave_torch_compile()
        op.leave_torch_compile()  # should not crash
        self.assertFalse(op.is_torch_compile)
        self.assertIsNone(op._original_forward_method)


if __name__ == "__main__":
    unittest.main()
