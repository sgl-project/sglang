"""Unit tests for sglang.srt.utils.device — no server, no model loading."""

import unittest
from unittest import mock

import torch

from sglang.srt.utils.device import get_xpu_memory_capacity
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestGetXpuMemoryCapacity(CustomTestCase):
    """`get_xpu_memory_capacity()` feeds the mem-fraction-static derivation at
    server startup.

    `torch.xpu.mem_get_info()` relies on Level Zero Sysman, which is
    unavailable in some environments (e.g. WSL2 GPU-PV paravirtualized
    iGPU), where it raises RuntimeError and would crash startup. The
    function must fall back to `get_device_properties().total_memory`
    (Level Zero core API) instead.
    """

    def test_normal_path_reports_total_from_mem_get_info(self):
        fake_xpu = mock.Mock()
        fake_xpu.is_available.return_value = True
        # mem_get_info() returns (free, total) in bytes; the function reports
        # the total converted to MB.
        fake_xpu.mem_get_info.return_value = (100 << 20, 2048 << 20)
        with mock.patch.object(torch, "xpu", fake_xpu):
            self.assertEqual(get_xpu_memory_capacity(), 2048)
        fake_xpu.get_device_properties.assert_not_called()

    def test_falls_back_to_core_api_when_sysman_unavailable(self):
        fake_xpu = mock.Mock()
        fake_xpu.is_available.return_value = True
        fake_xpu.mem_get_info.side_effect = RuntimeError(
            "The device (Intel(R) Graphics) doesn't support querying the "
            "available free memory."
        )
        fake_xpu.current_device.return_value = 0
        # 16841 MB is the total memory observed on a real WSL2 GPU-PV iGPU
        # (Intel(R) Graphics [0xb081]) where this crash was reproduced.
        fake_xpu.get_device_properties.return_value = mock.Mock(
            total_memory=16841 << 20
        )
        with mock.patch.object(torch, "xpu", fake_xpu):
            self.assertEqual(get_xpu_memory_capacity(), 16841)
        fake_xpu.get_device_properties.assert_called_once_with(0)

    def test_raises_value_error_when_xpu_unavailable(self):
        fake_xpu = mock.Mock()
        fake_xpu.is_available.return_value = False
        with mock.patch.object(torch, "xpu", fake_xpu):
            with self.assertRaises(ValueError):
                get_xpu_memory_capacity()

    def test_raises_runtime_error_when_torch_xpu_missing(self):
        # Simulates torch builds without the xpu module (AttributeError from
        # the torch.xpu attribute access); the function must translate it to
        # the documented RuntimeError.
        fake_xpu = mock.Mock()
        fake_xpu.is_available.side_effect = AttributeError("no torch.xpu")
        with mock.patch.object(torch, "xpu", fake_xpu):
            with self.assertRaises(RuntimeError):
                get_xpu_memory_capacity()


if __name__ == "__main__":
    unittest.main()
