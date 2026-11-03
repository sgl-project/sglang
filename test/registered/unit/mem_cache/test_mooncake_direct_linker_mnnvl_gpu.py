"""Unit tests for Mooncake Direct Linker MNNVL GPU identity resolution."""

import os
import unittest
from unittest import mock

from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    _configure_mnnvl_physical_gpu,
    _resolve_mnnvl_physical_gpu,
    _uses_mnnvl,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestMooncakeMnnvlProtocolDetection(CustomTestCase):
    def test_detects_store_mnnvl_protocols(self):
        self.assertTrue(_uses_mnnvl({"protocol": "mnnvl"}))
        self.assertTrue(_uses_mnnvl({"protocol": "NVLINK_HOST"}))

    def test_classic_nvlink_protocol_is_not_store_mnnvl(self):
        self.assertFalse(_uses_mnnvl({"protocol": "nvlink"}))

    def test_explicit_protocol_overrides_environment(self):
        with mock.patch.dict(os.environ, {"MOONCAKE_PROTOCOL": "mnnvl"}):
            self.assertFalse(_uses_mnnvl({"protocol": "rdma"}))

    def test_uses_protocol_environment_when_config_is_omitted(self):
        with mock.patch.dict(os.environ, {"MOONCAKE_PROTOCOL": "mnnvl"}):
            self.assertTrue(_uses_mnnvl({}))


class TestMooncakeMnnvlPhysicalGpuResolution(CustomTestCase):
    def test_resolves_tp2_numeric_visible_devices(self):
        self.assertEqual(_resolve_mnnvl_physical_gpu(0, "1,3", None), 1)
        self.assertEqual(_resolve_mnnvl_physical_gpu(1, "1,3", None), 3)

    def test_resolves_arbitrary_tp_size_and_device_order(self):
        visible_devices = "7,2,5,1,6,0,4,3"
        resolved = [
            _resolve_mnnvl_physical_gpu(rank, visible_devices, None)
            for rank in range(8)
        ]
        self.assertEqual(resolved, [7, 2, 5, 1, 6, 0, 4, 3])

    def test_uses_logical_device_when_visibility_is_unrestricted(self):
        self.assertEqual(_resolve_mnnvl_physical_gpu(3, None, None), 3)

    def test_accepts_matching_explicit_physical_gpu(self):
        self.assertEqual(_resolve_mnnvl_physical_gpu(1, "1,3", "3"), 3)

    def test_explicit_physical_gpu_is_authoritative_without_visibility_map(self):
        self.assertEqual(_resolve_mnnvl_physical_gpu(0, None, "3"), 3)

    def test_rejects_container_wide_gpu_value_that_conflicts_with_worker(self):
        with self.assertRaisesRegex(RuntimeError, "conflicts with CUDA logical"):
            _resolve_mnnvl_physical_gpu(1, "0,1", "0")

    def test_rejects_invalid_explicit_physical_gpu(self):
        for value in ("gpu0", "-1"):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(RuntimeError, "must be a non-negative integer"),
            ):
                _resolve_mnnvl_physical_gpu(0, "0,1", value)

    def test_rejects_logical_device_outside_visibility(self):
        with self.assertRaisesRegex(RuntimeError, "outside CUDA_VISIBLE_DEVICES"):
            _resolve_mnnvl_physical_gpu(2, "1,3", None)

    def test_uuid_and_mig_visibility_require_explicit_physical_gpu(self):
        for visible_device in ("GPU-1234", "MIG-1234"):
            with (
                self.subTest(visible_device=visible_device),
                self.assertRaisesRegex(RuntimeError, "Set MC_PHYSICAL_GPU explicitly"),
            ):
                _resolve_mnnvl_physical_gpu(0, visible_device, None)
            self.assertEqual(_resolve_mnnvl_physical_gpu(0, visible_device, "2"), 2)


class TestMooncakeMnnvlPhysicalGpuConfiguration(CustomTestCase):
    def setUp(self):
        self.environment = mock.patch.dict(os.environ, {}, clear=False)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "MC_PHYSICAL_GPU",
            "MOONCAKE_PROTOCOL",
        ):
            os.environ.pop(name, None)

    def test_configures_each_tp2_worker_independently(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
        for logical_gpu, expected_physical_gpu in ((0, 1), (1, 3)):
            os.environ.pop("MC_PHYSICAL_GPU", None)
            with mock.patch("torch.cuda.current_device", return_value=logical_gpu):
                physical_gpu = _configure_mnnvl_physical_gpu({"protocol": "mnnvl"})
            self.assertEqual(physical_gpu, expected_physical_gpu)
            self.assertEqual(os.environ["MC_PHYSICAL_GPU"], str(expected_physical_gpu))

    def test_does_not_touch_non_mnnvl_protocols(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        with mock.patch(
            "torch.cuda.current_device",
            side_effect=AssertionError("CUDA lookup should not run"),
        ):
            self.assertIsNone(_configure_mnnvl_physical_gpu({"protocol": "rdma"}))
        self.assertNotIn("MC_PHYSICAL_GPU", os.environ)


if __name__ == "__main__":
    unittest.main()
