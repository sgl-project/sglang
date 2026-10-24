import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pynvml
import torch

from sglang.srt.utils import common, cuda_vmm_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMnnvlFabricDetection(unittest.TestCase):
    def setUp(self) -> None:
        common.is_mnnvl_fabric_device.cache_clear()
        self.addCleanup(common.is_mnnvl_fabric_device.cache_clear)

        self.nvml = Mock(spec=pynvml)
        self.fabric = pynvml.c_nvmlGpuFabricInfo_v3_t()
        self.fabric.state = cuda_vmm_utils._NVML_GPU_FABRIC_STATE_COMPLETED
        self.fabric.clusterUuid[:] = b"0123456789abcdef"
        self.fabric.cliqueId = 7
        self.fabric.status = pynvml.NVML_SUCCESS
        self.nvml.NVML_SUCCESS = pynvml.NVML_SUCCESS
        self.resolve_nvml_index = torch.cuda._get_nvml_device_index

        for patcher in (
            patch.object(common, "is_cuda", return_value=True),
            patch.object(cuda_vmm_utils, "pynvml", self.nvml),
            patch.object(
                cuda_vmm_utils,
                "_NVML_GPU_FABRIC_INFO_V3_TYPE",
                return_value=self.fabric,
            ),
            patch.object(torch.cuda, "_get_nvml_device_index", return_value=3),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_joined_fabric_is_independent_of_device_name(self) -> None:
        for name in ("CF-NG-GBZZ3-O-L", "NVIDIA GB200", "NVIDIA GB300"):
            with (
                self.subTest(name=name),
                patch.object(torch.cuda, "get_device_name", return_value=name),
            ):
                common.is_mnnvl_fabric_device.cache_clear()
                self.assertTrue(common.is_mnnvl_fabric_device())

    def test_device_name_does_not_override_unjoined_fabric(self) -> None:
        for state in (0, 1, 2):
            with (
                self.subTest(state=state),
                patch.object(
                    torch.cuda, "get_device_name", return_value="NVIDIA GB300"
                ),
            ):
                common.is_mnnvl_fabric_device.cache_clear()
                self.fabric.state = state
                self.assertFalse(common.is_mnnvl_fabric_device())

    def test_unsuccessful_fabric_status_returns_false(self) -> None:
        self.fabric.status = pynvml.NVML_ERROR_UNKNOWN
        self.assertFalse(common.is_mnnvl_fabric_device())

    def test_legacy_fabric_query_accepts_joined_device(self) -> None:
        fabric = pynvml.c_nvmlGpuFabricInfo_t()
        fabric.state = cuda_vmm_utils._NVML_GPU_FABRIC_STATE_COMPLETED
        fabric.status = pynvml.NVML_SUCCESS
        fabric.clusterUuid[:] = b"0123456789abcdef"
        fabric.cliqueId = 7
        self.nvml.c_nvmlGpuFabricInfo_t.return_value = fabric
        with patch.object(cuda_vmm_utils, "_NVML_GPU_FABRIC_INFO_V3_TYPE", None):
            self.assertTrue(common.is_mnnvl_fabric_device())

    def test_fabric_probe_uses_visible_device_mapping(self) -> None:
        cuda_vmm_utils.is_gpu_fabric_ready(torch.device("cuda", 0))
        self.nvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(3)

    def test_reordered_visible_devices_select_the_correct_gpu(self) -> None:
        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3,1"}),
            patch.object(torch.cuda, "_get_nvml_device_index", self.resolve_nvml_index),
        ):
            common.is_mnnvl_fabric_device()
        self.nvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(3)

    def test_uuid_visible_devices_select_the_correct_gpu(self) -> None:
        device_uuids = [
            "GPU-00000000-0000-0000-0000-000000000000",
            "GPU-11111111-1111-1111-1111-111111111111",
        ]
        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": device_uuids[1]}),
            patch.object(torch.cuda, "_get_nvml_device_index", self.resolve_nvml_index),
            patch.object(
                torch.cuda, "_raw_device_uuid_nvml", return_value=device_uuids
            ),
        ):
            common.is_mnnvl_fabric_device()
        self.nvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(1)

    def test_nvml_query_failure_returns_false(self) -> None:
        self.nvml.nvmlDeviceGetGpuFabricInfoV.side_effect = (
            pynvml.NVMLError_NotSupported
        )
        self.assertFalse(common.is_mnnvl_fabric_device())

    def test_nvml_is_released_after_query_failure(self) -> None:
        self.nvml.nvmlDeviceGetGpuFabricInfoV.side_effect = (
            pynvml.NVMLError_NotSupported
        )
        common.is_mnnvl_fabric_device()
        self.nvml.nvmlShutdown.assert_called_once_with()

    def test_missing_nvml_returns_false(self) -> None:
        with patch.object(cuda_vmm_utils, "pynvml", None):
            self.assertFalse(common.is_mnnvl_fabric_device())

    def test_non_cuda_device_returns_false(self) -> None:
        with patch.object(common, "is_cuda", return_value=False):
            self.assertFalse(common.is_mnnvl_fabric_device())

    def test_non_cuda_device_does_not_initialize_nvml(self) -> None:
        with patch.object(common, "is_cuda", return_value=False):
            common.is_mnnvl_fabric_device()
        self.nvml.nvmlInit.assert_not_called()

    def test_cross_node_dcp_accepts_joined_oem_device(self) -> None:
        with (
            patch.object(
                common, "get_platform", return_value=SimpleNamespace(is_sm100=True)
            ),
            patch.object(torch.cuda, "get_device_name", return_value="CF-NG-GBZZ3-O-L"),
        ):
            self.assertTrue(
                common.is_fi_a2a_supported(dcp_size=8, tp_size=8, pp_size=1, nnodes=2)
            )

    def test_cross_node_dcp_rejects_unjoined_device(self) -> None:
        self.fabric.state = 0
        with patch.object(
            common, "get_platform", return_value=SimpleNamespace(is_sm100=True)
        ):
            self.assertFalse(
                common.is_fi_a2a_supported(dcp_size=8, tp_size=8, pp_size=1, nnodes=2)
            )

    def test_node_local_dcp_remains_supported_without_fabric(self) -> None:
        self.fabric.state = 0
        with patch.object(
            common, "get_platform", return_value=SimpleNamespace(is_sm100=True)
        ):
            self.assertTrue(
                common.is_fi_a2a_supported(dcp_size=4, tp_size=8, pp_size=1, nnodes=2)
            )


if __name__ == "__main__":
    unittest.main()
