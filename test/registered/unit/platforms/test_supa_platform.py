"""Unit tests for the Biren SUPA SRT platform and platform discovery."""

from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.platforms import _resolve_platform
from sglang.srt.platforms.device_mixin import DeviceCapability, PlatformEnum
from sglang.srt.platforms.supa import SupaSRTPlatform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _mock_supa_props(major=10, minor=4, total_memory=64 * 1024**3):
    # SimpleNamespace (not MagicMock) so that missing attributes such as
    # ``uuid`` genuinely raise, matching BR1xx device properties.
    return SimpleNamespace(
        major=major,
        minor=minor,
        total_memory=total_memory,
        multi_processor_count=128,
    )


class TestSupaPlatformIdentity(CustomTestCase):
    def test_identity_flags(self):
        platform = SupaSRTPlatform()
        self.assertTrue(platform.is_out_of_tree())
        self.assertFalse(platform.is_cuda())
        self.assertFalse(platform.is_cuda_alike())
        self.assertEqual(platform.device_name, "supa")
        self.assertEqual(platform.device_type, "supa")
        self.assertEqual(platform._enum, PlatformEnum.OOT)

    def test_capabilities_are_conservative(self):
        platform = SupaSRTPlatform()
        self.assertFalse(platform.supports_fp8())
        self.assertFalse(platform.support_cuda_graph())
        self.assertFalse(platform.support_piecewise_cuda_graph())

    def test_distributed_backend_is_sccl(self):
        self.assertEqual(SupaSRTPlatform().get_torch_distributed_backend_str(), "sccl")

    def test_pin_memory_available_for_supa_targets(self):
        platform = SupaSRTPlatform()
        self.assertTrue(platform.is_pin_memory_available())
        self.assertTrue(platform.is_pin_memory_available(device="supa"))
        self.assertFalse(platform.is_pin_memory_available(device="cpu"))


class TestSupaDeviceOperations(CustomTestCase):
    def setUp(self):
        self._supa_patch = patch("sglang.srt.platforms.supa._supa")
        self._supa_fn = self._supa_patch.start()
        # ``_supa()`` is called by the mixin; it returns the mock module.
        self.mock_supa = self._supa_fn.return_value
        self.platform = SupaSRTPlatform()

    def tearDown(self):
        self._supa_patch.stop()

    def test_get_device_returns_supa_device(self):
        with patch("sglang.srt.platforms.supa.torch.device") as mock_device:
            self.platform.get_device(2)
        mock_device.assert_called_once_with("supa", 2)

    def test_memory_queries_delegate_to_torch_supa(self):
        self.mock_supa.get_device_properties.return_value = _mock_supa_props(
            total_memory=32 * 1024**3
        )
        self.mock_supa.max_memory_allocated.return_value = 5 * 10**8
        self.mock_supa.mem_get_info.return_value = (10**9, 2 * 10**9)

        self.assertEqual(self.platform.get_device_total_memory(1), 32 * 1024**3)
        self.mock_supa.get_device_properties.assert_called_once_with(1)
        self.assertEqual(self.platform.get_current_memory_usage(), 5 * 10**8)
        self.mock_supa.max_memory_allocated.assert_called_once_with(None)
        self.assertEqual(self.platform.get_available_memory(2), (10**9, 2 * 10**9))
        self.mock_supa.mem_get_info.assert_called_once_with(2)

    def test_device_info_queries_delegate_to_torch_supa(self):
        self.mock_supa.get_device_name.return_value = "Biren166M"
        self.mock_supa.get_device_properties.return_value = _mock_supa_props(
            major=10, minor=4
        )

        self.assertEqual(self.platform.get_device_name(1), "Biren166M")
        self.mock_supa.get_device_name.assert_called_once_with(1)
        self.assertEqual(
            self.platform.get_device_capability(0), DeviceCapability(10, 4)
        )
        # BR1xx device properties have no uuid attribute.
        self.assertEqual(self.platform.get_device_uuid(3), "supa:3")

    def test_device_state_ops_delegate_to_torch_supa(self):
        with patch("sglang.srt.platforms.supa.torch.device") as mock_device:
            device = mock_device.return_value
            self.platform.set_device(device)
        self.mock_supa.set_device.assert_called_once_with(device)
        self.platform.empty_cache()
        self.mock_supa.empty_cache.assert_called_once()
        self.platform.synchronize()
        self.mock_supa.synchronize.assert_called_once()

    @patch("torch.manual_seed")
    @patch("sglang.srt.platforms.device_mixin.np.random.seed")
    @patch("sglang.srt.platforms.device_mixin.random.seed")
    def test_seed_everything_seeds_supa(
        self, mock_random_seed, mock_np_seed, mock_torch_seed
    ):
        SupaSRTPlatform.seed_everything(123)
        mock_random_seed.assert_called_once_with(123)
        mock_np_seed.assert_called_once_with(123)
        mock_torch_seed.assert_called_once_with(123)
        self.mock_supa.manual_seed_all.assert_called_once_with(123)

    def test_seed_everything_none_is_noop(self):
        SupaSRTPlatform.seed_everything(None)
        self.mock_supa.manual_seed_all.assert_not_called()


class TestSupaDiscovery(CustomTestCase):
    @patch("sglang.srt.platforms.torch")
    def test_is_supa_available_reads_torch_supa(self, mock_torch):
        import sglang.srt.platforms as plat_mod

        mock_torch.supa.is_available.return_value = True
        self.assertTrue(plat_mod._is_supa_available())

        mock_torch.supa.is_available.return_value = False
        self.assertFalse(plat_mod._is_supa_available())

    @patch("sglang.srt.platforms.load_plugins_by_group")
    @patch("sglang.srt.platforms._is_cuda_available")
    @patch("sglang.srt.platforms._is_cpu_available")
    @patch("sglang.srt.platforms._is_xpu_available")
    @patch("sglang.srt.platforms._is_supa_available")
    @patch("sglang.srt.platforms.envs")
    def test_no_plugin_supa_fallback(
        self, mock_envs, mock_supa, mock_xpu, mock_cpu, mock_cuda, mock_load
    ):
        mock_envs.SGLANG_PLATFORM.get.return_value = ""
        mock_cpu.return_value = False
        mock_cuda.return_value = False
        mock_xpu.return_value = False
        mock_supa.return_value = True
        mock_load.return_value = {}

        self.assertIsInstance(_resolve_platform(), SupaSRTPlatform)


if __name__ == "__main__":
    import unittest

    unittest.main()
