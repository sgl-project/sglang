"""
Unit tests for SGLang platform abstraction layer.

Tests DeviceMixin, SRTPlatform, PlatformEnum, CpuArchEnum, DeviceCapability,
and the platform discovery / lazy initialization mechanism.
"""

from unittest.mock import MagicMock, patch

import torch

from sglang.srt.platforms import _load_platform_class, _resolve_platform
from sglang.srt.platforms.cuda import CudaDeviceMixin, CudaSRTPlatform
from sglang.srt.platforms.device_mixin import (
    CpuArchEnum,
    DeviceCapability,
    DeviceMixin,
    PlatformEnum,
)
from sglang.srt.platforms.interface import SRTPlatform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="stage-a-test-cpu")


# ---------------------------------------------------------------------------
# Helpers: factory functions to reduce boilerplate
# ---------------------------------------------------------------------------


def _make_device_mixin(enum, name, dtype):
    """Create a concrete DeviceMixin subclass for testing."""

    class M(DeviceMixin):
        _enum = enum
        device_name = name
        device_type = dtype

        def get_device_total_memory(self, device_id=0):
            return 10**9

        def get_current_memory_usage(self, device=None):
            return 5 * 10**8

    return M()


class _StubPlatform(SRTPlatform):
    """Concrete SRTPlatform with minimal defaults for testing overrides."""

    _enum = PlatformEnum.CUDA
    device_name = "cuda"
    device_type = "cuda"

    def get_device_total_memory(self, device_id=0):
        return 10**9

    def get_current_memory_usage(self, device=None):
        return 5 * 10**8

    def get_default_attention_backend(self):
        return "flashinfer"

    def get_graph_runner_cls(self):
        return object

    def get_mha_kv_pool_cls(self):
        return object

    def get_mla_kv_pool_cls(self):
        return object

    def get_nsa_kv_pool_cls(self):
        return object

    def get_paged_allocator_cls(self):
        return object

    def get_piecewise_backend_cls(self):
        return object


def _make_platform_ep(name, load_fn=None):
    """Create a mock entry point for platform plugins."""
    ep = MagicMock()
    ep.name = name
    if load_fn is not None:
        ep.load.return_value = load_fn
    else:
        ep.load.return_value = MagicMock()
    return ep


# ---------------------------------------------------------------------------
# PlatformEnum & CpuArchEnum
# ---------------------------------------------------------------------------


class TestPlatformEnum(CustomTestCase):
    """Tests for PlatformEnum enumeration."""

    def test_all_expected_values_exist(self):
        expected = {
            "CUDA",
            "ROCM",
            "CPU",
            "XPU",
            "MUSA",
            "NPU",
            "TPU",
            "MPS",
            "OOT",
            "UNSPECIFIED",
        }
        actual = {member.name for member in PlatformEnum}
        self.assertEqual(actual, expected)


class TestCpuArchEnum(CustomTestCase):
    """Tests for CpuArchEnum enumeration."""

    def test_all_expected_values_exist(self):
        expected = {"X86", "ARM", "UNSPECIFIED"}
        actual = {member.name for member in CpuArchEnum}
        self.assertEqual(actual, expected)


# ---------------------------------------------------------------------------
# DeviceCapability
# ---------------------------------------------------------------------------


class TestDeviceCapability(CustomTestCase):
    """Tests for DeviceCapability custom logic (formatting, conversion)."""

    def test_as_version_str(self):
        self.assertEqual(DeviceCapability(major=9, minor=0).as_version_str(), "9.0")
        self.assertEqual(DeviceCapability(major=8, minor=9).as_version_str(), "8.9")

    def test_to_int(self):
        self.assertEqual(DeviceCapability(major=9, minor=0).to_int(), 90)
        self.assertEqual(DeviceCapability(major=8, minor=9).to_int(), 89)
        self.assertEqual(DeviceCapability(major=0, minor=0).to_int(), 0)


# ---------------------------------------------------------------------------
# DeviceMixin
# ---------------------------------------------------------------------------

# Platform identity test data: (enum, name, dtype, true_method)
_PLATFORM_IDENTITY = [
    (PlatformEnum.CUDA, "cuda", "cuda", "is_cuda"),
    (PlatformEnum.ROCM, "rocm", "hip", "is_rocm"),
    (PlatformEnum.CPU, "cpu", "cpu", "is_cpu"),
    (PlatformEnum.XPU, "xpu", "xpu", "is_xpu"),
    (PlatformEnum.MUSA, "musa", "musa", "is_musa"),
    (PlatformEnum.NPU, "npu", "npu", "is_npu"),
    (PlatformEnum.TPU, "tpu", "tpu", "is_tpu"),
    (PlatformEnum.MPS, "mps", "mps", "is_mps"),
]

# is_cuda_alike test data: (enum, name, dtype, expected)
_CUDA_ALIKE = [
    (PlatformEnum.CUDA, "cuda", "cuda", True),
    (PlatformEnum.ROCM, "rocm", "hip", True),
    (PlatformEnum.MUSA, "musa", "musa", True),
    (PlatformEnum.CPU, "cpu", "cpu", False),
    (PlatformEnum.NPU, "npu", "npu", False),
]


class TestDeviceMixin(CustomTestCase):
    """Tests for DeviceMixin base class."""

    def test_platform_identity_methods(self):
        """Each platform type returns True for its identity method."""
        for enum_val, name, dtype, method in _PLATFORM_IDENTITY:
            with self.subTest(method=method, enum=enum_val.name):
                mixin = _make_device_mixin(enum_val, name, dtype)
                self.assertTrue(getattr(mixin, method)())

    def test_is_cuda_alike(self):
        """is_cuda_alike is True for CUDA/ROCM/MUSA, False otherwise."""
        for enum_val, name, dtype, expected in _CUDA_ALIKE:
            with self.subTest(enum=enum_val.name):
                mixin = _make_device_mixin(enum_val, name, dtype)
                self.assertEqual(mixin.is_cuda_alike(), expected)

    def test_is_out_of_tree(self):
        oot = _make_device_mixin(PlatformEnum.OOT, "custom", "custom")
        self.assertTrue(oot.is_out_of_tree())
        cuda = _make_device_mixin(PlatformEnum.CUDA, "cuda", "cuda")
        self.assertFalse(cuda.is_out_of_tree())

    @patch("platform.machine")
    def test_get_cpu_architecture(self, mock_machine):
        """get_cpu_architecture maps common strings to CpuArchEnum."""
        cases = [
            ("x86_64", CpuArchEnum.X86),
            ("amd64", CpuArchEnum.X86),
            ("i386", CpuArchEnum.X86),
            ("i686", CpuArchEnum.X86),
            ("X86_64", CpuArchEnum.X86),  # case insensitive
            ("arm64", CpuArchEnum.ARM),
            ("aarch64", CpuArchEnum.ARM),
            ("unknown_arch", CpuArchEnum.UNSPECIFIED),
        ]
        for machine_str, expected in cases:
            with self.subTest(machine=machine_str):
                mock_machine.return_value = machine_str
                self.assertEqual(DeviceMixin.get_cpu_architecture(), expected)


# ---------------------------------------------------------------------------
# SRTPlatform
# ---------------------------------------------------------------------------


class TestSRTPlatform(CustomTestCase):
    """Tests for SRTPlatform base class and default behaviors."""

    def test_compile_backend_signature_compatibility(self):
        """get_compile_backend accepts mode keyword arg without error."""
        base = SRTPlatform()
        self.assertEqual(base.get_compile_backend(mode="npugraph_ex"), "inductor")

    def test_base_device_identity_stays_unspecified(self):
        """The abstract SRT base should not claim any concrete in-tree device."""
        base = SRTPlatform()
        self.assertFalse(base.is_cuda())
        self.assertFalse(base.is_cuda_alike())


class TestCudaDeviceMixin(CustomTestCase):
    """Tests for CUDA device operation defaults."""

    def test_default_get_device_returns_cuda_device(self):
        base = CudaSRTPlatform()
        self.assertEqual(base.get_device(2), torch.device("cuda", 2))

    def test_cuda_platform_identity(self):
        base = CudaSRTPlatform()
        self.assertTrue(base.is_cuda())
        self.assertTrue(base.is_cuda_alike())
        self.assertIsInstance(base, CudaDeviceMixin)

    @patch("torch.cuda.get_device_properties")
    def test_default_get_device_total_memory_uses_cuda(
        self, mock_get_device_properties
    ):
        mock_get_device_properties.return_value.total_memory = 123
        base = CudaSRTPlatform()
        self.assertEqual(base.get_device_total_memory(1), 123)
        mock_get_device_properties.assert_called_once_with(1)

    @patch("torch.cuda.max_memory_allocated", return_value=456)
    def test_default_get_current_memory_usage_uses_cuda(
        self, mock_max_memory_allocated
    ):
        base = CudaSRTPlatform()
        device = torch.device("cuda", 1)
        self.assertEqual(base.get_current_memory_usage(device), 456.0)
        mock_max_memory_allocated.assert_called_once_with(device)

    @patch("torch.cuda.set_device")
    def test_default_set_device_uses_cuda(self, mock_set_device):
        base = CudaSRTPlatform()
        device = torch.device("cuda", 1)
        base.set_device(device)
        mock_set_device.assert_called_once_with(device)

    @patch("torch.cuda.get_device_name", return_value="NVIDIA H100")
    def test_default_get_device_name_uses_cuda(self, mock_get_device_name):
        base = CudaSRTPlatform()
        self.assertEqual(base.get_device_name(1), "NVIDIA H100")
        mock_get_device_name.assert_called_once_with(1)

    @patch("torch.cuda.get_device_properties")
    def test_default_get_device_uuid_uses_cuda(self, mock_get_device_properties):
        mock_get_device_properties.return_value.uuid = "1234"
        base = CudaSRTPlatform()
        self.assertEqual(base.get_device_uuid(1), "1234")
        mock_get_device_properties.assert_called_once_with(1)

    @patch("torch.cuda.get_device_capability", return_value=(9, 0))
    def test_default_get_device_capability_uses_cuda(self, mock_get_device_capability):
        base = CudaSRTPlatform()
        self.assertEqual(base.get_device_capability(1), DeviceCapability(9, 0))
        mock_get_device_capability.assert_called_once_with(1)

    @patch("torch.cuda.empty_cache")
    def test_default_empty_cache_uses_cuda(self, mock_empty_cache):
        base = CudaSRTPlatform()
        base.empty_cache()
        mock_empty_cache.assert_called_once_with()

    @patch("torch.cuda.synchronize")
    def test_default_synchronize_uses_cuda(self, mock_synchronize):
        base = CudaSRTPlatform()
        base.synchronize()
        mock_synchronize.assert_called_once_with()

    @patch("torch.cuda.mem_get_info", return_value=(123, 456), create=True)
    def test_default_get_available_memory_uses_cuda(self, mock_mem_get_info):
        base = CudaSRTPlatform()
        self.assertEqual(base.get_available_memory(1), (123, 456))
        mock_mem_get_info.assert_called_once_with(1)

    def test_default_distributed_backend_is_nccl(self):
        base = CudaSRTPlatform()
        self.assertEqual(base.get_torch_distributed_backend_str(), "nccl")

    @patch("torch.cuda.manual_seed_all")
    @patch("torch.manual_seed")
    @patch("sglang.srt.platforms.device_mixin.np.random.seed")
    @patch("sglang.srt.platforms.device_mixin.random.seed")
    def test_default_seed_everything_seeds_cuda(
        self, mock_random_seed, mock_np_seed, mock_torch_seed, mock_cuda_seed
    ):
        CudaSRTPlatform.seed_everything(123)
        mock_random_seed.assert_called_once_with(123)
        mock_np_seed.assert_called_once_with(123)
        mock_torch_seed.assert_called_once_with(123)
        mock_cuda_seed.assert_called_once_with(123)

    def test_cuda_srt_platform_capabilities(self):
        base = CudaSRTPlatform()
        self.assertTrue(base.supports_fp8())
        self.assertTrue(base.support_cuda_graph())
        self.assertTrue(base.support_piecewise_cuda_graph())


class TestSRTPlatformOverrides(CustomTestCase):
    """Tests for SRTPlatform method overrides via plugins."""

    def test_custom_get_dispatch_key_name(self):
        class P(_StubPlatform):
            _enum = PlatformEnum.NPU
            device_name = "npu"
            device_type = "npu"

            def get_dispatch_key_name(self):
                return "npu"

        self.assertEqual(P().get_dispatch_key_name(), "npu")

    def test_custom_get_compile_backend(self):
        class P(_StubPlatform):
            _enum = PlatformEnum.NPU
            device_name = "npu"
            device_type = "npu"

            def get_compile_backend(self, mode=None):
                return "inductor"

        self.assertEqual(P().get_compile_backend(mode="npugraph_ex"), "inductor")


# ---------------------------------------------------------------------------
# Platform Discovery: _resolve_platform
# ---------------------------------------------------------------------------


class TestResolvePlatformWithEnv(CustomTestCase):
    """Tests for _resolve_platform when SGLANG_PLATFORM is set."""

    @patch("sglang.srt.platforms.entry_points")
    @patch("sglang.srt.platforms.envs")
    def test_selected_plugin_activates(self, mock_envs, mock_ep):
        """When SGLANG_PLATFORM matches an entry point, it activates that plugin."""
        mock_envs.SGLANG_PLATFORM.get.return_value = "my_hardware"
        plugin_fn = MagicMock(return_value="pkg.Mod:MyPlatform")
        mock_ep.return_value = [_make_platform_ep("my_hardware", plugin_fn)]
        with patch("sglang.srt.platforms._load_platform_class") as mock_load:
            mock_instance = MagicMock()
            mock_load.return_value = MagicMock(return_value=mock_instance)
            result = _resolve_platform()
            mock_load.assert_called_once_with("pkg.Mod:MyPlatform")
            self.assertEqual(result, mock_instance)

    @patch("sglang.srt.platforms.entry_points")
    @patch("sglang.srt.platforms.envs")
    def test_selected_plugin_not_found(self, mock_envs, mock_ep):
        """When SGLANG_PLATFORM names a nonexistent plugin, raise RuntimeError."""
        mock_envs.SGLANG_PLATFORM.get.return_value = "nonexistent"
        mock_ep.return_value = []
        with self.assertRaises(RuntimeError):
            _resolve_platform()

    @patch("sglang.srt.platforms.entry_points")
    @patch("sglang.srt.platforms.envs")
    def test_selected_plugin_hardware_unavailable(self, mock_envs, mock_ep):
        """When activate() returns None, hardware is not available."""
        mock_envs.SGLANG_PLATFORM.get.return_value = "my_hardware"
        plugin_fn = MagicMock(return_value=None)
        mock_ep.return_value = [_make_platform_ep("my_hardware", plugin_fn)]
        with self.assertRaises(RuntimeError):
            _resolve_platform()

    @patch("sglang.srt.platforms.entry_points")
    @patch("sglang.srt.platforms.envs")
    def test_selected_plugin_load_exception(self, mock_envs, mock_ep):
        """When ep.load() or activate() throws, exception is re-raised."""
        mock_envs.SGLANG_PLATFORM.get.return_value = "my_hardware"
        plugin_fn = MagicMock(side_effect=ImportError("missing dep"))
        mock_ep.return_value = [_make_platform_ep("my_hardware", plugin_fn)]
        with self.assertRaises(ImportError):
            _resolve_platform()

    @patch("sglang.srt.platforms.entry_points")
    @patch("sglang.srt.platforms.envs")
    def test_other_plugins_not_loaded(self, mock_envs, mock_ep):
        """When SGLANG_PLATFORM is set, other plugins are not imported."""
        mock_envs.SGLANG_PLATFORM.get.return_value = "target_hw"
        target_fn = MagicMock(return_value="pkg.Mod:TargetPlatform")
        other_ep = _make_platform_ep("other_hw")  # default load returns MagicMock
        target_ep = _make_platform_ep("target_hw", target_fn)
        mock_ep.return_value = [other_ep, target_ep]
        with patch("sglang.srt.platforms._load_platform_class") as mock_load:
            mock_load.return_value = MagicMock(return_value=MagicMock())
            _resolve_platform()
            # Only the target entry point should be loaded
            target_ep.load.assert_called_once()
            other_ep.load.assert_not_called()


class TestResolvePlatformAutoDiscover(CustomTestCase):
    """Tests for _resolve_platform auto-discovery when SGLANG_PLATFORM is not set."""

    @patch("sglang.srt.platforms.torch")
    def test_is_cuda_available_excludes_rocm(self, mock_torch):
        """ROCm exposes torch.cuda, but should not use the CUDA platform identity."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.version.hip = "6.0"

        import sglang.srt.platforms as plat_mod

        self.assertFalse(plat_mod._is_cuda_available())

    @patch("sglang.srt.platforms.load_plugins_by_group")
    @patch("sglang.srt.platforms.envs")
    def test_single_plugin_activates(self, mock_envs, mock_load):
        """When exactly one plugin activates, return its platform instance."""
        mock_envs.SGLANG_PLATFORM.get.return_value = ""
        plugin_fn = MagicMock(return_value="pkg.Mod:MyPlatform")
        mock_load.return_value = {"my_hw": (plugin_fn, "my-hw-dist")}
        with patch("sglang.srt.platforms._load_platform_class") as mock_resolve:
            mock_instance = MagicMock()
            mock_resolve.return_value = MagicMock(return_value=mock_instance)
            result = _resolve_platform()
            mock_resolve.assert_called_once_with("pkg.Mod:MyPlatform")
            self.assertEqual(result, mock_instance)

    @patch("sglang.srt.platforms.load_plugins_by_group")
    @patch("sglang.srt.platforms._is_cuda_available")
    @patch("sglang.srt.platforms.envs")
    def test_no_plugin_activates_cuda_fallback(
        self, mock_envs, mock_is_cuda_available, mock_load
    ):
        """When CUDA is available and no plugin activates, return CUDA defaults."""
        mock_envs.SGLANG_PLATFORM.get.return_value = ""
        mock_is_cuda_available.return_value = True
        mock_load.return_value = {}
        result = _resolve_platform()
        self.assertIsInstance(result, CudaSRTPlatform)

    @patch("sglang.srt.platforms.load_plugins_by_group")
    @patch("sglang.srt.platforms._is_cuda_available")
    @patch("sglang.srt.platforms.envs")
    def test_no_plugin_no_cuda_activates_base_fallback(
        self, mock_envs, mock_is_cuda_available, mock_load
    ):
        """When no plugin or CUDA is available, return the abstract base platform."""
        mock_envs.SGLANG_PLATFORM.get.return_value = ""
        mock_is_cuda_available.return_value = False
        mock_load.return_value = {}
        result = _resolve_platform()
        self.assertIsInstance(result, SRTPlatform)
        self.assertNotIsInstance(result, CudaSRTPlatform)

    @patch("sglang.srt.platforms.load_plugins_by_group")
    @patch("sglang.srt.platforms.torch")
    @patch("sglang.srt.platforms.envs")
    def test_no_plugin_rocm_does_not_activate_cuda_fallback(
        self, mock_envs, mock_torch, mock_load
    ):
        """ROCm exposes torch.cuda but must not use the CUDA fallback platform."""
        mock_envs.SGLANG_PLATFORM.get.return_value = ""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.version.hip = "6.0"
        mock_load.return_value = {}

        result = _resolve_platform()

        self.assertIsInstance(result, SRTPlatform)
        self.assertNotIsInstance(result, CudaSRTPlatform)

    @patch("sglang.srt.platforms.load_plugins_by_group")
    @patch("sglang.srt.platforms.envs")
    def test_multiple_plugins_activate_raises(self, mock_envs, mock_load):
        """When multiple plugins activate, raise RuntimeError."""
        mock_envs.SGLANG_PLATFORM.get.return_value = ""
        fn1 = MagicMock(return_value="pkg1.Mod:Platform1")
        fn2 = MagicMock(return_value="pkg2.Mod:Platform2")
        mock_load.return_value = {"hw1": (fn1, "hw1-dist"), "hw2": (fn2, "hw2-dist")}
        with self.assertRaises(RuntimeError):
            _resolve_platform()

    @patch("sglang.srt.platforms.load_plugins_by_group")
    @patch("sglang.srt.platforms.envs")
    def test_plugin_exception_does_not_crash(self, mock_envs, mock_load):
        """When a plugin's activate() throws, it is skipped, others continue."""
        mock_envs.SGLANG_PLATFORM.get.return_value = ""
        bad_fn = MagicMock(side_effect=RuntimeError("broken"))
        good_fn = MagicMock(return_value="pkg.Mod:GoodPlatform")
        mock_load.return_value = {
            "bad": (bad_fn, "bad-dist"),
            "good": (good_fn, "good-dist"),
        }
        with patch("sglang.srt.platforms._load_platform_class") as mock_resolve:
            mock_instance = MagicMock()
            mock_resolve.return_value = MagicMock(return_value=mock_instance)
            result = _resolve_platform()
            mock_resolve.assert_called_once_with("pkg.Mod:GoodPlatform")
            self.assertEqual(result, mock_instance)

    @patch("sglang.srt.platforms.load_plugins_by_group")
    @patch("sglang.srt.platforms.envs")
    def test_plugin_returns_none_is_skipped(self, mock_envs, mock_load):
        """When a plugin's activate() returns None, it is skipped (hardware unavailable)."""
        mock_envs.SGLANG_PLATFORM.get.return_value = ""
        none_fn = MagicMock(return_value=None)
        good_fn = MagicMock(return_value="pkg.Mod:GoodPlatform")
        mock_load.return_value = {
            "unavailable": (none_fn, "unavail-dist"),
            "good": (good_fn, "good-dist"),
        }
        with patch("sglang.srt.platforms._load_platform_class") as mock_resolve:
            mock_instance = MagicMock()
            mock_resolve.return_value = MagicMock(return_value=mock_instance)
            result = _resolve_platform()
            # Only the good plugin activated; single activation succeeds
            mock_resolve.assert_called_once_with("pkg.Mod:GoodPlatform")


# ---------------------------------------------------------------------------
# Platform Discovery: _load_platform_class
# ---------------------------------------------------------------------------


class TestLoadPlatformClass(CustomTestCase):
    """Tests for _load_platform_class qualname resolution."""

    @patch("sglang.srt.platforms.pkgutil.resolve_name")
    def test_valid_subclass(self, mock_resolve):
        """Valid SRTPlatform subclass resolves successfully."""
        mock_resolve.return_value = type("MyPlatform", (SRTPlatform,), {})
        result = _load_platform_class("pkg.Mod:MyPlatform")
        self.assertTrue(issubclass(result, SRTPlatform))

    @patch("sglang.srt.platforms.pkgutil.resolve_name")
    def test_non_subclass_raises_type_error(self, mock_resolve):
        """Non-SRTPlatform class raises TypeError."""
        mock_resolve.return_value = str
        with self.assertRaises(TypeError):
            _load_platform_class("builtins.str")

    @patch("sglang.srt.platforms.pkgutil.resolve_name")
    def test_non_type_raises_type_error(self, mock_resolve):
        """Non-type object raises TypeError."""
        mock_resolve.return_value = "not a class"
        with self.assertRaises(TypeError):
            _load_platform_class("something")


# ---------------------------------------------------------------------------
# Platform Discovery: current_platform lazy init
# ---------------------------------------------------------------------------


class TestCurrentPlatformLazyInit(CustomTestCase):
    """Tests for current_platform lazy initialization via module __getattr__."""

    def setUp(self):
        """Reset module-level cache before each test."""
        import sglang.srt.platforms as plat_mod

        self._saved_platform = plat_mod._current_platform
        plat_mod._current_platform = None

    def tearDown(self):
        """Restore original _current_platform after each test."""
        import sglang.srt.platforms as plat_mod

        plat_mod._current_platform = self._saved_platform

    @patch("sglang.srt.platforms._resolve_platform")
    def test_first_access_triggers_resolve(self, mock_resolve):
        """First access to current_platform calls _resolve_platform."""
        mock_instance = MagicMock(spec=SRTPlatform)
        mock_resolve.return_value = mock_instance
        import sglang.srt.platforms as plat_mod

        result = plat_mod.current_platform
        mock_resolve.assert_called_once()
        self.assertEqual(result, mock_instance)

    @patch("sglang.srt.platforms._resolve_platform")
    def test_subsequent_access_uses_cache(self, mock_resolve):
        """Subsequent accesses return cached instance without re-resolving."""
        mock_instance = MagicMock(spec=SRTPlatform)
        mock_resolve.return_value = mock_instance
        import sglang.srt.platforms as plat_mod

        _ = plat_mod.current_platform
        _ = plat_mod.current_platform
        mock_resolve.assert_called_once()

    def test_other_attribute_raises_error(self):
        """Accessing non-existent module attribute raises AttributeError."""
        import sglang.srt.platforms as plat_mod

        with self.assertRaises(AttributeError):
            _ = plat_mod.nonexistent_attribute


if __name__ == "__main__":
    import unittest

    unittest.main()
