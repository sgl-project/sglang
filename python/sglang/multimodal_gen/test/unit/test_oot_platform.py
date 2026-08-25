"""
Unit tests for out-of-tree (OOT) diffusion platform discovery.

Run:
  python -m pytest python/sglang/multimodal_gen/test/unit/test_oot_platform.py -v

NOTE: never import sglang.multimodal_gen.runtime.platforms.cuda here - it runs
NVML at import time and crashes on non-NVIDIA machines. The same holds for the
cuda device communicator.
"""

import contextlib
import os
import unittest
from unittest.mock import MagicMock, patch

from sglang.multimodal_gen.runtime import platforms as platforms_mod
from sglang.multimodal_gen.runtime.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from sglang.multimodal_gen.runtime.platforms.interface import Platform, PlatformEnum
from sglang.srt.environ import envs

CPU_QUALNAME = "sglang.multimodal_gen.runtime.platforms.cpu.CpuPlatform"
XPU_QUALNAME = "sglang.multimodal_gen.runtime.platforms.xpu.XpuPlatform"
PLATFORMS = "sglang.multimodal_gen.runtime.platforms"


class FakeOOTPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name = "fake_oot"
    device_type = "fake_oot"
    dispatch_key = "CPU"


FAKE_OOT_QUALNAME = f"{__name__}.FakeOOTPlatform"


def _override(value: str):
    """Patch SGLANG_DIFFUSION_PLATFORM_OVERRIDE ('' means unset).

    Deliberately patches os.environ directly rather than using
    ``EnvField.override``: this pins that the descriptor reads os.environ at
    call time, so plain env patching keeps working after the switch from
    ``os.environ.get`` to ``envs.SGLANG_DIFFUSION_PLATFORM_OVERRIDE.get()``.
    """
    return patch.dict(
        os.environ, {"SGLANG_DIFFUSION_PLATFORM_OVERRIDE": value}, clear=False
    )


def _make_platform_ep(name, activate_result=None, raises=False):
    ep = MagicMock()
    ep.name = name
    ep.dist = MagicMock()
    ep.dist.name = f"{name}-dist"

    def activate():
        if raises:
            raise RuntimeError("activate blew up")
        return activate_result

    ep.load.return_value = activate
    return ep


class _PlatformTestBase(unittest.TestCase):
    """Saves/restores the lazily-initialized platform singleton.

    Required because an installed OOT plugin may pre-seed _current_platform in
    this environment.
    """

    def setUp(self):
        self._saved_platform = platforms_mod._current_platform
        platforms_mod._current_platform = None
        self._stack = contextlib.ExitStack()
        self.addCleanup(self._stack.close)

    def tearDown(self):
        platforms_mod._current_platform = self._saved_platform

    def select_platform(self, name: str) -> None:
        """Set SGLANG_DIFFUSION_PLATFORM for the duration of the test."""
        self._stack.enter_context(envs.SGLANG_DIFFUSION_PLATFORM.override(name))

    def resolve(self) -> str:
        return platforms_mod.resolve_current_platform_cls_qualname()


class TestOverrideEnvVar(_PlatformTestBase):
    def test_short_name_cpu(self):
        with _override("cpu"):
            self.assertEqual(self.resolve(), CPU_QUALNAME)

    def test_short_name_xpu_is_supported(self):
        with _override("xpu"):
            self.assertEqual(self.resolve(), XPU_QUALNAME)

    def test_dotted_value_is_taken_as_qualname(self):
        with _override(FAKE_OOT_QUALNAME):
            self.assertEqual(self.resolve(), FAKE_OOT_QUALNAME)

    def test_unknown_short_name_raises_value_error(self):
        with _override("nosuchplatform"):
            with self.assertRaises(ValueError):
                self.resolve()

    def test_env_descriptor_reads_os_environ(self):
        with _override("cpu"):
            self.assertEqual(envs.SGLANG_DIFFUSION_PLATFORM_OVERRIDE.get(), "cpu")


class TestSelectedPlatformPlugin(_PlatformTestBase):
    def test_selected_plugin_is_used(self):
        ep = _make_platform_ep("fakeoot", activate_result=FAKE_OOT_QUALNAME)
        self.select_platform("fakeoot")
        with _override(""), patch(f"{PLATFORMS}.entry_points", return_value=[ep]):
            self.assertEqual(self.resolve(), FAKE_OOT_QUALNAME)

    def test_selected_plugin_not_installed_raises(self):
        ep = _make_platform_ep("other", activate_result=FAKE_OOT_QUALNAME)
        self.select_platform("missing")
        with _override(""), patch(f"{PLATFORMS}.entry_points", return_value=[ep]):
            with self.assertRaises(RuntimeError) as ctx:
                self.resolve()
        self.assertIn("'other'", str(ctx.exception))

    def test_selected_plugin_returning_none_raises(self):
        ep = _make_platform_ep("fakeoot", activate_result=None)
        self.select_platform("fakeoot")
        with _override(""), patch(f"{PLATFORMS}.entry_points", return_value=[ep]):
            with self.assertRaises(RuntimeError):
                self.resolve()

    def test_unselected_plugin_is_never_loaded(self):
        """Level-2 front-loading filter: only the selected ep is imported."""
        selected = _make_platform_ep("fakeoot", activate_result=FAKE_OOT_QUALNAME)
        unselected = _make_platform_ep("otheroot", activate_result=CPU_QUALNAME)
        self.select_platform("fakeoot")
        with _override(""), patch(
            f"{PLATFORMS}.entry_points", return_value=[selected, unselected]
        ):
            self.assertEqual(self.resolve(), FAKE_OOT_QUALNAME)
        selected.load.assert_called_once()
        unselected.load.assert_not_called()

    def test_selected_plugin_activate_raising_is_logged_and_reraised(self):
        ep = _make_platform_ep("fakeoot", raises=True)
        self.select_platform("fakeoot")
        with _override(""), patch(f"{PLATFORMS}.entry_points", return_value=[ep]):
            with self.assertLogs(PLATFORMS, level="ERROR") as cm:
                with self.assertRaises(RuntimeError) as ctx:
                    self.resolve()
        self.assertIn("activate blew up", str(ctx.exception))
        self.assertTrue(any("fakeoot" in msg for msg in cm.output))


class TestPlatformAutoDiscovery(_PlatformTestBase):
    def _run(self, discovered):
        self.select_platform("")
        with _override(""), patch(
            f"{PLATFORMS}.discover_diffusion_plugins", return_value=discovered
        ):
            return self.resolve()

    def test_single_activated_plugin_wins(self):
        discovered = {"fakeoot": (lambda: FAKE_OOT_QUALNAME, "fake-dist")}
        self.assertEqual(self._run(discovered), FAKE_OOT_QUALNAME)

    def test_multiple_activated_plugins_raise(self):
        discovered = {
            "a": (lambda: FAKE_OOT_QUALNAME, "a-dist"),
            "b": (lambda: CPU_QUALNAME, "b-dist"),
        }
        with self.assertRaises(RuntimeError) as ctx:
            self._run(discovered)
        self.assertIn("SGLANG_DIFFUSION_PLATFORM", str(ctx.exception))

    def test_activation_error_is_isolated(self):
        def boom():
            raise RuntimeError("no hardware")

        discovered = {
            "bad": (boom, "bad-dist"),
            "good": (lambda: FAKE_OOT_QUALNAME, "good-dist"),
        }
        self.assertEqual(self._run(discovered), FAKE_OOT_QUALNAME)

    def test_no_plugin_falls_back_to_builtin_chain(self):
        self.select_platform("")
        with _override(""), patch(
            f"{PLATFORMS}.discover_diffusion_plugins", return_value={}
        ), patch(
            f"{PLATFORMS}.mps_platform_plugin", return_value=None
        ) as mock_mps, patch(
            f"{PLATFORMS}.xpu_platform_plugin", return_value=None
        ), patch(
            f"{PLATFORMS}.rocm_platform_plugin", return_value=None
        ), patch(
            f"{PLATFORMS}.cuda_platform_plugin", return_value=None
        ), patch(
            f"{PLATFORMS}.npu_platform_plugin", return_value=None
        ), patch(
            f"{PLATFORMS}.musa_platform_plugin", return_value=None
        ), patch(
            f"{PLATFORMS}.cpu_platform_plugin", return_value=CPU_QUALNAME
        ):
            self.assertEqual(self.resolve(), CPU_QUALNAME)
            mock_mps.assert_called_once()


class TestLoadPlatformClass(_PlatformTestBase):
    def test_accepts_platform_subclass(self):
        self.assertIs(
            platforms_mod._load_platform_class(FAKE_OOT_QUALNAME), FakeOOTPlatform
        )

    def test_accepts_colon_separated_form(self):
        module, _, name = FAKE_OOT_QUALNAME.rpartition(".")
        self.assertIs(
            platforms_mod._load_platform_class(f"{module}:{name}"), FakeOOTPlatform
        )

    def test_rejects_non_platform_type(self):
        with self.assertRaises(TypeError):
            platforms_mod._load_platform_class("builtins.dict")

    def test_rejects_non_type(self):
        with self.assertRaises(TypeError):
            platforms_mod._load_platform_class("os.getcwd")

    def test_rejects_dotless_value(self):
        with self.assertRaises(TypeError):
            platforms_mod._load_platform_class("os")

    def test_current_platform_goes_through_load_platform_class(self):
        with patch.object(
            platforms_mod,
            "resolve_current_platform_cls_qualname",
            return_value=FAKE_OOT_QUALNAME,
        ):
            self.assertIsInstance(platforms_mod.current_platform, FakeOOTPlatform)

    def test_current_platform_rejects_non_platform_qualname(self):
        platforms_mod._current_platform = None
        with patch.object(
            platforms_mod,
            "resolve_current_platform_cls_qualname",
            return_value="builtins.dict",
        ):
            with self.assertRaises(TypeError):
                platforms_mod.current_platform


class FakeCommunicator(DeviceCommunicatorBase):
    """Stand-in device communicator class for OOT platform tests.

    The base __init__ needs an initialized process group, so it is bypassed;
    these tests only assert on class identity.
    """

    def __init__(self, cpu_group=None, device=None, device_group=None, unique_name=""):
        self.unique_name = unique_name


class NotACommunicator:
    """Resolvable but not a DeviceCommunicatorBase subclass."""


class OOTCommPlatform(FakeOOTPlatform):
    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return f"{__name__}.FakeCommunicator"


class OOTBadCommPlatform(FakeOOTPlatform):
    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return f"{__name__}.NotACommunicator"


class OOTPlatformClaimingEveryDevice(FakeOOTPlatform):
    """OOT platform that also answers True to the in-tree predicates.

    Pins the mandated dispatch order in CustomOp.dispatch_forward: if the OOT
    branch were moved after the hip/npu/xpu/musa branches, dispatch would pick
    one of those instead of forward_oot.
    """

    def is_hip(self) -> bool:
        return True

    def is_npu(self) -> bool:
        return True

    def is_xpu(self) -> bool:
        return True

    def is_musa(self) -> bool:
        return True


class TestCustomOpOOTDispatch(unittest.TestCase):
    def test_out_of_tree_platform_dispatches_to_forward_oot(self):
        from sglang.multimodal_gen.runtime.layers import custom_op as custom_op_mod

        class MyOp(custom_op_mod.CustomOp):
            def forward_native(self, x):
                return "native"

            def forward_cuda(self, x):
                return "cuda"

            def forward_oot(self, x):
                return "oot"

            def forward_npu(self, x):
                return "npu"

            def forward_xpu(self, x):
                return "xpu"

        with patch.object(custom_op_mod, "_is_cuda", False), patch.object(
            custom_op_mod, "current_platform", OOTPlatformClaimingEveryDevice()
        ):
            op = MyOp()
        self.assertEqual(op(1), "oot")

    def test_cuda_platform_still_dispatches_to_forward_cuda(self):
        from sglang.multimodal_gen.runtime.layers import custom_op as custom_op_mod

        class MyOp(custom_op_mod.CustomOp):
            def forward_native(self, x):
                return "native"

            def forward_cuda(self, x):
                return "cuda"

        with patch.object(custom_op_mod, "_is_cuda", True):
            op = MyOp()
        self.assertEqual(op(1), "cuda")


class TestDeviceCommunicatorSelection(unittest.TestCase):
    def test_out_of_tree_platform_uses_platform_hook(self):
        from sglang.multimodal_gen.runtime.distributed import group_coordinator as gc

        with patch.object(gc, "current_platform", OOTCommPlatform()):
            self.assertIs(gc._resolve_device_communicator_cls(), FakeCommunicator)

    def test_out_of_tree_non_communicator_raises(self):
        from sglang.multimodal_gen.runtime.distributed import group_coordinator as gc

        with patch.object(gc, "current_platform", OOTBadCommPlatform()):
            with self.assertRaises(TypeError) as ctx:
                gc._resolve_device_communicator_cls()
        self.assertIn("NotACommunicator", str(ctx.exception))

    def test_cpu_platform_keeps_cpu_communicator(self):
        from sglang.multimodal_gen.runtime.distributed import group_coordinator as gc
        from sglang.multimodal_gen.runtime.distributed.device_communicators.cpu_communicator import (
            CpuCommunicator,
        )
        from sglang.multimodal_gen.runtime.platforms.cpu import CpuPlatform

        with patch.object(gc, "current_platform", CpuPlatform()):
            self.assertIs(gc._resolve_device_communicator_cls(), CpuCommunicator)

    def test_mps_platform_keeps_cpu_communicator(self):
        from sglang.multimodal_gen.runtime.distributed import group_coordinator as gc
        from sglang.multimodal_gen.runtime.distributed.device_communicators.cpu_communicator import (
            CpuCommunicator,
        )
        from sglang.multimodal_gen.runtime.platforms.mps import MpsPlatform

        # MpsPlatform.get_device_communicator_cls() returns the *base* class,
        # but the in-tree path must keep giving MPS the CPU communicator.
        with patch.object(gc, "current_platform", MpsPlatform()):
            self.assertIs(gc._resolve_device_communicator_cls(), CpuCommunicator)


if __name__ == "__main__":
    unittest.main()
