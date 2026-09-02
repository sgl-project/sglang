# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import sglang.multimodal_gen.runtime.platforms as runtime_platforms
from sglang.multimodal_gen.runtime.distributed import group_coordinator, parallel_state
from sglang.multimodal_gen.runtime.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from sglang.multimodal_gen.runtime.distributed.device_communicators.cpu_communicator import (
    CpuCommunicator,
)
from sglang.multimodal_gen.runtime.distributed.device_communicators.cuda_communicator import (
    CudaCommunicator,
)
from sglang.multimodal_gen.runtime.layers import custom_op
from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp
from sglang.multimodal_gen.runtime.platforms.interface import (
    Platform,
    PlatformEnum,
)


class _OotPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name = "test"
    device_type = "test"


class _DispatchKeyOotPlatform(_OotPlatform):
    dispatch_key = "PrivateUse1"


class _LegacyOotPlatform(_OotPlatform):
    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "test.LegacyCommunicator"


class _TestOp(CustomOp):
    def forward_native(self, value):
        return ("native", value)


class _TestCommunicator(DeviceCommunicatorBase):
    pass


class TestOotCustomOpDispatch(unittest.TestCase):
    def tearDown(self):
        CustomOp._oot_forward_registry.pop("test", None)
        CustomOp._oot_forward_registry.pop("other", None)

    def test_registered_forward_is_used(self):
        forward = MagicMock(return_value=("registered", 7))
        CustomOp.register_oot_forward(_TestOp, fn=forward, platform_key="test")
        with patch.object(custom_op.platforms, "_current_platform", _OotPlatform()):
            op = _TestOp()
        self.assertEqual(op._forward_method(7), ("registered", 7))
        forward.assert_called_once_with(op, 7)

    def test_missing_registration_uses_native_fallback(self):
        CustomOp.register_oot_forward(_TestOp, fn=MagicMock(), platform_key="other")
        with patch.object(custom_op.platforms, "_current_platform", _OotPlatform()):
            op = _TestOp()
        self.assertEqual(op._forward_method(7), ("native", 7))


class TestOotRequiredConfiguration(unittest.TestCase):
    def test_device_and_dispatch_defaults_fail_loudly(self):
        platform = type("Oot", (Platform,), {"_enum": PlatformEnum.OOT})()

        with self.assertRaisesRegex(NotImplementedError, "implement get_device"):
            platform.get_device(0)
        with self.assertRaisesRegex(NotImplementedError, "define dispatch_key"):
            platform.get_torch_library_dispatch_key()
        with self.assertRaisesRegex(
            NotImplementedError, "implement get_all_to_all_communicator_cls"
        ):
            platform.get_all_to_all_communicator_cls()

        self.assertEqual(
            _DispatchKeyOotPlatform().get_torch_library_dispatch_key(),
            "PrivateUse1",
        )

    def test_builtin_torch_library_dispatch_is_preserved(self):
        platform = Platform()
        for is_npu, expected in ((False, "CUDA"), (True, "PrivateUse1")):
            with (
                self.subTest(is_npu=is_npu),
                patch.object(platform, "is_out_of_tree", return_value=False),
                patch.object(platform, "is_npu", return_value=is_npu),
            ):
                self.assertEqual(platform.get_torch_library_dispatch_key(), expected)

        from sglang.multimodal_gen.runtime.platforms.xpu import XpuPlatform

        self.assertEqual(XpuPlatform().get_torch_library_dispatch_key(), "CUDA")

    def test_legacy_communicator_override_is_respected(self):
        self.assertEqual(
            _LegacyOotPlatform.get_all_to_all_communicator_cls(),
            "test.LegacyCommunicator",
        )


class TestOotRuntimeHooks(unittest.TestCase):
    def test_builtin_communicator_behavior_is_preserved(self):
        cuda_communicator = (
            "sglang.multimodal_gen.runtime.distributed.device_communicators."
            "cuda_communicator.CudaCommunicator"
        )
        cpu_communicator = (
            "sglang.multimodal_gen.runtime.distributed.device_communicators."
            "cpu_communicator.CpuCommunicator"
        )
        cases = (
            ("cuda", cuda_communicator, CudaCommunicator),
            ("rocm", cuda_communicator, CudaCommunicator),
            ("musa", cuda_communicator, CudaCommunicator),
            ("cpu", cpu_communicator, CpuCommunicator),
            ("mps", cpu_communicator, CpuCommunicator),
            ("npu", cpu_communicator, CpuCommunicator),
            ("xpu", cpu_communicator, CpuCommunicator),
        )
        for platform_name, qualname, communicator_cls in cases:
            platform = MagicMock(device_name=platform_name)
            platform.get_all_to_all_communicator_cls.return_value = qualname
            with (
                self.subTest(platform=platform_name),
                patch.object(group_coordinator, "current_platform", platform),
                patch.object(
                    group_coordinator,
                    "resolve_obj_by_qualname",
                    return_value=communicator_cls,
                ) as resolve_obj_by_qualname,
            ):
                self.assertIs(
                    group_coordinator._resolve_all_to_all_communicator_cls(),
                    communicator_cls,
                )
                resolve_obj_by_qualname.assert_called_once_with(qualname)

    def test_builtin_overrides_keep_non_cuda_all_to_all_on_cpu(self):
        from sglang.multimodal_gen.runtime.platforms.mps import MpsPlatform
        from sglang.multimodal_gen.runtime.platforms.npu import NPUPlatformBase
        from sglang.multimodal_gen.runtime.platforms.xpu import XpuPlatform

        for platform_cls in (MpsPlatform, NPUPlatformBase, XpuPlatform):
            with self.subTest(platform=platform_cls.__name__):
                with patch.object(
                    group_coordinator, "current_platform", platform_cls()
                ):
                    self.assertIs(
                        group_coordinator._resolve_all_to_all_communicator_cls(),
                        CpuCommunicator,
                    )

    def test_platform_selects_all_to_all_communicator(self):
        platform = MagicMock()
        platform.get_all_to_all_communicator_cls.return_value = "vendor.Communicator"

        with (
            patch.object(group_coordinator, "current_platform", platform),
            patch.object(
                group_coordinator,
                "resolve_obj_by_qualname",
                return_value=_TestCommunicator,
            ),
        ):
            self.assertIs(
                group_coordinator._resolve_all_to_all_communicator_cls(),
                _TestCommunicator,
            )

    def test_rejects_invalid_all_to_all_communicator(self):
        platform = MagicMock()
        platform.get_all_to_all_communicator_cls.return_value = "vendor.Communicator"

        with (
            patch.object(group_coordinator, "current_platform", platform),
            patch.object(
                group_coordinator, "resolve_obj_by_qualname", return_value=object
            ),
            self.assertRaisesRegex(TypeError, "DeviceCommunicatorBase subclass"),
        ):
            group_coordinator._resolve_all_to_all_communicator_cls()

    def test_platform_controls_distributed_device_id(self):
        device_id = object()
        for supported in (False, True):
            platform = MagicMock(device_name="test")
            platform.get_torch_distributed_backend_str.return_value = "gloo"
            platform.supports_distributed_device_id.return_value = supported

            with (
                self.subTest(supported=supported),
                patch.object(runtime_platforms, "_current_platform", platform),
                patch.object(parallel_state, "_WORLD", SimpleNamespace(world_size=1)),
                patch.object(
                    parallel_state.torch.distributed,
                    "is_initialized",
                    return_value=False,
                ),
                patch.object(
                    parallel_state.torch.distributed, "init_process_group"
                ) as init_process_group,
                patch.object(
                    parallel_state.torch.distributed,
                    "get_world_size",
                    return_value=1,
                ),
                patch.object(parallel_state, "_sync_srt_world_group"),
            ):
                parallel_state.init_distributed_environment(device_id=device_id)

            kwargs = init_process_group.call_args.kwargs
            if supported:
                self.assertIs(kwargs["device_id"], device_id)
            else:
                self.assertNotIn("device_id", kwargs)


if __name__ == "__main__":
    unittest.main()
