# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import sglang.multimodal_gen.runtime.platforms as runtime_platforms
from sglang.multimodal_gen.runtime.distributed import group_coordinator, parallel_state
from sglang.multimodal_gen.runtime.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from sglang.multimodal_gen.runtime.distributed.device_communicators.cpu_communicator import (
    CpuCommunicator,
)
from sglang.multimodal_gen.runtime.layers import custom_op
from sglang.multimodal_gen.runtime.layers.activation import SiluAndMul
from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
from sglang.multimodal_gen.runtime.managers import gpu_worker
from sglang.multimodal_gen.runtime.platforms.interface import (
    Platform,
    PlatformEnum,
)


class _OotPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name = "test"
    device_type = "test"

    def get_dispatch_key_name(self) -> str:
        return "test"


class _DispatchKeyOotPlatform(_OotPlatform):
    dispatch_key = "PrivateUse1"


class _ExistingCommunicatorPlatform(_OotPlatform):
    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "test.LegacyCommunicator"


class _TestOp(CustomOp):
    def forward_native(self, value):
        return ("native", value)


class _CudaCompatibleTestOp(_TestOp):
    def forward_cuda(self, value):
        return ("cuda", value)


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
            self.assertEqual(op(7), ("registered", 7))
        forward.assert_called_once_with(op, 7)

    def test_missing_registration_uses_native_fallback(self):
        CustomOp.register_oot_forward(_TestOp, fn=MagicMock(), platform_key="other")
        with patch.object(custom_op.platforms, "_current_platform", _OotPlatform()):
            op = _TestOp()
            self.assertEqual(op(7), ("native", 7))

    def test_compiled_forward_does_not_recompile_after_dispatch(self):
        def oot_forward(op, value):
            return op.forward_native(value)

        CustomOp.register_oot_forward(SiluAndMul, fn=oot_forward, platform_key="test")
        cases = (
            (SiluAndMul, torch.randn(2, 8)),
            (lambda: RMSNorm(4), torch.randn(2, 4)),
        )

        for factory, value in cases:
            with self.subTest(op=factory):
                compile_count = 0

                def counting_backend(graph_module, _example_inputs):
                    nonlocal compile_count
                    compile_count += 1
                    return graph_module.forward

                with patch.object(
                    custom_op.platforms, "_current_platform", _OotPlatform()
                ):
                    op = factory()
                selected_forward = op._forward_method
                compiled = torch.compile(op, backend=counting_backend, fullgraph=True)
                expected = op.forward_native(value)
                torch.testing.assert_close(compiled(value), expected)
                torch.testing.assert_close(compiled(value), expected)

                self.assertEqual(compile_count, 1)
                self.assertIs(op._forward_method, selected_forward)

    def test_platform_dispatch_key_can_reuse_an_existing_forward(self):
        platform = _OotPlatform()
        platform.get_dispatch_key_name = lambda: "cuda"
        with patch.object(custom_op.platforms, "_current_platform", platform):
            self.assertEqual(_CudaCompatibleTestOp()(7), ("cuda", 7))

    def test_platform_dispatch_key_must_be_nonempty(self):
        platform = _OotPlatform()
        platform.get_dispatch_key_name = lambda: " "
        with (
            patch.object(custom_op.platforms, "_current_platform", platform),
            self.assertRaisesRegex(ValueError, "non-empty"),
        ):
            _TestOp()(7)


class TestOotBackendInit(unittest.TestCase):
    def setUp(self):
        state = (
            runtime_platforms._backend_init_done,
            runtime_platforms._backend_init_error,
        )
        self.addCleanup(self._restore_backend_state, state)
        runtime_platforms._backend_init_done = False
        runtime_platforms._backend_init_error = None

    @staticmethod
    def _restore_backend_state(state):
        (
            runtime_platforms._backend_init_done,
            runtime_platforms._backend_init_error,
        ) = state

    def test_backend_initialization_runs_once(self):
        platform = _OotPlatform()
        with (
            patch.object(runtime_platforms, "_current_platform", platform),
            patch.object(platform, "init_backend") as init_backend,
        ):
            runtime_platforms.initialize_current_platform()
            runtime_platforms.initialize_current_platform()

        init_backend.assert_called_once_with()

    def test_backend_initialization_failure_is_not_retried(self):
        platform = _OotPlatform()
        error = RuntimeError("backend unavailable")
        with (
            patch.object(runtime_platforms, "_current_platform", platform),
            patch.object(platform, "init_backend", side_effect=error) as init_backend,
        ):
            with self.assertRaisesRegex(RuntimeError, "backend unavailable"):
                runtime_platforms.initialize_current_platform()
            with self.assertRaisesRegex(RuntimeError, "previously failed"):
                runtime_platforms.initialize_current_platform()

        init_backend.assert_called_once_with()

    def test_interrupted_backend_initialization_is_not_reported_as_success(self):
        """An interrupt must leave the process failed, not silently initialized."""
        platform = _OotPlatform()
        with (
            patch.object(runtime_platforms, "_current_platform", platform),
            patch.object(
                platform, "init_backend", side_effect=KeyboardInterrupt
            ) as init_backend,
        ):
            with self.assertRaises(KeyboardInterrupt):
                runtime_platforms.initialize_current_platform()
            with self.assertRaisesRegex(RuntimeError, "previously failed"):
                runtime_platforms.initialize_current_platform()

        init_backend.assert_called_once_with()

    def test_worker_runs_init_backend_before_building_the_scheduler(self):
        order = []
        platform = MagicMock()
        platform.is_cuda.return_value = False
        platform.is_musa.return_value = False

        with (
            patch.object(gpu_worker, "current_platform", platform),
            patch.object(
                gpu_worker,
                "initialize_current_platform",
                side_effect=lambda: order.append("init_backend"),
            ),
            patch.object(gpu_worker, "kill_itself_when_parent_died"),
            patch.object(gpu_worker, "configure_logger"),
            patch.object(gpu_worker, "globally_suppress_loggers"),
            patch.object(
                gpu_worker,
                "init_diffusion_tracing",
                side_effect=lambda *a, **k: order.append("tracing"),
            ),
            patch.object(
                gpu_worker.PortArgs,
                "from_server_args",
                side_effect=RuntimeError("stop before Scheduler"),
            ),
            self.assertRaisesRegex(RuntimeError, "stop before Scheduler"),
        ):
            gpu_worker.run_scheduler_process(
                local_rank=0,
                rank=0,
                master_port=0,
                server_args=MagicMock(),
                pipe_writer=None,
                task_pipe_r=None,
                result_pipe_w=None,
            )

        self.assertEqual(order, ["init_backend", "tracing"])


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

    def test_existing_communicator_override_remains_the_fallback(self):
        self.assertEqual(
            _ExistingCommunicatorPlatform.get_all_to_all_communicator_cls(),
            "test.LegacyCommunicator",
        )


class TestOotRuntimeHooks(unittest.TestCase):
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
            ) as resolve_obj_by_qualname,
        ):
            self.assertIs(
                group_coordinator._resolve_all_to_all_communicator_cls(),
                _TestCommunicator,
            )

        resolve_obj_by_qualname.assert_called_once_with("vendor.Communicator")

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
