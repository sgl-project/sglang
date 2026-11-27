"""Check SAIL DeepEP FABRIC isolation and the dispatch dtype state switch."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.sail.moe import deepep as sail_deepep
from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.layers.moe.token_dispatcher import deepep
from sglang.srt.layers.moe.utils import DeepEPMode, DispatcherOutputDtype
from sglang.srt.runtime_context import get_context, get_resources
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _PlainBuffer:
    @staticmethod
    def get_low_latency_rdma_size_hint(tokens, hidden, ranks, experts):
        return tokens * hidden * ranks * experts

    def __init__(
        self,
        group,
        nvl_bytes,
        rdma_bytes,
        *,
        low_latency_mode,
        num_qps_per_rank,
        allow_mnnvl,
    ):
        self.group = group
        self.nvl_bytes = nvl_bytes
        self.rdma_bytes = rdma_bytes
        self.low_latency_mode = low_latency_mode
        self.num_qps_per_rank = num_qps_per_rank
        self.allow_mnnvl = allow_mnnvl


class _FabricBuffer(_PlainBuffer):
    def __init__(self, *args, use_fabric=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_fabric = use_fabric


class TestDeepEPFabric(CustomTestCase):
    def _build_ll_buffer(self, buffer_facade, buffer_type, *extra_patches):
        group = SimpleNamespace(size=lambda: 2)
        with ExitStack() as stack:
            for context in (
                patch.dict(get_resources().buffers, {}, clear=True),
                patch.object(deepep, "Buffer", buffer_type, create=True),
                patch.object(deepep, "_is_npu", False),
                patch.object(deepep, "_set_nvshmem_qp_depth"),
                patch.object(
                    torch.cuda,
                    "get_device_properties",
                    return_value=SimpleNamespace(multi_processor_count=32),
                ),
                *extra_patches,
            ):
                stack.enter_context(context)
            buffer = buffer_facade.get_deepep_buffer(
                group,
                hidden_size=128,
                param_bytes=2,
                deepep_mode=DeepEPMode.LOW_LATENCY,
                num_max_dispatch_tokens_per_rank=64,
                num_experts=4,
            )
            self.assertIs(buffer.group, group)
            self.assertEqual(buffer.nvl_bytes, 0)
            self.assertEqual(buffer.rdma_bytes, 64 * 128 * 2 * 4)
            self.assertEqual(buffer.num_qps_per_rank, 2)
            self.assertTrue(buffer.low_latency_mode)
            # The cache stays in the runtime resources; it is not rebuilt for
            # later call arguments.
            self.assertIs(
                buffer_facade.get_deepep_buffer(None, 0, 0, DeepEPMode.NORMAL),
                buffer,
            )
        return buffer

    def test_shared_buffer_keys_use_fabric_on_the_cuda_major_version(self):
        """Upstream builds: cu13 hybrid-ep takes use_fabric, cu12 fzyzcjy/DeepEP
        has no such kwarg and enables fabric in C++."""
        for cuda_major, buffer_type in ((12, _PlainBuffer), (13, _FabricBuffer)):
            for advertised in (False, True):
                with self.subTest(cuda_major=cuda_major, advertised=advertised):
                    buffer = self._build_ll_buffer(
                        deepep.DeepEPBuffer,
                        buffer_type,
                        patch.object(
                            deepep, "get_cuda_version", return_value=(cuda_major, 0)
                        ),
                        patch.object(
                            deepep,
                            "_is_mnnvl_fabric_supported",
                            return_value=advertised,
                        ),
                    )
                    self.assertEqual(buffer.allow_mnnvl, advertised)
                    if cuda_major != 12:
                        self.assertEqual(buffer.use_fabric, advertised)

    def test_sail_buffer_kwargs_follow_installed_signature(self):
        """A SAIL vendor without a use_fabric parameter must still be able to create
        an LL buffer, even when FABRIC is advertised."""
        for buffer_type in (_PlainBuffer, _FabricBuffer):
            supports_fabric = sail_deepep._buffer_supports_use_fabric(buffer_type)
            self.assertEqual(supports_fabric, buffer_type is _FabricBuffer)
            for advertised in (False, True):
                with self.subTest(
                    buffer_type=buffer_type.__name__, advertised=advertised
                ):
                    buffer = self._build_ll_buffer(
                        sail_deepep.SailDeepEPBuffer,
                        buffer_type,
                        patch.object(
                            sail_deepep, "DEEPEP_SUPPORT_USE_FABRIC", supports_fabric
                        ),
                        patch.object(
                            sail_deepep, "DEEPEP_SUPPORT_TIMEOUT_CONTROL", False
                        ),
                        envs.SGLANG_SAIL_MNNVL_FABRIC_SUPPORTED.override(advertised),
                    )
                    self.assertEqual(buffer.allow_mnnvl, advertised)
                    if supports_fabric:
                        self.assertEqual(buffer.use_fabric, advertised)

    def test_uninspectable_vendor_signature_keeps_default_behavior(self):
        """An optional extension that cannot be inspected must neither block the module
        import nor let us guess support for an unknown keyword."""
        for error in (TypeError, ValueError):
            with (
                self.subTest(error=error),
                patch.object(sail_deepep.inspect, "signature", side_effect=error),
            ):
                self.assertFalse(sail_deepep._buffer_supports_use_fabric(object))

    def test_sail_override_without_flashinfer(self):
        """SAIL reads its own fabric flag instead of probing NVIDIA FABRIC."""
        with patch.object(
            deepep,
            "is_flashinfer_available",
            side_effect=AssertionError("SAIL must not probe NVIDIA FABRIC"),
        ):
            with envs.SGLANG_SAIL_MNNVL_FABRIC_SUPPORTED.override(False):
                self.assertFalse(sail_deepep.SailDeepEPBuffer.mnnvl_fabric_supported())
            with envs.SGLANG_SAIL_MNNVL_FABRIC_SUPPORTED.override(True):
                self.assertTrue(sail_deepep.SailDeepEPBuffer.mnnvl_fabric_supported())

    def test_sail_fabric_flag_does_not_leak_into_the_shared_buffer(self):
        with (
            patch.object(deepep, "is_flashinfer_available", return_value=False),
            envs.SGLANG_SAIL_MNNVL_FABRIC_SUPPORTED.override(True),
        ):
            self.assertFalse(deepep.DeepEPBuffer.mnnvl_fabric_supported())


class TestDeepEPDispatcherDtype(CustomTestCase):
    def _dispatcher(self, *, sail=False, npu=False):
        self.enterContext(
            get_context().override_server_args(deepep_dispatcher_output_dtype="auto")
        )
        self.enterContext(envs.SGLANG_DEEPEP_BF16_DISPATCH.override(False))
        self.enterContext(
            envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(64)
        )
        self.enterContext(patch.object(moe_utils, "_is_npu", npu))
        self.enterContext(patch.object(deepep, "use_deepep", True))
        self.enterContext(patch.object(deepep, "_is_npu", npu))
        impl_cls = (
            sail_deepep.SailDeepEPDispatcherImplBase
            if sail
            else deepep._DeepEPDispatcherImplBase
        )
        return impl_cls(
            group=None,
            router_topk=2,
            permute_fusion=False,
            num_experts=4,
            num_local_experts=4,
            hidden_size=128,
            params_dtype=torch.bfloat16,
            deepep_mode=DeepEPMode.NORMAL,
        )

    def test_sail_dtype_switch_clears_previous_transport_flags(self):
        """A missing key must not break initialization, and switching dtype must not
        leave INT8/FP4 state behind."""
        dispatcher = self._dispatcher(sail=True)
        flag_names = ("use_fp8", "use_int8", "use_mxfp4", "use_mxfp8", "use_nvfp4")
        for dtype, active in (
            ("int8", "use_int8"),
            ("bf16", None),
            ("uint8", "use_mxfp4"),
            ("fp8", "use_fp8"),
            ("mxfp4", "use_mxfp4"),
            ("mxfp8", "use_mxfp8"),
            ("bf16", None),
        ):
            with self.subTest(dtype=dtype):
                dispatcher.set_quant_config({"dispatcher_output_dtype": dtype})
                self.assertEqual(dispatcher.deepep_output_dtype.value, dtype)
                self.assertEqual(
                    {name for name in flag_names if getattr(dispatcher, name)},
                    {active} if active else set(),
                )
        with self.assertRaises(RuntimeError):
            dispatcher.set_quant_config({"dispatcher_output_dtype": "nvfp4"})

    def test_nvidia_int8_fallback_remains_fp8(self):
        dispatcher = self._dispatcher()
        dispatcher.set_quant_config({"dispatcher_output_dtype": "int8"})
        self.assertEqual(dispatcher.deepep_output_dtype, DispatcherOutputDtype.FP8)
        self.assertTrue(dispatcher.use_fp8)
        self.assertFalse(dispatcher.use_int8)
        self.assertFalse(dispatcher.use_mxfp4)

    def test_npu_int8_reuses_fp8_transport_without_stale_flags(self):
        dispatcher = self._dispatcher(npu=True)
        dispatcher.set_quant_config({"dispatcher_output_dtype": "int8"})
        self.assertEqual(dispatcher.deepep_output_dtype, DispatcherOutputDtype.INT8)
        self.assertTrue(dispatcher.use_fp8)
        self.assertFalse(dispatcher.use_int8)
        dispatcher.set_quant_config({"dispatcher_output_dtype": "bf16"})
        self.assertFalse(dispatcher.use_fp8)
        self.assertFalse(dispatcher.use_int8)

    def test_dispatcher_installs_sail_impls_only_on_sail(self):
        for sail, expected in (
            (
                True,
                (
                    sail_deepep.SailDeepEPDispatcherImplNormal,
                    sail_deepep.SailDeepEPDispatcherImplLowLatency,
                ),
            ),
            (
                False,
                (
                    deepep._DeepEPDispatcherImplNormal,
                    deepep._DeepEPDispatcherImplLowLatency,
                ),
            ),
        ):
            with (
                self.subTest(sail=sail),
                get_context().override_server_args(
                    deepep_dispatcher_output_dtype="auto"
                ),
                envs.SGLANG_DEEPEP_BF16_DISPATCH.override(False),
                envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(64),
                patch.object(deepep, "use_deepep", True),
                patch.object(deepep, "_is_npu", False),
                patch.object(deepep, "is_sail", return_value=sail),
            ):
                dispatcher = deepep.DeepEPDispatcher(
                    group=None,
                    router_topk=2,
                    num_experts=4,
                    num_local_experts=4,
                    hidden_size=128,
                    params_dtype=torch.bfloat16,
                    deepep_mode=DeepEPMode.AUTO,
                )

            self.assertIsInstance(dispatcher._normal_dispatcher, expected[0])
            self.assertIsInstance(dispatcher._low_latency_dispatcher, expected[1])


if __name__ == "__main__":
    import unittest

    unittest.main()
