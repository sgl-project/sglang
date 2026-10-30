"""Check SAIL DeepEP FABRIC isolation and the dispatch dtype state switch."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.layers.moe.token_dispatcher import deepep
from sglang.srt.layers.moe.utils import DeepEPMode, DispatcherOutputDtype
from sglang.srt.runtime_context import get_context, get_resources
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDeepEPFabric(CustomTestCase):
    def test_buffer_kwargs_follow_installed_signature_and_device_capability(self):
        """A vendor without a use_fabric parameter must still be able to create an LL
        buffer, even when FABRIC is advertised."""

        class PlainBuffer:
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

        class FabricBuffer(PlainBuffer):
            def __init__(self, *args, use_fabric=False, **kwargs):
                super().__init__(*args, **kwargs)
                self.use_fabric = use_fabric

        group = SimpleNamespace(size=lambda: 2)
        for buffer_type in (PlainBuffer, FabricBuffer):
            supports_fabric = deepep._buffer_supports_use_fabric(buffer_type)
            self.assertEqual(supports_fabric, buffer_type is FabricBuffer)
            for advertised in (False, True):
                with (
                    self.subTest(
                        buffer_type=buffer_type.__name__, advertised=advertised
                    ),
                    patch.dict(get_resources().buffers, {}, clear=True),
                    patch.object(deepep, "Buffer", buffer_type, create=True),
                    patch.object(deepep, "DEEPEP_SUPPORT_USE_FABRIC", supports_fabric),
                    patch.object(deepep, "DEEPEP_SUPPORT_TIMEOUT_CONTROL", False),
                    patch.object(deepep, "_is_npu", False),
                    patch.object(deepep, "_set_nvshmem_qp_depth"),
                    patch.object(
                        deepep, "_is_mnnvl_fabric_supported", return_value=advertised
                    ),
                    patch.object(
                        torch.cuda,
                        "get_device_properties",
                        return_value=SimpleNamespace(multi_processor_count=32),
                    ),
                ):
                    buffer = deepep.DeepEPBuffer.get_deepep_buffer(
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
                    self.assertEqual(buffer.allow_mnnvl, advertised)
                    if supports_fabric:
                        self.assertEqual(buffer.use_fabric, advertised)
                    # The cache stays in the runtime resources; it is not rebuilt
                    # for later call arguments.
                    self.assertIs(
                        deepep.DeepEPBuffer.get_deepep_buffer(
                            None, 0, 0, DeepEPMode.NORMAL
                        ),
                        buffer,
                    )

    def test_uninspectable_vendor_signature_keeps_default_behavior(self):
        """An optional extension that cannot be inspected must neither block the module
        import nor let us guess support for an unknown keyword."""
        for error in (TypeError, ValueError):
            with (
                self.subTest(error=error),
                patch.object(deepep.inspect, "signature", side_effect=error),
            ):
                self.assertFalse(deepep._buffer_supports_use_fabric(object))

    def test_sail_override_without_flashinfer(self):
        with (
            patch.object(deepep, "_is_sail", True),
            patch.object(
                deepep,
                "is_flashinfer_available",
                side_effect=AssertionError("SAIL must not probe NVIDIA FABRIC"),
            ),
        ):
            with envs.SGLANG_SAIL_MNNVL_FABRIC_SUPPORTED.override(False):
                self.assertFalse(deepep._is_mnnvl_fabric_supported())
            with envs.SGLANG_SAIL_MNNVL_FABRIC_SUPPORTED.override(True):
                self.assertTrue(deepep._is_mnnvl_fabric_supported())

    def test_sail_override_does_not_enable_other_platforms(self):
        with (
            patch.object(deepep, "_is_sail", False),
            patch.object(deepep, "is_flashinfer_available", return_value=False),
            envs.SGLANG_SAIL_MNNVL_FABRIC_SUPPORTED.override(True),
        ):
            self.assertFalse(deepep._is_mnnvl_fabric_supported())


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
        self.enterContext(patch.object(deepep, "_is_sail", sail))
        self.enterContext(patch.object(deepep, "_is_npu", npu))
        return deepep._DeepEPDispatcherImplBase(
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


if __name__ == "__main__":
    import unittest

    unittest.main()
