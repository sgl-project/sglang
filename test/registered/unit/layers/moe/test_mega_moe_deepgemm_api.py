"""Unit tests for the DeepGEMM MegaMoE interface."""

import sys
import unittest
from contextlib import ExitStack, contextmanager, nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch

from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.hardware_backend.sail.moe import mega_gate as sail_mega_gate
from sglang.srt.hardware_backend.sail.moe import mega_moe as sail_mega_moe
from sglang.srt.hardware_backend.sail.quantization import mxfp4 as sail_mxfp4
from sglang.srt.layers.moe import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.moe import mega_gate as mega_gate_runtime
from sglang.srt.layers.moe import mega_moe
from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.fused_moe_triton import layer as fused_moe_layer_module
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.moe.utils import draft_model_build_scope
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import (
    get_context,
    get_exec,
    get_flags,
    get_model,
    get_parallel,
)
from sglang.srt.utils import is_sail

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestDeepGemmMegaMoeApi(CustomTestCase):
    def setUp(self):
        super().setUp()
        platform = patch.object(mega_moe, "_is_sail", False)
        platform.start()
        self.addCleanup(platform.stop)
        stack = ExitStack()
        self.addCleanup(stack.close)
        for var in (
            envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A8,
            envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16,
            envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16_MMA,
        ):
            stack.enter_context(var.override(False))
        mega_moe._MEGA_MOE_SYMM_BUFFER.clear()
        sail_mega_moe._MEGA_MOE_SYMM_BUFFER.clear()
        self.original_max_num_sms = mega_moe._mega_moe_max_num_sms
        self.original_sail_max_num_sms = sail_mega_moe._mega_moe_max_num_sms
        self.deep_gemm = ModuleType("deep_gemm")
        self.deep_gemm.num_sms = 132
        self.deep_gemm.get_num_sms = MagicMock(
            side_effect=lambda: self.deep_gemm.num_sms
        )
        self.deep_gemm.set_num_sms = MagicMock(
            side_effect=lambda num_sms: setattr(self.deep_gemm, "num_sms", num_sms)
        )
        self.deep_gemm.destroy_localizer = MagicMock()
        max_num_sms = patch.object(mega_moe, "_mega_moe_max_num_sms", return_value=130)
        self.max_num_sms = max_num_sms.start()
        self.addCleanup(max_num_sms.stop)
        self.sail_max_num_sms = stack.enter_context(
            patch.object(sail_mega_moe, "_mega_moe_max_num_sms", return_value=130)
        )
        stack.enter_context(patch.object(sail_mega_moe, "_device_sm", 89))

    def tearDown(self):
        mega_moe._MEGA_MOE_SYMM_BUFFER.clear()
        sail_mega_moe._MEGA_MOE_SYMM_BUFFER.clear()
        super().tearDown()

    def test_sail_precision_aliases_share_a_workspace(self):
        self.deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(
            side_effect=lambda *args, **kwargs: object()
        )
        group = object()
        with (
            patch.dict(sys.modules, {"deep_gemm": self.deep_gemm}),
            patch.object(mega_moe, "_is_sail", True),
        ):
            for framework, vendor in (
                ("mxf4xmxf4", "fp4xfp4"),
                ("fp8xmxfp4", "fp8xfp4"),
                ("bf16xmxfp4", "bf16xfp4"),
            ):
                with self.subTest(framework=framework):
                    buffers = [
                        mega_moe._get_mega_moe_symm_buffer(
                            group, 8, 64, 2, 128, 256, mma_type=kind
                        )
                        for kind in (framework, vendor)
                    ]
                    self.assertIs(buffers[0], buffers[1])
                    self.assertEqual(
                        self.deep_gemm.get_symm_buffer_for_mega_moe.call_args.kwargs[
                            "mma_type"
                        ],
                        vendor,
                    )
        self.assertEqual(self.deep_gemm.get_symm_buffer_for_mega_moe.call_count, 3)

    def test_mma_type_follows_layer_w4a4_mode(self):
        experts = SimpleNamespace(
            _mega_moe_nvfp4=False,
            _mega_moe_w4a4=False,
            w13_weight=torch.empty(0, dtype=torch.uint8),
        )
        with patch.object(mega_moe, "_is_sail", False):
            self.assertEqual(mega_moe._mega_moe_mma_type(experts), "fp8xfp4")
            experts._mega_moe_w4a4 = True
            self.assertEqual(mega_moe._mega_moe_mma_type(experts), "mxf4xmxf4")

    def test_ppu_mma_type_follows_transformed_weights(self):
        experts = SimpleNamespace(
            _mega_moe_nvfp4=False,
            _mega_moe_w4a4=False,
            _mega_moe_mma_type="mxf4xmxf4",
            w13_weight=torch.empty(0, dtype=torch.uint8),
        )
        with patch.object(mega_moe, "_is_sail", True):
            self.assertEqual(mega_moe._mega_moe_mma_type(experts), "mxf4xmxf4")
            experts._mega_moe_mma_type = "fp8xmxfp4"
            self.assertEqual(mega_moe._mega_moe_mma_type(experts), "fp8xmxfp4")
        with patch.object(mega_moe, "_is_sail", False):
            self.assertEqual(mega_moe._mega_moe_mma_type(experts), "fp8xfp4")

    def test_mxf4_buffer_uses_typed_api(self):
        deep_gemm = self.deep_gemm
        expected_buffer = object()
        deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(return_value=expected_buffer)
        group = object()

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch.object(mega_moe, "_is_sail", False),
        ):
            actual_buffer = mega_moe._get_mega_moe_symm_buffer(
                group,
                num_experts=8,
                num_max_tokens_per_rank=64,
                num_topk=2,
                hidden=128,
                intermediate_hidden=256,
                mma_type="mxf4xmxf4",
            )

        self.assertIs(actual_buffer, expected_buffer)
        call = deep_gemm.get_symm_buffer_for_mega_moe.call_args
        self.assertEqual(call.kwargs.get("mma_type"), "mxf4xmxf4")
        self.assertNotIn("use_fp8_dispatch", call.kwargs)

    def test_ppu_buffer_uses_mma_type_without_legacy_dispatch_flag(self):
        deep_gemm = self.deep_gemm
        deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(
            side_effect=(object(), object(), object())
        )
        group = object()

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch.object(mega_moe, "_is_sail", True),
        ):
            buffers = [
                mega_moe._get_mega_moe_symm_buffer(
                    group, 8, 64, 2, 128, 256, mma_type=mma_type, activation="situ"
                )
                for mma_type in (
                    "fp8xmxfp4",
                    "mxf4xmxf4",
                    "fp8xmxfp4",
                    "bf16xmxfp4",
                    "fp8xfp4",
                    "fp4xfp4",
                    "bf16xfp4",
                )
            ]

        self.assertIs(buffers[0], buffers[2])
        self.assertIsNot(buffers[0], buffers[1])
        self.assertIsNot(buffers[0], buffers[3])
        self.assertIs(buffers[0], buffers[4])
        self.assertIs(buffers[1], buffers[5])
        self.assertIs(buffers[3], buffers[6])
        self.assertEqual(
            deep_gemm.get_symm_buffer_for_mega_moe.call_args_list,
            [
                call(
                    group,
                    8,
                    64,
                    2,
                    128,
                    256,
                    num_shared_experts=0,
                    mma_type=mma_type,
                    activation="situ",
                )
                for mma_type in ("fp8xfp4", "fp4xfp4", "bf16xfp4")
            ],
        )
        deep_gemm.get_num_sms.assert_called()
        self.assertEqual(deep_gemm.num_sms, 132)

    def test_ppu_buffer_cache_separates_situ_activation(self):
        deep_gemm = self.deep_gemm
        deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(
            side_effect=(object(), object())
        )
        group = object()
        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch.object(mega_moe, "_is_sail", True),
        ):
            buffers = [
                mega_moe._get_mega_moe_symm_buffer(
                    group,
                    8,
                    64,
                    2,
                    128,
                    256,
                    mma_type="mxf4xmxf4",
                    activation=activation,
                )
                for activation in ("swiglu", "situ", "swiglu")
            ]
        self.assertIs(buffers[0], buffers[2])
        self.assertIsNot(buffers[0], buffers[1])
        self.assertEqual(
            [
                call.kwargs["activation"]
                for call in deep_gemm.get_symm_buffer_for_mega_moe.call_args_list
            ],
            ["swiglu", "situ"],
        )

    def test_ppu_sm89_and_newer_use_physical_sm_budget(self):
        with (
            patch.object(mega_moe, "_is_sail", True),
            patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(multi_processor_count=140),
            ),
            patch.object(
                mega_moe.envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS,
                "get",
                return_value=8,
            ),
        ):
            for sm in (89, 90):
                with patch.object(sail_mega_moe, "_device_sm", sm):
                    self.assertEqual(self.original_sail_max_num_sms.__wrapped__(), 132)
        with (
            patch.object(mega_moe, "_is_sail", False),
            patch.object(mega_moe, "_device_sm", 89),
        ):
            self.assertIsNone(self.original_max_num_sms.__wrapped__())

    def test_sail_sm89_keeps_odd_sm_budget_and_restores_on_failure(self):
        """SM89 allows odd CTA counts and restores the SM budget on failure."""
        self.deep_gemm.num_sms = 131
        self.sail_max_num_sms.return_value = 129
        with (
            patch.object(mega_moe, "_is_sail", True),
            patch.object(sail_mega_moe, "_device_sm", 89),
            self.assertRaisesRegex(RuntimeError, "compute failed"),
        ):
            with mega_moe._configure_mega_moe_deep_gemm_num_sms(self.deep_gemm):
                self.assertEqual(self.deep_gemm.num_sms, 129)
                raise RuntimeError("compute failed")
        self.assertEqual(self.deep_gemm.num_sms, 131)

    def test_buffer_cache_separates_mma_types(self):
        deep_gemm = self.deep_gemm
        expected_buffers = (object(), object())
        deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(side_effect=expected_buffers)
        group = object()

        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}):
            actual_buffers = tuple(
                mega_moe._get_mega_moe_symm_buffer(
                    group,
                    num_experts=8,
                    num_max_tokens_per_rank=64,
                    num_topk=2,
                    hidden=128,
                    intermediate_hidden=256,
                    mma_type=mma_type,
                )
                for mma_type in ("fp8xfp4", "mxf4xmxf4")
            )

        self.assertEqual(actual_buffers, expected_buffers)
        self.assertEqual(deep_gemm.get_symm_buffer_for_mega_moe.call_count, 2)
        self.assertEqual(
            [
                call.kwargs["mma_type"]
                for call in deep_gemm.get_symm_buffer_for_mega_moe.call_args_list
            ],
            ["fp8xfp4", "mxf4xmxf4"],
        )

    def test_buffer_cache_uses_effective_sm_budget(self):
        deep_gemm = self.deep_gemm
        deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(
            side_effect=lambda *_args, **_kwargs: SimpleNamespace(
                num_sms=deep_gemm.get_num_sms()
            )
        )
        group = object()
        buffers = []

        for current_num_sms, expected_num_sms in (
            (132, 130),
            (131, 130),
            (96, 96),
            (97, 96),
            (132, 130),
        ):
            with self.subTest(current_num_sms=current_num_sms):
                deep_gemm.set_num_sms(current_num_sms)
                buf = self._get_test_buffer(group)
                buffers.append(buf)
                self.assertEqual(buf.num_sms, expected_num_sms)
                self.assertEqual(deep_gemm.get_num_sms(), current_num_sms)
                with mega_moe._configure_mega_moe_deep_gemm_num_sms(deep_gemm):
                    self.assertEqual(buf.num_sms, deep_gemm.get_num_sms())
                self.assertEqual(deep_gemm.get_num_sms(), current_num_sms)

        self.assertIs(buffers[0], buffers[1])
        self.assertIs(buffers[0], buffers[4])
        self.assertIs(buffers[2], buffers[3])
        self.assertIsNot(buffers[0], buffers[2])
        self.assertEqual(deep_gemm.get_symm_buffer_for_mega_moe.call_count, 2)

    def test_buffer_cache_separates_activation_contracts(self):
        self.deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(
            side_effect=lambda *args, **kw: object()
        )
        group = object()
        with patch.dict(sys.modules, {"deep_gemm": self.deep_gemm}):
            buffers = [
                mega_moe._get_mega_moe_symm_buffer(
                    group,
                    num_experts=8,
                    num_max_tokens_per_rank=64,
                    num_topk=2,
                    hidden=128,
                    intermediate_hidden=256,
                    mma_type="nvfp4xnvfp4",
                    activation=activation,
                )
                for activation in ("swiglu", "swigluoai", "swiglu")
            ]
        self.assertIs(buffers[0], buffers[2])
        self.assertIsNot(buffers[0], buffers[1])

    def test_buffer_allocation_without_sm_override(self):
        deep_gemm = self.deep_gemm
        self.max_num_sms.return_value = None
        deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(
            side_effect=lambda *_args, **_kwargs: deep_gemm.get_num_sms()
        )

        buf = self._get_test_buffer(object())
        self.assertEqual(buf, 132)
        deep_gemm.set_num_sms.assert_not_called()

    def test_new_buffer_stops_mlopart_helper_on_sm100_only(self):
        """SM100 must stop the MLOPart helper, which blocks cuda-checkpoint, once per
        new buffer. SM90 must not call it, since that would start the helper."""
        self.deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(
            side_effect=lambda *_args, **_kwargs: object()
        )
        for sm100, expected_calls in ((False, 0), (True, 1)):
            with (
                self.subTest(sm100=sm100),
                patch.object(mega_moe, "is_sm100_supported", return_value=sm100),
            ):
                mega_moe._MEGA_MOE_SYMM_BUFFER.clear()
                self.deep_gemm.destroy_localizer.reset_mock()
                group = object()
                self._get_test_buffer(group)
                self._get_test_buffer(group)
                self.assertEqual(
                    self.deep_gemm.destroy_localizer.call_count, expected_calls
                )

    def test_buffer_allocation_failure_restores_sm_budget(self):
        deep_gemm = self.deep_gemm
        deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(
            side_effect=RuntimeError("allocation failed")
        )

        with self.assertRaisesRegex(RuntimeError, "allocation failed"):
            self._get_test_buffer(object())

        self.assertEqual(deep_gemm.get_num_sms(), 132)
        self.assertEqual(deep_gemm.set_num_sms.call_args_list, [call(130), call(132)])
        self.assertEqual(mega_moe._MEGA_MOE_SYMM_BUFFER, {})

    @unittest.skipIf(
        is_sail(),
        "the PPU MegaMoE path takes per-layer MMA types and its own DeepGEMM APIs",
    )
    def test_mxf4_weight_transform_uses_matching_mma_type(self):
        from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod

        deep_gemm = ModuleType("deep_gemm")
        deep_gemm.transform_sf_into_required_layout = MagicMock(
            side_effect=lambda _sf, mn, k, recipe, num_groups, disable_ue8m0_cast: (
                torch.zeros((num_groups, mn, max(1, k // 32)), dtype=torch.int32)
            )
        )
        deep_gemm.transform_weights_for_mega_moe = MagicMock(
            side_effect=lambda l1, l2, **_kwargs: (l1, l2)
        )
        method = object.__new__(Mxfp4MoEMethod)
        method.use_marlin = False
        method.use_deep_gemm = False
        method.use_mega_moe = True
        layer = SimpleNamespace(
            w13_weight=torch.nn.Parameter(
                torch.zeros((1, 32, 16), dtype=torch.uint8), requires_grad=False
            ),
            w13_weight_scale=torch.nn.Parameter(
                torch.zeros((1, 32, 1), dtype=torch.uint8), requires_grad=False
            ),
            w2_weight=torch.nn.Parameter(
                torch.zeros((1, 32, 16), dtype=torch.uint8), requires_grad=False
            ),
            w2_weight_scale=torch.nn.Parameter(
                torch.zeros((1, 32, 1), dtype=torch.uint8), requires_grad=False
            ),
        )

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch("sglang.srt.layers.quantization.mxfp4._is_sail", False),
            patch.object(mega_moe, "_mega_moe_mma_type", return_value="mxf4xmxf4"),
            # Toy shapes.
            patch.object(mega_moe, "check_mega_moe_shapes"),
        ):
            method.process_weights_after_loading(layer)

        call = deep_gemm.transform_weights_for_mega_moe.call_args
        self.assertEqual(call.kwargs.get("mma_type"), "mxf4xmxf4")

    def test_ppu_mxf4_transform_releases_legacy_layout(self):
        from sglang.srt.layers.quantization import mxfp4

        deep_gemm = ModuleType("deep_gemm")

        def transform(l1, l2):
            return (
                (l1[0].clone(), l1[1].to(torch.uint16)),
                (l2[0], l2[1].to(torch.uint16)),
            )

        deep_gemm.transform_weights_for_mega_moe = MagicMock(side_effect=transform)
        method = object.__new__(mxfp4.Mxfp4MoEMethod)
        method._mega_moe_mma_type = "mxf4xmxf4"
        method.use_marlin = False
        method.use_deep_gemm = False
        method.use_mega_moe = True
        method.use_deepgemm_mxfp4_w4a16_valu = False
        method.use_deepgemm_mxfp4_w4a16_mma = False
        method.with_bias = False
        layer = SimpleNamespace(
            w13_weight=torch.nn.Parameter(
                torch.zeros((1, 32, 16), dtype=torch.uint8), requires_grad=False
            ),
            w13_weight_scale=torch.nn.Parameter(
                torch.full((1, 32, 1), 127, dtype=torch.uint8), requires_grad=False
            ),
            w2_weight=torch.nn.Parameter(
                torch.zeros((1, 32, 16), dtype=torch.uint8), requires_grad=False
            ),
            w2_weight_scale=torch.nn.Parameter(
                torch.full((1, 32, 1), 126, dtype=torch.uint8), requires_grad=False
            ),
        )
        original_weight_ptr = layer.w13_weight.data_ptr()

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch.object(mxfp4, "_is_sail", True),
            patch.object(sail_mxfp4, "preprocess_sail_mxfp4_scales") as preprocess,
            patch.object(torch.cuda, "empty_cache"),
        ):
            method.process_weights_after_loading(layer)
            method.process_weights_after_loading(layer)

        self.assertEqual(deep_gemm.transform_weights_for_mega_moe.call_count, 1)
        l1, l2 = deep_gemm.transform_weights_for_mega_moe.call_args.args
        self.assertEqual(l1[1].dtype, torch.uint8)
        self.assertEqual(l2[1].dtype, torch.uint8)
        self.assertTrue(torch.all(l1[1] == 127))
        self.assertTrue(torch.all(l2[1] == 126))
        self.assertNotEqual(layer.w13_weight.data_ptr(), original_weight_ptr)
        self.assertEqual(
            layer.w13_weight.data_ptr(), layer.mega_l1_weights[0].data_ptr()
        )
        self.assertEqual(
            layer.w13_weight_scale.data_ptr(), layer.mega_l1_weights[1].data_ptr()
        )
        self.assertEqual(
            layer.w2_weight.data_ptr(), layer.mega_l2_weights[0].data_ptr()
        )
        self.assertEqual(
            layer.w2_weight_scale.data_ptr(), layer.mega_l2_weights[1].data_ptr()
        )
        self.assertEqual(layer.w13_weight_scale.dtype, torch.uint16)
        preprocess.assert_not_called()
        self.assertEqual(layer._mxfp4_backend, "deep_gemm")
        self.assertEqual(layer._mega_moe_mma_type, "mxf4xmxf4")
        self.assertTrue(layer._mega_moe_weights_built)

    def test_ppu_mega_moe_precision_is_available_before_weight_postprocessing(self):
        """Kimi preallocates its buffer inside load_weights, before quant repacking."""
        from sglang.srt.layers.quantization import mxfp4

        for precision, expected in (
            ("w4a4", "mxf4xmxf4"),
            ("w4a8", "fp8xmxfp4"),
            ("w4a16", "bf16xmxfp4"),
            ("w4a16_mma", "bf16xmxfp4"),
        ):
            with (
                self.subTest(precision=precision),
                patch.object(mxfp4, "_is_sail", True),
                patch.object(mega_moe, "_is_sail", True),
                get_flags().moe.override(
                    runner_backend=MoeRunnerBackend.DEEP_GEMM,
                    a2a_backend=MoeA2ABackend.MEGAMOE,
                ),
                patch.object(
                    mxfp4,
                    "get_exec",
                    return_value=SimpleNamespace(
                        moe=SimpleNamespace(flashinfer_mxfp4_moe_precision="default")
                    ),
                ),
                envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A8.override(precision == "w4a8"),
                envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16.override(precision == "w4a16"),
                envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16_MMA.override(
                    precision == "w4a16_mma"
                ),
            ):
                method = mxfp4.Mxfp4MoEMethod("model.layers.0.experts")
                layer = torch.nn.Module()
                layer.num_local_experts = 1
                layer.hidden_size = 1024
                layer.intermediate_size_per_partition = 128
                layer.moe_runner_config = SimpleNamespace(activation="situ")
                layer._mega_moe_nvfp4 = False
                layer._mega_moe_w4a4 = False
                layer._mega_moe_weights_built = False
                method.create_weights(layer, 1, 1024, 128, torch.bfloat16)

                self.assertEqual(mega_moe._mega_moe_mma_type(layer), expected)
                self.assertFalse(layer._mega_moe_weights_built)
                self.assertEqual(layer.w13_weight_scale.dtype, torch.uint8)

    def test_ppu_mega_moe_preserves_precision_specific_weight_layouts(self):
        """MegaMoE must see raw scales and retain both W4A8 inverse scales."""
        from sglang.srt.layers.quantization import mxfp4

        for precision in ("w4a8", "w4a16", "w4a16_mma"):
            with self.subTest(precision=precision):
                layer = SimpleNamespace(
                    w13_weight=torch.nn.Parameter(
                        torch.zeros((1, 32, 16), dtype=torch.uint8), requires_grad=False
                    ),
                    w13_weight_scale=torch.nn.Parameter(
                        torch.full((1, 32, 1), 127, dtype=torch.uint8),
                        requires_grad=False,
                    ),
                    w2_weight=torch.nn.Parameter(
                        torch.zeros((1, 32, 16), dtype=torch.uint8), requires_grad=False
                    ),
                    w2_weight_scale=torch.nn.Parameter(
                        torch.full((1, 32, 1), 126, dtype=torch.uint8),
                        requires_grad=False,
                    ),
                )

                def transform(l1, l2):
                    self.assertTrue(torch.all(l1[1] == 127))
                    self.assertTrue(torch.all(l2[1] == 126))
                    if precision == "w4a8":
                        return (*l1, 2.0), (*l2, 4.0)
                    return (l1[0], l1[1].to(torch.uint16)), (
                        l2[0],
                        l2[1].to(torch.uint16),
                    )

                deep_gemm = ModuleType("deep_gemm")
                deep_gemm.transform_weights_for_mega_moe = MagicMock(
                    side_effect=AssertionError("W4A4 transform changes the precision")
                )
                deep_gemm.transform_weights_for_w4a8_mega_moe = transform
                deep_gemm.transform_weights_for_w4a16_mega_moe = transform
                with (
                    patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
                    patch.object(mxfp4, "_is_sail", True),
                    get_flags().moe.override(
                        runner_backend=MoeRunnerBackend.DEEP_GEMM,
                        a2a_backend=MoeA2ABackend.MEGAMOE,
                    ),
                    patch.object(
                        mxfp4,
                        "get_exec",
                        return_value=SimpleNamespace(
                            moe=SimpleNamespace(
                                flashinfer_mxfp4_moe_precision="default"
                            )
                        ),
                    ),
                    envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A8.override(precision == "w4a8"),
                    envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16.override(
                        precision == "w4a16"
                    ),
                    envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16_MMA.override(
                        precision == "w4a16_mma"
                    ),
                    patch.object(torch.cuda, "empty_cache"),
                ):
                    method = mxfp4.Mxfp4MoEMethod("model.layers.0.experts")
                    method.process_weights_after_loading(layer)
                    method.process_weights_after_loading(layer)

                self.assertFalse(method.use_deepgemm_mxfp4_w4a16_valu)
                self.assertFalse(method.use_deepgemm_mxfp4_w4a16_mma)
                self.assertTrue(layer._mega_moe_weights_built)
                if precision == "w4a8":
                    self.assertEqual(layer._mega_moe_mma_type, "fp8xmxfp4")
                    self.assertEqual(layer.mega_l1_weights[2], 2.0)
                    self.assertEqual(layer.mega_l2_weights[2], 4.0)
                else:
                    self.assertEqual(layer._mega_moe_mma_type, "bf16xmxfp4")
                    self.assertEqual(layer.mega_l1_weights[1].dtype, torch.uint16)
                self.assertEqual(
                    layer.w13_weight.data_ptr(), layer.mega_l1_weights[0].data_ptr()
                )
                self.assertEqual(
                    layer.w2_weight_scale.data_ptr(),
                    layer.mega_l2_weights[1].data_ptr(),
                )

    def test_ppu_mxf4_rejects_expert_bias_before_conversion(self):
        from sglang.srt.layers.quantization import mxfp4

        method = object.__new__(mxfp4.Mxfp4MoEMethod)
        method.use_marlin = False
        method.use_deep_gemm = False
        method.use_mega_moe = True
        method.use_deepgemm_mxfp4_w4a16_valu = False
        method.use_deepgemm_mxfp4_w4a16_mma = False
        method.with_bias = True
        method._mega_moe_mma_type = "mxf4xmxf4"
        with patch.object(mxfp4, "_is_sail", True):
            with self.assertRaisesRegex(ValueError, "expert bias"):
                method.process_weights_after_loading(SimpleNamespace())

    @unittest.skipIf(
        is_sail(),
        "the PPU MegaMoE path takes per-layer MMA types and its own DeepGEMM APIs",
    )
    def test_mxf4_pre_dispatch_uses_typed_api(self):
        deep_gemm = ModuleType("deep_gemm")
        deep_gemm.mega_moe_pre_dispatch = MagicMock()
        deep_gemm.fp8_fp4_mega_moe = MagicMock()
        buffer = SimpleNamespace(
            x=object(),
            x_sf=object(),
            topk_idx=object(),
            topk_weights=object(),
        )
        experts = SimpleNamespace(
            num_experts=8,
            mega_l1_weights=object(),
            mega_l2_weights=object(),
            should_fuse_routed_scaling_factor_in_topk=True,
            moe_runner_config=SimpleNamespace(swiglu_limit=None),
            uses_expert_location=False,
            _mega_moe_weights_built=True,
            _mega_moe_nvfp4=False,
        )
        topk_output = SimpleNamespace(
            topk_ids=torch.tensor([[0, 1]]),
            topk_weights=torch.tensor([[0.6, 0.4]]),
        )
        moe = SimpleNamespace(
            config=SimpleNamespace(
                hidden_size=4,
                num_experts_per_tok=2,
                moe_intermediate_size=8,
                swiglu_limit=None,
            ),
            experts=experts,
            gate=MagicMock(
                return_value=torch.empty((1, 8)), e_score_correction_bias_vl=None
            ),
            topk=MagicMock(return_value=topk_output),
            is_hash=False,
            num_fused_shared_experts=0,
            layer_id=0,
            routed_scaling_factor=1.0,
            routed_scaling_factor_fused_in_topk=True,
            mega_shared_l1_weights=None,
            mega_shared_l2_weights=None,
        )

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch.object(mega_moe, "_is_sail", False),
            patch.object(mega_moe, "_device_sm", 100),
            patch.object(mega_moe, "_mega_moe_mma_type", return_value="mxf4xmxf4"),
            patch.object(
                mega_moe,
                "_get_mega_moe_symm_buffer",
                return_value=buffer,
            ),
            patch.object(
                mega_moe,
                "_configure_mega_moe_deep_gemm_num_sms",
                return_value=nullcontext(),
            ),
            # Toy shapes.
            patch.object(mega_moe, "check_mega_moe_shapes"),
            get_parallel().override(
                moe_ep_group=SimpleNamespace(device_group=object())
            ),
        ):
            mega_moe._run_mega_routed(
                moe,
                torch.zeros((1, 4)),
                forward_batch=None,
                input_ids_global=None,
                num_tokens=1,
            )

        self.assertTrue(deep_gemm.mega_moe_pre_dispatch.called)
        call = deep_gemm.mega_moe_pre_dispatch.call_args
        self.assertEqual(call.kwargs.get("mma_type"), "mxf4xmxf4")
        self.assertNotIn("use_fp4_acts", call.kwargs)

    @contextmanager
    def _reject_sail_imports(self):
        """CUDA-compatible device detection must not load the SAIL runtime."""
        import builtins

        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.startswith("sglang.srt.hardware_backend.sail"):
                raise AssertionError(f"CUDA path imported {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            yield

    def test_run_mega_routed_experts_generic_entry(self):
        deep_gemm = ModuleType("deep_gemm")
        deep_gemm.fp8_fp4_mega_moe = MagicMock()
        buffer = SimpleNamespace(
            x=object(),
            x_sf=object(),
            topk_idx=object(),
            topk_weights=object(),
        )
        experts = SimpleNamespace(
            num_experts=8,
            mega_l1_weights=object(),
            mega_l2_weights=object(),
            _mega_moe_weights_built=True,
            _mega_moe_nvfp4=False,
        )
        hidden_states = torch.zeros((3, 4), dtype=torch.bfloat16)
        topk_ids = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int64)
        topk_weights = torch.full((3, 2), 0.5, dtype=torch.bfloat16)

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            self._reject_sail_imports(),
            patch.object(mega_moe, "_device_sm", 100),
            patch.object(mega_moe, "_mega_moe_mma_type", return_value="fp8xfp4"),
            patch.object(mega_moe, "mega_moe_pre_dispatch") as pre_dispatch,
            patch.object(
                mega_moe, "_get_mega_moe_symm_buffer", return_value=buffer
            ) as get_buffer,
            patch.object(
                mega_moe,
                "_configure_mega_moe_deep_gemm_num_sms",
                return_value=nullcontext(),
            ),
            # Toy shapes.
            patch.object(mega_moe, "check_mega_moe_shapes"),
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=SimpleNamespace(
                    moe_ep_group=SimpleNamespace(device_group=object())
                ),
            ),
        ):
            out = mega_moe.run_mega_routed_experts(
                experts,
                hidden_states,
                topk_ids,
                topk_weights,
                hidden_size=4,
                intermediate_size=8,
                top_k=2,
                num_tokens=3,
                activation_clamp=7.0,
                routed_scaling_factor=1.0,
            )
            mega_call = deep_gemm.fp8_fp4_mega_moe.call_args
            eplb_load_row = torch.zeros(4, dtype=torch.int32)
            mega_moe.run_mega_routed_experts(
                experts,
                hidden_states,
                topk_ids,
                topk_weights,
                hidden_size=4,
                intermediate_size=8,
                top_k=2,
                num_tokens=3,
                eplb_load_row=eplb_load_row,
            )
            self.assertIs(
                deep_gemm.fp8_fp4_mega_moe.call_args.kwargs[
                    "cumulative_local_expert_recv_stats"
                ],
                eplb_load_row,
            )

        self.assertEqual(out.shape, (3, 4))
        self.assertEqual(out.dtype, torch.bfloat16)
        buf_call = get_buffer.call_args
        self.assertEqual(buf_call.kwargs.get("num_topk"), 2)
        self.assertEqual(buf_call.kwargs.get("hidden"), 4)
        self.assertEqual(buf_call.kwargs.get("intermediate_hidden"), 8)
        # The kernel wants int32 ids and fp32 weights regardless of the router dtype.
        ids_arg, weights_arg = pre_dispatch.call_args.args[1:3]
        self.assertEqual(ids_arg.dtype, torch.int32)
        self.assertEqual(weights_arg.dtype, torch.float32)
        self.assertIs(mega_call.args[1], experts.mega_l1_weights)
        self.assertIs(mega_call.args[2], experts.mega_l2_weights)
        self.assertEqual(mega_call.kwargs.get("activation_clamp"), 7.0)
        self.assertNotIn("cumulative_local_expert_recv_stats", mega_call.kwargs)

    def test_eplb_load_row_asks_the_recorder_for_this_ranks_slice(self):
        row = torch.zeros(2, dtype=torch.int32)
        recorder = MagicMock()
        recorder.get_current_pass_local_count_buffer.return_value = row
        group = SimpleNamespace(size=lambda: 4, rank=lambda: 2)
        parallel = SimpleNamespace(moe_ep_group=SimpleNamespace(device_group=group))
        moe = SimpleNamespace(
            experts=SimpleNamespace(num_experts=8), num_fused_shared_experts=0
        )
        with (
            patch.object(
                mega_moe,
                "get_global_expert_distribution_recorder",
                return_value=recorder,
            ),
            patch("sglang.srt.runtime_context.get_parallel", return_value=parallel),
            patch.object(mega_moe, "_device_sm", 100),
            patch.object(mega_moe, "_mega_moe_mma_type", return_value="fp8xfp4"),
        ):
            self.assertIs(mega_moe._local_eplb_load_row(moe), row)
            recorder.get_current_pass_local_count_buffer.assert_called_once_with(
                num_experts=8, ep_rank=2, ep_size=4
            )
            for name, value in (
                ("_device_sm", 90),
                ("_mega_moe_mma_type", lambda experts: "mxf4xmxf4"),
            ):
                with patch.object(mega_moe, name, value):
                    self.assertIsNone(mega_moe._local_eplb_load_row(moe))
            shared = SimpleNamespace(experts=moe.experts, num_fused_shared_experts=1)
            self.assertIsNone(mega_moe._local_eplb_load_row(shared))

    def test_v41_megamoe_selects_image_bias_on_local_rows(self):
        """Image rows must retain their router bias after attention-TP sharding.

        Equal logits make the two biases select disjoint experts. A numerical
        expert boundary exposes accidentally routing image rows as text, and
        padded rows must contribute nothing.
        """
        config = get_context().override_server_args(
            model_path="dummy",
            enable_waterfill=False,
            ep_dispatch_algorithm=None,
            enable_eplb=False,
        )
        config.install()
        self.addCleanup(config.restore)
        for fused_shared in (0, 1):
            for scale_in_topk in (False, True):
                with self.subTest(
                    fused_shared=fused_shared, scale_in_topk=scale_in_topk
                ):
                    moe = self._v41_moe(fused_shared, scale_in_topk)
                    ids = torch.tensor([7, 99, 99])
                    hidden = torch.ones((3, 4))
                    batch = SimpleNamespace(
                        moe_num_token_non_padded=lambda: torch.tensor(
                            2, dtype=torch.int32
                        )
                    )

                    def run_experts(_experts, x, topk_ids, weights, **kwargs):
                        # Routed experts multiply x by their physical ID + 1;
                        # the rank-1 shared slot (ID 5) is an identity expert.
                        factors = (topk_ids + 1).float()
                        factors[topk_ids == 5] = 1.0
                        self.assertTrue((topk_ids[2] == -1).all())
                        return x * (
                            (factors * weights).sum(-1, keepdim=True)
                            * kwargs["routed_scaling_factor"]
                        )

                    with (
                        get_flags().moe.override(a2a_backend=MoeA2ABackend.MEGAMOE),
                        get_parallel().override(
                            tp_size=2,
                            tp_rank=1,
                            attn_tp_size=2,
                            attn_tp_rank=1,
                            moe_ep_size=2,
                            moe_ep_rank=1,
                            moe_tp_size=1,
                            moe_tp_rank=0,
                        ),
                        forward_context(ForwardContext(None), forward_batch=batch),
                        patch.object(mega_moe, "run_mega_routed_experts", run_experts),
                        patch.object(mega_moe, "_mega_moe_buffer"),
                        # The fused padded-row fill requires Triton/GPU.
                        patch(
                            "sglang.srt.multimodal.dsv41.vl_routing.is_cuda",
                            return_value=False,
                        ),
                        patch.object(topk_module, "_is_cuda", False),
                        patch.object(
                            topk_module, "_can_fuse_padded_region", return_value=False
                        ),
                    ):
                        out = mega_moe._run_mega_routed(
                            moe, hidden, batch, ids, num_tokens=3
                        )
                        text_out = mega_moe._run_mega_routed(
                            moe, hidden, batch, None, num_tokens=3
                        )
                    # Text selects logical 0/1; image selects logical 2/3. With
                    # a shared slot, rank-1 routed IDs shift by one (2/3 -> 3/4).
                    expected = torch.tensor(
                        [3.0 + fused_shared, 7.0 + 3.0 * fused_shared, 0.0]
                    )[:, None].expand_as(hidden)
                    torch.testing.assert_close(out, expected)
                    text_expected = torch.tensor(
                        [3.0 + fused_shared, 3.0 + fused_shared, 0.0]
                    )[:, None].expand_as(hidden)
                    torch.testing.assert_close(text_out, text_expected)

    def test_v41_megamoe_empty_rank_still_participates(self):
        """An idle DP rank must reach A2A combine without running its router."""
        moe = self._v41_moe(0, False)
        hidden = torch.empty((0, 4))

        def run_experts(_experts, x, ids, weights, **kwargs):
            self.assertIsNone(ids)
            self.assertIsNone(weights)
            # The collective boundary owns the result even on an idle rank.
            return x.new_empty((0, kwargs["hidden_size"]))

        with (
            patch.object(mega_moe, "run_mega_routed_experts", run_experts),
            patch.object(mega_moe, "_mega_moe_buffer"),
        ):
            out = mega_moe._run_mega_routed(
                moe, hidden, None, torch.empty(0, dtype=torch.long), num_tokens=0
            )
        self.assertEqual(out.shape, (0, 4))
        moe.gate.assert_not_called()

    @staticmethod
    def _mega_gate_reference(x, weight, top_k, **kwargs):
        unmasked = x if kwargs["mask"] is None else x[kwargs["mask"]]
        if not torch.isfinite(unmasked).all():
            raise ValueError("MegaGate rejects nonfinite scores on unmasked rows")
        scores = torch.log1p(torch.exp(x.float() @ weight.float().T)).sqrt()
        if kwargs["unmapped_topk_idx"] is not None:
            ids = kwargs["unmapped_topk_idx"]
        else:
            bias = kwargs["bias"]
            if kwargs["image_bias"] is not None:
                bias = torch.where(
                    kwargs["image_token_mask"][:, None], kwargs["image_bias"], bias
                )
            ids = (scores + bias).topk(top_k, dim=-1).indices
        weights = scores.gather(1, ids)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        weights = weights * kwargs["routed_scaling_factor"]
        if kwargs["to_physical_map"] is not None:
            physical_map, count = kwargs["to_physical_map"], kwargs["logical_count"]
            assert physical_map.dtype == count.dtype == torch.int32
            token = torch.arange(ids.shape[0])[:, None]
            replica = (kwargs["ep_rank"] + token * 23333) % count[ids]
            ids = physical_map[ids, replica]
        if kwargs["mask"] is not None:
            weights = weights.masked_fill(~kwargs["mask"][:, None], 0)
            ids = ids.masked_fill(~kwargs["mask"][:, None], -1)
        kwargs["out"][0].copy_(ids)
        kwargs["out"][1].copy_(weights)

    def _mega_gate_moe(self, scale_in_topk=False, hash_routing=False):
        moe = self._v41_moe(0, scale_in_topk)
        moe.config.model_type = "deepseek_v41"
        moe.config.hidden_size = 256
        moe.gate.weight = torch.zeros((8, 256), dtype=torch.bfloat16)
        moe.gate.e_score_correction_bias = torch.tensor(
            [4.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        moe.gate.e_score_correction_bias_vl = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0]
        )
        moe.topk.topk_config.sqrtsoftplus_log1p = True
        moe.topk.enable_waterfill = False
        moe.mega_gate_supported = True
        if hash_routing:
            moe.is_hash = True
            moe.gate.e_score_correction_bias = None
            moe.topk = SimpleNamespace(
                enable_waterfill=False,
                topk=2,
                num_shared_topk_slots=0,
                routed_scaling_factor=2.0,
                apply_routed_scaling_factor_on_output=scale_in_topk,
                score_func="sqrtsoftplus",
                tid2eid=torch.tensor([[6, 7]] * 100, dtype=torch.int32),
            )
        return moe

    @contextmanager
    def _mega_gate_env(self):
        deep_gemm = ModuleType("deep_gemm")
        deep_gemm.bf16_mega_gate = self._mega_gate_reference
        with ExitStack() as stack:
            stack.enter_context(
                get_context().override_server_args(
                    model_path="dummy", enable_deterministic_inference=False
                )
            )
            stack.enter_context(get_parallel().override(moe_ep_rank=0))
            stack.enter_context(patch.object(mega_gate_runtime, "_is_sail", False))
            stack.enter_context(patch.dict(sys.modules, {"deep_gemm": deep_gemm}))
            stack.enter_context(patch.object(topk_module, "_is_cuda", True))
            stack.enter_context(
                patch.object(topk_module, "_can_fuse_padded_region", return_value=False)
            )
            yield

    def test_mega_gate_admission_falls_back_for_unsupported_routers(self):
        moe = self._mega_gate_moe()
        supported = mega_gate_runtime.mega_gate_supported
        with (
            self._mega_gate_env(),
            self._reject_sail_imports(),
            patch.object(mega_gate_runtime, "_device_sm", 100),
        ):
            self.assertTrue(supported(moe))
            self.assertTrue(supported(self._mega_gate_moe(hash_routing=True)))
            with patch.object(moe.topk.topk_config, "scoring_func", "sigmoid"):
                self.assertFalse(supported(moe))
            with patch.object(moe, "num_fused_shared_experts", 1):
                self.assertFalse(supported(moe))
            with get_exec().deterministic.override(enable_deterministic_inference=True):
                self.assertFalse(supported(moe))

            hidden = torch.zeros((32, 256), dtype=torch.bfloat16)
            use = mega_gate_runtime.should_use_mega_gate
            decode = SimpleNamespace(forward_mode=ForwardMode.DECODE)
            with envs.SGLANG_OPT_DEEPGEMM_MEGA_GATE.override(False):
                self.assertFalse(use(moe, hidden, decode))
            with envs.SGLANG_OPT_DEEPGEMM_MEGA_GATE.override(True):
                self.assertTrue(use(moe, hidden, decode))
                for mode in (ForwardMode.TARGET_VERIFY, ForwardMode.EXTEND):
                    self.assertTrue(
                        use(moe, hidden, SimpleNamespace(forward_mode=mode))
                    )
                self.assertFalse(
                    use(
                        moe,
                        hidden,
                        SimpleNamespace(forward_mode=ForwardMode.DRAFT_EXTEND_V2),
                    )
                )
                self.assertFalse(use(moe, hidden.float(), decode))
                with envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.override(
                    16
                ):
                    self.assertFalse(use(moe, hidden, decode))
                self.assertFalse(use(moe, hidden, None))

    def test_sail_mega_gate_requires_compatible_vendor_api_and_decode_rows(self):
        """Require a compatible API, supported SM, and bounded decode/verify rows."""
        moe = self._mega_gate_moe()
        with (
            self._mega_gate_env(),
            patch.object(mega_gate_runtime, "_is_sail", True),
            envs.SGLANG_OPT_DEEPGEMM_MEGA_GATE.override(True),
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.override(4096),
        ):
            self.addCleanup(sail_mega_gate._sail_mega_gate_available.cache_clear)
            for sm, vendor_fn, expected in (
                (88, self._mega_gate_reference, False),
                (90, self._mega_gate_reference, False),
                (89, lambda x, weight, top_k: None, False),
                (89, None, False),
                (89, self._mega_gate_reference, True),
            ):
                with (
                    self.subTest(sm=sm, vendor_fn=vendor_fn),
                    patch.object(sail_mega_gate, "_device_sm", sm),
                    patch("deep_gemm.bf16_mega_gate", vendor_fn),
                ):
                    sail_mega_gate._sail_mega_gate_available.cache_clear()
                    self.assertEqual(
                        mega_gate_runtime.mega_gate_supported(moe), expected
                    )
            hidden = torch.zeros((2048, 256), dtype=torch.bfloat16)
            for mode, expected in (
                (ForwardMode.DECODE, True),
                (ForwardMode.TARGET_VERIFY, True),
                (ForwardMode.EXTEND, False),
                (ForwardMode.IDLE, False),
            ):
                with self.subTest(mode=mode):
                    self.assertEqual(
                        mega_gate_runtime.should_use_mega_gate(
                            moe, hidden, SimpleNamespace(forward_mode=mode)
                        ),
                        expected,
                    )
            decode = SimpleNamespace(forward_mode=ForwardMode.DECODE)
            self.assertFalse(
                mega_gate_runtime.should_use_mega_gate(
                    moe, torch.zeros((2049, 256), dtype=torch.bfloat16), decode
                )
            )
            with envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.override(16):
                self.assertFalse(
                    mega_gate_runtime.should_use_mega_gate(moe, hidden, decode)
                )

    def test_mega_gate_routes_vision_hash_and_padding(self):
        for is_hash in (False, True):
            for scale_in_topk in (False, True):
                for in_buffer in (False, True):
                    with self.subTest(
                        is_hash=is_hash,
                        scale_in_topk=scale_in_topk,
                        in_buffer=in_buffer,
                    ):
                        moe = self._mega_gate_moe(scale_in_topk, is_hash)
                        hidden = torch.ones((4, 256), dtype=torch.bfloat16)
                        hidden[3:] = float("nan")
                        input_ids = torch.tensor([7, 99, 7, 1000])
                        buf = SimpleNamespace(
                            topk_idx=torch.full((8, 2), 5, dtype=torch.int64),
                            topk_weights=torch.full((8, 2), 9.0),
                        )
                        with (
                            self._mega_gate_env(),
                            patch.object(
                                moe.topk,
                                "routing_inputs",
                                return_value=(torch.tensor([3]), None),
                                create=True,
                            ),
                        ):
                            routing = mega_gate_runtime.run_mega_gate(
                                moe, hidden, input_ids, buf if in_buffer else None
                            )
                        self.assertEqual(routing.in_buffer, in_buffer)
                        if in_buffer:
                            ids, weights = buf.topk_idx[:4], buf.topk_weights[:4]
                            self.assertEqual(routing.routed_scaling_factor, 1.0)
                            self.assertTrue((buf.topk_idx[4:] == 5).all())
                            scale = 2.0
                        else:
                            ids, weights = routing.topk_ids, routing.topk_weights
                            self.assertEqual(
                                routing.routed_scaling_factor,
                                1.0 if scale_in_topk else 2.0,
                            )
                            scale = 2.0 if scale_in_topk else 1.0
                        image = [6, 7] if is_hash else [7, 6]
                        text = [6, 7] if is_hash else [0, 1]
                        torch.testing.assert_close(
                            ids,
                            torch.tensor([text, image, text, [-1, -1]]),
                        )
                        torch.testing.assert_close(
                            weights,
                            torch.tensor([[0.5 * scale] * 2] * 3 + [[0.0, 0.0]]),
                        )

    def test_mega_gate_row_masks_rebuilt_inside_graph_capture(self):
        # The decoder replay tail captures after eager warm-ups in the same forward.
        bound = torch.tensor([3], dtype=torch.int32)
        forward_batch = SimpleNamespace()

        def masks(capturing):
            with patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=capturing
            ):
                return mega_gate_runtime._gate_row_masks(
                    None, 4, bound, None, "cpu", forward_batch
                )[0]

        with (
            envs.SGLANG_OPT_DEEPGEMM_MEGA_GATE.override(True),
            mega_gate_runtime.mega_gate_forward_scope(forward_batch),
        ):
            eager = masks(False)
            self.assertIs(masks(False), eager)
            captured = masks(True)
            self.assertIsNot(captured, eager)
            self.assertIs(masks(True), captured)
        torch.testing.assert_close(captured, torch.tensor([True, True, True, False]))

    def test_mega_gate_maps_eplb_in_kernel_and_records_from_buffer(self):
        moe = self._mega_gate_moe()
        hidden = torch.ones((4, 256), dtype=torch.bfloat16)
        hidden[2:] = float("nan")
        dispatch_info = ExpertLocationDispatchInfo(
            ep_dispatch_algorithm="static",
            partial_logical_to_rank_dispatch_physical_map=torch.tensor(
                [9, 8, 7, 6, 5, 4, 3, 2]
            ),
            partial_logical_to_all_physical_map=None,
            partial_logical_to_all_physical_map_num_valid=None,
            num_physical_experts=10,
        )
        recorded = []
        recorder = SimpleNamespace(
            on_select_experts=lambda **kw: recorded.append(kw["topk_ids"].clone())
        )
        buf = SimpleNamespace(
            topk_idx=torch.zeros((8, 2), dtype=torch.int64),
            topk_weights=torch.zeros((8, 2)),
        )
        with (
            self._mega_gate_env(),
            patch.object(
                mega_gate_runtime,
                "get_global_expert_distribution_recorder",
                return_value=recorder,
            ),
            patch.object(
                moe.topk,
                "routing_inputs",
                return_value=(torch.tensor([2]), dispatch_info),
            ),
        ):
            routing = mega_gate_runtime.run_mega_gate(
                moe, hidden, torch.tensor([7, 99, 1000, 1000]), buf
            )
        self.assertTrue(routing.in_buffer)
        torch.testing.assert_close(
            buf.topk_idx[:4],
            torch.tensor([[9, 8], [2, 3], [-1, -1], [-1, -1]]),
        )
        self.assertTrue((buf.topk_weights[2:4] == 0).all())
        self.assertEqual(len(recorded), 1)
        torch.testing.assert_close(recorded[0], buf.topk_idx[:4])
        for algorithm in ("lp", "fake"):
            with patch.object(
                moe.topk,
                "routing_inputs",
                return_value=(None, SimpleNamespace(ep_dispatch_algorithm=algorithm)),
            ):
                self.assertIsNone(
                    mega_gate_runtime.run_mega_gate(moe, hidden, None, buf)
                )

    @staticmethod
    def _v41_moe(fused_shared, scale_in_topk):
        with (
            get_context().override_server_args(),
            get_model().override(
                shared_experts=SharedExpertsConfig(
                    num_fused_shared_experts=fused_shared
                )
            ),
        ):
            return SimpleNamespace(
                config=SimpleNamespace(
                    hidden_size=4,
                    num_experts_per_tok=2,
                    moe_intermediate_size=8,
                    image_token_id=99,
                ),
                experts=SimpleNamespace(
                    should_fuse_routed_scaling_factor_in_topk=scale_in_topk,
                    moe_runner_config=SimpleNamespace(swiglu_limit=None),
                ),
                gate=MagicMock(
                    return_value=torch.zeros((3, 4)),
                    e_score_correction_bias=torch.tensor([4.0, 3.0, 0.0, 0.0]),
                    e_score_correction_bias_vl=torch.tensor([0.0, 0.0, 3.0, 4.0]),
                ),
                topk=TopK(
                    top_k=2,
                    layer_id=0,
                    scoring_func="sqrtsoftplus",
                    renormalize=True,
                    routed_scaling_factor=2.0,
                    apply_routed_scaling_factor_on_output=scale_in_topk,
                ),
                is_hash=False,
                num_fused_shared_experts=fused_shared,
                layer_id=0,
                routed_scaling_factor=2.0,
                routed_scaling_factor_fused_in_topk=scale_in_topk,
                mega_shared_l1_weights=None,
                mega_shared_l2_weights=None,
            )

    def test_shape_check_uses_pinned_fp8_fp4_alignment(self):
        # Packed SF groups need 128 elements, not a 16-byte token SF row.
        mega_moe.check_mega_moe_shapes(2048, 768, "fp8xfp4")
        mega_moe.check_mega_moe_shapes(5120, 2304, "fp8xfp4")
        mega_moe.check_mega_moe_shapes(512, 128, "fp8xfp4")
        with self.assertRaisesRegex(ValueError, "intermediate_size.*multiple of 128"):
            mega_moe.check_mega_moe_shapes(2048, 736, "fp8xfp4")
        with self.assertRaisesRegex(ValueError, "hidden_size.*multiple of 512"):
            mega_moe.check_mega_moe_shapes(2016, 768, "fp8xfp4")
        with self.assertRaisesRegex(ValueError, "hidden_size.*multiple of 512"):
            mega_moe.check_mega_moe_shapes(256, 128, "fp8xfp4")
        mega_moe.check_mega_moe_shapes(4096, 1536, "fp8xfp4")
        mega_moe.check_mega_moe_shapes(4096, 1024, "mxf4xmxf4")
        # 768 is a multiple of 256, so the NVFP4 (g16) rule accepts it.
        mega_moe.check_mega_moe_shapes(2048, 768, "nvfp4xnvfp4")
        with self.assertRaisesRegex(ValueError, "intermediate_size.*multiple of 256"):
            mega_moe.check_mega_moe_shapes(2048, 384, "nvfp4xnvfp4")

    def test_shape_check_mxf4_takes_intermediate_multiples_of_128(self):
        # The MXFP4 intermediate SFs are MN-major like fp8xfp4's, so DeepSeek-V4.1-Flash's
        # 2304 (not a multiple of 512) is accepted; the input SF rows still need hidden % 512.
        mega_moe.check_mega_moe_shapes(5120, 2304, "mxf4xmxf4")
        mega_moe.check_mega_moe_shapes(512, 128, "mxf4xmxf4")
        with self.assertRaisesRegex(ValueError, "intermediate_size.*multiple of 128"):
            mega_moe.check_mega_moe_shapes(5120, 2240, "mxf4xmxf4")
        with self.assertRaisesRegex(ValueError, "hidden_size.*multiple of 512"):
            mega_moe.check_mega_moe_shapes(4864, 2304, "mxf4xmxf4")

    def test_glm_mxfp4_shapes_pass_alignment_check(self):
        # GLM mxfp4 W4A4 checkpoints (verified against the on-disk configs):
        # GLM-5.3 (glm_moe_dsa): hidden 6144, moe_intermediate 2048, 256 experts.
        # GLM-5.3-Flash (glm5_next): hidden 4096, moe_intermediate 2048, 288 experts.
        for hidden, intermediate in ((6144, 2048), (4096, 2048)):
            mega_moe.check_mega_moe_shapes(hidden, intermediate, "mxf4xmxf4")

    def test_glm_megamoe_inheritance_chain(self):
        # GLM-5.3 (glm_moe_dsa) reuses the DeepseekV2 stack and GLM-5.3-Flash
        # aliases DeepseekV2MoE directly, so both inherit the megamoe forward
        # hook without model-side code. Guard against future refactors that
        # break this assumption.
        from sglang.srt.models import glm5_next as glm5_next_module
        from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM, DeepseekV2MoE
        from sglang.srt.models.glm4_moe import GlmMoeDsaForCausalLM

        self.assertTrue(issubclass(GlmMoeDsaForCausalLM, DeepseekV2ForCausalLM))
        self.assertIs(
            getattr(glm5_next_module, "Glm5NextMoE", None)
            or getattr(glm5_next_module, "DeepseekV2MoE", None),
            DeepseekV2MoE,
        )
        self.assertTrue(
            hasattr(DeepseekV2MoE, "forward"),
            "mega hook lives in DeepseekV2MoE.forward",
        )

    def _fake_hit_moe(self, tag):
        experts = SimpleNamespace(
            num_experts=256,
            _mega_moe_nvfp4=False,
            _mega_moe_w4a4=True,
            _mega_moe_mma_type="mxf4xmxf4",
        )
        return SimpleNamespace(prefix=tag, experts=experts, n_shared_experts=1)

    def _reset_hit_marker(self):
        state = sail_mega_moe._mega_moe_hit_state()
        old = state["logged"]
        self.addCleanup(state.update, logged=old)
        state["logged"] = False

    def test_mega_moe_hit_marker_logs_once_for_all_layers(self):
        # The marker is one line per process (the first admitted batch is the
        # startup warmup), not one line per MoE layer.
        self._reset_hit_marker()
        with self.assertLogs("mega_moe", level="WARNING") as captured:
            for tag in ("layers.0.mlp", "layers.1.mlp", "layers.2.mlp"):
                sail_mega_moe._log_mega_moe_hit(self._fake_hit_moe(tag))
        self.assertEqual(len(captured.output), 1, captured.output)
        self.assertIn("MEGA_MOE_HIT", captured.output[0])
        self.assertIn("layers.0.mlp", captured.output[0])
        self.assertIn("experts=256", captured.output[0])
        self.assertIn("mxf4xmxf4", captured.output[0])
        self.assertTrue(sail_mega_moe._mega_moe_hit_state()["logged"])

    def test_mega_moe_hit_marker_stays_silent_after_first_hit(self):
        self._reset_hit_marker()
        sail_mega_moe._mega_moe_hit_state()["logged"] = False
        with patch.object(sail_mega_moe._MEGA_HIT_LOGGER, "warning") as warning:
            sail_mega_moe._log_mega_moe_hit(self._fake_hit_moe("layers.0.mlp"))
            self.assertEqual(warning.call_count, 1)
            for tag in ("layers.1.mlp", "layers.2.mlp", "layers.3.mlp"):
                sail_mega_moe._log_mega_moe_hit(self._fake_hit_moe(tag))
            self.assertEqual(warning.call_count, 1)

    def test_ppu_precision_uses_typed_pre_dispatch_and_matching_compute(self):
        for mma_type, vendor_mma_type, kernel_name in (
            ("mxf4xmxf4", "fp4xfp4", "fp4_fp4_mega_moe"),
            ("fp8xmxfp4", "fp8xfp4", "fp8_fp4_mega_moe"),
            ("bf16xmxfp4", "bf16xfp4", "bf16_fp4_mega_moe"),
        ):
            with self.subTest(mma_type=mma_type):
                deep_gemm = ModuleType("deep_gemm")
                deep_gemm.mega_moe_pre_dispatch = MagicMock()
                # This build exposes only the selected precision's compute API.
                selected_kernel = MagicMock()
                setattr(deep_gemm, kernel_name, selected_kernel)
                buffer = SimpleNamespace(
                    x=object(), x_sf=object(), topk_idx=object(), topk_weights=object()
                )
                experts = SimpleNamespace(
                    num_experts=8,
                    mega_l1_weights=object(),
                    mega_l2_weights=object(),
                    _mega_moe_nvfp4=False,
                )
                with (
                    patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
                    patch.object(mega_moe, "_is_sail", True),
                    patch.object(sail_mega_moe, "_device_sm", 89),
                    patch.object(
                        sail_mega_moe, "_mega_moe_mma_type", return_value=mma_type
                    ),
                    patch.object(
                        sail_mega_moe, "_get_mega_moe_symm_buffer", return_value=buffer
                    ),
                    patch.object(
                        sail_mega_moe,
                        "_configure_mega_moe_deep_gemm_num_sms",
                        return_value=nullcontext(),
                    ),
                    patch.object(mega_moe, "check_mega_moe_shapes"),
                    patch(
                        "sglang.srt.runtime_context.get_parallel",
                        return_value=SimpleNamespace(
                            moe_ep_group=SimpleNamespace(device_group=object())
                        ),
                    ),
                ):
                    out = mega_moe.run_mega_routed_experts(
                        experts,
                        torch.zeros((1, 8), dtype=torch.bfloat16)[:, ::2],
                        torch.tensor([[0, 2, 1, 3]])[:, ::2],
                        torch.tensor([[0.5, 1.0, 0.5, 1.0]])[:, ::2],
                        hidden_size=4,
                        intermediate_size=8,
                        top_k=2,
                        num_tokens=1,
                    )

                self.assertEqual(out.shape, (1, 4))
                for actual, expected in zip(
                    deep_gemm.mega_moe_pre_dispatch.call_args.args[3:],
                    (buffer.x, buffer.x_sf, buffer.topk_idx, buffer.topk_weights),
                    strict=True,
                ):
                    self.assertIs(actual, expected)
                staged_input, staged_ids, staged_weights = (
                    deep_gemm.mega_moe_pre_dispatch.call_args.args[:3]
                )
                self.assertEqual(staged_input.dtype, torch.bfloat16)
                self.assertEqual(staged_ids.dtype, torch.int64)
                self.assertEqual(staged_weights.dtype, torch.float32)
                for tensor in (staged_input, staged_ids, staged_weights):
                    self.assertTrue(tensor.is_contiguous())
                torch.testing.assert_close(staged_ids, torch.tensor([[0, 1]]))
                self.assertEqual(
                    deep_gemm.mega_moe_pre_dispatch.call_args.kwargs,
                    {"num_tokens": 1, "group_size": 32, "mma_type": vendor_mma_type},
                )
                self.assertNotIn(
                    "shared_l1_weights",
                    getattr(deep_gemm, kernel_name).call_args.kwargs,
                )
                for name in (
                    "fp4_fp4_mega_moe",
                    "fp8_fp4_mega_moe",
                    "bf16_fp4_mega_moe",
                ):
                    if name == kernel_name:
                        selected_kernel.assert_called_once()
                    else:
                        self.assertFalse(hasattr(deep_gemm, name))

    def test_sail_collective_keeps_empty_ranks_on_vendor_path(self):
        """Even an empty SM90-emulated PPU rank must join the vendor collective."""
        for precision, kernel_name in (
            ("mxf4xmxf4", "fp4_fp4_mega_moe"),
            ("fp8xmxfp4", "fp8_fp4_mega_moe"),
            ("bf16xmxfp4", "bf16_fp4_mega_moe"),
        ):
            with self.subTest(precision=precision):
                events = []

                def dispatch(x, ids, weights, *args, **kwargs):
                    self.assertEqual(ids.dtype, torch.int64)
                    self.assertEqual(weights.dtype, torch.float32)
                    self.assertEqual(ids.shape, (x.shape[0], 2))
                    events.append(("dispatch", x.shape[0]))

                def compute(y, l1, l2, buf, **kwargs):
                    events.append(("compute", y.shape[0]))
                    y.fill_(7)

                deep_gemm = self.deep_gemm
                deep_gemm.mega_moe_pre_dispatch = dispatch
                setattr(deep_gemm, kernel_name, compute)
                experts = SimpleNamespace(
                    _mega_moe_nvfp4=False,
                    _mega_moe_mma_type=precision,
                    mega_l1_weights=object(),
                    mega_l2_weights=object(),
                )
                buf = SimpleNamespace(
                    x=object(), x_sf=object(), topk_idx=object(), topk_weights=object()
                )
                with (
                    patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
                    patch.object(mega_moe, "_is_sail", True),
                    patch.object(mega_moe, "_device_sm", 90),
                    patch.object(
                        mega_moe,
                        "mega_moe_pre_dispatch",
                        side_effect=AssertionError("SAIL must not run the CUDA JIT"),
                    ),
                ):
                    for num_tokens in (0, 2):
                        result = mega_moe.run_mega_routed_experts(
                            experts,
                            torch.zeros((num_tokens, 4), dtype=torch.bfloat16),
                            torch.zeros((num_tokens, 2), dtype=torch.int32),
                            torch.ones((num_tokens, 2)),
                            hidden_size=4,
                            intermediate_size=8,
                            top_k=2,
                            num_tokens=num_tokens,
                            buf=buf,
                        )
                        self.assertEqual(result.shape, (num_tokens, 4))
                        torch.testing.assert_close(result, torch.full_like(result, 7))
                self.assertEqual(
                    events,
                    [("dispatch", 0), ("compute", 0), ("dispatch", 2), ("compute", 2)],
                )

    def test_ppu_experts_ready_from_sm89(self):
        """Admit transformed weights of all three supported precisions only."""
        with patch.object(mega_moe, "_is_sail", True):
            for sm in (88, 89, 90):
                for precision in ("mxf4xmxf4", "fp8xmxfp4", "bf16xmxfp4", "fp8xfp8"):
                    with (
                        self.subTest(sm=sm, precision=precision),
                        patch.object(sail_mega_moe, "_device_sm", sm),
                    ):
                        experts = SimpleNamespace(
                            _mega_moe_weights_built=True,
                            _mega_moe_mma_type=precision,
                            _mega_moe_nvfp4=False,
                        )
                        self.assertEqual(
                            mega_moe.is_mega_moe_experts_ready(experts),
                            sm >= 89 and precision != "fp8xfp8",
                        )
                        experts._mega_moe_weights_built = False
                        self.assertFalse(mega_moe.is_mega_moe_experts_ready(experts))

    def test_kimi_k3_situ_selects_ppu_mma_api_without_changing_cuda_path(self):
        from sglang.srt.models.kimi_k3 import KimiK3MoE

        buffer = SimpleNamespace(
            x=object(),
            x_sf=object(),
            topk_idx=object(),
            topk_weights=object(),
        )
        experts = SimpleNamespace(
            num_experts=8,
            mega_l1_weights=object(),
            mega_l2_weights=object(),
            moe_runner_config=SimpleNamespace(
                gemm1_alpha=3.0,
                gemm1_clamp_limit=17.0,
            ),
            should_fuse_routed_scaling_factor_in_topk=True,
        )
        owner = SimpleNamespace(
            experts=experts,
            _mega_top_k=2,
            moe_hidden_size=4,
            _mega_intermediate_size=8,
        )
        owner._mega_moe_buffer = lambda: KimiK3MoE._mega_moe_buffer(owner)
        topk_output = SimpleNamespace(
            topk_ids=torch.tensor([[0, 1]], dtype=torch.int32),
            topk_weights=torch.tensor([[0.6, 0.4]]),
        )

        for ppu, mma_type, num_tokens in (
            (True, "mxf4xmxf4", 1),
            (True, "fp8xmxfp4", 1),
            (True, "bf16xmxfp4", 1),
            (True, "mxf4xmxf4", 0),
            (True, "fp8xmxfp4", 0),
            (True, "bf16xmxfp4", 0),
            (False, "mxf4xmxf4", 1),
            (False, "fp8xfp4", 1),
            (False, "fp8xfp4", 0),
        ):
            with self.subTest(ppu=ppu, mma_type=mma_type, num_tokens=num_tokens):
                runtime = sail_mega_moe if ppu else mega_moe
                parallel_owner = (
                    sail_mega_moe if ppu else sys.modules["sglang.srt.models.kimi_k3"]
                )
                deep_gemm = ModuleType("deep_gemm")
                deep_gemm.mega_moe_pre_dispatch = MagicMock()
                deep_gemm.fp4_fp4_mega_moe = MagicMock()
                deep_gemm.fp8_fp4_mega_moe = MagicMock()
                deep_gemm.bf16_fp4_mega_moe = MagicMock()
                with (
                    patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
                    patch("sglang.srt.models.kimi_k3._is_sail", ppu),
                    nullcontext() if ppu else self._reject_sail_imports(),
                    patch.object(
                        runtime, "_mega_moe_mma_type", return_value=mma_type
                    ) as select_type,
                    patch.object(
                        runtime, "_get_mega_moe_symm_buffer", return_value=buffer
                    ) as get_buffer,
                    patch.object(
                        runtime,
                        "_configure_mega_moe_deep_gemm_num_sms",
                        return_value=nullcontext(),
                    ),
                    patch.object(
                        parallel_owner,
                        "get_parallel",
                        return_value=SimpleNamespace(
                            moe_ep_group=SimpleNamespace(device_group=object())
                        ),
                    ),
                    patch(
                        "sglang.kernels.ops.moe.dsv4.mega_moe_pre_dispatch"
                    ) as triton_dispatch,
                ):
                    result = KimiK3MoE._forward_mega_experts(
                        owner,
                        torch.zeros((num_tokens, 4), dtype=torch.bfloat16),
                        topk_output,
                    )

                self.assertEqual(result.shape, (num_tokens, 4))
                self.assertEqual(select_type.call_args_list, [call(experts)] * 2)
                self.assertEqual(get_buffer.call_args.kwargs["mma_type"], mma_type)
                if ppu:
                    self.assertEqual(get_buffer.call_args.kwargs["activation"], "situ")
                else:
                    self.assertNotIn("activation", get_buffer.call_args.kwargs)
                if ppu:
                    dispatch_args = deep_gemm.mega_moe_pre_dispatch.call_args.args
                    self.assertEqual(dispatch_args[1].dtype, torch.int64)
                    for tensor in dispatch_args[:3]:
                        self.assertTrue(tensor.is_contiguous())
                    for actual, expected in zip(
                        dispatch_args[3:],
                        (buffer.x, buffer.x_sf, buffer.topk_idx, buffer.topk_weights),
                        strict=True,
                    ):
                        self.assertIs(actual, expected)
                    self.assertEqual(
                        deep_gemm.mega_moe_pre_dispatch.call_args.kwargs,
                        {
                            "num_tokens": num_tokens,
                            "group_size": 32,
                            "mma_type": {
                                "mxf4xmxf4": "fp4xfp4",
                                "fp8xmxfp4": "fp8xfp4",
                                "bf16xmxfp4": "bf16xfp4",
                            }[mma_type],
                        },
                    )
                    triton_dispatch.assert_not_called()
                    kernel = {
                        "mxf4xmxf4": deep_gemm.fp4_fp4_mega_moe,
                        "fp8xmxfp4": deep_gemm.fp8_fp4_mega_moe,
                        "bf16xmxfp4": deep_gemm.bf16_fp4_mega_moe,
                    }[mma_type]
                    kernel.assert_called_once()
                    self.assertEqual(kernel.call_args.args[0].shape, (num_tokens, 4))
                    if mma_type != "mxf4xmxf4":
                        self.assertNotIn("recipe", kernel.call_args.kwargs)
                    self.assertEqual(kernel.call_args.kwargs["situ_beta"], 3.0)
                    self.assertEqual(
                        kernel.call_args.kwargs["situ_linear_beta"],
                        17.0,
                    )
                else:
                    if mma_type == "mxf4xmxf4":
                        deep_gemm.mega_moe_pre_dispatch.assert_called_once()
                        self.assertEqual(
                            deep_gemm.mega_moe_pre_dispatch.call_args.args[1].dtype,
                            torch.int32,
                        )
                        triton_dispatch.assert_not_called()
                    else:
                        triton_dispatch.assert_called_once()
                        self.assertEqual(
                            triton_dispatch.call_args.args[1].dtype, torch.int32
                        )
                        self.assertEqual(
                            triton_dispatch.call_args.kwargs["quant_group_size"], 32
                        )
                    deep_gemm.fp8_fp4_mega_moe.assert_called_once()
                    deep_gemm.fp4_fp4_mega_moe.assert_not_called()

    def test_kimi_ppu_precision_masks_padded_rows_and_scales_routes_once(self):
        from sglang.srt.models.kimi_k3 import KimiK3MoE

        routed_input = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], [float("nan")] * 4],
            dtype=torch.bfloat16,
        )
        valid_token_count = torch.tensor(1)
        buffer = SimpleNamespace(
            x=object(), x_sf=object(), topk_idx=object(), topk_weights=object()
        )
        staged = {}

        def pre_dispatch(
            x,
            ids,
            weights,
            buf_x,
            buf_x_sf,
            buf_topk_idx,
            buf_topk_weights,
            num_tokens,
            group_size=32,
            mma_type="mxf4xmxf4",
        ):
            self.assertIs(buf_x, buffer.x)
            self.assertIs(buf_x_sf, buffer.x_sf)
            self.assertIs(buf_topk_idx, buffer.topk_idx)
            self.assertIs(buf_topk_weights, buffer.topk_weights)
            self.assertEqual(num_tokens, 3)
            self.assertEqual(group_size, 32)
            self.assertEqual(mma_type, expected_mma_type)
            self.assertEqual(ids.dtype, torch.int64)
            self.assertTrue(torch.isfinite(x).all())
            valid = int(valid_token_count)
            self.assertTrue(torch.all(ids[valid:] == -1))
            self.assertTrue(torch.all(weights[valid:] == 0))
            staged.update(x=x, weights=weights)

        def kernel(y, l1, l2, buf, **kwargs):
            y.copy_(staged["x"] * staged["weights"].sum(-1, keepdim=True))
            y[int(valid_token_count) :].fill_(float("nan"))

        for mma_type in ("mxf4xmxf4", "fp8xmxfp4", "bf16xmxfp4"):
            expected_mma_type = {
                "mxf4xmxf4": "fp4xfp4",
                "fp8xmxfp4": "fp8xfp4",
                "bf16xmxfp4": "bf16xfp4",
            }[mma_type]
            for scale_in_topk in (False, True):
                with self.subTest(mma_type=mma_type, scale_in_topk=scale_in_topk):
                    experts = SimpleNamespace(
                        mega_l1_weights=object(),
                        mega_l2_weights=object(),
                        moe_runner_config=SimpleNamespace(
                            gemm1_alpha=4.0, gemm1_clamp_limit=25.0
                        ),
                        should_fuse_routed_scaling_factor_in_topk=scale_in_topk,
                    )
                    owner = SimpleNamespace(
                        experts=experts,
                        _mega_top_k=2,
                        moe_hidden_size=4,
                        _mega_moe_buffer=lambda: buffer,
                        routed_scaling_factor=2.0,
                    )
                    topk = SimpleNamespace(
                        topk_ids=torch.tensor([[0, 1]] * 3, dtype=torch.int32),
                        topk_weights=torch.tensor([[0.25, 0.75]] * 3)
                        * (2.0 if scale_in_topk else 1.0),
                    )
                    deep_gemm = ModuleType("deep_gemm")
                    deep_gemm.mega_moe_pre_dispatch = pre_dispatch
                    deep_gemm.fp4_fp4_mega_moe = kernel
                    deep_gemm.fp8_fp4_mega_moe = kernel
                    deep_gemm.bf16_fp4_mega_moe = kernel
                    with (
                        patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
                        patch("sglang.srt.models.kimi_k3._is_sail", True),
                        patch.object(
                            sail_mega_moe, "_mega_moe_mma_type", return_value=mma_type
                        ),
                        patch.object(
                            sail_mega_moe,
                            "_configure_mega_moe_deep_gemm_num_sms",
                            return_value=nullcontext(),
                        ),
                    ):
                        for valid in (1, 0, 2):
                            valid_token_count.fill_(valid)
                            actual = KimiK3MoE._forward_mega_experts(
                                owner,
                                routed_input,
                                topk,
                                valid_token_count=valid_token_count,
                            )
                            expected = torch.zeros_like(routed_input)
                            expected[:valid] = routed_input[:valid] * 2
                            torch.testing.assert_close(actual, expected)

    def test_mxf4_l1_uses_packed_gate_up_interleave(self):
        source = torch.arange(32).reshape(1, 32)
        expected = torch.tensor(
            [
                0,
                2,
                4,
                6,
                8,
                10,
                12,
                14,
                16,
                18,
                20,
                22,
                24,
                26,
                28,
                30,
                1,
                3,
                5,
                7,
                9,
                11,
                13,
                15,
                17,
                19,
                21,
                23,
                25,
                27,
                29,
                31,
            ]
        ).reshape(1, 32)

        actual = mega_moe._interleave_mega_moe_gate_up(source, gran=16)

        torch.testing.assert_close(actual, expected)

    @unittest.skipIf(
        is_sail(),
        "the PPU MegaMoE path takes per-layer MMA types and its own DeepGEMM APIs",
    )
    def test_draft_layers_keep_their_own_w4a4_choice(self):
        """Draft MegaMoE layers must not follow the target's MMA type once built.

        The draft forward runs outside draft_model_build_scope, so a layer that
        re-read the global flag would pair draft weights with the wrong kernel.
        """
        for target_flag, draft_flag, expected_target, expected_draft in (
            (True, False, "mxf4xmxf4", "fp8xfp4"),
            (True, None, "mxf4xmxf4", "mxf4xmxf4"),
            (False, True, "fp8xfp4", "mxf4xmxf4"),
            (False, None, "fp8xfp4", "fp8xfp4"),
        ):
            with self.subTest(
                enable_w4a4_mxfp4_megamoe=target_flag,
                speculative_enable_w4a4_mxfp4_megamoe=draft_flag,
            ):
                with get_context().override_server_args(
                    model_path="dummy",
                    enable_w4a4_mxfp4_megamoe=target_flag,
                    speculative_enable_w4a4_mxfp4_megamoe=draft_flag,
                ):
                    target = self._build_fused_moe()
                    with draft_model_build_scope():
                        draft = self._build_fused_moe()

                    self.assertEqual(
                        get_exec().moe.enable_w4a4_mxfp4_megamoe, target_flag
                    )
                    self.assertEqual(
                        mega_moe._mega_moe_mma_type(target), expected_target
                    )
                    self.assertEqual(mega_moe._mega_moe_mma_type(draft), expected_draft)

    @unittest.skipIf(
        is_sail(),
        "the PPU MegaMoE path takes per-layer MMA types and its own DeepGEMM APIs",
    )
    def test_draft_build_scope_restores_w4a4_on_exception(self):
        """A failed draft build must not leave the target on the draft's MMA type."""
        with get_context().override_server_args(
            model_path="dummy",
            enable_w4a4_mxfp4_megamoe=False,
            speculative_enable_w4a4_mxfp4_megamoe=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "draft build failed"):
                with draft_model_build_scope():
                    self.assertTrue(get_exec().moe.enable_w4a4_mxfp4_megamoe)
                    raise RuntimeError("draft build failed")

            self.assertFalse(get_exec().moe.enable_w4a4_mxfp4_megamoe)
            target = self._build_fused_moe()
            self.assertEqual(mega_moe._mega_moe_mma_type(target), "fp8xfp4")

    def _build_fused_moe(self):
        method = UnquantizedFusedMoEMethod()
        with (
            patch.object(method, "create_weights"),
            patch.object(method, "create_moe_runner"),
            patch.object(
                fused_moe_layer_module,
                "create_moe_dispatcher",
                return_value=SimpleNamespace(
                    quant_config={}, set_quant_config=MagicMock()
                ),
            ),
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend.AUTO,
                a2a_backend=MoeA2ABackend.MEGAMOE,
            ),
            get_parallel().override(
                moe_ep_size=1,
                moe_ep_rank=0,
                moe_tp_size=1,
                moe_tp_rank=0,
                tp_size=1,
                tp_rank=0,
            ),
        ):
            experts = fused_moe_layer_module.FusedMoE(
                num_experts=2,
                hidden_size=4,
                intermediate_size=8,
                layer_id=0,
                quant_method=method,
            )

        experts.w13_weight = torch.nn.Parameter(
            torch.zeros(0, dtype=torch.uint8), requires_grad=False
        )
        return experts

    def _get_test_buffer(self, group):
        with patch.dict(sys.modules, {"deep_gemm": self.deep_gemm}):
            return mega_moe._get_mega_moe_symm_buffer(
                group,
                num_experts=8,
                num_max_tokens_per_rank=64,
                num_topk=2,
                hidden=128,
                intermediate_hidden=256,
                mma_type="fp8xfp4",
            )


if __name__ == "__main__":
    unittest.main()
