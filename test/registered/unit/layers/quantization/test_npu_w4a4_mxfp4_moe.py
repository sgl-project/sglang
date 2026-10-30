"""CPU regression tests for W4A4 MXFP4 MoE support on Ascend NPU."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import torch

# Initialize quantization before its NPU helpers to avoid a cold-import cycle.
import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.hardware_backend.npu.moe.matmul import GroupedMatmulSwigluQuant
from sglang.srt.hardware_backend.npu.quantization import (
    moe_methods,
    online_moe_methods,
)
from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUMXFP8MoEMethod,
    NPUW4A4MXFP4MoEMethod,
    NPUW4A8MXFP4MoEMethod,
)
from sglang.srt.hardware_backend.npu.quantization.online_moe_methods import (
    NPUW4A4MXFP4OnlineMoEMethod,
)
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.moe_runner import ascend as ascend_runner
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimW4A4MXFP4MoE
from sglang.srt.layers.quantization.npu_mxfp4_w4a4 import Mxfp4W4A4Config
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestNPUW4A4MXFP4MoE(CustomTestCase):
    @staticmethod
    def _runner_config(**kwargs):
        defaults = dict(
            activation="silu",
            swiglu_limit=None,
            gemm1_alpha=None,
            gemm1_clamp_limit=None,
            layer=SimpleNamespace(),
        )
        return SimpleNamespace(**(defaults | kwargs))

    def test_config_selects_online_experts_and_respects_ignored_layers(self):
        layer = FusedMoE.__new__(FusedMoE)
        torch.nn.Module.__init__(layer)
        layer.use_triton_kernels = False
        layer.use_flashinfer_trtllm_moe = False
        prefix = "model.layers.0.mlp.experts"
        with (
            patch(
                "sglang.srt.layers.quantization.npu_mxfp4_w4a4.is_npu",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.quantization.unquant.get_moe_runner_backend",
                return_value=MoeRunnerBackend.AUTO,
            ),
        ):
            method = Mxfp4W4A4Config().get_quant_method(layer, prefix)
            skipped = Mxfp4W4A4Config(ignored_layers=[prefix]).get_quant_method(
                layer, prefix
            )
        self.assertIsInstance(method, NPUW4A4MXFP4OnlineMoEMethod)
        self.assertIs(type(skipped), UnquantizedFusedMoEMethod)

    def test_runner_installs_prefix_kernels_and_meta_weights(self):
        for backend in (MoeRunnerBackend.AUTO, MoeRunnerBackend.ASCEND):
            with (
                self.subTest(backend=backend),
                patch(
                    "sglang.srt.layers.quantization.unquant.get_moe_runner_backend",
                    return_value=backend,
                ),
                patch.object(
                    online_moe_methods, "get_moe_runner_backend", return_value=backend
                ),
                patch.object(online_moe_methods, "MoeRunner") as runner,
            ):
                method = NPUW4A4MXFP4OnlineMoEMethod()
                layer = torch.nn.Module()
                config = SimpleNamespace(layer=None, is_gated=True)
                method.create_moe_runner(layer, config)
                self.assertIsInstance(layer.w13_kernel, NPUW4A4MXFP4MoEMethod)
                self.assertIsInstance(layer.w2_kernel, NPUW4A4MXFP4MoEMethod)
                self.assertIs(config.layer, layer)
                runner.assert_called_once_with(MoeRunnerBackend.ASCEND, config)
                self.assertIsNone(method._aiter_runner)
                layer.moe_runner_config = config
                method.create_weights(
                    layer, 2, 128, 64, torch.bfloat16, weight_loader=Mock()
                )
                self.assertTrue(layer.w13_weight.is_meta)
                self.assertTrue(layer.w2_weight.is_meta)
                self.assertTrue(layer.w13_kernel.use_fused_gmm1)
                self.assertFalse(layer.w2_kernel.use_fused_gmm1)
                self.assertIsNone(layer.w2_kernel.fused_matmul)
                self.assertEqual(layer.w13_weight.shape, (2, 128, 128))
                self.assertEqual(layer.w2_weight.shape, (2, 128, 64))
                self.assertEqual(layer.w13_weight.dtype, torch.bfloat16)
                self.assertFalse(hasattr(layer, "w13_weight_scale"))

    def test_runner_rejects_incompatible_backend_before_binding(self):
        backend = MoeRunnerBackend.FLASHINFER_CUTLASS
        with (
            patch(
                "sglang.srt.layers.quantization.unquant.get_moe_runner_backend",
                return_value=backend,
            ),
            patch.object(
                online_moe_methods, "get_moe_runner_backend", return_value=backend
            ),
            patch.object(online_moe_methods, "MoeRunner") as runner,
        ):
            method = NPUW4A4MXFP4OnlineMoEMethod()
            layer = torch.nn.Module()
            with self.assertRaisesRegex(ValueError, "requires --moe-runner-backend"):
                method.create_moe_runner(layer, SimpleNamespace(layer=None))
            runner.assert_not_called()
            self.assertFalse(hasattr(layer, "w13_kernel"))

    def test_online_quantization_passes_fp4_operator_dtype(self):
        weight = torch.zeros(2, 96, 128, dtype=torch.bfloat16)
        expected = (torch.empty(2, 96, 64), torch.empty(2, 96, 2, 2))
        with (
            patch.object(
                torch.Tensor, "is_npu", new_callable=PropertyMock, create=True
            ) as is_npu,
            patch.object(
                torch.ops.npu,
                "npu_dynamic_mx_quant",
                create=True,
                return_value=expected,
            ) as quantize,
        ):
            is_npu.return_value = True
            actual = NPUW4A4MXFP4MoEMethod._quantize_weight_online(weight, 296)
            self.assertIs(actual, expected)
            quantize.assert_called_once_with(
                weight, dst_type=296, round_mode="rint", scale_alg=2, dst_type_max=7.25
            )

    def test_online_and_offline_weights_share_layout_without_requantizing_packed(self):
        packed = torch.arange(2 * 96 * 64).to(torch.uint8).reshape(2, 96, 64)
        scales = torch.arange(2 * 96 * 4).to(torch.uint8).reshape(2, 96, 4)
        method = NPUW4A4MXFP4MoEMethod()
        for prefix in ("w13", "w2"):
            for dtype, flat_scale in (
                (torch.bfloat16, False),
                (torch.bfloat16, True),
                (torch.float16, False),
                (torch.float16, True),
                (torch.uint8, True),
            ):
                with self.subTest(prefix=prefix, dtype=dtype, flat_scale=flat_scale):
                    online = dtype != torch.uint8
                    layer = torch.nn.Module()
                    layer.dispatcher = Mock()
                    layer.register_parameter(
                        f"{prefix}_weight",
                        torch.nn.Parameter(
                            torch.zeros(2, 96, 128, dtype=dtype)
                            if online
                            else packed.clone(),
                            requires_grad=False,
                        ),
                    )
                    if not online:
                        layer.register_parameter(
                            f"{prefix}_weight_scale",
                            torch.nn.Parameter(scales.clone(), requires_grad=False),
                        )
                    with (
                        patch.object(
                            moe_methods, "npu_format_cast", side_effect=lambda x: x
                        ),
                        patch.object(
                            method,
                            "_quantize_weight_online",
                            return_value=(
                                packed.clone(),
                                scales.clone()
                                if flat_scale
                                else scales.reshape(2, 96, 2, 2),
                            ),
                        ) as quantize,
                    ):
                        method.process_weights_after_loading(layer, prefix)
                    self.assertEqual(quantize.call_count, int(online))
                    weight = getattr(layer, f"{prefix}_weight")
                    scale = getattr(layer, f"{prefix}_weight_scale")
                    self.assertTrue(torch.equal(weight, packed.transpose(1, 2)))
                    self.assertTrue(
                        torch.equal(scale, scales.reshape(2, 96, 2, 2).transpose(1, 2))
                    )
                    self.assertFalse(weight.is_contiguous())
                    self.assertFalse(scale.is_contiguous())
                    if prefix == "w13":
                        layer.dispatcher.set_quant_config.assert_called_once_with(
                            {"dispatcher_output_dtype": "bf16"}
                        )
                    else:
                        layer.dispatcher.set_quant_config.assert_not_called()

    def test_online_method_stays_on_the_unquant_fused_moe_base(self):
        """FusedMoE picks its w1/w3 shard order from this base class.

        Dropping it makes the weight loader swap gate and up on every expert, so
        gmm1 computes silu(up) * gate with no error -- only degenerate output.
        """
        self.assertTrue(
            issubclass(NPUW4A4MXFP4OnlineMoEMethod, UnquantizedFusedMoEMethod)
        )

    def test_kernel_constructs_off_device(self):
        """The kernel holds a HiddenStatesDynamicQuant, built here on CPU.

        Binding torch.ops.npu.* in its constructor makes every scheme below
        un-constructible off-device, which is what this whole file needs.
        """
        ModelSlimW4A4MXFP4MoE({}, "w13")

        with self.assertRaises(ValueError):
            ModelSlimW4A4MXFP4MoE({}, "w1")

    def test_kernel_exposes_the_fused_gmm1_entry(self):
        """AscendRunnerCore calls this by name once it takes the fused branch.

        Renaming it, or dropping the GroupedMatmulSwigluQuant behind it, leaves
        an AttributeError that nothing short of an A5 run reaches.
        """
        kernel = NPUW4A4MXFP4MoEMethod("w13", use_fused_gmm1=True)

        self.assertTrue(callable(kernel.apply_fused_gmm1_swiglu))
        self.assertIsInstance(kernel.fused_matmul, GroupedMatmulSwigluQuant)

    def test_fused_gmm1_respects_dispatch_and_clamp_contract(self):
        kernel = NPUW4A4MXFP4MoEMethod("w13", use_fused_gmm1=True)
        for transport in (
            MoeA2ABackend.NONE,
            MoeA2ABackend.DEEPEP,
            MoeA2ABackend.ASCEND_OPS,
        ):
            with (
                self.subTest(transport=transport),
                patch.object(
                    ascend_runner, "get_moe_a2a_backend", return_value=transport
                ),
            ):
                self.assertEqual(
                    ascend_runner._uses_fused_gmm1(kernel, self._runner_config()),
                    transport.is_none(),
                )
                with self.assertRaisesRegex(NotImplementedError, "swiglu_limit"):
                    ascend_runner._uses_fused_gmm1(
                        kernel, self._runner_config(swiglu_limit=7.0)
                    )

    def test_custom_activations_keep_the_unfused_runner_path(self):
        kernel = NPUW4A4MXFP4MoEMethod("w13", use_fused_gmm1=True)
        with patch.object(
            ascend_runner, "get_moe_a2a_backend", return_value=MoeA2ABackend.NONE
        ):
            for override in (
                dict(activation="situ"),
                dict(activation="gelu"),
                dict(activation="npu_swiglu_oai"),
                dict(gemm1_alpha=1.0),
                dict(gemm1_clamp_limit=7.0),
            ):
                with self.subTest(override=override):
                    self.assertFalse(
                        ascend_runner._uses_fused_gmm1(
                            kernel, self._runner_config(**override)
                        )
                    )

    def test_fused_gmm1_passes_explicit_fp4_and_scale_contracts(self):
        kernel = NPUW4A4MXFP4MoEMethod("w13", use_fused_gmm1=True)
        x = torch.zeros(3, 128, dtype=torch.bfloat16)
        qx = torch.zeros(3, 64, dtype=torch.uint8)
        scale = torch.zeros(3, 2, 2, dtype=torch.uint8)
        tokens = torch.tensor([1, 2], dtype=torch.int32)
        info = SimpleNamespace(
            w13_weight=torch.zeros(2, 64, 96, dtype=torch.uint8),
            w13_weight_scale=torch.zeros(2, 2, 96, 2),
        )
        expected = (qx, scale)
        with (
            patch.object(kernel, "hidden_states_quantizer") as quantize,
            patch.object(
                torch.ops.npu,
                "npu_grouped_matmul_swiglu_quant_v2",
                create=True,
                return_value=expected,
            ) as fused,
            patch.object(moe_methods, "_require_e8m0_dtype", return_value=293),
        ):
            quantize.quant_dtype = 296
            quantize.return_value = (qx, scale)
            result = kernel.apply_fused_gmm1_swiglu(info, x, tokens, None, 1)
            quantize.assert_called_once_with(x)
            self.assertIs(result, expected)
            kwargs = fused.call_args.kwargs
            self.assertIs(kwargs["x"], qx)
            self.assertIs(kwargs["weight"][0], info.w13_weight)
            self.assertEqual(kwargs["group_list"].dtype, torch.int64)
            self.assertTrue(torch.equal(kwargs["group_list"], torch.tensor([1, 3])))
            for key in ("quant_dtype", "x_dtype", "weight_dtype"):
                self.assertEqual(kwargs[key], 296)
            for key in ("weight_scale_dtype", "x_scale_dtype"):
                self.assertEqual(kwargs[key], 293)
            self.assertIs(kwargs["x_scale"], scale)

    def test_gmm2_preserves_paired_scales_and_pairs_flat_scales(self):
        kernel = NPUW4A4MXFP4MoEMethod()
        x = torch.zeros(3, 64, dtype=torch.uint8)
        paired = torch.arange(12).to(torch.uint8).reshape(3, 2, 2)
        info = SimpleNamespace(
            w2_weight=torch.zeros(2, 64, 96, dtype=torch.uint8),
            w2_weight_scale=torch.zeros(2, 2, 96, 2),
        )
        for scale in (paired, paired.flatten(1)):
            with (
                self.subTest(shape=scale.shape),
                patch.object(kernel, "hidden_states_quantizer") as quantize,
                patch.object(
                    torch.ops.npu,
                    "npu_grouped_matmul",
                    create=True,
                    return_value=[torch.zeros(3, 96, dtype=torch.bfloat16)],
                ) as matmul,
                patch.object(moe_methods, "_require_e8m0_dtype", return_value=293),
            ):
                kernel.apply(
                    info, x, torch.tensor([1, 2]), scale, torch.bfloat16, "w2", 1
                )
                quantize.assert_not_called()
                kwargs = matmul.call_args.kwargs
                actual = kwargs["per_token_scale"][0]
                self.assertTrue(torch.equal(actual, paired))
                self.assertEqual(kwargs["group_list_type"], 1)
                self.assertTrue(torch.equal(kwargs["group_list"], torch.tensor([1, 2])))
                if scale is paired:
                    self.assertIs(actual, paired)

    def test_config_resolves_w4a4_mxfp4_moe_scheme(self):
        prefix = "model.layers.0.mlp.experts"
        quant_config = ModelSlimConfig(
            {
                f"{prefix}.0.gate_proj.weight": "W4A4_MXFP4",
                f"{prefix}.0.up_proj.weight": "W4A4_MXFP4",
                f"{prefix}.0.down_proj.weight": "W4A4_MXFP4",
            }
        )

        w13_scheme, w2_scheme = quant_config.get_moe_scheme(torch.nn.Module(), prefix)

        self.assertIsInstance(w13_scheme, ModelSlimW4A4MXFP4MoE)
        self.assertIsInstance(w2_scheme, ModelSlimW4A4MXFP4MoE)
        self.assertEqual(w13_scheme.weight_prefix, "w13")
        self.assertEqual(w2_scheme.weight_prefix, "w2")
        for scheme in (w13_scheme, w2_scheme):
            self.assertFalse(scheme.kernel.use_fused_gmm1)
            self.assertIsNone(scheme.kernel.fused_matmul)

    def test_fp4_activation_does_not_inherit_weight_policy(self):
        kernel = NPUW4A4MXFP4MoEMethod("w2")
        x = torch.zeros(3, 128, dtype=torch.bfloat16)
        with patch.object(
            torch.ops.npu,
            "npu_dynamic_mx_quant",
            create=True,
            return_value=(torch.zeros(3, 64, dtype=torch.uint8), torch.zeros(3, 2, 2)),
        ) as quantize:
            kernel.hidden_states_quantizer(x)
        quantize.assert_called_once_with(
            x, dst_type=kernel.hidden_states_quantizer.quant_dtype
        )

    def test_fused_fp4_input_scale_skips_requantization(self):
        kernel = NPUW4A4MXFP4MoEMethod("w13", use_fused_gmm1=True)
        x = torch.zeros(3, 64, dtype=torch.uint8)
        scale = torch.zeros(3, 2, 2, dtype=torch.uint8)
        info = SimpleNamespace(w13_weight_scale=torch.zeros(2, 2, 96, 2))
        with (
            patch.object(kernel, "hidden_states_quantizer") as quantize,
            patch.object(kernel.fused_matmul, "forward") as matmul,
        ):
            kernel.apply_fused_gmm1_swiglu(info, x, torch.tensor([1, 2]), scale, 0)
            quantize.assert_not_called()
            self.assertIs(matmul.call_args.kwargs["x_scale"], scale)
            self.assertEqual(matmul.call_args.kwargs["group_list_type"], 0)

    def test_rejects_unsupported_source_dtype_before_quantization(self):
        kernel = NPUW4A4MXFP4MoEMethod()
        for dtype in (torch.float32, torch.int8):
            layer = torch.nn.Module()
            layer.w13_weight = torch.nn.Parameter(
                torch.zeros(1, 64, 128, dtype=dtype), requires_grad=False
            )
            with (
                self.subTest(dtype=dtype),
                patch.object(
                    torch.ops.npu, "npu_dynamic_mx_quant", create=True
                ) as quantize,
            ):
                with self.assertRaisesRegex(ValueError, "FP16/BF16"):
                    kernel.process_weights_after_loading(layer, "w13")
                quantize.assert_not_called()

    def test_odd_offline_scales_fail_before_format_cast(self):
        layer = torch.nn.Module()
        layer.w2_weight = torch.nn.Parameter(
            torch.zeros(2, 64, 48, dtype=torch.uint8), requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            torch.zeros(2, 64, 3, dtype=torch.uint8), requires_grad=False
        )
        with patch.object(moe_methods, "npu_format_cast") as cast:
            with self.assertRaisesRegex(ValueError, "divisible by 2"):
                NPUW4A4MXFP4MoEMethod().process_weights_after_loading(layer, "w2")
            cast.assert_not_called()

    def test_fusion_requires_opt_in_and_no_expert_bias(self):
        with patch.object(
            ascend_runner, "get_moe_a2a_backend", return_value=MoeA2ABackend.NONE
        ):
            for enabled in (False, True):
                kernel = NPUW4A4MXFP4MoEMethod("w13", use_fused_gmm1=enabled)
                for attr in (None, "w13_weight_bias", "w13_scale_bias"):
                    layer = SimpleNamespace()
                    if attr:
                        setattr(layer, attr, torch.ones(2, 128))
                    with self.subTest(enabled=enabled, bias=attr):
                        self.assertEqual(
                            ascend_runner._uses_fused_gmm1(
                                kernel, self._runner_config(layer=layer)
                            ),
                            enabled and attr is None,
                        )

    def test_runner_initialization_and_execution_agree_on_w13_fusion(self):
        x = torch.zeros(3, 64, dtype=torch.bfloat16)
        scale = torch.zeros(3, 1, 2, dtype=torch.uint8)
        for fused in (False, True):
            layer = SimpleNamespace(
                w13_kernel=NPUW4A4MXFP4MoEMethod("w13", use_fused_gmm1=fused),
                w2_kernel=NPUW4A4MXFP4MoEMethod("w2", use_fused_gmm1=True),
            )
            with (
                self.subTest(fused=fused),
                patch.object(
                    ascend_runner,
                    "get_moe_a2a_backend",
                    return_value=MoeA2ABackend.NONE,
                ),
            ):
                runner = ascend_runner.AscendRunnerCore(
                    self._runner_config(layer=layer)
                )
                self.assertEqual(runner.activation is None, fused)
                self.assertIsNone(layer.w2_kernel.fused_matmul)
                if not fused:
                    runner.activation._apply_activation = Mock(return_value=(x, None))
                with (
                    patch.object(layer.w13_kernel, "apply", return_value=x) as plain,
                    patch.object(
                        layer.w13_kernel,
                        "apply_fused_gmm1_swiglu",
                        return_value=(x, scale),
                    ) as fusion,
                    patch.object(layer.w2_kernel, "apply", return_value=x) as down,
                ):
                    result = runner.run(
                        SimpleNamespace(
                            hidden_states=x,
                            hidden_states_scale=None,
                            expert_tokens=torch.tensor([1, 2]),
                            group_list_type=1,
                        ),
                        SimpleNamespace(),
                        {},
                    )
                    self.assertIs(result.hidden_states, x)
                    self.assertEqual(fusion.call_count, int(fused))
                    self.assertEqual(plain.call_count, int(not fused))
                    self.assertIs(
                        down.call_args.kwargs["pertoken_scale"],
                        scale if fused else None,
                    )

    def test_mxfp8_clamp_keeps_main_unfused_behavior(self):
        kernel = NPUMXFP8MoEMethod("w13")
        self.assertFalse(
            ascend_runner._uses_fused_gmm1(
                kernel, self._runner_config(swiglu_limit=7.0)
            )
        )

    def test_shared_fused_w4a8_preserves_incoming_scale_and_dtype_contract(self):
        kernel = NPUW4A8MXFP4MoEMethod()
        x = torch.zeros(3, 128, dtype=torch.float8_e4m3fn)
        paired = torch.zeros(3, 2, 2, dtype=torch.uint8)
        info = SimpleNamespace(w13_weight_scale=torch.zeros(2, 2, 96, 2))
        for scale in (None, paired, paired.flatten(1)):
            with (
                self.subTest(scale_shape=None if scale is None else scale.shape),
                patch.object(
                    kernel, "hidden_states_quantizer", return_value=(x, paired)
                ) as quantize,
                patch.object(
                    kernel.fused_matmul, "forward", return_value=(x, paired)
                ) as matmul,
                patch.object(
                    moe_methods, "_get_float4_e2m1fn_x2_dtype", return_value=296
                ),
                patch.object(moe_methods, "_require_e8m0_dtype", return_value=293),
            ):
                kernel.apply_fused_gmm1_swiglu(info, x, torch.tensor([1, 2]), scale, 1)
                self.assertEqual(quantize.call_count, int(scale is None))
                kwargs = matmul.call_args.kwargs
                self.assertIsNone(kwargs["x_dtype"])
                self.assertEqual(kwargs["weight_dtype"], 296)
                self.assertEqual(kwargs["quant_dtype"], torch.float8_e4m3fn)
                self.assertEqual(kwargs["weight_scale_dtype"], 293)
                self.assertEqual(kwargs["x_scale_dtype"], 293)
                torch.testing.assert_close(kwargs["x_scale"], paired)

    def test_unfused_bias_is_forwarded(self):
        kernel = NPUW4A4MXFP4MoEMethod("w13")
        for attr in ("w13_weight_bias", "w13_scale_bias"):
            bias = torch.ones(2, 96)
            info = SimpleNamespace(
                w13_weight_scale=torch.zeros(2, 1, 96, 2), **{attr: bias}
            )
            with (
                self.subTest(bias=attr),
                patch.object(kernel.matmul, "forward") as matmul,
            ):
                kernel.apply(
                    info,
                    torch.zeros(3, 32, dtype=torch.uint8),
                    torch.tensor([1, 2]),
                    torch.zeros(3, 1, 2, dtype=torch.uint8),
                    torch.bfloat16,
                    "w13",
                    1,
                )
                self.assertIs(matmul.call_args.kwargs["bias"][0], bias)

    def test_creates_packed_weights_and_ceil_divided_scales(self):
        """K=80 is not a multiple of the 32-element block.

        A truncating scale count would allocate 2 scales instead of 3 and drop
        the last partial block, so the checkpoint fails to load.
        """
        layer = torch.nn.Module()
        for weight_prefix in ("w13", "w2"):
            ModelSlimW4A4MXFP4MoE({}, weight_prefix).create_weights(
                layer=layer,
                num_experts=2,
                hidden_size=64,
                intermediate_size_per_partition=80,
            )

        self.assertEqual(layer.w13_weight.shape, (2, 160, 32))
        self.assertEqual(layer.w13_weight_scale.shape, (2, 160, 2))
        self.assertEqual(layer.w2_weight.shape, (2, 64, 40))
        self.assertEqual(layer.w2_weight_scale.shape, (2, 64, 3))


if __name__ == "__main__":
    unittest.main()
