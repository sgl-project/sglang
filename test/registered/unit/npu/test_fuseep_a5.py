import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu.moe import fuseep
from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUMXFP8MoEMethod,
    NPUW4A4MXFP4MoEMethod,
    NPUW4A8MXFP4MoEMethod,
)
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_layer(kernel_type=NPUMXFP8MoEMethod, online=False):
    layer = torch.nn.Module()
    layer.w13_kernel = object.__new__(kernel_type)
    layer.w2_kernel = object.__new__(kernel_type)
    layer.hidden_size, layer.intermediate_size_per_partition = 128, 64
    layer.num_local_experts, layer.num_experts = 2, 4
    layer.moe_ep_size, layer.moe_tp_size = 2, 1
    layer.num_fused_shared_experts = 0
    layer.moe_runner_config = MoeRunnerConfig(activation="silu", top_k=2, layer=layer)
    fp4 = kernel_type is not NPUMXFP8MoEMethod
    for prefix, channels, inputs in (("w13", 128, 128), ("w2", 128, 64)):
        shape = (2, channels, inputs // (2 if fp4 else 1))
        dtype = (
            torch.bfloat16 if online else torch.uint8 if fp4 else torch.float8_e4m3fn
        )
        weight = torch.arange(torch.tensor(shape).prod().item()).reshape(shape) % 127
        setattr(
            layer,
            f"{prefix}_weight",
            torch.nn.Parameter(weight.to(dtype), requires_grad=False),
        )
        if not online:
            scale = (
                torch.arange(2 * channels * inputs // 32, dtype=torch.int64).reshape(
                    2, channels, inputs // 32
                )
                % 16
                + 120
            )
            setattr(
                layer,
                f"{prefix}_weight_scale",
                torch.nn.Parameter(scale.to(torch.uint8), requires_grad=False),
            )
    return layer


class TestA5MegaMoE(CustomTestCase):
    def setUp(self):
        for patcher in (
            patch.object(fuseep, "_check_mega_moe_support"),
            patch.object(
                fuseep.envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK,
                "get",
                return_value=8192,
            ),
            patch.object(
                fuseep.envs.SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT,
                "get",
                return_value=False,
            ),
            patch.dict(
                sys.modules,
                {
                    "torch_npu": SimpleNamespace(
                        float4_e2m1fn_x2=296, get_npu_format=lambda _: 29
                    )
                },
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_offline_precision_activation_matrix(self):
        for kernel_type, mode in (
            (NPUMXFP8MoEMethod, "mx_fp8_e4m3"),
            (NPUW4A8MXFP4MoEMethod, "mx_fp8_e4m3_w4"),
            (NPUW4A4MXFP4MoEMethod, "mx_fp4_e2m1"),
        ):
            for activation in ("silu", "situ"):
                with self.subTest(mode=mode, activation=activation):
                    layer = make_layer(kernel_type)
                    layer.moe_runner_config.activation = activation
                    layer.moe_runner_config.gemm1_alpha = (
                        2.0 if activation == "situ" else None
                    )
                    originals = {
                        prefix: (
                            getattr(layer, prefix + "_weight").detach().clone(),
                            getattr(layer, prefix + "_weight_scale").detach().clone(),
                        )
                        for prefix in ("w13", "w2")
                    }
                    with patch.object(
                        fuseep,
                        "npu_format_cast",
                        side_effect=lambda weight, **_: weight,
                    ) as cast:
                        fuseep._process_mega_moe_weights(layer)
                    for prefix, (weight, scale) in originals.items():
                        self.assertEqual(len(getattr(layer, prefix + "_weight")), 1)
                        torch.testing.assert_close(
                            getattr(layer, prefix + "_weight")[0].view(torch.uint8),
                            weight.view(torch.uint8),
                        )
                        prepared = getattr(layer, prefix + "_weight_scale")[0]
                        self.assertEqual(prepared.dtype, torch.float8_e8m0fnu)
                        torch.testing.assert_close(
                            prepared.view(torch.uint8).reshape_as(scale), scale
                        )
                        self.assertTrue(prepared.is_contiguous())
                    self.assertEqual(layer._mega_moe_quant_mode, mode)
                    self.assertEqual(
                        cast.call_count, 0 if kernel_type is NPUMXFP8MoEMethod else 2
                    )
                    if kernel_type is NPUW4A8MXFP4MoEMethod:
                        self.assertTrue(
                            all(
                                call.kwargs["input_dtype"] == 296
                                for call in cast.call_args_list
                            )
                        )
                    elif kernel_type is NPUW4A4MXFP4MoEMethod:
                        self.assertEqual(cast.call_args_list[0].kwargs, {})
                        self.assertEqual(
                            cast.call_args_list[1].kwargs["input_dtype"], 296
                        )
                    captured = {}
                    buffer = SimpleNamespace(
                        fused_deep_moe=lambda **kw: (
                            captured.update(kw) or kw["x"],
                            None,
                        )
                    )
                    with (
                        patch.object(fuseep, "_get_fuseep_buffer", return_value=buffer),
                        patch.object(
                            fuseep,
                            "get_exec",
                            return_value=SimpleNamespace(
                                moe=SimpleNamespace(fuseep_mode=3)
                            ),
                        ),
                    ):
                        x = torch.zeros(5000, 128, dtype=torch.bfloat16)
                        routes = SimpleNamespace(
                            topk_ids=torch.zeros(5000, 2, dtype=torch.int64),
                            topk_weights=torch.ones(5000, 2),
                        )
                        self.assertIs(fuseep.forward_fuseep(layer, x, routes), x)
                    self.assertEqual(captured["quant_mode"], mode)
                    self.assertIsNone(captured["l1_bias"])
                    self.assertIsNone(captured["l2_bias"])
                    self.assertEqual(
                        captured["activation"],
                        "situ" if activation == "situ" else "swiglu",
                    )
                    if activation == "situ":
                        self.assertEqual(captured["beta"], 2.0)
                        self.assertIsNone(captured["linear_beta"])
                    first = layer.w13_weight
                    fuseep._process_mega_moe_weights(layer)
                    self.assertIs(layer.w13_weight, first)

    def test_online_mxfp8_postload_converts_each_projection_once(self):
        layer = make_layer(online=True)
        calls = []

        def quantize(weight, prefix):
            calls.append(prefix)
            return weight.to(torch.float8_e4m3fn), torch.full(
                (*weight.shape[:2], weight.shape[-1] // 64, 2), 127, dtype=torch.uint8
            ).view(torch.float8_e8m0fnu)

        with (
            patch.object(
                NPUMXFP8MoEMethod, "_quantize_weight_online", side_effect=quantize
            ),
            patch(
                "sglang.srt.layers.moe.get_moe_a2a_backend",
                return_value=SimpleNamespace(is_ascend_fuseep=lambda: True),
            ),
            patch(
                "sglang.srt.runtime_context.get_exec",
                return_value=SimpleNamespace(moe=SimpleNamespace(fuseep_mode=3)),
            ),
            patch.object(
                fuseep,
                "get_exec",
                return_value=SimpleNamespace(moe=SimpleNamespace(fuseep_mode=3)),
            ),
        ):
            layer.w13_kernel.process_weights_after_loading(layer, "w13")
            layer.w2_kernel.process_weights_after_loading(layer, "w2")
        self.assertEqual(calls, ["w13", "w2"])
        self.assertEqual(layer.w2_weight_scale[0].shape, (2, 128, 1, 2))

    def test_offline_fp4_postload_hooks_prepare_both_projections_once(self):
        for kernel_type in (NPUW4A8MXFP4MoEMethod, NPUW4A4MXFP4MoEMethod):
            with self.subTest(kernel_type=kernel_type):
                layer = make_layer(kernel_type)
                with (
                    patch.object(
                        fuseep,
                        "npu_format_cast",
                        side_effect=lambda weight, **_: weight,
                    ) as cast,
                    patch(
                        "sglang.srt.layers.moe.get_moe_a2a_backend",
                        return_value=SimpleNamespace(is_ascend_fuseep=lambda: True),
                    ),
                    patch(
                        "sglang.srt.runtime_context.get_exec",
                        return_value=SimpleNamespace(
                            moe=SimpleNamespace(fuseep_mode=3)
                        ),
                    ),
                    patch.object(
                        fuseep,
                        "get_exec",
                        return_value=SimpleNamespace(
                            moe=SimpleNamespace(fuseep_mode=3)
                        ),
                    ),
                ):
                    layer.w13_kernel.process_weights_after_loading(layer, "w13")
                    layer.w2_kernel.process_weights_after_loading(layer, "w2")
                self.assertEqual(cast.call_count, 2)
                self.assertTrue(layer._fuseep_weights_processed)

    def test_invalid_checkpoint_rejected_before_parameter_replacement(self):
        for defect in ("mixed", "bias", "scale", "shape", "tp", "activation"):
            with self.subTest(defect=defect):
                layer = make_layer()
                original = layer.w13_weight
                if defect == "mixed":
                    layer.w2_kernel = object.__new__(NPUW4A4MXFP4MoEMethod)
                elif defect == "bias":
                    layer.w2_weight_bias = torch.nn.Parameter(
                        torch.ones(2, 128), requires_grad=False
                    )
                elif defect == "scale":
                    layer.w2_weight_scale = torch.nn.Parameter(
                        torch.ones(2, 128, 2), requires_grad=False
                    )
                elif defect == "shape":
                    layer.w2_weight = torch.nn.Parameter(
                        torch.zeros(2, 128, 32, dtype=torch.float8_e4m3fn),
                        requires_grad=False,
                    )
                elif defect == "tp":
                    layer.moe_tp_size = 2
                else:
                    layer.moe_runner_config.swiglu_limit = 7.0
                with self.assertRaises(ValueError):
                    fuseep._process_mega_moe_weights(layer)
                self.assertIs(layer.w13_weight, original)
                self.assertFalse(getattr(layer, "_mega_moe_weights_processed", False))

    def test_fp4_format_failure_is_atomic(self):
        layer = make_layer(NPUW4A8MXFP4MoEMethod)
        original = layer.w13_weight
        with (
            patch.dict(
                sys.modules,
                {
                    "torch_npu": SimpleNamespace(
                        float4_e2m1fn_x2=296, get_npu_format=lambda _: 2
                    )
                },
            ),
            patch.object(
                fuseep, "npu_format_cast", side_effect=lambda weight, **_: weight
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "requires NZ"):
                fuseep._process_mega_moe_weights(layer)
        self.assertIs(layer.w13_weight, original)

    def test_cli_accepts_existing_online_mxfp8_configuration(self):
        from sglang.srt.arg_groups.overrides import _a2a_fusion_adjustments

        args = SimpleNamespace(
            fuseep_mode=3,
            moe_a2a_backend="ascend_fuseep",
            device="npu",
            quantization="mxfp8",
            disable_shared_experts_fusion=True,
            speculative_algorithm=None,
            enable_eplb=False,
        )
        self.assertEqual(_a2a_fusion_adjustments(args), {})
        args.quantization = "mxfp4"
        with self.assertRaises(ValueError):
            _a2a_fusion_adjustments(args)

    def test_platform_capacity_and_native_keyword_contract(self):
        self.assertEqual(fuseep._mega_moe_capacity("a5"), 8192)
        with self.assertRaisesRegex(ValueError, "4096"):
            fuseep._mega_moe_capacity("a3")

        def operator(
            *,
            x,
            topk_ids,
            topk_weights,
            l1_weights,
            l2_weights,
            sym_buffer,
            l1_weights_sf,
            l2_weights_sf,
            l1_bias,
            l2_bias,
            x_active_mask,
            activation,
            activation_params,
        ):
            pass

        fuseep._check_mega_moe_operator(operator)
        with self.assertRaisesRegex(RuntimeError, "weight1_type"):
            fuseep._check_mega_moe_operator(operator, mxfp=True)


if __name__ == "__main__":
    unittest.main()
