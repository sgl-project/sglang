import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.hardware_backend.npu.moe import fuseep
from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUUnquantMoEMethod,
    NPUW4A8Int8MoEMethod,
    NPUW8A8Int8MoEMethod,
)
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def make_layer(w4a8=True, bf16=False, intermediate=1024):
    layer = torch.nn.Module()
    kernel_type = (
        NPUUnquantMoEMethod
        if bf16
        else NPUW4A8Int8MoEMethod
        if w4a8
        else NPUW8A8Int8MoEMethod
    )
    kernel = object.__new__(kernel_type)
    kernel.is_per_channel_weight = True
    kernel.activation_use_clip = False
    layer.w13_kernel = layer.w2_kernel = kernel
    layer.moe_runner_config = MoeRunnerConfig(
        activation="silu",
        swiglu_limit=None,
        gemm1_alpha=None,
        gemm1_clamp_limit=None,
        top_k=1,
        layer=layer,
    )
    layer.num_fused_shared_experts = 0
    layer.moe_ep_size, layer.moe_tp_size = 2, 1
    layer.num_experts, layer.num_local_experts = 2, 1
    layer.hidden_size = 1024
    layer.intermediate_size_per_partition = intermediate
    for prefix, channels, inputs in (
        ("w13", 2 * intermediate, 1024),
        ("w2", 1024, intermediate),
    ):
        shape = (1, channels // (2 if w4a8 and not bf16 else 1), inputs)
        weight = (
            torch.arange(torch.tensor(shape).prod().item())
            .reshape(shape)
            .to(torch.bfloat16 if bf16 else torch.int8)
        )
        setattr(
            layer, prefix + "_weight", torch.nn.Parameter(weight, requires_grad=False)
        )
        if not bf16:
            setattr(
                layer,
                prefix + "_weight_scale",
                torch.nn.Parameter(
                    torch.full((1, channels, 1), 2.0), requires_grad=False
                ),
            )
        if w4a8 and not bf16:
            setattr(
                layer,
                prefix + "_scale_bias",
                torch.nn.Parameter(torch.ones(1, channels, 2), requires_grad=False),
            )
    return layer


class TestA3MegaMoE(CustomTestCase):
    def setUp(self):
        self.capacity = patch.object(
            fuseep.envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK,
            "get",
            return_value=1024,
        )
        self.capacity.start()
        self.addCleanup(self.capacity.stop)

    def prepare(self, layer):
        with (
            patch.object(fuseep, "npu_format_cast", side_effect=lambda x: x),
            patch.object(fuseep, "_check_mega_moe_support"),
        ):
            fuseep._process_mega_moe_weights(layer)

    def test_w4a8_layout_scale_bits_and_compensation(self):
        layer = make_layer()
        expected = layer.w13_weight[0].transpose(-2, -1).contiguous().view(torch.int32)
        self.prepare(layer)
        torch.testing.assert_close(layer.w13_weight[0], expected)
        self.assertEqual(layer.w2_weight[0].shape, (1024, 128))
        self.assertEqual(layer.w13_weight_scale[0].tolist(), [0x40000000] * 2048)
        torch.testing.assert_close(layer.w2_scale_bias[0], torch.full((1024,), 2.0))
        first = layer.w13_weight
        self.prepare(layer)
        self.assertIs(layer.w13_weight, first)

    def test_w8a8_keeps_int8_weights_without_compensation(self):
        layer = make_layer(w4a8=False)
        expected = layer.w2_weight[0].transpose(-2, -1).contiguous()
        self.prepare(layer)
        torch.testing.assert_close(layer.w2_weight[0], expected)
        self.assertEqual(layer.w13_weight[0].dtype, torch.int8)
        self.assertFalse(hasattr(layer, "w13_scale_bias"))

    def test_missing_compensation_rejected_before_mutation(self):
        layer = make_layer()
        before = layer.w13_weight
        layer.w2_scale_bias.data.fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "loaded w2_scale_bias"):
            self.prepare(layer)
        self.assertIs(layer.w13_weight, before)
        self.assertFalse(getattr(layer, "_mega_moe_weights_processed", False))

    def test_unsupported_activation_rejected(self):
        layer = make_layer()
        layer.moe_runner_config.gemm1_alpha = 4.0
        with self.assertRaisesRegex(ValueError, "Ordinary SwiGLU"):
            self.prepare(layer)

    def test_groupwise_w4a8_rejected(self):
        layer = make_layer()
        layer.w13_kernel.is_per_channel_weight = False
        with self.assertRaisesRegex(ValueError, "per-channel scales"):
            self.prepare(layer)

    def test_capacity_and_dtype_rejected_before_buffer_allocation(self):
        layer = SimpleNamespace(moe_ep_size=2, moe_tp_size=1)
        for rows, dtype, message in (
            (1025, torch.bfloat16, "exceeding"),
            (1, torch.float32, "BF16"),
        ):
            with (
                self.subTest(rows=rows, dtype=dtype),
                patch.object(
                    fuseep,
                    "get_exec",
                    return_value=SimpleNamespace(moe=SimpleNamespace(fuseep_mode=3)),
                ),
                patch.object(fuseep, "_get_fuseep_buffer") as allocate,
            ):
                with self.assertRaisesRegex(ValueError, message):
                    fuseep.forward_fuseep(
                        layer, torch.zeros(rows, 1024, dtype=dtype), None
                    )
                allocate.assert_not_called()

    def test_forward_matches_fuse_mode_api_and_uses_swiglu(self):
        received = {}

        class Buffer:
            def fused_deep_moe(
                self,
                *,
                x,
                topk_idx,
                topk_weights,
                gmm1_permuted_weight,
                gmm1_permuted_weight_scale,
                gmm2_weight,
                gmm2_weight_scale,
                num_max_dispatch_tokens_per_rank,
                num_experts,
                fuse_mode,
                quant_mode,
                activation,
                l1_bias,
                l2_bias,
            ):
                received.update(
                    mode=fuse_mode,
                    quant_mode=quant_mode,
                    activation=activation,
                    ids=topk_idx,
                    weights=topk_weights,
                    bias=l2_bias,
                )
                return x, None

        layer = make_layer()
        self.prepare(layer)
        hidden = torch.zeros(2, 1024, dtype=torch.bfloat16)
        topk = SimpleNamespace(
            topk_ids=torch.zeros(2, 1, dtype=torch.int64),
            topk_weights=torch.ones(2, 1, dtype=torch.bfloat16),
        )
        with (
            patch.object(
                fuseep,
                "get_exec",
                return_value=SimpleNamespace(moe=SimpleNamespace(fuseep_mode=3)),
            ),
            patch.object(fuseep, "_get_fuseep_buffer", return_value=Buffer()),
        ):
            self.assertIs(fuseep.forward_fuseep(layer, hidden, topk), hidden)
        self.assertEqual(
            (received["mode"], received["quant_mode"], received["activation"]),
            (3, 1, "swiglu"),
        )
        self.assertEqual(received["ids"].dtype, torch.int32)
        self.assertEqual(received["weights"].dtype, torch.float32)
        self.assertIs(received["bias"], layer.w2_scale_bias)

    def test_activation_and_precision_matrix(self):
        received = {}

        class Buffer:
            def fused_deep_moe(
                self,
                *,
                x,
                topk_idx,
                topk_weights,
                gmm1_permuted_weight,
                gmm1_permuted_weight_scale,
                gmm2_weight,
                gmm2_weight_scale,
                num_max_dispatch_tokens_per_rank,
                num_experts,
                fuse_mode,
                quant_mode,
                activation,
                l1_bias,
                l2_bias,
                beta=4.0,
                linear_beta=25.0,
            ):
                received.update(
                    mode=quant_mode,
                    activation=activation,
                    beta=beta,
                    linear_beta=linear_beta,
                    scale=gmm1_permuted_weight_scale,
                    bias=l1_bias,
                )
                return x, None

        cases = (
            ("silu", None, None, "swiglu", 4.0, 25.0),
            ("swiglu", None, None, "swiglu", 4.0, 25.0),
            ("situ", 2.0, 16.0, "situ", 2.0, 16.0),
            ("situ", None, None, "situ", 4.0, None),
            ("situglu", 3.0, None, "situ", 3.0, None),
        )
        hidden = torch.zeros(1, 1024, dtype=torch.bfloat16)
        topk = SimpleNamespace(
            topk_ids=torch.zeros(1, 1, dtype=torch.int64), topk_weights=torch.ones(1, 1)
        )
        for precision in ("bf16", "w8a8", "w4a8"):
            for activation, alpha, clamp, expected, beta, linear_beta in cases:
                with self.subTest(
                    precision=precision, activation=activation, clamp=clamp
                ):
                    layer = make_layer(
                        w4a8=precision == "w4a8", bf16=precision == "bf16"
                    )
                    layer.moe_runner_config.activation = activation
                    layer.moe_runner_config.gemm1_alpha = alpha
                    layer.moe_runner_config.gemm1_clamp_limit = clamp
                    self.prepare(layer)
                    with (
                        patch.object(
                            fuseep,
                            "get_exec",
                            return_value=SimpleNamespace(
                                moe=SimpleNamespace(fuseep_mode=3)
                            ),
                        ),
                        patch.object(
                            fuseep, "_get_fuseep_buffer", return_value=Buffer()
                        ),
                    ):
                        self.assertIs(
                            fuseep.forward_fuseep(layer, hidden, topk), hidden
                        )
                    self.assertEqual(
                        (
                            received["activation"],
                            received["beta"],
                            received["linear_beta"],
                        ),
                        (expected, beta, linear_beta),
                    )
                    self.assertEqual(received["mode"], 0 if precision == "bf16" else 1)
                    self.assertEqual(received["scale"] is None, precision == "bf16")
                    self.assertEqual(received["bias"] is None, precision != "w4a8")

    def test_bf16_weight_layout_and_post_load_hook(self):
        layer = make_layer(bf16=True)
        first = layer.w13_weight[0].transpose(-2, -1).contiguous()
        second = layer.w2_weight[0].transpose(-2, -1).contiguous()
        kernel = layer.w13_kernel
        with (
            patch(
                "sglang.srt.runtime_context.get_exec",
                return_value=SimpleNamespace(moe=SimpleNamespace(fuseep_mode=3)),
            ),
            patch(
                "sglang.srt.layers.moe.get_moe_a2a_backend",
                return_value=SimpleNamespace(is_ascend_fuseep=lambda: True),
            ),
            patch.object(
                fuseep,
                "get_exec",
                return_value=SimpleNamespace(moe=SimpleNamespace(fuseep_mode=3)),
            ),
            patch.object(fuseep, "npu_format_cast", side_effect=lambda x: x),
            patch.object(fuseep, "_check_mega_moe_support"),
            patch.object(kernel, "_set_moe_dispatch_dtype") as dtype,
        ):
            kernel.process_weights_after_loading(layer, "w13")
            kernel.process_weights_after_loading(layer, "w2")
            dtype.assert_called_once_with(layer, "bf16")
        torch.testing.assert_close(layer.w13_weight[0], first)
        torch.testing.assert_close(layer.w2_weight[0], second)
        self.assertEqual(layer._mega_moe_quant_mode, 0)
        self.assertFalse(hasattr(layer, "w13_weight_scale"))

    def test_bf16_rejects_bias_and_fp16_before_mutation(self):
        for invalid in ("bias", "dtype"):
            with self.subTest(invalid=invalid):
                layer = make_layer(bf16=True)
                before = layer.w13_weight
                if invalid == "bias":
                    layer.w2_weight_bias = torch.nn.Parameter(
                        torch.zeros(1, 1024), requires_grad=False
                    )
                else:
                    layer.w2_weight.data = layer.w2_weight.data.to(torch.float16)
                with self.assertRaises(ValueError):
                    self.prepare(layer)
                self.assertIs(layer.w13_weight, before)
                self.assertFalse(getattr(layer, "_mega_moe_weights_processed", False))

    def test_other_activation_contracts_are_not_silently_reinterpreted(self):
        config = dict(
            activation="silu",
            gemm1_alpha=None,
            gemm1_clamp_limit=None,
            swiglu_limit=None,
        )
        cases = (
            dict(activation="gelu"),
            dict(activation="npu_swiglu_oai"),
            dict(swiglu_limit=7.0),
            dict(is_gated=False),
            dict(gemm1_beta=1.0),
            dict(apply_router_weight_on_input=True),
            dict(no_combine=True),
            dict(activation="situ", gemm1_alpha=float("nan")),
            dict(activation="situ", gemm1_clamp_limit=-1.0),
        )
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                fuseep._mega_moe_activation(SimpleNamespace(**(config | changes)))

    def test_mixed_or_unknown_precision_is_rejected_before_conversion(self):
        for replacement in (object(), object.__new__(NPUUnquantMoEMethod)):
            layer = make_layer()
            before = layer.w13_weight
            layer.w2_kernel = replacement
            with self.assertRaisesRegex(ValueError, "homogeneous"):
                self.prepare(layer)
            self.assertIs(layer.w13_weight, before)

    def test_cann92_a3_intermediate_boundary(self):
        layer = make_layer(intermediate=3584)
        self.prepare(layer)
        self.assertEqual(layer.w13_weight[0].shape, (1024, 896))
        self.assertEqual(layer.w2_weight[0].shape, (3584, 128))
        outside = make_layer(intermediate=4096)
        with self.assertRaisesRegex(ValueError, "Expert geometry"):
            self.prepare(outside)

    def test_identity_swiglu_parameters_are_supported(self):
        config = SimpleNamespace(
            activation="swiglu", gemm1_alpha=1.0, gemm1_clamp_limit=None, swiglu_limit=0
        )
        self.assertEqual(fuseep._mega_moe_activation(config), dict(activation="swiglu"))

    def test_modes_one_and_two_keep_the_existing_invocation(self):
        hidden = torch.zeros(1, 1024, dtype=torch.bfloat16)
        layer = make_layer(w4a8=False)
        topk = SimpleNamespace(
            topk_ids=torch.zeros(1, 1, dtype=torch.int64), topk_weights=torch.ones(1, 1)
        )
        for mode in (1, 2):
            buffer = SimpleNamespace(fused_deep_moe=Mock(return_value=(hidden, None)))
            with (
                self.subTest(mode=mode),
                patch.object(
                    fuseep,
                    "get_exec",
                    return_value=SimpleNamespace(moe=SimpleNamespace(fuseep_mode=mode)),
                ),
                patch.object(fuseep, "_get_fuseep_buffer", return_value=buffer),
            ):
                self.assertIs(fuseep.forward_fuseep(layer, hidden, topk), hidden)
            self.assertEqual(buffer.fused_deep_moe.call_args.kwargs["fuse_mode"], mode)
            self.assertNotIn("activation", buffer.fused_deep_moe.call_args.kwargs)

    def test_native_int_operator_contract_is_checked(self):
        def operator(
            x,
            topk_ids,
            topk_weights,
            l1_weights,
            l2_weights,
            sym_buffer,
            *,
            l1_weights_sf=None,
            l2_weights_sf=None,
            l1_bias=None,
            l2_bias=None,
            x_active_mask=None,
            activation="swiglu",
            activation_params=None,
        ):
            return x

        self.assertIn("activation_params", fuseep._check_mega_moe_operator(operator))

    def test_operator_without_activation_params_requires_cann_update(self):
        def operator(
            x,
            topk_ids,
            topk_weights,
            l1_weights,
            l2_weights,
            sym_buffer,
            *,
            l1_weights_sf=None,
            l2_weights_sf=None,
            l1_bias=None,
            l2_bias=None,
            x_active_mask=None,
            activation="swiglu",
        ):
            return x

        with self.assertRaisesRegex(RuntimeError, "activation_params"):
            fuseep._check_mega_moe_operator(operator)

    def test_dependency_loader_owns_operator_compatibility(self):
        def compatible(**kwargs):
            return None

        def public_operator(x):
            return x

        class Buffer:
            def fused_deep_moe(
                self,
                *,
                fuse_mode,
                quant_mode,
                activation,
                beta,
                linear_beta,
                l1_bias,
                l2_bias,
            ):
                pass

        def module(name, **attributes):
            result = ModuleType(name)
            result.__file__ = "/dependency/" + name + ".py"
            result.__dict__.update(attributes)
            return result

        loader = Mock(return_value=(lambda: None, compatible))
        strategy = module(
            "fused_strategy", MegaMoeStrategy=SimpleNamespace(_load_mega_moe_ops=loader)
        )
        packages = {
            "deep_ep": module("deep_ep", Buffer=Buffer),
            "deep_ep.buffer": module(
                "deep_ep.buffer", FuseMode=SimpleNamespace(MEGA_MOE=3)
            ),
            "deep_ep.strategies": module("deep_ep.strategies", fused_strategy=strategy),
            "cann_ops_transformer": module("cann_ops_transformer"),
            "cann_ops_transformer.ops": module(
                "cann_ops_transformer.ops",
                mega_moe=public_operator,
                get_symm_buffer_for_mega_moe=lambda: None,
            ),
        }
        fuseep._check_mega_moe_support.cache_clear()
        self.addCleanup(fuseep._check_mega_moe_support.cache_clear)
        with (
            patch.dict(sys.modules, packages),
            patch.object(
                torch,
                "npu",
                SimpleNamespace(get_device_name=lambda: "Ascend910_9382"),
                create=True,
            ),
            patch.object(fuseep.importlib.metadata, "version", return_value="test"),
        ):
            _, operator = fuseep._check_mega_moe_support()
        self.assertIs(operator, compatible)
        loader.assert_called_once_with()

    def test_cli_rejects_unsupported_mode_three_combinations(self):
        from sglang.srt.arg_groups.overrides import _a2a_fusion_adjustments

        valid = dict(
            fuseep_mode=3,
            moe_a2a_backend="ascend_fuseep",
            device="npu",
            quantization="modelslim",
            disable_shared_experts_fusion=True,
            speculative_algorithm=None,
            enable_eplb=False,
        )
        self.assertEqual(_a2a_fusion_adjustments(SimpleNamespace(**valid)), {})
        self.assertEqual(
            _a2a_fusion_adjustments(
                SimpleNamespace(**(valid | dict(quantization=None)))
            ),
            {},
        )
        for changes in (
            dict(device="cuda"),
            dict(moe_a2a_backend="none"),
            dict(quantization="fp8"),
            dict(disable_shared_experts_fusion=False),
            dict(speculative_algorithm="EAGLE"),
            dict(enable_eplb=True),
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                _a2a_fusion_adjustments(SimpleNamespace(**(valid | changes)))


if __name__ == "__main__":
    unittest.main()
