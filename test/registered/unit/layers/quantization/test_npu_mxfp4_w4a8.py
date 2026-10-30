"""CPU contracts for online W4A8 Linear/MoE and shared ModelSlim preparation."""

import unittest
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

import torch

import sglang.srt.layers.quantization  # noqa: F401 (initialize the registry first)
from sglang.srt.hardware_backend.npu.moe.quant import HiddenStatesDynamicQuant
from sglang.srt.hardware_backend.npu.quantization import (
    linear_method_npu,
    moe_methods,
    online_moe_methods,
)
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.moe_runner import ascend as ascend_runner
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization.npu_mxfp4 import Mxfp4W4A8Config
from sglang.srt.layers.quantization.unquant import (
    UnquantizedFusedMoEMethod,
    UnquantizedLinearMethod,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

FP4_DTYPE = 296  # torch_npu's FP4 scalar-type enum.
WEIGHT_POLICY = dict(
    dst_type=FP4_DTYPE, round_mode="rint", scale_alg=2, dst_type_max=7.25
)


class TestW4A8Linear(unittest.TestCase):
    def test_bias_contract(self):
        prepare = linear_method_npu._prepare_mxfp4_w4a8_bias
        self.assertIsNone(prepare(None))
        for shape in ((8,), (1, 8)):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(shape=shape, dtype=dtype):
                    bias = torch.arange(8, dtype=dtype).reshape(shape)
                    actual = prepare(bias)
                    torch.testing.assert_close(actual, bias.bfloat16().reshape(1, 8))
                    if dtype == torch.bfloat16:
                        self.assertEqual(actual.data_ptr(), bias.data_ptr())
                        if len(shape) == 2:
                            self.assertIs(actual, bias)

    def test_unaligned_linear_falls_back_but_aligned_requires_npu(self):
        for size in (4304, 4096):
            with (
                self.subTest(size=size),
                patch(
                    "sglang.srt.layers.quantization.npu_mxfp4.is_npu",
                    return_value=False,
                ),
            ):
                layer = LinearBase.__new__(LinearBase)
                torch.nn.Module.__init__(layer)
                layer.input_size = size
                config = Mxfp4W4A8Config()
                if size == 4304:
                    self.assertIsInstance(
                        config.get_quant_method(layer, "model.visual.mlp.linear_fc2"),
                        UnquantizedLinearMethod,
                    )
                else:
                    with self.assertRaises(NotImplementedError):
                        config.get_quant_method(
                            layer, "model.layers.0.self_attn.o_proj"
                        )

    def test_alignment_uses_the_bound_group_local_k(self):
        prefix = "model.layers.0.self_attn.o_proj"
        for cls, size, tp, aligned in (
            (RowParallelLinear, 4352, 16, False),
            (RowParallelLinear, 4096, 16, True),
            (RowParallelLinear, 4352, 1, True),
            (RowParallelLinear, 4304, 1, False),
            (ColumnParallelLinear, 4352, 16, True),
            (ReplicatedLinear, 4352, 16, True),
        ):
            layer = cls.__new__(cls)
            torch.nn.Module.__init__(layer)
            layer.input_size = size
            layer.tp_group = SimpleNamespace(rank=0, size=tp)
            self.assertFalse(hasattr(layer, "input_size_per_partition"))
            with (
                self.subTest(layer=cls, size=size, tp=tp),
                patch(
                    "sglang.srt.layers.quantization.npu_mxfp4.is_npu", return_value=True
                ),
                patch(
                    "sglang.srt.distributed.utils.get_group_rank_size",
                    return_value=(0, tp),
                ) as group,
            ):
                method = Mxfp4W4A8Config().get_quant_method(layer, prefix)
                expected = (
                    linear_method_npu.NPUMXFP4W4A8LinearMethod
                    if aligned
                    else UnquantizedLinearMethod
                )
                self.assertIsInstance(method, expected)
                if cls is RowParallelLinear:
                    group.assert_called_once_with(layer.tp_group)
                else:
                    group.assert_not_called()
                ignored = Mxfp4W4A8Config(ignored_layers=[prefix]).get_quant_method(
                    layer, prefix
                )
                self.assertIsInstance(ignored, UnquantizedLinearMethod)

    def test_weight_policy_and_packed_layout(self):
        for dtype in (torch.bfloat16, torch.float16):
            for flat_scale in (False, True):
                with self.subTest(dtype=dtype, flat_scale=flat_scale):
                    layer = torch.nn.Module()
                    layer.weight = torch.nn.Parameter(
                        torch.ones(16, 128, dtype=dtype), requires_grad=False
                    )
                    packed = torch.arange(16 * 64).to(torch.uint8).reshape(16, 64)
                    scale = torch.arange(16 * 4).to(torch.uint8).reshape(16, 2, 2)
                    quantize = Mock(
                        return_value=(
                            packed,
                            scale.flatten(-2) if flat_scale else scale,
                        )
                    )
                    with (
                        patch.object(
                            torch.Tensor,
                            "is_npu",
                            property(lambda _: True),
                            create=True,
                        ),
                        patch.object(
                            torch.ops,
                            "npu",
                            SimpleNamespace(npu_dynamic_mx_quant=quantize),
                        ),
                        patch.object(
                            linear_method_npu,
                            "_get_float4_e2m1fn_x2_dtype",
                            return_value=FP4_DTYPE,
                        ),
                        patch.object(
                            linear_method_npu,
                            "npu_format_cast",
                            side_effect=lambda weight, *a, **kw: weight,
                        ),
                    ):
                        linear_method_npu.NPUMXFP4W4A8LinearMethod()._quantize_weights(
                            layer
                        )
                    quantize.assert_called_once_with(ANY, **WEIGHT_POLICY)
                    self.assertEqual(quantize.call_args.args[0].dtype, dtype)
                    torch.testing.assert_close(layer.weight, packed.T)
                    torch.testing.assert_close(
                        layer.weight_scale, scale.transpose(0, 1)
                    )

    def test_activation_policy_and_output_shape(self):
        layer = SimpleNamespace(
            weight=torch.zeros(32, 16, dtype=torch.uint8),
            weight_scale=torch.zeros(1, 16, 2, dtype=torch.uint8),
        )
        quantize = Mock(
            return_value=(
                torch.zeros(3, 64, dtype=torch.float8_e4m3fn),
                torch.zeros(3, 1, 2, dtype=torch.uint8),
            )
        )
        matmul = Mock(return_value=torch.zeros(3, 16, dtype=torch.bfloat16))
        with (
            patch.object(
                torch.ops,
                "npu",
                SimpleNamespace(npu_dynamic_mx_quant=quantize, npu_quant_matmul=matmul),
            ),
            patch.object(
                linear_method_npu, "_get_float4_e2m1fn_x2_dtype", return_value=FP4_DTYPE
            ),
            patch.object(
                linear_method_npu, "_get_float8_e8m0fnu_dtype", return_value=torch.uint8
            ),
        ):
            output = linear_method_npu.NPUMXFP4W4A8LinearMethod().apply(
                layer, torch.ones(1, 3, 64, dtype=torch.bfloat16)
            )
        quantize.assert_called_once_with(ANY, dst_type=torch.float8_e4m3fn)
        self.assertEqual(output.shape, (1, 3, 16))
        self.assertEqual(matmul.call_args.kwargs["x2_dtype"], FP4_DTYPE)


class TestW4A8MoE(unittest.TestCase):
    def test_config_selects_online_experts_and_respects_ignored_layers(self):
        layer = FusedMoE.__new__(FusedMoE)
        torch.nn.Module.__init__(layer)
        layer.use_triton_kernels = layer.use_flashinfer_trtllm_moe = False
        prefix = "model.layers.0.mlp.experts"
        with (
            patch("sglang.srt.layers.quantization.npu_mxfp4.is_npu", return_value=True),
            patch(
                "sglang.srt.layers.quantization.unquant.get_moe_runner_backend",
                return_value=MoeRunnerBackend.AUTO,
            ),
        ):
            method = Mxfp4W4A8Config().get_quant_method(layer, prefix)
            skipped = Mxfp4W4A8Config(ignored_layers=[prefix]).get_quant_method(
                layer, prefix
            )
        self.assertIsInstance(method, online_moe_methods.NPUW4A8MXFP4OnlineMoEMethod)
        self.assertIs(type(skipped), UnquantizedFusedMoEMethod)

    def test_runner_and_meta_allocation(self):
        for backend in (
            MoeRunnerBackend.AUTO,
            MoeRunnerBackend.ASCEND,
            MoeRunnerBackend.FLASHINFER_CUTLASS,
        ):
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
                method = online_moe_methods.NPUW4A8MXFP4OnlineMoEMethod()
                # This base class preserves gate/up shard ordering in FusedMoE.
                self.assertIsInstance(method, UnquantizedFusedMoEMethod)
                layer = torch.nn.Module()
                config = SimpleNamespace(layer=None, is_gated=True)
                if backend == MoeRunnerBackend.FLASHINFER_CUTLASS:
                    with self.assertRaisesRegex(
                        ValueError, "requires --moe-runner-backend"
                    ):
                        method.create_moe_runner(layer, config)
                    runner.assert_not_called()
                    self.assertFalse(hasattr(layer, "w13_kernel"))
                    continue
                method.create_moe_runner(layer, config)
                for prefix in ("w13", "w2"):
                    self.assertIsInstance(
                        getattr(layer, f"{prefix}_kernel"),
                        moe_methods.NPUW4A8MXFP4MoEMethod,
                    )
                self.assertTrue(layer.w13_kernel.use_fused_gmm1)
                self.assertIsNotNone(layer.w13_kernel.fused_matmul)
                self.assertFalse(layer.w2_kernel.use_fused_gmm1)
                self.assertIsNone(layer.w2_kernel.fused_matmul)
                self.assertIs(config.layer, layer)
                runner.assert_called_once_with(MoeRunnerBackend.ASCEND, config)
                self.assertIsNone(method._aiter_runner)
                layer.moe_runner_config = config
                method.create_weights(
                    layer, 2, 128, 64, torch.bfloat16, weight_loader=Mock()
                )
                for prefix, shape in (("w13", (2, 128, 128)), ("w2", (2, 128, 64))):
                    weight = getattr(layer, f"{prefix}_weight")
                    self.assertTrue(weight.is_meta)
                    self.assertEqual(weight.shape, shape)
                    self.assertEqual(weight.dtype, torch.bfloat16)
                    self.assertFalse(hasattr(layer, f"{prefix}_weight_scale"))

    def test_prefix_constructor_preserves_positional_quant_kwargs(self):
        # The existing first positional argument must remain the quant policy.
        legacy = moe_methods.NPUW4A8MXFP4MoEMethod(None)
        self.assertIsNone(legacy.dynamic_quant_kwargs)
        self.assertTrue(legacy.use_fused_gmm1)
        policy = {"dst_type": torch.float8_e4m3fn}
        down = moe_methods.NPUW4A8MXFP4MoEMethod(policy, weight_prefix="w2")
        self.assertIs(down.dynamic_quant_kwargs, policy)
        self.assertFalse(down.use_fused_gmm1)
        self.assertIsNone(down.fused_matmul)

    def test_offline_entries_bind_per_matmul_kernels(self):
        from sglang.srt.hardware_backend.npu.quantization.fp4_moe_methods import (
            NPUW4A8MXFP4FusedMoEMethod,
        )
        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a8_mxfp4_moe import (
            NPUCompressedTensorsW4A8mxfp4MoE,
        )
        from sglang.srt.layers.quantization.modelslim.schemes.modelslim_w4a8_mxfp4_moe import (
            ModelSlimW4A8MXFP4MoE,
        )

        deepseek = NPUW4A8MXFP4FusedMoEMethod()
        entries = (
            SimpleNamespace(
                w13_kernel=ModelSlimW4A8MXFP4MoE({}, "w13").kernel,
                w2_kernel=ModelSlimW4A8MXFP4MoE({}, "w2").kernel,
            ),
            deepseek,
            NPUCompressedTensorsW4A8mxfp4MoE(),
        )
        for entry in entries:
            with self.subTest(entry=type(entry)):
                self.assertTrue(entry.w13_kernel.use_fused_gmm1)
                self.assertIsNotNone(entry.w13_kernel.fused_matmul)
                self.assertFalse(entry.w2_kernel.use_fused_gmm1)
                self.assertIsNone(entry.w2_kernel.fused_matmul)
        self.assertIsNone(deepseek.w13_kernel.dynamic_quant_kwargs)
        self.assertIsNone(deepseek.w2_kernel.dynamic_quant_kwargs)

    def test_fp8_config_fp4_expert_entry_binds_per_matmul_kernels(self):
        from sglang.srt.layers.quantization import fp8

        method = fp8.Fp8MoEMethod.__new__(fp8.Fp8MoEMethod)
        method.is_fp4_expert = True
        layer = torch.nn.Module()
        config = SimpleNamespace(layer=None)
        with (
            patch.object(fp8, "_is_npu", True),
            patch.object(fp8, "MoeRunner") as runner,
        ):
            method.create_moe_runner(layer, config)
            runner.assert_called_once_with(MoeRunnerBackend.ASCEND, config)
        self.assertTrue(layer.w13_kernel.use_fused_gmm1)
        self.assertIsNotNone(layer.w13_kernel.fused_matmul)
        self.assertFalse(layer.w2_kernel.use_fused_gmm1)
        self.assertIsNone(layer.w2_kernel.fused_matmul)

    @staticmethod
    def _runner_config(layer, **kwargs):
        return SimpleNamespace(
            **{
                "layer": layer,
                "activation": "silu",
                "swiglu_limit": None,
                "gemm1_alpha": None,
                "gemm1_clamp_limit": None,
                **kwargs,
            }
        )

    def test_runner_construction_and_execution_both_consult_w13(self):
        x = torch.zeros(3, 64, dtype=torch.bfloat16)
        scale = torch.zeros(3, 1, 2, dtype=torch.uint8)
        for fused in (False, True):
            layer = SimpleNamespace(
                w13_kernel=moe_methods.NPUW4A8MXFP4MoEMethod(weight_prefix="w13"),
                w2_kernel=moe_methods.NPUW4A8MXFP4MoEMethod(weight_prefix="w2"),
            )
            self.assertIsNone(layer.w2_kernel.fused_matmul)
            layer.w13_kernel.use_fused_gmm1 = fused
            # Deliberately disagree: gmm2 must not select the gmm1 activation path.
            layer.w2_kernel.use_fused_gmm1 = not fused
            with (
                self.subTest(fused=fused),
                patch.object(
                    ascend_runner,
                    "get_moe_a2a_backend",
                    return_value=MoeA2ABackend.NONE,
                ),
            ):
                runner = ascend_runner.AscendRunnerCore(self._runner_config(layer))
                self.assertEqual(runner.activation is None, fused)
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

    def test_clamp_and_situ_still_disable_w4a8_fusion(self):
        layer = SimpleNamespace(
            w13_kernel=moe_methods.NPUW4A8MXFP4MoEMethod(weight_prefix="w13"),
            w2_kernel=moe_methods.NPUW4A8MXFP4MoEMethod(weight_prefix="w2"),
        )
        config = self._runner_config(layer, swiglu_limit=7.0)
        self.assertFalse(ascend_runner._uses_fused_gmm1(layer.w13_kernel, config))
        runner = ascend_runner.AscendRunnerCore(config)
        self.assertIsInstance(runner.activation, ascend_runner.NPUSwigluMxfp8Quant)
        self.assertFalse(
            ascend_runner._uses_fused_gmm1(
                layer.w13_kernel, self._runner_config(layer, activation="situ")
            )
        )

    def test_online_policy_and_offline_layout_without_requantization(self):
        packed = torch.arange(2 * 96 * 64).to(torch.uint8).reshape(2, 96, 64)
        scales = torch.arange(2 * 96 * 4).to(torch.uint8).reshape(2, 96, 4)
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
                    source = (
                        torch.zeros(2, 96, 128, dtype=dtype)
                        if online
                        else packed.clone()
                    )
                    layer.register_parameter(
                        f"{prefix}_weight",
                        torch.nn.Parameter(source, requires_grad=False),
                    )
                    if not online:
                        layer.register_parameter(
                            f"{prefix}_weight_scale",
                            torch.nn.Parameter(scales.clone(), requires_grad=False),
                        )
                    quantize = Mock(
                        return_value=(
                            packed.clone(),
                            scales.clone()
                            if flat_scale
                            else scales.reshape(2, 96, 2, 2),
                        )
                    )
                    with (
                        patch.object(
                            torch.Tensor,
                            "is_npu",
                            property(lambda _: True),
                            create=True,
                        ),
                        patch.object(
                            torch.ops,
                            "npu",
                            SimpleNamespace(npu_dynamic_mx_quant=quantize),
                        ),
                        patch.object(
                            moe_methods,
                            "_get_float4_e2m1fn_x2_dtype",
                            return_value=FP4_DTYPE,
                        ),
                        patch.object(
                            moe_methods,
                            "npu_format_cast",
                            side_effect=lambda weight, **kw: weight,
                        ),
                    ):
                        moe_methods.NPUW4A8MXFP4MoEMethod().process_weights_after_loading(
                            layer, prefix
                        )
                    if online:
                        quantize.assert_called_once_with(ANY, **WEIGHT_POLICY)
                        self.assertEqual(quantize.call_args.args[0].dtype, dtype)
                    else:
                        quantize.assert_not_called()
                    torch.testing.assert_close(
                        getattr(layer, f"{prefix}_weight"), packed.transpose(1, 2)
                    )
                    torch.testing.assert_close(
                        getattr(layer, f"{prefix}_weight_scale"),
                        scales.reshape(2, 96, 2, 2).transpose(1, 2),
                    )
                    if prefix == "w13":
                        layer.dispatcher.set_quant_config.assert_called_once_with(
                            {
                                "normal_dispatcher_output_dtype": "bf16",
                                "low_latency_dispatcher_output_dtype": "mxfp8",
                            }
                        )
                    else:
                        layer.dispatcher.set_quant_config.assert_not_called()

    def test_activation_policy_is_separate_from_weight_policy(self):
        info = SimpleNamespace(
            w2_weight=torch.zeros(2, 32, 16, dtype=torch.uint8),
            w2_weight_scale=torch.zeros(2, 1, 16, 2, dtype=torch.uint8),
        )
        with patch.object(moe_methods, "w4a8_mxfp_gmm") as gmm:
            moe_methods.NPUW4A8MXFP4MoEMethod().apply(
                info,
                torch.ones(3, 64, dtype=torch.bfloat16),
                torch.tensor([1, 3]),
                None,
                torch.bfloat16,
                "w2",
                0,
            )
        self.assertEqual(
            gmm.call_args.kwargs["dynamic_quant_kwargs"],
            {"dst_type": torch.float8_e4m3fn},
        )

    def test_dynamic_quant_resolves_only_when_called(self):
        for dtype, use_mx, name in (
            (torch.float8_e4m3fn, False, "npu_dynamic_mx_quant"),
            (torch.int8, False, "npu_dynamic_quant"),
            (torch.quint4x2, False, "npu_dynamic_quant"),
            (FP4_DTYPE, True, "npu_dynamic_mx_quant"),
        ):
            with self.subTest(dtype=dtype, use_mx=use_mx):
                with patch.object(torch.ops, "npu", SimpleNamespace()):
                    quantizer = HiddenStatesDynamicQuant(dtype, use_mx_quant=use_mx)
                x = torch.ones(2, 64)
                expected = (x, torch.ones(2, 2))
                op = Mock(return_value=expected)
                with patch.object(torch.ops, "npu", SimpleNamespace(**{name: op})):
                    actual = quantizer(x)
                self.assertIs(actual[0], expected[0])
                self.assertIs(actual[1], expected[1])
                op.assert_called_once_with(x, dst_type=dtype)
        with self.assertRaises(ValueError):
            HiddenStatesDynamicQuant(torch.float32)


if __name__ == "__main__":
    unittest.main()
