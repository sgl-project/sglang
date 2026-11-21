"""Diffusion Linear routing, exact-quality fallback and cold/hot graph behavior."""

import gc
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.kernels.ops.gemm.sm120_bf16 import (
    linear_bf16_cublaslt,
    linear_bf16_quack,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.cosmos3video import (
    Cosmos3GenDecoderLayer,
    _Sm120LinearMethod,
)
from sglang.multimodal_gen.test.quality_tier_admission import (
    assert_error_no_worse_than_reference,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-diffusion", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestSm120Bf16Linear(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        prop = torch.cuda.get_device_properties(0)
        if (prop.major, prop.minor) != (12, 0):
            raise unittest.SkipTest("SM120 required")

    def context(self, quality="lossless"):
        return set_forward_context(
            0,
            None,
            forward_batch=SimpleNamespace(
                sampling_params=SimpleNamespace(quality=quality)
            ),
        )

    def inputs(self, n=6144, k=4096, m=448):
        x = torch.randn(1, m, k, device="cuda", dtype=torch.bfloat16)
        layer = SimpleNamespace(
            weight=torch.randn(n, k, device="cuda", dtype=x.dtype) / k**0.5,
            prefix="gen_layers.0.self_attn.to_qkv",
            tp_size=1,
        )
        return x, layer

    @torch.inference_mode()
    def test_exact_and_generic_fallbacks_preserve_native_output(self):
        x, layer = self.inputs()
        with envs.SGLANG_OPT_COSMOS3_SM120_GEMM.override(True):
            for method, quality in (
                (_Sm120LinearMethod(), "exact"),
                (UnquantizedLinearMethod(), "lossless"),
            ):
                with (
                    self.subTest(method=type(method), quality=quality),
                    self.context(quality),
                ):
                    self.assertTrue(
                        torch.equal(method.apply(layer, x), F.linear(x, layer.weight))
                    )
                    if isinstance(method, _Sm120LinearMethod):
                        self.assertFalse(method._gemm._warmed)
        method = _Sm120LinearMethod()
        with self.context():
            compiled = torch.compile(method.apply, backend="eager", fullgraph=True)
            self.assertTrue(torch.equal(compiled(layer, x), F.linear(x, layer.weight)))
        self.assertFalse(method._gemm._warmed)

    def test_only_gen_tp1_unquantized_layers_opt_in_at_construction(self):
        class ProjectionOnlyAttention(torch.nn.Module):
            def __init__(self, hidden_size, prefix, quant_config, **kwargs):
                super().__init__()
                self.to_qkv = ColumnParallelLinear(
                    hidden_size,
                    hidden_size,
                    bias=False,
                    prefix=prefix,
                    quant_config=quant_config,
                )
                self.to_out = RowParallelLinear(
                    hidden_size,
                    hidden_size,
                    bias=False,
                    prefix=prefix,
                    quant_config=quant_config,
                )

        class CustomMethod(UnquantizedLinearMethod):
            pass

        custom_config = SimpleNamespace(
            get_quant_method=lambda *a, **kw: CustomMethod()
        )
        for enabled, tp, config, expected in (
            (False, 1, None, UnquantizedLinearMethod),
            (True, 2, None, UnquantizedLinearMethod),
            (True, 1, custom_config, CustomMethod),
            (True, 1, None, _Sm120LinearMethod),
        ):
            with (
                self.subTest(enabled=enabled, tp=tp, config=config),
                envs.SGLANG_OPT_COSMOS3_SM120_GEMM.override(enabled),
                patch(
                    "sglang.multimodal_gen.runtime.layers.linear.get_tp_group",
                    return_value=SimpleNamespace(world_size=tp, rank_in_group=0),
                ),
            ):
                block = Cosmos3GenDecoderLayer(
                    hidden_size=256,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    head_dim=128,
                    intermediate_size=512,
                    rms_norm_eps=1e-6,
                    hidden_act="silu",
                    layer_idx=0,
                    quant_config=config,
                    cross_attention_cls=ProjectionOnlyAttention,
                )
                projections = [
                    block.cross_attention.to_qkv,
                    block.cross_attention.to_out,
                    block.mlp.gate_up_proj,
                    block.mlp.down_proj,
                ]
                for projection in projections:
                    self.assertIs(type(projection.quant_method), expected)
                # A separate ordinary Linear never opts in, even with a GEN prefix.
                ordinary = ColumnParallelLinear(256, 256, prefix="gen_layers.0.other")
                self.assertIs(type(ordinary.quant_method), UnquantizedLinearMethod)

    @torch.inference_mode()
    def test_cold_capture_then_warmed_real_backends(self):
        from flashinfer.autotuner import AutoTuner
        from quack import gemm_interface

        # Exercise the real backends on CI's SM120 5090 as well. Only the SKU
        # performance gate is relaxed; shapes, dtypes and kernel calls are real.
        for m, n, module, name in (
            (394, 6144, AutoTuner.get(), "choose_one"),
            (808, 24576, gemm_interface, "gemm_out"),
        ):
            with (
                self.subTest(m=m, n=n),
                patch(
                    "sglang.multimodal_gen.runtime.models.dits.cosmos3video._sm120_gemm_device_supported",
                    return_value=True,
                ),
            ):
                method = _Sm120LinearMethod()
                x, layer = self.inputs(n=n, m=m)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                modes = []
                call = getattr(module, name)

                def record_mode(*args, **kwargs):
                    modes.append(AutoTuner.get().is_tuning_mode)
                    return call(*args, **kwargs)

                with (
                    torch.cuda.stream(stream),
                    self.context(),
                    envs.SGLANG_OPT_COSMOS3_SM120_GEMM.override(True),
                ):
                    F.linear(x, layer.weight)
                    stream.synchronize()
                    with patch.object(
                        module, name, side_effect=record_mode
                    ) as actual_backend:
                        cold = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(cold, stream=stream):
                            cold_out = method.apply(layer, x)
                        cold.replay()
                        stream.synchronize()
                        self.assertTrue(
                            torch.equal(cold_out, F.linear(x, layer.weight))
                        )
                        self.assertEqual(actual_backend.call_count, 0)
                        method.apply(layer, x)
                        calls = actual_backend.call_count
                        self.assertGreater(calls, 0)
                        self.assertEqual(modes[-1], n == 6144)
                        method.apply(layer, x)
                        self.assertFalse(modes[-1])
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            out = method.apply(layer, x)
                        self.assertGreater(actual_backend.call_count, calls)
                        self.assertFalse(modes[-1])
                        x.mul_(0.5)
                        layer.weight.neg_()
                        graph.replay()
                        stream.synchronize()
                        torch.testing.assert_close(
                            out.float(),
                            F.linear(x.float(), layer.weight.float()),
                            rtol=0.02,
                            atol=0.03,
                        )

    @torch.inference_mode()
    def test_fallback_does_not_retain_workspace(self):
        # Outside the measured intervals, including their gaps, execution
        # stays native even though the generic executor accepts these shapes.
        for m, n, k in (
            (63, 24576, 4096),
            (2049, 24576, 4096),
            (513, 4096, 12288),
            (192, 6144, 4096),
            (641, 6144, 4096),
            (1023, 6144, 4096),
            (127, 4096, 4096),
            (224, 512, 256),
        ):
            with (
                self.subTest(m=m, n=n, k=k),
                patch(
                    "sglang.multimodal_gen.runtime.models.dits.cosmos3video._sm120_gemm_device_supported",
                    return_value=True,
                ),
            ):
                x, layer = self.inputs(n=n, k=k, m=m)
                method = _Sm120LinearMethod()
                F.linear(x, layer.weight)
                gc.collect()
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated()
                with self.context(), envs.SGLANG_OPT_COSMOS3_SM120_GEMM.override(True):
                    y = method.apply(layer, x)
                    self.assertTrue(torch.equal(y, F.linear(x, layer.weight)))
                    del y
                gc.collect()
                torch.cuda.synchronize()
                self.assertLessEqual(torch.cuda.memory_allocated(), before)
                self.assertFalse(method._gemm._warmed)

    @torch.inference_mode()
    def test_range_interiors_execute_real_backends_with_bf16_parity(self):
        # Odd lengths must not silently fall back to the former M224/M448
        # whitelist. Check real execution as well as the FP64 error gate.
        for m, n, k, backend in (
            (65, 24576, 4096, linear_bf16_quack),
            (511, 4096, 12288, linear_bf16_quack),
            (193, 6144, 4096, linear_bf16_quack),
            (385, 6144, 4096, linear_bf16_cublaslt),
            (449, 6144, 4096, linear_bf16_quack),
            (1025, 6144, 4096, linear_bf16_quack),
            (129, 4096, 4096, linear_bf16_quack),
            (2047, 4096, 4096, linear_bf16_quack),
        ):
            with (
                self.subTest(m=m, n=n, k=k),
                self.context(),
                patch(
                    "sglang.multimodal_gen.runtime.models.dits.cosmos3video._sm120_gemm_device_supported",
                    return_value=True,
                ),
            ):
                x, layer = self.inputs(n=n, k=k, m=m)
                method = _Sm120LinearMethod()
                out = method.apply(layer, x)
                self.assertIn((backend, x.device.index, m, n, k), method._gemm._warmed)
                assert_error_no_worse_than_reference(
                    reference_fp64=F.linear(x.double(), layer.weight.double()),
                    baseline=F.linear(x, layer.weight),
                    candidate=out,
                )


if __name__ == "__main__":
    unittest.main()
