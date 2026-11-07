"""Dense SwiGLU/block-FP8 integration: real MLPs, dispatch and graph replay."""

import unittest
from functools import partial
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.fp8_utils import Fp8GemmRunnerBackend
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.layer_ut_utils import init_single_process_dist
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=45, stage="base-b", runner_config="1-gpu-large")


def _unfused(mlp, x):
    gate_up, _ = mlp.gate_up_proj(x)
    return mlp.down_proj(mlp.act_fn(gate_up))[0]


@unittest.skipUnless(
    get_device_sm() in (90, 100, 103), "Dense block-FP8 fusion on SM90/SM100"
)
@mock.patch.object(fp8_utils, "FP8_GEMM_RUNNER_BACKEND", Fp8GemmRunnerBackend.DEEP_GEMM)
class TestDenseSiluAndMulQuant(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist()

    def _make_mlp(self, model):
        from sglang.srt.models.llama import LlamaMLP
        from sglang.srt.models.qwen2 import Qwen2MLP
        from sglang.srt.models.qwen2_moe import Qwen2MoeMLP

        config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
        mlp_cls = {"llama": LlamaMLP, "qwen": Qwen2MLP, "qwen_moe": Qwen2MoeMLP}
        with set_default_torch_dtype(torch.bfloat16):
            mlp = mlp_cls[model](
                512,
                768,
                "silu",
                quant_config=config,
                reduce_results=False,
            ).cuda()
        with torch.no_grad():
            for layer in (mlp.gate_up_proj, mlp.down_proj):
                weight = (
                    torch.randn(layer.weight.shape, device="cuda", dtype=torch.bfloat16)
                    * 0.02
                )
                layer.weight.copy_(weight.to(torch.float8_e4m3fn))
                layer.weight_scale_inv.fill_(1)
                layer.quant_method.process_weights_after_loading(layer)
        return mlp

    def test_mlp_parity_and_graph_replay(self):
        """Cold first dispatch and graph replay must match the unfused MLP."""
        torch.manual_seed(123)
        with torch.inference_mode():
            for model in ("llama", "qwen", "qwen_moe"):
                mlp = self._make_mlp(model)
                self.assertTrue(mlp.activation_quant.enabled)
                if model != "qwen_moe":
                    self.assertIsNone(mlp.dual_gemm.mode)
                self.assertIsNone(mlp.act_fn._forward_method)
                for m in (1, 3, 32, 128, 1000):
                    with self.subTest(model=model, m=m):
                        x = torch.randn(m, 512, device="cuda", dtype=torch.bfloat16)
                        gate_up, _ = mlp.gate_up_proj(x)
                        quantized = mlp.activation_quant(
                            x=gate_up, act_fn=mlp.act_fn, down_proj=mlp.down_proj
                        )
                        self.assertIsInstance(quantized, tuple)
                        expected = _unfused(mlp=mlp, x=x)
                        with mock.patch.object(
                            mlp.act_fn,
                            "forward",
                            side_effect=AssertionError("used separate activation"),
                        ):
                            actual = mlp(x)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            actual = mlp(x)
                        for amplitude in (0, 0.1, 3):
                            x.normal_().mul_(amplitude)
                            graph.replay()
                            torch.testing.assert_close(
                                actual, _unfused(mlp=mlp, x=x), rtol=0, atol=0
                            )

    @torch.inference_mode()
    def test_decoder_uses_fusion(self):
        """The real decoder layers must reach the fused activation in every mode."""
        from sglang.srt.models.llama import LlamaDecoderLayer
        from sglang.srt.models.qwen2 import Qwen2DecoderLayer

        boundary = SimpleNamespace(
            prepare=lambda x, batch, **kw: x, finish=lambda x, batch: x
        )
        for model, decoder_cls in (
            ("llama", LlamaDecoderLayer),
            ("qwen", Qwen2DecoderLayer),
        ):
            mlp = self._make_mlp(model)
            x = torch.randn(1, 512, device="cuda", dtype=torch.bfloat16)
            expected = _unfused(mlp=mlp, x=x)
            # Only attention/communication are stubbed; run the real decoder and MLP.
            decoder = SimpleNamespace(
                attn_boundary=boundary,
                ffn_boundary=boundary,
                self_attn=lambda **kw: kw["hidden_states"],
                mlp=mlp,
            )
            for mode in (ForwardMode.DECODE, ForwardMode.EXTEND):
                batch = SimpleNamespace(forward_mode=mode)
                with (
                    self.subTest(model=model, mode=mode),
                    mock.patch.object(
                        mlp.act_fn,
                        "forward",
                        side_effect=AssertionError("used separate activation"),
                    ),
                ):
                    actual = decoder_cls.forward(decoder, None, x, batch)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_fallback_overrides_and_inputs(self):
        """Overrides and unsupported inputs must retain the original activation."""
        from sglang.kernels.fused_op import get_fused_op_backend, set_fused_op_backend
        from sglang.kernels.spec import KernelBackend
        from sglang.srt.runtime_context import get_exec, get_forward, get_lora

        mlp = self._make_mlp("llama")
        x = torch.randn(3, 1536, device="cuda", dtype=torch.bfloat16)

        def check(*, value=x, projection=mlp.down_proj):
            actual = mlp.activation_quant(
                x=value, act_fn=mlp.act_fn, down_proj=projection
            )
            self.assertIsInstance(actual, torch.Tensor)
            torch.testing.assert_close(actual, mlp.act_fn(value), rtol=0, atol=0)

        with get_forward().scoped(sp_active=True):
            check()
        with get_lora().override(enable_lora=True):
            check()
        with get_exec().deterministic.override(enable_deterministic_inference=True):
            check()
        with mock.patch.object(
            mlp.act_fn, "_forward_method", mlp.act_fn.forward_native
        ):
            check()
        old_backend = get_fused_op_backend()
        try:
            set_fused_op_backend(KernelBackend.TORCH)
            check()
        finally:
            set_fused_op_backend(old_backend)
        for dtype in (torch.float16, torch.float32):
            check(value=x.to(dtype))
        # Contiguous single-row views may still have an unaligned row stride.
        check(value=x[:1].as_strided((1, 1536), (1, 1)))

        wrapped_projection = SimpleNamespace(
            quant_method=mlp.down_proj.quant_method,
            weight=mlp.down_proj.weight,
        )
        check(projection=wrapped_projection)
        actual = mlp.activation_quant(
            x=x, act_fn=torch.nn.Identity(), down_proj=mlp.down_proj
        )
        self.assertIs(actual, x)

        mlp.act_fn.enter_torch_compile(x.shape[0])
        try:
            check()
            compiled = torch.compile(
                lambda value: mlp.activation_quant(
                    x=value, act_fn=mlp.act_fn, down_proj=mlp.down_proj
                ),
                backend="eager",
                fullgraph=True,
            )
            torch.testing.assert_close(
                compiled(x), mlp.act_fn.forward_native(x), rtol=0, atol=0
            )
        finally:
            mlp.act_fn.leave_torch_compile()

    def test_constructor_gates(self):
        """Other quantization/backends must retain activation tensors for down_proj."""
        from sglang.srt.layers.activation import SiluAndMul
        from sglang.srt.layers.linear import RowParallelLinear
        from sglang.srt.layers.quantization.fp8_activation import DenseSiluAndMulQuant
        from sglang.srt.runtime_context import override_platform
        from sglang.test.parallel_groups import parallel_scope

        cases = [
            (None, "deep_gemm"),
            ({}, "deep_gemm"),
            ({"weight_block_size": [128, 128]}, "triton"),
            ({"use_mxfp8": True}, "auto"),
            ({"weight_block_size": [32, 32], "scale_fmt": "ue8m0"}, "auto"),
        ]
        act = SiluAndMul()
        x = torch.randn(3, 1536, device="cuda", dtype=torch.bfloat16)
        expected = act(x)
        for options, backend in cases:
            config = (
                Fp8Config(is_checkpoint_fp8_serialized=True, **options)
                if options is not None
                else None
            )
            with (
                self.subTest(config=options, backend=backend),
                mock.patch.object(
                    fp8_utils, "FP8_GEMM_RUNNER_BACKEND", Fp8GemmRunnerBackend(backend)
                ),
            ):
                layer = RowParallelLinear(
                    768,
                    512,
                    bias=False,
                    params_dtype=torch.bfloat16,
                    quant_config=config,
                ).cuda()
                helper = DenseSiluAndMulQuant(layer)
                torch.testing.assert_close(
                    helper(x=x, act_fn=act, down_proj=layer), expected, rtol=0, atol=0
                )
        mlp = self._make_mlp("llama")
        # down_proj takes the local shard, so TP does not change its input.
        with parallel_scope(tp_size=2, attn_tp_size=2, moe_tp_size=2):
            self.assertTrue(DenseSiluAndMulQuant(mlp.down_proj).enabled)
        # SM90 quantizes UE8M0 checkpoints with row-major fp32 scales.
        with mock.patch.object(
            mlp.down_proj.quant_method,
            "w8a8_block_fp8_linear",
            partial(
                fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback,
                act_scale_ue8m0=True,
            ),
        ):
            self.assertEqual(
                DenseSiluAndMulQuant(mlp.down_proj).enabled, get_device_sm() != 90
            )
        for scope in (
            override_platform(is_sm90=False, is_sm100=False),
            mock.patch.object(fp8_utils, "_fp8_block_gemv_enabled", return_value=True),
        ):
            with scope:
                helper = DenseSiluAndMulQuant(mlp.down_proj)
            torch.testing.assert_close(
                helper(x=x, act_fn=act, down_proj=mlp.down_proj),
                expected,
                rtol=0,
                atol=0,
            )

    @torch.inference_mode()
    def test_fixed_geometry_and_live_projection_guards(self):
        """Cache support rules, but reject changed geometry/methods and other layers."""
        from sglang.srt.layers.linear import RowParallelLinear
        from sglang.srt.layers.quantization.fp8_activation import DenseSiluAndMulQuant

        mlp = self._make_mlp("llama")
        layer = mlp.down_proj
        helper = mlp.activation_quant
        x = torch.randn(3, 1536, device="cuda", dtype=torch.bfloat16)
        expected = mlp.act_fn(x)
        # Same-shape weight reloads remain eligible without redoing support checks.
        with (
            mock.patch.object(
                layer, "weight", torch.nn.Parameter(layer.weight.clone(), False)
            ),
            mock.patch.object(
                fp8_utils, "_deepgemm_block_fp8_supported", side_effect=AssertionError
            ),
        ):
            self.assertIsInstance(
                helper(x=x, act_fn=mlp.act_fn, down_proj=layer), tuple
            )
        for attr, value in (
            ("weight", torch.nn.Parameter(layer.weight.reshape(1024, 384), False)),
            ("quant_method", object()),
        ):
            with self.subTest(attr=attr), mock.patch.object(layer, attr, value):
                torch.testing.assert_close(
                    helper(x=x, act_fn=mlp.act_fn, down_proj=layer),
                    expected,
                    rtol=0,
                    atol=0,
                )
        for n, k in ((512, 768), (513, 768), (512, 769)):
            other = RowParallelLinear(
                k,
                n,
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=layer.quant_method.quant_config,
            )
            with self.subTest(n=n, k=k):
                self.assertEqual(
                    DenseSiluAndMulQuant(other).enabled, (n, k) == (512, 768)
                )
                torch.testing.assert_close(
                    helper(x=x, act_fn=mlp.act_fn, down_proj=other),
                    expected,
                    rtol=0,
                    atol=0,
                )


if __name__ == "__main__":
    unittest.main()
