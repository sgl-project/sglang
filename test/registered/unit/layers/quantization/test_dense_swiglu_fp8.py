"""Dense SwiGLU/block-FP8 integration: real MLPs, dispatch and graph replay."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.fp8_utils import Fp8GemmRunnerBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.layer_ut_utils import init_single_process_dist
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b", runner_config="4-gpu-b200")


@unittest.skipUnless(get_device_sm() in (100, 103), "Dense block-FP8 fusion on SM100")
@mock.patch.object(fp8_utils, "FP8_GEMM_RUNNER_BACKEND", Fp8GemmRunnerBackend.DEEP_GEMM)
class TestDenseSiluAndMulQuant(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist()

    def _make_mlp(self, model):
        from sglang.srt.models.llama import LlamaMLP
        from sglang.srt.models.qwen2 import Qwen2MLP

        config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
        with set_default_torch_dtype(torch.bfloat16):
            mlp = {"llama": LlamaMLP, "qwen": Qwen2MLP}[model](
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
        """Cold first-decode dispatch and graph replay must match the unfused MLP."""
        from sglang.srt.models.llama import LlamaDecoderLayer
        from sglang.srt.models.qwen2 import Qwen2DecoderLayer

        batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
        torch.manual_seed(123)
        with torch.inference_mode():
            for model in ("llama", "qwen"):
                mlp = self._make_mlp(model)
                self.assertTrue(mlp.activation_quant.enabled)
                self.assertIsNone(mlp.dual_gemm.mode)
                self.assertIsNone(mlp.act_fn._forward_method)
                for m in (1, 3, 32, 128):
                    with self.subTest(model=model, m=m):
                        x = torch.randn(m, 512, device="cuda", dtype=torch.bfloat16)
                        gate_up, _ = mlp.gate_up_proj(x)
                        quantized = mlp.activation_quant(
                            gate_up, mlp.act_fn, mlp.down_proj, batch
                        )
                        self.assertIsInstance(quantized, tuple)
                        expected = mlp(x)
                        actual = mlp(x, batch)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        if m == 1:
                            # Non-decode output must not depend on the fusion helper.
                            with mock.patch.object(
                                type(mlp.activation_quant),
                                "__call__",
                                side_effect=AssertionError(
                                    "non-decode entered fusion dispatch"
                                ),
                            ):
                                for mode in (None, *ForwardMode):
                                    if mode == ForwardMode.DECODE:
                                        continue
                                    candidate = (
                                        SimpleNamespace(forward_mode=mode)
                                        if mode is not None
                                        else None
                                    )
                                    torch.testing.assert_close(
                                        mlp(x, candidate), expected, rtol=0, atol=0
                                    )
                            # Isolate attention/communication, but exercise the actual
                            # decoder -> real MLP handoff. Dropping forward_batch must fail.
                            boundary = SimpleNamespace(
                                prepare=lambda x, batch, **kw: x,
                                finish=lambda x, batch: x,
                            )
                            decoder = SimpleNamespace(
                                attn_boundary=boundary,
                                ffn_boundary=boundary,
                                self_attn=lambda **kw: kw["hidden_states"],
                                mlp=mlp,
                            )
                            forward = {
                                "llama": LlamaDecoderLayer.forward,
                                "qwen": Qwen2DecoderLayer.forward,
                            }[model]
                            with mock.patch.object(
                                mlp.act_fn,
                                "forward",
                                side_effect=AssertionError(
                                    "decode used separate activation"
                                ),
                            ):
                                actual = forward(decoder, None, x, batch)
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            actual = mlp(x, batch)
                        for amplitude in (0, 0.1, 3):
                            x.normal_().mul_(amplitude)
                            graph.replay()
                            torch.testing.assert_close(actual, mlp(x), rtol=0, atol=0)

    def test_fallback_modes_and_overrides(self):
        """Excluded execution modes must return the original activation, not a tuple."""
        from sglang.kernels.fused_op import get_fused_op_backend, set_fused_op_backend
        from sglang.kernels.spec import KernelBackend
        from sglang.srt.runtime_context import get_exec, get_forward, get_lora

        mlp = self._make_mlp("llama")
        x = torch.randn(3, 1536, device="cuda", dtype=torch.bfloat16)
        batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)

        def check(candidate_batch=batch, *, value=x, projection=mlp.down_proj):
            actual = mlp.activation_quant(
                value, mlp.act_fn, projection, candidate_batch
            )
            self.assertIsInstance(actual, torch.Tensor)
            torch.testing.assert_close(actual, mlp.act_fn(value), rtol=0, atol=0)

        for mode in (ForwardMode.EXTEND, ForwardMode.IDLE, ForwardMode.TARGET_VERIFY):
            check(SimpleNamespace(forward_mode=mode))
        check(None)
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

        wrapped_projection = SimpleNamespace(
            quant_method=mlp.down_proj.quant_method,
            weight=mlp.down_proj.weight,
        )
        check(projection=wrapped_projection)
        actual = mlp.activation_quant(x, torch.nn.Identity(), mlp.down_proj, batch)
        self.assertIs(actual, x)

        mlp.act_fn.enter_torch_compile(x.shape[0])
        try:
            check()
            compiled = torch.compile(
                lambda value: mlp.activation_quant(
                    value, mlp.act_fn, mlp.down_proj, batch
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
        batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
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
                    helper(x, act, layer, batch), expected, rtol=0, atol=0
                )
        mlp = self._make_mlp("llama")
        for scope in (
            parallel_scope(tp_size=2, attn_tp_size=2, moe_tp_size=2),
            override_platform(is_sm100=False),
        ):
            with scope:
                helper = DenseSiluAndMulQuant(mlp.down_proj)
            torch.testing.assert_close(
                helper(x, act, mlp.down_proj, batch), expected, rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
