"""Dense SiLU-and-mul + dynamic per-token FP8 on CUDA: real MLPs, dispatch and graph replay."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.utils import is_cuda
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.layer_ut_utils import init_single_process_dist
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")
register_cuda_ci(est_time=30, stage="base-b", runner_config="4-gpu-b200")


@unittest.skipUnless(is_cuda(), "CUDA dynamic per-token FP8 SiLU+mul fusion")
class TestDenseSiluAndMulQuantPerTokenCuda(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist()

    def _make_mlp(
        self,
        model: str,
        *,
        hidden_size: int = 512,
        intermediate_size: int = 1024,
        dtype: torch.dtype = torch.bfloat16,
    ):
        from sglang.srt.models.llama import LlamaMLP
        from sglang.srt.models.qwen2 import Qwen2MLP
        from sglang.srt.models.qwen2_moe import Qwen2MoeMLP

        config = Fp8Config(activation_scheme="dynamic")
        cls = {"llama": LlamaMLP, "qwen2": Qwen2MLP, "qwen2_moe": Qwen2MoeMLP}[model]
        with set_default_torch_dtype(dtype):
            mlp = cls(
                hidden_size,
                intermediate_size,
                "silu",
                quant_config=config,
                reduce_results=False,
            ).cuda()
        with torch.no_grad():
            for layer in (mlp.gate_up_proj, mlp.down_proj):
                layer.weight.copy_(
                    torch.randn(layer.weight.shape, device="cuda", dtype=dtype) * 0.02
                )
                layer.quant_method.process_weights_after_loading(layer)
        # On SM100 (B200), DualGemm intercepts small-M LlamaMLP/Qwen2MLP before
        # activation_quant; disable it here so this test exercises activation_quant
        # across all batch sizes on every CUDA architecture.
        if hasattr(mlp, "dual_gemm"):
            mlp.dual_gemm.mode = None
        return mlp

    def _run_unfused_mlp(self, mlp, x):
        gate_up, _ = mlp.gate_up_proj(x)
        act = mlp.act_fn(gate_up)
        out, _ = mlp.down_proj(act)
        return out

    def test_mlp_parity_decode_prefill_and_graph_replay(self):
        """Fused MLP must match unfused MLP bit-for-bit across decode, extend, and CUDA Graph."""
        torch.manual_seed(123)
        with torch.inference_mode():
            for dtype in (torch.bfloat16, torch.float16):
                for model in ("llama", "qwen2", "qwen2_moe"):
                    mlp = self._make_mlp(model, dtype=dtype)
                    self.assertTrue(mlp.activation_quant.enabled)
                    self.assertTrue(mlp.activation_quant.supports_all_modes)

                    for mode in (ForwardMode.DECODE, ForwardMode.EXTEND):
                        batch = SimpleNamespace(forward_mode=mode)
                        for m in (1, 4, 32, 128):
                            with self.subTest(dtype=dtype, model=model, mode=mode, m=m):
                                x = torch.randn(m, 512, device="cuda", dtype=dtype)
                                gate_up, _ = mlp.gate_up_proj(x)
                                quantized = mlp.activation_quant.forward_per_token(
                                    gate_up, mlp.act_fn, mlp.down_proj
                                )
                                self.assertIsInstance(quantized, tuple)
                                self.assertEqual(len(quantized), 3)
                                self.assertEqual(
                                    quantized[0].dtype, torch.float8_e4m3fn
                                )
                                self.assertEqual(quantized[1].shape, (m, 1))
                                self.assertEqual(quantized[2], dtype)

                                expected = self._run_unfused_mlp(mlp, x)
                                actual = (
                                    mlp(x) if model == "qwen2_moe" else mlp(x, batch)
                                )
                                torch.testing.assert_close(
                                    actual, expected, rtol=0, atol=0
                                )

                    # CUDA Graph capture & replay
                    x = torch.randn(16, 512, device="cuda", dtype=dtype)
                    batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
                    _ = mlp(x) if model == "qwen2_moe" else mlp(x, batch)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        actual = mlp(x) if model == "qwen2_moe" else mlp(x, batch)
                    for amplitude in (0.1, 1.0, 3.0):
                        x.normal_().mul_(amplitude)
                        graph.replay()
                        expected = self._run_unfused_mlp(mlp, x)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @torch.inference_mode()
    def test_fallback_conditions(self):
        """Unsupported inputs, static input_scale, or overrides must fall back to act_fn."""
        from sglang.srt.runtime_context import get_exec, get_forward, get_lora

        mlp = self._make_mlp("qwen2")
        x = torch.randn(8, 512, device="cuda", dtype=torch.bfloat16)
        gate_up, _ = mlp.gate_up_proj(x)

        # Non-contiguous input falls back (use a sentinel act_fn since JIT silu_and_mul
        # itself requires contiguous input)
        non_contig = gate_up.transpose(0, 1).contiguous().transpose(0, 1)
        fallback_called = False

        def sentinel_act(t):
            nonlocal fallback_called
            fallback_called = True
            return t[..., : t.shape[-1] // 2]

        out = mlp.activation_quant.forward_per_token(
            non_contig, sentinel_act, mlp.down_proj
        )
        self.assertTrue(fallback_called)
        self.assertIsInstance(out, torch.Tensor)

        # Empty batch falls back
        empty = gate_up[:0]
        out = mlp.activation_quant.forward_per_token(empty, mlp.act_fn, mlp.down_proj)
        self.assertIsInstance(out, torch.Tensor)

        # Deterministic mode falls back
        with get_exec().deterministic.override(enable_deterministic_inference=True):
            out = mlp.activation_quant.forward_per_token(
                gate_up, mlp.act_fn, mlp.down_proj
            )
            self.assertIsInstance(out, torch.Tensor)

        # LoRA enabled falls back
        with get_lora().override(enable_lora=True):
            out = mlp.activation_quant.forward_per_token(
                gate_up, mlp.act_fn, mlp.down_proj
            )
            self.assertIsInstance(out, torch.Tensor)

        # SP active falls back
        with get_forward().scoped(sp_active=True):
            out = mlp.activation_quant.forward_per_token(
                gate_up, mlp.act_fn, mlp.down_proj
            )
            self.assertIsInstance(out, torch.Tensor)

        # Static input_scale disables fusion lazily
        mlp_static = self._make_mlp("qwen2_moe")
        mlp_static.down_proj.input_scale = torch.tensor(
            1.0, device="cuda", dtype=torch.float32
        )
        out = mlp_static.activation_quant.forward_per_token(
            gate_up, mlp_static.act_fn, mlp_static.down_proj
        )
        self.assertIsInstance(out, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
