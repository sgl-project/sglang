"""Dense SiLU-and-mul + per-tensor FP8 on gfx95: real MLPs, dispatch and graph replay."""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

# Set before importing SGLang: the unfused reference quantizer is chosen at import.
os.environ.setdefault("SGLANG_USE_AITER", "1")

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.layer_ut_utils import init_single_process_dist
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(is_gfx95_supported(), "the per-tensor fusion is gfx95's")
class TestDenseSiluAndMulQuantPerTensor(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist()

    def _make_mlp(self, model):
        from sglang.srt.models.llama import LlamaMLP
        from sglang.srt.models.qwen2 import Qwen2MLP

        # Online FP8: bf16 weights quantized per tensor at load, activations
        # quantized per tensor at run time.
        config = Fp8Config(activation_scheme="dynamic")
        with (
            envs.SGLANG_USE_AITER.override(True),
            set_default_torch_dtype(torch.bfloat16),
        ):
            mlp = {"llama": LlamaMLP, "qwen": Qwen2MLP}[model](
                512,
                768,
                "silu",
                quant_config=config,
                reduce_results=False,
            ).cuda()
        with torch.no_grad():
            for layer in (mlp.gate_up_proj, mlp.down_proj):
                layer.weight.copy_(
                    torch.randn(layer.weight.shape, device="cuda", dtype=torch.bfloat16)
                    * 0.02
                )
                layer.quant_method.process_weights_after_loading(layer)
        return mlp

    def test_mlp_parity_and_graph_replay(self):
        """The fused decode activation must match the unfused MLP bit for bit."""
        batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
        torch.manual_seed(123)
        with torch.inference_mode():
            for model in ("llama", "qwen"):
                mlp = self._make_mlp(model)
                self.assertTrue(mlp.activation_quant.enabled)
                for m in (1, 3, 32, 128):
                    with self.subTest(model=model, m=m):
                        x = torch.randn(m, 512, device="cuda", dtype=torch.bfloat16)
                        gate_up, _ = mlp.gate_up_proj(x)
                        quantized = mlp.activation_quant.forward_decode(
                            gate_up, mlp.act_fn, mlp.down_proj
                        )
                        self.assertIsInstance(quantized, tuple)
                        expected = mlp(x)
                        torch.testing.assert_close(
                            mlp(x, batch), expected, rtol=0, atol=0
                        )
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            actual = mlp(x, batch)
                        for amplitude in (0.1, 3):
                            x.normal_().mul_(amplitude)
                            graph.replay()
                            torch.testing.assert_close(actual, mlp(x), rtol=0, atol=0)

    @torch.inference_mode()
    def test_large_batches_keep_the_separate_activation(self):
        """Past the fused kernel's row limit decode keeps SiluAndMul."""
        mlp = self._make_mlp("qwen")
        x = torch.randn(129, 512, device="cuda", dtype=torch.bfloat16)
        gate_up, _ = mlp.gate_up_proj(x)
        out = mlp.activation_quant.forward_decode(gate_up, mlp.act_fn, mlp.down_proj)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.dtype, torch.bfloat16)

    @torch.inference_mode()
    def test_aiter_silu_and_mul_keeps_the_separate_activation(self):
        """Decode keeps aiter's SiluAndMul, whose rounding the fusion does not copy."""
        with envs.SGLANG_OPT_USE_AITER_SILU_MUL.override(True):
            mlp = self._make_mlp("qwen")
        x = torch.randn(1, 512, device="cuda", dtype=torch.bfloat16)
        gate_up, _ = mlp.gate_up_proj(x)
        out = mlp.activation_quant.forward_decode(gate_up, mlp.act_fn, mlp.down_proj)
        self.assertIsInstance(out, torch.Tensor)
        torch.testing.assert_close(out, mlp.act_fn(gate_up), rtol=0, atol=0)

    @torch.inference_mode()
    def test_non_decode_bypass(self):
        """Every non-decode mode must avoid the helper and preserve MLP output."""
        mlp = self._make_mlp("qwen")
        x = torch.randn(1, 512, device="cuda", dtype=torch.bfloat16)
        expected = mlp(x)
        with mock.patch.object(
            type(mlp.activation_quant),
            "forward_decode",
            side_effect=AssertionError("non-decode entered fusion dispatch"),
        ):
            for mode in ForwardMode:
                if mode == ForwardMode.DECODE:
                    continue
                with self.subTest(mode=mode):
                    batch = SimpleNamespace(forward_mode=mode)
                    torch.testing.assert_close(mlp(x, batch), expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
