"""Dense SiLU-and-mul + 1x128 FP8 group quant on gfx942: real block-FP8 MLPs."""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

# Set before importing SGLang: the aiter block-FP8 GEMM is chosen at import.
os.environ.setdefault("SGLANG_USE_AITER", "1")

import torch

from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.fp8_activation import (
    aiter_gfx942_takes_fp8_group_input,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.utils import is_gfx942_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.layer_ut_utils import init_single_process_dist
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="base-b-test-1-gpu-small-amd")


@unittest.skipUnless(is_gfx942_supported(), "the gfx942 block-FP8 fusion")
class TestDenseSiluAndMulQuantGfx942(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist()

    def _make_mlp(self):
        from sglang.srt.models.qwen2 import Qwen2MLP

        config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
        with set_default_torch_dtype(torch.bfloat16):
            mlp = Qwen2MLP(
                512, 768, "silu", quant_config=config, reduce_results=False
            ).cuda()
        with torch.no_grad():
            for layer in (mlp.gate_up_proj, mlp.down_proj):
                weight = (
                    torch.randn(layer.weight.shape, device="cuda", dtype=torch.bfloat16)
                    * 0.02
                )
                layer.weight.copy_(weight.to(layer.weight.dtype))
                layer.weight_scale_inv.fill_(1)
                layer.quant_method.process_weights_after_loading(layer)
        return mlp

    def test_mlp_parity_and_graph_replay(self):
        """The fused decode activation hands down_proj the fnuz fp8 input and
        row-major scale its own quant would produce, so the MLP output matches
        the unfused MLP bit for bit."""
        import aiter

        batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
        torch.manual_seed(123)
        with torch.inference_mode():
            mlp = self._make_mlp()
            self.assertTrue(aiter_gfx942_takes_fp8_group_input(mlp.down_proj))
            for m in (1, 3, 32, 64):
                with self.subTest(m=m):
                    x = torch.randn(m, 512, device="cuda", dtype=torch.bfloat16)
                    gate_up, _ = mlp.gate_up_proj(x)
                    quantized = mlp.activation_quant.forward_decode(
                        gate_up, mlp.act_fn, mlp.down_proj
                    )
                    self.assertIsInstance(quantized, tuple)
                    q, scale = quantized
                    self.assertEqual(q.dtype, aiter.dtypes.fp8)
                    self.assertEqual(tuple(scale.shape), (m, 768 // 128))
                    expected = mlp(x)
                    torch.testing.assert_close(mlp(x, batch), expected, rtol=0, atol=0)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        actual = mlp(x, batch)
                    for amplitude in (0.1, 3):
                        x.normal_().mul_(amplitude)
                        graph.replay()
                        torch.testing.assert_close(actual, mlp(x), rtol=0, atol=0)

    @torch.inference_mode()
    def test_only_the_plain_block_fp8_gemm_takes_the_fused_input(self):
        """A preshuffled, biased, LoRA-wrapped or bf16-fallback linear, or a
        process without the aiter gfx942 GEMM, keeps its own input quant."""
        down = self._make_mlp().down_proj
        bias = torch.nn.Parameter(torch.zeros(512, device="cuda", dtype=torch.bfloat16))
        for attr, value in (
            ("aiter_bpreshuffled", True),
            ("bias", bias),
            ("base_layer", torch.nn.Identity()),
            ("_block_fp8_bf16_weight", down.weight),
        ):
            with (
                self.subTest(attr=attr),
                mock.patch.object(down, attr, value, create=True),
            ):
                self.assertFalse(aiter_gfx942_takes_fp8_group_input(down))
        with mock.patch.object(fp8_utils, "_use_aiter_gfx942", False):
            self.assertFalse(aiter_gfx942_takes_fp8_group_input(down))


if __name__ == "__main__":
    unittest.main()
