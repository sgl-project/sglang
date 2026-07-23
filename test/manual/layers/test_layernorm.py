import itertools
import unittest

import torch

from sglang.srt.layers.layernorm import (
    Gemma3RMSNorm,
    GemmaRMSNorm,
    LayerNorm,
    RMSNorm,
)
from sglang.test.test_utils import CustomTestCase


class TestRMSNorm(CustomTestCase):
    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [7, 83, 4096]
    HIDDEN_SIZES = [768, 769, 770, 771, 5120, 5124, 5125, 5126, 8192, 8199]
    ADD_RESIDUAL = [False, True]
    SEEDS = [0]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_rms_norm_test(self, num_tokens, hidden_size, add_residual, dtype, seed):
        torch.manual_seed(seed)

        layer = RMSNorm(hidden_size).to(dtype=dtype)
        layer.weight.data.normal_(mean=1.0, std=0.1)
        scale = 1 / (2 * hidden_size)
        x = torch.randn(num_tokens, hidden_size, dtype=dtype) * scale
        residual = torch.randn_like(x) * scale if add_residual else None

        with torch.inference_mode():
            ref_out = layer.forward_native(x, residual)
            out = layer(x, residual)

        if add_residual:
            self.assertTrue(torch.allclose(out[0], ref_out[0], atol=1e-2, rtol=1e-2))
            self.assertTrue(torch.allclose(out[1], ref_out[1], atol=1e-2, rtol=1e-2))
        else:
            self.assertTrue(torch.allclose(out, ref_out, atol=1e-2, rtol=1e-2))

    def test_rms_norm(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.HIDDEN_SIZES,
            self.ADD_RESIDUAL,
            self.DTYPES,
            self.SEEDS,
        ):
            with self.subTest(
                num_tokens=params[0],
                hidden_size=params[1],
                add_residual=params[2],
                dtype=params[3],
                seed=params[4],
            ):
                self._run_rms_norm_test(*params)


class TestGemmaRMSNorm(CustomTestCase):
    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [7, 83, 4096]
    HIDDEN_SIZES = [768, 769, 770, 771, 5120, 5124, 5125, 5126, 8192, 8199]
    ADD_RESIDUAL = [False, True]
    SEEDS = [0]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_gemma_rms_norm_test(
        self, num_tokens, hidden_size, add_residual, dtype, seed
    ):
        torch.manual_seed(seed)

        layer = GemmaRMSNorm(hidden_size).to(dtype=dtype)
        layer.weight.data.normal_(mean=1.0, std=0.1)
        scale = 1 / (2 * hidden_size)
        x = torch.randn(num_tokens, hidden_size, dtype=dtype) * scale
        residual = torch.randn_like(x) * scale if add_residual else None

        with torch.inference_mode():
            ref_out = layer.forward_native(x, residual)
            out = layer(x, residual)

        if add_residual:
            self.assertTrue(torch.allclose(out[0], ref_out[0], atol=1e-3, rtol=1e-3))
            self.assertTrue(torch.allclose(out[1], ref_out[1], atol=1e-3, rtol=1e-3))
        else:
            self.assertTrue(torch.allclose(out, ref_out, atol=1e-3, rtol=1e-3))

    def test_gemma_rms_norm(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.HIDDEN_SIZES,
            self.ADD_RESIDUAL,
            self.DTYPES,
            self.SEEDS,
        ):
            with self.subTest(
                num_tokens=params[0],
                hidden_size=params[1],
                add_residual=params[2],
                dtype=params[3],
                seed=params[4],
            ):
                self._run_gemma_rms_norm_test(*params)


class TestGemma3RMSNorm(CustomTestCase):
    """Guards Gemma3RMSNorm.forward_cuda, which routes contiguous inputs to the
    fused gemma_rmsnorm kernel and falls back to forward_native otherwise.

    Two failure modes are covered:
    1. Numerical: the fused kernel must match the fp32 native reference.
    2. Layout: for the non-contiguous transposed q/k tensors that
       Gemma3Attention feeds in, forward_cuda must preserve the input's
       memory layout. An earlier fused implementation used .contiguous(),
       which silently re-laid-out memory to standard order; the values stayed
       correct but a downstream permute().view() (the attention KV write)
       then failed with a stride error. Comparing only values would not catch
       this, so we also assert stride parity and replay that permute().view().
    """

    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [7, 83, 4096]
    HIDDEN_SIZES = [768, 2560, 5376]
    SEEDS = [0]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _make_layer(self, dim, dtype):
        layer = Gemma3RMSNorm(dim).to(dtype=dtype)
        # Gemma weights are zero-centered (kernel applies the +1 internally).
        layer.weight.data.normal_(mean=0.0, std=0.1)
        return layer

    def test_contiguous_matches_native(self):
        # 2D hidden-size norms (the common, fused path).
        for num_tokens, hidden_size, dtype, seed in itertools.product(
            self.NUM_TOKENS, self.HIDDEN_SIZES, self.DTYPES, self.SEEDS
        ):
            with self.subTest(
                num_tokens=num_tokens,
                hidden_size=hidden_size,
                dtype=dtype,
                seed=seed,
            ):
                torch.manual_seed(seed)
                layer = self._make_layer(hidden_size, dtype)
                x = torch.randn(num_tokens, hidden_size, dtype=dtype) / (
                    2 * hidden_size
                )
                with torch.inference_mode():
                    ref = layer.forward_native(x)
                    out = layer(x)
                self.assertTrue(torch.allclose(out, ref, atol=1e-2, rtol=1e-2))
                self.assertEqual(out.stride(), ref.stride())

    def test_transposed_qk_layout_preserved(self):
        # Reproduces the exact non-contiguous q/k tensor Gemma3Attention builds:
        # [s, h*d] -> unflatten -> [s, h, d] -> transpose(0,1).unsqueeze(0)
        #   -> [1, h, s, d]  (non-contiguous view over [s, h, d] memory)
        head_dim = 256
        for num_tokens, num_heads, dtype in itertools.product(
            [7, 83], [4, 8], self.DTYPES
        ):
            with self.subTest(num_tokens=num_tokens, num_heads=num_heads, dtype=dtype):
                torch.manual_seed(0)
                layer = self._make_layer(head_dim, dtype)
                flat = torch.randn(num_tokens, num_heads * head_dim, dtype=dtype)
                x = flat.unflatten(-1, (num_heads, head_dim))
                x = x.transpose(0, 1).unsqueeze(0)
                self.assertFalse(x.is_contiguous())
                with torch.inference_mode():
                    ref = layer.forward_native(x)
                    out = layer(x)
                # values match ...
                self.assertTrue(torch.allclose(out, ref, atol=1e-2, rtol=1e-2))
                # ... and the memory layout is preserved (the regression guard).
                self.assertEqual(out.stride(), ref.stride())
                # the downstream op that crashed on the buggy layout must work.
                permuted = out.permute(0, 2, 1, 3)
                permuted.reshape(-1, num_heads, head_dim).view(-1, num_heads * head_dim)

    def test_empty_input(self):
        layer = self._make_layer(2560, torch.bfloat16)
        x = torch.empty(0, 2560, dtype=torch.bfloat16)
        with torch.inference_mode():
            out = layer(x)
        self.assertEqual(out.shape, x.shape)


class TestLayerNorm(CustomTestCase):
    DTYPES = [torch.half, torch.bfloat16]
    PARAM_DTYPES = [torch.bfloat16, torch.float32]
    NUM_TOKENS = [7, 83, 1024]
    HIDDEN_SIZES = [128, 512, 1536, 5120, 5124, 5125, 5126, 7168]
    USE_AFFINE = [False, True]
    USE_BIAS = [False, True]
    SEEDS = [0]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_layer_norm_test(
        self, num_tokens, hidden_size, use_affine, use_bias, dtype, seed, param_dtype
    ):
        torch.manual_seed(seed)

        layer = LayerNorm(
            hidden_size, elementwise_affine=use_affine, bias=use_bias, dtype=param_dtype
        )
        if use_affine:
            layer.weight.data.normal_(mean=1.0, std=0.1)
            if use_bias:
                layer.bias.data.normal_(mean=0.0, std=0.1)

        scale = 1 / (2 * hidden_size)
        x = torch.randn(num_tokens, hidden_size, dtype=dtype) * scale

        with torch.inference_mode():
            ref_out = layer.forward_native(x)
            out = layer(x)

        self.assertTrue(torch.allclose(out, ref_out, atol=1e-2, rtol=1e-3))

        if (
            use_affine
            and use_bias
            and not (dtype == torch.bfloat16 and param_dtype == torch.float32)
        ):
            layer.dtype = torch.float32
            layer.weight.data = layer.weight.data.to(torch.float32)
            layer.bias.data = layer.bias.data.to(torch.float32)
            with torch.inference_mode():
                cuda_out = layer(x.to(torch.bfloat16)).to(x.dtype)

            self.assertTrue(torch.allclose(cuda_out, ref_out, atol=2e-2, rtol=1e-3))

    def test_layer_norm(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.HIDDEN_SIZES,
            self.USE_AFFINE,
            self.USE_BIAS,
            self.DTYPES,
            self.SEEDS,
            self.PARAM_DTYPES,
        ):
            with self.subTest(
                num_tokens=params[0],
                hidden_size=params[1],
                use_affine=params[2],
                use_bias=params[3],
                dtype=params[4],
                seed=params[5],
                param_dtype=params[6],
            ):
                self._run_layer_norm_test(*params)


if __name__ == "__main__":
    unittest.main(verbosity=2)
