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
    """Covers Gemma3RMSNorm, whose CUDA path had no test.

    Includes the rank-3 shapes that q_norm/k_norm receive, non-contiguous
    inputs, the residual path, and an fp32 weight against half-precision
    activations -- which is how the module is constructed when it is built
    outside the loader's `set_default_torch_dtype` context
    (`nn.Parameter(torch.zeros(dim))`).
    """

    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [1, 7, 83, 4096]
    HIDDEN_SIZES = [256, 768, 1152, 5120, 5126]
    SEEDS = [0]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _make(self, hidden_size, dtype, seed, weight_fp32):
        torch.manual_seed(seed)
        layer = Gemma3RMSNorm(hidden_size)
        layer.weight.data.normal_(mean=0.0, std=0.1)
        if not weight_fp32:
            layer.weight.data = layer.weight.data.to(dtype)
        return layer

    def _check(self, layer, x):
        with torch.inference_mode():
            ref_out = layer.forward_native(x)
            out = layer(x)
        self.assertEqual(out.shape, ref_out.shape)
        self.assertFalse(torch.isnan(out).any() or torch.isinf(out).any())
        self.assertTrue(torch.allclose(out, ref_out, atol=1e-2, rtol=1e-2))

    def test_gemma3_rms_norm(self):
        for num_tokens, hidden_size, dtype, seed, weight_fp32 in itertools.product(
            self.NUM_TOKENS,
            self.HIDDEN_SIZES,
            self.DTYPES,
            self.SEEDS,
            [True, False],
        ):
            with self.subTest(
                num_tokens=num_tokens,
                hidden_size=hidden_size,
                dtype=dtype,
                seed=seed,
                weight_fp32=weight_fp32,
            ):
                scale = 1 / (2 * hidden_size)
                x = torch.randn(num_tokens, hidden_size, dtype=dtype) * scale
                self._check(self._make(hidden_size, dtype, seed, weight_fp32), x)

    def test_gemma3_rms_norm_3d(self):
        """q_norm / k_norm are called with [tokens, heads, head_dim]."""
        for num_tokens, heads, head_dim, dtype, weight_fp32 in itertools.product(
            [1, 37], [1, 4, 8], [128, 256], self.DTYPES, [True, False]
        ):
            with self.subTest(
                num_tokens=num_tokens,
                heads=heads,
                head_dim=head_dim,
                dtype=dtype,
                weight_fp32=weight_fp32,
            ):
                scale = 1 / (2 * head_dim)
                x = torch.randn(num_tokens, heads, head_dim, dtype=dtype) * scale
                self._check(self._make(head_dim, dtype, 0, weight_fp32), x)

    def test_gemma3_rms_norm_non_contiguous(self):
        """A transposed or sliced view must not be assumed contiguous."""
        for dtype in self.DTYPES:
            head_dim = 256
            scale = 1 / (2 * head_dim)
            layer = self._make(head_dim, dtype, 0, weight_fp32=False)

            # [tokens, heads, head_dim] produced by transposing [heads, tokens, .]
            base = torch.randn(4, 37, head_dim, dtype=dtype) * scale
            transposed = base.transpose(0, 1)
            self.assertFalse(transposed.is_contiguous())
            with self.subTest(dtype=dtype, case="transposed"):
                self._check(layer, transposed)

            # a strided slice along the last dimension
            wide = torch.randn(37, 4, head_dim * 2, dtype=dtype) * scale
            sliced = wide[..., :head_dim]
            self.assertFalse(sliced.is_contiguous())
            with self.subTest(dtype=dtype, case="sliced"):
                self._check(layer, sliced)

    def test_gemma3_rms_norm_residual(self):
        """The residual path updates both tensors in place."""
        for hidden_size, dtype, weight_fp32 in itertools.product(
            [256, 1152], self.DTYPES, [True, False]
        ):
            with self.subTest(
                hidden_size=hidden_size, dtype=dtype, weight_fp32=weight_fp32
            ):
                scale = 1 / (2 * hidden_size)
                layer = self._make(hidden_size, dtype, 0, weight_fp32)
                x = torch.randn(83, hidden_size, dtype=dtype) * scale
                residual = torch.randn(83, hidden_size, dtype=dtype) * scale

                with torch.inference_mode():
                    ref_out, ref_residual = layer.forward_native(
                        x.clone(), residual.clone()
                    )
                    out, out_residual = layer(x.clone(), residual.clone())

                for got, ref in ((out, ref_out), (out_residual, ref_residual)):
                    self.assertEqual(got.shape, ref.shape)
                    self.assertFalse(torch.isnan(got).any() or torch.isinf(got).any())
                    self.assertTrue(torch.allclose(got, ref, atol=1e-2, rtol=1e-2))


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
