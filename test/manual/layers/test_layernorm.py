import itertools
import unittest

import torch

from sglang.srt.batch_invariant_ops import true_on_policy_rms_norm
from sglang.srt.layers.layernorm import GemmaRMSNorm, LayerNorm, RMSNorm
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

    def test_rms_norm_cuda_uses_native_for_fp32_weight(self):
        torch.manual_seed(0)

        hidden_size = 256
        layer = RMSNorm(
            hidden_size,
            cast_x_before_out_mul=True,
            weight_dtype=torch.float32,
            override_orig_dtype=torch.float32,
        )
        layer.weight.data.normal_(mean=1.0, std=0.1)
        x = torch.randn(17, hidden_size, dtype=torch.bfloat16) / hidden_size

        with torch.inference_mode():
            ref_out = layer.forward_native(x)
            out = layer(x)

        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(torch.equal(out, ref_out))

    def test_true_on_policy_fused_rms_norm_qk_dtype_boundary(self):
        torch.manual_seed(0)

        hidden_size = 128
        x = torch.randn(13, hidden_size, dtype=torch.bfloat16)
        weight = torch.randn(hidden_size, dtype=torch.float32)

        actual = true_on_policy_rms_norm(
            x,
            weight,
            eps=1e-6,
            cast_x_before_out_mul=True,
            norm_cast_dtype=torch.bfloat16,
            weight_cast_dtype=torch.float32,
        )

        x_float = x.float()
        expected = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + 1e-6)
        expected = weight * expected.to(torch.bfloat16)

        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_true_on_policy_fused_rms_norm_residual_pair_dtype_boundary(self):
        torch.manual_seed(0)

        hidden_size = 128
        x = torch.randn(13, hidden_size, dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        post_residual = torch.randn_like(x)
        weight = torch.randn(hidden_size, dtype=torch.float32)

        actual, actual_residual = true_on_policy_rms_norm(
            x,
            weight,
            eps=1e-6,
            residual=residual,
            post_residual_addition=post_residual,
            cast_x_before_out_mul=True,
            norm_cast_dtype=torch.float32,
            weight_cast_dtype=torch.float32,
            residual_dtype=torch.float32,
        )

        x_float = x.float() + residual.float() + post_residual.float()
        expected = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + 1e-6)
        expected = weight * expected

        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual_residual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(actual_residual, x_float)

    def test_true_on_policy_fused_rms_norm_final_norm_dtype_boundary(self):
        torch.manual_seed(0)

        hidden_size = 128
        x = torch.randn(13, hidden_size, dtype=torch.float32)
        weight = torch.randn(hidden_size, dtype=torch.float32)

        actual = true_on_policy_rms_norm(
            x,
            weight,
            eps=1e-6,
            cast_x_before_out_mul=True,
            norm_cast_dtype=torch.bfloat16,
            weight_cast_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
        )

        x_float = x.float()
        expected = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + 1e-6)
        expected = weight.to(torch.bfloat16) * expected.to(torch.bfloat16)

        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


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
