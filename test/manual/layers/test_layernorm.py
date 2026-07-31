import itertools
import unittest

import torch

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


class TestRMSNormFp8QuantFusion(CustomTestCase):
    DTYPES = [torch.bfloat16, torch.half]
    NUM_TOKENS = [7, 83, 512]
    HIDDEN_SIZES = [512, 4096]
    ADD_RESIDUAL = [False, True]
    SEED = 0
    FP8_DTYPE = torch.float8_e4m3fn

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        from sglang.srt.layers.layernorm import _flashinfer_rmsnorm_quant_available

        if not _flashinfer_rmsnorm_quant_available:
            raise unittest.SkipTest("flashinfer rmsnorm_quant is not available")
        torch.set_default_device("cuda")

    def _run_fusion_test(self, num_tokens, hidden_size, add_residual, dtype):
        torch.manual_seed(self.SEED)

        layer = RMSNorm(hidden_size).to(dtype=dtype)
        layer.weight.data.normal_(mean=1.0, std=0.1)
        x = torch.randn(num_tokens, hidden_size, dtype=dtype)
        residual = torch.randn_like(x) if add_residual else None
        # Per-tensor reciprocal scale (as carried by a static FP8 linear).
        scale = torch.tensor([0.05], dtype=torch.float32)

        with torch.inference_mode():
            ref = layer.forward_native(
                x.clone(), residual.clone() if add_residual else None
            )
            normed_ref = ref[0] if add_residual else ref
            residual_ref = ref[1] if add_residual else None

            result = layer.forward_with_per_tensor_quant_fusion(
                x.clone(), scale, residual.clone() if add_residual else None
            )

        if add_residual:
            (q, s, out_dtype), r = result
        else:
            q, s, out_dtype = result
            r = None

        # Output contract.
        self.assertEqual(q.dtype, self.FP8_DTYPE)
        self.assertIs(s, scale)
        self.assertEqual(out_dtype, dtype)
        self.assertEqual(tuple(q.shape), (num_tokens, hidden_size))
        if add_residual:
            self.assertEqual(r.dtype, dtype)
            self.assertTrue(
                torch.allclose(r.float(), residual_ref.float(), atol=1e-2, rtol=1e-2)
            )

        # Numerical: dequantized (q * scale) matches the reference normed output
        # within FP8 e4m3 precision.
        deq = q.float() * scale
        ref_flat = normed_ref.float().flatten()
        cos = torch.nn.functional.cosine_similarity(deq.flatten(), ref_flat, dim=0)
        self.assertGreater(cos.item(), 0.99)
        rel_err = (
            deq.flatten() - ref_flat
        ).abs().mean() / ref_flat.abs().mean().clamp_min(1e-6)
        self.assertLess(rel_err.item(), 0.1)

    def test_rms_norm_fp8_quant_fusion(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.HIDDEN_SIZES,
            self.ADD_RESIDUAL,
            self.DTYPES,
        ):
            with self.subTest(
                num_tokens=params[0],
                hidden_size=params[1],
                add_residual=params[2],
                dtype=params[3],
            ):
                self._run_fusion_test(*params)

    def test_forward_cuda_quant_linear_dispatch(self):
        """forward_cuda routes to the fused path only when enabled + applicable."""
        import sglang.srt.layers.layernorm as ln_mod

        torch.manual_seed(self.SEED)
        hidden_size, num_tokens = 512, 32
        x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        scale = torch.tensor([0.05], dtype=torch.float32)

        class _ServerArgs:
            enable_flashinfer_rmsnorm_fp8_quant = True
            rl_on_policy_target = None

        server_args = _ServerArgs()
        orig_get_server_args = ln_mod.get_server_args
        orig_static_scale = ln_mod._fp8_static_input_scale
        ln_mod.get_server_args = lambda: server_args
        ln_mod._fp8_static_input_scale = lambda linear: scale
        try:
            plain = RMSNorm(hidden_size).to(dtype=torch.bfloat16)
            plain.weight.data.normal_(mean=1.0, std=0.1)

            with torch.inference_mode():
                # Enabled + plain norm -> fused (fp8, scale, dtype) + bf16 residual.
                (q, s, out_dtype), r = plain(
                    x.clone(), residual.clone(), quant_linear=object()
                )
                # Disabled -> normal path (plain tensors), even with quant_linear.
                server_args.enable_flashinfer_rmsnorm_fp8_quant = False
                out_off = plain(x.clone(), residual.clone(), quant_linear=object())
                server_args.enable_flashinfer_rmsnorm_fp8_quant = True

                # variance_size_override is incompatible -> must not fuse.
                var_layer = RMSNorm(hidden_size, var_hidden_size=hidden_size // 2).to(
                    dtype=torch.bfloat16
                )
                var_out = var_layer(x.clone(), residual.clone(), quant_linear=object())
                # cast_x_before_out_mul (HF semantics) is incompatible -> must not fuse.
                cast_layer = RMSNorm(hidden_size, cast_x_before_out_mul=True).to(
                    dtype=torch.bfloat16
                )
                cast_out = cast_layer(
                    x.clone(), residual.clone(), quant_linear=object()
                )
        finally:
            ln_mod.get_server_args = orig_get_server_args
            ln_mod._fp8_static_input_scale = orig_static_scale

        self.assertEqual(q.dtype, self.FP8_DTYPE)
        self.assertIs(s, scale)
        self.assertEqual(out_dtype, torch.bfloat16)
        self.assertEqual(r.dtype, torch.bfloat16)

        self.assertEqual(out_off[0].dtype, torch.bfloat16)
        self.assertEqual(var_out[0].dtype, torch.bfloat16)
        self.assertEqual(cast_out[0].dtype, torch.bfloat16)


class TestApplyFp8LinearPrequantOutputDtype(CustomTestCase):
    """apply_fp8_linear with a pre-quantized fp8 activation must emit the
    caller-supplied ``pre_quant_output_dtype`` (the model's activation dtype),
    not the fp8 input dtype. Regression test for FP16 models where hardcoding
    bf16 caused a query/key dtype mismatch in attention."""

    DTYPES = [torch.float16, torch.bfloat16]
    FP8_DTYPE = torch.float8_e4m3fn

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run(self, dtype):
        from sglang.srt.layers.quantization.fp8_utils import (
            apply_fp8_linear,
            cutlass_fp8_supported,
        )

        torch.manual_seed(0)
        M, K, N = 33, 512, 256
        cf = cutlass_fp8_supported()
        fp8_info = torch.finfo(self.FP8_DTYPE)

        normed = torch.randn(M, K, dtype=dtype)
        input_scale = torch.tensor([0.05], dtype=torch.float32)
        # Per-channel fp8 weight in column-major (K, N) layout.
        w = torch.randn(N, K, dtype=dtype) * 0.05
        w_scale = (w.abs().amax(dim=1) / fp8_info.max).float()
        weight = (
            (w.float() / w_scale[:, None])
            .clamp(fp8_info.min, fp8_info.max)
            .to(self.FP8_DTYPE)
            .t()
        )

        # Reference: non-pre-quantized input -> output dtype == input dtype.
        ref = apply_fp8_linear(
            input=normed,
            weight=weight,
            weight_scale=w_scale,
            input_scale=input_scale,
            cutlass_fp8_supported=cf,
        )
        self.assertEqual(ref.dtype, dtype)

        qinput = (
            (normed.float() * input_scale.reciprocal())
            .clamp(fp8_info.min, fp8_info.max)
            .to(self.FP8_DTYPE)
        )

        # Pre-quantized input with the dtype propagated -> output matches dtype.
        out = apply_fp8_linear(
            input=qinput,
            weight=weight,
            weight_scale=w_scale,
            input_scale=input_scale,
            cutlass_fp8_supported=cf,
            pre_quant_output_dtype=dtype,
        )
        self.assertEqual(out.dtype, dtype)
        self.assertTrue(torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2))

        # Without the dtype hint, the pre-quantized path falls back to bf16.
        out_default = apply_fp8_linear(
            input=qinput,
            weight=weight,
            weight_scale=w_scale,
            input_scale=input_scale,
            cutlass_fp8_supported=cf,
        )
        self.assertEqual(out_default.dtype, torch.bfloat16)

    def test_prequant_output_dtype(self):
        for dtype in self.DTYPES:
            with self.subTest(dtype=dtype):
                self._run(dtype)


if __name__ == "__main__":
    unittest.main(verbosity=2)
