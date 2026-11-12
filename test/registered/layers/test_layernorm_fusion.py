import itertools
import unittest
from unittest import mock

import torch

from sglang.srt.layers import layernorm
from sglang.srt.layers.layernorm import Gemma3RMSNorm, RMSNorm
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=2, suite="stage-b-test-1-gpu-small-amd")


class TestRMSNormInputShape(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")

    def test_higher_rank_residual(self):
        torch.manual_seed(0)
        shape = (2, 3, 512)

        cast_modes = (False,) if torch.version.hip is not None else (False, True)
        for cast_x_before_out_mul in cast_modes:
            with self.subTest(cast_x_before_out_mul=cast_x_before_out_mul):
                layer = RMSNorm(
                    shape[-1], cast_x_before_out_mul=cast_x_before_out_mul
                ).to(device="cuda", dtype=torch.bfloat16)
                layer.weight.data.normal_(mean=1.0, std=0.1)
                x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                residual = torch.randn_like(x)

                with torch.inference_mode():
                    expected = layer.forward_native(x.clone(), residual.clone())
                    actual = layer(x.clone(), residual.clone())

                self.assertEqual(actual[0].shape, x.shape)
                self.assertEqual(actual[1].shape, residual.shape)
                torch.testing.assert_close(
                    actual[0], expected[0], atol=1e-2, rtol=1.5e-2
                )
                torch.testing.assert_close(actual[1], expected[1], atol=1e-2, rtol=1e-2)


class TestGemma3RMSNormInputShape(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")

    def test_head_slice_uses_fused_kernel(self):
        torch.manual_seed(0)
        head_dim = 256
        for dtype, num_tokens, num_heads in itertools.product(
            (torch.bfloat16, torch.half), (1, 7, 83), (1, 8)
        ):
            with self.subTest(dtype=dtype, num_tokens=num_tokens, num_heads=num_heads):
                layer = Gemma3RMSNorm(head_dim).to(device="cuda", dtype=dtype)
                layer.weight.data.normal_(mean=0.0, std=0.1)
                # q_norm and k_norm see a head slice of the fused qkv tensor.
                qkv = torch.randn(
                    num_tokens, (num_heads + 2) * head_dim, device="cuda", dtype=dtype
                )
                x = qkv[:, : num_heads * head_dim].unflatten(-1, (num_heads, head_dim))

                with torch.inference_mode():
                    expected = layer.forward_native(x)
                    if torch.version.hip is None:
                        with mock.patch.object(
                            layernorm, "gemma_rmsnorm", wraps=layernorm.gemma_rmsnorm
                        ) as fused:
                            actual = layer(x)
                        fused.assert_called_once()
                    else:
                        actual = layer(x)

                self.assertEqual(actual.shape, x.shape)
                torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


class TestRMSNormFp8QuantFusion(CustomTestCase):
    DTYPES = [torch.bfloat16, torch.half]
    NUM_TOKENS = [7, 83, 512]
    HIDDEN_SIZES = [512, 4096]
    ADD_RESIDUAL = [False, True]
    # Static FP8 linears carry the input scale as [1] or 0-D (``input_scale.max()``).
    SCALE_SHAPES = [(1,), ()]
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

    def _run_fusion_test(
        self, num_tokens, hidden_size, add_residual, dtype, scale_shape=(1,)
    ):
        torch.manual_seed(self.SEED)

        layer = RMSNorm(hidden_size).to(dtype=dtype)
        layer.weight.data.normal_(mean=1.0, std=0.1)
        x = torch.randn(num_tokens, hidden_size, dtype=dtype)
        residual = torch.randn_like(x) if add_residual else None
        # Per-tensor reciprocal scale (as carried by a static FP8 linear).
        scale = torch.full(scale_shape, 0.05, dtype=torch.float32)

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
            self.SCALE_SHAPES,
        ):
            with self.subTest(
                num_tokens=params[0],
                hidden_size=params[1],
                add_residual=params[2],
                dtype=params[3],
                scale_shape=params[4],
            ):
                self._run_fusion_test(*params)

    def test_forward_cuda_quant_linear_dispatch(self):
        """forward_cuda routes to the fused path only when applicable."""
        import sglang.srt.layers.layernorm as ln_mod

        torch.manual_seed(self.SEED)
        hidden_size, num_tokens = 512, 32
        x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        scale = torch.tensor([0.05], dtype=torch.float32)

        orig_static_scale = ln_mod._fp8_static_input_scale
        ln_mod._fp8_static_input_scale = lambda linear: scale
        try:
            plain = RMSNorm(hidden_size).to(dtype=torch.bfloat16)
            plain.weight.data.normal_(mean=1.0, std=0.1)

            with torch.inference_mode():
                # Plain norm -> fused (fp8, scale, dtype) + bf16 residual.
                (q, s, out_dtype), r = plain(
                    x.clone(), residual.clone(), quant_linear=object()
                )

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
            ln_mod._fp8_static_input_scale = orig_static_scale

        self.assertEqual(q.dtype, self.FP8_DTYPE)
        self.assertIs(s, scale)
        self.assertEqual(out_dtype, torch.bfloat16)
        self.assertEqual(r.dtype, torch.bfloat16)

        self.assertEqual(var_out[0].dtype, torch.bfloat16)
        self.assertEqual(cast_out[0].dtype, torch.bfloat16)

    def test_forward_cuda_quant_linear_0d_scale(self):
        """forward_cuda fuses with a 0-D static input scale, with and without a residual."""
        import sglang.srt.layers.layernorm as ln_mod

        torch.manual_seed(self.SEED)
        hidden_size, num_tokens = 4096, 32
        x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        scale = torch.tensor(0.05, dtype=torch.float32)
        layer = RMSNorm(hidden_size).to(dtype=torch.bfloat16)
        layer.weight.data.normal_(mean=1.0, std=0.1)

        orig_static_scale = ln_mod._fp8_static_input_scale
        ln_mod._fp8_static_input_scale = lambda linear: scale
        try:
            for add_residual in self.ADD_RESIDUAL:
                with self.subTest(add_residual=add_residual), torch.inference_mode():
                    res = residual.clone() if add_residual else None
                    ref = layer.forward_native(x.clone(), res)
                    out = layer(x.clone(), res, quant_linear=object())
                    (q, s, _), normed_ref = (
                        (out[0], ref[0]) if add_residual else (out, ref)
                    )
                    self.assertEqual(q.dtype, self.FP8_DTYPE)
                    self.assertIs(s, scale)
                    deq = (q.float() * scale).flatten()
                    cos = torch.nn.functional.cosine_similarity(
                        deq, normed_ref.float().flatten(), dim=0
                    )
                    self.assertGreater(cos.item(), 0.99)
        finally:
            ln_mod._fp8_static_input_scale = orig_static_scale


if __name__ == "__main__":
    unittest.main()
