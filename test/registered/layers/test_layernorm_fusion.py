import itertools
import unittest

import torch

import sglang.srt.layers.layernorm as ln_mod
from sglang.kernels.ops.gemm import try_sm120_fp8_linear
from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8
from sglang.srt.layers.layernorm import (
    GemmaRMSNorm,
    RMSNorm,
)
from sglang.srt.layers.quantization.fp4_utils import fp4_quantize
from sglang.srt.layers.quantization.fp8_utils import (
    apply_fp8_linear_bmm_flashinfer,
    flashinfer_per_tensor_fp8_supported,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=11, stage="base-b", runner_config="1-gpu-large")
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

        if torch.version.hip is not None:
            raise unittest.SkipTest("FlashInfer fusion requires CUDA")
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
        """forward_cuda routes to the fused path only when applicable."""

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


class TestGemmaRMSNormInputQuantFusion(CustomTestCase):
    NUM_TOKENS = [1, 7, 200]
    HIDDEN_SIZE = 2048

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if torch.version.hip is not None:
            raise unittest.SkipTest("FlashInfer fusion requires CUDA")
        torch.set_default_device("cuda")

    def _layer_and_inputs(self, num_tokens, add_residual):
        torch.manual_seed(0)
        layer = GemmaRMSNorm(self.HIDDEN_SIZE).to(dtype=torch.bfloat16)
        layer.weight.weight_loader(
            layer.weight, torch.randn(self.HIDDEN_SIZE, dtype=torch.bfloat16) * 0.1
        )
        x = torch.randn(num_tokens, self.HIDDEN_SIZE, dtype=torch.bfloat16)
        residual = torch.randn_like(x) if add_residual else None
        return layer, x, residual

    def _run_fused(self, layer, x, residual, fp8_scale=None, fp4_scale=None):

        orig = ln_mod._fp8_static_input_scale, ln_mod._nvfp4_input_global_scale
        ln_mod._fp8_static_input_scale = lambda linear: fp8_scale
        ln_mod._nvfp4_input_global_scale = lambda linear: fp4_scale
        try:
            layer.fuse_input_quant(object())
            with torch.inference_mode():
                ref = layer.forward_native(
                    x.clone(), None if residual is None else residual.clone()
                )
                out = layer(x.clone(), None if residual is None else residual.clone())
        finally:
            ln_mod._fp8_static_input_scale, ln_mod._nvfp4_input_global_scale = orig
        if residual is None:
            return out, None, ref, None
        return out[0], out[1], ref[0], ref[1]

    def test_fp8(self):

        # 0-dim, as ModelOpt FP8 linears hold their static input scale.
        scale = torch.tensor(0.05, dtype=torch.float32)
        for num_tokens, add_residual in itertools.product(
            self.NUM_TOKENS, (False, True)
        ):
            with self.subTest(num_tokens=num_tokens, add_residual=add_residual):
                layer, x, residual = self._layer_and_inputs(num_tokens, add_residual)
                (q, s, out_dtype), r, normed_ref, r_ref = self._run_fused(
                    layer, x, residual, fp8_scale=scale
                )
                self.assertEqual(q.dtype, torch.float8_e4m3fn)
                self.assertIs(s, scale)
                self.assertEqual(out_dtype, torch.bfloat16)
                if add_residual:
                    torch.testing.assert_close(r, r_ref, atol=1e-2, rtol=1e-2)
                deq = (q.float() * scale).flatten()
                ref_flat = normed_ref.float().flatten()
                cos = torch.nn.functional.cosine_similarity(deq, ref_flat, dim=0)
                self.assertGreater(cos.item(), 0.999)

    def test_nvfp4(self):

        if torch.cuda.get_device_capability()[0] < 10:
            self.skipTest("NVFP4 quantization needs SM100 or newer")
        # 0-dim, as ModelOpt NVFP4 linears hold input_scale_inv.
        global_scale = torch.tensor(448 * 6 / 8.0, dtype=torch.float32)
        kb = self.HIDDEN_SIZE // 16
        for num_tokens, add_residual in itertools.product(
            self.NUM_TOKENS, (False, True)
        ):
            with self.subTest(num_tokens=num_tokens, add_residual=add_residual):
                layer, x, residual = self._layer_and_inputs(num_tokens, add_residual)
                (q, sf), r, normed_ref, r_ref = self._run_fused(
                    layer, x, residual, fp4_scale=global_scale
                )
                q_ref, sf_ref = fp4_quantize(normed_ref, global_scale)
                self.assertEqual((q.dtype, q.shape), (q_ref.dtype, q_ref.shape))
                self.assertEqual((sf.dtype, sf.shape), (sf_ref.dtype, sf_ref.shape))
                if add_residual:
                    torch.testing.assert_close(r, r_ref, atol=1e-2, rtol=1e-2)
                # The fused kernel skips the bf16 rounding of the normed value, so
                # a few codes land one step away; padded scale rows are unspecified.
                self.assertLess((q != q_ref).float().mean().item(), 0.03)
                rows = torch.arange(num_tokens).view(-1, 1)
                cols = torch.arange(kb).view(1, -1)
                idx = (
                    ((rows // 128) * (kb // 4) + cols // 4) * 512
                    + (rows % 32) * 16
                    + ((rows // 32) % 4) * 4
                    + cols % 4
                ).flatten()
                sf_diff = (sf.flatten()[idx].int() - sf_ref.flatten()[idx].int()).abs()
                self.assertLessEqual(sf_diff.max().item(), 1)
                self.assertLess((sf_diff != 0).float().mean().item(), 0.03)

    def test_unfused_without_quant_linear(self):
        layer, x, residual = self._layer_and_inputs(7, True)
        out, r, normed_ref, _ = self._run_fused(layer, x, residual)
        self.assertEqual(out.dtype, torch.bfloat16)
        torch.testing.assert_close(out, normed_ref, atol=1e-2, rtol=1.5e-2)


class TestPrequantizedFp8Linear(CustomTestCase):
    """An already-quantized fp8 input must give the same output as the bf16 input."""

    M, K, N = 1, 2048, 1024

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if torch.version.hip is not None:
            raise unittest.SkipTest("FlashInfer fusion requires CUDA")
        torch.set_default_device("cuda")

    def _operands(self):

        torch.manual_seed(0)
        x = torch.randn(self.M, self.K, dtype=torch.bfloat16)
        input_scale = torch.tensor(0.02, dtype=torch.float32)
        weight = (torch.randn(self.N, self.K) * 0.05).to(torch.float8_e4m3fn).t()
        weight_scale = torch.tensor(0.01, dtype=torch.float32)
        q, _ = static_quant_fp8(x, input_scale.reshape(1), repeat_scale=False)
        return x, q, input_scale, weight, weight_scale

    def test_flashinfer_bmm(self):

        if not flashinfer_per_tensor_fp8_supported():
            self.skipTest("flashinfer bmm_fp8 is not supported here")
        x, q, input_scale, weight, weight_scale = self._operands()
        ref = apply_fp8_linear_bmm_flashinfer(x, weight, weight_scale, input_scale)
        out = apply_fp8_linear_bmm_flashinfer(
            q, weight, weight_scale, input_scale, pre_quant_output_dtype=torch.bfloat16
        )
        self.assertEqual(out.dtype, torch.bfloat16)
        torch.testing.assert_close(out, ref, atol=0, rtol=0)

    def test_sm12x_small_m(self):
        if torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("SM12x only")

        x, q, input_scale, weight, weight_scale = self._operands()
        input_scale = input_scale.reshape(1)
        alpha = (input_scale * weight_scale).reshape(1)
        ref = try_sm120_fp8_linear(x, weight, input_scale, alpha)
        if ref is None:
            self.skipTest("no SM12x small-M kernel qualifies for this shape")
        out = try_sm120_fp8_linear(q, weight, input_scale, alpha)
        torch.testing.assert_close(out, ref, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
