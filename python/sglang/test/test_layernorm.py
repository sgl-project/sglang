import itertools
import unittest

import torch

from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
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


class TestRMSNormQuant(CustomTestCase):
    DTYPES = [torch.bfloat16]
    NUM_TOKENS = [7, 83, 4096]
    HIDDEN_SIZES = [256, 512, 2048, 7168, 8192]
    ADD_RESIDUAL = [False, True]
    COLUMN_MAJOR_SCALES = [False, True]
    SCALE_TMA_ALIGNED = [False, True]
    GROUP_SIZE = [32, 64, 128, 256]
    SCALE_UE8M0 = [False, True]
    SEEDS = [42]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_rms_norm_test(
        self,
        num_tokens,
        hidden_size,
        add_residual,
        dtype,
        column_major_scales,
        scale_tma_aligned,
        group_size,
        scale_ue8m0,
        seed,
    ):
        # Invalid configuration, skip
        if scale_ue8m0 and not (
            scale_tma_aligned and column_major_scales and group_size == 128
        ):
            return

        torch.manual_seed(seed)

        layer = RMSNorm(
            hidden_size,
            output_quant=True,
            column_major_scales=column_major_scales,
            scale_tma_aligned=scale_tma_aligned,
            group_size=group_size,
            scale_ue8m0=scale_ue8m0,
        ).to(dtype=dtype)
        layer.weight.data.normal_(mean=1.0, std=0.1)
        scale = 1 / (2 * hidden_size)
        x = torch.randn(num_tokens, hidden_size, dtype=dtype) * scale
        residual = torch.randn_like(x) * scale if add_residual else None

        with torch.inference_mode():
            ref_out = layer.forward_native(x, residual)
            out = layer(x, residual)
        quant_atol = 5e-1
        quant_rtol = 1e-3

        if add_residual:
            (q, s, _), res = out
            ref_rms, ref_res = ref_out
            ref_q, ref_s = sglang_per_token_group_quant_fp8(
                ref_rms,
                group_size,
                column_major_scales=column_major_scales,
                scale_tma_aligned=scale_tma_aligned,
                scale_ue8m0=scale_ue8m0,
            )
            if scale_ue8m0:
                s = s.flatten().contiguous().view(dtype=torch.uint8)
                ref_s = ref_s.flatten().contiguous().view(dtype=torch.uint8)
                # ue8m0 allocates bigger buffers so we clean non zero elements
                # so that shapes fit
                s = 2 ** (s[s.nonzero()].to(dtype) - 127)
                ref_s = 2 ** (ref_s[ref_s.nonzero()].to(dtype) - 127)

            s = s.repeat_interleave(group_size).reshape(q.shape)
            ref_s = ref_s.repeat_interleave(group_size).reshape(q.shape)
            dequant = q.to(dtype) * s
            ref_dequant = ref_q.to(dtype) * ref_s

            self.assertTrue(
                torch.allclose(dequant, ref_dequant, atol=quant_atol, rtol=quant_rtol)
            )
            self.assertTrue(torch.allclose(res, ref_res, atol=1e-2, rtol=1e-2))
        else:
            q, s, _ = out
            ref_rms = ref_out
            ref_q, ref_s = sglang_per_token_group_quant_fp8(
                ref_rms,
                group_size,
                column_major_scales=column_major_scales,
                scale_tma_aligned=scale_tma_aligned,
                scale_ue8m0=scale_ue8m0,
            )

            if scale_ue8m0:
                s = s.flatten().contiguous().view(dtype=torch.uint8)
                ref_s = ref_s.flatten().contiguous().view(dtype=torch.uint8)

                # ue8m0 allocates bigger buffers so we clean non zero elements
                # so that shapes fit
                s = 2 ** (s[s.nonzero()].to(dtype) - 127)
                ref_s = 2 ** (ref_s[ref_s.nonzero()].to(dtype) - 127)

            s = s.repeat_interleave(group_size).reshape(q.shape)
            ref_s = ref_s.repeat_interleave(group_size).reshape(q.shape)
            dequant = q.to(dtype) * s
            ref_dequant = ref_q.to(dtype) * ref_s

            self.assertTrue(
                torch.allclose(dequant, ref_dequant, atol=quant_atol, rtol=quant_rtol)
            )

    def test_rms_norm(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.HIDDEN_SIZES,
            self.ADD_RESIDUAL,
            self.DTYPES,
            self.COLUMN_MAJOR_SCALES,
            self.SCALE_TMA_ALIGNED,
            self.GROUP_SIZE,
            self.SCALE_UE8M0,
            self.SEEDS,
        ):
            with self.subTest(
                num_tokens=params[0],
                hidden_size=params[1],
                add_residual=params[2],
                dtype=params[3],
                column_major_scales=params[4],
                scale_tma_aligned=params[5],
                group_size=params[6],
                scale_ue8m0=params[7],
                seed=params[8],
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
