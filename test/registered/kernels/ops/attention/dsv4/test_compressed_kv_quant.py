import unittest

import torch

from sglang.srt.layers.attention.dsv4.torch_quant import (
    fake_quant_compressed_kv,
    fake_quant_fp4,
)
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")


class TestCompressedKVQuant(CustomTestCase):
    @unittest.skipUnless(
        torch.cuda.is_available()
        and (torch.version.cuda is not None or is_gfx95_supported()),
        "the C1 decode kernel serves CUDA and gfx95 ROCm",
    )
    def test_cuda_compressor_scale_boundaries(self):
        from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache
        from sglang.kernels.ops.attention.dsv4.c1 import c1_decode_norm_rope_store

        maxima = torch.tensor(
            [0, 2**-12, 6 * 2**-9, 6 * 1.0625, 6 * 1.1875, 6 * 448, 1e6, 8.25],
            device="cuda",
            dtype=torch.bfloat16,
        )
        weight = maxima.repeat_interleave(16).repeat(4)
        weight[112:116] = torch.tensor(
            [8.25, 4.125, 3.4375, -4.8125], device="cuda", dtype=torch.bfloat16
        )
        quantized = (
            torch.tensor(
                [0, 0, 6 * 2**-9, 6, 7.5, 2688, 2688, 8.25],
                device="cuda",
                dtype=torch.bfloat16,
            )
            .repeat_interleave(16)
            .repeat(4)
        )
        # Scale 1.375: ties at 2.5 and -3.5 round to 2 and -4.
        quantized[112:116] = torch.tensor(
            [8.25, 4.125, 2.75, -5.5], device="cuda", dtype=torch.bfloat16
        )
        x = torch.ones(1, 512, device="cuda", dtype=torch.bfloat16)
        freqs = torch.view_as_real(
            torch.ones(1, 32, device="cuda", dtype=torch.complex64)
        )
        positions = torch.zeros(1, device="cuda", dtype=torch.int64)
        slots = torch.ones_like(positions)
        page_size = 128
        page_bytes = -(-page_size * 584 // 576) * 576
        cache = torch.zeros(1, page_bytes, device="cuda", dtype=torch.uint8)
        expected = torch.zeros_like(cache)
        latent = c1_decode_norm_rope_store(
            x,
            weight,
            positions,
            slots,
            0.0,
            freqs.flatten(-2),
            cache,
            page_size=page_size,
        )
        self.assertTrue(torch.equal(latent, weight.unsqueeze(0)))
        fused_store_cache(
            quantized.unsqueeze(0),
            expected,
            slots,
            page_size=page_size,
            type="flashmla",
        )
        self.assertTrue(torch.equal(cache, expected))

    @unittest.skipUnless(torch.cuda.is_available(), "requires a GPU")
    def test_triton_matches_torch_for_both_quantization_rules(self):
        from sglang.kernels.ops.attention.dsv4.rope_fake_quant_fp4 import (
            rope_tail_fake_quant_fp4,
        )
        from sglang.srt.layers.attention.dsv4.dsv41_sparse import _rope_fq4, rope_tail

        generator = torch.Generator(device="cuda").manual_seed(17)
        for rows in (0, 1, 33, 129):
            x = torch.randn(
                rows, 512, generator=generator, device="cuda", dtype=torch.bfloat16
            )
            angles = torch.randn(rows, 32, generator=generator, device="cuda")
            freqs = torch.polar(torch.ones_like(angles), angles)
            for compressed_kv in (False, True):
                with self.subTest(rows=rows, compressed_kv=compressed_kv):
                    quant = (
                        fake_quant_compressed_kv if compressed_kv else fake_quant_fp4
                    )
                    expected = quant(rope_tail(x, freqs, 64))
                    actual = _rope_fq4(x, freqs, 64, compressed_kv=compressed_kv)
                    self.assertTrue(torch.equal(actual, expected))

        # Identity RoPE isolates quantization boundaries from trigonometric rounding.
        maxima = torch.tensor(
            [0, 2**-12, 6 * 2**-9, 6 * 1.0625, 6 * 1.1875, 6 * 448, 1e6],
            device="cuda",
            dtype=torch.bfloat16,
        )
        x = maxima[:, None].expand(-1, 512).contiguous()
        freqs = torch.ones(x.shape[0], 32, device="cuda", dtype=torch.complex64)
        actual = rope_tail_fake_quant_fp4(x, freqs, 64, compressed_kv=True)
        expected = torch.tensor(
            [0, 0, 6 * 2**-9, 6, 7.5, 2688, 2688],
            device="cuda",
            dtype=torch.bfloat16,
        )[:, None].expand_as(x)
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
