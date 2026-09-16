import unittest

import torch

from sglang.srt.layers.attention.dsv4.torch_quant import (
    fake_quant_compressed_kv,
    fake_quant_fp4,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def _rope_fq4(x, freqs, rope_dim, *, compressed_kv=False):
    """RoPE plus fake FP4 quantization, fused for CUDA BF16 inputs."""
    if x.is_cuda and torch.version.cuda is not None and x.dtype == torch.bfloat16:
        from sglang.kernels.ops.attention.dsv4.rope_fake_quant_fp4 import (
            rope_tail_fake_quant_fp4,
        )

        return rope_tail_fake_quant_fp4(x, freqs, rope_dim, compressed_kv=compressed_kv)
    quant = fake_quant_compressed_kv if compressed_kv else fake_quant_fp4
    return quant(rope_tail(x, freqs, rope_dim))


def rope_tail(
    x: torch.Tensor, freqs: torch.Tensor, rope_dim: int, inverse: bool = False
) -> torch.Tensor:
    """Rotate the last rope_dim features of x [T, ..., D] with complex freqs [T, rope_dim // 2]."""
    head, tail = x[..., :-rope_dim], x[..., -rope_dim:]
    tc = torch.view_as_complex(tail.float().unflatten(-1, (-1, 2)).contiguous())
    f = freqs.conj() if inverse else freqs
    f = f.view(x.shape[0], *([1] * (x.ndim - 2)), rope_dim // 2)
    rotated = torch.view_as_real(tc * f).flatten(-2).to(x.dtype)
    return torch.cat([head, rotated], dim=-1)


class TestCompressedKVQuant(CustomTestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_triton_matches_torch_for_both_quantization_rules(self):
        from sglang.kernels.ops.attention.dsv4.rope_fake_quant_fp4 import (
            rope_tail_fake_quant_fp4,
        )

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
