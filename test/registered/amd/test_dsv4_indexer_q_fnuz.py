"""The DeepSeek-V4 indexer q must reach gfx94x's FNUZ consumers as E4M3FNUZ."""

import unittest

import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")


def _on_fnuz_gpu() -> bool:
    if not (is_hip() and torch.cuda.is_available()):
        return False
    from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz

    return is_fp8_fnuz()


@unittest.skipUnless(_on_fnuz_gpu(), "requires a gfx94x (E4M3FNUZ) GPU")
class TestDsv4IndexerQFnuz(unittest.TestCase):
    """The AOT kernel writes E4M3FN bytes, while the rest of the gfx94x indexer reads
    E4M3FNUZ: there the FN negative zero (0x80) is NaN and every other byte is worth half.
    """

    def test_every_finite_fn_byte_keeps_its_value(self):
        from sglang.kernels.ops.attention.dsv4.elementwise import _as_indexer_fp8

        codes = [b for b in range(256) if b not in (0x7F, 0xFF)]  # the FN NaNs
        q_fn = (
            torch.tensor(codes, dtype=torch.uint8, device="cuda")
            .view(torch.float8_e4m3fn)
            .reshape(2, 1, -1)
        )
        weights = torch.rand(2, 1, 1, device="cuda") + 0.5
        expected = q_fn.float() * weights

        q, w = _as_indexer_fp8(q_fn.clone(), weights.clone())

        self.assertEqual(q.dtype, torch.float8_e4m3fnuz)
        self.assertFalse(torch.isnan(q.float()).any(), "FN -0 must not become NaN")
        torch.testing.assert_close(q.float() * w, expected, rtol=0, atol=0)

    def test_the_fused_quant_returns_fnuz_q_worth_what_the_kernel_wrote(self):
        import sgl_kernel  # noqa: F401  (registers torch.ops.sgl_kernel)

        from sglang.kernels.ops.attention.dsv4.elementwise import (
            fused_q_indexer_rope_hadamard_quant,
        )

        torch.manual_seed(0)
        tokens, heads, head_dim, rope_dim, max_pos = 256, 64, 128, 64, 512
        q_input = torch.randn(
            tokens, heads, head_dim, dtype=torch.bfloat16, device="cuda"
        )
        weight = torch.randn(tokens, heads, dtype=torch.bfloat16, device="cuda")
        freqs_cis = torch.polar(
            torch.ones(max_pos, rope_dim // 2),
            torch.rand(max_pos, rope_dim // 2) * 6.28,
        ).cuda()
        positions = torch.randint(
            0, max_pos, (tokens,), dtype=torch.int32, device="cuda"
        )

        # The kernel's own output, written into an E4M3FN buffer and left as it is.
        q_fn = torch.empty(q_input.shape, dtype=torch.float8_e4m3fn, device="cuda")
        w_fn = torch.empty(tokens, heads, 1, dtype=torch.float32, device="cuda")
        torch.ops.sgl_kernel.dsv4_fused_q_indexer_rope_hadamard_quant(
            q_input,
            q_fn,
            weight,
            w_fn,
            0.5,
            torch.view_as_real(freqs_cis).flatten(-2),
            positions,
        )

        q, w = fused_q_indexer_rope_hadamard_quant(
            q_input, weight, 0.5, freqs_cis, positions
        )

        self.assertEqual(q.dtype, torch.float8_e4m3fnuz)
        self.assertFalse(torch.isnan(q.float()).any())
        torch.testing.assert_close(q.float() * w, q_fn.float() * w_fn, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
