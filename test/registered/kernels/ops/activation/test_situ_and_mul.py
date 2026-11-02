import unittest

import torch

from sglang.kernels.ops.activation import (
    situ_and_mul,
    situ_and_mul_masked,
    situ_and_mul_masked_post_quant_mxfp4,
    situ_and_mul_post_quant_mxfp4,
)
from sglang.srt.utils import get_device_sm, is_sail
from sglang.test.ci.ci_register import register_cuda_ci, register_ppu_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=70, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
# backend-specific: SAIL FP4 instructions and the DeepEP masked SiTU ABI.
register_ppu_ci(est_time=70, suite="stage-a-test-1-gpu-ppu")

_BETA = 4.0
_LINEAR_BETA = 25.0


def _situ_reference(gate_up, linear_beta=_LINEAR_BETA):
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.float()
    up = up.float()
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return _BETA * torch.tanh(gate / _BETA) * torch.sigmoid(gate) * up


class TestSituAndMul(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if not is_sail() and get_device_sm() < 100:
            raise unittest.SkipTest("Kimi K3 compute kernels require SM100a+")

    def test_situ_and_mul(self):
        generator = torch.Generator(device="cuda").manual_seed(2)
        hidden_size = 1024
        storage = torch.randn(
            (7, 2 * hidden_size + 16),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        gate_up = storage[:, : 2 * hidden_size]
        output = torch.empty(
            (gate_up.shape[0], hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
        )

        returned = situ_and_mul(gate_up, output, beta=_BETA, linear_beta=_LINEAR_BETA)

        self.assertIs(returned, output)
        torch.testing.assert_close(
            returned.float(),
            _situ_reference(gate_up).to(torch.bfloat16).float(),
            rtol=2e-2,
            atol=4e-2,
        )

    def test_masked_situ_preserves_padding_and_ignores_expected_m_limit(self):
        """expected_m=2 must not truncate the 128 valid rows, and empty experts must
        not overwrite the padding."""
        generator = torch.Generator(device="cuda").manual_seed(3)
        gate_up = torch.randn(
            (8, 128, 6144), generator=generator, device="cuda", dtype=torch.bfloat16
        )
        counts = [0, 128, 64, 128, 128, 128, 128, 128]
        masked_m = torch.tensor(counts, device="cuda", dtype=torch.int32)
        for linear_beta in (_LINEAR_BETA, None):
            with self.subTest(linear_beta=linear_beta):
                output = torch.full(
                    (8, 128, 3072), 42, device="cuda", dtype=torch.bfloat16
                )
                returned = situ_and_mul_masked(
                    gate_up, output, masked_m, _BETA, linear_beta, topk=16, expected_m=2
                )
                self.assertIs(returned, output)
                reference = _situ_reference(gate_up, linear_beta).to(torch.bfloat16)
                for expert, count in enumerate(counts):
                    torch.testing.assert_close(
                        output[expert, :count],
                        reference[expert, :count],
                        rtol=2e-2,
                        atol=4e-2,
                    )
                    torch.testing.assert_close(
                        output[expert, count:],
                        torch.full_like(output[expert, count:], 42),
                        rtol=0,
                        atol=0,
                    )

    def test_masked_mxfp4_matches_compact_and_independent_dequantization(self):
        """Check BF16 rounding, E8M0 scale bytes, E2M1 packing and the MN-major
        physical layout."""
        generator = torch.Generator(device="cuda").manual_seed(4)
        gate_up = torch.randn(
            (8, 128, 6144), generator=generator, device="cuda", dtype=torch.bfloat16
        )
        counts = [0, 128, 64, 128, 128, 128, 128, 128]
        masked_m = torch.tensor(counts, device="cuda", dtype=torch.int32)
        for linear_beta in (_LINEAR_BETA, None):
            with self.subTest(linear_beta=linear_beta):
                output = torch.full(
                    (8, 128, 1536), 0x7F, device="cuda", dtype=torch.uint8
                )
                scales = torch.full(
                    (8, 48, 128), 0x7F7F, device="cuda", dtype=torch.uint16
                )
                situ_and_mul_masked_post_quant_mxfp4(
                    gate_up,
                    output,
                    scales,
                    masked_m,
                    _BETA,
                    linear_beta,
                    topk=16,
                    expected_m=2,
                )
                compact = torch.cat([gate_up[e, :m] for e, m in enumerate(counts)])
                compact_q = torch.empty(
                    (sum(counts), 1536), device="cuda", dtype=torch.uint8
                )
                compact_s = torch.empty(
                    (48, sum(counts)), device="cuda", dtype=torch.uint16
                )
                situ_and_mul_post_quant_mxfp4(
                    compact, compact_q, compact_s, _BETA, linear_beta
                )
                offset = 0
                for expert, count in enumerate(counts):
                    torch.testing.assert_close(
                        output[expert, :count],
                        compact_q[offset : offset + count],
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        scales[expert, :, :count],
                        compact_s[:, offset : offset + count],
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        output[expert, count:],
                        torch.full_like(output[expert, count:], 0x7F),
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        scales[expert, :, count:],
                        torch.full_like(scales[expert, :, count:], 0x7F7F),
                        rtol=0,
                        atol=0,
                    )
                    offset += count
                # Decode independently, without calling vendor quant/dequant or
                # another FP4 kernel.
                raw_scales = compact_s.t().contiguous().view(torch.uint8)
                scale_values = torch.exp2(raw_scales.float() - 127)
                # Validate SiTU on its own first, then validate the quantization
                # against the same-precision BF16 producer output; small
                # fast-math vs torch differences must not turn power-of-two
                # boundaries into false scale mismatches.
                produced = torch.empty(
                    (sum(counts), 3072), device="cuda", dtype=torch.bfloat16
                )
                situ_and_mul(compact, produced, _BETA, linear_beta)
                torch.testing.assert_close(
                    produced,
                    _situ_reference(compact, linear_beta).to(torch.bfloat16),
                    rtol=2e-2,
                    atol=4e-2,
                )
                reference = produced.float()
                max_values = (
                    reference.reshape(-1, 96, 32).abs().amax(-1).clamp_min(1e-10)
                )
                expected_scales = torch.exp2(torch.ceil(torch.log2(max_values / 6)))
                torch.testing.assert_close(
                    scale_values, expected_scales, rtol=0, atol=0
                )
                codes = (
                    torch.stack((compact_q & 15, compact_q >> 4), dim=-1)
                    .flatten(-2)
                    .long()
                )
                table = torch.tensor(
                    [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
                    device="cuda",
                )
                dequantized = table[codes].reshape(-1, 96, 32) * scale_values.unsqueeze(
                    -1
                )
                normalized_error = (
                    dequantized - reference.reshape(-1, 96, 32)
                ).abs() / scale_values.unsqueeze(-1)
                # The largest adjacent E2M1 gap is 2, so the nearest-rounding
                # error stays within one scale.
                self.assertLessEqual(normalized_error.max().item(), 1.05)

    def test_masked_empty_and_alignment_contract(self):
        for experts, tokens in ((0, 3), (2, 0)):
            x = torch.empty((experts, tokens, 512), device="cuda", dtype=torch.bfloat16)
            out = torch.empty(
                (experts, tokens, 256), device="cuda", dtype=torch.bfloat16
            )
            counts = torch.zeros(experts, device="cuda", dtype=torch.int32)
            self.assertIs(situ_and_mul_masked(x, out, counts, _BETA, None), out)
            q = torch.empty((experts, tokens, 128), device="cuda", dtype=torch.uint8)
            scales = torch.empty(
                (experts, 4, tokens), device="cuda", dtype=torch.uint16
            )
            situ_and_mul_masked_post_quant_mxfp4(x, q, scales, counts, _BETA, None)
        x = torch.empty((1, 2, 512), device="cuda", dtype=torch.bfloat16)
        counts = torch.ones(1, device="cuda", dtype=torch.int32)
        # The contiguous shape is valid, but the vector start address is offset by
        # 2B; it must be rejected before launch.
        out = torch.empty(513, device="cuda", dtype=torch.bfloat16)[1:].view(1, 2, 256)
        with self.assertRaisesRegex(RuntimeError, "align"):
            situ_and_mul_masked(x, out, counts, _BETA, None)


if __name__ == "__main__":
    unittest.main()
