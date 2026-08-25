import unittest

import torch

from sglang.srt.layers.quantization.fp8_utils import quantize_block_fp8_weight_to_mxfp4
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

_E2M1_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _dequant_mxfp4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Reference dequant of the MXFP4 layout the trtllm-gen kernel (and the DSV4
    checkpoint's routed experts) consume: low nibble = even column, sign in code
    bit 3, one e8m0 scale per 32 in-row elements."""
    as_u8 = packed.view(torch.uint8)
    codes_lo = (as_u8 & 0x0F).long()
    codes_hi = (as_u8 >> 4).long()

    def decode(codes):
        magnitudes = _E2M1_LUT[codes & 0x7]  # bits 2:0 index the magnitude table
        return torch.where(codes >= 8, -magnitudes, magnitudes)

    vals = torch.stack([decode(codes_lo), decode(codes_hi)], dim=-1)
    vals = vals.reshape(packed.shape[0], packed.shape[1] * 2)
    exponents = scales.view(torch.uint8).float() - 127.0
    return vals * torch.pow(2.0, exponents).repeat_interleave(32, dim=-1)


class TestQuantizeBlockFp8WeightToMxfp4(CustomTestCase):
    """Pins the wire format of the load-time FP8→MXFP4 shared-expert requant
    (FusedMoE._maybe_load_fp8_shared_expert_as_fp4): the produced bytes land in
    the same expert tensors as the checkpoint's MXFP4-packed routed experts, so
    nibble order / sign bit / e8m0 bias are a contract with the trtllm-gen
    kernel, not an implementation detail."""

    def _requant(self, weight_bf16, block=(128, 128)):
        rows, cols = weight_bf16.shape
        scale_rows = (rows + block[0] - 1) // block[0]
        scale_cols = (cols + block[1] - 1) // block[1]
        # Identity block scale: fp8 payload == dequantized value.
        fp8_scale = torch.ones(scale_rows, scale_cols, dtype=torch.float32)
        fp8_weight = weight_bf16.to(torch.float8_e4m3fn)
        return quantize_block_fp8_weight_to_mxfp4(fp8_weight, fp8_scale, list(block))

    def test_packing_layout_contract(self):
        w = torch.zeros(1, 32, dtype=torch.bfloat16)
        w[0, 0] = 0.5  # code 1
        w[0, 1] = -3.0  # magnitude idx 5, sign bit -> code 13
        w[0, 2] = 6.0  # code 7
        w[0, 3] = 1.5  # code 3
        packed, scales = self._requant(w)

        self.assertEqual(packed.dtype, torch.int8)
        self.assertEqual(packed.shape, (1, 16))
        self.assertEqual(scales.dtype, torch.float8_e8m0fnu)
        self.assertEqual(scales.shape, (1, 1))
        # group amax 6.0 -> scale 2**0 -> biased e8m0 exponent 127
        self.assertEqual(scales.view(torch.uint8)[0, 0].item(), 127)
        as_u8 = packed.view(torch.uint8)
        # byte 0 = code(0.5) | code(-3.0) << 4 ; low nibble is the even column
        self.assertEqual(as_u8[0, 0].item(), 0x01 | (0x0D << 4))
        # byte 1 = code(6.0) | code(1.5) << 4
        self.assertEqual(as_u8[0, 1].item(), 0x07 | (0x03 << 4))
        # Zero padding is not asserted byte-exactly: the quantizer encodes 0.0
        # as -0.0 (code 8), which the kernel decodes back to zero. Check the
        # padding dequantizes to zero without pinning its sign bit.
        deq = _dequant_mxfp4(packed, scales)
        self.assertTrue((deq[0, 4:] == 0).all())

    def test_roundtrip_error_is_mxfp4_sized(self):
        torch.manual_seed(0)
        w = (torch.randn(64, 128) * 0.1).to(torch.bfloat16)
        packed, scales = self._requant(w)
        deq = _dequant_mxfp4(packed, scales)
        # The fp8 cast itself is lossy; compare against what was actually quantized.
        ref = w.to(torch.float8_e4m3fn).float()
        rel_err = (deq - ref).norm() / ref.norm()
        self.assertLess(rel_err.item(), 0.15)

    def test_exactly_representable_values_roundtrip(self):
        w = torch.zeros(1, 64, dtype=torch.bfloat16)
        w[0, :32] = 4.0  # amax 4 -> quantizes exactly
        w[0, 32:] = 48.0  # amax 48 -> 6 * 2**3, exact
        packed, scales = self._requant(w)
        deq = _dequant_mxfp4(packed, scales)
        torch.testing.assert_close(deq, w.float())

    def test_block_scale_is_applied(self):
        torch.manual_seed(1)
        w = (torch.randn(16, 64) * 0.1).to(torch.bfloat16)
        fp8_weight = w.to(torch.float8_e4m3fn)
        fp8_scale = torch.full((1, 1), 2.0, dtype=torch.float32)
        packed, scales = quantize_block_fp8_weight_to_mxfp4(
            fp8_weight, fp8_scale, [128, 128]
        )
        expected = fp8_weight.float() * 2.0
        deq = _dequant_mxfp4(packed, scales)
        rel_err = (deq - expected).norm() / expected.norm()
        self.assertLess(rel_err.item(), 0.15)
        self.assertEqual(packed.shape, (16, 32))
        self.assertEqual(scales.shape, (16, 2))


if __name__ == "__main__":
    unittest.main()
