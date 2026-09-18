"""MXFP4 KV-cache codec conformance.

Two implementations are locked against the independent OCP oracle
(``mxfp4_quantize_reference``):

1. the eager ``MXFP4KVQuantizeUtil`` -- OCP MX v1.0 §6.3 surface: all E2M1 codes
   and nibble order, saturating round-to-nearest-even, E8M0 scale exponent
   boundaries, all-zero / NaN / +-Inf blocks, per-head block isolation,
   partial-block padding, random parity, CPU/CUDA byte determinism;
2. the fused Triton write kernel ``quant_store_kv_mxfp4`` -- bit-exactness vs the
   eager codec and the oracle (bf16/fp16, special values, ties), the reserved
   slot-0 contract, and the support gate that falls back to the eager codec for
   fp32 / CPU / non-contiguous / shape-mismatched inputs.
"""

import unittest

import torch

from sglang.kernels.ops.quantization.mxfp4_quant import (
    mxfp4_fused_store_supported,
    quant_store_kv_mxfp4,
)
from sglang.srt.layers.quantization.kvfp4_tensor import MXFP4KVQuantizeUtil
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_HAS_CUDA = torch.cuda.is_available()
_DEVICE = "cuda"


# ---------------------------------------------------------------------------
# Independent OCP MXFP4 reference (test-only). Shares no code with the runtime
# MXFP4KVQuantizeUtil, so the bit-exactness assertions below are not circular.
# OCP MX v1.0 6.3: block-32, E2M1 saturating round-to-nearest-even, one E8M0
# scale per head block = 2^(floor(log2(amax)) - 2).
# ---------------------------------------------------------------------------
MXFP4_BLOCK_SIZE = 32
E8M0_MIN_EXP = -127
E8M0_MAX_EXP = 127
E8M0_NAN_BYTE = 0xFF

_E2M1_POSITIVE_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _round_e2m1_rne_reference(values: torch.Tensor) -> torch.Tensor:
    """Return positive E2M1 codes using round-to-nearest, ties-to-even."""
    values = values.to(torch.float32).abs().clamp(max=6.0)
    table = values.new_tensor(_E2M1_POSITIVE_VALUES)
    distance = (values.unsqueeze(-1) - table).abs()
    minimum = distance.amin(dim=-1, keepdim=True)
    tied = distance == minimum
    code_ids = torch.arange(8, dtype=torch.int64, device=values.device)
    even_tied = tied & ((code_ids & 1) == 0)
    has_even = even_tied.any(dim=-1)
    even_code = even_tied.to(torch.uint8).argmax(dim=-1).to(torch.uint8)
    first_code = tied.to(torch.uint8).argmax(dim=-1).to(torch.uint8)
    return torch.where(has_even, even_code, first_code)


def mxfp4_quantize_reference(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[tokens, heads, dim]`` to OCP MXFP4 bytes.

    Blocks are independent along each head's final dimension. The final partial
    block and an odd high nibble are padded with positive zero.
    """
    if tensor.ndim != 3:
        raise ValueError(
            f"MXFP4 expects a 3-D [tokens, heads, dim] tensor, got {tensor.shape}"
        )
    tokens, heads, logical_dim = tensor.shape
    if logical_dim <= 0:
        raise ValueError("MXFP4 logical_dim must be positive")

    num_blocks = (logical_dim + MXFP4_BLOCK_SIZE - 1) // MXFP4_BLOCK_SIZE
    padded_dim = num_blocks * MXFP4_BLOCK_SIZE
    values = tensor.to(torch.float32)
    if padded_dim != logical_dim:
        values = torch.nn.functional.pad(values, (0, padded_dim - logical_dim))
    blocks = values.reshape(tokens, heads, num_blocks, MXFP4_BLOCK_SIZE)

    nan_blocks = torch.isnan(blocks).any(dim=-1)
    inf_blocks = torch.isinf(blocks).any(dim=-1) & ~nan_blocks
    finite_abs = torch.nan_to_num(blocks.abs(), nan=0.0, posinf=0.0, neginf=0.0)
    amax = finite_abs.amax(dim=-1)

    scale_exp = torch.floor(torch.log2(amax)) - 2.0
    scale_exp = torch.where(amax == 0, scale_exp.new_full((), E8M0_MIN_EXP), scale_exp)
    scale_exp = torch.where(inf_blocks, scale_exp.new_full((), E8M0_MAX_EXP), scale_exp)
    scale_exp = scale_exp.clamp(E8M0_MIN_EXP, E8M0_MAX_EXP)
    scale_bytes = (scale_exp.to(torch.int32) + 127).to(torch.uint8)
    scale_bytes = torch.where(
        nan_blocks,
        scale_bytes.new_full((), E8M0_NAN_BYTE),
        scale_bytes,
    )

    scale = torch.exp2(scale_exp).unsqueeze(-1)
    scaled = blocks / scale
    scaled = torch.nan_to_num(scaled, nan=0.0, posinf=6.0, neginf=-6.0)
    magnitude = _round_e2m1_rne_reference(scaled)
    codes = magnitude | (torch.signbit(scaled).to(torch.uint8) << 3)
    codes = torch.where(nan_blocks.unsqueeze(-1), torch.zeros_like(codes), codes)
    codes = codes.reshape(tokens, heads, padded_dim)[..., :logical_dim]

    if logical_dim % 2:
        codes = torch.nn.functional.pad(codes, (0, 1))
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return packed.contiguous(), scale_bytes.contiguous()


def mxfp4_dequantize_reference(
    packed: torch.Tensor,
    scale_bytes: torch.Tensor,
    *,
    logical_dim: int,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize raw packed E2M1 and E8M0 bytes to ``dtype``."""
    if packed.ndim != 3 or scale_bytes.ndim != 3:
        raise ValueError("MXFP4 packed data and scales must both be 3-D")
    if logical_dim <= 0 or packed.shape[-1] != (logical_dim + 1) // 2:
        raise ValueError(
            f"packed last dim {packed.shape[-1]} does not match logical_dim {logical_dim}"
        )
    expected_blocks = (logical_dim + MXFP4_BLOCK_SIZE - 1) // MXFP4_BLOCK_SIZE
    if (
        scale_bytes.shape[:-1] != packed.shape[:-1]
        or scale_bytes.shape[-1] != expected_blocks
    ):
        raise ValueError(
            f"scale shape {scale_bytes.shape} does not match packed shape {packed.shape} "
            f"and logical_dim {logical_dim}"
        )

    packed_bytes = packed.view(torch.uint8)
    raw_codes = torch.empty(
        *packed_bytes.shape[:-1],
        packed_bytes.shape[-1] * 2,
        dtype=torch.uint8,
        device=packed.device,
    )
    raw_codes[..., 0::2] = packed_bytes & 0x0F
    raw_codes[..., 1::2] = (packed_bytes >> 4) & 0x0F
    raw_codes = raw_codes[..., :logical_dim]

    table = packed_bytes.new_tensor(_E2M1_POSITIVE_VALUES, dtype=torch.float32)
    magnitude = table[(raw_codes & 0x07).long()]
    elements = torch.where((raw_codes & 0x08) != 0, -magnitude, magnitude)

    num_blocks = scale_bytes.shape[-1]
    padded_dim = num_blocks * MXFP4_BLOCK_SIZE
    if padded_dim != logical_dim:
        elements = torch.nn.functional.pad(elements, (0, padded_dim - logical_dim))
    elements = elements.reshape(*elements.shape[:-1], num_blocks, MXFP4_BLOCK_SIZE)

    raw_scales = scale_bytes.view(torch.uint8)
    nan_blocks = raw_scales == E8M0_NAN_BYTE
    scale_exp = raw_scales.to(torch.int16) - 127
    scales = torch.exp2(scale_exp.to(torch.float32))
    scales = torch.where(nan_blocks, scales.new_full((), float("nan")), scales)
    output = (elements * scales.unsqueeze(-1)).flatten(-2)[..., :logical_dim]
    return output.to(dtype)


class TestMXFP4CodecConformance(CustomTestCase):
    def _assert_production_matches_oracle(self, x):
        actual_data, actual_scales = MXFP4KVQuantizeUtil.batched_quantize(x)
        expected_data, expected_scales = mxfp4_quantize_reference(x)
        self.assertTrue(torch.equal(actual_data, expected_data))
        self.assertTrue(torch.equal(actual_scales, expected_scales))

        actual_dq = MXFP4KVQuantizeUtil.batched_dequantize(
            actual_data,
            actual_scales,
            logical_dim=x.shape[-1],
            dtype=torch.float32,
        )
        expected_dq = mxfp4_dequantize_reference(
            expected_data,
            expected_scales,
            logical_dim=x.shape[-1],
            dtype=torch.float32,
        )
        torch.testing.assert_close(
            actual_dq, expected_dq, rtol=0, atol=0, equal_nan=True
        )
        return actual_data, actual_scales, actual_dq

    def test_all_e2m1_codes_and_nibble_order(self):
        values = (
            torch.tensor(
                [
                    0.0,
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                    3.0,
                    4.0,
                    6.0,
                    -0.0,
                    -0.5,
                    -1.0,
                    -1.5,
                    -2.0,
                    -3.0,
                    -4.0,
                    -6.0,
                ],
                dtype=torch.float32,
            )
            .repeat(2)
            .view(1, 1, 32)
        )
        packed, scales, reconstructed = self._assert_production_matches_oracle(values)
        expected_codes = torch.arange(16, dtype=torch.uint8).repeat(2)
        expected_packed = expected_codes[0::2] | (expected_codes[1::2] << 4)
        self.assertTrue(torch.equal(packed.flatten(), expected_packed))
        self.assertEqual(scales.item(), 127)  # scale = 1.0
        torch.testing.assert_close(reconstructed, values, rtol=0, atol=0)

    def test_round_ties_to_even_and_saturation(self):
        values = torch.tensor(
            [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0],
            dtype=torch.float32,
        )
        values = torch.cat([values, -values, torch.tensor([6.0] * 16)]).view(1, 1, 32)
        packed, scales, reconstructed = self._assert_production_matches_oracle(values)
        self.assertEqual(scales.item(), 127)
        expected = torch.tensor(
            [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0],
            dtype=torch.float32,
        )
        torch.testing.assert_close(reconstructed[0, 0, :8], expected, rtol=0, atol=0)
        torch.testing.assert_close(reconstructed[0, 0, 8:16], -expected, rtol=0, atol=0)
        self.assertEqual(packed.dtype, torch.uint8)

    def test_scale_exponent_boundaries_and_zero_block(self):
        x = torch.zeros(1, 3, 32, dtype=torch.float32)
        x[0, 0, 0] = 3.9  # floor_pow2=2, scale=0.5, byte=126
        x[0, 1, 0] = 8.0  # floor_pow2=8, scale=2, byte=128
        _, scales, reconstructed = self._assert_production_matches_oracle(x)
        self.assertEqual(scales.flatten().tolist(), [126, 128, 0])
        self.assertEqual(reconstructed[0, 2].abs().sum().item(), 0.0)

    def test_nan_and_infinity_policy(self):
        x = torch.zeros(1, 2, 32, dtype=torch.float32)
        x[0, 0, 3] = float("nan")
        x[0, 1, 4] = float("inf")
        x[0, 1, 5] = -float("inf")
        _, scales, reconstructed = self._assert_production_matches_oracle(x)
        self.assertEqual(scales[0, 0, 0].item(), E8M0_NAN_BYTE)
        self.assertTrue(torch.isnan(reconstructed[0, 0]).all())
        self.assertEqual(scales[0, 1, 0].item(), 254)
        self.assertTrue(torch.isinf(reconstructed[0, 1, 4]))
        self.assertTrue(torch.isinf(reconstructed[0, 1, 5]))

    def test_head_isolation_and_partial_block(self):
        x = torch.zeros(2, 2, 33, dtype=torch.float32)
        x[0, 0, 0] = 4.0
        x[0, 0, 32] = 0.5
        x[0, 1, 0] = 16.0
        packed, scales, reconstructed = self._assert_production_matches_oracle(x)
        self.assertEqual(packed.shape, (2, 2, 17))
        self.assertEqual(scales.shape, (2, 2, 2))
        self.assertEqual(scales[0, 0].tolist(), [127, 124])
        self.assertEqual(scales[0, 1].tolist(), [129, 0])
        self.assertEqual(reconstructed.shape, x.shape)

    def test_random_cpu_parity(self):
        torch.manual_seed(20260831)
        for head_dim in (1, 2, 31, 32, 33, 64, 256):
            with self.subTest(head_dim=head_dim):
                x = torch.randn(3, 4, head_dim, dtype=torch.bfloat16)
                self._assert_production_matches_oracle(x)

    @unittest.skipUnless(_HAS_CUDA, "CUDA is required")
    def test_fixed_vector_cpu_cuda_byte_parity(self):
        torch.manual_seed(20260901)
        x_cpu = torch.randn(3, 4, 65, dtype=torch.bfloat16)
        cpu_data, cpu_scales = MXFP4KVQuantizeUtil.batched_quantize(x_cpu)
        cuda_data, cuda_scales = MXFP4KVQuantizeUtil.batched_quantize(x_cpu.cuda())
        self.assertTrue(torch.equal(cpu_data, cuda_data.cpu()))
        self.assertTrue(torch.equal(cpu_scales, cuda_scales.cpu()))


def _make_buffers(slots, heads, dim, device=_DEVICE):
    packed_dim = (dim + 1) // 2
    num_blocks = (dim + 31) // 32
    data = torch.zeros((slots, heads, packed_dim), dtype=torch.uint8, device=device)
    scales = torch.zeros((slots, heads, num_blocks), dtype=torch.uint8, device=device)
    return data, scales


def _fused_write(k, v, loc):
    """Run the fused kernel into fresh pool-shaped buffers; returns
    (k_data, v_data, k_sf, v_sf) at the written slots."""
    slots = int(loc.max().item()) + 2
    heads, dim = k.shape[1], k.shape[2]
    kd, ks = _make_buffers(slots, heads, dim, device=k.device)
    vd, vs = _make_buffers(slots, heads, dim, device=v.device)
    quant_store_kv_mxfp4(k, v, loc, kd, vd, ks, vs)
    return kd, vd, ks, vs


@unittest.skipUnless(_HAS_CUDA, "CUDA is required")
class TestMxfp4FusedStore(CustomTestCase):
    def _check_bit_exact(self, k, v, loc, context):
        kd, vd, ks, vs = _fused_write(k, v, loc)
        valid = loc != 0
        locs_valid = loc[valid]

        eager_k, eager_ks = MXFP4KVQuantizeUtil.batched_quantize(k)
        eager_v, eager_vs = MXFP4KVQuantizeUtil.batched_quantize(v)
        oracle_k, oracle_ks = mxfp4_quantize_reference(k)
        oracle_v, oracle_vs = mxfp4_quantize_reference(v)

        # The eager codec and the oracle must agree first; otherwise this test
        # would chase a moving target.
        self.assertTrue(
            torch.equal(eager_k, oracle_k) and torch.equal(eager_ks, oracle_ks),
            f"[{context}] eager codec drifted from the OCP oracle (K scales)",
        )
        self.assertTrue(
            torch.equal(eager_v, oracle_v) and torch.equal(eager_vs, oracle_vs),
            f"[{context}] eager codec drifted from the OCP oracle (V scales)",
        )

        self.assertTrue(
            torch.equal(kd[locs_valid], eager_k[valid]),
            f"[{context}] fused K packed bytes differ from eager codec",
        )
        self.assertTrue(
            torch.equal(ks[locs_valid], eager_ks[valid]),
            f"[{context}] fused K E8M0 bytes differ from eager codec",
        )
        self.assertTrue(
            torch.equal(vd[locs_valid], eager_v[valid]),
            f"[{context}] fused V packed bytes differ from eager codec",
        )
        self.assertTrue(
            torch.equal(vs[locs_valid], eager_vs[valid]),
            f"[{context}] fused V E8M0 bytes differ from eager codec",
        )

    def test_random_bf16_dims(self):
        g = torch.Generator(device=_DEVICE).manual_seed(20260903)
        for dim in (32, 48, 64, 128, 256):
            for heads in (1, 4):
                with self.subTest(dim=dim, heads=heads):
                    k = (
                        torch.randn((7, heads, dim), generator=g, device=_DEVICE) * 2
                    ).to(torch.bfloat16)
                    v = (
                        torch.randn((7, heads, dim), generator=g, device=_DEVICE) * 2
                    ).to(torch.bfloat16)
                    loc = torch.arange(1, 8, dtype=torch.int64, device=_DEVICE)
                    self._check_bit_exact(k, v, loc, f"random_bf16_d{dim}_h{heads}")

    def test_random_fp16(self):
        g = torch.Generator(device=_DEVICE).manual_seed(5)
        k = torch.randn((5, 2, 128), generator=g, device=_DEVICE).to(torch.float16)
        v = torch.randn((5, 2, 128), generator=g, device=_DEVICE).to(torch.float16)
        loc = torch.arange(1, 6, dtype=torch.int64, device=_DEVICE)
        self._check_bit_exact(k, v, loc, "random_fp16")

    def test_special_values(self):
        # NaN blocks, +-Inf blocks, signed zeros, bf16 subnormals and
        # extremes; one (heads=4, dim=64) row per scenario so each 32-block
        # isolates a case.
        row = torch.zeros((4, 64), dtype=torch.float32)
        row[0] = float("nan")  # NaN block -> scale 0xFF, zero codes
        row[1, :16] = float("inf")
        row[1, 16:] = -float("inf")  # mixed +-Inf block -> scale 2^127
        row[2, ::2] = 0.0
        row[2, 1::2] = -0.0  # signed zeros -> codes 0x0 / 0x8
        row[3, 0] = 3.3895e38  # bf16 max
        row[3, 1] = -3.3895e38
        row[3, 2] = 1e-40  # fp32 subnormal -> rounds to 0 in bf16
        row[3, 3] = 2.0**-130  # bf16 subnormal
        k = (
            row.unsqueeze(0)
            .to(torch.bfloat16)
            .expand(3, 4, 64)
            .contiguous()
            .to(_DEVICE)
        )
        v = (-k).clone()
        loc = torch.arange(1, 4, dtype=torch.int64, device=_DEVICE)
        self._check_bit_exact(k, v, loc, "special_values")

    def test_all_zero_block(self):
        k = torch.zeros((2, 2, 64), dtype=torch.bfloat16, device=_DEVICE)
        v = torch.zeros((2, 2, 64), dtype=torch.bfloat16, device=_DEVICE)
        loc = torch.arange(1, 3, dtype=torch.int64, device=_DEVICE)
        self._check_bit_exact(k, v, loc, "all_zero")
        # amax == 0 -> scale byte 0 (2^-127) with zero codes.
        _, _, ks, _ = _fused_write(k, v, loc)
        self.assertTrue((ks[loc] == 0).all())

    def test_tie_midpoints(self):
        # Exact RNE midpoints after scaling. amax = 6.0 pins the block scale
        # to 2^(floor(log2(6))-2) = 1, so the bf16-exact tie values below hit
        # the distance-table midpoints (.25/.75/1.25/1.75/2.5/3.5/5) bit for
        # bit; ties-to-even must pick codes 0/2/2/4/4/6/6.
        base = torch.tensor(
            [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0], dtype=torch.float32
        )
        row = base.repeat(8)[:64]  # dim 64 = two 32-blocks, amax 6.0
        k = row.unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
        v = (-row).unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
        loc = torch.tensor([1], dtype=torch.int64, device=_DEVICE)
        self._check_bit_exact(
            k.contiguous().to(_DEVICE), v.contiguous().to(_DEVICE), loc, "ties"
        )

    def test_slot0_untouched(self):
        g = torch.Generator(device=_DEVICE).manual_seed(9)
        k = torch.randn((4, 2, 64), generator=g, device=_DEVICE).to(torch.bfloat16)
        v = torch.randn((4, 2, 64), generator=g, device=_DEVICE).to(torch.bfloat16)
        # Token 1 targets the reserved slot 0; its bytes must not change.
        loc = torch.tensor([1, 0, 2, 3], dtype=torch.int64, device=_DEVICE)

        kd, ks = _make_buffers(8, 2, 64)
        vd, vs = _make_buffers(8, 2, 64)
        sentinel = torch.tensor(0xA5, dtype=torch.uint8, device=_DEVICE)
        kd[0].fill_(sentinel)
        ks[0].fill_(sentinel)
        vd[0].fill_(sentinel)
        vs[0].fill_(sentinel)
        quant_store_kv_mxfp4(k, v, loc, kd, vd, ks, vs)

        self.assertTrue((kd[0] == sentinel).all(), "slot 0 K data was written")
        self.assertTrue((ks[0] == sentinel).all(), "slot 0 K scale was written")
        self.assertTrue((vd[0] == sentinel).all(), "slot 0 V data was written")
        self.assertTrue((vs[0] == sentinel).all(), "slot 0 V scale was written")

        # The non-reserved rows must match the eager codec byte-for-byte:
        # token 0 -> slot 1, token 1 -> slot 0 (skipped), token 2 -> slot 2,
        # token 3 -> slot 3.
        eager_k, eager_ks = MXFP4KVQuantizeUtil.batched_quantize(k)
        for token, slot in ((0, 1), (2, 2), (3, 3)):
            self.assertTrue(torch.equal(kd[slot], eager_k[token]))
            self.assertTrue(torch.equal(ks[slot], eager_ks[token]))

    def test_gate(self):
        loc = torch.arange(1, 4, dtype=torch.int64, device=_DEVICE)
        with self.subTest(supported_bf16=True):
            k = torch.randn((3, 2, 64), device=_DEVICE).to(torch.bfloat16)
            self.assertTrue(mxfp4_fused_store_supported(k, k.clone(), loc))
        with self.subTest(rejects_fp32=True):
            k = torch.randn((3, 2, 64), device=_DEVICE)
            self.assertFalse(mxfp4_fused_store_supported(k, k.clone(), loc))
        with self.subTest(rejects_non_contiguous_last_dim=True):
            base = torch.randn((3, 2, 128), device=_DEVICE).to(torch.bfloat16)
            k = base[..., ::2]  # last-dim stride 2
            self.assertFalse(mxfp4_fused_store_supported(k, k.clone(), loc))
        with self.subTest(rejects_cpu=True):
            k = torch.randn((3, 2, 64)).to(torch.bfloat16)
            cpu_loc = torch.arange(1, 4, dtype=torch.int64)
            self.assertFalse(mxfp4_fused_store_supported(k, k.clone(), cpu_loc))
        with self.subTest(rejects_shape_mismatch=True):
            k = torch.randn((3, 2, 64), device=_DEVICE).to(torch.bfloat16)
            v = torch.randn((3, 2, 32), device=_DEVICE).to(torch.bfloat16)
            self.assertFalse(mxfp4_fused_store_supported(k, v, loc))


if __name__ == "__main__":
    unittest.main()
