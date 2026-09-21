import unittest

import torch

from sglang.kernels.ops.attention.fla.layernorm_gated import rms_norm_gated
from sglang.kernels.ops.attention.fla.rmsnorm_gated_mxfp4 import (
    MXFP4_ROUND_EVEN,
    MXFP4_ROUND_UP,
    can_use_rmsnorm_gated_mxfp4,
    rmsnorm_gated_mxfp4_quant,
)
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=12, stage="jit-kernel-unit", runner_config="amd")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is not None,
    "ROCm GPU is required",
)
class TestRMSNormGatedMXFP4Kernel(unittest.TestCase):
    def setUp(self):
        if not is_gfx95_supported():
            self.skipTest("MXFP4 requires gfx95")

    @staticmethod
    def _valid_shuffled_scale_indices(
        num_rows: int, num_groups: int, scale_n: int
    ) -> torch.Tensor:
        indices = []
        for row in range(num_rows):
            for group in range(num_groups):
                indices.append(
                    (row // 32 * scale_n) * 32
                    + (group // 8) * 256
                    + (group % 4) * 64
                    + (row % 16) * 4
                    + ((group % 8) // 4) * 2
                    + ((row % 32) // 16)
                )
        return torch.tensor(indices, device="cuda")

    def _run_case(
        self,
        *,
        num_tokens: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        z_is_3d: bool,
        round_mode: int,
        shuffle_scales: bool,
        activation: str = "swish",
        zero_from_token: int | None = None,
        eps: float = 1e-6,
    ):
        from aiter import per_1x32_f4_quant_hip
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.utility import dtypes

        torch.manual_seed(42)
        num_head_rows = num_tokens * num_heads
        x_storage = torch.randn(
            num_head_rows,
            head_dim + 32,
            device="cuda",
            dtype=dtype,
        )
        x = x_storage[:, :head_dim]
        if zero_from_token is not None:
            x[zero_from_token * num_heads :] = 0

        if z_is_3d:
            z_storage = torch.randn(
                num_tokens,
                num_heads,
                head_dim * 2,
                device="cuda",
                dtype=dtype,
            )
            z = z_storage[..., head_dim:]
        else:
            z_storage = torch.randn(
                num_head_rows,
                head_dim + 32,
                device="cuda",
                dtype=dtype,
            )
            z = z_storage[:, :head_dim]
        weight = torch.randn(head_dim, device="cuda", dtype=dtype)
        # Force the first gated-norm element to remain IEEE -0.0. AITER's
        # public quantizers preserve its E2M1 sign nibble.
        x[0, 0] = -0.0
        weight[0] = 1.0
        if z_is_3d:
            z[0, 0, 0] = 1.0
        else:
            z[0, 0] = 1.0

        # rms_norm_gated materializes dtype-rounded output before the reference
        # quantizer, so byte equality also verifies the fused BF16/FP16 rounding
        # point and norm-before-SiLU operation order.
        dense = rms_norm_gated(
            x=x,
            weight=weight,
            bias=None,
            z=z,
            eps=eps,
            group_size=None,
            norm_before_gate=True,
            is_rms_norm=True,
            activation=activation,
        ).view(num_tokens, num_heads * head_dim)
        if round_mode == MXFP4_ROUND_UP:
            expected_q, expected_scales = per_1x32_f4_quant_hip(
                dense, shuffle=shuffle_scales
            )
        else:
            expected_q, expected_scales = dynamic_mxfp4_quant(dense)

        self.assertEqual(int(expected_q.view(torch.uint8)[0, 0].item()) & 0xF, 0x8)
        self.assertTrue(
            can_use_rmsnorm_gated_mxfp4(
                x,
                z,
                weight,
                num_heads=num_heads,
                activation=activation,
                shuffle_scales=shuffle_scales,
            )
        )
        actual_q, actual_scales = rmsnorm_gated_mxfp4_quant(
            x,
            z,
            weight,
            eps,
            num_heads=num_heads,
            activation=activation,
            round_mode=round_mode,
            shuffle_scales=shuffle_scales,
            use_native_dtypes=shuffle_scales,
        )

        self.assertTrue(
            torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
        )
        if shuffle_scales:
            self.assertEqual(actual_q.dtype, dtypes.fp4x2)
            self.assertEqual(actual_scales.dtype, dtypes.fp8_e8m0)
            valid_indices = self._valid_shuffled_scale_indices(
                num_tokens,
                num_heads * head_dim // 32,
                expected_scales.shape[1],
            )
            actual_scales = actual_scales.view(torch.uint8).flatten()[valid_indices]
            expected_scales = expected_scales.view(torch.uint8).flatten()[valid_indices]
        else:
            self.assertEqual(actual_q.dtype, torch.uint8)
            self.assertEqual(actual_scales.dtype, torch.uint8)
        self.assertTrue(
            torch.equal(
                actual_scales.view(torch.uint8),
                expected_scales.view(torch.uint8),
            )
        )

    def test_row_major_even_qwen_prefill_with_dp_padding(self):
        self._run_case(
            num_tokens=4,
            num_heads=48,
            head_dim=128,
            dtype=torch.bfloat16,
            z_is_3d=True,
            round_mode=MXFP4_ROUND_EVEN,
            shuffle_scales=False,
            zero_from_token=2,
        )

    def test_row_major_even_decode_layout(self):
        self._run_case(
            num_tokens=1,
            num_heads=48,
            head_dim=128,
            dtype=torch.bfloat16,
            z_is_3d=False,
            round_mode=MXFP4_ROUND_EVEN,
            shuffle_scales=False,
            eps=1e-5,
        )

    def test_row_major_even_non_power_of_two_head_width(self):
        self._run_case(
            num_tokens=1,
            num_heads=3,
            head_dim=96,
            dtype=torch.float16,
            z_is_3d=True,
            round_mode=MXFP4_ROUND_EVEN,
            shuffle_scales=False,
            activation="silu",
        )

    def test_shuffled_round_up_qwen_decode(self):
        self._run_case(
            num_tokens=1,
            num_heads=48,
            head_dim=128,
            dtype=torch.bfloat16,
            z_is_3d=True,
            round_mode=MXFP4_ROUND_UP,
            shuffle_scales=True,
        )

    def test_shuffled_round_up_qwen_crosses_256_row_tail(self):
        self._run_case(
            num_tokens=257,
            num_heads=48,
            head_dim=128,
            dtype=torch.bfloat16,
            z_is_3d=True,
            round_mode=MXFP4_ROUND_UP,
            shuffle_scales=True,
        )

    def test_shuffled_round_up_rejects_unaligned_total_width(self):
        num_tokens, num_heads, head_dim = 257, 3, 96
        x = torch.empty(
            (num_tokens * num_heads, head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
        z = torch.empty(
            (num_tokens, num_heads, head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
        weight = torch.empty(head_dim, device="cuda", dtype=torch.bfloat16)

        self.assertFalse(
            can_use_rmsnorm_gated_mxfp4(
                x,
                z,
                weight,
                num_heads=num_heads,
                activation="swish",
                shuffle_scales=True,
            )
        )
        with self.assertRaisesRegex(ValueError, "device, shape, or layout"):
            rmsnorm_gated_mxfp4_quant(
                x,
                z,
                weight,
                1e-6,
                num_heads=num_heads,
                activation="swish",
                round_mode=MXFP4_ROUND_UP,
                shuffle_scales=True,
                use_native_dtypes=True,
            )


if __name__ == "__main__":
    unittest.main()
