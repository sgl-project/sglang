"""Unit tests for the NVFP4 checkpoint conversion of SM90 FlashInfer MegaMoE."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.moe import mega_moe_sm90_fi_nvfp4 as fi_nvfp4

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _e4m3_bytes(values: list[float], shape) -> torch.Tensor:
    return torch.tensor(values).to(torch.float8_e4m3fn).view(torch.uint8).reshape(shape)


class TestNvfp4CheckpointConversion(CustomTestCase):
    def test_rebalance_preserves_scales_and_fits_decoder(self):
        # Local scales up to 256 (ModelOpt DeepSeek-V4 range) and gate / up
        # globals a factor of two apart.
        generator = torch.Generator().manual_seed(7)
        e, n, groups = 2, 8, 4
        choices = torch.tensor([4.0, 16.0, 128.0, 192.0, 256.0])
        local = choices[torch.randint(0, 5, (e, 2 * n, groups), generator=generator)]
        scale = local.to(torch.float8_e4m3fn).view(torch.uint8)
        global_scale = torch.tensor([[2.0**-13, 2.0**-12], [2.0**-10, 2.0**-10]])

        out_scale, out_global = fi_nvfp4._rebalance_scales(
            scale, global_scale, gated=True
        )

        out_local = out_scale.view(torch.float8_e4m3fn).float()
        self.assertTrue((6 * out_local <= 448).all())
        row_global = torch.cat(
            (global_scale[:, :1].expand(-1, n), global_scale[:, 1:].expand(-1, n)),
            dim=1,
        )
        self.assertTrue(
            torch.equal(
                out_local * out_global[:, None, None], local * row_global[..., None]
            )
        )

    def test_unrepresentable_scale_range_is_rejected(self):
        # 448 and 2^-9 in one expert: moving 448 into range pushes 2^-9 below E4M3.
        scale = _e4m3_bytes([448.0, 2.0**-9], (1, 1, 2))
        with self.assertRaises(ValueError):
            fi_nvfp4._rebalance_scales(scale, torch.ones(1), gated=False)

    def test_nibble_order_matches_flashinfer_layout(self):
        # Checkpoint byte i holds K[2i] low and K[2i + 1] high; FlashInfer expects
        # byte b = K[b] << 4 | K[b + 4] within every 4-byte word.
        packed = torch.tensor([[[0x10, 0x32, 0x54, 0x76]]], dtype=torch.uint8)
        out = fi_nvfp4._low_nibble_first_to_marlin(packed)
        self.assertEqual(out.flatten().tolist(), [0x04, 0x15, 0x26, 0x37])


if __name__ == "__main__":
    unittest.main()
