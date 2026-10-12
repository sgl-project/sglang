"""Idle Mega mHC output must remain consumable by MegaMoE dispatch."""

import unittest

import torch

from sglang.kernels.ops.layernorm.mhc_mega import mhc_mega_boundary
from sglang.kernels.ops.moe.dsv4 import mega_moe_pre_dispatch
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


class TestMegaMHC(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            raise unittest.SkipTest("Mega mHC requires SM10x")

    def test_idle_output_feeds_mega_moe_dispatch(self):
        hidden, hc, top_k = 5120, 4, 6
        residual = torch.empty((0, hc, hidden), dtype=torch.bfloat16, device="cuda")
        x = residual[:, 0, :]
        self.assertEqual(x.stride(), (hc * hidden, 1))
        self.assertTrue(x.is_contiguous())
        mix = torch.empty((0, hc), dtype=torch.float32, device="cuda")
        comb = torch.empty((0, hc, hc), dtype=torch.float32, device="cuda")
        _, normalized, _ = mhc_mega_boundary(
            x,
            residual,
            mix,
            mix,
            comb,
            torch.zeros(
                (hc * (hc + 2), hc * hidden), dtype=torch.float32, device="cuda"
            ),
            torch.ones(3, dtype=torch.float32, device="cuda"),
            torch.zeros(hc * (hc + 2), dtype=torch.float32, device="cuda"),
            torch.ones(hidden, dtype=torch.bfloat16, device="cuda"),
            1e-6,
            1e-6,
            1e-6,
            20,
        )
        topk_ids = torch.empty((0, top_k), dtype=torch.int32, device="cuda")
        topk_weights = torch.empty((0, top_k), dtype=torch.float32, device="cuda")
        padded_ids = torch.full((1, top_k), 123, dtype=torch.int64, device="cuda")
        padded_weights = torch.ones((1, top_k), dtype=torch.float32, device="cuda")
        mega_moe_pre_dispatch(
            normalized,
            topk_ids,
            topk_weights,
            torch.empty((1, hidden), dtype=torch.float8_e4m3fn, device="cuda"),
            torch.empty((1, hidden // 32 // 4), dtype=torch.int32, device="cuda"),
            padded_ids,
            padded_weights,
        )
        torch.testing.assert_close(padded_ids, torch.full_like(padded_ids, -1))
        torch.testing.assert_close(padded_weights, torch.zeros_like(padded_weights))


if __name__ == "__main__":
    unittest.main()
