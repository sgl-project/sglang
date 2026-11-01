import unittest
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(
    torch.version.hip
    and torch.cuda.is_available()
    and torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"),
    "gfx950 (MI35x) only",
)
class TestKDAVerifyTileGfx950(CustomTestCase):
    def test_launch_config_boundaries(self):
        from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
            _select_recurrent_launch_config as cfg,
        )

        for n, bv in ((1, 4), (2, 4), (3, 8), (8, 8), (9, 16), (16, 16), (17, 32)):
            self.assertEqual(cfg(n, 16, 16, 128, 128, True, True), (bv, 1), f"n={n}")
        self.assertEqual(cfg(1, 8, 8, 128, 128, True, True), (4, 1))
        self.assertEqual(cfg(1, 17, 17, 128, 128, True, True), (32, 1))
        self.assertEqual(cfg(1, 32, 32, 128, 128, True, True), (32, 1))
        self.assertEqual(cfg(1, 16, 16, 128, 128, True, False), (32, 1))
        self.assertEqual(cfg(1, 16, 16, 64, 64, True, True), (32, 1))

    def test_narrow_tile_is_bit_exact(self):
        import sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent as R

        torch.manual_seed(0)
        H, K, T = 16, 128, 6
        for B in (1, 3, 9):
            q, k, v = (
                torch.randn(1, B * T, H, K, dtype=torch.bfloat16, device="cuda")
                for _ in range(3)
            )
            a = torch.randn(B * T, H * K, dtype=torch.bfloat16, device="cuda")
            b = torch.randn(1, B * T, H, dtype=torch.bfloat16, device="cuda")
            A_log = torch.randn(H, dtype=torch.float32, device="cuda")
            dt_bias = torch.randn(H * K, dtype=torch.float32, device="cuda")
            pool = torch.randn(B + 1, H, K, K, dtype=torch.float32, device="cuda")
            slots = torch.arange(B, dtype=torch.int32, device="cuda")

            def run():
                states = torch.zeros(B + 1, T, H, K, K, device="cuda")
                out = R.fused_sigmoid_gating_delta_rule_update(
                    A_log=A_log,
                    dt_bias=dt_bias,
                    q=q,
                    k=k,
                    v=v,
                    a=a,
                    b=b,
                    initial_state_source=pool.clone(),
                    initial_state_indices=slots,
                    cu_seqlens=torch.arange(
                        0, B * T + 1, T, dtype=torch.int32, device="cuda"
                    ),
                    use_qk_l2norm_in_kernel=True,
                    softplus_beta=1.0,
                    softplus_threshold=20.0,
                    is_kda=True,
                    disable_state_update=True,
                    intermediate_states_buffer=states,
                    intermediate_state_indices=slots,
                    cache_steps=T,
                )
                return out, states

            out, states = run()
            with mock.patch.object(
                R, "_select_recurrent_launch_config", lambda *args, **kwargs: (32, 1)
            ):
                ref_out, ref_states = run()
            self.assertTrue(torch.equal(out, ref_out), f"B={B}")
            self.assertTrue(torch.equal(states, ref_states), f"B={B}")


if __name__ == "__main__":
    unittest.main()
