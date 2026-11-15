"""Mamba SSM commit row loop (ssm_scatter_rows_loop): one launch over a padded request list
is bitwise identical to the masked per-request scatters it replaces (accept set, accept +
track set, all padding)."""

import unittest

import torch

from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
    fused_mamba_state_scatter_with_mask,
    ssm_scatter_rows_loop,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA")
class TestSsmCommitRowLoop(unittest.TestCase):
    def _case(self, n, with_track, pad, layers=4, cache=48, rows=40, steps=4):
        g = torch.Generator(device="cuda").manual_seed(
            n * 13 + int(with_track) + 7 * pad
        )
        state_shape = (2, 16, 32)
        src = torch.randn(
            layers, rows + 1, steps, *state_shape, device="cuda", generator=g
        )
        base = torch.randn(layers, cache, *state_shape, device="cuda", generator=g)
        perm = torch.randperm(cache, device="cuda", generator=g)
        total = n + pad
        slots = torch.zeros(total, dtype=torch.int64, device="cuda")
        slots[:n] = perm[:n]
        step = torch.full((total,), -1, dtype=torch.int32, device="cuda")
        step[:n] = torch.randint(0, steps, (n,), device="cuda", generator=g).int()
        # Scattered scratch rows; padding rows point at the scratch padding row.
        src_rows = torch.full((total,), rows, dtype=torch.int32, device="cuda")
        src_rows[:n] = torch.randperm(rows, device="cuda", generator=g)[:n].int()
        track_slots = track_steps = None
        if with_track:
            track_slots = torch.zeros(total, dtype=torch.int64, device="cuda")
            track_slots[:n] = perm[n : 2 * n]
            track_steps = torch.full((total,), -1, dtype=torch.int32, device="cuda")
            track_steps[:n] = torch.randint(
                -1, steps, (n,), device="cuda", generator=g
            ).int()
        ref = base.clone()
        fused_mamba_state_scatter_with_mask(ref, src, slots, step, src_rows)
        if with_track:
            fused_mamba_state_scatter_with_mask(
                ref, src, track_slots, track_steps, src_rows
            )
        out = base.clone()
        ssm_scatter_rows_loop(out, src, slots, step, src_rows, track_slots, track_steps)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)

    def test_accept_only(self):
        for n, pad in ((1, 0), (3, 5), (8, 24)):
            self._case(n, with_track=False, pad=pad)

    def test_accept_and_track(self):
        for n, pad in ((1, 7), (5, 0), (8, 24)):
            self._case(n, with_track=True, pad=pad)

    def test_all_padding(self):
        self._case(0, with_track=True, pad=16)


if __name__ == "__main__":
    unittest.main()
