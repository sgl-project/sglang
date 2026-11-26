"""Test circular checkpoint materialization independently of verify and prefill."""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
    commit_gdn_replayssm_circular,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is None, "requires NVIDIA CUDA"
)
class TestGDNReplaySSMInt8Commit(unittest.TestCase):
    def setUp(self):
        self.gen = torch.Generator(device="cuda").manual_seed(11)
        self.layers, self.slots, self.heads, self.value_heads = 2, 8, 2, 6
        self.dim, self.ring = 128, 32
        layer_shape = (
            self.layers,
            self.slots,
            self.value_heads,
            self.dim,
            self.dim,
        )
        self.checkpoint = torch.randint(
            -127, 128, layer_shape, device="cuda", dtype=torch.int8, generator=self.gen
        )
        self.scales = (
            torch.rand(
                (self.slots, self.layers, self.value_heads, self.dim),
                device="cuda",
                generator=self.gen,
            )
            * 0.02
            + 0.001
        )
        self.d = self.randn(
            self.layers, self.slots, self.value_heads, self.ring, self.dim
        )
        self.k = self.randn(self.layers, self.slots, self.heads, self.ring, self.dim)
        self.g = (
            -self.randn(self.layers, self.slots, self.value_heads, self.ring)
            .float()
            .abs()
        )

    def randn(self, *shape):
        return (
            torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=self.gen)
            * 0.2
        )

    @staticmethod
    def index(values, dtype=torch.int32):
        return torch.tensor(values, dtype=dtype, device="cuda")

    def cursor(self, values, dtype=torch.int32):
        # Pitched columns exercise all cursor strides in the persistent grid.
        out = torch.zeros((len(values), 3), device="cuda", dtype=dtype)[:, 1]
        out.copy_(self.index(values, dtype))
        return out

    def run_commit(self, flush, *, track, ordinary=False, with_residual=False):
        state = self.checkpoint.clone()
        scales = self.scales.clone()
        write_pos = self.cursor([4, 9, 3, 7])
        cache_base = self.cursor([4, 8, 2, 29])
        is_flush = self.cursor(flush, torch.int8)
        track_args = (
            {
                "mamba_track_indices": self.index([-1, 5, -1, 6]),
                "mamba_steps_to_track": self.index([-1, 1, -1, 2]),
            }
            if track
            else {}
        )
        kwargs = dict(
            checkpoint_state=state,
            checkpoint_scale=scales.transpose(0, 1),
            d_cache=self.d,
            k_cache=self.k,
            g_cache=self.g,
            d_residual_cache=self.d * 0.25 if with_residual else None,
            k_residual_cache=self.k * 0.25 if with_residual else None,
            state_batch_indices=self.index([1, 3, 4, -1]),
            replay_indices=self.index([3, 0, 2, 1]),
            write_pos=write_pos,
            cache_base=cache_base,
            is_flush=is_flush,
            accept_lens=self.index([3, 2, 2, 2]),
            **track_args,
        )
        if ordinary:
            with patch("torch.cuda.get_device_capability", return_value=(10, 0)):
                commit_gdn_replayssm_circular(**kwargs)
        else:
            commit_gdn_replayssm_circular(**kwargs)
        return state, scales, write_pos, cache_base, is_flush

    def test_fold_tracking_and_padding(self):
        state, scales, pos, base, flush = self.run_commit([0, 1, 0, 1], track=True)
        # The first row folds across the ring boundary, the second only tracks,
        # the third stays clean, and the fourth is padding.
        for untouched in (0, 2, 3, 4, 6, 7):
            torch.testing.assert_close(
                state[:, untouched], self.checkpoint[:, untouched]
            )
            torch.testing.assert_close(scales[untouched], self.scales[untouched])
        self.assertFalse(torch.equal(state[:, 1], self.checkpoint[:, 1]))
        self.assertFalse(torch.equal(state[:, 5], self.checkpoint[:, 5]))
        self.assertEqual(pos.tolist(), [4, 9, 3, 0])
        self.assertEqual(base.tolist(), [4, 8, 2, 4])
        self.assertEqual(flush.tolist(), [0, 1, 0, 0])

        ordinary = self.run_commit([0, 1, 0, 1], track=True, ordinary=True)
        for got, expected in zip((state, scales, pos, base, flush), ordinary):
            torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_no_flush_keeps_state_and_cursors(self):
        state, scales, pos, base, flush = self.run_commit([0, 0, 0, 0], track=False)
        torch.testing.assert_close(state, self.checkpoint)
        torch.testing.assert_close(scales, self.scales)
        self.assertEqual(pos.tolist(), [4, 9, 3, 7])
        self.assertEqual(base.tolist(), [4, 8, 2, 29])
        self.assertEqual(flush.tolist(), [0, 0, 0, 0])

    def test_residual_inputs_use_the_compensated_commit(self):
        with patch(
            "sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode."
            "gdn_replayssm_compact_commit_persistent_kernel",
            None,
        ):
            with_residual = self.run_commit(
                [0, 0, 0, 1], track=False, with_residual=True
            )
        no_residual = self.run_commit([0, 0, 0, 1], track=False)
        self.assertFalse(torch.equal(with_residual[0][:, 1], no_residual[0][:, 1]))


if __name__ == "__main__":
    unittest.main()
