"""Test circular checkpoint materialization independently of verify and prefill."""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
    commit_gdn_replayssm_circular,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")


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

    def make_commit(self, flush, *, track, with_residual=False):
        state = self.checkpoint.clone()
        scales = self.scales.clone()
        write_pos = self.cursor([4, 9, 3, 7])
        cache_base = self.cursor([4, 8, 2, 29])
        is_flush = self.cursor(flush, torch.int8)
        track_args = (
            {
                "mamba_track_indices": self.index([-1, 5, -1, 6]),
                "mamba_steps_to_track": self.index([-1, 0, -1, 2]),
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
        return kwargs, (state, scales, write_pos, cache_base, is_flush)

    def run_commit(self, flush, *, track, ordinary=False, with_residual=False):
        kwargs, result = self.make_commit(
            flush, track=track, with_residual=with_residual
        )
        if ordinary:
            with patch("torch.cuda.get_device_capability", return_value=(10, 0)):
                commit_gdn_replayssm_circular(**kwargs)
        else:
            commit_gdn_replayssm_circular(**kwargs)
        return result

    def assert_same_commit(self, actual, expected):
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, rtol=0, atol=0)

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
        self.assert_same_commit((state, scales, pos, base, flush), ordinary)

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

    def test_rejected_nonfinite_tail_is_ignored(self):
        clean = self.run_commit([0, 0, 0, 1], track=True)
        # Active row 3 wraps from position 29 and accepts through position 3.
        # Track-only row 0 accepts through position 7.
        poisoned = (
            self.d[:, 3, :, 4],
            self.k[:, 3, :, 4],
            self.g[:, 3, :, 4],
            self.d[:, 0, :, 8],
            self.k[:, 0, :, 8],
            self.g[:, 0, :, 8],
        )
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                for tensor in poisoned:
                    tensor.fill_(value)
                dirty = self.run_commit([0, 0, 0, 1], track=True)
                self.assert_same_commit(dirty, clean)

    def test_tracking_snapshot_excludes_later_accepted_updates(self):
        if torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("tests the SM120 persistent path")
        kwargs, (state, scales, *_cursors) = self.make_commit([0, 0, 0, 1], track=True)
        # Both writes use row 3. The tracking snapshot ends at history length
        # five, two updates before the active checkpoint.
        kwargs["mamba_track_indices"][0] = 2
        kwargs["mamba_steps_to_track"][0] = 0
        commit_gdn_replayssm_circular(**kwargs)
        expected = state[:, 2].clone(), scales[2].clone()
        self.d[:, 3, :, 2].fill_(float("nan"))
        self.k[:, 3, :, 2].fill_(float("inf"))
        dirty, (state, scales, *_cursors) = self.make_commit([0, 0, 0, 1], track=True)
        dirty["mamba_track_indices"][0] = 2
        dirty["mamba_steps_to_track"][0] = 0
        commit_gdn_replayssm_circular(**dirty)
        self.assert_same_commit((state[:, 2], scales[2]), expected)

    def test_commit_matches_fp64_recurrence(self):
        # Advance scalar decay and outer products one token at a time so the
        # oracle does not share the kernel's prefix scan or batched dot.
        group = self.value_heads // self.heads
        for with_residual in (False, True):
            kwargs, (state, scales, *_cursors) = self.make_commit(
                [0, 0, 0, 1], track=True, with_residual=with_residual
            )
            active_ref = self.checkpoint[:, 1].double() * self.scales[
                1
            ].double().unsqueeze(-1)
            track_ref = self.checkpoint[:, 3].double() * self.scales[
                3
            ].double().unsqueeze(-1)
            active_error, track_error = 0.0, 0.0

            def quantized_state(x):
                scale = (x.abs().amax(dim=-1) / 127).clamp_min(1e-12).float().double()
                return (x / scale.unsqueeze(-1)).round().clamp(
                    -127, 127
                ) * scale.unsqueeze(-1)

            def advance(ref, row, base, length):
                for offset in range(length):
                    pos = (base + offset) % self.ring
                    update = self.d[:, row, :, pos].double()
                    key = self.k[:, row, :, pos].double()
                    if with_residual:
                        update += kwargs["d_residual_cache"][:, row, :, pos].double()
                        key += kwargs["k_residual_cache"][:, row, :, pos].double()
                    key = key.repeat_interleave(group, dim=1)
                    decay = self.g[:, row, :, pos].double().exp()[..., None, None]
                    ref = decay * ref + update.unsqueeze(-1) * key.unsqueeze(-2)
                return quantized_state(ref)

            def relative_l2(slot, ref):
                decoded = state[:, slot].double() * scales[slot].double().unsqueeze(-1)
                return float(
                    torch.linalg.vector_norm(decoded - ref)
                    / torch.linalg.vector_norm(ref)
                )

            for step in range(8):
                active_base = (29 + 7 * step) % self.ring
                track_base = (4 + 5 * step) % self.ring
                kwargs["write_pos"].copy_(self.index([4, 9, 3, 7]))
                kwargs["cache_base"].copy_(self.index([track_base, 8, 2, active_base]))
                kwargs["is_flush"].copy_(self.index([0, 0, 0, 1], torch.int8))
                active_ref = advance(active_ref, 3, active_base, 7)
                expected_track = advance(track_ref, 0, track_base, 3)
                commit_gdn_replayssm_circular(**kwargs)
                active_error = max(active_error, relative_l2(1, active_ref))
                track_error = max(track_error, relative_l2(5, expected_track))

            self.assertLess(active_error, 0.015, (with_residual, active_error))
            self.assertLess(track_error, 0.015, (with_residual, track_error))
            print(
                f"FP64 reference residual={with_residual}: active={active_error:.7f}, tracking={track_error:.7f}"
            )

    def test_cuda_graph_replay_matches_regular_grid(self):
        if torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("tests the SM120 persistent path")
        graph_args, graph_output = self.make_commit([0, 0, 0, 1], track=True)
        regular_args, regular_output = self.make_commit([0, 0, 0, 1], track=True)
        commit_gdn_replayssm_circular(**graph_args)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            commit_gdn_replayssm_circular(**graph_args)

        cases = (
            ([0, 0, 0, 0], False),
            ([0, 0, 0, 1], False),
            ([1, 0, 1, 1], False),
            ([0, 0, 0, 0], True),
            ([0, 0, 0, 1], True),
        )
        for flags, tracking in cases:
            with self.subTest(flags=flags, tracking=tracking):
                for args in (graph_args, regular_args):
                    args["checkpoint_state"].copy_(self.checkpoint)
                    args["checkpoint_scale"].copy_(self.scales.transpose(0, 1))
                    args["write_pos"].copy_(self.index([4, 9, 3, 7]))
                    args["cache_base"].copy_(self.index([4, 8, 2, 29]))
                    args["is_flush"].copy_(self.index(flags, torch.int8))
                    args["mamba_track_indices"].copy_(
                        self.index([-1, 5, -1, 6] if tracking else [-1] * 4)
                    )
                graph.replay()
                with patch("torch.cuda.get_device_capability", return_value=(10, 0)):
                    commit_gdn_replayssm_circular(**regular_args)
                self.assert_same_commit(graph_output, regular_output)


if __name__ == "__main__":
    unittest.main()
