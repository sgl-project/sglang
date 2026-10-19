import itertools
import unittest

import torch

from sglang.kernels.ops.mamba import causal_conv1d_prefill_vectorized
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _same_bytes(actual, expected):
    return torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestCausalConv1dPrefill(CustomTestCase):
    def _case(
        self,
        lengths,
        channels=1536,
        initial=True,
        bias=False,
        block_m=64,
        table=False,
        pad=False,
        pitch_extra=0,
        activation="silu",
    ):
        torch.manual_seed(791)
        total = sum(lengths)
        pitch = channels * 5 // 3 + pitch_extra
        packed = torch.randn(total, pitch, device="cuda", dtype=torch.bfloat16)
        x = packed[:, :channels].T
        if total:
            x[0, 0] = -0.0
        weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(channels, device="cuda", dtype=x.dtype) if bias else None
        batch = len(lengths)
        before = torch.randn(batch + 3, channels, 3, device="cuda", dtype=x.dtype)
        before[:, 0, :] = -0.0
        actual_state, expected_state = before.clone(), before.clone()
        ids = list(reversed(range(1, batch + 1)))
        if pad:
            ids[-1] = -1
        indices = torch.tensor(ids, device="cuda", dtype=torch.int32)
        flags = (
            torch.tensor([i % 2 == 0 for i in range(batch)], device="cuda")
            if initial
            else None
        )
        cu = torch.tensor(
            [0] + list(itertools.accumulate(lengths)), device="cuda", dtype=torch.int32
        )
        blocks = None
        if table:
            pairs = [
                (i, j)
                for i, length in enumerate(lengths)
                for j in range((length + block_m - 1) // block_m)
            ]
            # Reversed scheduling must preserve the first-chunk state ownership.
            blocks = torch.tensor(
                list(reversed(pairs)) + [(-1, 0)], device="cuda", dtype=torch.int32
            )
        actual = causal_conv1d_prefill_vectorized(
            x,
            weight,
            bias,
            actual_state,
            cu,
            lengths,
            indices,
            flags,
            activation=activation,
            block_table=blocks,
            block_m=block_m,
        )
        if total:
            expected = causal_conv1d_fn(
                x,
                weight,
                bias,
                expected_state,
                cu,
                lengths,
                indices,
                flags,
                activation=activation,
                block_table=blocks if block_m == 8 else None,
            )
            for i, length in enumerate(lengths):
                if ids[i] != -1:
                    lo, hi = cu[i].item(), cu[i + 1].item()
                    self.assertTrue(
                        _same_bytes(
                            actual[:, lo:hi].T.contiguous().view(torch.uint8),
                            expected[:, lo:hi].T.contiguous().view(torch.uint8),
                        )
                    )
        self.assertTrue(_same_bytes(actual_state, expected_state))
        self.assertEqual(actual.shape, x.shape)
        self.assertTrue(actual.T.is_contiguous())
        # The persisted history is raw input, independent of weights and SiLU.
        offset = 0
        for i, length in enumerate(lengths):
            slot = ids[i]
            if length and slot != -1:
                prefix = (
                    before[slot]
                    if initial and i % 2 == 0
                    else torch.zeros_like(before[slot])
                )
                raw = torch.cat((prefix, x[:, offset : offset + length]), dim=1)
                self.assertTrue(_same_bytes(actual_state[slot], raw[:, -3:]))
            offset += length

    def test_short_state_and_tile_boundaries(self):
        """History merge and empty sequences survive either token tiling."""
        for block_m in (8, 64):
            for lengths in ([0], [1, 2, 3], [0, 63, 65], [64, 3, 2]):
                with self.subTest(block_m=block_m, lengths=lengths):
                    self._case(
                        lengths,
                        channels=130,
                        block_m=block_m,
                        table=True,
                        bias=True,
                        pitch_extra=16,
                    )
        self._case([1, 2, 65], initial=False, pad=True, table=True, activation=None)

    def test_block_table_exceeds_grid_z_limit(self):
        """Large block tables use grid.x, whose bound exceeds the 16-bit axes."""
        self._case([65536 * 8], channels=1, block_m=8, table=True)

    def test_long_strided_prefill(self):
        """Packed projection strides, TP head splits and long offsets agree with stock."""
        for total, channels in ((4096, 3072), (32768, 1536), (131072, 12288)):
            with self.subTest(total=total, channels=channels):
                self._case(
                    [total // 7, total // 3, total - total // 7 - total // 3],
                    channels=channels,
                    pitch_extra=128,
                )

    def test_split_continuation_equals_one_pass(self):
        """Continuing at a non-tile boundary preserves both outputs and raw history."""
        torch.manual_seed(883)
        x = torch.randn(197, 256, device="cuda", dtype=torch.bfloat16).T
        weight = torch.randn(256, 4, device="cuda", dtype=x.dtype)
        state = torch.randn(1, 256, 3, device="cuda", dtype=x.dtype)
        split_state = state.clone()
        initial = torch.ones(1, device="cuda", dtype=torch.bool)

        def call(part, cache):
            cu = torch.tensor([0, part.shape[1]], device="cuda", dtype=torch.int32)
            return causal_conv1d_prefill_vectorized(
                part,
                weight,
                None,
                cache,
                cu,
                [part.shape[1]],
                has_initial_state=initial,
            )

        full = call(x, state)
        split = torch.cat(
            (
                call(x[:, :2], split_state),
                call(x[:, 2:65], split_state),
                call(x[:, 65:], split_state),
            ),
            dim=1,
        )
        self.assertTrue(_same_bytes(full, split))
        self.assertTrue(_same_bytes(state, split_state))

    def test_channel_slices_preserve_pool_stride(self):
        """A channel slice keeps the full pool's sequence stride and base offset."""
        torch.manual_seed(881)
        full_channels, total = 12288, 65
        packed = torch.randn(total, 20608, device="cuda", dtype=torch.bfloat16)
        x = packed[:, :full_channels].T
        weight = torch.randn(full_channels, 4, device="cuda", dtype=x.dtype)
        before = torch.randn(5, full_channels, 3, device="cuda", dtype=x.dtype)
        cu = torch.tensor([0, 2, total], device="cuda", dtype=torch.int32)
        ids = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
        initial = torch.tensor([True, False], device="cuda")
        reference_state = before.clone()
        reference = causal_conv1d_fn(
            x,
            weight,
            None,
            reference_state,
            cu,
            [2, total - 2],
            ids,
            initial,
        )
        for channels in (256, 512, 2048, 8192):
            with self.subTest(channels=channels):
                lo, hi = 256, 256 + channels
                state = before.clone()
                actual = causal_conv1d_prefill_vectorized(
                    x[lo:hi],
                    weight[lo:hi],
                    None,
                    state[:, lo:hi],
                    cu,
                    [2, total - 2],
                    ids,
                    initial,
                )
                self.assertTrue(_same_bytes(actual, reference[lo:hi]))
                self.assertTrue(_same_bytes(state[:, lo:hi], reference_state[:, lo:hi]))
                self.assertTrue(_same_bytes(state[:, :lo], before[:, :lo]))
                self.assertTrue(_same_bytes(state[:, hi:], before[:, hi:]))


if __name__ == "__main__":
    unittest.main()
