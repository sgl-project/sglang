from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

"""NaN-semantics parity with torch.argmax for the speculative argmax kernels.

The two-stage kernels must give NaNs priority and break all ties to the
lowest index, exactly like torch.argmax. Before the fix, an all-NaN row made
row_argmax return the vocab width and the markov greedy step return INT_MAX
-- both out of range, so feeding the result to the next embedding lookup is a
device-side failure. The medium-row path (rows > 64) already reduced with
_argmax_pair and is covered here as a regression guard.
"""

import unittest

import torch

from sglang.kernels.ops.speculative.dspark.dspark_draft_model import (
    markov_greedy_step,
    markov_greedy_step_triton,
)
from sglang.kernels.ops.speculative.row_argmax import row_argmax
from sglang.test.test_utils import CustomTestCase

DEVICE = torch.device("cuda")


class TestRowArgmaxNanSemantics(CustomTestCase):
    def _check(self, x):
        out = row_argmax(x)
        ref = x.argmax(dim=-1)
        self.assertTrue(torch.equal(out, ref), f"row_argmax {out} != torch {ref}")
        return out

    def test_all_nan_row_returns_first_index(self):
        # The pre-fix kernel returned the vocab width for these shapes.
        for rows, width in [(1, 131072), (4, 129280), (64, 152064)]:
            x = torch.full((rows, width), float("nan"), device=DEVICE)
            out = row_argmax(x)
            self.assertTrue(
                torch.equal(out, torch.zeros_like(out)),
                f"all-NaN rows={rows} width={width}: got {out}, want 0",
            )

    def test_first_nan_wins(self):
        for rows in (1, 64):
            x = torch.randn(rows, 131072, device=DEVICE)
            x[:, 90000] = float("nan")
            x[:, 5000] = float("nan")
            self._check(x)
            self.assertEqual(row_argmax(x)[0].item(), 5000)

    def test_nan_beats_finite_max(self):
        x = torch.zeros(2, 131072, device=DEVICE)
        x[0, 100] = 1e30
        x[0, 70000] = float("nan")
        x[1, 70000] = 1e30
        x[1, 50] = float("nan")
        self.assertEqual(row_argmax(x).tolist(), [70000, 50])

    def test_all_minus_inf_row(self):
        x = torch.full((4, 129280), float("-inf"), device=DEVICE)
        out = row_argmax(x)
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))

    def test_matches_torch_random_and_ties(self):
        g = torch.Generator(device=DEVICE).manual_seed(0)
        # rows <= 64 takes the two-stage path, rows > 64 the medium path.
        for rows, width in [
            (1, 129280),
            (7, 131072),
            (64, 152064),
            (65, 131072),
            (100, 129280),
        ]:
            x = torch.randn(rows, width, generator=g, device=DEVICE)
            self._check(x)
        # Exact ties resolve to the lowest index.
        x = torch.zeros(3, 131072, device=DEVICE)
        x[:, 60000] = 5.0
        x[:, 1234] = 5.0
        self.assertEqual(row_argmax(x).tolist(), [1234, 1234, 1234])
        # NaN mixed into random rows keeps parity on both paths.
        for rows in (64, 65):
            x = torch.randn(rows, 131072, generator=g, device=DEVICE)
            x[::3, 77777] = float("nan")
            self._check(x)


class TestMarkovGreedyNanSemantics(CustomTestCase):
    def _inputs(self, rows, vocab, rank=64, seed=0):
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        base = torch.randn(rows, vocab, generator=g, device=DEVICE)
        embeds = torch.randn(rows, rank, generator=g, device=DEVICE)
        w2 = torch.randn(vocab, rank, generator=g, device=DEVICE)
        return base, embeds, w2

    def _check(self, base, embeds, w2):
        out = markov_greedy_step_triton(
            base_logits=base, prev_embeds=embeds, w2_weight=w2
        )
        ref = markov_greedy_step(base_logits=base, prev_embeds=embeds, w2_weight=w2)
        self.assertTrue(torch.equal(out, ref), f"triton {out} != torch {ref}")
        return out

    def test_all_nan_base_returns_first_index(self):
        # The pre-fix combine kernel returned INT_MAX for these shapes.
        for rows, vocab in [(1, 152064), (4, 1000)]:
            base = torch.full((rows, vocab), float("nan"), device=DEVICE)
            embeds = torch.zeros(rows, 64, device=DEVICE)
            w2 = torch.randn(vocab, 64, device=DEVICE)
            out = markov_greedy_step_triton(
                base_logits=base, prev_embeds=embeds, w2_weight=w2
            )
            self.assertTrue(
                torch.equal(out, torch.zeros_like(out)),
                f"all-NaN rows={rows} vocab={vocab}: got {out}, want 0",
            )

    def test_first_nan_wins(self):
        base, embeds, w2 = self._inputs(2, 152064)
        base[:, 90000] = float("nan")
        base[:, 5000] = float("nan")
        self.assertEqual(
            markov_greedy_step_triton(
                base_logits=base, prev_embeds=embeds, w2_weight=w2
            ).tolist(),
            [5000, 5000],
        )

    def test_nan_beats_finite_max(self):
        base, embeds, w2 = self._inputs(2, 152064)
        base[0, 100] = 1e30
        base[0, 70000] = float("nan")
        base[1, 70000] = 1e30
        base[1, 50] = float("nan")
        self.assertEqual(
            markov_greedy_step_triton(
                base_logits=base, prev_embeds=embeds, w2_weight=w2
            ).tolist(),
            [70000, 50],
        )

    def test_all_minus_inf_base(self):
        # Padded lanes must not win even when every valid score is -inf.
        base = torch.full((4, 1000), float("-inf"), device=DEVICE)
        embeds = torch.zeros(4, 64, device=DEVICE)
        w2 = torch.randn(1000, 64, device=DEVICE)
        out = markov_greedy_step_triton(
            base_logits=base, prev_embeds=embeds, w2_weight=w2
        )
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))

    def test_matches_torch_reference_random(self):
        # vocab=1000 leaves padded lanes inside the last 256-wide tile.
        for rows, vocab in [(1, 152064), (8, 1000), (64, 129280)]:
            base, embeds, w2 = self._inputs(rows, vocab, seed=rows)
            self._check(base, embeds, w2)


if __name__ == "__main__":
    unittest.main()
