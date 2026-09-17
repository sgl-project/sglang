import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch

from sglang.kernels.ops.speculative.dflash import selector_walk_triton
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _walk(scores, greedy):
    batch, slots, top_k, _ = scores.shape
    candidate_ids = torch.arange(batch * slots * top_k).view(batch, slots, top_k)
    tokens, q_rows = selector_walk_triton(
        candidate_ids=candidate_ids,
        scores=scores,
        uniforms=torch.full((batch, slots), 0.5),
        temperatures=torch.ones(batch),
        greedy_mask=torch.full((batch,), greedy, dtype=torch.bool),
    )
    return candidate_ids, tokens, q_rows


class TestDFlashSelectorWalkNanRow(CustomTestCase):
    def test_all_nan_greedy_row_stays_inside_its_candidate_row(self):
        batch, slots, top_k = 2, 3, 4
        scores = torch.randn(batch, slots, top_k, top_k)
        scores[0] = float("nan")
        candidate_ids, tokens, q_rows = _walk(scores, greedy=True)

        for slot in range(slots):
            self.assertIn(int(tokens[0, slot]), candidate_ids[0, slot].tolist())
            self.assertEqual(float(q_rows[0, slot].sum()), 1.0)
        # Matches torch.argmax on the same rows (index 0 for all-NaN).
        self.assertEqual(tokens[0].tolist(), candidate_ids[0, :, 0].tolist())

    def test_finite_greedy_row_matches_torch_argmax(self):
        batch, slots, top_k = 3, 4, 8
        scores = torch.randn(batch, slots, top_k, top_k)
        scores[1, 2, :, :] = 0.0  # ties break to the left
        candidate_ids, tokens, q_rows = _walk(scores, greedy=True)

        previous = torch.zeros(batch, dtype=torch.int64)
        for slot in range(slots):
            rows = scores[torch.arange(batch), slot, previous]
            expected = rows.argmax(dim=-1)
            self.assertEqual(
                tokens[:, slot].tolist(),
                candidate_ids[torch.arange(batch), slot, expected].tolist(),
            )
            torch.testing.assert_close(
                q_rows[:, slot], torch.nn.functional.one_hot(expected, top_k).float()
            )
            previous = expected


if __name__ == "__main__":
    import unittest

    unittest.main()
