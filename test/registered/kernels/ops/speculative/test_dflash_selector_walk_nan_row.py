import os
import unittest

import torch

# The Triton execution mode is bound when @triton.jit decorates the kernel
# at import time, so the interpreter fallback for GPU-less hosts must be
# enabled before the kernel module is imported. Hosts with a GPU (including
# the CUDA CI runner) keep the compiled kernel and exercise the real launch
# path below. Registered test files run one file per forked worker process,
# so the variable cannot leak into other test modules sharing a process.
if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

from sglang.kernels.ops.speculative.dflash import selector_walk_triton
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _walk(scores, greedy):
    batch, slots, top_k, _ = scores.shape
    device = scores.device
    candidate_ids = torch.arange(batch * slots * top_k, device=device).view(
        batch, slots, top_k
    )
    tokens, q_rows = selector_walk_triton(
        candidate_ids=candidate_ids,
        scores=scores,
        uniforms=torch.full((batch, slots), 0.5, device=device),
        temperatures=torch.ones(batch, device=device),
        greedy_mask=torch.full((batch,), greedy, dtype=torch.bool, device=device),
    )
    return candidate_ids, tokens, q_rows


class TestDFlashSelectorWalkNanRow(CustomTestCase):
    def test_all_nan_greedy_row_stays_inside_its_candidate_row(self):
        batch, slots, top_k = 2, 3, 4
        scores = torch.randn(batch, slots, top_k, top_k, device=DEVICE)
        scores[0] = float("nan")
        candidate_ids, tokens, q_rows = _walk(scores, greedy=True)

        for slot in range(slots):
            self.assertIn(int(tokens[0, slot]), candidate_ids[0, slot].tolist())
            self.assertEqual(float(q_rows[0, slot].sum()), 1.0)
        # Matches torch.argmax on the same rows (index 0 for all-NaN).
        self.assertEqual(tokens[0].tolist(), candidate_ids[0, :, 0].tolist())

    def test_finite_greedy_row_matches_torch_argmax(self):
        batch, slots, top_k = 3, 4, 8
        scores = torch.randn(batch, slots, top_k, top_k, device=DEVICE)
        scores[1, 2, :, :] = 0.0  # ties break to the left
        candidate_ids, tokens, q_rows = _walk(scores, greedy=True)

        batch_idx = torch.arange(batch, device=DEVICE)
        previous = torch.zeros(batch, dtype=torch.int64, device=DEVICE)
        for slot in range(slots):
            rows = scores[batch_idx, slot, previous]
            expected = rows.argmax(dim=-1)
            self.assertEqual(
                tokens[:, slot].tolist(),
                candidate_ids[batch_idx, slot, expected].tolist(),
            )
            torch.testing.assert_close(
                q_rows[:, slot], torch.nn.functional.one_hot(expected, top_k).float()
            )
            previous = expected

    def test_mixed_nan_greedy_row_matches_torch_argmax(self):
        batch, slots, top_k = 4, 3, 8
        nan = float("nan")
        patterns = [
            [2.0, nan, 1.0, 0.5, nan, -1.0, 0.25, -0.5],  # finite max precedes a NaN
            [nan, 3.0, 2.0, 1.0, 0.0, -1.0, 0.5, -0.25],  # NaN before every finite
            [1.0, nan, 2.0, nan, 0.5, 3.0, nan, -2.0],  # interleaved NaNs
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # finite control
        ]
        scores = torch.empty(batch, slots, top_k, top_k)
        for b, pattern in enumerate(patterns):
            row = torch.tensor(pattern)
            for slot in range(slots):
                for prev in range(top_k):
                    # Rotate so the first NaN lands at every position somewhere.
                    scores[b, slot, prev] = torch.roll(row, slot + prev)
        scores = scores.to(DEVICE)
        candidate_ids, tokens, q_rows = _walk(scores, greedy=True)

        batch_idx = torch.arange(batch, device=DEVICE)
        previous = torch.zeros(batch, dtype=torch.int64, device=DEVICE)
        for slot in range(slots):
            rows = scores[batch_idx, slot, previous]
            expected = rows.argmax(dim=-1)
            for b in range(batch):
                self.assertIn(int(tokens[b, slot]), candidate_ids[b, slot].tolist())
            self.assertEqual(
                tokens[:, slot].tolist(),
                candidate_ids[batch_idx, slot, expected].tolist(),
            )
            torch.testing.assert_close(
                q_rows[:, slot], torch.nn.functional.one_hot(expected, top_k).float()
            )
            previous = expected


if __name__ == "__main__":
    unittest.main()
