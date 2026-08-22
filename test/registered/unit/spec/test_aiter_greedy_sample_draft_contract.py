"""Pins the aiter `greedy_sample` contract the topk=1 EAGLE draft branch relies on.

For topk=1 the draft tree is a single chain, so `topk_p` carries no information
and only the index matters. `EagleDraftWorker` therefore skips the full-vocab
softmax on ROCm and calls `aiter.greedy_sample` directly. Softmax is monotonic,
so the index must be identical either way.

Two properties make that substitution safe, and both are asserted here because
both live in aiter rather than in sglang:

1. `argmax_impl` reduces with `hipcub::ArgMax`, whose tie-break is an explicit
   lowest-index rule — so the winner among equal logits is independent of block
   reduction order. `torch.argmax` makes no such guarantee, which is what #26397
   cited when it gated the fast path to CUDA.
2. The kernel launches `BlockSize=1024 x VecSize=16` and is only correct above
   `1023 * 16`; at or below that a thread whose first offset already exceeds the
   vocab skips its load yet still reduces an uninitialized accumulator under a
   negative key. Above the floor every thread loads, so correctness there is
   structural, not statistical — that is the direction asserted below. The
   breakage *under* the floor is undefined behaviour and shows up only when the
   stale registers happen to win the reduction, so it is deliberately not
   asserted; the draft branch simply refuses that range.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=20, stage="stage-b", runner_config="1-gpu-small-amd")

try:
    from aiter import greedy_sample
except ImportError:
    greedy_sample = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB = 32768
# Mirrors the guard in EagleDraftWorker: BlockSize(1024) * VecSize(16).
AITER_VOCAB_FLOOR = 1023 * 16


def _argmax(logits: torch.Tensor) -> torch.Tensor:
    """The draft branch's inline body, as a test-local helper."""
    selected = torch.empty(logits.shape[0], dtype=torch.int32, device=logits.device)
    greedy_sample(selected, logits)
    return selected.to(torch.int64).unsqueeze(-1)


def _tied_logits(bs: int, num_ties: int = 4, seed: int = 0) -> torch.Tensor:
    """Rows whose maximum is duplicated at `num_ties` random indices."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn(bs, VOCAB, generator=generator) * 0.1
    for row in range(bs):
        tied = torch.randperm(VOCAB, generator=generator)[:num_ties]
        logits[row, tied] = 3.0
    return logits.to(DEVICE)


@unittest.skipIf(greedy_sample is None, "aiter not available on this platform")
class TestAiterGreedySampleDraftContract(CustomTestCase):
    def test_matches_softmax_slow_path(self):
        # The whole point of the branch: softmax cannot move the argmax.
        for bs in (1, 8, 64):
            with self.subTest(bs=bs):
                logits = torch.randn(bs, VOCAB, device=DEVICE)
                slow = torch.softmax(logits, dim=-1).max(dim=-1, keepdim=True).indices
                self.assertEqual(_argmax(logits).tolist(), slow.tolist())

    def test_index_shape_and_dtype_match_torch_argmax(self):
        # The branch feeds the same downstream code as the CUDA
        # torch.argmax(..., keepdim=True), so these must line up.
        logits = torch.randn(8, VOCAB, device=DEVICE)
        index = _argmax(logits)
        reference = torch.argmax(logits, dim=-1, keepdim=True)
        self.assertEqual(index.shape, reference.shape)
        self.assertEqual(index.dtype, reference.dtype)

    def test_empty_batch(self):
        index = _argmax(torch.empty(0, VOCAB, device=DEVICE))
        self.assertEqual(index.shape, (0, 1))
        self.assertEqual(index.dtype, torch.int64)

    def test_tie_break_is_stable_across_batch_shapes(self):
        # #26358: an argmax whose winner among tied logits shifts with the batch
        # shape makes the draft chain diverge run to run. The same rows must
        # select the same tokens regardless of what else shares the batch.
        head = _tied_logits(8, seed=1)
        reference = _argmax(head)
        for pad in (1, 7, 16, 33, 64):
            with self.subTest(pad=pad):
                padded = torch.cat([head, _tied_logits(pad, seed=pad + 2)], dim=0)
                self.assertEqual(_argmax(padded)[:8].tolist(), reference.tolist())

    def test_tie_break_is_stable_across_repeats(self):
        logits = _tied_logits(16, seed=3)
        reference = _argmax(logits)
        for _ in range(10):
            self.assertEqual(_argmax(logits).tolist(), reference.tolist())

    def test_tie_break_picks_lowest_index(self):
        # hipcub::ArgMax resolves equal values to the smaller key; that is what
        # makes the reduction order-independent.
        logits = torch.full((4, VOCAB), -10.0, device=DEVICE)
        expected = []
        for row, cols in enumerate(([5, 900, 3000], [0, 1], [VOCAB - 1, 42], [77, 78])):
            logits[row, cols] = 3.0
            expected.append([min(cols)])
        self.assertEqual(_argmax(logits).tolist(), expected)

    def test_correct_immediately_above_vocab_floor(self):
        # The draft branch admits anything above this, so it must be correct here.
        logits = torch.randn(8, AITER_VOCAB_FLOOR + 1, device=DEVICE)
        self.assertEqual(
            _argmax(logits).tolist(),
            torch.argmax(logits, dim=-1, keepdim=True).tolist(),
        )


if __name__ == "__main__":
    unittest.main()
