"""Seeded sampling hashes Gumbel noise by token id, even when the batch took
the top-k/top-p sort path."""

import unittest

import torch

from sglang.srt.layers.sampler import (
    sampling_from_probs_torch,
    top_k_top_p_min_p_sampling_from_probs_torch,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=15, suite="stage-b-test-1-gpu-small-amd")

VOCAB = 32
DEVICE = "cuda"
# Near-uniform so Gumbel noise, not the distribution, picks the token.
LOGIT_SCALE = 0.05


def _seeded_simple(probs: torch.Tensor, seed: int, position: int) -> int:
    device = probs.device
    return int(
        sampling_from_probs_torch(
            probs,
            sampling_seed=torch.tensor([seed], device=device, dtype=torch.int64),
            positions=torch.tensor([position], device=device, dtype=torch.int64),
        ).item()
    )


def _seeded_complex(
    probs: torch.Tensor,
    top_ks: list[int],
    seeds: list[int],
    positions: list[int],
) -> torch.Tensor:
    batch = probs.shape[0]
    device = probs.device
    return top_k_top_p_min_p_sampling_from_probs_torch(
        probs=probs,
        top_ks=torch.tensor(top_ks, device=device, dtype=torch.int32),
        top_ps=torch.ones(batch, device=device, dtype=torch.float32),
        min_ps=torch.zeros(batch, device=device, dtype=torch.float32),
        need_min_p_sampling=False,
        sampling_seed=torch.tensor(seeds, device=device, dtype=torch.int64),
        positions=torch.tensor(positions, device=device, dtype=torch.int64),
    )


class TestDeterministicVocabIdNoise(CustomTestCase):
    def test_seeded_row_unaffected_by_batchmate_filtering(self):
        """A seeded unfiltered row must pick the same token when a batchmate
        triggers the top-k sort path."""
        torch.manual_seed(0)
        seeded_probs = torch.softmax(
            torch.randn(1, VOCAB, device=DEVICE) * LOGIT_SCALE, dim=-1
        )
        mate_probs = torch.softmax(torch.randn(1, VOCAB, device=DEVICE) * 3.0, dim=-1)
        seed, position = 4321, 6
        solo = _seeded_simple(seeded_probs, seed=seed, position=position)
        mixed = _seeded_complex(
            probs=torch.cat([seeded_probs, mate_probs], dim=0),
            top_ks=[VOCAB, 2],
            seeds=[seed, seed + 1],
            positions=[position, position + 5],
        )
        self.assertEqual(int(mixed[0].item()), solo)

    def test_noop_topk_matches_simple_path(self):
        """top_k equal to vocab size masks nothing, so the complex path must
        still match the simple path."""
        torch.manual_seed(1)
        probs = torch.softmax(
            torch.randn(1, VOCAB, device=DEVICE) * LOGIT_SCALE, dim=-1
        )
        seed, position = 12345, 4
        simple = _seeded_simple(probs, seed=seed, position=position)
        complex_tok = int(
            _seeded_complex(
                probs=probs,
                top_ks=[VOCAB],
                seeds=[seed],
                positions=[position],
            ).item()
        )
        self.assertEqual(complex_tok, simple)

    def test_seeded_topk_never_samples_masked_token(self):
        """Tokens zeroed by top-k must stay unreachable under seeded Gumbel-max."""
        logits = torch.zeros(1, VOCAB, device=DEVICE)
        logits[0, 7] = 5.0
        logits[0, 3] = 4.0
        logits[0, 11] = 3.0
        tok = int(
            _seeded_complex(
                probs=torch.softmax(logits, dim=-1),
                top_ks=[2],
                seeds=[0],
                positions=[0],
            ).item()
        )
        self.assertIn(tok, {3, 7})


if __name__ == "__main__":
    unittest.main()
