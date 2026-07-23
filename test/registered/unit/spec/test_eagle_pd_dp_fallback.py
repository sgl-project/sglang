import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.eagle_worker_v2 import (
    EAGLEWorkerV2,
    _slice_draft_output_to_local_tokens,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


class TestEaglePDDPFallback(CustomTestCase):
    def test_seedless_pd_draft_requests_rank_consistent_eager_forward(self):
        worker = object.__new__(EAGLEWorkerV2)
        worker._draft_worker = SimpleNamespace(seed_dsa_topk_from_draft_extend=True)

        for seed, future_seed, expect_eager in (
            (None, False, True),
            (torch.ones((1, 1)), False, False),
            (None, True, False),
        ):
            with self.subTest(
                seed_present=seed is not None,
                future_seed=future_seed,
            ):
                batch = SimpleNamespace(
                    spec_info=SimpleNamespace(
                        dsa_topk_indices=seed,
                        future_dsa_topk_indices_available=future_seed,
                    )
                )
                self.assertEqual(
                    worker.requires_dp_attention_eager_forward(batch),
                    expect_eager,
                )

        worker._draft_worker.seed_dsa_topk_from_draft_extend = False
        self.assertFalse(
            worker.requires_dp_attention_eager_forward(
                SimpleNamespace(spec_info=SimpleNamespace(dsa_topk_indices=None))
            )
        )

    def test_eager_draft_discards_dp_padding_rows(self):
        logits = torch.arange(24, dtype=torch.float32).reshape(3, 8)
        hidden_states = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        positions = torch.tensor([7, 100, 100])

        local_logits, local_hidden_states, local_positions = (
            _slice_draft_output_to_local_tokens(
                logits,
                hidden_states,
                positions,
                num_local_tokens=1,
            )
        )

        self.assertEqual(local_logits.shape, (1, 8))
        self.assertEqual(local_hidden_states.shape, (1, 4))
        self.assertEqual(local_positions.tolist(), [7])
        local_positions.add_(1)
        self.assertEqual(positions.tolist(), [8, 100, 100])

    def test_idle_eager_draft_discards_all_dp_padding_rows(self):
        logits = torch.empty((2, 8))
        hidden_states = torch.empty((2, 4))
        positions = torch.tensor([100, 100])

        local_logits, local_hidden_states, local_positions = (
            _slice_draft_output_to_local_tokens(
                logits,
                hidden_states,
                positions,
                num_local_tokens=0,
            )
        )

        self.assertEqual(local_logits.shape, (0, 8))
        self.assertEqual(local_hidden_states.shape, (0, 4))
        self.assertEqual(local_positions.shape, (0,))

    def test_eager_draft_rejects_missing_local_rows(self):
        with self.assertRaisesRegex(RuntimeError, "next_token_logits has 0 rows"):
            _slice_draft_output_to_local_tokens(
                torch.empty((0, 8)),
                torch.empty((1, 4)),
                torch.tensor([7]),
                num_local_tokens=1,
            )


if __name__ == "__main__":
    unittest.main()
