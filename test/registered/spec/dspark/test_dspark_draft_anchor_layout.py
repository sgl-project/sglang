import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockProposer,
    DsparkDraftSampler,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSparkDraftAnchorLayout(CustomTestCase):
    def test_draft_forward_runs_anchor_plus_gamma_and_crops_anchor_hidden(self):
        seen = {}
        bs = 2
        gamma = 4
        draft_width = gamma + 1
        hidden_size = 8

        class FakeDraftRunner:
            device = "cpu"

            def forward(self, forward_batch):
                seen["input_ids"] = forward_batch.input_ids.clone()
                seen["positions"] = forward_batch.positions.clone()
                seen["out_cache_loc"] = forward_batch.out_cache_loc.clone()
                hidden = torch.arange(
                    bs * draft_width * hidden_size, dtype=torch.float32
                ).view(bs * draft_width, hidden_size)
                return SimpleNamespace(
                    logits_output=SimpleNamespace(hidden_states=hidden),
                    can_run_graph=False,
                )

        proposer = DraftBlockProposer(
            draft_model=SimpleNamespace(),
            draft_model_runner=FakeDraftRunner(),
            gamma=gamma,
            mask_token_id=0,
            draft_block_spec_info=SimpleNamespace(),
        )
        batch = SimpleNamespace(
            seq_lens=torch.tensor([10, 20], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([10, 20], dtype=torch.int32),
            seq_lens_sum=30,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
        )
        draft_input = SimpleNamespace(
            bonus_tokens=torch.tensor([7, 8], dtype=torch.int64),
        )
        verify_window = SimpleNamespace(
            positions_2d=torch.arange(bs * draft_width, dtype=torch.int64).view(
                bs, draft_width
            ),
            verify_cache_loc_2d=torch.arange(
                100, 100 + bs * draft_width, dtype=torch.int64
            ).view(bs, draft_width),
        )

        out = proposer._run_forward(
            batch=batch,
            draft_input=draft_input,
            verify_window=verify_window,
            bs=bs,
            device="cpu",
            embed_module=torch.nn.Embedding(16, hidden_size),
        )

        self.assertEqual(tuple(out.draft_block_ids.shape), (bs, draft_width))
        self.assertEqual(out.draft_block_ids[:, 0].tolist(), [7, 8])
        self.assertEqual(seen["input_ids"].numel(), bs * draft_width)
        self.assertEqual(seen["positions"].tolist(), list(range(bs * draft_width)))
        self.assertEqual(
            seen["out_cache_loc"].tolist(),
            list(range(100, 100 + bs * draft_width)),
        )
        self.assertEqual(tuple(out.draft_hidden_3d.shape), (bs, gamma, hidden_size))
        self.assertEqual(tuple(out.raw_hidden.shape), (bs * gamma, hidden_size))
        self.assertEqual(out.raw_hidden[0].tolist(), list(range(hidden_size, 16)))

    def test_folded_sampler_requires_full_anchor_plus_gamma_blocks(self):
        class FakeModel:
            def __init__(self):
                self.markov_head = SimpleNamespace()

            def compute_base_logits(self, hidden_states):
                vocab_size = 16
                logits = torch.zeros((hidden_states.shape[0], vocab_size))
                return logits, None

        sampler = DsparkDraftSampler(
            model=FakeModel(),
            gamma=4,
            max_bs=2,
            device=torch.device("cpu"),
        )

        with self.assertRaisesRegex(RuntimeError, "anchor \\+ gamma"):
            sampler(torch.empty((8, 8)), torch.empty((8,), dtype=torch.int64))


if __name__ == "__main__":
    unittest.main()
