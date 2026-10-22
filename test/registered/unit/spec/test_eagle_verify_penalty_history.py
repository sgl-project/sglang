"""Penalty history must advance after every committed speculative token."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative import eagle_utils
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestVerifyPenaltyHistory(CustomTestCase):
    def _verify(self, additive=False, scaling=False, unpenalized_neighbor=False):
        # All three rows agree with the proposed chain. Without penalties the
        # target accepts two drafts and emits a bonus; with penalties only its
        # first distribution has the complete history.
        device = "cuda"
        bs = 2 if unpenalized_neighbor else 1
        params = [
            SimpleNamespace(
                frequency_penalty=0.5 if additive else 0.0,
                presence_penalty=0.0,
                repetition_penalty=2.0 if scaling else 1.0,
                min_new_tokens=0,
            )
        ]
        if unpenalized_neighbor:
            params.append(
                SimpleNamespace(
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    repetition_penalty=1.0,
                    min_new_tokens=0,
                )
            )
        info = SimpleNamespace(
            is_all_greedy=True,
            acc_additive_penalties=(
                torch.tensor(
                    [[-p.frequency_penalty] * 4 for p in params], device=device
                )
                if additive
                else None
            ),
            acc_scaling_penalties=(
                torch.tensor(
                    [[p.repetition_penalty] * 4 for p in params], device=device
                )
                if scaling
                else None
            ),
            logit_bias=None,
        )
        batch = SimpleNamespace(
            device=device,
            reqs=[SimpleNamespace(sampling_params=p, output_ids=[]) for p in params],
            seq_lens=torch.tensor([4] * bs, device=device),
            sampling_info=info,
            forward_mode=SimpleNamespace(is_idle=lambda: False),
        )
        verify = SimpleNamespace(
            draft_token=torch.tensor([0, 1, 2] * bs, dtype=torch.int64, device=device),
            draft_token_num=3,
            max_tree_depth=3,
            tree_topk=1,
            retrieve_index=torch.arange(3 * bs, device=device).reshape(bs, 3),
            retrieve_next_token=torch.tensor([[1, 2, -1]] * bs, device=device),
            retrieve_next_sibling=torch.tensor([[-1, -1, -1]] * bs, device=device),
        )
        logits = SimpleNamespace(
            next_token_logits=torch.tensor(
                [[0.0, 5.0, 0.0, 0.0], [0.0, 0.0, 5.0, 0.0], [0.0, 0.0, 0.0, 5.0]] * bs,
                device=device,
            )
        )
        spec = SimpleNamespace(speculative_use_rejection_sampling=False)
        with patch.object(eagle_utils, "get_spec", return_value=spec):
            predict, lengths, indices = eagle_utils.eagle_sample(verify, batch, logits)
        self.assertEqual(predict[indices[0, 0]].item(), 1)
        return lengths.tolist(), indices.tolist()

    def test_unpenalized_chain_retains_acceleration(self):
        lengths, indices = self._verify()
        self.assertEqual(lengths, [3])
        self.assertEqual(indices, [[0, 1, 2]])

    def test_additive_penalties_commit_only_the_root(self):
        lengths, indices = self._verify(additive=True)
        self.assertEqual(lengths, [1])
        self.assertEqual(indices, [[0, -1, -1]])

    def test_scaling_penalties_commit_only_the_root(self):
        lengths, indices = self._verify(scaling=True)
        self.assertEqual(lengths, [1])
        self.assertEqual(indices, [[0, -1, -1]])

    def test_mixed_batch_keeps_unpenalized_drafts(self):
        lengths, indices = self._verify(additive=True, unpenalized_neighbor=True)
        self.assertEqual(lengths, [1, 3])
        self.assertEqual(indices, [[0, -1, -1], [3, 4, 5]])

    def test_combined_penalties_commit_only_the_root(self):
        lengths, indices = self._verify(additive=True, scaling=True)
        self.assertEqual(lengths, [1])
        self.assertEqual(indices, [[0, -1, -1]])


if __name__ == "__main__":
    unittest.main()
