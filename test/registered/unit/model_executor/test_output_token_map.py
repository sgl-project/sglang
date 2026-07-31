# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.model_executor.output_token_map import (
    apply_projected_vocab_mask,
    map_output_token_indices,
    project_vocab_tensor,
    validate_output_token_ids,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestOutputTokenMap(unittest.TestCase):
    def test_validate_output_token_ids(self):
        actual = validate_output_token_ids([5, 1, 9], vocab_size=10)
        torch.testing.assert_close(actual, torch.tensor([5, 1, 9]))

        for invalid in ([], [-1], [10], [1, 1], [[1, 2]]):
            with self.assertRaises(ValueError):
                validate_output_token_ids(invalid, vocab_size=10)

    def test_project_vocab_tensor(self):
        tensor = torch.arange(20).reshape(2, 10)
        output_token_ids = torch.tensor([7, 2, 9])
        expected = torch.tensor([[7, 2, 9], [17, 12, 19]])
        torch.testing.assert_close(
            project_vocab_tensor(tensor, output_token_ids), expected
        )

    def test_apply_projected_vocab_mask(self):
        logits = torch.zeros(2, 3)
        output_token_ids = torch.tensor([1, 33, 63])
        vocab_mask = torch.tensor(
            [[2, 2], [0, -2147483646]],
            dtype=torch.int32,
        )

        apply_projected_vocab_mask(logits, vocab_mask, output_token_ids)

        self.assertTrue(torch.isfinite(logits[0, 0]))
        self.assertTrue(torch.isfinite(logits[0, 1]))
        self.assertTrue(torch.isneginf(logits[0, 2]))
        self.assertTrue(torch.isneginf(logits[1, 0]))
        self.assertTrue(torch.isfinite(logits[1, 1]))
        self.assertTrue(torch.isfinite(logits[1, 2]))

    def test_map_output_token_indices(self):
        output_token_ids = torch.tensor([10, 30, 20])
        actual = map_output_token_indices(
            [torch.tensor([2, 0, -1]), [1, None]],
            output_token_ids,
        )
        torch.testing.assert_close(actual[0], torch.tensor([20, 10, -1]))
        self.assertEqual(actual[1], [30, None])

    def test_sampling_biases_are_projected(self):
        info = SamplingBatchInfo(
            temperatures=torch.ones(1, 1),
            top_ps=torch.ones(1),
            top_ks=torch.full((1,), TOP_K_ALL, dtype=torch.int32),
            min_ps=torch.zeros(1),
            is_all_greedy=True,
            is_any_greedy=True,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            vocab_size=5,
            device="cpu",
            penalizer_orchestrator=MagicMock(is_required=False),
            acc_additive_penalties=torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]]),
            acc_scaling_penalties=None,
            logit_bias=torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0]]),
        )
        logits = torch.zeros(1, 2)

        info.apply_logits_bias(logits, torch.tensor([3, 1]))

        torch.testing.assert_close(logits, torch.tensor([[43.0, 21.0]]))

    def test_compact_softmax_matches_conditioned_full_softmax(self):
        torch.manual_seed(20260729)
        full_logits = torch.randn(1000, 128)
        output_token_ids = torch.randperm(128)[:37]
        compact_logits = project_vocab_tensor(full_logits, output_token_ids)

        full_probs = torch.softmax(full_logits, dim=-1)
        conditioned_probs = full_probs.index_select(-1, output_token_ids)
        conditioned_probs /= conditioned_probs.sum(dim=-1, keepdim=True)
        compact_probs = torch.softmax(compact_logits, dim=-1)

        torch.testing.assert_close(
            compact_probs,
            conditioned_probs,
            rtol=1e-5,
            atol=1e-7,
        )
        torch.testing.assert_close(
            compact_logits.argmax(dim=-1),
            full_logits.index_select(-1, output_token_ids).argmax(dim=-1),
        )


if __name__ == "__main__":
    unittest.main()
