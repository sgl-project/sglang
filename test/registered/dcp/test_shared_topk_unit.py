# Copyright 2023-2026 SGLang Team
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

import torch

from sglang.srt.layers.dcp.shared_topk import (
    pack_owner_candidates,
    stable_topk_from_candidates,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSharedTopKUnit(unittest.TestCase):
    def test_owner_candidates_merge_matches_global_topk(self):
        owner_scores = (
            torch.tensor(
                [[1.0, 9.0, 3.0, 7.0], [5.0, 4.0, 3.0, 2.0]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[8.0, 2.0, 6.0, 4.0], [1.0, 7.0, 8.0, 6.0]],
                dtype=torch.float32,
            ),
        )
        candidates = []
        for rank, scores in enumerate(owner_scores):
            local_indices = torch.topk(scores, 3, dim=1).indices.to(torch.int32)
            candidates.append(
                pack_owner_candidates(
                    scores,
                    local_indices,
                    dcp_rank=rank,
                    dcp_size=2,
                )
            )

        actual = stable_topk_from_candidates(torch.cat(candidates, dim=1), 3)
        expected = torch.tensor([[2, 1, 6], [5, 3, 7]], dtype=torch.int32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_stable_tie_break_prefers_lower_global_id(self):
        candidates = torch.empty((1, 4, 2), dtype=torch.float32)
        candidates[..., 0] = 1.0
        candidates[..., 1] = torch.tensor([[9, 3, 7, 1]], dtype=torch.int32).view(
            torch.float32
        )

        actual = stable_topk_from_candidates(candidates, 3)
        expected = torch.tensor([[1, 3, 7]], dtype=torch.int32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_candidate_ids_do_not_round_through_fp32(self):
        local_indices = torch.tensor([[16_777_217]], dtype=torch.int32)
        candidates = pack_owner_candidates(
            torch.tensor([[1.0]], dtype=torch.float32),
            local_indices,
            dcp_rank=0,
            dcp_size=1,
        )
        recovered = candidates[..., 1].contiguous().view(torch.int32)
        torch.testing.assert_close(recovered, local_indices, rtol=0, atol=0)

    def test_ragged_row_starts_preserve_request_local_ids(self):
        row_starts = torch.tensor([1, 2], dtype=torch.int32)
        owner_logits = (
            torch.tensor(
                [
                    [-99.0, 4.0, 8.0, -99.0],
                    [-99.0, -99.0, 7.0, 1.0],
                ],
                dtype=torch.float32,
            ),
            torch.tensor(
                [
                    [-99.0, 9.0, 2.0, -99.0],
                    [-99.0, -99.0, 6.0, 10.0],
                ],
                dtype=torch.float32,
            ),
        )
        owner_indices = (
            torch.tensor([[1, 0], [0, 1]], dtype=torch.int32),
            torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
        )
        candidates = [
            pack_owner_candidates(
                logits,
                indices,
                dcp_rank=rank,
                dcp_size=2,
                row_starts=row_starts,
            )
            for rank, (logits, indices) in enumerate(
                zip(owner_logits, owner_indices, strict=True)
            )
        ]

        actual = stable_topk_from_candidates(torch.cat(candidates, dim=1), 2)
        expected = torch.tensor([[1, 2], [3, 0]], dtype=torch.int32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_invalid_candidates_stay_padded(self):
        logits = torch.tensor([[2.0]], dtype=torch.float32)
        local_indices = torch.tensor([[0, -1, -1]], dtype=torch.int32)
        candidates = pack_owner_candidates(
            logits,
            local_indices,
            dcp_rank=1,
            dcp_size=2,
        )
        actual = stable_topk_from_candidates(candidates, 3)
        expected = torch.tensor([[1, -1, -1]], dtype=torch.int32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
