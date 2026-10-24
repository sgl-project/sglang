# Copyright 2026 SGLang Team
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
"""Selection replay preserves native values and owns every request's history."""

import unittest
from types import MethodType, SimpleNamespace

import torch

from sglang.srt.beam_search.beam_group import BeamGroup
from sglang.srt.beam_search.coordinator import BeamCoordinator
from sglang.srt.beam_search.selection_cuda_graph import TrieSelectionGraphCache
from sglang.srt.beam_search.trie_config import TrieOutputHeadConfig
from sglang.srt.beam_search.trie_constraint import BeamTrieConstraint, _SparseLevel2
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestBeamSelection(CustomTestCase):
    @staticmethod
    def constraint():
        size = 128
        keys = (
            torch.arange(8, device="cuda", dtype=torch.int32)[:, None] * size
            + torch.arange(8, device="cuda", dtype=torch.int32)
        ).reshape(-1)
        sparse = _SparseLevel2(
            keys=keys,
            offsets=torch.arange(65, device="cuda", dtype=torch.int32) * 8,
            values=torch.arange(8, device="cuda", dtype=torch.int16).repeat(64),
            max_children=8,
        )
        mask = torch.zeros((size, size), device="cuda", dtype=torch.bool)
        mask[:8, :8] = True
        return BeamTrieConstraint(
            TrieOutputHeadConfig("test", 16, size, 3),
            torch.arange(size, device="cuda") < 8,
            mask,
            None,
            sparse,
            (8, 64, 512),
        )

    @staticmethod
    def group(width, depth, cumulative, shift):
        result = BeamGroup(
            beam_width=width, max_new_tokens=3, device="cuda", device_history=True
        )
        roots = (torch.arange(width, device="cuda", dtype=torch.int64) + shift) % 8
        result.num_generated = depth
        result.trie_prefix_codes = (
            roots[:, None]
            if depth == 1
            else torch.stack((roots, (roots + shift) % 8), dim=1)
        )
        result.frontier_cum_logprobs = cumulative.clone()
        return result

    def test_replay_matches_native_and_does_not_overwrite_prior_requests(self):
        torch.manual_seed(13)
        width = 32
        coordinator = SimpleNamespace(trie_constraint=self.constraint())
        coordinator._group_topk_logprobs = MethodType(
            BeamCoordinator._group_topk_logprobs, coordinator
        )
        coordinator._select_group = MethodType(
            BeamCoordinator._select_group, coordinator
        )
        for normalized in (False, True):
            with (
                envs.SGLANG_BEAM_SHARED_CONTEXT_WIDTH.override(width),
                envs.SGLANG_OPT_BEAM_TRIE_TOPK_LOGPROB.override(not normalized),
            ):
                cache = TrieSelectionGraphCache()
                for depth in (1, 2):
                    retained = []
                    for shift in (0, 1, 0):
                        pieces = [
                            torch.randn((1, 128), device="cuda"),
                            torch.randn((width - 1, 128), device="cuda"),
                        ]
                        normalizers = (
                            [
                                torch.randn((len(part), 2), device="cuda")
                                for part in pieces
                            ]
                            if normalized
                            else None
                        )
                        cumulative = torch.randn(width, device="cuda")
                        expected = self.group(width, depth, cumulative, shift)
                        actual = self.group(width, depth, cumulative, shift)
                        values, tokens = coordinator._group_topk_logprobs(
                            expected, pieces, normalizers
                        )
                        coordinator._select_group(expected, values, tokens, 7)
                        self.assertIsNotNone(
                            cache.select(coordinator, actual, pieces, normalizers, 7)
                        )
                        torch.cuda.synchronize()
                        left, right = (
                            expected._pending_steps[-1][1],
                            actual._pending_steps[-1][1],
                        )
                        for field in type(left).__struct_fields__:
                            self.assertTrue(
                                torch.equal(
                                    getattr(left, field), getattr(right, field)
                                ),
                                field,
                            )
                        self.assertTrue(
                            torch.equal(
                                expected.trie_prefix_codes, actual.trie_prefix_codes
                            )
                        )
                        for prior, fields, prefix in retained:
                            for field, snapshot in fields.items():
                                self.assertTrue(
                                    torch.equal(
                                        getattr(prior._pending_steps[-1][1], field),
                                        snapshot,
                                    )
                                )
                            self.assertTrue(
                                torch.equal(prior.trie_prefix_codes, prefix)
                            )
                        retained.append(
                            (
                                actual,
                                {
                                    field: getattr(right, field).clone()
                                    for field in type(right).__struct_fields__
                                },
                                actual.trie_prefix_codes.clone(),
                            )
                        )
                self.assertEqual(cache.captures, 2)
                self.assertEqual(cache.replays, 6)

    def test_fused_sparse_mask_and_fallback_match_native_supported_layout(self):
        constraint = self.constraint()
        prefix = torch.tensor([[0, 0], [1, 3], [7, 7]], device="cuda")
        logits = torch.randn((128, 3), device="cuda").T
        from sglang.kernels.ops.grammar.trie_mask import mask_sparse_logits

        sparse = constraint.level2_sparse
        rows = constraint._sparse_level2_rows(prefix)
        reference = mask_sparse_logits(
            logits.contiguous(), rows, sparse.offsets, sparse.values, 8
        )
        actual, fallback = constraint._mask_sparse_level2_logits(logits, prefix)
        self.assertTrue(torch.equal(reference, actual))
        self.assertTrue(
            torch.equal(
                fallback, sparse.values[sparse.offsets[rows].long()].long()[:, None]
            )
        )


if __name__ == "__main__":
    unittest.main()
