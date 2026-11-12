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

"""Replayed trie selection graph against the eager selection path."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.beam_search import BeamGroup
from sglang.srt.beam_search.coordinator import BeamCoordinator
from sglang.srt.beam_search.selection_cuda_graph import TrieSelectionGraphCache
from sglang.srt.beam_search.trie_config import TrieOutputHeadConfig
from sglang.srt.beam_search.trie_constraint import BeamTrieConstraint
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

CODEBOOK_SIZE = 16
NUM_CODEBOOKS = 3
TOKEN_START = 4
VOCAB_SIZE = 64


def _random_mask(rows: int, generator: torch.Generator) -> torch.Tensor:
    # At least 10 valid children per row, so every beam width tested has enough
    # candidates at every depth.
    scores = torch.rand(rows, CODEBOOK_SIZE, generator=generator)
    return scores >= scores.sort(dim=1, descending=True).values[:, 9:10]


def _make_constraint() -> BeamTrieConstraint:
    generator = torch.Generator().manual_seed(0)
    config = TrieOutputHeadConfig(
        tensor_path="",
        token_start=TOKEN_START,
        codebook_size=CODEBOOK_SIZE,
        num_codebooks=NUM_CODEBOOKS,
    )
    return BeamTrieConstraint(
        config,
        root_mask=_random_mask(1, generator)[0].cuda(),
        level1_masks=_random_mask(CODEBOOK_SIZE, generator).cuda(),
        level2_masks=_random_mask(CODEBOOK_SIZE**2, generator).cuda(),
        level2_sparse=None,
        level_cardinalities=(10, 10, 10),
    )


def _make_coordinator(constraint: BeamTrieConstraint) -> BeamCoordinator:
    coordinator = BeamCoordinator(
        model_config=SimpleNamespace(beam_trie_config=None),
        spec_algorithm=None,
        dllm_enabled=False,
        max_req_len=0,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        tree_cache=None,
        future_map=None,
    )
    coordinator.trie_constraint = constraint
    return coordinator


def _make_group(
    constraint: BeamTrieConstraint, beam_width: int, depth: int, seed: int
) -> BeamGroup:
    generator = torch.Generator().manual_seed(seed)
    group = BeamGroup(
        beam_width=beam_width,
        max_new_tokens=NUM_CODEBOOKS,
        device="cuda",
        device_history=True,
    )
    group.num_generated = depth
    group.frontier_cum_logprobs = (
        -torch.rand(beam_width, generator=generator).sort().values.cuda()
    )
    prefix = torch.empty((beam_width, 0), dtype=torch.int64)
    for level in range(depth):
        if level == 0:
            parent_mask = constraint.root_mask.cpu().expand(beam_width, -1)
        else:
            parent_mask = constraint.valid_child_mask(prefix.cuda(), level).cpu()
        choice = torch.multinomial(parent_mask.float(), 1, generator=generator)
        prefix = torch.cat([prefix, choice], dim=1)
    group.trie_prefix_codes = prefix.cuda()
    return group


def _make_pieces(beam_width: int, depth: int, seed: int):
    """Compact codebook logits and their full-vocabulary normalizers, as the
    compact output head emits them."""
    generator = torch.Generator().manual_seed(seed)
    token_start = TOKEN_START + depth * CODEBOOK_SIZE
    pieces, normalizers = [], []
    for rows in (1, beam_width - 1):
        full = torch.randn(rows, VOCAB_SIZE, generator=generator)
        row_max = full.max(dim=-1).values
        log_sum = torch.log(torch.exp(full - row_max[:, None]).sum(dim=-1))
        pieces.append(full[:, token_start : token_start + CODEBOOK_SIZE].cuda())
        normalizers.append(torch.stack([row_max, log_sum], dim=-1).cuda())
    return pieces, normalizers


class TestTrieSelectionGraph(CustomTestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_replay_matches_eager_selection(self):
        constraint = _make_constraint()
        coordinator = _make_coordinator(constraint)
        cache = TrieSelectionGraphCache()
        expected_captures = 0
        for beam_width in (8, 4):
            for depth in (1, 2):
                expected_captures += 1
                for round_idx in range(2):
                    seed = 100 * beam_width + 10 * depth + round_idx
                    pieces, normalizers = _make_pieces(beam_width, depth, seed)
                    eager = _make_group(constraint, beam_width, depth, seed)
                    graphed = _make_group(constraint, beam_width, depth, seed)

                    values, tokens = coordinator.group_topk_logprobs(
                        eager, pieces, normalizers
                    )
                    eager_tokens, eager_parents = coordinator.select_group(
                        eager, values, tokens, 0
                    )
                    selected = cache.select(
                        coordinator, graphed, pieces, normalizers, 0
                    )
                    self.assertIsNotNone(selected)
                    graph_tokens, graph_parents = selected

                    label = f"width {beam_width} depth {depth} round {round_idx}"
                    self.assertTrue(torch.equal(eager_tokens, graph_tokens), label)
                    if eager_parents is None:
                        self.assertIsNone(graph_parents, label)
                    else:
                        self.assertTrue(
                            torch.equal(eager_parents, graph_parents), label
                        )
                        self.assertTrue(
                            torch.equal(
                                eager.trie_prefix_codes, graphed.trie_prefix_codes
                            ),
                            label,
                        )
                        self.assertTrue(
                            torch.equal(
                                eager.frontier_cum_logprobs,
                                graphed.frontier_cum_logprobs,
                            ),
                            label,
                        )
                    self.assertEqual(eager.num_generated, graphed.num_generated)
        self.assertEqual(cache.captures, expected_captures)
        self.assertEqual(cache.replays, 2 * expected_captures)


if __name__ == "__main__":
    unittest.main()
