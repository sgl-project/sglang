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
"""Replay trie selection math while retaining request-owned active state."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Optional, Sequence

import torch

from sglang.srt.beam_search.beam_group import BeamGroup
from sglang.srt.beam_search.joint_select import FinalSelect, SelectResult
from sglang.srt.beam_search.shared_context_attention import graph_beam_width
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.beam_search.coordinator import BeamCoordinator


class _CapturedSelection:
    def __init__(
        self,
        coordinator: BeamCoordinator,
        group: BeamGroup,
        pieces: Sequence[torch.Tensor],
        normalizers: Optional[Sequence[torch.Tensor]],
    ):
        self.final = group.next_step_is_final()
        depth = group.num_generated
        self.pieces = [part.clone() for part in pieces]
        self.normalizers = (
            None if normalizers is None else [part.clone() for part in normalizers]
        )
        self.prefix = group.trie_prefix_codes.clone()
        self.cumulative = group.frontier_cum_logprobs.clone()
        scratch = BeamGroup(
            beam_width=group.beam_width,
            max_new_tokens=group.max_new_tokens,
            device=pieces[0].device,
            device_history=True,
        )

        def execute():
            scratch.num_generated = depth
            scratch.frontier_cum_logprobs = self.cumulative
            scratch.trie_prefix_codes = self.prefix
            scratch._pending_steps.clear()
            values, tokens = coordinator._group_topk_logprobs(
                scratch, self.pieces, self.normalizers
            )
            coordinator._select_group(scratch, values, tokens, 0)
            self.selection = scratch._pending_steps[-1][1]
            self.next_prefix = scratch.trie_prefix_codes

        caller = torch.cuda.current_stream()
        side = torch.cuda.Stream()
        side.wait_stream(caller)
        with torch.cuda.stream(side):
            execute()
            execute()
        caller.wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=side):
            execute()
        caller.wait_stream(side)

    def replay(self, group, pieces, normalizers, tick):
        from sglang.kernels.ops.sampling.beam_selection import snapshot_selection

        for target, source in zip(self.pieces, pieces, strict=True):
            target.copy_(source)
        if self.normalizers is not None:
            for target, source in zip(self.normalizers, normalizers, strict=True):
                target.copy_(source)
        self.prefix.copy_(group.trie_prefix_codes)
        self.cumulative.copy_(group.frontier_cum_logprobs)
        self.graph.replay()
        result = self.selection
        ids, scores, prefix = snapshot_selection(
            result.tokens if self.final else result.next_tokens,
            result.parent_idx,
            result.cum_logprobs if self.final else result.new_cum_logprobs,
            None if self.final else self.next_prefix,
        )
        if self.final:
            group.advance_final_frontier(
                FinalSelect(tokens=ids[0], parent_idx=ids[1], cum_logprobs=scores),
                tick,
            )
            return ids[0], None
        # No-stop trie groups ignore finished outputs. These arrays remain
        # immutable zeros; active state above is independently owned.
        group.advance_frontier(
            SelectResult(
                next_tokens=ids[0],
                parent_idx=ids[1],
                new_cum_logprobs=scores,
                num_survivors=result.num_survivors,
                fin_tokens=result.fin_tokens,
                fin_parent_idx=result.fin_parent_idx,
                fin_cum_logprobs=result.fin_cum_logprobs,
                num_finished=result.num_finished,
            ),
            tick,
        )
        group.trie_prefix_codes = prefix
        return ids[0], ids[1]


class TrieSelectionGraphCache:
    def __init__(self):
        self.graphs = OrderedDict()
        self.signature = None
        self.captures = 0
        self.replays = 0
        self.resets = 0

    def select(self, coordinator, group, pieces, normalizers, tick):
        constraint = coordinator.trie_constraint
        if (
            constraint is None
            or not group.device_history
            or group.stop_token_ids.numel()
            or group.beam_width != graph_beam_width()
            or group.num_generated <= 0
            or group.num_generated >= group.max_new_tokens
            or len(pieces) != 2
            or not all(part.is_cuda for part in pieces)
            or pieces[0].shape != (1, constraint.config.codebook_size)
            or pieces[1].shape
            != (group.beam_width - 1, constraint.config.codebook_size)
        ):
            return None
        approximate = envs.SGLANG_OPT_BEAM_TRIE_TOPK_LOGPROB.get()
        if (normalizers is None) != approximate:
            raise ValueError("Trie score policy changed without recapturing the model")

        def layout(tensor):
            return (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)

        def table(tensor):
            return (tensor.data_ptr(), *layout(tensor))

        sparse = constraint.level2_sparse
        signature = (
            id(constraint),
            graph_beam_width(),
            approximate,
            constraint.config.token_start,
            constraint.config.codebook_size,
            constraint.config.num_codebooks,
            table(constraint.root_mask),
            table(constraint.level1_masks),
            table(constraint.level2_masks)
            if constraint.level2_masks is not None
            else None,
            (
                sparse.max_children,
                *(table(part) for part in (sparse.keys, sparse.offsets, sparse.values)),
            )
            if sparse is not None
            else None,
        )
        if self.signature != signature:
            self.graphs.clear()
            self.signature = signature
            self.resets += 1
        key = (
            pieces[0].device,
            torch.cuda.current_stream().cuda_stream,
            group.num_generated,
            group.max_new_tokens,
            tuple(layout(part) for part in pieces),
            None
            if normalizers is None
            else tuple(layout(part) for part in normalizers),
            layout(group.trie_prefix_codes),
            layout(group.frontier_cum_logprobs),
        )
        if key not in self.graphs:
            if len(self.graphs) >= 4:
                self.graphs.popitem(last=False)
            self.graphs[key] = _CapturedSelection(
                coordinator, group, pieces, normalizers
            )
            self.captures += 1
        captured = self.graphs[key]
        self.graphs.move_to_end(key)
        selected = captured.replay(group, pieces, normalizers, tick)
        self.replays += 1
        return selected
