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
"""Scheduler wiring for beam search (columnar member-row architecture).

A beam_width=k request runs as one leader Req plus k-1 bare member rows:
physical req_to_token rows tracked columnarly on the group, with no Req
object behind them. The member rows are appended to the decode batch's row
tensors just before allocation (batch_tail.append_beam_tail) and sliced
off the logits before sampling (tp_worker), so the reqs-aligned world never
sees them. Hooks: admission (validate_and_init), selection at the forward's
relay point, and lifecycle (member spawn at the leader's prefill relay,
finalize at group finish).

All rows decode in lockstep over [prompt_len, leader_allocated), but
share-on-fork lets several of them reference the same slot, so the group --
not the individual row -- owns that region: it is freed once, deduped, at
group finish.

Every tick splits in two, and the whole file follows the naming. select_* is
the launch half (tensor-side, no D2H) and must run before the next forward
resolves its inputs, so the relayed tokens and reparented KV are the selected
ones. commit_* is the deferred half (DAG build, finish/abort); under overlap it
lags one forward and discards overshoot steps. Sync callers run both within one
tick. Commits are tick-gated; BeamGroup.commit_pending documents why.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Sequence

import msgspec
import torch

from sglang.srt.beam_search.beam_group import BeamGroup, BeamGroupState
from sglang.srt.beam_search.fork import (
    MEMBER_LENGTH_MARGIN,
    StagedOrphans,
    alias_members_prompt_kv,
    collect_orphan_slots,
    free_member_rows,
    neutral_member_sampling_params,
    remap_kv_mapping,
)
from sglang.srt.beam_search.joint_select import joint_select, select_final_topk
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.managers.overlap_utils import FutureMap, RelayPayload
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    FINISH_MATCHED_TOKEN,
    Req,
    ScheduleBatch,
)
from sglang.srt.runtime_context import (
    get_disagg,
    get_memory,
    get_parallel,
    get_schedule,
)

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


def _rows_topk_logprobs(pieces: Sequence[torch.Tensor], num_candidates: int):
    # One logsumexp per piece: it is not batch-invariant on CUDA, and folding
    # pieces shifts the leader's lse by an ulp, flipping near-equal beams.
    vals, toks = [], []
    for piece in pieces:
        if piece.shape[0] == 0:
            continue
        x = piece.float()
        # lse instead of a full [rows, vocab] log_softmax; topk order on raw
        # logits is identical under the monotone shift.
        lse = torch.logsumexp(x, dim=-1, keepdim=True)
        v, t = torch.topk(x, num_candidates, dim=-1)
        vals.append(v - lse)
        toks.append(t)
    return torch.cat(vals), torch.cat(toks)


class BeamCoordinator(msgspec.Struct, kw_only=True):
    model_config: ModelConfig
    spec_algorithm: SpeculativeAlgorithm
    dllm_enabled: bool
    max_req_len: int
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    tree_cache: BasePrefixCache
    future_map: FutureMap

    # Live (non-retired) groups; the O(1) gate for the per-forward relay hook.
    _num_live_groups: int = 0

    @staticmethod
    def request_beam_width(recv_req) -> int:
        """beam_width of an incoming request (1 = not a beam request)."""
        return getattr(recv_req.sampling_params, "beam_width", None) or 1

    def validate_and_init(self, req: Req, recv_req) -> Optional[str]:
        """Validate a beam request and attach its group; returns an error or
        None. On success the leader's row params are neutralized."""
        user_params = req.sampling_params
        beam_width = user_params.beam_width

        if not self.spec_algorithm.is_none():
            return "Beam search is not supported with speculative decoding."
        if get_disagg().disaggregation_mode != "null":
            return "Beam search is not supported with PD disaggregation."
        if get_schedule().page_size > 1:
            return "Beam search currently requires --page-size 1."
        if self.dllm_enabled:
            return "Beam search is not supported with diffusion LLM."
        if get_memory().enable_hisparse:
            return "Beam search is not supported with hisparse."
        if get_parallel().pp_size > 1:
            return "Beam search is not supported with pipeline parallelism."
        if is_dp_attention_enabled():
            # The cross-rank token sync reads batch_size() (== len(reqs)), which
            # excludes the member rows the forward actually runs.
            return "Beam search is not supported with dp attention."
        if get_memory().enable_hierarchical_cache:
            return "Beam search is not supported with hierarchical cache."
        if self.model_config.is_encoder_decoder:
            return "Beam search is not supported with encoder-decoder models."
        if self.tree_cache.supports_swa() or self.tree_cache.supports_mamba():
            return "Beam search is not supported with SWA/mamba hybrid caches."

        if req.session_id is not None or recv_req.session_params is not None:
            return "Beam search is not supported for session requests."
        if req.lora_id is not None:
            return "Beam search is not supported with LoRA."
        if recv_req.return_logprob:
            return "Beam search does not support return_logprob."
        if recv_req.return_hidden_states:
            return "Beam search does not support return_hidden_states."
        if recv_req.return_sampling_mask or recv_req.return_routed_experts:
            return "Beam search does not support sampling-mask/routed-experts returns."
        if any(
            x is not None
            for x in (
                user_params.json_schema,
                user_params.regex,
                user_params.ebnf,
                user_params.structural_tag,
            )
        ):
            return "Beam search is not supported with constrained decoding."
        if user_params.stop_strs or user_params.stop_regex_strs:
            return "Beam search does not support stop strings/regex yet; use stop_token_ids."
        if user_params.min_new_tokens > 0:
            return "Beam search does not support min_new_tokens yet."
        if user_params.n > beam_width:
            return f"n ({user_params.n}) cannot exceed beam_width ({beam_width})."
        if 2 * beam_width > self.model_config.vocab_size:
            return f"beam_width ({beam_width}) is too large for the vocabulary."
        if beam_width > self.req_to_token_pool.size:
            return (
                f"beam_width ({beam_width}) needs {beam_width} req-to-token slots "
                f"but the pool holds only {self.req_to_token_pool.size}. Reduce "
                f"beam_width or raise --max-running-requests."
            )

        # Effective generation budget: keep prompt + budget + member margin
        # within the row width so member-side length can never truncate first.
        prompt_len = len(req.origin_input_ids)
        max_new_tokens = min(
            (
                user_params.max_new_tokens
                if user_params.max_new_tokens is not None
                else 1 << 30
            ),
            self.max_req_len - prompt_len - 1 - MEMBER_LENGTH_MARGIN,
        )
        if max_new_tokens < 1:
            return (
                f"Beam search needs at least 1 generated token within the "
                f"context budget (prompt_len={prompt_len}, max_req_len={self.max_req_len})."
            )

        group = BeamGroup(
            beam_width=beam_width,
            stop_token_ids=self._collect_stop_token_ids(req, user_params),
            max_new_tokens=max_new_tokens,
            num_return=user_params.n,
            # Frontier state lives on device: selection consumes device top-2k
            # tensors in place; only k-sized results ever reach the host.
            device=self.req_to_token_pool.device,
        )
        group.leader = req
        group.prompt_len = prompt_len

        # Neutralize the leader's row params (raw log_softmax scoring, no
        # self-finish path); the user's semantics now live on the group.
        neutral = neutral_member_sampling_params(user_params)
        neutral.max_new_tokens = max_new_tokens + MEMBER_LENGTH_MARGIN
        neutral.no_stop_trim = user_params.no_stop_trim
        req.sampling_params = neutral
        req.beam_group = group
        # The leader's decode suffix is a beam path, never a tree entry; this
        # also skips the prefill-time unfinished insert.
        req.skip_radix_cache_insert = True
        self._num_live_groups += 1
        return None

    def pending_member_rows(self, batch: ScheduleBatch) -> int:
        """Rows admitted-but-not-yet-spawned groups will claim; the admission
        gate subtracts this so it never over-commits the req slot pool."""
        if self._num_live_groups == 0:
            return 0
        return sum(
            r.beam_group.beam_width - 1
            for r in batch.reqs
            if r.beam_group is not None
            and r.beam_group.member_rows is None
            and not r.beam_group.retired
            and not r.finished()
        )

    @staticmethod
    def _collect_stop_token_ids(req: Req, user_params) -> List[int]:
        if user_params.ignore_eos:
            return []
        stop_ids = set(user_params.stop_token_ids or ())
        stop_ids |= set(req.eos_token_ids or ())
        tokenizer = req.tokenizer
        if tokenizer is not None:
            if getattr(tokenizer, "eos_token_id", None) is not None:
                stop_ids.add(tokenizer.eos_token_id)
            stop_ids |= set(getattr(tokenizer, "additional_stop_token_ids", None) or ())
        return sorted(stop_ids)

    def maybe_select_and_relay(
        self, batch: ScheduleBatch, batch_result, chunked_req: Optional[Req] = None
    ) -> None:
        """Per-forward relay hook: overwrite beam rows' relayed tokens with
        joint-selected ones. O(1) when no beam group is live."""
        if self._num_live_groups == 0:
            return
        if not batch.spec_algorithm.is_none():
            return
        logits_output = batch_result.logits_output
        if logits_output is None or logits_output.next_token_logits is None:
            return
        if batch.forward_mode.is_decode():
            self.select_and_relay_decode(batch, logits_output)
        elif batch.forward_mode.is_extend():
            capture = logits_output.beam
            leader_pos = {
                row: pos
                for pos, row in enumerate(
                    capture.leader_rows if capture is not None else ()
                )
            }
            for i, req in enumerate(batch.reqs):
                group = req.beam_group
                if (
                    group is None
                    or group.state != BeamGroupState.DECODING
                    or group.num_generated > 0
                    or req is chunked_req  # mid-chunk leader: no selection yet
                    or req.is_retracted
                    or req.finished()
                ):
                    continue
                assert i in leader_pos, (
                    "beam leader prefill logits were not captured pre-sample "
                    "(worker capture_pre_sample_logits wiring)"
                )
                self.select_leader_prefill(
                    req, leader_pos[i], logits_output, tick=batch.forward_iter
                )

    def select_leader_prefill(
        self,
        req: Req,
        pos: int,
        logits_output: LogitsProcessorOutput,
        tick: int = 0,
    ) -> None:
        """Launch half of the leader's prefill tick: first selection, member-row
        spawn, and the relay overwrite (the sampled token is void)."""
        group: BeamGroup = req.beam_group
        top_logprobs, top_tokens = _rows_topk_logprobs(
            [logits_output.beam.leader_logits[pos : pos + 1]], group.num_candidates
        )
        final = group.next_step_is_final()
        next_tokens, _ = self._select_group(group, top_logprobs, top_tokens, tick)
        if final:
            # Prefill-terminated (max_new_tokens == 1): no spawn, but the relay
            # slot still needs a token so an overshoot step has a valid input.
            self._stash_next_tokens([req.req_pool_idx], next_tokens[:1])
            return
        self._spawn_member_rows(group, req)
        req.output_ids.append(0)  # length placeholder; DAG owns history
        self._stash_next_tokens(group.all_rows, next_tokens)

    def commit_prefill(self, req: Req, up_to_tick: Optional[int] = None) -> None:
        """Deferred half of the leader's prefill tick: fold the staged selection
        into the DAG (the designated D2H point) and apply finish/abort."""
        group: BeamGroup = req.beam_group
        if group.retired:
            return
        if req.to_finish is not None:
            self._abort_group(group)
            return
        self._reclaim_orphans(group, up_to_tick)
        if group.commit_pending(up_to_tick):
            self._finish_group(group)

    def _spawn_member_rows(self, group: BeamGroup, leader: Req) -> None:
        rows = self.req_to_token_pool.alloc_rows(group.beam_width - 1)
        assert rows is not None, (
            f"Beam member spawn needs {group.beam_width - 1} req-to-token slots "
            f"but only {self.req_to_token_pool.available_size()} are free; the "
            f"admission gate (get_num_allocatable_reqs) must reserve them."
        )
        device = self.req_to_token_pool.device
        member_rows = torch.tensor(rows, dtype=torch.int64, device=device)
        alias_members_prompt_kv(
            self.req_to_token_pool.req_to_token,
            member_rows,
            leader.req_pool_idx,
            group.prompt_len,
        )
        group.member_rows = member_rows
        group.member_rows_cpu = torch.tensor(rows, dtype=torch.int64)
        leader_row = torch.tensor(
            [leader.req_pool_idx], dtype=torch.int64, device=device
        )
        group.all_rows = torch.cat([leader_row, member_rows])

    def select_and_relay_decode(
        self, batch: ScheduleBatch, logits_output: LogitsProcessorOutput
    ) -> None:
        """Launch half: joint-select every group in this decode batch on device,
        reparent KV, and overwrite the relayed next tokens."""
        tail = batch.beam_tail
        if tail is None:
            return
        capture = logits_output.beam
        for gi, entry in enumerate(tail.entries):
            group = entry.group
            if group.retired or group.state != BeamGroupState.DECODING:
                continue
            top_logprobs, top_tokens = _rows_topk_logprobs(
                [
                    capture.leader_logits[gi : gi + 1],
                    capture.tail_logits[entry.start : entry.end],
                ],
                group.num_candidates,
            )
            next_tokens, parent_idx = self._select_group(
                group, top_logprobs, top_tokens, batch.forward_iter
            )
            self._apply_survivors(group, next_tokens, parent_idx, batch.forward_iter)

    def commit_decode(self, batch: ScheduleBatch) -> set:
        """Deferred half: fold staged selections into the DAG and apply
        finish/abort. Returns groups finished by THIS call -- under overlap a
        finished leader reappears for one overshoot tick, so the caller runs the
        shared finish machinery only on the committing tick."""
        newly_finished = set()
        if self._num_live_groups == 0:
            return newly_finished
        for req in batch.reqs:
            group = req.beam_group
            if group is None or group.retired:
                continue
            # Reclaim even if the DAG commit is discarded: the launch path
            # already mutated req_to_token, so the orphans are real either way.
            self._reclaim_orphans(group, batch.forward_iter)
            if req.to_finish is not None:
                self._abort_group(group)
                newly_finished.add(id(group))
                continue
            if group.commit_pending(batch.forward_iter):
                self._finish_group(group)
                newly_finished.add(id(group))
        return newly_finished

    def _select_group(
        self,
        group: BeamGroup,
        top_logprobs: torch.Tensor,
        top_tokens: torch.Tensor,
        tick: int,
    ):
        k = group.beam_width
        if group.next_step_is_final():
            # parent_idx is None on a final step: every selection finishes, so
            # no row moves onto another's slots.
            fsel = select_final_topk(
                group.frontier_cum_logprobs, top_logprobs, top_tokens, k
            )
            group.advance_final_frontier(fsel, tick)
            return fsel.tokens, None
        sel = joint_select(
            group.frontier_cum_logprobs,
            top_logprobs,
            top_tokens,
            group.stop_token_ids,
            k,
        )
        group.advance_frontier(sel, tick)
        return sel.next_tokens[:k], sel.parent_idx[:k]

    def _apply_survivors(
        self,
        group: BeamGroup,
        next_tokens: torch.Tensor,
        parent_idx: Optional[torch.Tensor],
        tick: int,
    ) -> None:
        rows = group.all_rows
        if parent_idx is not None:
            # Final steps skip this: their KV is never read again. Orphans are
            # staged, not freed here -- the set difference must not block launch.
            old_map, new_map = remap_kv_mapping(
                self.req_to_token_pool.req_to_token,
                rows=rows,
                parent_idx=parent_idx,
                prefix_len=group.prompt_len,
                # All rows are synchronized; the leader's committed length
                # covers the KV computed through this step.
                seq_len=group.leader.kv_committed_len,
            )
            group.pending_orphans.append(StagedOrphans(tick, old_map, new_map))
            # Length placeholder only; the DAG owns history and member rows
            # have no host state.
            group.leader.output_ids.append(0)
        self._stash_next_tokens(rows, next_tokens)

    def _reclaim_orphans(
        self, group: BeamGroup, up_to_tick: Optional[int] = None
    ) -> None:
        # Callers must be past the tick's copy_done sync: collect_orphan_slots
        # synchronizes, which would stall the launch path.
        if not group.pending_orphans:
            return
        if up_to_tick is None:
            # Teardown drains every staged tick.
            staged, group.pending_orphans = group.pending_orphans, []
        else:
            staged = [e for e in group.pending_orphans if e.tick <= up_to_tick]
            group.pending_orphans = [
                e for e in group.pending_orphans if e.tick > up_to_tick
            ]
        if not staged:
            return
        # Plain free(), never a nested free_group: an inner begin/end pair
        # inside the decode path's group double-frees.
        allocator = self.token_to_kv_pool_allocator
        for entry in staged:
            orphans = collect_orphan_slots(entry.old_mapping, entry.new_mapping)
            if orphans.numel():
                group.slots_freed += orphans.numel()
                allocator.free(orphans)

    def _finish_group(self, group: BeamGroup) -> None:
        # The leader carries the best sequence's finish reason.
        group.final_results = group.finalize()
        top = group.final_results[0]
        leader = group.leader
        if top.matched_token is not None:
            leader.finished_reason = FINISH_MATCHED_TOKEN(matched=top.matched_token)
        else:
            leader.finished_reason = FINISH_LENGTH(length=group.num_committed)
        self._free_member_rows(group)
        self._retire_group(group)

    def _abort_group(self, group: BeamGroup) -> None:
        group.state = BeamGroupState.FINISHED
        group.final_results = []
        leader = group.leader
        leader.finished_reason = leader.to_finish or FINISH_ABORT("Beam group aborted.")
        leader.to_finish = None
        self._free_member_rows(group)
        self._retire_group(group)

    def retire_group(self, req: Req) -> None:
        """Leader ended outside the commit path -- retracted, or aborted while
        still queued. Member rows, if any, were released by the caller."""
        group = req.beam_group
        if group is None or group.retired:
            return
        group.state = BeamGroupState.FINISHED
        group.final_results = []
        self._retire_group(group)

    def _free_member_rows(self, group: BeamGroup) -> None:
        # Staged orphans are unreachable from every row, so the group-wide
        # dedup free below would miss them.
        self._reclaim_orphans(group)
        free_member_rows(group, self.req_to_token_pool, self.token_to_kv_pool_allocator)

    def _retire_group(self, group: BeamGroup) -> None:
        # Exactly once per group, so the O(1) live gate stays accurate.
        if not group.retired:
            # Staged orphans are referenced by no row, so the retract-abort
            # path's direct fork.free_member_rows cannot see them.
            self._reclaim_orphans(group)
            group.retired = True
            # Drops overshoot selections staged after the terminal commit.
            group._pending_steps.clear()
            self._num_live_groups -= 1

    def _stash_next_tokens(self, rows, tokens) -> None:
        # Accepts GPU tensors (decode path, no D2H) or host lists (prefill).
        device = self.req_to_token_pool.device
        if not torch.is_tensor(rows):
            rows = torch.tensor(rows, dtype=torch.int64, device=device)
        if not torch.is_tensor(tokens):
            tokens = torch.tensor(tokens, dtype=torch.int64, device=device)
        self.future_map.stash(rows, RelayPayload(bonus_tokens=tokens))
