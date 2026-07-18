"""Scheduler wiring for beam search (plain-req group architecture).

A beam_width=k request runs as k plain member rows: the leader (the user's
Req, member 0) plus k-1 internal members spawned decode-ready after the
leader's prefill. Members run the standard decode path end to end; this
processor only hooks three points:

1. admission -- validate_and_init: validation, neutral row params, the
   internal top-2k logprob channel, the group overlay
2. selection -- on_leader_prefill / process_decode: joint_select over per-row
   top-2k, overwrite next tokens via the FutureMap relay, reparent KV
   (copy-on-fork) where the frontier moved
3. lifecycle -- member spawn (prebuilt decode batch) and group finish
   (finalize + beam_results on the leader; members exit silently)

Member KV bookkeeping is born correct on the standard per-req fields:
committed == allocated == prompt_len, cache_protected_len == prompt_len
(the aliased prompt is not the member's to free).
"""

from __future__ import annotations

import logging
from array import array
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from sglang.srt.beam_search.beam_group import BeamGroup, BeamGroupState
from sglang.srt.beam_search.fork import (
    MEMBER_LENGTH_MARGIN,
    init_member_kv_state,
    neutral_member_sampling_params,
    reparent_kv,
    spawn_member,
)
from sglang.srt.beam_search.joint_select import joint_select, select_final_topk
from sglang.srt.managers.overlap_utils import FutureMap, RelayPayload
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    FINISH_MATCHED_TOKEN,
    Req,
    ScheduleBatch,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class BeamCoordinator:
    server_args: ServerArgs
    model_config: ModelConfig
    spec_algorithm: SpeculativeAlgorithm
    enable_overlap: bool
    dllm_enabled: bool
    max_req_len: int
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    tree_cache: BasePrefixCache
    future_map: FutureMap

    _pending_spawn_groups: List[BeamGroup] = field(default_factory=list)
    _kv_buffers: Optional[List[torch.Tensor]] = None

    # ==================== admission ====================

    @staticmethod
    def request_beam_width(recv_req) -> int:
        """beam_width of an incoming request (1 = not a beam request)."""
        return getattr(recv_req.sampling_params, "beam_width", None) or 1

    def validate_and_init(self, req: Req, recv_req) -> Optional[str]:
        """Validate a beam request and attach its group; returns an error or None.

        On success the leader's row params are neutralized (the user's
        semantics move onto the group) and the top-2k channel is armed.
        """
        user_params = req.sampling_params
        beam_width = user_params.beam_width

        # Server-level compatibility (sync transition + long-term exclusions).
        if self.enable_overlap:
            return (
                "Beam search currently requires the server to run with "
                "--disable-overlap-schedule."
            )
        if not self.spec_algorithm.is_none():
            return "Beam search is not supported with speculative decoding."
        if self.server_args.disaggregation_mode != "null":
            return "Beam search is not supported with PD disaggregation."
        if self.server_args.page_size > 1:
            return "Beam search currently requires --page-size 1."
        if self.dllm_enabled:
            return "Beam search is not supported with diffusion LLM."
        if getattr(self.server_args, "enable_hisparse", False):
            return "Beam search is not supported with hisparse."
        if self.server_args.pp_size > 1:
            return "Beam search is not supported with pipeline parallelism."
        if self.server_args.enable_hierarchical_cache:
            return "Beam search is not supported with hierarchical cache."

        # Request-level compatibility.
        if req.session_id is not None or recv_req.session_params is not None:
            return "Beam search is not supported for session requests."
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
            num_return=user_params.n if user_params.n > 1 else beam_width,
            # Frontier state lives on device: selection consumes the sampler's
            # top-2k tensors in place; only k-sized results ever reach the host.
            device=self.req_to_token_pool.device,
        )
        group.leader = req
        group.member_reqs = [req]
        group.prompt_len = prompt_len

        # Neutralize the leader's row params (raw log_softmax scoring, no
        # self-finish path); the user's semantics now live on the group.
        neutral = neutral_member_sampling_params(user_params)
        neutral.max_new_tokens = max_new_tokens + MEMBER_LENGTH_MARGIN
        neutral.no_stop_trim = user_params.no_stop_trim
        req.sampling_params = neutral
        req.group = group
        # The leader's decode suffix is a beam path; never insert it into the
        # radix tree (this also skips the prefill-time unfinished insert: in v1
        # the leader row keeps sole ownership of any non-tree prompt KV).
        req.skip_radix_cache_insert = True
        return None

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

    # ==================== prefill hook ====================

    def on_leader_prefill(
        self, req: Req, i: int, logits_output: LogitsProcessorOutput
    ) -> None:
        """First joint selection, replacing the normal sampled-token append:
        the leader adopts survivor 0, the remaining first tokens are staged
        for the member-spawn tick."""
        group: BeamGroup = req.group
        if req.to_finish is not None:
            # The leader was aborted mid-prefill; the group never starts.
            self._abort_group(group)
            return
        device = group.stop_token_ids.device
        top_logprobs = self._rows_to_tensor(
            [logits_output.next_token_top_logprobs_val[i]], torch.float32, device
        )
        top_tokens = self._rows_to_tensor(
            [logits_output.next_token_top_logprobs_idx[i]], torch.int64, device
        )

        sel_tokens = self._advance(group, top_logprobs, top_tokens)
        if sel_tokens is None:
            self._finish_group(group)
            return

        req.output_ids.append(sel_tokens[0])
        group.pending_first_tokens = sel_tokens
        self._pending_spawn_groups.append(group)
        # Overwrite the leader's relayed next token (the sampled one is void).
        self._stash_next_tokens([req.req_pool_idx], sel_tokens[:1])

    # ==================== member spawn ====================

    def build_pending_member_batch(self) -> Optional[ScheduleBatch]:
        """Build one decode-ready batch of all pending members (rows allocated
        here, prompt mapping aliased from the leader, first tokens via the
        relay); the caller merges it into the running batch."""
        if not self._pending_spawn_groups:
            return None
        groups = self._pending_spawn_groups
        self._pending_spawn_groups = []

        members: List[Req] = []
        spawn_plan = []  # (member, leader_row, prompt_len, first_token)
        for group in groups:
            leader = group.leader
            if leader.finished() or leader.is_retracted:
                # The leader died between selection and spawn (abort/timeout);
                # the group never activates.
                group.state = BeamGroupState.FINISHED
                group.pending_first_tokens = None
                continue
            first_tokens = group.pending_first_tokens
            group.pending_first_tokens = None
            for j, token in enumerate(first_tokens[1:], start=1):
                member = spawn_member(leader, token, j)
                members.append(member)
                spawn_plan.append(
                    (member, leader.req_pool_idx, group.prompt_len, token)
                )
                group.member_reqs.append(member)

        if not members:
            return None

        rows = self.req_to_token_pool.alloc(members)
        assert rows is not None, (
            f"Beam member spawn needs {len(members)} req-to-token slots but only "
            f"{self.req_to_token_pool.available_size()} are free; the admission "
            f"gate (get_num_allocatable_reqs) must reserve them."
        )
        req_to_token = self.req_to_token_pool.req_to_token
        for member, leader_row, prompt_len, _ in spawn_plan:
            init_member_kv_state(member, req_to_token, leader_row, prompt_len)

        batch = ScheduleBatch.init_new(
            reqs=members,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            model_config=self.model_config,
            enable_overlap=self.enable_overlap,
            spec_algorithm=self.spec_algorithm,
        )
        device = batch.device
        batch.req_pool_indices = torch.tensor(rows, dtype=torch.int64, device=device)
        batch.req_pool_indices_cpu = torch.tensor(rows, dtype=torch.int64)
        seq_lens = [plan[2] for plan in spawn_plan]  # KV present = prompt only
        batch.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device=device)
        batch.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        batch.orig_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        batch.seq_lens_sum = sum(seq_lens)
        # First decode input rides the relay, like every other decode row.
        self._stash_next_tokens(rows, [plan[3] for plan in spawn_plan])
        batch.input_ids = None
        batch.multimodal_inputs = [None] * len(members)
        batch.top_logprobs_nums = [m.logprob.top_logprobs_num for m in members]
        batch.token_ids_logprobs = [None] * len(members)
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch, self.model_config.vocab_size
        )
        return batch

    # ==================== decode hook ====================

    def process_decode(
        self, batch: ScheduleBatch, logits_output: LogitsProcessorOutput
    ) -> None:
        """Joint-select every group in this decode batch (runs once per decode
        result, before the per-req loop): rewrites member histories and next
        tokens, reparents KV, finishes groups."""
        row_of: Dict[int, int] = {}
        for i, req in enumerate(batch.reqs):
            if req.group is not None and not req.is_retracted:
                row_of[id(req)] = i
        if not row_of:
            return

        seen = set()
        for req in batch.reqs:
            group = req.group
            if group is None or id(group) in seen:
                continue
            seen.add(id(group))
            if group.state != BeamGroupState.DECODING:
                continue
            self._process_group_decode(group, row_of, logits_output)

    def _process_group_decode(
        self,
        group: BeamGroup,
        row_of: Dict[int, int],
        logits_output: LogitsProcessorOutput,
    ) -> None:
        members = group.member_reqs
        assert len(members) == group.beam_width and all(
            id(m) in row_of for m in members
        ), f"beam group of {group.leader.rid} lost rows mid-decode (group atomicity bug)"

        if any(m.to_finish is not None for m in members):
            # The leader was aborted mid-decode (members never carry their own
            # to_finish); the whole group exits atomically.
            self._abort_group(group)
            return

        idxs = [row_of[id(m)] for m in members]
        device = group.stop_token_ids.device
        top_logprobs = self._rows_to_tensor(
            [logits_output.next_token_top_logprobs_val[i] for i in idxs],
            torch.float32,
            device,
        )
        top_tokens = self._rows_to_tensor(
            [logits_output.next_token_top_logprobs_idx[i] for i in idxs],
            torch.int64,
            device,
        )

        sel_tokens_and_parents = self._advance(
            group, top_logprobs, top_tokens, return_parents=True
        )
        if sel_tokens_and_parents is None:
            self._finish_group(group)
            return
        tokens, parents = sel_tokens_and_parents
        self._apply_survivors(group, parents, tokens)

    def _advance(
        self,
        group: BeamGroup,
        top_logprobs: torch.Tensor,
        top_tokens: torch.Tensor,
        return_parents: bool = False,
    ):
        """Run one selection step; returns survivors or None if the group ended."""
        k = group.beam_width
        if group.next_step_is_final():
            group.advance_final(
                select_final_topk(
                    group.frontier_cum_logprobs, top_logprobs, top_tokens, k
                )
            )
            return None
        sel = joint_select(
            group.frontier_cum_logprobs,
            top_logprobs,
            top_tokens,
            group.stop_token_ids,
            k,
        )
        if group.advance(sel):
            return None
        tokens = sel.next_tokens[:k].tolist()
        if return_parents:
            return tokens, sel.parent_idx[:k].tolist()
        return tokens

    def _apply_survivors(
        self, group: BeamGroup, parents: List[int], tokens: List[int]
    ) -> None:
        """Move rows onto the surviving paths: reparent KV, rewrite histories,
        and relay the selected next tokens."""
        members = group.member_reqs
        moved = [(j, p) for j, p in enumerate(parents) if p != j]
        if moved:
            device = self.req_to_token_pool.device
            dst_rows = torch.tensor(
                [members[j].req_pool_idx for j, _ in moved],
                dtype=torch.int64,
                device=device,
            )
            src_rows = torch.tensor(
                [members[p].req_pool_idx for _, p in moved],
                dtype=torch.int64,
                device=device,
            )
            reparent_kv(
                self.req_to_token_pool.req_to_token,
                self._kv_data_buffers(),
                dst_rows=dst_rows,
                src_rows=src_rows,
                prefix_len=group.prompt_len,
                # All rows are synchronized; the leader's committed length is
                # the group's (it covers the KV computed through this step).
                seq_len=members[0].kv_committed_len,
            )

        # Snapshot parents' histories first: a parent row may itself be
        # rewritten by an earlier survivor in the same step.
        old_outputs = {p: list(members[p].output_ids) for _, p in moved}
        for j, (parent, token) in enumerate(zip(parents, tokens)):
            member = members[j]
            if parent != j:
                member.output_ids = array("q", old_outputs[parent])
            member.output_ids.append(token)

        self._stash_next_tokens([m.req_pool_idx for m in members], tokens)

    # ==================== group finish ====================

    def _finish_group(self, group: BeamGroup) -> None:
        """Mark every row finished (group-atomic): the leader carries the best
        sequence's finish reason; members exit with an innocuous one."""
        group.final_results = group.finalize()
        top = group.final_results[0]
        leader = group.member_reqs[0]
        if top.matched_token is not None:
            leader.finished_reason = FINISH_MATCHED_TOKEN(matched=top.matched_token)
        else:
            leader.finished_reason = FINISH_LENGTH(length=group.num_generated)
        for member in group.member_reqs[1:]:
            member.finished_reason = FINISH_LENGTH(length=len(member.output_ids))

    def _abort_group(self, group: BeamGroup) -> None:
        """Terminate a group whose leader was aborted: no beam results."""
        group.state = BeamGroupState.FINISHED
        group.final_results = []
        leader = group.member_reqs[0] if group.member_reqs else group.leader
        leader.finished_reason = leader.to_finish or FINISH_ABORT("Beam group aborted.")
        leader.to_finish = None
        for member in group.member_reqs[1:]:
            member.finished_reason = FINISH_LENGTH(length=len(member.output_ids))
            member.to_finish = None

    # ==================== helpers ====================

    @staticmethod
    def _rows_to_tensor(entries, dtype, device) -> torch.Tensor:
        """Stack per-row top-2k entries into [rows, 2k] on the target device.

        Entries are device tensors on the standard decode path (kept on device
        by _normalize_decode_outputs) and python lists on the prefill path.
        """
        if torch.is_tensor(entries[0]):
            return torch.stack(entries).to(dtype)
        return torch.tensor(entries, dtype=dtype, device=device)

    def _stash_next_tokens(self, rows: List[int], tokens: List[int]) -> None:
        device = self.req_to_token_pool.device
        self.future_map.stash(
            torch.tensor(rows, dtype=torch.int64, device=device),
            RelayPayload(
                bonus_tokens=torch.tensor(tokens, dtype=torch.int64, device=device)
            ),
        )

    def _kv_data_buffers(self) -> List[torch.Tensor]:
        if self._kv_buffers is None:
            pool = self.token_to_kv_pool_allocator.get_kvcache()
            if hasattr(pool, "k_buffer") and hasattr(pool, "v_buffer"):
                self._kv_buffers = list(pool.k_buffer) + list(pool.v_buffer)
            elif hasattr(pool, "kv_buffer"):
                self._kv_buffers = list(pool.kv_buffer)
            else:
                raise NotImplementedError(
                    f"beam search reparent does not support KV pool {type(pool)}"
                )
        return self._kv_buffers
