from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Deque, List, Optional, Tuple, Union

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.routed_experts_capturer import get_global_experts_capturer
from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchEmbeddingOutput,
    BatchTokenIDOutput,
)
from sglang.srt.managers.schedule_batch import (
    BaseFinishReason,
    Req,
    ScheduleBatch,
)
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.decoupled_spec_io import (
    DraftClose,
    DraftControlMessage,
    DraftReqKey,
    DraftSync,
    DraftTailStreamOutput,
    DraftTailStreamOutputBatch,
    VerifyCommit,
    build_draft_scheduler_rid,
    parse_draft_scheduler_rid,
)
from sglang.srt.utils import broadcast_pyobj

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import (
        EmbeddingBatchResult,
        GenerationBatchResult,
        ScheduleBatch,
        Scheduler,
    )

logger = logging.getLogger(__name__)

DEFAULT_FORCE_STREAM_INTERVAL = 50


@dataclass
class DraftReqState:
    key: DraftReqKey
    req: Optional[Req] = None
    verifier_committed_prefix_len: int = 0
    pending_verify_commits: Deque[VerifyCommit] = field(default_factory=deque)
    pending_close: Optional[DraftClose] = None


@dataclass
class DraftCommitApplyResult:
    stream_output: Optional[Tuple[str, int, int, int, int]] = None
    applied_commits: list[VerifyCommit] = field(default_factory=list)
    deferred_commits: list[VerifyCommit] = field(default_factory=list)


class SchedulerOutputProcessorMixin:
    """
    This class implements the output processing logic for Scheduler.
    We put them into a separate file to make the `scheduler.py` shorter.
    """

    def _is_decoupled_draft_entry_rank(self: Scheduler) -> bool:
        return (
            self.spec_algorithm.is_decoupled_draft()
            and self.pp_rank == 0
            and self.attn_tp_rank == 0
            and self.attn_cp_rank == 0
        )

    def _broadcast_draft_control_messages(
        self: Scheduler,
        messages: list[DraftControlMessage] | None,
    ) -> list[DraftControlMessage]:
        """
        Broadcast draft control messages among all ranks:
        DraftSync: build a new draft request based on its prompt token_ids
        VerifyCommit: overwrite the bonus token and truncate the suffix if needed
        """
        if getattr(self.server_args, "enable_dp_attention", False):
            if self.attn_tp_size != 1:
                messages = broadcast_pyobj(
                    messages,
                    self.attn_tp_group.rank,
                    self.attn_tp_cpu_group,
                    src=self.attn_tp_group.ranks[0],
                )
            if self.attn_cp_size != 1:
                messages = broadcast_pyobj(
                    messages,
                    self.attn_cp_group.rank,
                    self.attn_cp_cpu_group,
                    src=self.attn_cp_group.ranks[0],
                )
            return list(messages or [])

        if self.tp_size != 1:
            messages = broadcast_pyobj(
                messages,
                self.tp_group.rank,
                self.tp_cpu_group,
                src=self.tp_group.ranks[0],
            )
        return list(messages or [])

    def _get_draft_adapter_thread(self: Scheduler):
        adapter = self.draft_adapter_thread
        if adapter is None:
            raise RuntimeError("Decoupled draft entry rank has no draft adapter thread")
        return adapter

    def _draft_get_or_create_state(
        self: Scheduler,
        draft_key: DraftReqKey,
    ) -> DraftReqState:
        state = self.draft_req_table.get(draft_key)
        if state is None:
            state = DraftReqState(key=draft_key)
            self.draft_req_table[draft_key] = state
        return state

    def _draft_get_state_by_req(self: Scheduler, req: Req) -> DraftReqState:
        draft_key = parse_draft_scheduler_rid(req.rid)
        state = self.draft_req_table.get(draft_key)
        if state is None:
            raise RuntimeError(
                "Decoupled draft request has no scheduler state: "
                f"rid={req.rid} draft_key={draft_key}"
            )
        if state.req is not req:
            raise RuntimeError(
                "Decoupled draft scheduler state points to a different request: "
                f"rid={req.rid} draft_key={draft_key}"
            )
        return state

    def _submit_draft_tokens_stream(
        self: Scheduler,
        stream_output_batch: DraftTailStreamOutputBatch,
    ) -> None:
        """
        Submit new draft tokens produced by a decode round to the verifier.
        """
        if not stream_output_batch.outputs:
            return
        if not self._is_decoupled_draft_entry_rank():
            return
        self._get_draft_adapter_thread().submit_draft_results(stream_output_batch)


    def _draft_apply_verify_commit(
        self: Scheduler,
        req: Req,
        message: VerifyCommit,
        *,
        batch: Optional[ScheduleBatch] = None,
        req_batch_idx: Optional[int] = None,
    ) -> None:
        """
        apply the verify result (pre_verify_committed_len, bonus_token_pos,
        bonus_token_id) to the draft request:
        1. overwrite the bonus token, update the related states, including output_ids, grammar, kv cache, stream output... if needed
        2. update the scheduler state's verifier_committed_prefix_len to (bonus_token_pos + 1)
        """
        state = self._draft_get_state_by_req(req)
        if state.key != message.draft_key:
            raise RuntimeError(
                "VerifyCommit arrived for a mismatched draft request: "
                f"req_rid={req.rid} req_draft_key={state.key} "
                f"message_draft_key={message.draft_key}"
            )
        pre_verify_committed_len = int(message.pre_verify_committed_len)
        bonus_token_pos = int(message.bonus_token_pos)
        bonus_token_id = int(message.bonus_token_id)
        current_committed_len = int(state.verifier_committed_prefix_len)
        new_committed_len = bonus_token_pos + 1
        output_len = len(req.output_ids)
        prompt_len = len(req.origin_input_ids)
        materialized_kv_len = prompt_len + max(output_len - 1, 0)

        if new_committed_len <= current_committed_len:
            raise RuntimeError(
                "VerifyCommit must advance the drafter committed prefix: "
                f"request_id={message.request_id} "
                f"src_verifier_rank={message.src_verifier_rank} "
                f"pre_verify_committed_len={pre_verify_committed_len} "
                f"bonus_token_pos={bonus_token_pos} "
                f"current_committed_len={current_committed_len} "
                f"new_committed_len={new_committed_len}"
            )

        if pre_verify_committed_len > current_committed_len:
            raise RuntimeError(
                "VerifyCommit depends on a prefix the drafter has not committed: "
                f"request_id={message.request_id} "
                f"src_verifier_rank={message.src_verifier_rank} "
                f"pre_verify_committed_len={pre_verify_committed_len} "
                f"current_committed_len={current_committed_len}"
            )

        if bonus_token_pos < pre_verify_committed_len:
            raise RuntimeError(
                "VerifyCommit bonus token is before its pre-verify prefix: "
                f"request_id={message.request_id} "
                f"src_verifier_rank={message.src_verifier_rank} "
                f"pre_verify_committed_len={pre_verify_committed_len} "
                f"bonus_token_pos={bonus_token_pos}"
            )

        if bonus_token_pos >= output_len:
            raise RuntimeError(
                "Decoupled draft attempted to apply an unmaterialized verify commit: "
                f"request_id={state.key.request_id} "
                f"bonus_token_pos={bonus_token_pos} "
                f"output_len={output_len} "
                f"committed_prefix_len={state.verifier_committed_prefix_len} "
                f"pre_verify_committed_len={pre_verify_committed_len}"
            )

        if req.kv_committed_freed:
            raise RuntimeError(
                "Decoupled draft verify commit found freed KV cache: "
                f"request_id={state.key.request_id} "
                f"bonus_token_pos={bonus_token_pos} "
                f"output_len={output_len} "
                f"kv_committed_len={req.kv_committed_len} "
                f"kv_allocated_len={req.kv_allocated_len}"
            )

        if (
            req.kv_committed_len < materialized_kv_len
            or req.kv_allocated_len < materialized_kv_len
        ):
            raise RuntimeError(
                "Decoupled draft KV prefix is shorter than the materialized "
                "output prefix: "
                f"request_id={state.key.request_id} "
                f"bonus_token_pos={bonus_token_pos} "
                f"output_len={output_len} "
                f"prompt_len={prompt_len} "
                f"materialized_kv_len={materialized_kv_len} "
                f"kv_committed_len={req.kv_committed_len} "
                f"kv_allocated_len={req.kv_allocated_len}"
            )


        bonus_token_matches = (
            bonus_token_pos < output_len
            and int(req.output_ids[bonus_token_pos]) == bonus_token_id
        )

        if bonus_token_matches:
            # if the bonus token matches, only need to push forward the committed prefix
            state.verifier_committed_prefix_len = new_committed_len
            return

        # The verifier-selected bonus token replaces the drafter suffix starting at
        # `bonus_token_pos`.
        #
        # Positions here are in req.output_ids, not in the full prompt+output
        # sequence. The kept output range is [0, truncate_from), and the removed
        # output range is [truncate_from, len(req.output_ids)). In other words,
        # `truncate_from` itself is removed. After the removal, `bonus_token_id`
        # is appended at exactly that position.
        truncate_from = bonus_token_pos

        # Number of output tokens removed from the drafter suffix:
        # len(req.output_ids[truncate_from:]).
        removed = output_len - truncate_from

        # KV positions are in the full sequence coordinate system:
        # [0, prompt_len) are prompt tokens, and output_ids[i] corresponds to
        # full-sequence position prompt_len + i. Therefore the KV entries to
        # discard start at `kv_truncate_from`, inclusive.
        kv_truncate_from = prompt_len + truncate_from

        if kv_truncate_from > min(req.kv_committed_len, req.kv_allocated_len):
            raise RuntimeError(
                "Decoupled draft cannot truncate beyond materialized KV prefix: "
                f"request_id={state.key.request_id} "
                f"bonus_token_pos={bonus_token_pos} "
                f"output_len={output_len} "
                f"prompt_len={prompt_len} "
                f"kv_truncate_from={kv_truncate_from} "
                f"kv_committed_len={req.kv_committed_len} "
                f"kv_allocated_len={req.kv_allocated_len}"
            )

        if removed > 0:
            if req.grammar is not None:
                try:
                    req.grammar.rollback(removed)
                except Exception:
                    logger.debug("Draft grammar rollback failed for req %s", req.rid)

            if req.req_pool_idx is not None and not req.kv_committed_freed:
                # Only free KV slots that are currently allocated for this req.
                # `trimmed_end` is exclusive. The freed full-sequence KV range is
                # [kv_truncate_from, trimmed_end). If kv_truncate_from ==
                # trimmed_end, there is nothing to free.
                trimmed_end = min(
                    req.kv_allocated_len, prompt_len + len(req.output_ids)
                )
                if kv_truncate_from < trimmed_end:
                    indices_to_free = self.req_to_token_pool.req_to_token[
                        req.req_pool_idx, kv_truncate_from:trimmed_end
                    ]
                    if len(indices_to_free) > 0:
                        self.token_to_kv_pool_allocator.free(indices_to_free)
                req.kv_committed_len = min(req.kv_committed_len, kv_truncate_from)
                req.kv_allocated_len = min(req.kv_allocated_len, kv_truncate_from)
                req.cache_protected_len = min(
                    req.cache_protected_len, kv_truncate_from
                )

            # Truncate per-output arrays with the same output-index interval:
            # delete [truncate_from, old_output_len).
            del req.output_ids[truncate_from:]
            if req.return_logprob:
                del req.output_token_logprobs_val[truncate_from:]
                del req.output_token_logprobs_idx[truncate_from:]
                del req.output_top_logprobs_val[truncate_from:]
                del req.output_top_logprobs_idx[truncate_from:]
                del req.output_token_ids_logprobs_val[truncate_from:]
                del req.output_token_ids_logprobs_idx[truncate_from:]
            if req.hidden_states:
                del req.hidden_states[truncate_from:]

        req.output_ids.append(bonus_token_id)
        if req.grammar is not None:
            try:
                req.grammar.accept_token(bonus_token_id)
            except Exception:
                logger.debug(
                    "Draft grammar accept failed during bonus token update for req %s",
                    req.rid,
                )
        req.finished_reason = None
        req.finished_len = None
        req.finished_output = None
        req.to_finish = None
        req.decoded_text = ""

        if len(req.output_ids) != new_committed_len:
            raise RuntimeError(
                "Decoupled draft verify commit produced an unexpected output "
                "length: "
                f"request_id={state.key.request_id} "
                f"expected_output_len={new_committed_len} "
                f"actual_output_len={len(req.output_ids)} "
                f"bonus_token_pos={bonus_token_pos}"
            )
        if int(req.output_ids[-1]) != bonus_token_id:
            raise RuntimeError(
                "Decoupled draft verify commit failed to install bonus token: "
                f"request_id={state.key.request_id} "
                f"bonus_token_pos={bonus_token_pos} "
                f"bonus_token_id={bonus_token_id} "
                f"tail_token_id={int(req.output_ids[-1])}"
            )

        state.verifier_committed_prefix_len = new_committed_len

        if batch is not None and req_batch_idx is not None:
            if not (0 <= req_batch_idx < len(batch.reqs)):
                raise RuntimeError(
                    "Decoupled draft verify commit received an invalid batch "
                    "index: "
                    f"request_id={state.key.request_id} "
                    f"req_batch_idx={req_batch_idx} "
                    f"batch_size={len(batch.reqs)}"
                )
            if batch.reqs[req_batch_idx] is not req:
                raise RuntimeError(
                    "Decoupled draft verify commit batch index points to a "
                    "different request: "
                    f"request_id={state.key.request_id} "
                    f"req_batch_idx={req_batch_idx} "
                    f"batch_req_rid={batch.reqs[req_batch_idx].rid}"
                )
            # Keep the in-flight decode batch consistent with the rewritten request
            # state. This block is only needed when the verifier bonus token changed
            # req.output_ids above: either an existing suffix was truncated and
            # replaced, or the bonus token was appended at the current tail.
            #
            # Decode seq_len is the number of tokens **already present in KV** before the
            # next tail token is consumed. For a drafter request, output_ids[-1] is the
            # current tail token used as the next decode input, so the KV-backed prefix
            # is origin_input_ids plus output_ids[0:-1]. The slice [0, -1) excludes the
            # tail token itself, hence len(origin_input_ids) + max(len(output_ids)-1, 0).
            new_seq_len = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)

            # batch.output_ids[req_batch_idx] stores the single tail token that the
            # decode worker will consume next. Prefer the last output token. If no
            # output token exists yet, fall back to the last prompt token. In both
            # cases the selected token is included as the decode input, but excluded
            # from new_seq_len above.
            if req.output_ids:
                new_tail_token_id = int(req.output_ids[-1])
            elif req.origin_input_ids:
                new_tail_token_id = int(req.origin_input_ids[-1])
            else:
                raise AssertionError(
                    f"Draft request {req.rid} has no token to decode from"
                )

            old_seq_len = None
            if batch.seq_lens_cpu is not None:
                # Save the old per-request seq_len so seq_lens_sum can be adjusted by
                # delta. req_batch_idx is the inclusive index of this req in batch.reqs.
                old_seq_len = int(batch.seq_lens_cpu[req_batch_idx].item())
                batch.seq_lens_cpu[req_batch_idx] = new_seq_len

            # Mirror the same new_seq_len into every per-request seq_len buffer.
            if batch.seq_lens is not None:
                batch.seq_lens[req_batch_idx] = new_seq_len
            if batch.orig_seq_lens is not None:
                batch.orig_seq_lens[req_batch_idx] = new_seq_len

            # The batch-level output_ids entry is not the whole output sequence. It is
            # exactly the one tail token for this request's next decode step.
            if batch.output_ids is not None:
                batch.output_ids[req_batch_idx] = new_tail_token_id

            if batch.seq_lens_sum is not None:
                if old_seq_len is not None:
                    # Incrementally maintain sum(seq_lens). This is equivalent to
                    # replacing one element: sum' = sum - old_seq_len + new_seq_len.
                    batch.seq_lens_sum += new_seq_len - old_seq_len
                elif batch.seq_lens_cpu is not None:
                    # Fallback when the old value was unavailable: recompute the full
                    # sum over all requests in the batch.
                    batch.seq_lens_sum = int(batch.seq_lens_cpu.sum().item())

            if new_seq_len > min(req.kv_committed_len, req.kv_allocated_len):
                raise RuntimeError(
                    "Decoupled draft batch seq_len points beyond materialized KV "
                    "after verify commit: "
                    f"request_id={state.key.request_id} "
                    f"new_seq_len={new_seq_len} "
                    f"kv_committed_len={req.kv_committed_len} "
                    f"kv_allocated_len={req.kv_allocated_len}"
                )



    def _draft_release_req(self: Scheduler, req: Req) -> None:
        """
        release a draft request only when it has completed at the verifier side:
        1. evict the req from waiting_queue or running_batch
        2. remove the req from the draft request table
        3. release its kvcache
        """
        state = self._draft_get_state_by_req(req)

        # remove the req from waiting_queue or running_batch
        self.waiting_queue = [
            queued_req for queued_req in self.waiting_queue if queued_req is not req
        ]
        if getattr(self, "running_batch", None) is not None and self.running_batch.reqs:
            keep_indices = [
                i
                for i, running_req in enumerate(self.running_batch.reqs)
                if running_req is not req
            ]
            self.running_batch.filter_batch(keep_indices=keep_indices)
        self.draft_req_table.pop(state.key, None)
        release_kv_cache(req, self.tree_cache, is_insert=False)


    def _draft_create_req(
        self: Scheduler,
        message: DraftSync,
    ) -> Req:
        """
        Create and register a new drafter-side request from DraftSync.
        """
        state = self._draft_get_or_create_state(message.draft_key)
        if state.pending_close is not None:
            raise RuntimeError(
                "Received DraftSync for a closed decoupled draft request: "
                f"request_id={message.request_id}"
            )
        if state.req is not None:
            raise RuntimeError(
                "Received DraftSync for an existing decoupled draft request: "
                f"request_id={message.request_id}"
            )

        sampling_params = SamplingParams(
            max_new_tokens=1 << 30, # a very large number to ensure the drafter keeps sampling until receiving DraftClose
            temperature=0.0,
            top_k=1,
            ignore_eos=True,
        )
        sampling_params.normalize(self.tokenizer)
        sampling_params.verify(self.model_config.vocab_size)

        req = Req(
            build_draft_scheduler_rid(message.draft_key),
            "",
            list(message.prompt_token_ids),
            sampling_params,
            return_logprob=False,
            stream=False,
            eos_token_ids=self.model_config.hf_eos_token_id,
            vocab_size=self.model_config.vocab_size,
            metrics_collector=(self.metrics_collector if self.enable_metrics else None),
        )
        req.tokenizer = self.tokenizer
        req.output_ids = list(message.committed_output_ids)
        req.fill_ids = req.origin_input_ids + req.output_ids
        self.init_req_max_new_tokens(req)
        state.req = req
        state.verifier_committed_prefix_len = len(req.output_ids)
        return req


    def _drain_post_decode_draft_control_messages(
        self: Scheduler,
    ) -> list[VerifyCommit | DraftClose]:
        """
        (called by decoupled drafter)
        Drain all VerifyCommit and DraftClose messages from draft adapter thread
        """
        messages: list[DraftControlMessage] | None = None
        if self._is_decoupled_draft_entry_rank():
            messages = self._get_draft_adapter_thread().drain_post_result_messages()

        return [
            message
            for message in self._broadcast_draft_control_messages(messages)
            if isinstance(message, (VerifyCommit, DraftClose))
        ]


    def _handle_draft_sync_messages(self: Scheduler) -> None:
        """
        (called by decoupled drafter)
        Drain DraftSync messages from draft adapter thread, and handle them.
        DraftSync creates a new drafter-side request from verifier state.
        """

        trace_enabled = getattr(self.decoupled_spec_tracer, "enabled", False)
        trace_start_ns = time.perf_counter_ns() if trace_enabled else 0
        messages: list[DraftControlMessage] | None = None
        if self._is_decoupled_draft_entry_rank():
            messages = self._get_draft_adapter_thread().drain_sync_messages()

        messages = [
            message
            for message in self._broadcast_draft_control_messages(messages)
            if isinstance(message, DraftSync)
        ]
        if trace_enabled and messages:
            self.decoupled_spec_tracer.record(
                "drafter",
                "recv_sync_batch",
                duration_ms=(time.perf_counter_ns() - trace_start_ns) / 1_000_000,
                batch_size=len(messages),
                rids=[message.request_id for message in messages],
                committed_lens_by_req=[
                    len(message.committed_output_ids) for message in messages
                ],
                output_lens_by_req=[
                    len(message.committed_output_ids) for message in messages
                ],
            )

        if not messages:
            return

        create_start_ns = time.perf_counter_ns() if trace_enabled else 0
        created_reqs: list[Req] = []
        for message in messages:
            entry = self.draft_req_table.get(message.draft_key)
            if entry is not None:
                if entry.pending_close is not None:
                    self.draft_req_table.pop(message.draft_key, None)
                    continue
            req = self._draft_create_req(message)
            running_batch = self.running_batch
            if req not in self.waiting_queue and req not in running_batch.reqs:
                self._add_request_to_queue(req)
            created_reqs.append(req)
        if trace_enabled:
            self.decoupled_spec_tracer.record(
                "drafter",
                "create_draft_req_batch",
                duration_ms=(time.perf_counter_ns() - create_start_ns) / 1_000_000,
                batch_size=len(created_reqs),
                rids=[
                    parse_draft_scheduler_rid(req.rid).request_id
                    for req in created_reqs
                ],
                committed_lens_by_req=[
                    int(
                        self._draft_get_state_by_req(req).verifier_committed_prefix_len
                    )
                    for req in created_reqs
                ],
                output_lens_by_req=[len(req.output_ids) for req in created_reqs],
            )


    def _draft_apply_commits_and_maybe_emit(
        self: Scheduler,
        req: Req,
        *,
        commits: Optional[list[VerifyCommit]] = None,
        batch: ScheduleBatch,
        req_batch_idx: int,
        decoded_token: Optional[tuple[int, int]] = None,
    ) -> DraftCommitApplyResult:
        """
        1. apply the pending verify commits to the req,
          and update its scheduler state and other states(if needed)
        2. if the bonus token is not overwritten,
          this req is considered having decoded a new valid draft token,
          therefore, drafter will send this draft token to verifier,
          as streaming draft token output
        """
        state = self._draft_get_state_by_req(req)
        result = DraftCommitApplyResult()

        commits_to_apply: list[VerifyCommit] = []
        if state.pending_verify_commits:
            commits_to_apply.extend(state.pending_verify_commits)
            state.pending_verify_commits.clear()

        if commits:
            commits_to_apply.extend(commits)

        for commit_idx, verify_commit in enumerate(commits_to_apply):
            if int(verify_commit.bonus_token_pos) >= len(req.output_ids):
                result.deferred_commits.extend(commits_to_apply[commit_idx:])
                state.pending_verify_commits.extend(commits_to_apply[commit_idx:])
                break
            self._draft_apply_verify_commit(
                req,
                verify_commit,
                batch=batch,
                req_batch_idx=req_batch_idx,
            )
            result.applied_commits.append(verify_commit)

        if not self._is_decoupled_draft_entry_rank() or decoded_token is None:
            return result

        token_pos, token_id = (int(decoded_token[0]), int(decoded_token[1]))
        committed_len = int(state.verifier_committed_prefix_len)
        if (
            token_pos >= committed_len
            and token_pos < len(req.output_ids)
            and int(req.output_ids[token_pos]) == token_id
        ):
            result.stream_output = (
                state.key.request_id,
                int(state.key.src_verifier_rank),
                committed_len,
                token_pos,
                token_id,
            )
        return result


    def _draft_process_post_decode_controls(
        self: Scheduler,
        batch: ScheduleBatch,
        decoded_tokens: list[Optional[tuple[int, int]]],
        control_messages: list[VerifyCommit | DraftClose],
    ) -> DraftTailStreamOutputBatch:
        """
        args:
          decoded_tokens: the list of (token_pos, token_id) for each new decoded token
          control_messages: VerifyCommit and DraftClose message received from verifier

        called by decoupled drafter, during `process_batch_result_decode()`:
        1. apply DraftClose message: release the draft req
        2. apply VerifyCommit message to the req, and collect & send new draft token
        """
        trace_enabled = getattr(self.decoupled_spec_tracer, "enabled", False)
        trace_start_ns = time.perf_counter_ns() if trace_enabled else 0
        # build draft_key -> req mapping
        current_req_by_key: dict[DraftReqKey, Req] = {}
        decoded_token_by_key: dict[DraftReqKey, Optional[tuple[int, int]]] = {}
        for req_batch_idx, req in enumerate(batch.reqs):
            state = self._draft_get_state_by_req(req)
            current_req_by_key[state.key] = req
            decoded_token_by_key[state.key] = (
                decoded_tokens[req_batch_idx]
                if req_batch_idx < len(decoded_tokens)
                else None
            )

        # collect each req's VerifyCommit message
        commits_by_key: dict[DraftReqKey, list[VerifyCommit]] = {}
        deferred_commit_messages: list[VerifyCommit] = []
        closed_keys: set[DraftReqKey] = set()
        for message in control_messages:
            draft_key = message.draft_key
            entry = self.draft_req_table.get(draft_key)
            req = entry.req if entry is not None else None
            if isinstance(message, DraftClose):
                # this req will be release upon recv DraftClose
                # discard its pending VerifyCommits
                closed_keys.add(draft_key)
                commits_by_key.pop(draft_key, None)
                deferred_commit_messages = [
                    commit
                    for commit in deferred_commit_messages
                    if commit.draft_key != draft_key
                ]
                if entry is None:
                    entry = self._draft_get_or_create_state(draft_key)
                entry.pending_verify_commits.clear()
                if req is not None:
                    self._draft_release_req(req)
                else:
                    entry.req = None
                    entry.pending_close = message
                continue

            if draft_key in closed_keys:
                continue

            if entry is None:
                entry = self._draft_get_or_create_state(draft_key)

            if entry.pending_close is not None:
                continue

            if req is None:
                entry.pending_verify_commits.append(message)
                deferred_commit_messages.append(message)
                continue

            assert (
                parse_draft_scheduler_rid(req.rid) == draft_key
            ), "draft_req_table contains a request under a mismatched draft_key"
            if draft_key in current_req_by_key:
                commits_by_key.setdefault(draft_key, []).append(message)
            else:
                # if the req is not in current batch, cache the pending VerifyCommit message
                entry.pending_verify_commits.append(message)
                deferred_commit_messages.append(message)

        commit_messages = [
            message for message in control_messages if isinstance(message, VerifyCommit)
        ]
        close_messages = [
            message for message in control_messages if isinstance(message, DraftClose)
        ]

        # apply VerifyCommit and send new draft token
        stream_output_batch = DraftTailStreamOutputBatch()
        src_drafter_rank = int(getattr(self, "dp_rank", 0) or 0)
        applied_commit_messages: list[VerifyCommit] = []
        for req_batch_idx, req in enumerate(batch.reqs):
            draft_key = self._draft_get_state_by_req(req).key
            decoded_token = decoded_token_by_key.get(draft_key)
            apply_result = self._draft_apply_commits_and_maybe_emit(
                req,
                commits=commits_by_key.get(draft_key),
                batch=batch,
                req_batch_idx=req_batch_idx,
                decoded_token=decoded_token,
            )
            applied_commit_messages.extend(apply_result.applied_commits)
            deferred_commit_messages.extend(apply_result.deferred_commits)
            stream_output_item = apply_result.stream_output
            if stream_output_item is not None:
                (
                    request_id,
                    dst_verifier_rank,
                    base_committed_len,
                    token_pos,
                    token_id,
                ) = stream_output_item
                stream_output_batch.outputs.append(
                    DraftTailStreamOutput(
                        src_drafter_rank=src_drafter_rank,
                        dst_verifier_rank=dst_verifier_rank,
                        request_id=request_id,
                        base_committed_len=base_committed_len,
                        new_token_pos=token_pos,
                        new_token_id=token_id,
                    )
                )
        if trace_enabled:
            control_duration_ms = (
                time.perf_counter_ns() - trace_start_ns
            ) / 1_000_000
            self.decoupled_spec_tracer.record(
                "drafter",
                "apply_commit_batch",
                duration_ms=control_duration_ms,
                forward_mode=str(batch.forward_mode),
                batch_size=len(batch.reqs),
                rids=[message.request_id for message in applied_commit_messages],
                committed_lens_by_req=[
                    int(message.bonus_token_pos) + 1
                    for message in applied_commit_messages
                ],
                num_applied_commit=len(applied_commit_messages),
                num_deferred_commit=len(deferred_commit_messages),
            )
            self.decoupled_spec_tracer.record(
                "drafter",
                "post_decode_control_batch",
                duration_ms=control_duration_ms,
                forward_mode=str(batch.forward_mode),
                batch_size=len(batch.reqs),
                rids=[message.request_id for message in control_messages],
                num_commit=len(commit_messages),
                num_close=len(close_messages),
                num_applied_commit=len(applied_commit_messages),
                num_deferred_commit=len(deferred_commit_messages),
            )
            num_stream_outputs = len(stream_output_batch.outputs)
            self.decoupled_spec_tracer.record(
                "drafter",
                "emit_tail_batch",
                forward_mode=str(batch.forward_mode),
                duration_ms=(time.perf_counter_ns() - trace_start_ns) / 1_000_000,
                batch_size=len(batch.reqs),
                rids=[
                    self._draft_get_state_by_req(req).key.request_id
                    for req in batch.reqs
                ],
                num_stream_outputs=num_stream_outputs,
                committed_lens_by_req=[
                    int(
                        self._draft_get_state_by_req(req).verifier_committed_prefix_len
                    )
                    for req in batch.reqs
                ],
                output_lens_by_req=[len(req.output_ids) for req in batch.reqs],
            )
        return stream_output_batch

    def _get_storage_backend_type(self) -> str:
        """Get storage backend type from tree_cache."""
        storage_backend_type = "none"
        cache_controller = getattr(self.tree_cache, "cache_controller", None)
        if cache_controller and hasattr(cache_controller, "storage_backend"):
            storage_backend = cache_controller.storage_backend
            if storage_backend is not None:
                storage_backend_type = type(storage_backend).__name__
        return storage_backend_type

    def _get_cached_tokens_details(self: Scheduler, req: Req) -> Optional[dict]:
        """Get detailed cache breakdown for a request, if available.

        Returns:
            - None if no cached tokens at all
            - {"device": X, "host": Y} without storage breakdown
            - {"device": X, "host": Y, "storage": Z} with storage breakdown
        """
        if (
            req.cached_tokens_device > 0
            or req.cached_tokens_host > 0
            or req.cached_tokens_storage > 0
        ):
            details = {
                "device": req.cached_tokens_device,
                "host": req.cached_tokens_host,
            }
            # Only include storage fields if L3 storage is enabled
            if getattr(self, "enable_hicache_storage", False):
                details["storage"] = req.cached_tokens_storage
                details["storage_backend"] = self._get_storage_backend_type()
            return details

        if req.cached_tokens > 0:
            return {
                "device": req.cached_tokens,
                "host": 0,
            }

        return None

    def process_batch_result_prebuilt(self: Scheduler, batch: ScheduleBatch):
        assert self.disaggregation_mode == DisaggregationMode.DECODE
        for req in batch.reqs:
            req.time_stats.set_decode_prebuilt_finish_time()
            req.check_finished()
            if req.finished():
                req.time_stats.set_quick_finish_time()
                release_kv_cache(req, self.tree_cache)

        # Note: Logprobs should be handled on the prefill engine.
        self.stream_output(batch.reqs, batch.return_logprob)

    def maybe_collect_routed_experts(self: Scheduler, req: Req):
        """Collect routed experts for a finished request."""
        req.routed_experts = get_global_experts_capturer().get_routed_experts(
            req_pool_idx=req.req_pool_idx,
            seqlen=req.seqlen,
            req_to_token_pool=self.req_to_token_pool,
        )

    def maybe_collect_customized_info(
        self: Scheduler, i: int, req: Req, logits_output: LogitsProcessorOutput
    ):
        if logits_output is not None and logits_output.customized_info is not None:
            if req.customized_info is None:
                req.customized_info = {}
            for k, v in logits_output.customized_info.items():
                if k not in req.customized_info:
                    req.customized_info[k] = []
                # Copy the element so it doesn't retain the entire batch
                # tensor/array via a view reference.
                elem = v[i]
                if isinstance(elem, torch.Tensor):
                    elem = elem.clone()
                elif hasattr(elem, "copy") and callable(elem.copy):
                    elem = elem.copy()
                req.customized_info[k].append(elem)

    def process_batch_result_prefill(
        self: Scheduler,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):
        skip_stream_req = None
        is_decoupled_draft = bool(
            self.is_generation and batch.spec_algorithm.is_decoupled_draft()
        )
        decoded_draft_tokens: list[Optional[tuple[int, int]]] = (
            [None] * len(batch.reqs) if is_decoupled_draft else []
        )

        if self.is_generation:
            if result.copy_done is not None:
                result.copy_done.synchronize()

            (
                logits_output,
                next_token_ids,
                extend_input_len_per_req,
                extend_logprob_start_len_per_req,
            ) = (
                result.logits_output,
                result.next_token_ids,
                result.extend_input_len_per_req,
                result.extend_logprob_start_len_per_req,
            )

            # Move next_token_ids and logprobs to cpu
            next_token_ids = next_token_ids.tolist()
            if batch.return_logprob:
                if logits_output.next_token_logprobs is not None:
                    logits_output.next_token_logprobs = (
                        logits_output.next_token_logprobs.tolist()
                    )
                if logits_output.input_token_logprobs is not None:
                    logits_output.input_token_logprobs = tuple(
                        logits_output.input_token_logprobs.tolist()
                    )
                if logits_output.next_token_top_logprobs_val:
                    logits_output.next_token_top_logprobs_val = [
                        v.tolist() for v in logits_output.next_token_top_logprobs_val
                    ]
                    logits_output.next_token_top_logprobs_idx = [
                        x.tolist() for x in logits_output.next_token_top_logprobs_idx
                    ]
                if logits_output.next_token_token_ids_logprobs_val:
                    logits_output.next_token_token_ids_logprobs_val = [
                        v.tolist()
                        for v in logits_output.next_token_token_ids_logprobs_val
                    ]

            hidden_state_offset = 0

            # Check finish conditions
            logprob_pt = 0

            for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
                if req.finished() or req.is_retracted:
                    # decode req in mixed batch or retracted req
                    continue

                if req.is_chunked <= 0:
                    req.time_stats.set_prefill_finished_time()

                    # req output_ids are set here
                    if is_decoupled_draft:
                        decoded_draft_tokens[i] = (
                            len(req.output_ids),
                            int(next_token_id),
                        )
                    req.output_ids.append(next_token_id)

                    self._maybe_update_reasoning_tokens(req, next_token_id)

                    if not is_decoupled_draft:
                        req.check_finished()
                    if req.finished():
                        self.maybe_collect_routed_experts(req)
                        release_kv_cache(req, self.tree_cache)
                        req.time_stats.set_completion_time()
                    elif not batch.decoding_reqs or req not in batch.decoding_reqs:
                        self.tree_cache.cache_unfinished_req(req)
                        if self.enable_hisparse:
                            self.hisparse_coordinator.admit_request_into_staging(req)

                    self.maybe_collect_customized_info(i, req, logits_output)

                    if batch.return_logprob:
                        assert extend_logprob_start_len_per_req is not None
                        assert extend_input_len_per_req is not None
                        extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                        extend_input_len = extend_input_len_per_req[i]

                        num_input_logprobs = self._calculate_num_input_logprobs(
                            req, extend_input_len, extend_logprob_start_len
                        )

                        if req.return_logprob:
                            self.add_logprob_return_values(
                                i,
                                req,
                                logprob_pt,
                                next_token_ids,
                                num_input_logprobs,
                                logits_output,
                            )
                        logprob_pt += num_input_logprobs

                    if (
                        req.return_hidden_states
                        and logits_output.hidden_states is not None
                    ):
                        req.hidden_states.append(
                            logits_output.hidden_states[
                                hidden_state_offset : (
                                    hidden_state_offset := hidden_state_offset
                                    + len(req.origin_input_ids)
                                )
                            ]
                            .cpu()
                            .clone()
                            .tolist()
                        )

                    if req.grammar is not None:
                        # FIXME: this try-except block is for handling unexpected xgrammar issue.
                        try:
                            req.grammar.accept_token(next_token_id)
                        except ValueError as e:
                            # Grammar accept_token can raise ValueError if the token is not in the grammar.
                            # This can happen if the grammar is not set correctly or the token is invalid.
                            logger.error(
                                f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
                            )
                            self.abort_request(AbortReq(rid=req.rid))
                        req.grammar.finished = req.finished()

                else:
                    # being chunked reqs' prefill is not finished
                    req.is_chunked -= 1
                    # There is only at most one request being currently chunked.
                    # Because this request does not finish prefill,
                    # we don't want to stream the request currently being chunked.
                    skip_stream_req = req

                    # Incrementally update input logprobs.
                    if batch.return_logprob:
                        extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                        extend_input_len = extend_input_len_per_req[i]
                        if extend_logprob_start_len < extend_input_len:
                            # Update input logprobs.
                            num_input_logprobs = self._calculate_num_input_logprobs(
                                req, extend_input_len, extend_logprob_start_len
                            )
                            if req.return_logprob:
                                self.add_input_logprob_return_values(
                                    i,
                                    req,
                                    logits_output,
                                    logprob_pt,
                                    num_input_logprobs,
                                    last_prefill_chunk=False,
                                )
                            logprob_pt += num_input_logprobs

                    req.time_stats.set_last_chunked_prefill_finish_time()

            if is_decoupled_draft:
                post_decode_messages = (
                    self._drain_post_decode_draft_control_messages()
                )
                stream_outputs = self._draft_process_post_decode_controls(
                    batch,
                    decoded_draft_tokens,
                    post_decode_messages,
                )
                self._submit_draft_tokens_stream(stream_outputs)

        else:  # embedding or reward model
            if result.copy_done is not None:
                result.copy_done.synchronize()

            is_sparse = envs.SGLANG_EMBEDDINGS_SPARSE_HEAD.is_set()

            embeddings = result.embeddings

            if is_sparse:
                batch_ids, token_ids = embeddings.indices()
                values = embeddings.values()

                embeddings = [{} for _ in range(embeddings.size(0))]
                for i in range(batch_ids.shape[0]):
                    embeddings[batch_ids[i].item()][token_ids[i].item()] = values[
                        i
                    ].item()
            else:
                if isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings.tolist()
                else:
                    embeddings = [tensor.tolist() for tensor in embeddings]

            # Check finish conditions
            for i, req in enumerate(batch.reqs):
                if req.is_retracted:
                    continue

                req.embedding = embeddings[i]
                if req.is_chunked <= 0:
                    req.time_stats.set_prefill_finished_time()
                    # Dummy output token for embedding models
                    req.output_ids.append(0)
                    req.check_finished()

                    if req.finished():
                        release_kv_cache(req, self.tree_cache)
                        req.time_stats.set_completion_time()
                    else:
                        self.tree_cache.cache_unfinished_req(req)
                else:
                    # being chunked reqs' prefill is not finished
                    req.is_chunked -= 1
                    req.time_stats.set_last_chunked_prefill_finish_time()

        self.stream_output(batch.reqs, batch.return_logprob, skip_stream_req)

        can_run_cuda_graph = getattr(result, "can_run_cuda_graph", False)
        self.report_prefill_stats(
            prefill_stats=batch.prefill_stats,
            can_run_cuda_graph=can_run_cuda_graph,
            dp_cooperation_info=batch.dp_cooperation_info,
        )

    def _resolve_spec_overlap_token_ids(
        self: Scheduler, result: GenerationBatchResult, batch: ScheduleBatch
    ) -> List[List[int]]:
        """Resolve the padding next token ids for speculative decoding with overlap."""
        assert result.next_token_ids.is_cpu
        assert result.accept_lens.is_cpu

        next_token_ids = result.next_token_ids.tolist()
        accept_lens = result.accept_lens.tolist()
        result.num_accepted_tokens = sum(accept_lens) - len(batch.reqs)
        result.accept_length_per_req_cpu = [x - 1 for x in accept_lens]

        predict_tokens = []
        stride = self.draft_worker.speculative_num_draft_tokens

        for i, req in enumerate(batch.reqs):
            req.kv_committed_len += accept_lens[i]
            predict_tokens.append(
                next_token_ids[i * stride : i * stride + accept_lens[i]]
            )
            req.spec_verify_ct += 1

            accepted_draft_tokens = result.accept_length_per_req_cpu[i]
            req.spec_accepted_tokens += accepted_draft_tokens
            req.update_spec_acceptance_histogram(accepted_draft_tokens)

        return predict_tokens

    def _validate_decoupled_verify_result(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:
        accept_lens = result.accept_length_per_req_cpu
        if accept_lens is None:
            raise RuntimeError("Decoupled verify result is missing accept lengths.")
        if len(accept_lens) != len(batch.reqs):
            raise RuntimeError(
                "Decoupled verify accept length count does not match batch size: "
                f"accept_lens={len(accept_lens)} batch_size={len(batch.reqs)}"
            )
        if result.next_token_ids is None:
            raise RuntimeError("Decoupled verify result is missing verified token ids.")

        verified_ids_obj = result.next_token_ids
        if isinstance(verified_ids_obj, torch.Tensor):
            verified_ids = verified_ids_obj.tolist()
        else:
            verified_ids = []
            for item in verified_ids_obj:
                if isinstance(item, torch.Tensor):
                    item = item.tolist()
                if isinstance(item, list):
                    verified_ids.extend(int(token_id) for token_id in item)
                else:
                    verified_ids.append(int(item))

        offset = 0
        for req, accept_len in zip(batch.reqs, accept_lens):
            accept_len = int(accept_len)
            segment_len = accept_len + 1
            segment = verified_ids[offset : offset + segment_len]
            if len(segment) != segment_len:
                raise RuntimeError(
                    "Decoupled verify returned too few verified ids: "
                    f"request_id={req.rid} accept_len={accept_len} "
                    f"remaining_verified_ids={len(verified_ids) - offset}"
                )
            offset += segment_len

            pre_committed_len = getattr(
                req, "_decoupled_verify_pre_committed_len", None
            )
            if pre_committed_len is None:
                pre_committed_len = len(req.output_ids) - segment_len
            pre_committed_len = int(pre_committed_len)
            if pre_committed_len < 0:
                raise RuntimeError(
                    "Decoupled verify output is shorter than the verified segment: "
                    f"request_id={req.rid} output_len={len(req.output_ids)} "
                    f"segment_len={segment_len}"
                )

            output_segment = req.output_ids[
                pre_committed_len : pre_committed_len + segment_len
            ]
            if output_segment != segment:
                raise RuntimeError(
                    "Decoupled verify result does not match committed output ids: "
                    f"request_id={req.rid} pre_committed_len={pre_committed_len} "
                    f"accept_len={accept_len} verified_segment={segment} "
                    f"output_segment={output_segment}"
                )

            draft_buffer = list(getattr(req, "draft_buffer", []) or [])
            accepted_draft_tokens = segment[:accept_len]
            expected_draft_tokens = draft_buffer[:accept_len]
            if accepted_draft_tokens != expected_draft_tokens:
                raise RuntimeError(
                    "Decoupled verify accepted tokens outside the draft snapshot: "
                    f"request_id={req.rid} accept_len={accept_len} "
                    f"accepted_draft_tokens={accepted_draft_tokens} "
                    f"draft_snapshot_prefix={expected_draft_tokens}"
                )

        if offset != len(verified_ids):
            raise RuntimeError(
                "Decoupled verify returned extra verified ids: "
                f"consumed={offset} total={len(verified_ids)}"
            )

    def process_batch_result_idle(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        if result.copy_done is not None:
            result.copy_done.synchronize()

        self.stream_output_generation(
            batch.reqs, batch.return_logprob, is_idle_batch=True
        )

    def process_batch_result_decode(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        is_decoupled_draft = bool(batch.spec_algorithm.is_decoupled_draft())
        is_decoupled_verify = bool(batch.spec_algorithm.is_decoupled_verify())
        if result.copy_done is not None:
            result.copy_done.synchronize()

        logits_output, next_token_ids, can_run_cuda_graph = (
            result.logits_output,
            result.next_token_ids,
            result.can_run_cuda_graph,
        )

        if batch.spec_algorithm.is_none() or is_decoupled_draft or batch.is_spec_v2:
            if batch.is_spec_v2 and not is_decoupled_draft:
                next_token_ids = self._resolve_spec_overlap_token_ids(result, batch)
            else:
                next_token_ids = next_token_ids.tolist()

            if batch.return_logprob:
                next_token_logprobs = logits_output.next_token_logprobs.tolist()
                if logits_output.next_token_top_logprobs_val:
                    logits_output.next_token_top_logprobs_val = [
                        v.tolist() for v in logits_output.next_token_top_logprobs_val
                    ]
                    logits_output.next_token_top_logprobs_idx = [
                        x.tolist() for x in logits_output.next_token_top_logprobs_idx
                    ]

                if logits_output.next_token_token_ids_logprobs_val:
                    logits_output.next_token_token_ids_logprobs_val = [
                        v.tolist()
                        for v in logits_output.next_token_token_ids_logprobs_val
                    ]
        elif is_decoupled_verify:
            # Decoupled verify reuses the EAGLE/spec-v1 verify path, which
            # mutates req.output_ids and checks finish inside the worker.
            # Keep scheduler-side handling aligned with the v0.5.9 spec-v1
            # contract: do not append the returned verified ids a second time.
            self._validate_decoupled_verify_result(batch, result)
            next_token_ids = [None] * len(batch.reqs)
        else:
            # for normal spec decoding: unify next_token_ids format
            next_token_ids = []
            cum_num_tokens = 0
            next_token_ids_list = result.next_token_ids.tolist()

            for i, req in enumerate(batch.reqs):
                accept_length = result.accept_length_per_req_cpu[i]
                next_token_ids.append(
                    next_token_ids_list[
                        cum_num_tokens : cum_num_tokens + accept_length + 1
                    ]
                )
                cum_num_tokens += accept_length + 1

        self.num_generated_tokens += len(batch.reqs)
        if not batch.spec_algorithm.is_none() or is_decoupled_draft:
            self.update_spec_metrics(batch.batch_size(), result.num_accepted_tokens)

        if self.enable_metrics:
            self.metrics_collector.increment_decode_cuda_graph_pass(
                value=can_run_cuda_graph
            )

        self.token_to_kv_pool_allocator.free_group_begin()

        # NOTE: in any case, we should check finish here
        # if finished, also clean up committed kv cache and over-allocated kv cache here

        # Check finish condition
        req_iter = (
            (i, req, next_token_id)
            for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids))
        )

        # newly decoded (token_pos, token_id) for all reqs in this batch
        decoded_draft_tokens: list[Optional[tuple[int, int]]] = (
            [None] * len(batch.reqs) if is_decoupled_draft else []
        )

        for i, req, next_token_id in req_iter:
            req: Req

            if self.enable_overlap and (req.finished() or req.is_retracted):
                # NOTE: This (req.finished() or req.is_retracted) should only happen when overlap scheduling is enabled.
                # (currently not, e.g. Eagle V1 still check finish during forward)
                # And all the over-allocated tokens will be freed in `release_kv_cache`.
                continue

            new_accepted_len = 1
            if (
                batch.spec_algorithm.is_none()
                or batch.spec_algorithm.is_decoupled_draft()
            ):
                if is_decoupled_draft:
                    decoded_draft_tokens[i] = (len(req.output_ids), int(next_token_id))
                req.output_ids.append(next_token_id)
            elif batch.is_spec_v2:
                # Only spec v2's output_ids are updated here.
                req.output_ids.extend(next_token_id)
                new_accepted_len = len(next_token_id)
            elif is_decoupled_verify:
                # Output ids were already committed by EAGLE/spec-v1 verify.
                pass

            self._maybe_update_reasoning_tokens(req, next_token_id)

            # Update Mamba last track seqlen
            self._mamba_prefix_cache_update(req, batch, result, i)

            req.time_stats.set_last_decode_finish_time()

            # External decoupled drafter must not finish locally based on draft
            # tokens. The verifier still owns finished state, matching the
            # v0.5.9 spec-v1 post-processing contract.
            if not is_decoupled_draft:
                req.check_finished(new_accepted_len)

            if (
                self.server_args.disaggregation_decode_enable_offload_kvcache
                and not req.finished()
            ):
                self.decode_offload_manager.offload_kv_cache(req)

            if req.finished():
                # delete feature to save memory
                if req.multimodal_inputs is not None and req.session is None:
                    req.multimodal_inputs.release_features()
                self.maybe_collect_routed_experts(req)

                if self.server_args.disaggregation_decode_enable_offload_kvcache:
                    # Asynchronously offload KV cache; release_kv_cache will be called after Device->Host transfer completes
                    if not self.decode_offload_manager.offload_kv_cache(req):
                        self.decode_offload_manager.finalize_release_on_finish(req)
                else:
                    if self.enable_hisparse:
                        self.hisparse_coordinator.request_finished(req)
                    release_kv_cache(req, self.tree_cache)

                req.time_stats.set_completion_time()

            self.maybe_collect_customized_info(i, req, logits_output)

            if req.return_logprob and (
                batch.spec_algorithm.is_none()
                or batch.is_spec_v2
                or is_decoupled_draft
            ):
                # Spec v1 handles logprobs inside its own worker.
                # Normalize: non-spec has 1 token, spec v2 has multiple.
                if batch.is_spec_v2:
                    accepted_logprobs = next_token_logprobs[i]
                    accepted_ids = next_token_id
                    max_accept = len(accepted_logprobs)
                else:
                    accepted_logprobs = [next_token_logprobs[i]]
                    accepted_ids = [next_token_id]
                    max_accept = 1

                for j, tok_id in enumerate(accepted_ids):
                    req.output_token_logprobs_val.append(accepted_logprobs[j])
                    req.output_token_logprobs_idx.append(tok_id)
                    if req.top_logprobs_num > 0:
                        flat_idx = i * max_accept + j
                        req.output_top_logprobs_val.append(
                            logits_output.next_token_top_logprobs_val[flat_idx]
                        )
                        req.output_top_logprobs_idx.append(
                            logits_output.next_token_top_logprobs_idx[flat_idx]
                        )
                    if req.token_ids_logprob is not None:
                        flat_idx = i * max_accept + j
                        req.output_token_ids_logprobs_val.append(
                            logits_output.next_token_token_ids_logprobs_val[flat_idx]
                        )
                        req.output_token_ids_logprobs_idx.append(
                            logits_output.next_token_token_ids_logprobs_idx[flat_idx]
                        )

            if req.return_hidden_states and logits_output.hidden_states is not None:
                req.hidden_states.append(
                    logits_output.hidden_states[i].cpu().clone().tolist()
                )

            if req.grammar is not None and not is_decoupled_verify:
                # FIXME: this try-except block is for handling unexpected xgrammar issue.
                try:
                    if (
                        batch.spec_algorithm.is_none()
                        or batch.spec_algorithm.is_decoupled_draft()
                    ):
                        # Normal decode: single token
                        req.grammar.accept_token(next_token_id)
                    elif batch.is_spec_v2:
                        # Speculative decode: next_token_id is a list of accepted tokens
                        for token_id in next_token_id:
                            req.grammar.accept_token(token_id)
                except ValueError as e:
                    # Grammar accept_token can raise ValueError if the token is not in the grammar.
                    # This can happen if the grammar is not set correctly or the token is invalid.
                    logger.error(
                        f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
                    )
                    self.abort_request(AbortReq(rid=req.rid))
                req.grammar.finished = req.finished()

        if is_decoupled_draft:
            post_decode_messages = self._drain_post_decode_draft_control_messages()
            stream_outputs = (
                self._draft_process_post_decode_controls(
                    batch,
                    decoded_draft_tokens,
                    post_decode_messages,
                )
            )
            self._submit_draft_tokens_stream(stream_outputs)

        self.stream_output(batch.reqs, batch.return_logprob)
        self.token_to_kv_pool_allocator.free_group_end()

        self.forward_ct_decode = (self.forward_ct_decode + 1) % (1 << 30)
        self.report_decode_stats(
            can_run_cuda_graph,
            running_batch=batch,
            num_accepted_tokens=result.num_accepted_tokens,
        )

    def _maybe_update_reasoning_tokens(
        self: Scheduler, req: Req, next_token_id: Union[int, List[int]]
    ):
        if req.require_reasoning and self._think_end_id is not None:
            req.update_reasoning_tokens(next_token_id, self._think_end_id)

    def _mamba_prefix_cache_update(
        self: Scheduler,
        req: Req,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        i: int,
    ) -> None:
        seq_len = len(req.origin_input_ids) + len(req.output_ids) - 1
        if req.mamba_ping_pong_track_buffer is not None:
            mamba_track_interval = get_global_server_args().mamba_track_interval
            if (
                (
                    batch.spec_algorithm.is_none()
                    or batch.spec_algorithm.is_decoupled_draft()
                )
                and seq_len % mamba_track_interval == 0
            ):
                # for non-spec decode, we update mamba_last_track_seqlen at the end of each track interval
                req.mamba_next_track_idx = (
                    batch.req_to_token_pool.get_mamba_ping_pong_other_idx(
                        req.mamba_next_track_idx
                    )
                )
                req.mamba_last_track_seqlen = seq_len
            elif (
                not batch.spec_algorithm.is_none()
                and not batch.spec_algorithm.is_decoupled_draft()
                and result.accept_length_per_req_cpu is not None
            ):
                # for spec decode, update mamba_last_track_seqlen if this iteration crosses a track interval
                actual_seq_len = req.seqlen - 1
                if (
                    actual_seq_len // mamba_track_interval
                    != (actual_seq_len - result.accept_length_per_req_cpu[i])
                    // mamba_track_interval
                ):
                    req.mamba_next_track_idx = (
                        batch.req_to_token_pool.get_mamba_ping_pong_other_idx(
                            req.mamba_next_track_idx
                        )
                    )
                    req.mamba_last_track_seqlen = (
                        actual_seq_len // mamba_track_interval * mamba_track_interval
                    )

    def _process_input_token_logprobs(
        self: Scheduler, req: Req, input_token_logprobs: List
    ) -> None:
        """Process input token logprobs values and indices."""
        is_multi_item_scoring = self._is_multi_item_scoring(req)

        # Process logprob values - handle multi-item scoring vs regular requests
        if is_multi_item_scoring:
            # Multi-item scoring: use all logprobs as-is
            req.input_token_logprobs_val = input_token_logprobs
        else:
            # Regular request: add None at start, remove last (sampling token)
            req.input_token_logprobs_val = [None] + input_token_logprobs[:-1]

        # Process logprob indices based on scoring type
        if is_multi_item_scoring:
            # Multi-item scoring: only include delimiter token positions
            relevant_tokens = req.origin_input_ids[req.logprob_start_len :]
            input_token_logprobs_idx = [
                token_id
                for token_id in relevant_tokens
                if token_id == self.server_args.multi_item_scoring_delimiter
            ]
        else:
            # Regular request: include all tokens from logprob_start_len onwards
            input_token_logprobs_idx = req.origin_input_ids[req.logprob_start_len :]

        # Clip padded hash values from image tokens to prevent detokenization errors
        req.input_token_logprobs_idx = [
            x if x < self.model_config.vocab_size - 1 else 0
            for x in input_token_logprobs_idx
        ]

    def _process_input_top_logprobs(self: Scheduler, req: Req) -> None:
        """Process input top logprobs."""
        if req.top_logprobs_num <= 0:
            return

        is_multi_item_scoring = self._is_multi_item_scoring(req)

        # Initialize arrays - multi-item scoring starts empty, others start with None
        req.input_top_logprobs_val = [] if is_multi_item_scoring else [None]
        req.input_top_logprobs_idx = [] if is_multi_item_scoring else [None]

        # Extend arrays with temp values
        for val, idx in zip(
            req.temp_input_top_logprobs_val,
            req.temp_input_top_logprobs_idx,
            strict=True,
        ):
            req.input_top_logprobs_val.extend(val)
            req.input_top_logprobs_idx.extend(idx)

        # Remove last token (sampling token) for non multi-item scoring requests
        if not is_multi_item_scoring:
            req.input_top_logprobs_val.pop()
            req.input_top_logprobs_idx.pop()

        # Clean up temp storage
        req.temp_input_top_logprobs_idx = None
        req.temp_input_top_logprobs_val = None

    def _process_input_token_ids_logprobs(self: Scheduler, req: Req) -> None:
        """Process input token IDs logprobs."""
        if req.token_ids_logprob is None:
            return

        is_multi_item_scoring = self._is_multi_item_scoring(req)

        # Initialize arrays - multi-item scoring starts empty, others start with None
        req.input_token_ids_logprobs_val = [] if is_multi_item_scoring else [None]
        req.input_token_ids_logprobs_idx = [] if is_multi_item_scoring else [None]

        # Process temp values - convert tensors to lists and extend arrays
        for val, idx in zip(
            req.temp_input_token_ids_logprobs_val,
            req.temp_input_token_ids_logprobs_idx,
            strict=True,
        ):
            val_list = val.tolist() if isinstance(val, torch.Tensor) else val
            req.input_token_ids_logprobs_val.extend(
                val_list if isinstance(val_list, list) else [val_list]
            )
            req.input_token_ids_logprobs_idx.extend(idx)

        # Remove last token (sampling token) for non multi-item scoring requests
        if not is_multi_item_scoring:
            req.input_token_ids_logprobs_val.pop()
            req.input_token_ids_logprobs_idx.pop()

        # Clean up temp storage
        req.temp_input_token_ids_logprobs_idx = None
        req.temp_input_token_ids_logprobs_val = None

    def _calculate_relevant_tokens_len(self: Scheduler, req: Req) -> int:
        """Calculate the expected length of logprob arrays based on whether multi-item scoring is enabled.

        For multi-item scoring, only delimiter positions have logprobs.
        For regular requests, all positions from logprob_start_len onwards have logprobs.
        """
        is_multi_item_scoring = self._is_multi_item_scoring(req)
        relevant_tokens = req.origin_input_ids[req.logprob_start_len :]

        if is_multi_item_scoring:
            # Multi-item scoring: count delimiter tokens from logprob_start_len onwards
            return sum(
                1
                for token_id in relevant_tokens
                if token_id == self.server_args.multi_item_scoring_delimiter
            )
        else:
            # Regular request: all tokens from logprob_start_len onwards
            return len(relevant_tokens)

    def _calculate_num_input_logprobs(
        self: Scheduler, req: Req, extend_input_len: int, extend_logprob_start_len: int
    ) -> int:
        """Calculate the number of input logprobs based on whether multi-item scoring is enabled.

        For multi-item scoring, only delimiter positions have logprobs.
        For regular requests, all positions in the range have logprobs.
        """
        is_multi_item_scoring = self._is_multi_item_scoring(req)

        if is_multi_item_scoring:
            # Multi-item scoring: count delimiter tokens in the relevant portion
            relevant_tokens = req.origin_input_ids[
                extend_logprob_start_len:extend_input_len
            ]
            return sum(
                1
                for token_id in relevant_tokens
                if token_id == self.server_args.multi_item_scoring_delimiter
            )
        else:
            # Regular request: all tokens in the range
            return extend_input_len - extend_logprob_start_len

    def _is_multi_item_scoring(self: Scheduler, req: Req) -> bool:
        """Check if request uses multi-item scoring.

        Multi-item scoring applies to prefill-only requests when a delimiter
        token is configured. In this mode, only positions containing the
        delimiter token receive logprobs.
        """
        return req.is_prefill_only and self.server_args.multi_item_scoring_delimiter

    def add_input_logprob_return_values(
        self: Scheduler,
        i: int,
        req: Req,
        output: LogitsProcessorOutput,
        logprob_pt: int,
        num_input_logprobs: int,
        last_prefill_chunk: bool,  # If True, it means prefill is finished.
    ):
        """Incrementally add input logprobs to `req`.

        Args:
            i: The request index in a batch.
            req: The request. Input logprobs inside req are modified as a
                consequence of the API
            fill_ids: The prefill ids processed.
            output: Logit processor output that's used to compute input logprobs
            last_prefill_chunk: True if it is the last prefill (when chunked).
                Some of input logprob operation should only happen at the last
                prefill (e.g., computing input token logprobs).
        """
        assert output.input_token_logprobs is not None
        if req.input_token_logprobs is None:
            req.input_token_logprobs = []
        if req.temp_input_top_logprobs_val is None:
            req.temp_input_top_logprobs_val = []
        if req.temp_input_top_logprobs_idx is None:
            req.temp_input_top_logprobs_idx = []
        if req.temp_input_token_ids_logprobs_val is None:
            req.temp_input_token_ids_logprobs_val = []
        if req.temp_input_token_ids_logprobs_idx is None:
            req.temp_input_token_ids_logprobs_idx = []

        if req.input_token_logprobs_val is not None:
            # The input logprob has been already computed. It only happens
            # upon retract.
            if req.top_logprobs_num > 0:
                assert req.input_token_logprobs_val is not None
            return

        # Important for the performance.
        assert isinstance(output.input_token_logprobs, tuple)
        input_token_logprobs: Tuple[int] = output.input_token_logprobs
        input_token_logprobs = input_token_logprobs[
            logprob_pt : logprob_pt + num_input_logprobs
        ]
        req.input_token_logprobs.extend(input_token_logprobs)

        if req.top_logprobs_num > 0:
            req.temp_input_top_logprobs_val.append(output.input_top_logprobs_val[i])
            req.temp_input_top_logprobs_idx.append(output.input_top_logprobs_idx[i])

        if req.token_ids_logprob is not None:
            req.temp_input_token_ids_logprobs_val.append(
                output.input_token_ids_logprobs_val[i]
            )
            req.temp_input_token_ids_logprobs_idx.append(
                output.input_token_ids_logprobs_idx[i]
            )

        if last_prefill_chunk:
            input_token_logprobs = req.input_token_logprobs
            req.input_token_logprobs = None
            assert req.input_token_logprobs_val is None
            assert req.input_token_logprobs_idx is None
            assert req.input_top_logprobs_val is None
            assert req.input_top_logprobs_idx is None

            # Process all input logprob types using helper functions
            self._process_input_token_logprobs(req, input_token_logprobs)
            self._process_input_top_logprobs(req)

            self._process_input_token_ids_logprobs(req)

            if req.return_logprob:
                relevant_tokens_len = self._calculate_relevant_tokens_len(req)
                assert len(req.input_token_logprobs_val) == relevant_tokens_len
                assert len(req.input_token_logprobs_idx) == relevant_tokens_len
                if req.top_logprobs_num > 0:
                    assert len(req.input_top_logprobs_val) == relevant_tokens_len
                    assert len(req.input_top_logprobs_idx) == relevant_tokens_len
                if req.token_ids_logprob is not None:
                    assert len(req.input_token_ids_logprobs_val) == relevant_tokens_len
                    assert len(req.input_token_ids_logprobs_idx) == relevant_tokens_len

    def add_logprob_return_values(
        self: Scheduler,
        i: int,
        req: Req,
        pt: int,
        next_token_ids: List[int],
        num_input_logprobs: int,
        output: LogitsProcessorOutput,
    ):
        """Attach logprobs to the return values."""
        if output.next_token_logprobs is not None:
            req.output_token_logprobs_val.append(output.next_token_logprobs[i])
            req.output_token_logprobs_idx.append(next_token_ids[i])

        # Only add input logprobs if there are input tokens to process
        # Note: For prefill-only requests with default logprob_start_len, this will be 0,
        # meaning we only compute output logprobs (which is the intended behavior)
        if num_input_logprobs > 0:
            self.add_input_logprob_return_values(
                i, req, output, pt, num_input_logprobs, last_prefill_chunk=True
            )
        else:
            self._initialize_empty_logprob_containers(req)

        if req.top_logprobs_num > 0:
            req.output_top_logprobs_val.append(output.next_token_top_logprobs_val[i])
            req.output_top_logprobs_idx.append(output.next_token_top_logprobs_idx[i])

        if (
            req.token_ids_logprob is not None
            and output.next_token_token_ids_logprobs_val is not None
        ):
            # Convert GPU tensor to list if needed
            logprobs_val = output.next_token_token_ids_logprobs_val[i]
            if isinstance(logprobs_val, torch.Tensor):
                logprobs_val = logprobs_val.tolist()
            req.output_token_ids_logprobs_val.append(logprobs_val)
            req.output_token_ids_logprobs_idx.append(
                output.next_token_token_ids_logprobs_idx[i]
            )

        return num_input_logprobs

    def _initialize_empty_logprob_containers(self: Scheduler, req: Req) -> None:
        """
        Initialize logprob fields to empty lists if unset.

        This is needed for prefill-only requests where the normal initialization
        flow might be bypassed, but downstream code expects these fields to be lists.
        """
        if req.input_token_logprobs_val is None:
            req.input_token_logprobs_val = []
        if req.input_token_logprobs_idx is None:
            req.input_token_logprobs_idx = []
        if req.input_top_logprobs_val is None:
            req.input_top_logprobs_val = []
        if req.input_top_logprobs_idx is None:
            req.input_top_logprobs_idx = []
        if req.input_token_ids_logprobs_val is None:
            req.input_token_ids_logprobs_val = []
        if req.input_token_ids_logprobs_idx is None:
            req.input_token_ids_logprobs_idx = []

    def stream_output(
        self: Scheduler,
        reqs: List[Req],
        return_logprob: bool,
        skip_req: Optional[Req] = None,
    ):
        """Stream the output to detokenizer."""
        if self.spec_algorithm.is_decoupled_draft():
            return
        if self.is_generation:
            self.stream_output_generation(reqs, return_logprob, skip_req)
        else:  # embedding or reward model
            self.stream_output_embedding(reqs)

        if envs.SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS.get() > 0:
            self._trigger_crash_for_tests(
                envs.SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS.get()
            )

    def _trigger_crash_for_tests(self: Scheduler, crash_threshold: int):
        # Crash trigger: crash after stream_output is called N times
        # This is used for testing purposes.
        if not hasattr(self, "_test_stream_output_count"):
            self._test_stream_output_count = 0
        self._test_stream_output_count += 1
        if self._test_stream_output_count >= crash_threshold:
            raise RuntimeError(
                f"Test crash after stream_output called {self._test_stream_output_count} times"
            )

    def stream_output_generation(
        self: Scheduler,
        reqs: List[Req],
        return_logprob: bool,
        skip_req: Optional[Req] = None,
        is_idle_batch: bool = False,
    ):
        rids = []
        http_worker_ipcs = []
        finished_reasons: List[BaseFinishReason] = []

        decoded_texts = []
        decode_ids_list = []
        read_offsets = []
        output_ids = []

        skip_special_tokens = []
        spaces_between_special_tokens = []
        no_stop_trim = []
        prompt_tokens = []
        reasoning_tokens = []
        completion_tokens = []
        cached_tokens = []
        cached_tokens_details = []  # Detailed breakdown by cache source
        spec_verify_ct = []
        spec_accepted_tokens = []
        spec_acceptance_histogram = []
        retraction_counts = []
        output_hidden_states = None
        load = self.get_load()
        routed_experts = None
        customized_info = {}

        time_stats = []

        if return_logprob:
            input_token_logprobs_val = []
            input_token_logprobs_idx = []
            output_token_logprobs_val = []
            output_token_logprobs_idx = []
            input_top_logprobs_val = []
            input_top_logprobs_idx = []
            output_top_logprobs_val = []
            output_top_logprobs_idx = []
            input_token_ids_logprobs_val = []
            input_token_ids_logprobs_idx = []
            output_token_ids_logprobs_val = []
            output_token_ids_logprobs_idx = []
        else:
            input_token_logprobs_val = input_token_logprobs_idx = (
                output_token_logprobs_val
            ) = output_token_logprobs_idx = input_top_logprobs_val = (
                input_top_logprobs_idx
            ) = output_top_logprobs_val = output_top_logprobs_idx = (
                input_token_ids_logprobs_val
            ) = input_token_ids_logprobs_idx = output_token_ids_logprobs_val = (
                output_token_ids_logprobs_idx
            ) = None

        for req in reqs:
            if req is skip_req:
                continue

            if req.finished():
                if req.finished_output:
                    # With the overlap schedule, a request will try to output twice and hit this line twice
                    # because of the one additional delayed token. This "continue" prevented the dummy output.
                    continue
                req.finished_output = True
                if req.finished_len is None:
                    req.finished_len = len(req.output_ids)
                should_output = True
            else:
                if req.stream:
                    stream_interval = (
                        req.sampling_params.stream_interval or self.stream_interval
                    )

                    # origin stream_interval logic
                    should_output = (
                        len(req.output_ids) % stream_interval == 1
                        if stream_interval > 1
                        else len(req.output_ids) % stream_interval == 0
                    )

                    if should_output:
                        # check_match_stop_str_prefix if  tail_str's suffix match stop_str prefix
                        should_output &= not req.check_match_stop_str_prefix()
                else:
                    should_output = (
                        len(req.output_ids) % DEFAULT_FORCE_STREAM_INTERVAL == 0
                    )

            if should_output:
                send_token_offset = req.send_token_offset
                send_output_token_logprobs_offset = (
                    req.send_output_token_logprobs_offset
                )
                rids.append(req.rid)
                http_worker_ipcs.append(req.http_worker_ipc)
                finished_reasons.append(
                    req.finished_reason.to_json() if req.finished_reason else None
                )
                decoded_texts.append(req.decoded_text)
                decode_ids, read_offset = req.init_incremental_detokenize()

                decode_ids_list.append(decode_ids[req.send_decode_id_offset :])

                # Exclude the tokens after stop condition
                output_ids_ = req.output_ids_through_stop

                req.send_decode_id_offset = len(decode_ids)
                read_offsets.append(read_offset)
                output_ids.append(output_ids_[send_token_offset:])
                req.send_token_offset = len(output_ids_)
                skip_special_tokens.append(req.sampling_params.skip_special_tokens)
                spaces_between_special_tokens.append(
                    req.sampling_params.spaces_between_special_tokens
                )
                no_stop_trim.append(req.sampling_params.no_stop_trim)
                prompt_tokens.append(len(req.origin_input_ids))
                reasoning_tokens.append(req.reasoning_tokens)
                completion_tokens.append(len(output_ids_))
                cached_tokens.append(req.cached_tokens)

                # Collect detailed cache breakdown if available
                cached_tokens_details.append(self._get_cached_tokens_details(req))

                retraction_counts.append(req.retraction_count)

                time_stats.append(req.time_stats)

                if not self.spec_algorithm.is_none() and not self.spec_algorithm.is_decoupled_draft():
                    spec_verify_ct.append(req.spec_verify_ct)
                    spec_accepted_tokens.append(req.spec_accepted_tokens)
                    spec_acceptance_histogram.append(req.spec_acceptance_histogram)

                if return_logprob:
                    if (
                        req.return_logprob
                        and not req.input_logprob_sent
                        # Decode server does not send input logprobs
                        and self.disaggregation_mode != DisaggregationMode.DECODE
                        # Only send when input logprobs have been computed (after prefill)
                        and req.input_token_logprobs_val is not None
                    ):
                        input_token_logprobs_val.append(req.input_token_logprobs_val)
                        input_token_logprobs_idx.append(req.input_token_logprobs_idx)
                        input_top_logprobs_val.append(req.input_top_logprobs_val)
                        input_top_logprobs_idx.append(req.input_top_logprobs_idx)
                        input_token_ids_logprobs_val.append(
                            req.input_token_ids_logprobs_val
                        )
                        input_token_ids_logprobs_idx.append(
                            req.input_token_ids_logprobs_idx
                        )
                        req.input_logprob_sent = True
                    else:
                        input_token_logprobs_val.append([])
                        input_token_logprobs_idx.append([])
                        input_top_logprobs_val.append([])
                        input_top_logprobs_idx.append([])
                        input_token_ids_logprobs_val.append([])
                        input_token_ids_logprobs_idx.append([])

                    if req.return_logprob:
                        logprob_end = max(len(output_ids_), 1)
                        output_token_logprobs_val.append(
                            req.output_token_logprobs_val[
                                send_output_token_logprobs_offset:logprob_end
                            ]
                        )
                        output_token_logprobs_idx.append(
                            req.output_token_logprobs_idx[
                                send_output_token_logprobs_offset:logprob_end
                            ]
                        )
                        output_top_logprobs_val.append(
                            req.output_top_logprobs_val[
                                send_output_token_logprobs_offset:logprob_end
                            ]
                        )
                        output_top_logprobs_idx.append(
                            req.output_top_logprobs_idx[
                                send_output_token_logprobs_offset:logprob_end
                            ]
                        )
                        output_token_ids_logprobs_val.append(
                            req.output_token_ids_logprobs_val[
                                send_output_token_logprobs_offset:logprob_end
                            ]
                        )
                        output_token_ids_logprobs_idx.append(
                            req.output_token_ids_logprobs_idx[
                                send_output_token_logprobs_offset:logprob_end
                            ]
                        )
                        req.send_output_token_logprobs_offset = logprob_end
                    else:
                        output_token_logprobs_val.append([])
                        output_token_logprobs_idx.append([])
                        output_top_logprobs_val.append([])
                        output_top_logprobs_idx.append([])
                        output_token_ids_logprobs_val.append([])
                        output_token_ids_logprobs_idx.append([])

                if req.return_hidden_states:
                    if output_hidden_states is None:
                        output_hidden_states = []
                    output_hidden_states.append(req.hidden_states)
                if req.return_routed_experts:
                    if routed_experts is None:
                        routed_experts = []
                    routed_experts.append(req.routed_experts)

                if req.customized_info is not None:
                    for k, v in req.customized_info.items():
                        if k not in customized_info:
                            customized_info[k] = []
                        customized_info[k].append(
                            v[send_token_offset : len(output_ids_)]
                        )

            if (
                req.finished()
                and self.attn_tp_rank == 0
                and self.server_args.enable_request_time_stats_logging
            ):
                req.log_time_stats()

        dp_ranks = [self.dp_rank] * len(rids) if rids else None

        # Send to detokenizer
        if reqs or is_idle_batch:
            self.send_to_detokenizer.send_output(
                BatchTokenIDOutput(
                    rids=rids,
                    http_worker_ipcs=http_worker_ipcs,
                    spec_verify_ct=spec_verify_ct,
                    spec_accepted_tokens=spec_accepted_tokens,
                    spec_acceptance_histogram=spec_acceptance_histogram,
                    time_stats=time_stats,
                    finished_reasons=finished_reasons,
                    decoded_texts=decoded_texts,
                    decode_ids=decode_ids_list,
                    read_offsets=read_offsets,
                    output_ids=output_ids,
                    skip_special_tokens=skip_special_tokens,
                    spaces_between_special_tokens=spaces_between_special_tokens,
                    no_stop_trim=no_stop_trim,
                    prompt_tokens=prompt_tokens,
                    reasoning_tokens=reasoning_tokens,
                    completion_tokens=completion_tokens,
                    cached_tokens=cached_tokens,
                    cached_tokens_details=cached_tokens_details,
                    input_token_logprobs_val=input_token_logprobs_val,
                    input_token_logprobs_idx=input_token_logprobs_idx,
                    output_token_logprobs_val=output_token_logprobs_val,
                    output_token_logprobs_idx=output_token_logprobs_idx,
                    input_top_logprobs_val=input_top_logprobs_val,
                    input_top_logprobs_idx=input_top_logprobs_idx,
                    output_top_logprobs_val=output_top_logprobs_val,
                    output_top_logprobs_idx=output_top_logprobs_idx,
                    input_token_ids_logprobs_val=input_token_ids_logprobs_val,
                    input_token_ids_logprobs_idx=input_token_ids_logprobs_idx,
                    output_token_ids_logprobs_val=output_token_ids_logprobs_val,
                    output_token_ids_logprobs_idx=output_token_ids_logprobs_idx,
                    output_token_entropy_val=None,
                    output_hidden_states=output_hidden_states,
                    routed_experts=routed_experts,
                    customized_info=customized_info,
                    placeholder_tokens_idx=None,
                    placeholder_tokens_val=None,
                    retraction_counts=retraction_counts,
                    load=load,
                    dp_ranks=dp_ranks,
                )
            )

    def stream_output_embedding(self: Scheduler, reqs: List[Req]):
        rids = []
        http_worker_ipcs = []
        finished_reasons: List[BaseFinishReason] = []

        embeddings = []
        prompt_tokens = []
        cached_tokens = []
        cached_tokens_details = []  # Detailed breakdown by cache source
        time_stats = []
        retraction_counts = []
        for req in reqs:
            if req.finished():
                rids.append(req.rid)
                http_worker_ipcs.append(req.http_worker_ipc)
                finished_reasons.append(req.finished_reason.to_json())
                embeddings.append(req.embedding)
                prompt_tokens.append(len(req.origin_input_ids))
                cached_tokens.append(req.cached_tokens)

                # Collect detailed cache breakdown if available
                cached_tokens_details.append(self._get_cached_tokens_details(req))
                time_stats.append(req.time_stats)
                retraction_counts.append(req.retraction_count)
        self.send_to_detokenizer.send_output(
            BatchEmbeddingOutput(
                rids=rids,
                http_worker_ipcs=http_worker_ipcs,
                time_stats=time_stats,
                finished_reasons=finished_reasons,
                embeddings=embeddings,
                prompt_tokens=prompt_tokens,
                cached_tokens=cached_tokens,
                cached_tokens_details=cached_tokens_details,
                placeholder_tokens_idx=None,
                placeholder_tokens_val=None,
                retraction_counts=retraction_counts,
            )
        )
