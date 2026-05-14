from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Deque, Dict, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.decoupled_spec_io import (
    DraftClose,
    DraftControlBatch,
    DraftControlMessage,
    DraftReqKey,
    DraftSync,
    DraftTailStreamOutput,
    DraftTailStreamOutputBatch,
    VerifyCommit,
    build_draft_scheduler_rid,
    parse_draft_scheduler_rid,
)
from sglang.srt.speculative.decoupled_spec_trace import (
    DecoupledSpecTraceEvent,
    build_decoupled_spec_tracer,
    trace_decoupled_spec,
)
from sglang.srt.speculative.draft_proxy import DraftProxyThread
from sglang.srt.speculative.draft_tail_buffer import DraftTailBuffer, DraftTailSnapshot
from sglang.srt.speculative.token_sync_thread import TokenSyncThread
from sglang.srt.utils import broadcast_pyobj

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler

logger = logging.getLogger(__name__)


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


class SchedulerDecoupledSpecMixin:
    """Decoupled-spec scheduler hooks and request lifecycle helpers."""

    def get_decoupled_spec_rank(self: Scheduler) -> int:
        rank = self.server_args.decoupled_spec_rank
        if rank is None:
            raise RuntimeError("decoupled_spec_rank is required for decoupled spec")
        return int(rank)

    def is_draft_worker_batch(
        self: Scheduler, batch: Optional[ScheduleBatch] = None
    ) -> bool:
        spec_algorithm = (
            batch.spec_algorithm if batch is not None else self.spec_algorithm
        )
        return bool(spec_algorithm.is_decoupled_draft())

    def is_verify_worker_batch(
        self: Scheduler, batch: Optional[ScheduleBatch] = None
    ) -> bool:
        spec_algorithm = (
            batch.spec_algorithm if batch is not None else self.spec_algorithm
        )
        return bool(spec_algorithm.is_decoupled_verify())

    def create_draft_tail_buffer(self: Scheduler) -> Optional[DraftTailBuffer]:
        if not self.is_verify_entry_rank():
            return None
        return DraftTailBuffer(
            verifier_rank=self.get_decoupled_spec_rank(),
            required_tail_len=max(
                0, int(self.server_args.speculative_num_draft_tokens) - 1
            ),
        )

    def start_verify_proxy(self: Scheduler, context) -> None:
        self.draft_proxy_thread = None
        if not self.is_verify_entry_rank():
            return
        ipc_config = self.port_args.decoupled_spec_ipc_config
        if ipc_config is None:
            raise RuntimeError(
                "Decoupled spec IPC config is required on decoupled_verify entry rank"
            )
        if self.draft_tail_buffer is None:
            raise RuntimeError(
                "DraftTailBuffer is required on decoupled_verify entry rank"
            )
        self.draft_proxy_thread = DraftProxyThread(
            context=context,
            verifier_rank=ipc_config.rank,
            result_bind_endpoint=ipc_config.bind_endpoint,
            drafter_control_endpoints=ipc_config.connect_endpoints,
            draft_tail_buffer=self.draft_tail_buffer,
            tracer=self.decoupled_spec_tracer,
        )
        self.draft_proxy_thread.start()

    def start_token_sync_thread(self: Scheduler) -> None:
        self.token_sync_thread = None
        if not self.is_draft_entry_rank():
            return
        ipc_config = self.port_args.decoupled_spec_ipc_config
        if ipc_config is None:
            raise RuntimeError(
                "Decoupled spec IPC config is required on decoupled_draft entry rank"
            )
        self.token_sync_thread = TokenSyncThread(
            context=getattr(self, "zmq_context", None),
            control_bind_endpoint=ipc_config.bind_endpoint,
            verifier_result_endpoints=ipc_config.connect_endpoints,
            drafter_rank=ipc_config.rank,
            tracer=self.decoupled_spec_tracer,
        )
        self.token_sync_thread.start()

    def init_draft_state_tables(self: Scheduler) -> None:
        self.draft_req_table: Dict[DraftReqKey, DraftReqState] = {}
        self.decoupled_verify_drafter_ranks: list[int] = []
        self.decoupled_verify_req_to_drafter_rank: Dict[str, int] = {}
        self.decoupled_verify_drafter_loads: Dict[int, int] = {}
        if not self.is_verify_entry_rank():
            return
        ipc_config = self.port_args.decoupled_spec_ipc_config
        if ipc_config is None:
            raise RuntimeError(
                "Decoupled spec IPC config is required on decoupled_verify entry rank"
            )
        self.decoupled_verify_drafter_ranks = list(
            range(len(ipc_config.connect_endpoints))
        )
        if not self.decoupled_verify_drafter_ranks:
            raise RuntimeError(
                "Decoupled verify requires at least one drafter control endpoint"
            )
        self.decoupled_verify_drafter_loads = {
            rank: 0 for rank in self.decoupled_verify_drafter_ranks
        }

    def start_forward_timer(self: Scheduler, batch: ScheduleBatch) -> Optional[int]:
        if (
            not batch
            or not self.is_generation
            or not (
                self.spec_algorithm.is_decoupled_verify()
                or self.spec_algorithm.is_decoupled_draft()
                or self.spec_algorithm.is_none()
            )
        ):
            return None
        start_ns = self.decoupled_spec_tracer.start_timer()
        if start_ns is None:
            return None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return self.decoupled_spec_tracer.start_timer()

    @trace_decoupled_spec(DecoupledSpecTraceEvent.SCHEDULER_FORWARD_BATCH)
    def record_forward_latency(
        self: Scheduler,
        batch: ScheduleBatch,
        start_ns: Optional[int],
        result: object | None = None,
    ) -> None:
        if start_ns is None:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def flush_draft_updates(
        self: Scheduler,
        batch: ScheduleBatch,
        decoded_tokens: list[Optional[tuple[int, int]]],
    ) -> None:
        if not self.is_draft_worker_batch(batch):
            return
        post_decode_messages = self._drain_draft_update_messages()
        stream_outputs = self._process_draft_updates(
            batch, decoded_tokens, post_decode_messages
        )
        self._submit_draft_tokens_stream(stream_outputs)

    def prepare_verify_inputs(self: Scheduler, batch: ScheduleBatch) -> None:
        if not self.is_verify_worker_batch(batch):
            return
        if batch.forward_mode.is_extend():
            self._sync_verify_requests(batch)
        elif batch.forward_mode.is_decode():
            self._snapshot_verify_inputs(batch)

    def create_spec_tracer(self):
        enabled = bool(
            getattr(self.server_args, "decoupled_spec_trace_dir", None)
            and (
                self.spec_algorithm.is_decoupled_verify()
                or self.spec_algorithm.is_decoupled_draft()
                or self.spec_algorithm.is_none()
            )
        )
        dp_rank = self.dp_rank or 0
        rank_suffix = f"dp{dp_rank}_tp{self.tp_rank}_pp{self.pp_rank}"
        if self.spec_algorithm.is_decoupled_verify():
            forward_trace_prefix = "verifier"
        elif self.spec_algorithm.is_decoupled_draft():
            forward_trace_prefix = "drafter"
        elif self.spec_algorithm.is_none():
            forward_trace_prefix = "decode"
        else:
            forward_trace_prefix = "scheduler"
        file_names = {
            "scheduler.forward_batch": (
                f"{forward_trace_prefix}-forward-batch_{rank_suffix}.csv"
            ),
            "verifier": f"verifier_{rank_suffix}.csv",
            "drafter": f"drafter_{rank_suffix}.csv",
            "draft_proxy": f"draft_proxy_verifier{dp_rank}.csv",
            "token_sync_thread": f"token_sync_thread_drafter{dp_rank}.csv",
        }
        return build_decoupled_spec_tracer(
            enabled=enabled,
            output_dir=getattr(self.server_args, "decoupled_spec_trace_dir", None),
            file_names=file_names,
        )

    def _infer_forward_graph_path(
        self: Scheduler, batch: ScheduleBatch, can_run_cuda_graph: bool
    ) -> str:
        if not can_run_cuda_graph:
            return "eager"
        if batch.forward_mode.is_decode():
            if self.spec_algorithm.is_decoupled_verify():
                return "cuda_graph_target_verify"
            return "cuda_graph_decode"
        if batch.forward_mode.is_extend(include_draft_extend_v2=True):
            return "piecewise_cuda_graph"
        if batch.forward_mode.is_idle():
            return "cuda_graph_idle"
        return "cuda_graph"

    def is_draft_entry_rank(self: Scheduler) -> bool:
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

    def _get_token_sync_thread(self: Scheduler):
        token_sync_thread = self.token_sync_thread
        if token_sync_thread is None:
            raise RuntimeError("Decoupled draft entry rank has no token sync thread")
        return token_sync_thread

    def _get_or_create_draft_state(
        self: Scheduler,
        draft_key: DraftReqKey,
    ) -> DraftReqState:
        state = self.draft_req_table.get(draft_key)
        if state is None:
            state = DraftReqState(key=draft_key)
            self.draft_req_table[draft_key] = state
        return state

    def _get_draft_state_by_req(self: Scheduler, req: Req) -> DraftReqState:
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
        if not self.is_draft_entry_rank():
            return
        self._get_token_sync_thread().submit_draft_results(stream_output_batch)

    def apply_verify_commit(
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
        1. overwrite the bonus token and update output_ids, grammar,
           KV cache, stream output, etc. when needed
        2. update the scheduler state's verifier_committed_prefix_len to (bonus_token_pos + 1)
        """
        state = self._get_draft_state_by_req(req)
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
                req.cache_protected_len = min(req.cache_protected_len, kv_truncate_from)

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

    def release_draft_request(self: Scheduler, req: Req) -> None:
        """
        release a draft request only when it has completed at the verifier side:
        1. evict the req from waiting_queue or running_batch
        2. remove the req from the draft request table
        3. release its kvcache
        """
        state = self._get_draft_state_by_req(req)

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

    def _create_draft_request(
        self: Scheduler,
        message: DraftSync,
    ) -> Req:
        """
        Create and register a new drafter-side request from DraftSync.
        """
        state = self._get_or_create_draft_state(message.draft_key)
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
            # Keep sampling until receiving DraftClose.
            max_new_tokens=1 << 30,
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

    def _drain_draft_update_messages(
        self: Scheduler,
    ) -> list[VerifyCommit | DraftClose]:
        """
        (called by decoupled drafter)
        Drain all VerifyCommit and DraftClose messages from token sync thread
        """
        messages: list[DraftControlMessage] | None = None
        if self.is_draft_entry_rank():
            messages = self._get_token_sync_thread().drain_post_result_messages()

        return [
            message
            for message in self._broadcast_draft_control_messages(messages)
            if isinstance(message, (VerifyCommit, DraftClose))
        ]

    @trace_decoupled_spec(DecoupledSpecTraceEvent.DRAFTER_SYNC_DRAFT_REQUESTS)
    def sync_draft_requests(self: Scheduler) -> dict | None:
        """
        (called by decoupled drafter)
        Drain DraftSync messages from token sync thread, and handle them.
        DraftSync creates a new drafter-side request from verifier state.
        """
        if not self.spec_algorithm.is_decoupled_draft():
            return None

        messages: list[DraftControlMessage] | None = None
        if self.is_draft_entry_rank():
            messages = self._get_token_sync_thread().drain_sync_messages()

        messages = [
            message
            for message in self._broadcast_draft_control_messages(messages)
            if isinstance(message, DraftSync)
        ]
        if not messages:
            return None

        created_reqs: list[Req] = []
        for message in messages:
            entry = self.draft_req_table.get(message.draft_key)
            if entry is not None:
                if entry.pending_close is not None:
                    self.draft_req_table.pop(message.draft_key, None)
                    continue
            req = self._create_draft_request(message)
            running_batch = self.running_batch
            if req not in self.waiting_queue and req not in running_batch.reqs:
                self._add_request_to_queue(req)
            created_reqs.append(req)
        return {
            "recv_batch_size": len(messages),
            "recv_rids": [message.request_id for message in messages],
            "recv_committed_lens_by_req": [
                len(message.committed_output_ids) for message in messages
            ],
            "recv_output_lens_by_req": [
                len(message.committed_output_ids) for message in messages
            ],
            "num_created_reqs": len(created_reqs),
            "created_rids": [
                parse_draft_scheduler_rid(req.rid).request_id for req in created_reqs
            ],
            "created_committed_lens_by_req": [
                int(self._get_draft_state_by_req(req).verifier_committed_prefix_len)
                for req in created_reqs
            ],
            "created_output_lens_by_req": [
                len(req.output_ids) for req in created_reqs
            ],
        }

    def _apply_commits_and_maybe_emit(
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
        state = self._get_draft_state_by_req(req)
        result = DraftCommitApplyResult()

        commits_to_apply: list[VerifyCommit] = []
        if state.pending_verify_commits:
            commits_to_apply.extend(state.pending_verify_commits)
            state.pending_verify_commits.clear()

        if commits:
            commits_to_apply.extend(commits)

        for commit_idx, verify_commit in enumerate(commits_to_apply):
            if int(verify_commit.bonus_token_pos) >= len(req.output_ids):
                # if a drafter req has multiple pending verify commits,
                # it can only apply the first few commits that update the bonus token
                # within the current output_ids range.
                result.deferred_commits.extend(commits_to_apply[commit_idx:])
                state.pending_verify_commits.extend(commits_to_apply[commit_idx:])
                break
            self.apply_verify_commit(
                req,
                verify_commit,
                batch=batch,
                req_batch_idx=req_batch_idx,
            )
            result.applied_commits.append(verify_commit)

        if not self.is_draft_entry_rank() or decoded_token is None:
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

    @trace_decoupled_spec(DecoupledSpecTraceEvent.DRAFTER_PROCESS_UPDATES)
    def _process_draft_updates(
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
        # build draft_key -> req mapping
        current_req_by_key: dict[DraftReqKey, Req] = {}
        decoded_token_by_key: dict[DraftReqKey, Optional[tuple[int, int]]] = {}
        for req_batch_idx, req in enumerate(batch.reqs):
            state = self._get_draft_state_by_req(req)
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
                    entry = self._get_or_create_draft_state(draft_key)
                entry.pending_verify_commits.clear()
                if req is not None:
                    self.release_draft_request(req)
                else:
                    entry.req = None
                    entry.pending_close = message
                continue

            if draft_key in closed_keys:
                continue

            if entry is None:
                entry = self._get_or_create_draft_state(draft_key)

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
        src_drafter_rank = self.get_decoupled_spec_rank()
        applied_commit_messages: list[VerifyCommit] = []
        for req_batch_idx, req in enumerate(batch.reqs):
            draft_key = self._get_draft_state_by_req(req).key
            decoded_token = decoded_token_by_key.get(draft_key)
            apply_result = self._apply_commits_and_maybe_emit(
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
        trace_payload = {
            "forward_mode": str(batch.forward_mode),
            "batch_size": len(batch.reqs),
            "control_rids": [message.request_id for message in control_messages],
            "num_commit": len(commit_messages),
            "num_close": len(close_messages),
            "applied_commit_rids": [
                message.request_id for message in applied_commit_messages
            ],
            "applied_committed_lens_by_req": [
                int(message.bonus_token_pos) + 1
                for message in applied_commit_messages
            ],
            "num_applied_commit": len(applied_commit_messages),
            "num_deferred_commit": len(deferred_commit_messages),
            "batch_rids": [
                self._get_draft_state_by_req(req).key.request_id for req in batch.reqs
            ],
            "num_stream_outputs": len(stream_output_batch.outputs),
            "committed_lens_by_req": [
                int(self._get_draft_state_by_req(req).verifier_committed_prefix_len)
                for req in batch.reqs
            ],
            "output_lens_by_req": [len(req.output_ids) for req in batch.reqs],
        }
        setattr(stream_output_batch, "_decoupled_spec_payload", trace_payload)
        return stream_output_batch

    def validate_verify_outputs(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:
        accept_lens = result.num_correct_drafts_per_req_cpu
        # Compatibility with older decoupled verifier results during rolling
        # migrations from v0.5.10-dev.
        if accept_lens is None:
            accept_lens = getattr(result, "num_accepted_drafts_per_req_cpu", None)
        if accept_lens is None:
            accept_lens = getattr(result, "accept_length_per_req_cpu", None)
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
        valid_draft_metric_updates: list[tuple[Req, int, int]] = []
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
            valid_draft_metric_updates.append((req, len(draft_buffer), accept_len))

        if offset != len(verified_ids):
            raise RuntimeError(
                "Decoupled verify returned extra verified ids: "
                f"consumed={offset} total={len(verified_ids)}"
            )
        for (
            req,
            valid_draft_tokens,
            valid_accepted_tokens,
        ) in valid_draft_metric_updates:
            req.spec_valid_draft_tokens += valid_draft_tokens
            req.spec_valid_accepted_tokens += valid_accepted_tokens

    def is_verify_entry_rank(self) -> bool:
        return (
            self.spec_algorithm.is_decoupled_verify()
            and self.pp_rank == 0
            and self.attn_tp_rank == 0
            and self.attn_cp_rank == 0
        )

    def assign_drafter_rank(self, request_id: str) -> int:
        """Assign a verifier request to the currently least-loaded drafter rank."""
        drafter_rank = self.decoupled_verify_req_to_drafter_rank.get(request_id)
        if drafter_rank is not None:
            return drafter_rank

        if not self.decoupled_verify_drafter_ranks:
            raise RuntimeError(
                "Decoupled verify drafter ranks are not initialized on entry rank"
            )
        drafter_rank = min(
            self.decoupled_verify_drafter_ranks,
            key=lambda rank: (
                self.decoupled_verify_drafter_loads.get(rank, 0),
                (rank - self.get_decoupled_spec_rank())
                % len(self.decoupled_verify_drafter_ranks),
            ),
        )
        self.decoupled_verify_req_to_drafter_rank[request_id] = drafter_rank
        self.decoupled_verify_drafter_loads[drafter_rank] = (
            self.decoupled_verify_drafter_loads.get(drafter_rank, 0) + 1
        )
        return drafter_rank

    def get_drafter_rank(self, request_id: str) -> int:
        """Return the drafter rank already assigned to a verifier request."""
        drafter_rank = self.decoupled_verify_req_to_drafter_rank.get(request_id)
        if drafter_rank is None:
            raise RuntimeError(
                "Missing decoupled verify drafter assignment for "
                f"request_id={request_id}"
            )
        return drafter_rank

    def release_drafter_rank(self, request_id: str) -> None:
        """Release one request's drafter assignment after close/abort."""
        drafter_rank = self.decoupled_verify_req_to_drafter_rank.pop(request_id, None)
        if drafter_rank is None:
            return
        self.decoupled_verify_drafter_loads[drafter_rank] = max(
            0,
            self.decoupled_verify_drafter_loads.get(drafter_rank, 0) - 1,
        )

    def _submit_verify_control_batch(self, batch: DraftControlBatch) -> None:
        """
        Submit one verifier-to-drafter control batch.

        Called after scheduler build DraftSync, VerifyCommit, or
        DraftClose messages. The entry rank forwards the batch to
        DraftProxyThread, which both updates the entry-rank DraftTailBuffer
        locally and sends the batch asynchronously. Non-entry ranks do not own
        draft transport state and return without side effects.

        Args:
            batch: A batch of control messages for one drafter rank.

        Returns:
            None.
        """
        if not self.is_verify_entry_rank():
            return

        if self.draft_proxy_thread is None:
            raise RuntimeError(
                "Draft proxy thread is not initialized on decoupled_verify entry rank"
            )
        self.draft_proxy_thread.submit_control_batch(batch)

    def _send_verify_control_batches(
        self,
        *,
        sync_messages: list[DraftSync] | None = None,
        verify_commit_messages: list[VerifyCommit] | None = None,
        close_messages: list[DraftClose] | None = None,
    ) -> None:
        """
        Group verifier control messages by destination drafter and submit them.

        Used by decoupled verify lifecycle hooks verify input/update processing
        and by abort handling. This keeps verifier-to-drafter communication
        batch-based: each destination drafter rank receives at most one
        DraftControlBatch from this call.

        Args:
            sync_messages: Optional DraftSync messages created when verifier
                first introduces requests to the drafter.
            verify_commit_messages: Optional VerifyCommit messages created
                after verifier accepts tokens for live requests.
            close_messages: Optional DraftClose messages created when verifier
                finishes, retracts, or aborts requests.

        Returns:
            None.
        """
        if not self.is_verify_entry_rank():
            return

        batches: dict[int, DraftControlBatch] = {}

        def get_batch(dst_drafter_rank: int) -> DraftControlBatch:
            dst_drafter_rank = int(dst_drafter_rank)
            batch = batches.get(dst_drafter_rank)
            if batch is None:
                batch = DraftControlBatch(dst_drafter_rank=dst_drafter_rank)
                batches[dst_drafter_rank] = batch
            return batch

        for message in sync_messages or []:
            get_batch(message.dst_drafter_rank).sync_messages.append(message)
        for message in verify_commit_messages or []:
            get_batch(message.dst_drafter_rank).verify_commit_messages.append(message)
        for message in close_messages or []:
            get_batch(message.dst_drafter_rank).close_messages.append(message)

        for batch in batches.values():
            self._submit_verify_control_batch(batch)

    def _broadcast_verify_snapshots(
        self, local_snapshots: list[DraftTailSnapshot] | None
    ) -> list[DraftTailSnapshot]:
        """
        Broadcast per-forward draft tail snapshots from the verifier entry rank.

        Used during decoupled verify batch preparation. The entry rank reads
        currently available draft tail tokens from its DraftTailBuffer, then
        this helper makes the same immutable per-forward snapshot visible to all
        verifier ranks that participate in the forward pass.

        Args:
            local_snapshots: Draft tail snapshots collected on the entry rank.
                This value is ignored on non-entry ranks.

        Returns:
            The broadcast list of DraftTailSnapshot objects.
        """
        source_payload = (
            list(local_snapshots or []) if self.is_verify_entry_rank() else []
        )
        if getattr(self.server_args, "enable_dp_attention", False):
            synced_snapshots = source_payload
            if self.attn_tp_size != 1:
                synced_snapshots = broadcast_pyobj(
                    synced_snapshots,
                    self.attn_tp_group.rank,
                    self.attn_tp_cpu_group,
                    src=self.attn_tp_group.ranks[0],
                )
            if self.attn_cp_size != 1:
                synced_snapshots = broadcast_pyobj(
                    synced_snapshots,
                    self.attn_cp_group.rank,
                    self.attn_cp_cpu_group,
                    src=self.attn_cp_group.ranks[0],
                )
            return list(synced_snapshots or [])

        if self.tp_size != 1:
            source_payload = broadcast_pyobj(
                source_payload,
                self.tp_group.rank,
                self.tp_cpu_group,
                src=self.tp_group.ranks[0],
            )
        return list(source_payload or [])

    def _bind_verify_snapshots(
        self,
        target_reqs: list[Req],
        snapshots: list[DraftTailSnapshot],
    ) -> None:
        """
        Bind one broadcast draft tail snapshot set to the local verifier batch.

        Called after the entry-rank snapshot
        has been broadcast. All ranks, including the entry rank, use this same
        snapshot set for req.draft_buffer so concurrent proxy updates cannot
        affect the current verifier forward pass.

        Args:
            target_reqs: Live verifier requests in the local batch.
            snapshots: Broadcast per-request draft tail snapshots.

        Returns:
            None.
        """
        snapshot_by_rid: dict[str, DraftTailSnapshot] = {}
        for snapshot in snapshots:
            if snapshot.request_id in snapshot_by_rid:
                raise RuntimeError(
                    "Duplicate decoupled verify draft tail snapshot: "
                    f"request_id={snapshot.request_id}"
                )
            snapshot_by_rid[snapshot.request_id] = snapshot

        for req in target_reqs:
            snapshot = snapshot_by_rid.get(req.rid)
            if snapshot is None:
                setattr(req, "draft_buffer", None)
                setattr(req, "_decoupled_verify_snapshot_raw_tail_tokens", [])
                continue
            committed_len = int(snapshot.committed_len)
            if committed_len != len(req.output_ids):
                raise RuntimeError(
                    "Decoupled verify draft tail snapshot is out of sync with "
                    "the verifier request: "
                    f"request_id={req.rid} snapshot_committed_len={committed_len} "
                    f"request_output_len={len(req.output_ids)}"
                )
            setattr(req, "draft_buffer", list(snapshot.tail_tokens))
            setattr(
                req,
                "_decoupled_verify_snapshot_raw_tail_tokens",
                list(snapshot.raw_tail_tokens),
            )

    @trace_decoupled_spec(DecoupledSpecTraceEvent.VERIFIER_BUILD_SYNC_BATCH)
    def _sync_verify_requests(self, batch: ScheduleBatch) -> dict | None:
        """
        Send DraftSync messages before verifier prefill/extend processing.

        Called from process_batch_result setup for decoupled verify batches
        before an extend batch is run. Only the entry rank owns DraftTailBuffer
        and draft transport; for each live, unsynced request, it records the
        verifier's current prompt/output prefix and sends one DraftSync to the
        corresponding drafter so draft generation can start from the committed
        prefix.

        Args:
            batch: The ScheduleBatch about to be processed by the verifier.

        Returns:
            None.
        """
        if not self.is_verify_entry_rank():
            return None

        if not batch.forward_mode.is_extend() or batch.is_dllm():
            return None

        draft_tail_buffer = self.draft_tail_buffer
        assert draft_tail_buffer is not None

        sync_messages: list[DraftSync] = []
        for req in batch.reqs:
            if not req.is_retracted and not req.finished():
                setattr(
                    req,
                    "_decoupled_verify_pre_committed_len",
                    len(req.output_ids),
                )
            if req.is_chunked > 0 or req.is_retracted or req.finished():
                continue
            if draft_tail_buffer.has_request(req.rid):
                continue
            sync_messages.append(
                DraftSync(
                    request_id=req.rid,
                    src_verifier_rank=self.get_decoupled_spec_rank(),
                    dst_drafter_rank=self.assign_drafter_rank(req.rid),
                    prompt_token_ids=list(req.origin_input_ids),
                    committed_output_ids=list(req.output_ids),
                )
            )
            setattr(req, "draft_buffer", None)
        trace_payload = {
            "forward_mode": str(batch.forward_mode),
            "batch_size": len(batch.reqs),
            "rids": [message.request_id for message in sync_messages],
            "committed_lens_by_req": [
                len(message.committed_output_ids) for message in sync_messages
            ],
            "output_lens_by_req": [
                len(message.committed_output_ids) for message in sync_messages
            ],
            "dst_drafter_ranks": [
                int(message.dst_drafter_rank) for message in sync_messages
            ],
        }
        self._send_verify_control_batches(sync_messages=sync_messages)
        return trace_payload

    @trace_decoupled_spec(DecoupledSpecTraceEvent.VERIFIER_SNAPSHOT_TAIL_BATCH)
    def _snapshot_verify_inputs(self, batch: ScheduleBatch) -> dict | None:
        """
        Collect currently available draft tails, and bind them to a verifier request batch.

        Called immediately before a decoupled verify forward pass is prepared.
        The default path is non-blocking: the verifier entry rank snapshots the
        draft tail tokens already received by DraftProxyThread, broadcasts that
        stable per-forward snapshot to peer TP ranks, and all ranks bind
        req.draft_buffer from the broadcast snapshot.

        Args:
            batch: The ScheduleBatch that will run verifier extend/decode.

        Returns:
            None.
        """
        live_reqs = []
        for req in batch.reqs:
            if req.is_retracted or req.finished():
                continue
            live_reqs.append(req)
            setattr(req, "draft_buffer", None)
            setattr(req, "_decoupled_verify_snapshot_raw_tail_tokens", [])
            setattr(
                req,
                "_decoupled_verify_pre_committed_len",
                len(req.output_ids),
            )
        target_reqs = live_reqs
        if not target_reqs:
            return None

        local_snapshots: list[DraftTailSnapshot] = []
        if self.is_verify_entry_rank():
            draft_tail_buffer = self.draft_tail_buffer
            assert draft_tail_buffer is not None
            local_snapshots = draft_tail_buffer.get_draft_snapshots(
                target_reqs,
                allow_partial=envs.SGLANG_DECOUPLED_SPEC_ALLOW_PARTIAL.get(),
                include_raw_tail_tokens=True,
            )

        synced_snapshots = self._broadcast_verify_snapshots(local_snapshots)
        self._bind_verify_snapshots(target_reqs, synced_snapshots)
        snapshot_by_rid = {
            snapshot.request_id: snapshot for snapshot in synced_snapshots
        }
        return {
            "forward_mode": str(batch.forward_mode),
            "batch_size": len(target_reqs),
            "rids": [req.rid for req in target_reqs],
            "valid_tail_lens_by_req": [
                len(getattr(req, "draft_buffer", None) or []) for req in target_reqs
            ],
            "raw_tail_lens_by_req": [
                int(getattr(snapshot_by_rid.get(req.rid), "raw_tail_len", 0))
                for req in target_reqs
            ],
            "committed_lens_by_req": [len(req.output_ids) for req in target_reqs],
            "output_lens_by_req": [len(req.output_ids) for req in target_reqs],
        }

    @trace_decoupled_spec(DecoupledSpecTraceEvent.VERIFIER_BUILD_UPDATE_BATCH)
    def submit_verify_updates(
        self,
        batch: ScheduleBatch,
    ) -> dict | None:
        """
        Send verifier commit or close messages after batch result processing.

        Called after verifier extend/decode results have updated request output
        ids and finish state. Only the entry rank owns DraftTailBuffer and emits
        control messages. Live synced requests emit VerifyCommit with the latest
        committed prefix and bonus token. Finished or retracted synced requests
        emit DraftClose so the drafter can release its request state.

        Args:
            batch: The ScheduleBatch whose verifier results were just applied.

        Returns:
            None.
        """
        if not self.is_verify_entry_rank():
            return None

        if batch.forward_mode.is_extend() and batch.is_dllm():
            return None

        if not (batch.forward_mode.is_extend() or batch.forward_mode.is_decode()):
            return None

        draft_tail_buffer = self.draft_tail_buffer
        assert draft_tail_buffer is not None

        verify_commit_messages: list[VerifyCommit] = []
        close_messages: list[DraftClose] = []
        commit_pre_committed_lens: list[int] = []
        commit_draft_buffer_lens: list[int] = []
        commit_accepted_tail_lens: list[int] = []
        commit_bonus_token_ids: list[int] = []
        commit_snapshot_candidate_token_ids: list[int] = []
        commit_output_lens: list[int] = []
        close_output_lens: list[int] = []
        for req in batch.reqs:
            has_request = draft_tail_buffer.has_request(req.rid)

            if req.is_retracted or req.finished():
                if has_request:
                    dst_drafter_rank = self.get_drafter_rank(req.rid)
                    close_messages.append(
                        DraftClose(
                            request_id=req.rid,
                            src_verifier_rank=self.get_decoupled_spec_rank(),
                            dst_drafter_rank=dst_drafter_rank,
                            reason="abort" if req.is_retracted else "finished",
                        )
                    )
                    close_output_lens.append(len(req.output_ids))
                    self.release_drafter_rank(req.rid)
                setattr(req, "draft_buffer", None)
                setattr(req, "_decoupled_verify_snapshot_raw_tail_tokens", [])
                continue

            if not has_request:
                continue
            if not req.output_ids:
                continue

            bonus_token_pos = len(req.output_ids) - 1
            pre_verify_committed_len = getattr(
                req,
                "_decoupled_verify_pre_committed_len",
                None,
            )
            if pre_verify_committed_len is None:
                pre_verify_committed_len = draft_tail_buffer.get_committed_len(req.rid)
            if pre_verify_committed_len is None:
                continue

            bonus_token_id = int(req.output_ids[bonus_token_pos])
            draft_buffer = list(getattr(req, "draft_buffer", None) or [])
            accepted_tail_len = max(
                0, int(bonus_token_pos) - int(pre_verify_committed_len)
            )
            snapshot_raw_tail_tokens = list(
                getattr(req, "_decoupled_verify_snapshot_raw_tail_tokens", []) or []
            )
            snapshot_candidate_token_id = (
                int(snapshot_raw_tail_tokens[accepted_tail_len])
                if 0 <= accepted_tail_len < len(snapshot_raw_tail_tokens)
                else -1
            )

            verify_commit_messages.append(
                VerifyCommit(
                    request_id=req.rid,
                    src_verifier_rank=self.get_decoupled_spec_rank(),
                    dst_drafter_rank=self.get_drafter_rank(req.rid),
                    pre_verify_committed_len=pre_verify_committed_len,
                    bonus_token_pos=bonus_token_pos,
                    bonus_token_id=bonus_token_id,
                )
            )
            commit_pre_committed_lens.append(int(pre_verify_committed_len))
            commit_draft_buffer_lens.append(len(draft_buffer))
            commit_accepted_tail_lens.append(accepted_tail_len)
            commit_bonus_token_ids.append(bonus_token_id)
            commit_snapshot_candidate_token_ids.append(snapshot_candidate_token_id)
            commit_output_lens.append(len(req.output_ids))
            if hasattr(req, "_decoupled_verify_pre_committed_len"):
                delattr(req, "_decoupled_verify_pre_committed_len")
            if hasattr(req, "_decoupled_verify_snapshot_raw_tail_tokens"):
                delattr(req, "_decoupled_verify_snapshot_raw_tail_tokens")
        if verify_commit_messages:
            # Applying a VerifyCommit needs the raw bonus-candidate anchor in
            # DraftTailBuffer; without it `_apply_commit_locked` falls into the
            # bonus_match=False branch and unconditionally advances
            # `can_accept_prefix_len`, which causes any in-flight drafter
            # stream outputs still based on the old committed prefix to be
            # rejected as `stale_base` and silently dropped. Block here until
            # each committed request has at least one tail token so the
            # anchor-match decision is exact and consistent with the drafter,
            # otherwise consistency is lost.
            draft_tail_buffer.wait_for_draft_tokens(
                [message.request_id for message in verify_commit_messages],
                1,
            )
        trace_payload = {
            "forward_mode": str(batch.forward_mode),
            "batch_size": len(batch.reqs),
            "commit_rids": [
                message.request_id for message in verify_commit_messages
            ],
            "close_rids": [message.request_id for message in close_messages],
            "num_commit": len(verify_commit_messages),
            "num_close": len(close_messages),
            "pre_committed_lens_by_req": commit_pre_committed_lens,
            "draft_buffer_lens_by_req": commit_draft_buffer_lens,
            "accepted_tail_lens_by_req": commit_accepted_tail_lens,
            "bonus_token_ids_by_req": commit_bonus_token_ids,
            "snapshot_candidate_token_ids_by_req": commit_snapshot_candidate_token_ids,
            "committed_lens_by_req": [
                int(message.bonus_token_pos) + 1
                for message in verify_commit_messages
            ],
            "commit_output_lens_by_req": commit_output_lens,
            "commit_dst_drafter_ranks": [
                int(message.dst_drafter_rank) for message in verify_commit_messages
            ],
            "close_output_lens_by_req": close_output_lens,
            "close_dst_drafter_ranks": [
                int(message.dst_drafter_rank) for message in close_messages
            ],
        }
        self._send_verify_control_batches(
            verify_commit_messages=verify_commit_messages,
            close_messages=close_messages,
        )
        return trace_payload

    def abort_verify_request(self, request_id: str) -> None:
        """
        Close a drafter-side request when the verifier aborts it.

        Called from scheduler abort paths. Only the entry rank owns
        DraftTailBuffer; if the request has decoupled verify state there, this
        sends a DraftClose with ABORT to the recorded drafter rank. Requests
        that were never synced have no drafter state and do not need a close
        message.

        Args:
            request_id: Verifier request id to abort on the drafter side.

        Returns:
            None.
        """
        if not self.is_verify_entry_rank():
            return
        draft_tail_buffer = self.draft_tail_buffer
        assert draft_tail_buffer is not None
        if draft_tail_buffer.has_request(request_id):
            dst_drafter_rank = self.get_drafter_rank(request_id)
            self._send_verify_control_batches(
                close_messages=[
                    DraftClose(
                        request_id=request_id,
                        src_verifier_rank=self.get_decoupled_spec_rank(),
                        dst_drafter_rank=dst_drafter_rank,
                        reason="abort",
                    )
                ]
            )
            self.release_drafter_rank(request_id)
