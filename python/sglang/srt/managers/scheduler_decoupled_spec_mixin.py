from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.decoupled_spec_io import (
    DraftClose,
    DraftControlBatch,
    DraftControlInbox,
    DraftReqKey,
    DraftSync,
    DraftTailStreamOutput,
    DraftTailStreamOutputBatch,
    ReadyDraftControls,
    VerifierCommitSegment,
    VerifyCommit,
    build_draft_scheduler_rid,
    parse_draft_scheduler_rid,
)
from sglang.srt.speculative.tracer import (
    SpecTraceEvent,
    build_tracer,
    trace_speculative,
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
    is_sleeping: bool = False
    mamba_checkpoint_positions: set[int] = field(default_factory=set)
    mamba_checkpoint_slots: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class DraftKVTruncation:
    req_pool_idx: int
    kv_start: int
    kv_end: int


@dataclass(frozen=True)
class DraftBatchMetadataUpdate:
    req_batch_idx: int
    new_seq_len: int
    new_tail_token_id: int


@triton.jit
def _flush_draft_batch_metadata_updates_kernel(
    metadata_ptr,
    seq_lens_ptr,
    orig_seq_lens_ptr,
    output_ids_ptr,
    num_updates,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_updates

    req_batch_indices = tl.load(metadata_ptr + offsets, mask=mask, other=0)
    new_seq_lens = tl.load(
        metadata_ptr + num_updates + offsets, mask=mask, other=0
    )
    new_tail_token_ids = tl.load(
        metadata_ptr + 2 * num_updates + offsets, mask=mask, other=0
    )

    tl.store(seq_lens_ptr + req_batch_indices, new_seq_lens, mask=mask)
    tl.store(
        orig_seq_lens_ptr + req_batch_indices,
        new_seq_lens.to(tl.int32),
        mask=mask,
    )
    tl.store(output_ids_ptr + req_batch_indices, new_tail_token_ids, mask=mask)


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
            tracer=self.tracer,
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
            tracer=self.tracer,
        )
        self.token_sync_thread.start()

    def init_draft_state_tables(self: Scheduler) -> None:
        self.draft_req_table: Dict[DraftReqKey, DraftReqState] = {}
        self.draft_sleeping_reqs: Dict[DraftReqKey, Req] = {}
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
        return self.tracer.start_timer()

    @trace_speculative(SpecTraceEvent.SCHEDULER_FORWARD_BATCH)
    def record_forward_latency(
        self: Scheduler,
        batch: ScheduleBatch,
        start_ns: Optional[int],
        result: object | None = None,
    ) -> None:
        if start_ns is None:
            return

    @trace_speculative(SpecTraceEvent.DRAFTER_EMIT_DRAFT_TOKENS)
    def flush_draft_updates(
        self: Scheduler,
        batch: ScheduleBatch,
        req_indices: Optional[list[int]] = None,
    ) -> Optional[DraftTailStreamOutputBatch]:
        if not self.is_draft_worker_batch(batch):
            return None
        if req_indices is None:
            emit_candidate_indices = list(range(len(batch.reqs)))
        else:
            emit_candidate_indices = list(req_indices)

        stream_output_batch = DraftTailStreamOutputBatch()
        src_drafter_rank = self.get_decoupled_spec_rank()
        for req_batch_idx in emit_candidate_indices:
            if not (0 <= req_batch_idx < len(batch.reqs)):
                continue
            req = batch.reqs[req_batch_idx]
            if not req.output_ids:
                continue
            state = self._get_draft_state_by_req(req)
            token_pos = len(req.output_ids) - 1
            token_id = int(req.output_ids[-1])
            committed_len = int(state.verifier_committed_prefix_len)
            if token_pos < committed_len:
                continue
            stream_output_batch.outputs.append(
                DraftTailStreamOutput(
                    src_drafter_rank=src_drafter_rank,
                    dst_verifier_rank=int(state.key.src_verifier_rank),
                    request_id=state.key.request_id,
                    base_committed_len=committed_len,
                    new_token_pos=token_pos,
                    new_token_id=token_id,
                )
            )

        trace_payload = {
            "num_emit_candidates": len(emit_candidate_indices),
            "num_stream_outputs": len(stream_output_batch.outputs),
            "committed_lens_by_req": [
                int(self._get_draft_state_by_req(req).verifier_committed_prefix_len)
                for req in batch.reqs
            ],
            "output_lens_by_req": [len(req.output_ids) for req in batch.reqs],
        }
        setattr(stream_output_batch, "_decoupled_spec_payload", trace_payload)
        if stream_output_batch.outputs and self.is_draft_entry_rank():
            self._get_token_sync_thread().submit_draft_results(stream_output_batch)
        return stream_output_batch

    def prepare_verify_inputs(self: Scheduler, batch: ScheduleBatch) -> None:
        if not self.is_verify_worker_batch(batch):
            return
        if batch.forward_mode.is_extend():
            self._sync_verify_requests(batch)
        elif batch.forward_mode.is_decode():
            self._snapshot_verify_inputs(batch)

    def create_tracer(self):
        enabled = bool(
            getattr(self.server_args, "spec_trace_dir", None)
            and (
                self.spec_algorithm.is_decoupled_verify()
                or self.spec_algorithm.is_decoupled_draft()
                or self.spec_algorithm.is_none()
                or self.spec_algorithm.is_eagle()
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
        elif self.spec_algorithm.is_eagle():
            forward_trace_prefix = "mtp"
        else:
            forward_trace_prefix = "scheduler"
        file_names = {
            "scheduler.forward_batch": (
                f"{forward_trace_prefix}-forward-batch_{rank_suffix}.csv"
            ),
            "mtp.phase": f"mtp-phase_{rank_suffix}.csv",
            "verifier": f"verifier_{rank_suffix}.csv",
            "drafter": f"drafter_{rank_suffix}.csv",
            "draft_proxy": f"draft_proxy_verifier{dp_rank}.csv",
            "token_sync_thread": f"token_sync_thread_drafter{dp_rank}.csv",
        }
        return build_tracer(
            enabled=enabled,
            output_dir=getattr(self.server_args, "spec_trace_dir", None),
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

    def _broadcast_ready_draft_controls(
        self: Scheduler,
        ready_controls: ReadyDraftControls | None,
    ) -> ReadyDraftControls:
        """
        Broadcast ready draft controls among all ranks:
        DraftSync: build a new draft request based on its prompt token_ids
        VerifierCommitSegment: apply the verifier-committed segment and
        truncate suffix if needed
        """
        def broadcast_ready_controls(rank, group, src) -> ReadyDraftControls | None:
            payload = (
                [ready_controls]
                if ready_controls is not None and rank == src
                else []
            )
            payload = broadcast_pyobj(payload, rank, group, src=src)
            if not payload:
                return None
            if len(payload) != 1:
                raise RuntimeError(
                    "Expected a single ReadyDraftControls payload, "
                    f"got {len(payload)}"
                )
            return payload[0]

        if getattr(self.server_args, "enable_dp_attention", False):
            if self.attn_tp_size != 1:
                ready_controls = broadcast_ready_controls(
                    self.attn_tp_group.rank,
                    self.attn_tp_cpu_group,
                    src=self.attn_tp_group.ranks[0],
                )
            if self.attn_cp_size != 1:
                ready_controls = broadcast_ready_controls(
                    self.attn_cp_group.rank,
                    self.attn_cp_cpu_group,
                    src=self.attn_cp_group.ranks[0],
                )
            return (
                ready_controls
                if ready_controls is not None
                else ReadyDraftControls()
            )

        if self.tp_size != 1:
            ready_controls = broadcast_ready_controls(
                self.tp_group.rank,
                self.tp_cpu_group,
                src=self.tp_group.ranks[0],
            )
        return (
            ready_controls
            if ready_controls is not None
            else ReadyDraftControls()
        )

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

    def _release_draft_mamba_ckpt_slots(
        self: Scheduler,
        state: DraftReqState,
    ) -> None:
        """
        Release the draft mamba checkpoint slots back to the pool when a draft request is released.
        """
        slots = state.mamba_checkpoint_slots
        state.mamba_checkpoint_positions.clear()
        state.mamba_checkpoint_slots = None
        if slots is None:
            return

        self.req_to_token_pool.mamba_pool.free(slots)

    def _ensure_draft_mamba_ckpt_slots(
        self: Scheduler,
        state: DraftReqState,
    ) -> torch.Tensor:
        """
        Ensure the draft request can allocate mamba checkpoint slots
        for its potential future rollback.
        """
        if state.mamba_checkpoint_slots is not None:
            return state.mamba_checkpoint_slots

        window = self._draft_ahead_window()
        if window <= 0:
            raise RuntimeError(
                "Decoupled drafter mamba rollback requires a positive draft "
                f"ahead window. request_id={state.key.request_id}, window={window}"
            )

        mamba_pool = self.req_to_token_pool.mamba_pool
        if mamba_pool is None:
            raise RuntimeError(
                "Decoupled drafter mamba checkpoint requested without a mamba pool: "
                f"request_id={state.key.request_id}"
            )

        slots = mamba_pool.alloc(window)
        if slots is None:
            raise RuntimeError(
                "Not enough space for decoupled drafter mamba rollback "
                "checkpoints. Try to increase --mamba-full-memory-ratio or "
                f"--max-mamba-cache-size. request_id={state.key.request_id}, "
                f"window={window}, mamba_pool_size={mamba_pool.size}, "
                f"mamba_available_size={mamba_pool.available_size()}"
            )

        state.mamba_checkpoint_slots = slots
        return slots

    def _draft_mamba_ckpt_slot(
        self: Scheduler,
        state: DraftReqState,
        token_pos: int,
        *,
        for_write: bool,
    ) -> torch.Tensor:
        if for_write:
            slots = self._ensure_draft_mamba_ckpt_slots(state)
        else:
            if token_pos not in state.mamba_checkpoint_positions:
                req = state.req
                raise RuntimeError(
                    "Missing decoupled drafter mamba checkpoint. "
                    f"request_id={state.key.request_id}, "
                    f"token_pos={token_pos}, "
                    f"output_len={len(req.output_ids) if req else None}, "
                    "available_checkpoint_positions="
                    f"{sorted(state.mamba_checkpoint_positions)}"
                )
            slots = state.mamba_checkpoint_slots
            if slots is None:
                raise RuntimeError(
                    "Decoupled drafter mamba checkpoint metadata exists without "
                    "allocated checkpoint slots. "
                    f"request_id={state.key.request_id}, "
                    f"token_pos={token_pos}"
                )

        slot_count = int(slots.numel())
        slot_offset = token_pos % slot_count
        if for_write:
            for existing_pos in state.mamba_checkpoint_positions:
                if (
                    existing_pos != token_pos
                    and existing_pos % slot_count == slot_offset
                ):
                    raise RuntimeError(
                        "Decoupled drafter mamba checkpoint ring would overwrite a "
                        "live checkpoint. This indicates the drafter exceeded its "
                        "rollback window. "
                        f"request_id={state.key.request_id}, token_pos={token_pos}, "
                        f"existing_pos={existing_pos}, slot_count={slot_count}"
                    )
        return slots[slot_offset : slot_offset + 1]

    def _prune_draft_mamba_ckpts(self: Scheduler, state: DraftReqState) -> None:
        """
        Prune the draft mamba checkpoints that are no longer needed
        after the verifier has committed a longer prefix.
        """
        req = state.req
        if req is None or not state.mamba_checkpoint_positions:
            return
        committed_len = int(state.verifier_committed_prefix_len)
        output_len = len(req.output_ids)
        tail_pos = output_len - 1
        # Only keep checkpoints for tokens that are still in the drafter's
        # uncommitted suffix.
        # Also keep the current tail checkpoint: the next decode consumes that
        # tail token even when the verifier has already committed it.
        positions_to_invalidate = [
            pos
            for pos in state.mamba_checkpoint_positions
            if pos >= output_len or (pos < committed_len and pos != tail_pos)
        ]
        for pos in positions_to_invalidate:
            state.mamba_checkpoint_positions.discard(pos)

    def commit_draft_mamba_ckpts(
        self: Scheduler,
        batch: ScheduleBatch,
        req_indices: Optional[list[int]] = None,
    ) -> None:
        """
        Commit the drafter mamba checkpoint metadata after the forward pass has
        written the routed dst slot.
        """
        if not self.is_draft_worker_batch(batch):
            return
        try:
            if req_indices is None:
                checkpoint_candidate_indices = range(len(batch.reqs))
            else:
                checkpoint_candidate_indices = req_indices

            for req_batch_idx in checkpoint_candidate_indices:
                if not (0 <= req_batch_idx < len(batch.reqs)):
                    continue
                req = batch.reqs[req_batch_idx]
                if req.mamba_pool_idx is None or not req.output_ids:
                    continue

                state = self._get_draft_state_by_req(req)
                self._prune_draft_mamba_ckpts(state)
                token_pos = len(req.output_ids) - 1
                if token_pos in state.mamba_checkpoint_positions:
                    continue

                if batch.mamba_cache_dst_indices is None:
                    raise RuntimeError(
                        "Decoupled drafter emitted a token without mamba "
                        "routing metadata. "
                        f"request_id={state.key.request_id}, token_pos={token_pos}"
                    )

                state.mamba_checkpoint_positions.add(token_pos)
        finally:
            if (
                batch.forward_mode.is_decode()
                or batch.forward_mode.is_extend(include_draft_extend_v2=True)
            ):
                batch.mamba_cache_src_indices = None
                batch.mamba_cache_dst_indices = None

    def prepare_draft_mamba_routing(self: Scheduler, batch: ScheduleBatch) -> None:
        if not self.is_draft_worker_batch(batch):
            return
        if not isinstance(self.req_to_token_pool, HybridReqToTokenPool):
            return
        is_decode = batch.forward_mode.is_decode()
        is_prefill = batch.forward_mode.is_extend(include_draft_extend_v2=True)
        if not is_decode and not is_prefill:
            return

        src_indices: list[torch.Tensor] = []
        dst_indices: list[torch.Tensor] = []
        for req in batch.reqs:
            if req.mamba_pool_idx is None:
                raise RuntimeError(
                    "Decoupled drafter mamba routing requires every req "
                    "to own a mamba slot. "
                    f"rid={req.rid}"
                )
            if is_decode and not req.output_ids:
                raise RuntimeError(
                    "Decoupled drafter mamba routing requires a tail token. "
                    f"rid={req.rid}"
                )

            state = self._get_draft_state_by_req(req)
            if is_decode:
                self._prune_draft_mamba_ckpts(state)
                token_pos = len(req.output_ids) - 1
                src_indices.append(
                    self._draft_mamba_ckpt_slot(
                        state, token_pos, for_write=False
                    )
                )
                dst_indices.append(
                    self._draft_mamba_ckpt_slot(
                        state, token_pos + 1, for_write=True
                    )
                )
            else:
                token_pos = len(req.output_ids)
                dst_slot = self._draft_mamba_ckpt_slot(
                    state, token_pos, for_write=True
                )
                src_indices.append(dst_slot)
                dst_indices.append(dst_slot)

        if not src_indices:
            return

        device = batch.seq_lens.device
        batch.mamba_cache_src_indices = torch.cat(src_indices).to(
            device=device, dtype=torch.int64, non_blocking=True
        )
        batch.mamba_cache_dst_indices = torch.cat(dst_indices).to(
            device=device, dtype=torch.int64, non_blocking=True
        )

    def _flush_draft_kv_truncations(
        self: Scheduler,
        kv_truncations: list[DraftKVTruncation],
    ) -> None:
        if not kv_truncations:
            return

        indices_to_free: list[torch.Tensor] = []
        req_to_token = self.req_to_token_pool.req_to_token
        for truncation in kv_truncations:
            if truncation.kv_start >= truncation.kv_end:
                continue
            kv_indices = req_to_token[
                truncation.req_pool_idx, truncation.kv_start : truncation.kv_end
            ]
            if len(kv_indices) > 0:
                indices_to_free.append(kv_indices)

        if indices_to_free:
            self.token_to_kv_pool_allocator.free(torch.cat(indices_to_free))
        kv_truncations.clear()

    def _flush_draft_batch_metadata_updates(
        self: Scheduler,
        batch_metadata_updates: list[DraftBatchMetadataUpdate],
    ) -> None:
        if not batch_metadata_updates:
            return

        batch = self.running_batch
        if batch is None or batch.is_empty():
            raise RuntimeError(
                "Decoupled draft batch metadata update requires a non-empty "
                "running_batch. Verifier commit segment metadata updates should "
                "only be queued for requests in running_batch."
            )
        if (
            batch.seq_lens_cpu is None
            or batch.seq_lens is None
            or batch.orig_seq_lens is None
            or batch.output_ids is None
            or batch.seq_lens_sum is None
        ):
            raise RuntimeError(
                "Decoupled draft batch metadata update requires complete "
                "running_batch metadata: seq_lens_cpu, seq_lens, "
                "orig_seq_lens, output_ids, and seq_lens_sum must be set."
            )

        req_batch_idx_set = {update.req_batch_idx for update in batch_metadata_updates}
        if len(req_batch_idx_set) != len(batch_metadata_updates):
            raise RuntimeError(
                "Decoupled draft batch metadata update received duplicate batch "
                "indices in one flush. This indicates multiple verifier commit "
                "segment rewrites for the same in-flight request."
            )

        device = batch.seq_lens.device
        num_updates = len(batch_metadata_updates)
        req_batch_indices = []
        new_seq_lens = []
        new_tail_token_ids = []
        seq_lens_delta = 0
        seq_lens_cpu_np = batch.seq_lens_cpu.numpy()
        for update in batch_metadata_updates:
            req_batch_idx = int(update.req_batch_idx)
            new_seq_len = int(update.new_seq_len)
            old_seq_len = int(seq_lens_cpu_np[req_batch_idx])

            req_batch_indices.append(req_batch_idx)
            new_seq_lens.append(new_seq_len)
            new_tail_token_ids.append(int(update.new_tail_token_id))
            seq_lens_delta += new_seq_len - old_seq_len

        metadata_cpu = torch.tensor(
            [req_batch_indices, new_seq_lens, new_tail_token_ids],
            dtype=torch.int64,
            pin_memory=device.type == "cuda",
        )

        for req_batch_idx, new_seq_len in zip(req_batch_indices, new_seq_lens):
            seq_lens_cpu_np[req_batch_idx] = new_seq_len

        metadata_device = metadata_cpu.to(device=device, non_blocking=True)
        block_size = 256
        _flush_draft_batch_metadata_updates_kernel[
            (triton.cdiv(num_updates, block_size),)
        ](
            metadata_device,
            batch.seq_lens,
            batch.orig_seq_lens,
            batch.output_ids,
            num_updates,
            BLOCK_SIZE=block_size,
        )

        batch.seq_lens_sum += seq_lens_delta

        batch_metadata_updates.clear()

    def apply_verifier_commit_segment(
        self: Scheduler,
        req: Req,
        segment: VerifierCommitSegment,
        *,
        req_batch_idx: Optional[int] = None,
        kv_truncations: list[DraftKVTruncation],
        batch_metadata_updates: list[DraftBatchMetadataUpdate],
    ) -> None:
        """
        Apply the verifier-committed output segment to the draft request.

        This path handles already-materialized matching segments and a single
        divergent verifier token. Callers must split verifier commit segments
        before applying them so mismatched suffixes are committed incrementally.
        """
        state = self._get_draft_state_by_req(req)
        if state.key != segment.draft_key:
            raise RuntimeError(
                "VerifierCommitSegment arrived for a mismatched draft request: "
                f"req_rid={req.rid} req_draft_key={state.key} "
                f"segment_draft_key={segment.draft_key}"
            )
        pre_verify_committed_len = int(segment.pre_verify_committed_len)
        committed_token_ids = [
            int(token_id) for token_id in segment.committed_token_ids
        ]
        if not committed_token_ids:
            raise ValueError(
                "VerifierCommitSegment committed_token_ids must be non-empty: "
                f"request_id={segment.draft_key.request_id} "
                f"pre_verify_committed_len={pre_verify_committed_len}"
            )
        committed_segment_len = len(committed_token_ids)
        current_committed_len = int(state.verifier_committed_prefix_len)
        new_committed_len = pre_verify_committed_len + committed_segment_len
        output_len = len(req.output_ids)
        prompt_len = len(req.origin_input_ids)
        materialized_kv_len = prompt_len + max(output_len - 1, 0)

        if new_committed_len <= current_committed_len:
            raise RuntimeError(
                "VerifierCommitSegment must advance the drafter committed prefix: "
                f"request_id={segment.draft_key.request_id} "
                f"src_verifier_rank={segment.draft_key.src_verifier_rank} "
                f"pre_verify_committed_len={pre_verify_committed_len} "
                f"committed_segment_len={committed_segment_len} "
                f"current_committed_len={current_committed_len} "
                f"new_committed_len={new_committed_len}"
            )

        if pre_verify_committed_len > current_committed_len:
            raise RuntimeError(
                "VerifierCommitSegment depends on a prefix the drafter has not "
                "committed: "
                f"request_id={segment.draft_key.request_id} "
                f"src_verifier_rank={segment.draft_key.src_verifier_rank} "
                f"pre_verify_committed_len={pre_verify_committed_len} "
                f"current_committed_len={current_committed_len}"
            )

        if req.kv_committed_freed:
            raise RuntimeError(
                "Decoupled draft verify commit found freed KV cache: "
                f"request_id={state.key.request_id} "
                f"new_committed_len={new_committed_len} "
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
                f"new_committed_len={new_committed_len} "
                f"output_len={output_len} "
                f"prompt_len={prompt_len} "
                f"materialized_kv_len={materialized_kv_len} "
                f"kv_committed_len={req.kv_committed_len} "
                f"kv_allocated_len={req.kv_allocated_len}"
            )

        matched_segment_len = 0
        max_possible_match_len = min(
            committed_segment_len,
            max(0, output_len - pre_verify_committed_len),
        )
        while (
            matched_segment_len < max_possible_match_len
            and int(req.output_ids[pre_verify_committed_len + matched_segment_len])
            == committed_token_ids[matched_segment_len]
        ):
            matched_segment_len += 1

        if matched_segment_len == committed_segment_len:
            # all committed_tokens match the drafter's output, simply advance the committed prefix
            state.verifier_committed_prefix_len = new_committed_len
            self._prune_draft_mamba_ckpts(state)
            return

        remaining_committed_token_ids = committed_token_ids[matched_segment_len:]
        if len(remaining_committed_token_ids) > 1:
            raise RuntimeError(
                "VerifierCommitSegment committed_token_ids contain a multi-token "
                "mismatched verifier segment. Split the segment before applying: "
                f"request_id={segment.draft_key.request_id} "
                f"src_verifier_rank={segment.draft_key.src_verifier_rank} "
                f"pre_verify_committed_len={pre_verify_committed_len} "
                f"matched_segment_len={matched_segment_len} "
                f"committed_token_ids={committed_token_ids} "
                f"draft_segment={req.output_ids[pre_verify_committed_len:new_committed_len]}"
            )

        committed_token_pos = pre_verify_committed_len + matched_segment_len
        committed_token_id = int(remaining_committed_token_ids[0])
        if committed_token_pos < current_committed_len:
            raise RuntimeError(
                "VerifierCommitSegment conflicts with the already committed "
                "drafter prefix: "
                f"request_id={segment.draft_key.request_id} "
                f"src_verifier_rank={segment.draft_key.src_verifier_rank} "
                f"committed_token_pos={committed_token_pos} "
                f"current_committed_len={current_committed_len} "
                f"committed_token_ids={committed_token_ids}"
            )
        if committed_token_pos >= output_len:
            raise RuntimeError(
                "VerifierCommitSegment cannot skip a non-materialized drafter "
                "output gap. Keep the segment pending until the drafter "
                "materializes the prefix: "
                f"request_id={segment.draft_key.request_id} "
                f"src_verifier_rank={segment.draft_key.src_verifier_rank} "
                f"committed_token_pos={committed_token_pos} "
                f"output_len={output_len} "
                f"committed_token_ids={committed_token_ids}"
            )

        # The verifier-selected token replaces the drafter suffix starting at
        # `committed_token_pos`.
        #
        # Positions here are in req.output_ids, not in the full prompt+output
        # sequence. The kept output range is [0, truncate_from), and the removed
        # output range is [truncate_from, len(req.output_ids)). In other words,
        # `truncate_from` itself is removed. After the removal, committed_token_id
        # is appended at exactly that position.
        truncate_from = committed_token_pos

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
                f"committed_token_pos={committed_token_pos} "
                f"output_len={output_len} "
                f"prompt_len={prompt_len} "
                f"kv_truncate_from={kv_truncate_from} "
                f"kv_committed_len={req.kv_committed_len} "
                f"kv_allocated_len={req.kv_allocated_len}"
            )

        if isinstance(self.req_to_token_pool, HybridReqToTokenPool):
            # check the committed_token_pos's mamba ckpt exiests
            self._draft_mamba_ckpt_slot(state, committed_token_pos, for_write=False)

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
                kv_truncations.append(
                    DraftKVTruncation(
                        req_pool_idx=int(req.req_pool_idx),
                        kv_start=kv_truncate_from,
                        kv_end=trimmed_end,
                    )
                )
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

        req.output_ids.append(committed_token_id)
        if req.grammar is not None:
            try:
                req.grammar.accept_token(committed_token_id)
            except Exception:
                logger.debug(
                    "Draft grammar accept failed during verify commit for req %s",
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
                f"committed_token_pos={committed_token_pos}"
            )
        if int(req.output_ids[-1]) != committed_token_id:
            raise RuntimeError(
                "Decoupled draft verify commit failed to install committed token: "
                f"request_id={state.key.request_id} "
                f"committed_token_pos={committed_token_pos} "
                f"committed_token_id={committed_token_id} "
                f"tail_token_id={int(req.output_ids[-1])}"
            )

        state.verifier_committed_prefix_len = new_committed_len
        self._prune_draft_mamba_ckpts(state)

        if req_batch_idx is not None:
            batch = self.running_batch
            if batch is None or batch.is_empty():
                raise RuntimeError(
                    "Decoupled draft verify commit received a running batch "
                    "index, but running_batch is empty: "
                    f"request_id={state.key.request_id} "
                    f"req_batch_idx={req_batch_idx}"
                )
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
            # state. This block is only needed when the verifier committed token
            # changed req.output_ids above: either an existing suffix was truncated
            # and replaced, or a committed token was appended at the current tail.
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

            if new_seq_len > min(req.kv_committed_len, req.kv_allocated_len):
                raise RuntimeError(
                    "Decoupled draft batch seq_len points beyond materialized KV "
                    "after verify commit: "
                    f"request_id={state.key.request_id} "
                    f"new_seq_len={new_seq_len} "
                    f"kv_committed_len={req.kv_committed_len} "
                    f"kv_allocated_len={req.kv_allocated_len}"
                )
            batch_metadata_updates.append(
                DraftBatchMetadataUpdate(
                    req_batch_idx=req_batch_idx,
                    new_seq_len=new_seq_len,
                    new_tail_token_id=new_tail_token_id,
                )
            )

    def release_draft_request(self: Scheduler, req: Req) -> None:
        """
        release a draft request only when it has completed at the verifier side:
        1. evict the req from waiting_queue or running_batch
        2. remove the req from the draft request table
        3. release its kvcache
        """
        state = self._get_draft_state_by_req(req)

        # remove the req from waiting_queue, running_batch, and sleeping table
        self.waiting_queue = [
            queued_req for queued_req in self.waiting_queue if queued_req is not req
        ]
        for batch in (self.running_batch, self.last_batch, self.cur_batch):
            if batch is None or batch.is_empty():
                continue
            keep_indices = [
                i for i, batch_req in enumerate(batch.reqs) if batch_req is not req
            ]
            if len(keep_indices) != len(batch.reqs):
                batch.filter_batch(keep_indices=keep_indices)
                batch.batch_is_full = False
        self.draft_sleeping_reqs.pop(state.key, None)
        self._release_draft_mamba_ckpt_slots(state)
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

    def _draft_commit_segment_consumable_len(
        self: Scheduler,
        segment: VerifierCommitSegment,
        req: Req,
        state: DraftReqState,
    ) -> int:
        pre_verify_committed_len = int(segment.pre_verify_committed_len)
        current_committed_len = int(state.verifier_committed_prefix_len)
        if pre_verify_committed_len != current_committed_len:
            raise RuntimeError(
                "Verifier commit segment does not match the drafter committed prefix: "
                f"request_id={segment.draft_key.request_id} "
                f"src_verifier_rank={segment.draft_key.src_verifier_rank} "
                f"pre_verify_committed_len={pre_verify_committed_len} "
                f"current_committed_len={current_committed_len}"
            )
        if not segment.committed_token_ids:
            return 0

        output_len = len(req.output_ids)
        if pre_verify_committed_len > output_len:
            raise RuntimeError(
                "Verifier commit segment is ahead of the drafter committed prefix: "
                f"request_id={segment.draft_key.request_id} "
                f"src_verifier_rank={segment.draft_key.src_verifier_rank} "
                f"pre_verify_committed_len={pre_verify_committed_len} "
                f"output_len={output_len}"
            )

        if pre_verify_committed_len == output_len:
            return 0

        matched_len = 0
        max_possible_match_len = min(
            len(segment.committed_token_ids),
            output_len - pre_verify_committed_len,
        )
        while (
            matched_len < max_possible_match_len
            and int(req.output_ids[pre_verify_committed_len + matched_len])
            == int(segment.committed_token_ids[matched_len])
        ):
            matched_len += 1

        if matched_len == len(segment.committed_token_ids):
            return matched_len
        if matched_len < max_possible_match_len:
            # the first token that doesn't match the drafter's suffix is still considered consumable
            return matched_len + 1
        return matched_len

    def _collect_ready_draft_controls(
        self: Scheduler,
        control_inbox: DraftControlInbox,
    ) -> ReadyDraftControls:
        def consumable_commit_len(segment: VerifierCommitSegment) -> int:
            state = self.draft_req_table.get(segment.draft_key)
            if state is None:
                return 0
            req = state.req
            if (
                req is None
                or req.req_pool_idx is None
                or req.kv_committed_freed
            ):
                return 0

            return self._draft_commit_segment_consumable_len(segment, req, state)

        return control_inbox.extract_ready_controls_locked(consumable_commit_len)

    def _apply_ready_verifier_commit_segments(
        self: Scheduler,
        ready_commit_segments: list[VerifierCommitSegment],
    ) -> tuple[list[VerifierCommitSegment], int]:
        kv_truncations: list[DraftKVTruncation] = []
        batch_metadata_updates: list[DraftBatchMetadataUpdate] = []
        applied_segments: list[VerifierCommitSegment] = []
        commit_echo_batch = DraftTailStreamOutputBatch()
        if not ready_commit_segments:
            return applied_segments, 0

        running_req_to_idx = {}
        if self.running_batch is not None and not self.running_batch.is_empty():
            running_req_to_idx = {
                id(req): req_batch_idx
                for req_batch_idx, req in enumerate(self.running_batch.reqs)
        }

        for segment in ready_commit_segments:
            state = self.draft_req_table.get(segment.draft_key)
            if state is None:
                raise RuntimeError(
                    "Ready verifier commit segment has no draft state: "
                    f"draft_key={segment.draft_key}"
                )
            req = state.req
            if req is None:
                raise RuntimeError(
                    "Ready verifier commit segment has no draft request: "
                    f"draft_key={segment.draft_key}"
                )

            req_batch_idx = running_req_to_idx.get(id(req))
            self.apply_verifier_commit_segment(
                req,
                segment,
                req_batch_idx=req_batch_idx,
                kv_truncations=kv_truncations,
                batch_metadata_updates=batch_metadata_updates,
            )
            applied_segments.append(segment)
            if self.is_draft_entry_rank():
                if not segment.committed_token_ids:
                    raise ValueError(
                        "VerifierCommitSegment committed_token_ids must be "
                        "non-empty before echoing applied segment: "
                        f"request_id={segment.draft_key.request_id} "
                        f"pre_verify_committed_len={segment.pre_verify_committed_len}"
                    )
                committed_token_pos = (
                    int(segment.pre_verify_committed_len)
                    + len(segment.committed_token_ids)
                    - 1
                )
                # Echo the last applied committed token so verifier-side
                # pending expected tokens can advance when no comparable
                # draft-tail anchor was available in its buffer.
                commit_echo_batch.outputs.append(
                    DraftTailStreamOutput(
                        src_drafter_rank=self.get_decoupled_spec_rank(),
                        dst_verifier_rank=int(segment.draft_key.src_verifier_rank),
                        request_id=segment.draft_key.request_id,
                        base_committed_len=committed_token_pos,
                        new_token_pos=committed_token_pos,
                        new_token_id=int(segment.committed_token_ids[-1]),
                    )
                )

        self._flush_draft_kv_truncations(kv_truncations)
        self._flush_draft_batch_metadata_updates(batch_metadata_updates)
        if commit_echo_batch.outputs:
            self._get_token_sync_thread().submit_draft_results(commit_echo_batch)
        return applied_segments, len(commit_echo_batch.outputs)

    def _handle_draft_sync_message(
        self: Scheduler,
        message: DraftSync,
    ) -> Optional[Req]:
        req = self._create_draft_request(message)
        running_batch = self.running_batch
        if (
            req not in self.waiting_queue
            and req not in running_batch.reqs
            and req not in self.draft_sleeping_reqs.values()
        ):
            self._add_request_to_queue(req)
        return req

    def _handle_draft_close_key(self: Scheduler, draft_key: DraftReqKey) -> None:
        entry = self.draft_req_table.get(draft_key)
        if entry is None:
            return

        req = entry.req
        if req is not None:
            self.release_draft_request(req)
            return

        raise RuntimeError(
            "DraftClose found drafter state without a live request: "
            f"draft_key={draft_key} is_sleeping={entry.is_sleeping}"
        )

    @trace_speculative(SpecTraceEvent.DRAFTER_SYNC_CONTROL_MESSAGES)
    def sync_draft_requests(self: Scheduler) -> dict | None:
        """
        (called by decoupled drafter)
        Collect ready verifier-to-drafter controls in arrival order.
        DraftSync creates requests, ready VerifierCommitSegment objects
        advance/truncate existing requests, and DraftClose releases drafter-side
        state.
        """
        if not self.spec_algorithm.is_decoupled_draft():
            return None

        ready_controls: ReadyDraftControls | None = None
        if self.is_draft_entry_rank():
            ready_controls = (
                self._get_token_sync_thread().collect_ready_draft_controls(
                    self._collect_ready_draft_controls
                )
            )

        ready_controls = self._broadcast_ready_draft_controls(ready_controls)

        num_sync = 0
        num_commit = len(ready_controls.ready_commit_segments)
        num_close = 0
        num_created_reqs = 0
        num_applied_commit = 0

        closed_keys = ready_controls.close_keys
        for draft_key in closed_keys:
            num_close += 1
            self._handle_draft_close_key(draft_key)

        for message in ready_controls.sync_messages:
            draft_key = message.draft_key
            if draft_key in closed_keys:
                continue
            num_sync += 1
            req = self._handle_draft_sync_message(message)
            if req is not None:
                num_created_reqs += 1

        applied_segments, num_commit_echo = self._apply_ready_verifier_commit_segments(
            ready_controls.ready_commit_segments
        )
        num_applied_commit += len(applied_segments)

        if ready_controls.is_empty() and num_applied_commit == 0:
            return None
        return {
            "num_sync": num_sync,
            "num_commit": num_commit,
            "num_close": num_close,
            "num_created_reqs": num_created_reqs,
            "num_applied_commit": num_applied_commit,
            "num_commit_echo": num_commit_echo,
            "num_sleeping_reqs": len(self.draft_sleeping_reqs),
        }

    def _draft_ahead_window(self: Scheduler) -> int:
        draft_tokens = self.server_args.speculative_num_draft_tokens
        return max(0, int(draft_tokens or 0) * 2)

    def _draft_req_ahead(self: Scheduler, state: DraftReqState) -> int:
        req = state.req
        if req is None:
            return 0
        return len(req.output_ids) - int(state.verifier_committed_prefix_len)

    def has_draft_sleeping_requests(self: Scheduler) -> bool:
        # check whether decoupled drafter has sleeping requests
        return bool(self.draft_sleeping_reqs)

    def _build_draft_decode_batch(self: Scheduler, reqs: list[Req]) -> ScheduleBatch:
        device = self.device
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            model_config=self.model_config,
            enable_overlap=self.enable_overlap,
            spec_algorithm=self.spec_algorithm,
        )

        batch.req_pool_indices = torch.tensor(
            [req.req_pool_idx for req in reqs], dtype=torch.int64, device=device
        )
        seq_lens = [
            len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
            for req in reqs
        ]
        batch.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device=device)
        batch.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        batch.orig_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        batch.seq_lens_sum = sum(seq_lens)
        batch.output_ids = torch.tensor(
            [
                int(req.output_ids[-1])
                if req.output_ids
                else int(req.origin_input_ids[-1])
                for req in reqs
            ],
            dtype=torch.int64,
            device=device,
        )
        batch.multimodal_inputs = [req.multimodal_inputs for req in reqs]
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch, self.model_config.vocab_size
        )
        return batch

    @trace_speculative(SpecTraceEvent.DRAFTER_SLEEP_REQUESTS)
    def sleep_overrun_draft_requests(
        self: Scheduler,
        batch: Optional[ScheduleBatch],
    ) -> Optional[ScheduleBatch]:
        if batch is None or batch.is_empty():
            return batch
        setattr(batch, "_decoupled_spec_payload", None)
        window = self._draft_ahead_window()
        if window <= 0:
            return batch

        keep_indices: list[int] = []
        slept_rids: list[str] = []
        slept_any = False
        for req_batch_idx, req in enumerate(batch.reqs):
            state = self._get_draft_state_by_req(req)
            ahead = self._draft_req_ahead(state)
            if ahead >= window:
                state.is_sleeping = True
                self.draft_sleeping_reqs[state.key] = req
                slept_rids.append(state.key.request_id)
                slept_any = True
            else:
                keep_indices.append(req_batch_idx)

        if slept_any:
            batch.filter_batch(keep_indices=keep_indices)
            batch.batch_is_full = False
            setattr(
                batch,
                "_decoupled_spec_payload",
                {
                    "num_slept": len(slept_rids),
                    "slept_rids": slept_rids,
                },
            )
        return batch

    @trace_speculative(SpecTraceEvent.DRAFTER_WAKE_REQUESTS)
    def wake_draft_sleeping_requests(self: Scheduler) -> Optional[dict]:
        window = self._draft_ahead_window()
        if window <= 0 or not self.draft_sleeping_reqs:
            return None
        max_batch_size = getattr(self.server_args, "pp_max_micro_batch_size", None)
        if not max_batch_size:
            max_batch_size = getattr(self, "max_running_requests", None)
        max_batch_size = int(max_batch_size or 0)
        if max_batch_size > 0:
            available_num_reqs = max(0, max_batch_size - len(self.running_batch.reqs))
            if available_num_reqs == 0:
                return None
        else:
            available_num_reqs = len(self.draft_sleeping_reqs)

        wake_reqs: list[Req] = []
        woken_rids: list[str] = []
        for draft_key, req in list(self.draft_sleeping_reqs.items()):
            state = self.draft_req_table.get(draft_key)
            if state is None or state.req is not req:
                self.draft_sleeping_reqs.pop(draft_key, None)
                continue
            ahead = self._draft_req_ahead(state)
            if ahead < window:
                state.is_sleeping = False
                self.draft_sleeping_reqs.pop(draft_key, None)
                wake_reqs.append(req)
                woken_rids.append(draft_key.request_id)
                if len(wake_reqs) >= available_num_reqs:
                    break

        if not wake_reqs:
            return None

        # build decode batch for these woken reqs, and merge them into the current batch
        wake_batch = self._build_draft_decode_batch(wake_reqs)
        if self.running_batch.is_empty():
            self.running_batch = wake_batch
        else:
            self.running_batch.merge_batch(wake_batch)
        self.running_batch.batch_is_full = False
        return {
            "num_woken": len(wake_reqs),
            "woken_rids": woken_rids,
        }

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

        spec_steps = int(self.server_args.speculative_num_steps or 0)
        if spec_steps <= 0:
            spec_steps = max(
                0, int(self.server_args.speculative_num_draft_tokens or 1) - 1
            )
        for (
            req,
            valid_draft_tokens,
            valid_accepted_tokens,
        ) in valid_draft_metric_updates:
            valid_draft_tokens = min(valid_draft_tokens, spec_steps)
            valid_accepted_tokens = min(valid_accepted_tokens, valid_draft_tokens)
            req.spec_valid_draft_tokens += valid_draft_tokens
            req.spec_valid_accepted_tokens += valid_accepted_tokens
            req.spec_valid_draft_tokens_by_position = (
                req.spec_valid_draft_tokens_by_position + [0] * spec_steps
            )[:spec_steps]
            req.spec_valid_accepted_tokens_by_position = (
                req.spec_valid_accepted_tokens_by_position + [0] * spec_steps
            )[:spec_steps]
            for pos in range(valid_draft_tokens):
                req.spec_valid_draft_tokens_by_position[pos] += 1
            for pos in range(valid_accepted_tokens):
                req.spec_valid_accepted_tokens_by_position[pos] += 1

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
    ) -> int:
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
            The number of snapshots skipped because their confirmed prefix
            lags behind the verifier request.
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
            if committed_len < len(req.output_ids):
                # the drafter has not caught up with the verifier req's committed output prefix
                setattr(req, "draft_buffer", None)
                setattr(req, "_decoupled_verify_snapshot_raw_tail_tokens", [])
                continue
            if committed_len > len(req.output_ids):
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
        return sum(
            1
            for req in target_reqs
            if snapshot_by_rid.get(req.rid) is not None
            and int(snapshot_by_rid[req.rid].committed_len) < len(req.output_ids)
        )

    @trace_speculative(SpecTraceEvent.VERIFIER_BUILD_SYNC_BATCH)
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

    @trace_speculative(SpecTraceEvent.VERIFIER_SNAPSHOT_TAIL_BATCH)
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
        num_stale_snapshots = self._bind_verify_snapshots(
            target_reqs, synced_snapshots
        )
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
            "committed_lens_by_req": [
                int(getattr(snapshot_by_rid.get(req.rid), "committed_len", 0))
                for req in target_reqs
            ],
            "output_lens_by_req": [len(req.output_ids) for req in target_reqs],
            "num_stale_snapshots": num_stale_snapshots,
        }

    @trace_speculative(SpecTraceEvent.VERIFIER_BUILD_UPDATE_BATCH)
    def submit_verify_updates(
        self,
        batch: ScheduleBatch,
    ) -> dict | None:
        """
        Send verifier commit or close messages after batch result processing.

        Called after verifier extend/decode results have updated request output
        ids and finish state. Only the entry rank owns DraftTailBuffer and emits
        control messages. Live synced requests emit VerifyCommit with the latest
        committed output segment. Finished or retracted synced requests
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
        commit_segment_lens: list[int] = []
        commit_last_token_ids: list[int] = []
        commit_committed_lens: list[int] = []
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

            pre_verify_committed_len = getattr(
                req,
                "_decoupled_verify_pre_committed_len",
                None,
            )
            if pre_verify_committed_len is None:
                pre_verify_committed_len = draft_tail_buffer.get_committed_len(req.rid)
            if pre_verify_committed_len is None:
                continue
            pre_verify_committed_len = int(pre_verify_committed_len)
            if pre_verify_committed_len > len(req.output_ids):
                raise RuntimeError(
                    "Verifier VerifyCommit pre-commit prefix is beyond the "
                    "current output ids: "
                    f"request_id={req.rid} "
                    f"pre_verify_committed_len={pre_verify_committed_len} "
                    f"output_len={len(req.output_ids)}"
                )
            if pre_verify_committed_len == len(req.output_ids):
                # no tokens are generated during this forward(e.g. chunked prefill)
                if hasattr(req, "_decoupled_verify_pre_committed_len"):
                    delattr(req, "_decoupled_verify_pre_committed_len")
                if hasattr(req, "_decoupled_verify_snapshot_raw_tail_tokens"):
                    delattr(req, "_decoupled_verify_snapshot_raw_tail_tokens")
                continue

            draft_buffer = list(getattr(req, "draft_buffer", None) or [])
            committed_token_ids = [
                int(token_id)
                for token_id in req.output_ids[pre_verify_committed_len:]
            ]

            verify_commit_messages.append(
                VerifyCommit(
                    request_id=req.rid,
                    src_verifier_rank=self.get_decoupled_spec_rank(),
                    dst_drafter_rank=self.get_drafter_rank(req.rid),
                    pre_verify_committed_len=pre_verify_committed_len,
                    committed_token_ids=committed_token_ids,
                )
            )
            commit_pre_committed_lens.append(pre_verify_committed_len)
            commit_draft_buffer_lens.append(len(draft_buffer))
            commit_segment_lens.append(len(committed_token_ids))
            commit_last_token_ids.append(int(committed_token_ids[-1]))
            commit_committed_lens.append(
                pre_verify_committed_len + len(committed_token_ids)
            )
            commit_output_lens.append(len(req.output_ids))
            if hasattr(req, "_decoupled_verify_pre_committed_len"):
                delattr(req, "_decoupled_verify_pre_committed_len")
            if hasattr(req, "_decoupled_verify_snapshot_raw_tail_tokens"):
                delattr(req, "_decoupled_verify_snapshot_raw_tail_tokens")
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
            "committed_segment_lens_by_req": commit_segment_lens,
            "last_committed_token_ids_by_req": commit_last_token_ids,
            "committed_lens_by_req": commit_committed_lens,
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
