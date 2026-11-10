from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.runtime_context import get_parallel, get_spec
from sglang.srt.speculative.dspark_components.dspark_pp import (
    pack_proposal,
    unpack_proposal,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.utils import GenerationBatchResult


@dataclass(frozen=True, slots=True)
class PPDSparkDraftWork:
    batch: ScheduleBatch
    draft_input: object
    pp_outputs: Optional[PPProxyTensors] = None


def snapshot_pp_dspark_batch(batch: ScheduleBatch) -> ScheduleBatch:
    snapshot = copy.copy(batch)
    snapshot.reqs = batch.reqs[:]
    snapshot.req_pool_indices = batch.req_pool_indices.clone()
    return snapshot


class PPDSparkDraftCoordinator:
    """Owns request-scoped PP DSpark draft scheduling state."""

    def __init__(self, scheduler: Scheduler):
        self._scheduler = scheduler
        self._pending: deque[PPDSparkDraftWork] = deque()
        self._send_work = []

    def on_batch_launched(
        self, batch: ScheduleBatch, result: GenerationBatchResult
    ) -> None:
        scheduler = self._scheduler
        if result.pp_dspark_draft_idle:
            if scheduler.pp_group.is_last_rank:
                self._run_idle_draft(batch)
            return

        if (
            result.pp_dspark_projected_context is None
            or result.accept_lens is None
            or result.pp_dspark_next_proposal is not None
        ):
            return

        if self._is_bubble_policy():
            self.enqueue(batch, result.next_draft_input)
        else:
            result.pp_dspark_next_proposal = self._run_draft(
                batch, result.next_draft_input
            )

    def on_relayed_idle(self, batch: ScheduleBatch, pp_outputs: PPProxyTensors) -> None:
        scheduler = self._scheduler
        if (
            "dspark_draft_idle" not in pp_outputs.tensors
            or not scheduler.pp_group.is_first_rank
        ):
            return

        scheduler.forward_stream.wait_stream(scheduler.copy_stream)
        with scheduler.forward_stream_ctx:
            self._run_idle_draft(batch)
        scheduler.copy_stream.wait_stream(scheduler.forward_stream)

    def on_relayed_proposals(
        self,
        batch: ScheduleBatch,
        draft_input: object,
        pp_outputs: PPProxyTensors,
    ) -> None:
        scheduler = self._scheduler
        deferred = (
            "dspark_next_1_identities" not in pp_outputs.tensors
            and self._is_bubble_policy()
        )
        if deferred:
            if not scheduler.pp_group.is_first_rank:
                raise RuntimeError(
                    "Only the first PP rank may receive a deferred DSpark "
                    "verify result."
                )
            self.enqueue(batch, draft_input, pp_outputs)
            return

        if scheduler.pp_group.is_first_rank:
            scheduler.forward_stream.wait_stream(scheduler.copy_stream)
            with scheduler.forward_stream_ctx:
                pp_outputs.tensors.update(
                    pack_proposal(0, self._run_draft(batch, draft_input))
                )
            scheduler.copy_stream.wait_stream(scheduler.forward_stream)

        for owner in range(get_parallel().pp_size):
            scheduler.model_worker.install_pp_draft(
                batch,
                owner,
                unpack_proposal(owner, pp_outputs.tensors),
            )

    def enqueue(
        self,
        batch: ScheduleBatch,
        draft_input: object,
        pp_outputs: Optional[PPProxyTensors] = None,
    ) -> None:
        self._pending.append(
            PPDSparkDraftWork(
                batch=snapshot_pp_dspark_batch(batch),
                draft_input=draft_input,
                pp_outputs=pp_outputs,
            )
        )

    def drain(self) -> None:
        if not self._pending:
            return

        scheduler = self._scheduler
        work = self._pending.popleft()
        scheduler._pp_commit_comm_work(self._send_work)
        with scheduler.forward_stream_ctx:
            proposal = self._run_draft(work.batch, work.draft_input)
            ready_event = scheduler.device_module.Event()
            ready_event.record(scheduler.device_module.current_stream())

        if scheduler.pp_group.is_last_rank:
            self._send_work = scheduler._pp_send_dict_to_next_stage(
                pack_proposal(get_parallel().pp_rank, proposal),
                async_send=True,
                msg_type="dspark_draft",
                ready_event=ready_event,
            )
            return

        if not scheduler.pp_group.is_first_rank or work.pp_outputs is None:
            raise RuntimeError("Invalid PP DSpark deferred draft work owner.")
        tensor_dict, recv_event = scheduler._pp_recv_typed_dict(
            expected_kind="dspark_draft",
            all_gather_group=scheduler.attn_tp_group,
        )
        if recv_event is not None:
            scheduler.forward_stream.wait_event(recv_event)
        work.pp_outputs.tensors.update(pack_proposal(0, proposal))
        work.pp_outputs.tensors.update(tensor_dict)
        for owner in range(get_parallel().pp_size):
            scheduler.model_worker.install_pp_draft(
                work.batch,
                owner,
                unpack_proposal(owner, work.pp_outputs.tensors),
            )

    def _run_draft(self, batch: ScheduleBatch, draft_input: object):
        scheduler = self._scheduler
        if not self._is_bubble_policy():
            return scheduler.model_worker.prepare_pp_draft(batch, draft_input)

        with torch.profiler.record_function("pp_dspark_draft_bubble"):
            return scheduler.model_worker.prepare_pp_draft(batch, draft_input)

    def _run_idle_draft(self, batch: ScheduleBatch) -> None:
        with torch.profiler.record_function("pp_dspark_draft_bubble"):
            self._scheduler.model_worker.prepare_pp_idle_draft(batch)

    @staticmethod
    def _is_bubble_policy() -> bool:
        return get_spec().speculative_draft_scheduling_policy == "bubble"
