"""Answer every reader about the live requests, and abort them."""

from __future__ import annotations

import logging
import time
from http import HTTPStatus
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set

from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.request_lists import RequestList
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.runtime_context import get_memory

if TYPE_CHECKING:
    from sglang.srt.constrained.grammar_manager import GrammarManager
    from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
        DecodeKVCacheOffloadManager,
    )
    from sglang.srt.disaggregation.encoder.receiver import MMReceiverBase
    from sglang.srt.managers.request_lists import CoreLists
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler_components.bundles import KvResources
    from sglang.srt.managers.scheduler_components.deferred_responses import (
        DeferredResponses,
    )
    from sglang.srt.managers.scheduler_components.mode_owners import ModeOwners
    from sglang.srt.managers.scheduler_components.pending_steps import PendingSteps
    from sglang.srt.managers.scheduler_components.request_retirer import (
        RequestRetirer,
    )

logger = logging.getLogger(__name__)


class RequestRegistry:
    def __init__(
        self,
        *,
        core_lists: CoreLists,
        grammar_manager: GrammarManager,
        request_retirer: RequestRetirer,
        kv: KvResources,
        mm_receiver: Optional[MMReceiverBase],
        pending: PendingSteps,
        decode_offload_manager: Optional[DecodeKVCacheOffloadManager],
        mode_owners: ModeOwners,
        deferred_responses: DeferredResponses,
    ):
        self.core_lists = core_lists
        self.grammar_manager = grammar_manager
        self.request_retirer = request_retirer
        self.tree_cache = kv.tree_cache
        self.mm_receiver = mm_receiver
        self.pending = pending
        self.decode_offload_manager = decode_offload_manager
        # Read at call time: a PD role switch replaces the owners.
        self.mode_owners = mode_owners
        self.deferred_responses = deferred_responses

    def reqs(self, where: Callable[[RequestList], bool]) -> List[Req]:
        """The members of the lists `where` selects, in list order (the same
        on every rank)."""
        return [
            req
            for entry in self.request_lists().values()
            if where(entry)
            for req in entry.reqs
        ]

    def queued_reqs(self) -> List[Req]:
        """The requests waiting for admission, in list order."""
        return self.reqs(lambda entry: entry.queued)

    def collect_inflight_reqs(self) -> List[Req]:
        """The requests past prefill admission that a step may still run, in
        list order (the same on every rank)."""
        return self.reqs(lambda entry: entry.inflight)

    def running_reqs(self) -> List[Req]:
        """The running requests, in list order."""
        return self.reqs(lambda entry: entry.counts_running)

    def lora_ids_in_use(self) -> Set[Optional[str]]:
        """The adapters LoRA admission keeps: those of the lists that pin
        them, and of the steps in flight (a request finished at the last
        commit may still run in one)."""
        lora_ids = {req.lora_id for req in self.reqs(lambda entry: entry.pins_lora)}
        lora_ids.update(self.pending.lora_ids())
        return lora_ids

    def request_lists(self) -> Dict[str, RequestList]:
        """The lists that own the live requests, by name: the core lists, then
        those of the PD mode and of the add-ons."""
        lists = [
            RequestList(
                name="waiting",
                reqs=self.core_lists.waiting_queue,
                queued=True,
                abort=self._abort_waiting_reqs,
                health_check=True,
                waiting=True,
                counts_queued=True,
            ),
            # Abort method 2: call `set_finish_with_abort`. The request still
            # runs one prefill forward, with its input cut to one token.
            RequestList(
                name="grammar",
                reqs=self.grammar_manager.grammar_queue,
                abort=self.grammar_manager.abort_requests,
            ),
            RequestList(
                name="decoding",
                reqs=self.core_lists.decoding,
                inflight=True,
                health_check=True,
                retract_on_pause=True,
                holds_kv=True,
                pins_lora=True,
                counts_running=True,
            ),
            # A PD prefill node keeps a live mid-chunk request across a pause:
            # its KV sender still reads the KV a retraction would free.
            RequestList(
                name="chunked",
                reqs=[self.core_lists.chunked_req]
                if self.core_lists.chunked_req is not None
                else [],
                inflight=True,
                abort=self._defer_chunked_abort,
                health_check=True,
                retract_on_pause=self.mode_owners.disagg_prefill is None,
                holds_kv=True,
                partly_prefilled=True,
            ),
        ]
        if self.mode_owners.disagg_prefill is not None:
            lists += self.mode_owners.disagg_prefill.lists()
        if self.mode_owners.disagg_decode is not None:
            lists += self.mode_owners.disagg_decode.lists()
        if self.mode_owners.dllm is not None:
            lists += self.mode_owners.dllm.lists()
        if self.mode_owners.hisparse_coordinator is not None:
            lists += self.mode_owners.hisparse_coordinator.lists()
        lists += self.deferred_responses.lists()
        return {entry.name: entry for entry in lists}

    def abort_request(self, recv_req: AbortReq):
        # Abort requests still waiting for encoder embeddings (EPD language-only)
        if self.mm_receiver is not None:
            self.mm_receiver.abort_waiting_requests(recv_req)

        for entry in self.request_lists().values():
            if entry.abort is not None:
                entry.abort(recv_req)
            if entry.inflight:
                self._abort_inflight_reqs(entry.reqs, recv_req)

    def _abort_waiting_reqs(self, recv_req: AbortReq) -> None:
        # Delete requests in the waiting queue
        to_del = []
        for i, req in enumerate(self.core_lists.waiting_queue):
            if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                to_del.append(i)

        # Sort in reverse order to avoid index issues when deleting
        for i in reversed(to_del):
            # Abort method 1: directly pop from the queue
            # This only works for requests that have not started anything.
            # We still need to send something back to TokenizerManager to clean up the state.
            req = self.core_lists.waiting_queue.pop(i)
            if self.mode_owners.disagg_prefill is not None:
                # A PD prefill request may hold KV here (an optimistic retry).
                self.mode_owners.disagg_prefill.release_aborted_waiting_req(req)
                self.request_retirer.retire_aborted(req)
            else:
                self.request_retirer.retire_aborted_waiting(req)
            # Without the initiator's reason the tokenizer falls back to a
            # generic abort message.
            self.request_retirer.send_abort_output(
                req, finished_reason=recv_req.finished_reason
            )
            logger.debug(f"Abort queued request. {req.rid=}")

    def _defer_chunked_abort(self, recv_req: AbortReq) -> None:
        """Mark the chunked request for `process_pending_chunked_abort`."""
        if (chunked_req := self.core_lists.chunked_req) is not None:
            if recv_req.abort_all or chunked_req.rid.startswith(recv_req.rid):
                self.core_lists._pending_chunked_abort_req = chunked_req

    def _abort_inflight_reqs(self, reqs: List[Req], recv_req: AbortReq) -> None:
        # Abort the requests past admission
        for req in reqs:
            if not req.finished() and (
                recv_req.abort_all or req.rid.startswith(recv_req.rid)
            ):
                # Abort method 3: set `to_finish`
                # The request will still run one decode forward pass.
                # Then we reuse all existing code to clean up the KV cache allocation.
                logger.debug(f"Abort running request. {req.rid=}")
                if recv_req.abort_message:
                    # Timeout aborts carry an SLA message + 503 for the client.
                    req.to_finish = FINISH_ABORT(
                        recv_req.abort_message, HTTPStatus.SERVICE_UNAVAILABLE
                    )
                else:
                    req.to_finish = FINISH_ABORT()

    def is_fully_idle(
        self, for_health_check=False, ignore_waiting=False, ignore_releasing=False
    ) -> bool:
        # The request lists, and the steps not committed yet. The health check
        # piggybacks on running requests in process_output, so it counts only
        # the lists whose requests run soon (`RequestList.health_check`).
        idle = len(self.pending) == 0 and not any(
            entry.keeps_busy(
                for_health_check=for_health_check,
                ignore_waiting=ignore_waiting,
                ignore_releasing=ignore_releasing,
            )
            for entry in self.request_lists().values()
        )

        if not for_health_check:
            # PD decode KV offload: copies in flight still read device KV.
            if self.decode_offload_manager is not None:
                idle &= len(self.decode_offload_manager.ongoing_offload) == 0

            # HiCache: in-flight async ops (GPU↔Host↔L3) must drain before
            # destructive operations like attach/detach/flush_cache.
            if get_memory().enable_hierarchical_cache:
                tc = self.tree_cache
                idle &= len(tc.ongoing_write_through) == 0
                idle &= len(tc.ongoing_load_back) == 0
                if tc.enable_storage:
                    idle &= len(tc.ongoing_prefetch) == 0
                    idle &= len(tc.ongoing_backup) == 0
                    if get_memory().hicache_host_memory_mode == "buffer_only":
                        # Queued writes, staged prefetches, and in-flight
                        # storage writes still hold host staging
                        # (buffer-mode unified tree only).
                        idle &= tc.buffer_pipeline.is_idle()
            elif get_memory().enable_lmcache:
                idle &= not self.tree_cache.has_pending_cache_operations()

        return idle

    def poll_timeout_aborts(self) -> List[AbortReq]:
        """Emit aborts only; every rank must drop the same requests in the
        same iteration, or the extend-vs-decode decision splits and the
        collectives hang.
        """
        aborts: List[AbortReq] = []

        if (timeout_s := envs.SGLANG_REQ_WAITING_TIMEOUT.get()) > 0:
            deadline = time.perf_counter() - timeout_s
            for req in self.queued_reqs():
                entry_time = req.time_stats.wait_queue_entry_time
                if 0 < entry_time < deadline:
                    aborts.append(
                        AbortReq(
                            rid=req.rid,
                            abort_message="Request waiting timeout reached.",
                            finished_reason={
                                "type": "abort",
                                "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                                "message": "Request waiting timeout reached.",
                            },
                        )
                    )

        if (timeout_s := envs.SGLANG_REQ_RUNNING_TIMEOUT.get()) > 0:
            deadline = time.perf_counter() - timeout_s
            for req in self.collect_inflight_reqs():
                # Polling must not re-emit an abort awaiting result processing.
                if req.finished() or req.to_finish is not None:
                    continue
                if 0 < req.time_stats.forward_entry_time < deadline:
                    aborts.append(
                        AbortReq(
                            rid=req.rid,
                            abort_message="Request running timeout reached.",
                            finished_reason={
                                "type": "abort",
                                "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                                "message": "Request running timeout reached.",
                            },
                        )
                    )

        return aborts
