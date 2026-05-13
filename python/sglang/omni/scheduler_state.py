# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from http import HTTPStatus
from queue import Empty, Queue
from threading import Condition, Event, RLock, Thread, get_ident
from typing import TYPE_CHECKING, Any, Callable

from sglang.omni.protocol import OmniContextBundle

if TYPE_CHECKING:
    from sglang.omni.coordinator import OmniCoordinator
    from sglang.srt.managers.io_struct import (
        OmniGenerateReqInput,
        OmniGenerateReqOutput,
        OmniGenerateStreamOutput,
    )
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.omni_session.runtime import (
        OmniSessionRecord as SRTOmniSessionRecord,
    )
    from sglang.srt.omni_session.srt_executor import OmniSRTSchedulerExecutor

    OmniTaskOutput = tuple[
        OmniGenerateReqOutput | OmniGenerateStreamOutput,
        OmniGenerateReqInput,
    ]
else:
    OmniTaskOutput = tuple[Any, Any]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OmniSessionRecord:
    context: OmniContextBundle
    turns: int
    created_at: float
    updated_at: float


@dataclass(slots=True)
class OmniSRTExecutionRequest:
    """A native SRT request submitted by an omni task and completed by scheduler."""

    executor: "OmniSRTSchedulerExecutor"
    record: "SRTOmniSessionRecord"
    req: "Req"
    state: object | None
    done: Event = field(default_factory=Event)
    error: BaseException | None = None

    def finish(self) -> None:
        self.done.set()

    def fail(self, error: BaseException) -> None:
        self.error = error
        self.done.set()

    def wait(self) -> None:
        self.done.wait()
        if self.error is not None:
            raise self.error


@dataclass(slots=True)
class OmniSRTBatchObservation:
    """Scheduler-owned observation needed to finalize native omni SRT reqs."""

    requests: list[OmniSRTExecutionRequest]
    executor_sessions: list[
        tuple["OmniSRTSchedulerExecutor", list[tuple[str, str, int | None]]]
    ]


@dataclass(slots=True)
class OmniSchedulerThreadCall:
    """A small resource-lifecycle callback that must run on scheduler thread."""

    callback: Callable[[], Any]
    description: str
    done: Event = field(default_factory=Event)
    result: Any = None
    error: BaseException | None = None

    def run(self) -> None:
        try:
            self.result = self.callback()
        except BaseException as exc:
            self.error = exc
        finally:
            self.done.set()

    def wait(self) -> Any:
        self.done.wait()
        if self.error is not None:
            raise self.error
        return self.result


@dataclass(slots=True)
class OmniSchedulerExclusiveLease:
    """Lease that keeps scheduler-owned SRT state idle for colocated generation."""

    scheduler_state: "OmniSchedulerState"
    reason: str
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.scheduler_state.leave_scheduler_exclusive_region(self.reason)


@dataclass(slots=True)
class OmniSchedulerState:
    """Scheduler-local state for async omni tasks and native SRT req handoff."""

    orchestrators: dict[str, "OmniCoordinator"] = field(default_factory=dict)
    sessions: dict[str, OmniSessionRecord] = field(default_factory=dict)
    pending_srt_requests: Queue[OmniSRTExecutionRequest] = field(default_factory=Queue)
    running_srt_requests: dict[str, OmniSRTExecutionRequest] = field(
        default_factory=dict
    )
    finished_srt_requests: dict[str, OmniSRTExecutionRequest] = field(
        default_factory=dict
    )
    pending_scheduler_thread_calls: Queue[OmniSchedulerThreadCall] = field(
        default_factory=Queue
    )
    completed_task_outputs: Queue[OmniTaskOutput] = field(default_factory=Queue)
    orchestrator_lock: RLock = field(default_factory=RLock)
    scheduler_thread_id: int | None = None
    scheduler_condition: Condition = field(default_factory=Condition)
    exclusive_waiters: int = 0
    exclusive_active: int = 0
    active_task_count: int = 0

    def bind_scheduler_thread(self) -> None:
        self.scheduler_thread_id = get_ident()

    def is_scheduler_thread(self) -> bool:
        return self.scheduler_thread_id == get_ident()

    def submit_srt_request(
        self,
        *,
        executor: "OmniSRTSchedulerExecutor",
        record: "SRTOmniSessionRecord",
        req: "Req",
        state: object | None,
    ) -> OmniSRTExecutionRequest:
        pending = OmniSRTExecutionRequest(
            executor=executor,
            record=record,
            req=req,
            state=state,
        )
        self.pending_srt_requests.put(pending)
        return pending

    def admit_srt_requests(self, scheduler: "Scheduler") -> None:
        """Move pending omni AR reqs into the normal SRT waiting queue."""
        if self.exclusive_waiters > 0 or self.exclusive_active > 0:
            return

        while True:
            try:
                pending = self.pending_srt_requests.get_nowait()
            except Empty:
                return
            try:
                # 1. materialize a native SRT req submitted by an omni task
                scheduler.init_req_max_new_tokens(pending.req)
                # 2. submit it to the normal SRT waiting queue for batching
                scheduler._add_request_to_queue(pending.req)
                self.running_srt_requests[pending.req.rid] = pending
            except BaseException as exc:
                pending.fail(exc)

    def observe_srt_batch_before_process(
        self, batch: "ScheduleBatch"
    ) -> OmniSRTBatchObservation | None:
        pending_requests = [
            pending
            for req in batch.reqs
            if (pending := self.running_srt_requests.get(req.rid)) is not None
        ]
        if not pending_requests:
            return None

        executor_sessions: list[
            tuple["OmniSRTSchedulerExecutor", list[tuple[str, str, int | None]]]
        ] = []
        seen_executors: set[int] = set()
        for pending in pending_requests:
            executor = pending.executor
            executor_id = id(executor)
            if executor_id in seen_executors:
                continue
            seen_executors.add(executor_id)
            sessions = executor.capture_batch_token_bindings_before_process(batch)
            executor_sessions.append((executor, sessions))
        return OmniSRTBatchObservation(
            requests=pending_requests,
            executor_sessions=executor_sessions,
        )

    def finalize_srt_batch_after_process(
        self, observation: OmniSRTBatchObservation | None
    ) -> None:
        if observation is None:
            return

        for executor, sessions in observation.executor_sessions:
            # 3. finalize session/KV bindings after scheduler processed the req
            executor.capture_session_token_bindings_after_process(sessions)

        for pending in observation.requests:
            if not pending.req.finished():
                continue
            self.running_srt_requests.pop(pending.req.rid, None)
            self.finished_srt_requests[pending.req.rid] = pending

        self.notify_scheduler_progress()

    def retire_finished_srt_requests(self, scheduler: "Scheduler") -> None:
        if not self.finished_srt_requests:
            return

        retired: list[str] = []
        for rid, pending in self.finished_srt_requests.items():
            if self._scheduler_still_owns_req(scheduler, pending.req):
                continue
            retired.append(rid)
            # 4. wake the omni task only after SRT no longer owns the req
            pending.finish()

        for rid in retired:
            self.finished_srt_requests.pop(rid, None)
        if retired:
            self.notify_scheduler_progress()

    def start_omni_generate_task(
        self,
        *,
        scheduler: "Scheduler",
        recv_req: "OmniGenerateReqInput",
    ) -> None:
        with self.scheduler_condition:
            self.active_task_count += 1
        thread = Thread(
            target=self._run_omni_generate_task,
            args=(scheduler, recv_req),
            daemon=True,
            name=f"omni-generate-{recv_req.rid}",
        )
        thread.start()

    def pop_completed_task_outputs(self) -> list[OmniTaskOutput]:
        outputs: list[OmniTaskOutput] = []
        while True:
            try:
                outputs.append(self.completed_task_outputs.get_nowait())
            except Empty:
                return outputs

    def run_on_scheduler_thread(
        self,
        *,
        callback: Callable[[], Any],
        description: str,
    ) -> Any:
        if self.is_scheduler_thread():
            return callback()

        call = OmniSchedulerThreadCall(
            callback=callback,
            description=description,
        )
        self.pending_scheduler_thread_calls.put(call)
        self.notify_scheduler_progress()
        return call.wait()

    def drain_scheduler_thread_calls(self) -> bool:
        drained = False
        while True:
            try:
                call = self.pending_scheduler_thread_calls.get_nowait()
            except Empty:
                return drained
            drained = True
            call.run()

    def has_pending_scheduler_side_work(self) -> bool:
        return (
            self.active_task_count > 0
            or not self.pending_srt_requests.empty()
            or bool(self.running_srt_requests)
            or bool(self.finished_srt_requests)
            or not self.pending_scheduler_thread_calls.empty()
            or not self.completed_task_outputs.empty()
            or self.exclusive_waiters > 0
            or self.exclusive_active > 0
        )

    def enter_scheduler_exclusive_region(
        self,
        *,
        scheduler: "Scheduler",
        reason: str,
    ) -> OmniSchedulerExclusiveLease:
        with self.scheduler_condition:
            self.exclusive_waiters += 1
            try:
                while not scheduler.is_fully_idle():
                    self.scheduler_condition.wait(timeout=0.01)
                self.exclusive_active += 1
            finally:
                self.exclusive_waiters -= 1
                self.scheduler_condition.notify_all()
        return OmniSchedulerExclusiveLease(self, reason)

    def leave_scheduler_exclusive_region(self, reason: str) -> None:
        with self.scheduler_condition:
            self.exclusive_active = max(0, self.exclusive_active - 1)
            self.scheduler_condition.notify_all()

    def notify_scheduler_progress(self) -> None:
        with self.scheduler_condition:
            self.scheduler_condition.notify_all()

    def _scheduler_still_owns_req(self, scheduler: "Scheduler", req: "Req") -> bool:
        if req in scheduler.waiting_queue:
            return True
        for batch in (
            scheduler.cur_batch,
            scheduler.last_batch,
            scheduler.running_batch,
        ):
            if batch is not None and req in batch.reqs:
                return True
        result_queue = getattr(scheduler, "result_queue", None)
        if result_queue is not None:
            for batch, _ in result_queue:
                if req in batch.reqs:
                    return True
        running_mbs = getattr(scheduler, "running_mbs", None)
        if running_mbs is not None:
            for batch in running_mbs:
                if batch is not None and req in batch.reqs:
                    return True
        return False

    def _run_omni_generate_task(
        self, scheduler: "Scheduler", recv_req: "OmniGenerateReqInput"
    ) -> None:
        from sglang.omni.srt_transport import (
            handle_omni_generate_with_omni_coordinator,
        )
        from sglang.omni.streaming import OmniStreamSink
        from sglang.srt.managers.io_struct import (
            OmniGenerateReqOutput,
            OmniGenerateStreamOutput,
        )

        def emit_stream_event(event: dict[str, Any]) -> None:
            self.completed_task_outputs.put(
                (
                    OmniGenerateStreamOutput(rid=recv_req.rid, event=event),
                    recv_req,
                )
            )
            self.notify_scheduler_progress()

        stream_sink = OmniStreamSink(emit_stream_event) if recv_req.stream else None
        try:
            payload = handle_omni_generate_with_omni_coordinator(
                scheduler=scheduler,
                payload=recv_req.payload,
                stream_sink=stream_sink,
            )
            if stream_sink is not None:
                stream_sink.done(payload)
            output = OmniGenerateReqOutput(
                rid=recv_req.rid,
                success=True,
                payload=payload,
            )
        except ValueError as exc:
            if stream_sink is not None:
                stream_sink.error(
                    message=str(exc),
                    status_code=int(HTTPStatus.BAD_REQUEST),
                )
            output = OmniGenerateReqOutput(
                rid=recv_req.rid,
                success=False,
                message=str(exc),
                status_code=int(HTTPStatus.BAD_REQUEST),
            )
        except RuntimeError as exc:
            if stream_sink is not None:
                stream_sink.error(
                    message=str(exc),
                    status_code=int(HTTPStatus.NOT_IMPLEMENTED),
                )
            output = OmniGenerateReqOutput(
                rid=recv_req.rid,
                success=False,
                message=str(exc),
                status_code=int(HTTPStatus.NOT_IMPLEMENTED),
            )
        except Exception as exc:
            logger.exception("Failed to handle omni generate request")
            if stream_sink is not None:
                stream_sink.error(
                    message=str(exc),
                    status_code=int(HTTPStatus.INTERNAL_SERVER_ERROR),
                )
            output = OmniGenerateReqOutput(
                rid=recv_req.rid,
                success=False,
                message=str(exc),
                status_code=int(HTTPStatus.INTERNAL_SERVER_ERROR),
            )

        self.completed_task_outputs.put((output, recv_req))
        self.notify_scheduler_progress()
        with self.scheduler_condition:
            self.active_task_count = max(0, self.active_task_count - 1)
            self.scheduler_condition.notify_all()
