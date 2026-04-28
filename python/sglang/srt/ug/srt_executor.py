# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any


class UGSRTSchedulerExecutorError(RuntimeError):
    """Raised when UG cannot synchronously execute an SRT scheduler request."""


class UGSRTRequestBoundaryExecutor:
    """Records materialized UG SRT requests at the execution boundary."""

    finish_request_after_execute = True

    def __init__(self) -> None:
        self.events: list[tuple[str, str, int]] = []

    def execute_ug_request(self, *, record, req, state) -> None:
        del record
        self.events.append((state.value, req.rid, len(req.origin_input_ids)))


class UGSRTSchedulerExecutor:
    """Minimal adapter from UG materialized requests into an SRT Scheduler."""

    finish_request_after_execute = False

    def __init__(
        self,
        scheduler: Any,
        *,
        run_synchronously: bool = True,
        max_sync_steps: int = 8,
        require_idle_scheduler: bool = True,
    ) -> None:
        if not hasattr(scheduler, "session_controller"):
            raise ValueError(
                "UGSRTSchedulerExecutor requires scheduler.session_controller"
            )
        self.scheduler = scheduler
        self.run_synchronously = run_synchronously
        self.max_sync_steps = max_sync_steps
        self.require_idle_scheduler = require_idle_scheduler
        self.events: list[tuple[str, str, int]] = []
        self.sync_step_count = 0

    @property
    def session_controller(self):
        return self.scheduler.session_controller

    def execute_ug_request(self, *, record, req, state) -> None:
        del record
        self._check_scheduler_idle(req)
        self.events.append((state.value, req.rid, len(req.origin_input_ids)))
        if hasattr(self.scheduler, "init_req_max_new_tokens"):
            self.scheduler.init_req_max_new_tokens(req)
        if not hasattr(self.scheduler, "_add_request_to_queue"):
            raise ValueError(
                "UGSRTSchedulerExecutor requires scheduler._add_request_to_queue"
            )
        self.scheduler._add_request_to_queue(req)
        if self.run_synchronously:
            self._run_until_request_complete(req)

    def _run_until_request_complete(self, req: Any) -> None:
        self._require_scheduler_methods(
            "get_next_batch_to_run",
            "run_batch",
            "process_batch_result",
        )

        for _ in range(self.max_sync_steps):
            if req.finished():
                self._run_idle_cleanup()
                return
            batch = self._run_scheduler_step()
            if batch is None and not req.finished():
                raise UGSRTSchedulerExecutorError(
                    f"SRT scheduler produced no batch for UG request {req.rid}"
                )

        if not req.finished():
            raise UGSRTSchedulerExecutorError(
                "SRT scheduler did not finish UG request "
                f"{req.rid} within {self.max_sync_steps} steps"
            )
        self._run_idle_cleanup()

    def _run_scheduler_step(self):
        batch = self.scheduler.get_next_batch_to_run()
        if hasattr(self.scheduler, "cur_batch"):
            self.scheduler.cur_batch = batch
        if batch:
            result = self.scheduler.run_batch(batch)
            self.scheduler.process_batch_result(batch, result)
            self.sync_step_count += 1
        elif hasattr(self.scheduler, "on_idle"):
            self.scheduler.on_idle()
        if hasattr(self.scheduler, "last_batch"):
            self.scheduler.last_batch = batch
        return batch

    def _run_idle_cleanup(self) -> None:
        if not hasattr(self.scheduler, "last_batch"):
            return
        if self.scheduler.last_batch is None:
            return
        self._run_scheduler_step()

    def _require_scheduler_methods(self, *method_names: str) -> None:
        missing = [
            method_name
            for method_name in method_names
            if not hasattr(self.scheduler, method_name)
        ]
        if missing:
            raise UGSRTSchedulerExecutorError(
                "UGSRTSchedulerExecutor synchronous mode requires scheduler methods: "
                f"{missing}"
            )

    def _check_scheduler_idle(self, req: Any) -> None:
        if not self.require_idle_scheduler:
            return
        if not hasattr(self.scheduler, "is_fully_idle"):
            return
        if self.scheduler.is_fully_idle():
            return
        raise UGSRTSchedulerExecutorError(
            "UG synchronous scheduler execution requires an idle scheduler before "
            f"enqueuing request {req.rid}"
        )
