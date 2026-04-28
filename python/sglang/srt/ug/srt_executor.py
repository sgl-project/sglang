# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any


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

    def __init__(self, scheduler: Any) -> None:
        if not hasattr(scheduler, "session_controller"):
            raise ValueError(
                "UGSRTSchedulerExecutor requires scheduler.session_controller"
            )
        self.scheduler = scheduler
        self.events: list[tuple[str, str, int]] = []

    @property
    def session_controller(self):
        return self.scheduler.session_controller

    def execute_ug_request(self, *, record, req, state) -> None:
        del record
        self.events.append((state.value, req.rid, len(req.origin_input_ids)))
        if hasattr(self.scheduler, "init_req_max_new_tokens"):
            self.scheduler.init_req_max_new_tokens(req)
        if not hasattr(self.scheduler, "_add_request_to_queue"):
            raise ValueError(
                "UGSRTSchedulerExecutor requires scheduler._add_request_to_queue"
            )
        self.scheduler._add_request_to_queue(req)
