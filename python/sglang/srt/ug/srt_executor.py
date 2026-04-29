# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch

from sglang.srt.ug.context import UGSRTKVTokenBinding


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

    def get_ug_request_token_binding(
        self,
        *,
        record,
        req,
        state,
    ) -> UGSRTKVTokenBinding | None:
        del state
        token_indices = self._request_token_indices(record, req)
        if token_indices is None:
            return None
        return UGSRTKVTokenBinding(
            session_id=record.session_id,
            request_id=req.rid,
            token_count=int(token_indices.numel()),
            token_indices=token_indices,
        )

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

    def _request_token_indices(self, record: Any, req: Any) -> torch.Tensor | None:
        tree_cache = getattr(self.scheduler, "tree_cache", None)
        if tree_cache is None:
            return None
        req_to_token_pool = getattr(tree_cache, "req_to_token_pool", None)
        req_to_token = getattr(req_to_token_pool, "req_to_token", None)
        if req_to_token is None:
            return None

        pool_idx = getattr(req, "req_pool_idx", None)
        token_count = int(getattr(req, "kv_committed_len", 0) or 0)
        if pool_idx is None or token_count <= 0:
            slot = self._streaming_session_slot(tree_cache, record.session_id)
            if slot is None:
                return None
            pool_idx = getattr(slot, "req_pool_idx", None)
            token_count = int(getattr(slot, "kv_committed_len", 0) or 0)
        if pool_idx is None or token_count <= 0:
            return None

        token_indices = req_to_token[pool_idx, :token_count].to(dtype=torch.int64)
        return token_indices.clone()

    @staticmethod
    def _streaming_session_slot(tree_cache: Any, session_id: str) -> Any | None:
        slots = getattr(tree_cache, "slots", None)
        if isinstance(slots, dict) and session_id in slots:
            return slots[session_id]
        streaming_session = getattr(tree_cache, "session", None)
        slots = getattr(streaming_session, "slots", None)
        if isinstance(slots, dict):
            return slots.get(session_id)
        return None
