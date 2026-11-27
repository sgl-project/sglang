# SPDX-License-Identifier: Apache-2.0
"""Opt-in completion-boundary scheduling for the native pi0.5 pipeline."""

from __future__ import annotations

import math
import time
from itertools import takewhile
from typing import TYPE_CHECKING, Any

import zmq

from sglang.multimodal_gen.configs.pipeline_configs.pi05 import Pi05PipelineConfig
from sglang.multimodal_gen.runtime.distributed.utils import broadcast_pyobj
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.vla.pi05_segment import Pi05FlowState

logger = init_logger(__name__)


def _flow(req: Req) -> Pi05FlowState | None:
    return req.extra.get("vla", {}).get("pi05_flow")


def _remaining(req: Req) -> int:
    flow = _flow(req)
    return req.num_inference_steps if flow is None else flow.remaining


class Pi05SegmentSchedulerMixin:
    def _pi05_segments_enabled(self) -> bool:
        config = self.server_args.pipeline_config
        return (
            isinstance(config, Pi05PipelineConfig) and config.enable_segmented_actions
        )

    def _is_pi05_segment_request(self, req: Any) -> bool:
        return (
            isinstance(req, Req)
            and not req.is_warmup
            and "vla" in (req.extra or {})
            and not req.realtime_session_id
            and req.session is None
            and req.num_outputs_per_prompt == 1
        )

    def _pi05_queue_signature(self) -> list[tuple]:
        return [
            (
                type(req).__name__,
                req.request_id if isinstance(req, Req) else None,
                req.num_inference_steps if self._is_pi05_segment_request(req) else None,
                _remaining(req) if self._is_pi05_segment_request(req) else None,
            )
            for _, req, _ in self.waiting_queue
        ]

    def _pi05_deadline(self, req: Req) -> float:
        """Absolute time on this rank's monotonic clock; inf without a deadline."""
        budget_ms = req.sampling_params.deadline_ms
        if budget_ms is None:
            return math.inf
        return self._pi05_arrivals[id(req)][0] + budget_ms / 1000.0

    def _select_pi05_segment(self) -> dict[str, Any]:
        q = self.waiting_queue
        if not q:
            return {"indices": [], "arrivals": [], "expired": []}
        # Queue position is a stable tie-breaker for fresh requests received in
        # one poll. Continuations retain their original sequence and timestamp.
        for _, req, arrived in q:
            if id(req) not in self._pi05_arrivals:
                if self._is_pi05_segment_request(req) and _flow(req) is not None:
                    raise RuntimeError("Continuation lost its original arrival")
                self._pi05_arrivals[id(req)] = (arrived, self._pi05_sequence)
                self._pi05_sequence += 1
        now = time.monotonic()
        expired = [
            i
            for i, (_, req, _) in enumerate(q)
            if self._is_pi05_segment_request(req) and self._pi05_deadline(req) <= now
        ]
        # Keep control requests in arrival order relative to continuations.
        ordered = sorted(
            set(range(len(q))).difference(expired),
            key=lambda i: self._pi05_arrivals[id(q[i][1])],
        )
        if not ordered:
            return {"indices": [], "arrivals": [], "expired": expired}
        # Earliest deadline first, but only among requests ahead of the next
        # control/warmup request; without deadlines this is arrival order.
        window = sorted(
            takewhile(lambda i: self._is_pi05_segment_request(q[i][1]), ordered),
            key=lambda i: (
                self._pi05_deadline(q[i][1]),
                self._pi05_arrivals[id(q[i][1])],
            ),
        )
        if not window:
            return {
                "indices": ordered[:1],
                "arrivals": [self._pi05_arrivals[id(q[ordered[0]][1])]],
                "expired": expired,
            }
        selected = window[:1]
        reqs = [q[window[0]][1]]
        head = reqs[0]
        if self._dynamic_batching_enabled() and self._can_dynamic_batch(head, head):
            for i in window[1:]:
                req = q[i][1]
                if len(
                    reqs
                ) >= self._batching_max_size or self._batch_admission.batch_is_full(
                    reqs
                ):
                    break
                if not self._can_dynamic_batch(head, req):
                    continue
                if self._batch_admission.reject_reason_for_candidate(reqs, req) is None:
                    selected.append(i)
                    reqs.append(req)
            continuing = any(
                self._is_pi05_segment_request(r) and _flow(r) is not None
                for _, r, _ in q
            )
            hold_until = (
                min(self._pi05_arrivals[id(r)][0] for r in reqs)
                + self._batching_delay_s
            )
            if (
                not continuing
                and len(reqs) < self._batching_max_size
                and not self._batch_admission.batch_is_full(reqs)
                and now < hold_until
                # Never hold a request until after its own deadline.
                and all(self._pi05_deadline(r) > hold_until for r in reqs)
            ):
                return {"indices": [], "arrivals": [], "expired": expired}
        return {
            "indices": selected,
            "arrivals": [self._pi05_arrivals[id(r)] for r in reqs],
            "expired": expired,
        }

    def _release_pi05_state(self, req: Req) -> None:
        self._pi05_arrivals.pop(id(req), None)
        for key in (
            "pi05_flow",
            "pi05_segment_length",
            "actions",
            "prefix_context",
            "prefix_context_group",
            "observation_group",
            "observation_batch",
        ):
            req.extra["vla"].pop(key, None)

    def _drop_expired_pi05_requests(
        self, entries: list[tuple[bytes | None, Req, float]]
    ) -> None:
        for identity, req, _ in entries:
            self._release_pi05_state(req)
            output = OutputBatch(
                error=(
                    f"Action deadline of {req.sampling_params.deadline_ms:g} ms "
                    "expired before launch"
                ),
                deadline_expired=True,
            )
            try:
                self._return_item_result((identity, req), output)
            except zmq.ZMQError as exc:
                logger.error("ZMQ error returning expired pi05 request: %s", exc)

    def _get_next_pi05_segment_batch(self) -> list[tuple[bytes | None, Any]] | None:
        rank = self.worker.tp_group.rank
        packet = None
        if rank == 0:
            try:
                packet = self._select_pi05_segment()
                packet["signature"] = self._pi05_queue_signature()
            except Exception as exc:
                packet = {"error": str(exc)}
        if self.server_args.tp_size > 1:
            packet = broadcast_pyobj(
                packet,
                rank,
                self.worker.tp_cpu_group,
                src=self.worker.tp_group.ranks[0],
            )
        if "error" in packet:
            raise RuntimeError(packet["error"])
        if packet["signature"] != self._pi05_queue_signature():
            raise RuntimeError("Pi05 continuation queue differs between TP ranks")
        expired = [self.waiting_queue[i] for i in packet["expired"]]
        selected = []
        enqueue_times = []
        for i, arrival in zip(packet["indices"], packet["arrivals"], strict=True):
            identity, req, enqueued = self.waiting_queue[i]
            if self._is_pi05_segment_request(req):
                self._pi05_arrivals[id(req)] = tuple(arrival)
            else:
                self._pi05_arrivals.pop(id(req), None)
            selected.append((identity, req))
            enqueue_times.append(enqueued)
        for i in sorted(packet["indices"] + packet["expired"], reverse=True):
            del self.waiting_queue[i]
        self._drop_expired_pi05_requests(expired)
        if not selected:
            return None
        if isinstance(selected[0][1], Req):
            reqs = [req for _, req in selected]
            output_count = sum(
                max(1, int(req.num_outputs_per_prompt or 1)) for req in reqs
            )
            segmented = self._is_pi05_segment_request(reqs[0])
            self._record_batch_dispatch_metrics(
                request_count=len(reqs),
                output_count=output_count,
                queue_wait_ms=(time.monotonic() - min(enqueue_times)) * 1000.0,
                effective_max_output_count=(
                    self._batch_admission.max_admissible_batch_size(reqs[0])
                    if segmented
                    else output_count
                ),
                stop_reason="completion_boundary" if segmented else "head_ineligible",
            )
        return selected

    def _run_pi05_segment(
        self, items: list[tuple[bytes | None, Req]]
    ) -> list[tuple[tuple[bytes | None, Req], OutputBatch]]:
        """Return terminal results only, retaining unfinished requests locally."""
        budgets = [req.num_inference_steps for _, req in items]
        before = [
            0 if _flow(req) is None else _flow(req).steps_done for _, req in items
        ]
        length = min(_remaining(req) for _, req in items)
        for _, req in items:
            req.extra["vla"]["pi05_segment_length"] = length
        try:
            result = self._dispatch_items(items)
            outputs = result if isinstance(result, list) else [result]
            if len(outputs) != len(items) or not all(
                isinstance(o, OutputBatch) for o in outputs
            ):
                raise RuntimeError(
                    "Segment dispatch did not return one result per request"
                )
        except Exception as exc:
            outputs = [OutputBatch(error=str(exc)) for _ in items]
        completed = []
        for item, output, budget, previous in zip(
            items, outputs, budgets, before, strict=True
        ):
            _, req = item
            flow = _flow(req)
            if not output.error and (
                flow is None
                or flow.num_steps != budget
                or req.num_inference_steps != budget
                or flow.steps_done != previous + length
                or flow.remaining < 0
            ):
                output = OutputBatch(error="Pi05 segment state/budget invariant failed")
            if not output.error and flow.remaining > 0:
                # Arrival metadata preserves FCFS age; queue time measures only
                # the wait since this continuation became ready again.
                self.waiting_queue.append((item[0], req, time.monotonic()))
            else:
                self._release_pi05_state(req)
                completed.append((item, output))
        return completed
