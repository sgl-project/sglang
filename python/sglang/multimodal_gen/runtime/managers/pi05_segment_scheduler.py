# SPDX-License-Identifier: Apache-2.0
"""Opt-in completion-boundary scheduling for the native pi0.5 pipeline."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from sglang.multimodal_gen.runtime.distributed.utils import broadcast_pyobj
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.vla.pi05_segment import Pi05FlowState


def _flow(req: Req) -> Pi05FlowState | None:
    return req.extra.get("vla", {}).get("pi05_flow")


def _remaining(req: Req) -> int:
    flow = _flow(req)
    return req.num_inference_steps if flow is None else flow.remaining


class Pi05SegmentSchedulerMixin:
    def _pi05_segments_enabled(self) -> bool:
        return bool(
            getattr(self.server_args.pipeline_config, "enable_segmented_actions", False)
        )

    def _is_pi05_segment_request(self, req: Any) -> bool:
        return (
            isinstance(req, Req)
            and not req.is_warmup
            and "vla" in (req.extra or {})
            and not getattr(req, "realtime_session_id", None)
            and getattr(req, "session", None) is None
            and getattr(req, "num_outputs_per_prompt", 1) == 1
        )

    def _pi05_queue_signature(self) -> list[tuple]:
        return [
            (
                type(req).__name__,
                getattr(req, "request_id", None),
                req.num_inference_steps if self._is_pi05_segment_request(req) else None,
                _remaining(req) if self._is_pi05_segment_request(req) else None,
            )
            for _, req, _ in self.waiting_queue
        ]

    def _select_pi05_segment(self) -> dict[str, Any]:
        q = self.waiting_queue
        if not q:
            return {"indices": []}
        # Queue position is a stable tie-breaker for fresh requests received in
        # one poll. Continuations retain their original sequence and timestamp.
        for _, req, arrived in q:
            if self._is_pi05_segment_request(req) and not hasattr(req, "_pi05_arrival"):
                if _flow(req) is not None:
                    raise RuntimeError("Continuation lost its original arrival")
                sequence = getattr(self, "_pi05_sequence", 0)
                req._pi05_arrival = (arrived, sequence)
                self._pi05_sequence = sequence + 1
        # Keep control requests in arrival order relative to continuations.
        ordered = sorted(
            range(len(q)),
            key=lambda i: getattr(q[i][1], "_pi05_arrival", (q[i][2], -1)),
        )
        head = q[ordered[0]][1]
        if not self._is_pi05_segment_request(head):
            return {"indices": ordered[:1], "arrivals": [None]}
        selected = [ordered[0]]
        reqs = [head]
        if self._dynamic_batching_enabled() and self._can_dynamic_batch(head, head):
            for i in ordered[1:]:
                req = q[i][1]
                if len(
                    reqs
                ) >= self._batching_max_size or self._batch_admission.batch_is_full(
                    reqs
                ):
                    break
                if not self._is_pi05_segment_request(req):
                    break  # Do not start requests across a control/warmup boundary.
                if not self._can_dynamic_batch(head, req):
                    continue
                if self._batch_admission.reject_reason_for_candidate(reqs, req) is None:
                    selected.append(i)
                    reqs.append(req)
            continuing = any(
                self._is_pi05_segment_request(r) and _flow(r) is not None
                for _, r, _ in q
            )
            if (
                not continuing
                and len(reqs) < self._batching_max_size
                and not self._batch_admission.batch_is_full(reqs)
                and time.monotonic() - head._pi05_arrival[0] < self._batching_delay_s
            ):
                return {"indices": []}
        return {"indices": selected, "arrivals": [r._pi05_arrival for r in reqs]}

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
        if not packet["indices"]:
            return None
        selected = []
        for i, arrival in zip(packet["indices"], packet["arrivals"], strict=True):
            identity, req, _ = self.waiting_queue[i]
            if arrival is not None:
                req._pi05_arrival = tuple(arrival)
            selected.append((identity, req))
        for i in sorted(packet["indices"], reverse=True):
            del self.waiting_queue[i]
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
            state = req.extra["vla"]
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
                # Keep the original queue timestamp, not the continuation time.
                self.waiting_queue.append((item[0], req, req._pi05_arrival[0]))
            else:
                for key in (
                    "pi05_flow",
                    "pi05_segment_length",
                    "actions",
                    "prefix_context",
                    "prefix_context_group",
                    "observation_group",
                ):
                    state.pop(key, None)
                completed.append((item, output))
        return completed
