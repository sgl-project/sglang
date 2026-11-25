# SPDX-License-Identifier: Apache-2.0
"""In-process stage ports for a robot policy. Model factories run once per worker."""

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib import import_module
from threading import Condition, Timer
from typing import Any, Callable

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.vla import vla_state
from sglang.multimodal_gen.runtime.realtime.action_chunk import (
    ActionDeadlineExceeded,
    remaining_ms,
)
from sglang.multimodal_gen.runtime.realtime.session import (
    BaseRealtimeState,
    resident_tensor_bytes,
)


@dataclass(frozen=True)
class ActionNode:
    """A request-triggered stage, or a periodic producer of a latest-value stream."""

    name: str
    inputs: tuple[str, ...]
    run: Callable[..., Any]
    period_s: float = 0
    repeat_latest: bool = False
    retained_gpu_bytes: int = 0


class _PeriodicPort:
    """One in-flight update and one replaceable input per session and producer."""

    def __init__(self, node, executor):
        self.node = node
        self.executor = executor
        self.changed = Condition()
        self.next_start = 0.0
        self.queued = None
        self.running = False
        self.timer = None
        self.closed = False
        self.ready = False
        self.error = None
        self.latest = None
        self.local = {}

    def capture(self, ports):
        device = torch.cuda.current_device() if torch.cuda.is_available() else None
        event = (
            torch.cuda.current_stream(device).record_event()
            if device is not None
            else None
        )
        inputs = {k: ports[k] for k in self.node.inputs if k not in ("emit", "state")}
        self._check_size(inputs)
        return inputs, device, event

    def _check_size(self, *roots):
        if resident_tensor_bytes(roots) > self.node.retained_gpu_bytes:
            raise ValueError(
                f"Stage {self.node.name} exceeded its retained_gpu_bytes reservation"
            )

    def push(self, update):
        with self.changed:
            try:
                self._check_size(update, self.latest, self.local)
            except ValueError as exc:
                self.error = exc.with_traceback(None)
                self.changed.notify_all()
                return
            self.queued = update
            self._schedule()

    def _schedule(self):
        if self.closed or self.queued is None or self.running or self.timer is not None:
            return
        delay = self.next_start - time.monotonic()
        if delay > 0:
            self.timer = Timer(delay, self._wake)
            self.timer.daemon = True
            self.timer.start()
        else:
            self.running = True
            self.executor.submit(self._run)

    def _wake(self):
        with self.changed:
            self.timer = None
            self._schedule()

    def _publish(self, value):
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()
        with self.changed:
            if not self.closed:
                self._check_size(self.queued, value, self.local)
                self.ready, self.latest, self.error = True, value, None
                self.changed.notify_all()

    @torch.inference_mode()
    def _run(self):
        with self.changed:
            if self.closed:
                return
            inputs, device, event = self.queued
            if not self.node.repeat_latest:
                self.queued = None
            self.next_start = time.monotonic() + self.node.period_s
            local = self.local.copy()
        inputs.update(state=local, emit=self._publish)
        try:

            def run():
                value = self.node.run(*(inputs[k] for k in self.node.inputs))
                if value is not None or "emit" not in self.node.inputs:
                    self._publish(value)

            if device is None:
                run()
            else:
                with (
                    torch.cuda.device(device),
                    torch.cuda.stream(torch.cuda.Stream(device=device)),
                ):
                    torch.cuda.current_stream(device).wait_event(event)
                    run()
                    torch.cuda.current_stream(device).synchronize()
            with self.changed:
                if not self.closed:
                    self._check_size(self.queued, self.latest, local)
                    self.local = local
        except Exception as exc:
            logging.getLogger(__name__).exception(
                "Action stage %s failed", self.node.name
            )
            with self.changed:
                self.error = exc.with_traceback(None)
                self.changed.notify_all()
        finally:
            with self.changed:
                self.running = False
                if not self.ready and self.error is None and not self.closed:
                    self.error = ValueError(
                        f"Stage {self.node.name} finished without publishing a value"
                    )
                    self.changed.notify_all()
                if self.error is not None and self.node.repeat_latest:
                    self.queued = None
                self._schedule()

    def read(self, timeout):
        with self.changed:
            if not self.changed.wait_for(
                lambda: self.ready or self.error is not None or self.closed, timeout
            ):
                raise ActionDeadlineExceeded(
                    f"Timed out waiting for stage {self.node.name}"
                )
            if self.closed:
                raise RuntimeError("Action graph session closed")
            if not self.ready and self.error is not None:
                raise self.error
            return self.latest

    def close(self):
        with self.changed:
            self.closed = True
            self.latest, self.queued = None, None
            self.local.clear()
            if self.timer is not None:
                self.timer.cancel()
            self.changed.notify_all()


class ActionGraphState(BaseRealtimeState):
    def __init__(self):
        self.streams = {}
        self.node_state = {}
        self.pending_inputs = {}

    def fork(self):
        other = ActionGraphState()
        other.streams = self.streams.copy()
        other.node_state = {
            name: values.copy() for name, values in self.node_state.items()
        }
        return other

    def tensor_roots(self):
        return self.node_state

    def reserved_bytes(self):
        # Background output and KV may change between action commits.
        return sum(port.node.retained_gpu_bytes for port in self.streams.values())

    def committed(self):
        for name, update in self.pending_inputs.items():
            self.streams[name].push(update)
        self.pending_inputs.clear()

    def discard(self, previous):
        for name, port in self.streams.items():
            if previous is None or name not in previous.streams:
                port.close()
        self.pending_inputs.clear()

    def dispose(self):
        for port in self.streams.values():
            port.close()
        self.streams.clear()
        self.node_state.clear()


class ActionStageGraph(PipelineStage):
    """Port values stay in the worker. Bind selected output ports to policy inputs."""

    def __init__(self, nodes: list[ActionNode], bindings: dict[str, str]):
        super().__init__()
        available = {"task", "observation", "deadline_unix_ms", "state", "emit"}
        self.pools = {}
        for node in nodes:
            if (
                node.name in available
                or not set(node.inputs) <= available
                or not math.isfinite(node.period_s)
                or node.period_s < 0
                or type(node.retained_gpu_bytes) is not int
                or node.retained_gpu_bytes < 0
            ):
                raise ValueError(
                    f"Invalid stage name, input ports or period for {node.name}"
                )
            if node.repeat_latest and not node.period_s:
                raise ValueError("Repeating a stage requires a positive period")
            if "emit" in node.inputs and not node.period_s:
                raise ValueError("Streaming output requires a periodic producer")
            available.add(node.name)
            if node.period_s:
                self.pools[node.name] = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix=node.name
                )
        if not set(bindings.values()) <= available - {"state", "emit"}:
            raise ValueError("Unknown action graph output port")
        self.nodes, self.bindings = tuple(nodes), dict(bindings)

    @classmethod
    def load(cls, spec, modules, server_args):
        module, sep, name = spec.partition(":")
        if not sep:
            raise ValueError("--action-stage-graph expects an installed module:factory")
        graph = getattr(import_module(module), name)(modules, server_args)
        if not isinstance(graph, cls):
            raise ValueError("Action graph factory must return ActionStageGraph")
        return graph

    def forward(self, batch, server_args):
        state = (
            batch.session.get_or_create_state(ActionGraphState)
            if batch.session
            else ActionGraphState()
        )
        vla = vla_state(batch)
        observation = dict(vla.get("observation") or {})
        deadline = batch.extra.get("action_realtime", {}).get("deadline_unix_ms")
        ports = {
            "task": batch.prompt,
            "observation": observation.copy(),
            "deadline_unix_ms": deadline,
        }
        try:
            for node in self.nodes:
                if deadline is not None:
                    remaining_ms(deadline)
                if not node.period_s:
                    ports["state"] = state.node_state.setdefault(node.name, {})
                    ports[node.name] = node.run(*(ports[name] for name in node.inputs))
                else:
                    port = state.streams.get(node.name)
                    if port is None:
                        port = state.streams[node.name] = _PeriodicPort(
                            node, self.pools[node.name]
                        )
                        port.push(port.capture(ports))
                    else:
                        # A timed-out policy step cannot update a live planner.
                        state.pending_inputs[node.name] = port.capture(ports)
                    timeout = (
                        None if deadline is None else remaining_ms(deadline) / 1000
                    )
                    ports[node.name] = port.read(timeout)
        finally:
            if not batch.session:
                state.dispose()
        for target, source in self.bindings.items():
            if target == "prompt":
                batch.prompt = ports[source]
                observation["prompt"] = ports[source]
            else:
                observation[target] = ports[source]
        vla["observation"] = observation
        return batch


class SRTReasoning:
    """One call to an SGLang LLM engine. Expose generated text and token IDs as a graph port."""

    def __init__(self, url, sampling_params=None, timeout=30):
        self.url = url.rstrip("/") + "/generate"
        self.sampling_params = {
            "max_new_tokens": 64,
            "temperature": 0,
            **(sampling_params or {}),
        }
        self.timeout = timeout

    def __call__(self, prompt, deadline_unix_ms=None):
        """Accept text or a trusted node's multimodal SRT request fields."""

        import requests

        request = dict(prompt) if isinstance(prompt, dict) else {"text": prompt}
        request["sampling_params"] = self.sampling_params
        timeout = self.timeout
        if deadline_unix_ms is not None:
            timeout = min(timeout, remaining_ms(deadline_unix_ms) / 1000)
        try:
            response = requests.post(self.url, json=request, timeout=timeout)
        except requests.Timeout:
            if deadline_unix_ms is not None:
                remaining_ms(deadline_unix_ms)
            raise
        response.raise_for_status()
        return response.json()
