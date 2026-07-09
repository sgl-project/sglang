# SPDX-License-Identifier: Apache-2.0
"""Async encode/decode worker for continuous diffusion batching.

Runs text-encode admission and VAE-decode completion on a side CUDA stream
so the scheduler can keep stepping the packed denoising batch.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass(slots=True)
class StageJob:
    kind: str  # "encode" | "finalize"
    ticket: Any
    payload: Any


@dataclass(slots=True)
class StageResult:
    kind: str
    ticket: Any
    value: Any = None
    error: BaseException | None = None


def async_stages_supported(server_args: Any) -> bool:
    """Return True if encode/decode can run async without component offload."""
    if not bool(getattr(server_args, "cb_async_stages", True)):
        return False
    offload_flags = (
        "dit_cpu_offload",
        "dit_layerwise_offload",
        "text_encoder_cpu_offload",
        "image_encoder_cpu_offload",
        "vae_cpu_offload",
    )
    return not any(getattr(server_args, flag, False) for flag in offload_flags)


class AsyncContinuousStageWorker:
    """One thread + side CUDA stream for running encode/finalize jobs."""

    def __init__(self, device: torch.device | None = None) -> None:
        self._jobs: queue.Queue[StageJob | None] = queue.Queue()
        self._results: queue.Queue[StageResult] = queue.Queue()
        self._device = device
        self._stream = None
        if torch.cuda.is_available():
            if device is None:
                device = torch.device("cuda", torch.cuda.current_device())
                self._device = device
            self._stream = torch.cuda.Stream(device=device)
        self._pending = 0
        self._thread = threading.Thread(
            target=self._run,
            name="cb-stage-worker",
            daemon=True,
        )
        self._thread.start()

    @property
    def pending(self) -> int:
        return self._pending

    def submit(self, kind: str, ticket: Any, fn: Callable[[], Any]) -> None:
        self._pending += 1
        self._jobs.put(StageJob(kind=kind, ticket=ticket, payload=fn))

    def poll_results(self, block_one: bool = False) -> list[StageResult]:
        """Drain finished results, optionally blocking for one."""
        results: list[StageResult] = []
        if block_one and self._pending > 0:
            try:
                results.append(self._results.get(timeout=600.0))
            except queue.Empty:
                logger.error("Timed out waiting for async stage result")
        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                break
        self._pending -= len(results)
        return results

    def shutdown(self) -> None:
        self._jobs.put(None)
        self._thread.join(timeout=60.0)

    def _run(self) -> None:
        if self._device is not None and self._device.type == "cuda":
            torch.cuda.set_device(self._device)
        while True:
            job = self._jobs.get()
            if job is None:
                return
            result = StageResult(kind=job.kind, ticket=job.ticket)
            try:
                with torch.no_grad():
                    if self._stream is not None:
                        # Wait for the scheduler stream to finish the step that
                        # produced the latents this job consumes.
                        self._stream.wait_stream(
                            torch.cuda.default_stream(self._device)
                        )
                        with torch.cuda.stream(self._stream):
                            result.value = job.payload()
                        # Synchronize before returning results to other streams/threads.
                        self._stream.synchronize()
                    else:
                        result.value = job.payload()
            except BaseException as e:  # noqa: BLE001 - forwarded to caller
                logger.error(
                    "Async %s stage job failed: %s", job.kind, e, exc_info=True
                )
                result.error = e
            self._results.put(result)
