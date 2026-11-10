"""Receives per-step progress from the schedulers and updates VIDEO_STORE."""

import os
from contextlib import suppress

import zmq
import zmq.asyncio

from sglang.multimodal_gen.runtime.entrypoints.openai.stores import VIDEO_STORE
from sglang.multimodal_gen.runtime.pipelines_core.request_utils import (
    split_output_request_id,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class VideoProgressTracker:
    def __init__(self) -> None:
        # job id -> completed fraction of each output
        self._jobs: dict[str, list[float]] = {}

    def register(self, job_id: str, num_outputs: int) -> None:
        self._jobs[job_id] = [0.0] * num_outputs

    def unregister(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)

    def update(
        self, request_id: str, completed: float, total: int
    ) -> tuple[str, int] | None:
        """Record progress of a request; return (job_id, percent), or None if it is not a video job."""
        if request_id in self._jobs:
            job_id, indices = request_id, range(len(self._jobs[request_id]))
        else:
            # a job with n outputs may run as n requests, see expand_request_outputs
            output = split_output_request_id(request_id)
            if output is None or output[0] not in self._jobs:
                return None
            job_id, index = output
            if index >= len(self._jobs[job_id]):
                return None
            indices = [index]
        fractions = self._jobs[job_id]
        if total > 0:
            for i in indices:
                fractions[i] = max(fractions[i], min(completed, total) / total)
        return job_id, int(100 * sum(fractions) / len(fractions))


VIDEO_PROGRESS = VideoProgressTracker()


async def apply_progress(job_id: str, progress: int) -> None:
    """Mark a queued/running job in_progress; progress only increases and is capped at 99."""
    job = await VIDEO_STORE.get(job_id)
    # no await between the check and the write, so a completed, failed or
    # deleted job cannot be overwritten here
    if job is not None and job["status"] in ("queued", "in_progress"):
        job["status"] = "in_progress"
        job["progress"] = max(job["progress"], min(progress, 99))


async def run_progress_listener(endpoint: str) -> None:
    context = zmq.asyncio.Context()
    socket = context.socket(zmq.PULL)
    try:
        socket.bind(endpoint)
    except zmq.ZMQError as e:
        logger.warning(
            "Failed to bind progress endpoint %s, progress updates disabled: %s",
            endpoint,
            e,
        )
        context.destroy(linger=0)
        return
    try:
        while True:
            try:
                request_id, completed, total = await socket.recv_json()
                if (
                    not isinstance(request_id, str)
                    or type(completed) not in (int, float)
                    or type(total) is not int
                    or not 0 <= completed <= total
                ):
                    raise ValueError("invalid progress message")
                result = VIDEO_PROGRESS.update(
                    request_id=request_id, completed=completed, total=total
                )
                if result is not None:
                    await apply_progress(job_id=result[0], progress=result[1])
            except (TypeError, ValueError, OverflowError) as e:
                logger.warning("Failed to apply progress update: %s", e)
    finally:
        context.destroy(linger=0)
        with suppress(FileNotFoundError):
            os.unlink(endpoint.removeprefix("ipc://"))
