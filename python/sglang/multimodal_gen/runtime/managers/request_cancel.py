# SPDX-License-Identifier: Apache-2.0
"""Stops cancelled requests before their forward and at each denoising step.

The HTTP server publishes the id of each cancelled request. A replica driver
cannot read its request socket while a forward runs, so it subscribes to these
ids on a separate socket and reads them at each check. The driver then
broadcasts its decision, so that every rank of the replica stops at the same
step and no rank waits in a collective that the others left.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
import zmq

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_replica_group,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.managers.request_cancel_error import (
    RequestCancelledError,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

logger = init_logger(__name__)

# a driver forgets the oldest cancelled ids first; most of them belong to
# requests of other replicas or requests that already finished
_MAX_CANCELLED_IDS = 4096


class CancelListener:
    """Receives cancelled request ids on a replica driver.

    Only the scheduler thread of the driver uses it.
    """

    def __init__(self, context: zmq.Context, endpoint: str):
        # pipelines_core imports the stages, which import this module, so
        # import it when the scheduler starts, not at module import
        from sglang.multimodal_gen.runtime.pipelines_core.request_utils import (
            split_output_request_id,
        )

        self._split_output_request_id = split_output_request_id
        self._socket = context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.connect(endpoint)
        self._cancelled: OrderedDict[str, None] = OrderedDict()

    def is_cancelled(self, request_ids: list[str]) -> bool:
        """Return True when each request, or the job that it is an output of, is cancelled."""
        self._receive()
        if not self._cancelled:
            return False
        return bool(request_ids) and all(
            self._is_cancelled(request_id) for request_id in request_ids
        )

    def _receive(self) -> None:
        while True:
            try:
                request_id = self._socket.recv(zmq.NOBLOCK).decode(errors="replace")
            except zmq.Again:
                return
            self._cancelled[request_id] = None
            self._cancelled.move_to_end(request_id)
            if len(self._cancelled) > _MAX_CANCELLED_IDS:
                self._cancelled.popitem(last=False)

    def _is_cancelled(self, request_id: str) -> bool:
        if request_id in self._cancelled:
            return True
        # a job with n outputs may run as n requests, see expand_request_outputs
        output = self._split_output_request_id(request_id)
        return output is not None and output[0] in self._cancelled


_enabled = False
_listener: CancelListener | None = None


def poll_cancellations() -> None:
    """Drain idle drivers too, so they do not backpressure other replicas."""
    if _listener is not None:
        _listener._receive()


def init_request_cancel(endpoint: str, context: zmq.Context | None) -> None:
    """Enable the checks on this rank; a replica driver also gives its context."""
    global _enabled, _listener
    _enabled = True
    if context is None:
        return
    try:
        _listener = CancelListener(context=context, endpoint=endpoint)
    except zmq.ZMQError as e:
        # the other ranks still take part in each decision, which is then False
        logger.warning(
            "Failed to connect cancel endpoint %s, request cancellation disabled: %s",
            endpoint,
            e,
        )


def _is_checked(req: Req) -> bool:
    return req.cancellable and not req.is_warmup


def raise_if_cancelled(reqs: list[Req]) -> None:
    """Raise RequestCancelledError when the client cancelled each of reqs.

    On a replica of more than one rank this is a collective: each rank must
    call it at the same point for the same requests.
    """
    if not _enabled or not reqs or not all(_is_checked(req) for req in reqs):
        return
    request_ids = [
        request_id for req in reqs for request_id in req.client_request_ids()
    ]
    cancelled = _listener is not None and _listener.is_cancelled(request_ids)
    if model_parallel_is_initialized():
        group = get_replica_group()
        if group.world_size > 1:
            # gloo on the CPU group, so the check does not wait for the GPU
            decision = torch.tensor([int(cancelled)], dtype=torch.int32)
            torch.distributed.broadcast(
                decision, src=group.ranks[0], group=group.cpu_group
            )
            cancelled = bool(decision.item())
    if cancelled:
        raise RequestCancelledError(request_ids)


def step_cancel_check(batch: Req | None) -> Callable[[], None] | None:
    """Return the check to run after each denoising step of batch, or None."""
    if not _enabled or batch is None or not _is_checked(batch):
        return None
    return lambda: raise_if_cancelled([batch])
