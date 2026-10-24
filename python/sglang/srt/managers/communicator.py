from __future__ import annotations

import asyncio
import copy
import logging
from typing import Callable, Generic, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Default deadline, in seconds, for a queueing control call to be admitted
#: and collect all rank replies. Bounds the wait so that a silent scheduler
#: rank fails the call with :class:`FanOutTimeoutError` instead of hanging
#: until the client's HTTP timeout. ``None`` disables the deadline (legacy
#: unbounded wait). Individual communicators can override it via the
#: ``queueing_timeout`` constructor argument.
DEFAULT_QUEUEING_TIMEOUT: float = 300.0


class FanOutTimeoutError(TimeoutError):
    """A queueing control call did not finish within its deadline.

    Raised to the caller when the deadline expires before all rank replies
    are collected (e.g. a silent scheduler rank). The in-flight request's
    slot stays owned by the timed-out call until its rank replies drain, so
    a late reply can never be consumed by -- or complete -- a later call,
    and partial results are never reported as successful all-rank completion.
    """


class FanOutCommunicator(Generic[T]):
    """Fan-out request + collect response primitive over zmq.

    One send is fanned out to `fan_out` recipients; the caller awaits until
    all `fan_out` responses are collected. Supports two modes:
    - "queueing": requests are serialized; concurrent callers wait in a FIFO queue.
    - "watching": concurrent callers share a single in-flight request and all
      receive the same result when it completes.

    Only one request is in-flight at any time in either mode.

    In "queueing" mode, a call that does not finish within `queueing_timeout`
    seconds raises :class:`FanOutTimeoutError` to the caller instead of
    waiting indefinitely.
    """

    def __init__(
        self,
        send: Callable[[T], None],
        fan_out: int,
        mode: str = "queueing",
        queueing_timeout: Optional[float] = DEFAULT_QUEUEING_TIMEOUT,
    ):
        self._send = send
        self._fan_out = fan_out
        self._mode = mode
        self._queueing_timeout = queueing_timeout
        self._result_event: Optional[asyncio.Event] = None
        self._result_values: Optional[List[T]] = None
        self._result_fan_out: Optional[int] = None
        self._queueing_lock = asyncio.Lock()
        # True while the slot is held by a dispatched call whose caller
        # already gave up on the deadline; late replies keep draining into
        # the orphaned collector until the slot is released.
        self._queueing_call_abandoned = False

        assert mode in ["queueing", "watching"]

    async def queueing_call(self, obj: T):
        # asyncio.Lock is FIFO-fair: a new caller cannot acquire while earlier
        # callers are still waiting, so requests are strictly serialized in
        # arrival order.
        #
        # The deadline covers the whole call: waiting for the slot behind an
        # earlier call, plus waiting for this call's rank replies. A call that
        # times out before dispatch never owned the slot and just reports the
        # timeout; a call that times out after dispatch reports the timeout
        # but keeps the slot until its rank replies drain (see handle_recv),
        # so late replies cannot complete a later call.
        cm = asyncio.timeout(self._queueing_timeout)
        dispatched = False
        try:
            async with cm:
                await self._queueing_lock.acquire()
                try:
                    self._result_event = asyncio.Event()
                    self._result_values = []
                    self._result_fan_out = self._fan_out
                    if obj is not None:
                        self._send(obj)
                    dispatched = True
                    await self._result_event.wait()
                    result_values = self._result_values
                    self._reset_queueing_slot()
                    return result_values
                except BaseException:
                    if dispatched and cm.expired():
                        # Deadline expired after dispatch: fail the caller,
                        # but keep the slot owned by this call until its rank
                        # replies drain. External cancellation and send
                        # failures free the slot immediately, as before.
                        self._queueing_call_abandoned = True
                    else:
                        self._reset_queueing_slot()
                    raise
        except TimeoutError as exc:
            # Only deadline expiries are converted; a TimeoutError raised by
            # _send itself (not by the deadline) propagates unchanged.
            if cm.expired():
                raise FanOutTimeoutError(
                    f"FanOutCommunicator queueing call timed out after "
                    f"{self._queueing_timeout} seconds waiting for "
                    f"{self._fan_out} rank replies"
                ) from exc
            raise

    def _reset_queueing_slot(self):
        self._result_event = self._result_values = None
        self._result_fan_out = None
        self._queueing_call_abandoned = False
        self._queueing_lock.release()

    async def watching_call(self, obj):
        if self._result_event is None:
            assert self._result_values is None
            self._result_values = []
            self._result_event = asyncio.Event()
            self._result_fan_out = self._fan_out

            if obj is not None:
                self._send(obj)

        # Capture local refs before await -- after event fires, the first
        # awakened coroutine clears shared state; later awaiters use local refs.
        values = self._result_values
        event = self._result_event
        await event.wait()

        result_values = copy.deepcopy(values)
        if self._result_event is event:
            self._result_event = self._result_values = None
            self._result_fan_out = None
        return result_values

    async def __call__(self, obj):
        if self._mode == "queueing":
            return await self.queueing_call(obj)
        else:
            return await self.watching_call(obj)

    def set_fan_out(self, fan_out: int):
        self._fan_out = fan_out

    def handle_recv(self, recv_obj: T):
        if (
            self._result_values is None
            or self._result_event is None
            or self._result_fan_out is None
        ):
            logger.debug(
                "Dropping communicator response without active waiter: %s",
                type(recv_obj).__name__,
            )
            return
        self._result_values.append(recv_obj)
        if len(self._result_values) == self._result_fan_out:
            self._result_event.set()
            if self._queueing_call_abandoned:
                # The dispatched call already failed with FanOutTimeoutError;
                # its rank replies have now drained, so the slot is free for
                # the next call. Late replies were consumed by the orphaned
                # collector and never by a later call.
                self._reset_queueing_slot()

    @staticmethod
    def merge_results(results):
        all_success = all([r.success for r in results])
        all_message = [r.message for r in results]
        all_message = " | ".join(all_message)
        return all_success, all_message
