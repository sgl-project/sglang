"""One-shot NIXL/UCX writes of staged encoder embeddings into host memory."""

import asyncio
import math
import threading
import time
from contextlib import ExitStack, contextmanager
from uuid import uuid4

import msgspec
import torch


class _Destination(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    address: int
    size: int
    notification: bytes
    agent_metadata: bytes


def _agent():
    # Keep NIXL optional for all other encoder transports.
    from nixl._api import nixl_agent, nixl_agent_config

    return nixl_agent(
        f"sglang-encoder-{uuid4().hex}",
        nixl_agent_config(backends=["UCX"], enable_prog_thread=True),
    )


_writer = threading.local()


@contextmanager
def _writer_agent():
    # The executor bounds the number of agents. One thread issues a single
    # write at a time, so it can safely remove P's per-target metadata before
    # the next write to P. Other threads have independent native agents.
    agent = getattr(_writer, "agent", None)
    if agent is None:
        agent = _writer.agent = _agent()
    try:
        yield agent
    except BaseException:
        # A failed native teardown can leave cached registrations on an agent.
        # Start clean on this thread's next transfer.
        del _writer.agent
        raise


class NixlTarget:
    """A receiver-owned registration. Tensor views remain valid after close()."""

    def __init__(self, receiver, size):
        self._receiver = receiver
        self.tensor = torch.empty(size, dtype=torch.uint8, device="cpu")
        self._registration = receiver._agent.register_memory(
            self.tensor, backends=["UCX"]
        )
        self._notification = uuid4().bytes
        self._completed = False
        self._abandoned = False
        try:
            metadata = receiver._agent.get_partial_agent_metadata(
                self._registration, inc_conn_info=True, backends=["UCX"]
            )
            self.descriptor = msgspec.msgpack.encode(
                _Destination(self.tensor.data_ptr(), size, self._notification, metadata)
            )
        except BaseException:
            receiver._agent.deregister_memory(self._registration)
            raise

    def close(self):
        """Close before publication, after Deliver success, or after a Release fence."""
        if self._registration is not None:
            self._receiver._agent.deregister_memory(self._registration)
            self._registration = None
            self._receiver._targets.pop(self._notification, None)

    def abandon(self):
        """Retain an unfenced destination under the receiver's admission limit."""
        if self._registration is not None:
            self._abandoned = True
            self._receiver._reap()

    def reaped(self):
        """Poll completion notifications while a Release fence is unavailable."""
        if self._registration is not None:
            self._receiver._reap()
        return self._registration is None


class NixlReceiver:
    """One scheduler-thread owner of active targets, including ambiguous writes."""

    def __init__(self, capacity=32):
        if capacity <= 0:
            raise ValueError("NIXL receiver capacity must be positive")
        self._agent = _agent()
        self._capacity = capacity
        self._targets = {}

    def _reap(self):
        # A WRITE's remote completion notification can arrive before the Deliver
        # RPC returns, or after the receiver has lost its connection to E.
        for messages in self._agent.get_new_notifs().values():
            for message in messages:
                target = self._targets.get(message)
                if target is not None:
                    target._completed = True
        for target in list(self._targets.values()):
            if target._abandoned and target._completed:
                target.close()

    def allocate(self, size):
        if type(size) is not int or size <= 0:
            raise ValueError("NIXL destination requires a positive byte count")
        self._reap()
        if len(self._targets) >= self._capacity:
            raise RuntimeError("NIXL receiver has no free transfer slots")
        target = NixlTarget(self, size)
        self._targets[target._notification] = target
        return target


def _write(source, destination, stop, deadline):
    with _writer_agent() as agent, ExitStack() as stack:
        registration = agent.register_memory(source, backends=["UCX"])
        stack.callback(agent.deregister_memory, registration)
        peer = agent.add_remote_agent(destination.agent_metadata)
        stack.callback(agent.remove_remote_agent, peer)
        local = agent.get_xfer_descs(source)
        remote = agent.get_xfer_descs(
            [(destination.address, destination.size, 0)], "DRAM"
        )
        handle = agent.initialize_xfer(
            "WRITE", local, remote, peer, destination.notification, backends=["UCX"]
        )
        stack.callback(agent.release_xfer_handle, handle)
        if stop.is_set() or time.monotonic() >= deadline:
            raise TimeoutError("NIXL encoder write expired before submission")
        state = agent.transfer(handle)
        while state == "PROC":
            # Release is a remote-write fence. A deadline or gRPC cancellation
            # cannot retire the registration until UCX reaches terminal state.
            time.sleep(0.001)
            state = agent.check_xfer_state(handle)
        if state != "DONE":
            raise RuntimeError(f"NIXL encoder write failed with status {state}")
        if time.monotonic() >= deadline:
            raise TimeoutError("NIXL encoder write exceeded its deadline")


async def write_embedding(source: torch.Tensor, descriptor: bytes, timeout: float):
    """Write one contiguous CPU tensor, draining all posted work before returning."""
    if source.device.type != "cpu" or not source.is_contiguous() or not source.numel():
        raise ValueError("NIXL encoder write requires a nonempty contiguous CPU tensor")
    if not 0 < timeout < math.inf or len(descriptor) > 1024 * 1024:
        raise ValueError("Invalid NIXL encoder timeout or destination descriptor")
    destination = msgspec.msgpack.decode(descriptor, type=_Destination)
    if (
        destination.size != source.numel() * source.element_size()
        or not 0 < destination.address <= (1 << 64) - destination.size
        or len(destination.notification) != 16
        or not destination.agent_metadata
    ):
        raise ValueError("Invalid NIXL encoder destination or embedding byte count")

    stop = threading.Event()
    deadline = time.monotonic() + timeout
    writer = asyncio.create_task(
        asyncio.to_thread(_write, source, destination, stop, deadline)
    )
    try:
        await asyncio.shield(writer)
    except asyncio.CancelledError:
        stop.set()
        # A second cancel (e.g. server shutdown during an RPC timeout) must
        # not let the staging service acknowledge Release while DMA is active.
        drain = asyncio.gather(writer, return_exceptions=True)
        while not drain.done():
            try:
                await asyncio.shield(drain)
            except asyncio.CancelledError:
                continue
        raise
