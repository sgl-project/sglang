"""NIXL/UCX writes of staged encoder embeddings into receiver-owned host memory."""

import asyncio
import logging
import math
import threading
import time
import weakref
from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from uuid import uuid4

import msgspec
import torch

logger = logging.getLogger(__name__)


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
_MAX_CACHED_PEERS = 16


class _NixlWriter:
    """One executor thread's UCX agent and its most recently used peers."""

    def __init__(self):
        self.agent = _agent()
        self._peers: OrderedDict[bytes, str] = OrderedDict()

    def peer_for(self, metadata: bytes) -> str:
        # All slices of P's registered pool advertise the same metadata.
        if metadata in self._peers:
            self._peers.move_to_end(metadata)
            return self._peers[metadata]

        # Only NIXL can decode the agent name. Older receivers send fresh
        # partial metadata per transfer. Loading it may merge registrations or
        # reject overlapping ranges on an already known peer.
        try:
            peer = self.agent.add_remote_agent(metadata)
        except Exception:
            if not self._peers:
                raise
            self.agent = _agent()
            self._peers.clear()
            peer = self.agent.add_remote_agent(metadata)

        for old_metadata, name in self._peers.items():
            if name == peer:
                self.agent.remove_remote_agent(peer)
                del self._peers[old_metadata]
                peer = self.agent.add_remote_agent(metadata)
                break
        if len(self._peers) >= _MAX_CACHED_PEERS:
            _, oldest_peer = self._peers.popitem(last=False)
            self.agent.remove_remote_agent(oldest_peer)
        self._peers[metadata] = peer
        return peer


@contextmanager
def _writer_agent():
    # A thread has one synchronous write in flight. Its previous transfer has
    # drained before a cache eviction can disconnect a peer.
    writer = getattr(_writer, "state", None)
    if writer is None:
        writer = _writer.state = _NixlWriter()
    try:
        yield writer
    except BaseException:
        # A failed native teardown can leave cached registrations on an agent.
        # Start clean on this thread's next transfer.
        del _writer.state
        raise


class _RegisteredMemory:
    """Keep the backing tensor alive until the UCX registration is retired."""

    def __init__(self, agent, tensor):
        self.agent = agent
        self.tensor = tensor
        self.registration = agent.register_memory(tensor, backends=["UCX"])
        try:
            self.metadata = agent.get_partial_agent_metadata(
                self.registration, inc_conn_info=True, backends=["UCX"]
            )
        except BaseException:
            self.close()
            raise

    def close(self):
        self.agent.deregister_memory(self.registration)


# Cancelling a scheduler or dropping its receiver cannot revoke a submitted WRITE.
# Hold abandoned registrations through receiver teardown until a real fence.
_unfenced_targets = set()


class NixlTarget:
    """A transfer destination. Tensor views can outlive its write fence."""

    def __init__(self, receiver, size):
        self._receiver = receiver
        self._notification = uuid4().bytes
        self._completed = False
        self._abandoned = False
        self._quarantined = False
        self._one_shot = None
        pool = receiver._pool
        slot = pool.try_alloc(size) if pool is not None else None
        if slot is None:
            # Preserve admission for a large embedding or a fragmented pool.
            self.tensor = torch.empty(size, dtype=torch.uint8, device="cpu")
            self._one_shot = _RegisteredMemory(receiver._agent, self.tensor)
            metadata = self._one_shot.metadata
        else:
            view, _, slot_id = slot
            try:
                # from_numpy makes the ndarray the owner of a distinct Torch
                # storage. Every slice, dtype view or detach holds that storage,
                # even after the original tensor and request disappear.
                owner = view.numpy()
                self.tensor = torch.from_numpy(owner)
                pool.release_on_gc(owner, slot_id).atexit = False
            except BaseException:
                pool.release(slot_id)
                raise
            metadata = receiver._pool_registration.metadata
        try:
            self.descriptor = msgspec.msgpack.encode(
                _Destination(self.tensor.data_ptr(), size, self._notification, metadata)
            )
        except BaseException:
            self.close()
            raise

    def close(self):
        """Close before publication, after Deliver success, or after a Release fence."""
        if self.tensor is not None:
            if self._one_shot is not None:
                self._one_shot.close()
                self._one_shot = None
            self._receiver._targets.pop(self._notification, None)
            if self._quarantined:
                self._receiver._quarantined -= 1
            _unfenced_targets.discard(self)
            # A pooled range stays occupied if a consumer retains any alias.
            self.tensor = None

    def abandon(self):
        """Retain an unfenced destination under the receiver's admission limit."""
        if self.tensor is not None:
            self._abandoned = True
            _unfenced_targets.add(self)
            self._receiver._reap()

    def quarantine(self):
        """Stop RPC recovery. Only a late native WRITE notification can free this memory."""
        if self.tensor is not None and not self._quarantined:
            self._quarantined = True
            self._receiver._quarantined += 1

    def reaped(self):
        """Poll completion notifications while a Release fence is unavailable."""
        if self.tensor is not None:
            self._receiver._reap()
        return self.tensor is None


class NixlReceiver:
    """Register the host embedding pool once and track outstanding remote writes."""

    def __init__(self, capacity=32, *, pool_bytes=None):
        from sglang.srt.disaggregation.encoder.receiver import EmbeddingPool
        from sglang.srt.environ import envs

        if capacity <= 0:
            raise ValueError("NIXL receiver capacity must be positive")
        if pool_bytes is None:
            pool_bytes = envs.SGLANG_ENCODER_NIXL_POOL_SIZE_MB.get() * 1024**2
        if type(pool_bytes) is not int or pool_bytes < 0:
            raise ValueError("NIXL receiver pool size must be a nonnegative byte count")
        self._agent = _agent()
        self._capacity = capacity
        self._quarantined = 0
        self._targets = {}
        self._pool = None
        if pool_bytes:
            try:
                pool = EmbeddingPool(device_type="cpu", gpu_id=0, size_bytes=pool_bytes)
                pool.buffer.zero_()
                registration = _RegisteredMemory(self._agent, pool.buffer)
            except Exception as exc:
                logger.warning(
                    "Cannot pre-register the %d-byte NIXL receive pool (%s); "
                    "using per-transfer registration",
                    pool_bytes,
                    exc,
                )
            else:
                self._pool = pool
                self._pool_registration = registration
                # A live transfer must retain its mapping through interpreter shutdown.
                weakref.finalize(self, registration.close).atexit = False

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
        # A lost encoder should not take the last serving slot. Reserve at most
        # one extra quarantine slot per configured live slot, then backpressure.
        if len(self._targets) >= self._capacity + min(
            self._quarantined, self._capacity
        ):
            raise RuntimeError("NIXL receiver has no free transfer slots")
        target = NixlTarget(self, size)
        self._targets[target._notification] = target
        return target


def _write(source, destination, stop, deadline):
    with _writer_agent() as writer, ExitStack() as stack:
        peer = writer.peer_for(destination.agent_metadata)
        agent = writer.agent
        registration = agent.register_memory(source, backends=["UCX"])
        stack.callback(agent.deregister_memory, registration)
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
