"""What page moves wait for.

A page move must not copy a page some unfinished work still writes, and must not
hand out a page it vacated while unfinished work still reads it. Forwards and
host-cache copies run on CUDA streams and signal with events, which a stream
waits on the GPU. A network transfer is driven by its engine outside every
stream: only the host learns that it ended, so a move that must wait for one
waits on the host (`NetworkTransfer`), never on the GPU.
"""

import ctypes
import threading
from typing import Callable, Iterator, List, Optional, Sequence

import torch


class CompletionFlag:
    """One completion, published by a host store into pinned host memory mapped
    to the device, and waited on a stream with ``cuStreamWaitValue32``.

    Slots of one `CompletionFlags` array are reused: each flag waits for its
    slot to reach its own generation, so a wait queued for an earlier flag of
    the slot is never undone by a later one."""

    __slots__ = ("_flags", "_slot", "_generation")

    def __init__(self, flags: "CompletionFlags", slot: int, generation: int):
        self._flags = flags
        self._slot = slot
        self._generation = generation

    def publish(self) -> None:
        """Mark the work done. Safe from any host thread."""
        self._flags._store(self._slot, self._generation)

    def query(self) -> bool:
        return self._flags._load(self._slot) >= self._generation

    def wait_on_stream(self, stream: torch.cuda.Stream) -> None:
        """Order ``stream``'s later work after the publish, without blocking the
        host."""
        self._flags._wait(stream, self._slot, self._generation)


class CompletionFlags:
    """An array of int32 completion slots in host-mapped pinned memory. Never
    in VMM memory: imported VMM memory does not support stream wait-value
    operations on every driver."""

    _CU_MEMHOSTALLOC_PORTABLE = 0x01
    _CU_MEMHOSTALLOC_DEVICEMAP = 0x02
    # cuStreamWaitValue32 flag: wait until *addr >= value (as unsigned).
    _CU_STREAM_WAIT_VALUE_GEQ = 0x0

    def __init__(self, num_slots: int = 4096):
        from cuda.bindings import driver as cuda

        from sglang.srt.utils.cuda_vmm_utils import check_drv

        self._cuda, self._check = cuda, check_drv
        size = 4 * num_slots
        self._host_ptr = int(
            check_drv(
                cuda.cuMemHostAlloc(
                    size,
                    self._CU_MEMHOSTALLOC_PORTABLE | self._CU_MEMHOSTALLOC_DEVICEMAP,
                ),
                "cuMemHostAlloc(completion flags)",
            )
        )
        ctypes.memset(self._host_ptr, 0, size)
        self._device_ptr = int(
            check_drv(
                cuda.cuMemHostGetDevicePointer(self._host_ptr, 0),
                "cuMemHostGetDevicePointer(completion flags)",
            )
        )
        self._values = (ctypes.c_uint32 * num_slots).from_address(self._host_ptr)
        self._num_slots = num_slots
        self._generations = [0] * num_slots
        self._free: List[int] = list(range(num_slots))
        self._lock = threading.Lock()

    def acquire(self) -> CompletionFlag:
        """A new unpublished flag. Its slot returns with `release`."""
        with self._lock:
            assert self._free, "every completion flag is in use"
            slot = self._free.pop()
            self._generations[slot] += 1
            return CompletionFlag(self, slot, self._generations[slot])

    def release(self, flag: CompletionFlag) -> None:
        """Return a flag's slot once nothing will query it again. A stream wait
        already queued on it stays satisfied: the slot only grows."""
        flag.publish()
        with self._lock:
            self._free.append(flag._slot)

    def _store(self, slot: int, generation: int) -> None:
        self._values[slot] = generation

    def _load(self, slot: int) -> int:
        return self._values[slot]

    def _wait(self, stream: torch.cuda.Stream, slot: int, generation: int) -> None:
        self._check(
            self._cuda.cuStreamWaitValue32(
                stream.cuda_stream,
                self._device_ptr + 4 * slot,
                generation,
                self._CU_STREAM_WAIT_VALUE_GEQ,
            ),
            "cuStreamWaitValue32(completion flag)",
        )


class TransferRelease:
    """The logical end of a transfer whose completion the host observes: set
    where every rank observes it, so move and reuse decisions that read it agree
    across ranks. ``quarantined``: the transfer failed and its engine has not
    confirmed that it stopped touching the pages."""

    __slots__ = ("released", "quarantined")

    def __init__(self):
        self.released = False
        self.quarantined = False


class NetworkTransfer(TransferRelease):
    """One send or receive a network engine runs on a request's rows. It is
    released where the transfer's owner observes its end on every rank; a move
    that cannot wait for that asks the storage's `NetworkWaiter`. ``peer`` is
    what the waiter judges the transfer by, opaque here."""

    __slots__ = ("peer",)

    def __init__(self, peer=None):
        super().__init__()
        self.peer = peer


# Blocks on the host until the engine is done with every transfer given, as
# judged alike on every rank of the storage's group, releasing the GIL while it
# waits. Installed by whatever runs the transfers (`KVStorage.set_network_waiter`).
NetworkWaiter = Callable[[Sequence[NetworkTransfer]], None]


def signals_of(key) -> Iterator:
    """The signals a wait or release key holds: a tuple holds several."""
    if isinstance(key, tuple):
        for k in key:
            yield from signals_of(k)
    elif key is not None:
        yield key


def unreleased_transfers(key) -> List[NetworkTransfer]:
    """The network transfers of ``key`` that have not been released."""
    return [
        s for s in signals_of(key) if isinstance(s, NetworkTransfer) and not s.released
    ]


def signal_fired(key) -> bool:
    """Whether every signal of ``key`` has completed: its events fired and its
    network transfers were released. A receive hold has nothing to wait for."""
    for s in signals_of(key):
        if isinstance(s, TransferRelease):
            if isinstance(s, NetworkTransfer) and not s.released:
                return False
        elif not s.query():
            return False
    return True


def wait_signals(
    key,
    stream: Optional[torch.cuda.Stream] = None,
    *,
    network_waiter: Optional[NetworkWaiter] = None,
) -> None:
    """Order ``stream``'s later work after every signal of ``key``: network
    transfers first, on the host through ``network_waiter`` (which every rank
    reaches at the same point, since the decision reads releases only), then
    events, on the GPU."""
    pending = unreleased_transfers(key)
    if pending:
        if network_waiter is None:
            raise RuntimeError(
                "a page move must wait for a network transfer, but no network "
                "waiter is installed on the storage"
            )
        network_waiter(pending)
        for transfer in pending:
            transfer.released = True
    stream = torch.cuda.current_stream() if stream is None else stream
    for s in signals_of(key):
        if isinstance(s, TransferRelease) or s.query():
            continue
        if isinstance(s, CompletionFlag):
            s.wait_on_stream(stream)
        else:
            stream.wait_event(s)
