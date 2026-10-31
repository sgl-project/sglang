"""Completion signals that page moves wait on.

A page move must not copy a page some unfinished work still writes, and must not
hand out a page it vacated while unfinished work still reads it. Forwards and
host-cache copies run on CUDA streams and signal with events. A network transfer
is driven by its engine outside every stream; its completion is published from
the host into a `CompletionFlag`, which a stream waits on the GPU.
"""

import ctypes
import threading
from typing import Iterator, List, Optional

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

    @property
    def device_mapped(self) -> bool:
        return self._flags._device_ptr is not None

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

    def __init__(self, num_slots: int = 4096, *, device_mapped: bool = True):
        self._num_slots = num_slots
        if not device_mapped:
            # No device to order: the flags only record completion (CPU pools).
            self._buffer = (ctypes.c_uint32 * num_slots)()
            self._values = self._buffer
            self._device_ptr = None
        else:
            self._map_to_device(num_slots)
        self._generations = [0] * num_slots
        self._free: List[int] = list(range(num_slots))
        self._lock = threading.Lock()

    def _map_to_device(self, num_slots: int) -> None:
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

    def acquire(self) -> Optional[CompletionFlag]:
        """A new unpublished flag, or None when every slot is in use. Its slot
        returns with `release`."""
        with self._lock:
            if not self._free:
                return None
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
        # Monotonic: a late publish of an earlier flag of the slot (a timer, a
        # status thread) must not undo a later flag's publish.
        with self._lock:
            if self._values[slot] < generation:
                self._values[slot] = generation

    def _load(self, slot: int) -> int:
        return self._values[slot]

    def _wait(self, stream: torch.cuda.Stream, slot: int, generation: int) -> None:
        if self._device_ptr is None:
            return
        self._check(
            self._cuda.cuStreamWaitValue32(
                stream.cuda_stream,
                self._device_ptr + 4 * slot,
                generation,
                self._CU_STREAM_WAIT_VALUE_GEQ,
            ),
            "cuStreamWaitValue32(completion flag)",
        )


class CompletionFlagPool:
    """Completion flags without a fixed limit: a new array of slots is mapped
    when every slot of the earlier ones is in use (one flag per request whose
    transfer is in flight)."""

    def __init__(self, slots_per_array: int = 4096, *, device_mapped: bool = True):
        self._slots = slots_per_array
        self._device_mapped = device_mapped
        self._arrays: List[CompletionFlags] = []

    def acquire(self) -> CompletionFlag:
        for flags in self._arrays:
            flag = flags.acquire()
            if flag is not None:
                return flag
        flags = CompletionFlags(self._slots, device_mapped=self._device_mapped)
        self._arrays.append(flags)
        return flags.acquire()

    @staticmethod
    def release(flag: CompletionFlag) -> None:
        flag._flags.release(flag)


class TransferRelease:
    """The logical end of a transfer whose completion the host observes: set
    where every rank observes it, so move and reuse decisions that read it agree
    across ranks."""

    __slots__ = ("released",)

    def __init__(self):
        self.released = False


def signals_of(key) -> Iterator:
    """The signals a wait or release key holds: a tuple holds several."""
    if isinstance(key, tuple):
        for k in key:
            yield from signals_of(k)
    elif key is not None:
        yield key


def signal_fired(key) -> bool:
    """Whether every GPU-waitable signal of ``key`` has completed. A
    `TransferRelease` has nothing to wait on the GPU."""
    return all(s.query() for s in signals_of(key) if not isinstance(s, TransferRelease))


def wait_signals(key, stream: Optional[torch.cuda.Stream] = None) -> None:
    """Order ``stream``'s later work after every GPU-waitable signal of
    ``key`` that has not completed."""
    for s in signals_of(key):
        if isinstance(s, TransferRelease) or s.query():
            continue
        if isinstance(s, CompletionFlag) and not s.device_mapped:
            continue  # a CPU pool has no stream to order
        if stream is None:
            stream = torch.cuda.current_stream()
        if isinstance(s, CompletionFlag):
            s.wait_on_stream(stream)
        else:
            stream.wait_event(s)
