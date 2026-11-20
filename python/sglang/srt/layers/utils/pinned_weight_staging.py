"""Copy checkpoint tensors to the GPU through pinned host memory.

Weight loaders copy slices of memory-mapped checkpoint tensors, which live in
pageable memory, to the GPU. A copy from pageable memory runs well below the
bandwidth of a copy from pinned memory, and a multithreaded CPU copy from the
page cache into pinned memory is faster than either, so staging through pinned
buffers shortens the load. Loaders call `staged_copy_` at their hot copy sites;
`maybe_stage_weight_copies` scopes a stager to the load when
SGLANG_PINNED_WEIGHT_STAGING is set.
"""

import contextlib
import logging
import math
import threading
from contextvars import ContextVar
from typing import Iterator, List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_cuda_alike, is_pin_memory_available

logger = logging.getLogger(__name__)

MIN_STAGED_BYTES = 1 << 20
BUFFER_BYTES = 64 << 20


class PinnedWeightStager:
    """Copies large pageable CPU tensors to the GPU through two pinned buffers.

    `copy_(dst, src)` makes the copy and returns True when `dst` is a CUDA
    tensor, `src` is a CPU tensor that is not pinned and holds at least
    MIN_STAGED_BYTES, the shapes are equal, one dim-0 row fits a buffer, the
    caller is the thread that entered the stager, neither tensor overrides
    `__torch_function__`, and no TorchDispatchMode is active (online
    quantization counts copied elements with `CopyNumelCounter`, and staging
    adds copies). For any other copy it returns False and the caller copies.

    A staged copy is split along dim 0 into chunks that fit a buffer. The CPU
    gathers a chunk (any strides, converted to `dst.dtype` on the CPU as a
    blocking `copy_` does) into one buffer while the previous chunk is copied
    from the other with `non_blocking=True` on the current stream of `dst`'s
    device, so later work on that stream is ordered after it. Each buffer's
    event is waited on before the buffer is refilled, and exiting waits for
    both. The gather runs on torch's intra-op threads, and with one thread it is
    about as slow as the pageable copy it replaces, so the stager raises them to
    at least `num_threads` while active and restores them on exit.
    """

    def __init__(self, num_threads: int, buffer_bytes: int = BUFFER_BYTES):
        self.num_threads = num_threads
        self.buffer_bytes = buffer_bytes
        self.staged_bytes = 0
        self._owner: Optional[int] = None
        self._saved_num_threads = 0
        self._device: Optional[torch.device] = None
        self._buffers: List[torch.Tensor] = []
        self._events: List[torch.cuda.Event] = []
        self._pending = [False, False]
        self._next = 0

    def __enter__(self) -> "PinnedWeightStager":
        self._owner = threading.get_ident()
        self._saved_num_threads = torch.get_num_threads()
        if self.num_threads > self._saved_num_threads:
            torch.set_num_threads(self.num_threads)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            for event, pending in zip(self._events, self._pending):
                if pending:
                    event.synchronize()
        finally:
            torch.set_num_threads(self._saved_num_threads)
            self._owner = None
            self._buffers = []
            self._events = []
            self._pending = [False, False]
        if self.staged_bytes:
            logger.info(
                "Copied %.2f GiB of weights to the GPU through pinned memory",
                self.staged_bytes / (1 << 30),
            )

    def copy_(self, dst: torch.Tensor, src: torch.Tensor) -> bool:
        """Stage `dst.copy_(src)` and return True, or return False untouched."""
        if not self._stages(dst, src):
            return False
        if not self._buffers:
            self._device = dst.device
            self._buffers = [
                torch.empty(
                    self.buffer_bytes, dtype=torch.uint8, device="cpu", pin_memory=True
                )
                for _ in range(2)
            ]
            self._events = [torch.cuda.Event() for _ in range(2)]
        row_bytes = math.prod(dst.shape[1:]) * dst.element_size()
        rows_per_chunk = self.buffer_bytes // row_bytes
        stream = torch.cuda.current_stream(dst.device)
        num_rows = src.shape[0]
        for start in range(0, num_rows, rows_per_chunk):
            k = self._next
            self._next ^= 1
            if self._pending[k]:
                self._events[k].synchronize()
            rows = min(rows_per_chunk, num_rows - start)
            src_chunk = src.narrow(0, start, rows)
            staged = self._buffers[k][: rows * row_bytes]
            staged = staged.view(dst.dtype).view(src_chunk.shape)
            staged.copy_(src_chunk)
            dst.narrow(0, start, rows).copy_(staged, non_blocking=True)
            self._events[k].record(stream)
            self._pending[k] = True
        self.staged_bytes += src.nbytes
        return True

    def _stages(self, dst: torch.Tensor, src: torch.Tensor) -> bool:
        return (
            threading.get_ident() == self._owner
            and dst.is_cuda
            and src.device.type == "cpu"
            and dst.shape == src.shape
            and src.nbytes >= MIN_STAGED_BYTES
            and math.prod(dst.shape[1:]) * dst.element_size() <= self.buffer_bytes
            and (self._device is None or dst.device == self._device)
            and _has_default_torch_function(dst)
            and _has_default_torch_function(src)
            and torch._C._len_torch_dispatch_stack() == 0
            and not src.is_pinned()
        )


def _has_default_torch_function(tensor: torch.Tensor) -> bool:
    """A Tensor, or a Parameter (subclass) that keeps Parameter's disabled
    `__torch_function__`."""
    return (
        type(tensor) is torch.Tensor
        or type(tensor).__torch_function__ is torch._C._disabled_torch_function_impl
    )


_stager: ContextVar[Optional[PinnedWeightStager]] = ContextVar(
    "pinned_weight_stager", default=None
)


def staged_copy_(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """`dst.copy_(src)` for weight loaders. Inside `maybe_stage_weight_copies`,
    the active stager makes the copies it accepts (see `PinnedWeightStager`)."""
    stager = _stager.get()
    if stager is None or not stager.copy_(dst, src):
        dst.copy_(src)
    return dst


@contextlib.contextmanager
def staging_weight_copies(stager: PinnedWeightStager) -> Iterator[None]:
    """Offer each `staged_copy_` call in this context to `stager`."""
    token = _stager.set(stager)
    try:
        with stager:
            yield
    finally:
        _stager.reset(token)


@contextlib.contextmanager
def maybe_stage_weight_copies() -> Iterator[None]:
    """Stage `staged_copy_` copies through pinned memory while the context is
    active, if SGLANG_PINNED_WEIGHT_STAGING is set (CUDA and ROCm)."""
    if not (
        envs.SGLANG_PINNED_WEIGHT_STAGING.get()
        and is_cuda_alike()
        and is_pin_memory_available()
    ):
        yield
        return
    stager = PinnedWeightStager(envs.SGLANG_PINNED_WEIGHT_STAGING_THREADS.get())
    with staging_weight_copies(stager):
        yield
