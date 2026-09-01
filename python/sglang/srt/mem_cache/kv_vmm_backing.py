from __future__ import annotations

import logging
from math import prod
from typing import TYPE_CHECKING, List, Optional, Sequence

import torch

from sglang.srt.utils.cuda_vmm_utils import (
    BumpArenaStub,
    VmmReservation,
    align_up,
    allocation_handle_type_name,
    get_device_granularity,
    make_device_allocation_prop,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KvBufferDesc

logger = logging.getLogger(__name__)


_DEFAULT_RESERVE_BYTES = 256 * (1024**3)  # 256 GiB virtual; free until committed


class KvVmmArena:
    """One device's CUDA virtual-memory reservation exposed as a ``torch.cuda.MemPool``."""

    def __init__(self, device_id: int, reserve_bytes: int = _DEFAULT_RESERVE_BYTES):
        self.device_id = int(device_id)
        with torch.cuda.device(self.device_id):
            prop = make_device_allocation_prop(self.device_id)
            self.handle_type = prop.requestedHandleTypes
            self.granularity = get_device_granularity(self.device_id)

            self.reserved = self._align(reserve_bytes)
            # Align the base to granularity so base + (granularity-aligned cursor) is
            # always a valid cuMemMap address for per-buffer commit_range().
            self._allocation = VmmReservation(
                self.reserved,
                prop,
                self.device_id,
                alignment=self.granularity,
            )
            self.base = self._allocation.base
            self._committed_by_offset = {}
            self._range_backed = 0
            self._closed = False

        self._stub = BumpArenaStub()
        self._stub.set_extents([(self.base, self.reserved)])
        self._stub.set_align(self.granularity)
        # no_split so the caching allocator hands our bump pointers back verbatim.
        self.pool = torch.cuda.MemPool(self._stub.allocator, no_split=True)
        logger.info(
            "KvVmmArena[%s] ready: device=%d reserved_va=%.1f GiB "
            "granularity=%d KiB handle_type=%s",
            self._stub.sfx,
            self.device_id,
            self.reserved / (1024**3),
            self.granularity // 1024,
            allocation_handle_type_name(self.handle_type),
        )

    def _align(self, v: int) -> int:
        return align_up(v, self.granularity)

    def commit_range(self, offset: int, want_bytes: int) -> None:
        """Back ``[base+offset, base+offset+want_bytes)`` (monotonic per offset).
        ``offset`` must be granularity-aligned (the bump allocator guarantees it).
        Maps one full handle per extension -- GB300 rejects partial-handle maps."""
        if self._closed:
            raise RuntimeError("KvVmmArena.commit_range after close")
        if offset % self.granularity != 0:
            raise ValueError(
                f"commit_range offset {offset} not granularity-aligned "
                f"({self.granularity})"
            )
        want = self._align(int(want_bytes))
        prev = self._committed_by_offset.get(offset, 0)
        if want <= prev:
            return
        if offset + want > self.reserved:
            raise RuntimeError(
                f"commit_range [{offset}, {offset + want}) exceeds reservation "
                f"{self.reserved}"
            )
        add = want - prev
        with torch.cuda.device(self.device_id):
            self._allocation.map(
                offset + prev,
                add,
                retain_handle=True,
            )
        self._committed_by_offset[offset] = want
        self._range_backed += add

    @property
    def backed_bytes(self) -> int:
        """Total physically-backed bytes (sum of scattered per-buffer ranges)."""
        return self._range_backed

    @property
    def cursor_bytes(self) -> int:
        return self._stub.cursor_bytes

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            torch.cuda.synchronize()
        except Exception as e:  # pragma: no cover
            logger.warning("KvVmmArena.close synchronize failed: %s", e)
        self._allocation.close()


# torch's caching allocator hands the pluggable allocator whole large-pool segments
# (rounded up to >= ~20 MiB) per tensor, so reserve slack beyond the tight tensor sum.
# VA is free until committed, so this costs only address space, not GPU memory.
_PER_BUFFER_VA_SLACK = 32 << 20


class _BufferSpec:
    """Per-buffer placement + backing state inside the shared VA reservation."""

    __slots__ = ("desc", "offset", "reserved_span", "aligned_reserved", "backed_to")

    def __init__(
        self,
        desc: KvBufferDesc,
        offset: int,
        reserved_span: int,
        aligned_reserved: int,
    ):
        self.desc = desc
        self.offset = offset  # granularity-aligned arena offset of this buffer
        self.reserved_span = reserved_span  # logical (unaligned) tensor bytes
        self.aligned_reserved = aligned_reserved  # reserved span rounded to granularity
        self.backed_to = 0  # bytes from offset currently backed


class KvVmmBufferOwner:
    """Owns one ``KvVmmArena`` plus its incrementally-backed KV buffers.

    ``buffer_descs`` is an ordered list of ``KvBufferDesc``; the created ``torch.empty``
    tensors are exposed in the same order as ``self.tensors``.
    """

    def __init__(
        self,
        *,
        device: str,
        device_id: int,
        store_dtype: torch.dtype,
        page_size: int,
        reserved_num_tokens: int,
        buffer_descs: Sequence[KvBufferDesc],
    ):
        self.device = device
        self.device_id = int(device_id)
        self.store_dtype = store_dtype
        self.page_size = int(page_size)
        self._reserved_num_tokens = int(reserved_num_tokens)
        self._final_num_tokens: Optional[int] = None
        self._arena: Optional[KvVmmArena] = None
        self._specs: List[_BufferSpec] = []
        self.tensors: List[torch.Tensor] = []

        itemsize = store_dtype.itemsize
        with torch.cuda.device(self.device_id):
            gran = get_device_granularity(self.device_id)
            reserved_spans = [d.reserved_span_bytes(itemsize) for d in buffer_descs]
            aligned = [align_up(s, gran) for s in reserved_spans]
            reserve_bytes = sum(a + _PER_BUFFER_VA_SLACK for a in aligned) + gran
            self._arena = KvVmmArena(self.device_id, reserve_bytes=reserve_bytes)

            # NORMAL torch tensors through the arena MemPool; torch.empty never touches
            # the unbacked tail.
            with torch.cuda.use_mem_pool(self._arena.pool):
                self.tensors = [
                    torch.empty(d.shape, dtype=store_dtype, device=self.device)
                    for d in buffer_descs
                ]

            specs: List[_BufferSpec] = []
            for desc, tensor, reserved_span, aligned_reserved in zip(
                buffer_descs, self.tensors, reserved_spans, aligned
            ):
                if prod(tensor.shape) * itemsize != reserved_span:
                    raise RuntimeError(
                        f"buffer {desc.name!r} tensor bytes "
                        f"{prod(tensor.shape) * itemsize} != reserved span {reserved_span}"
                    )
                offset = tensor.data_ptr() - self._arena.base
                if offset < 0 or offset % gran != 0:
                    raise RuntimeError(
                        f"buffer {desc.name!r} arena offset {offset} not "
                        f"granularity-aligned ({gran})"
                    )
                if offset + aligned_reserved > self._arena.reserved:
                    raise RuntimeError(
                        f"buffer {desc.name!r} [{offset}, {offset + aligned_reserved}) "
                        f"exceeds reservation {self._arena.reserved}"
                    )
                specs.append(_BufferSpec(desc, offset, reserved_span, aligned_reserved))
            self._specs = specs

            # Back one page so slot 0 is resident before capture: capture routes every
            # dummy KV write to slot 0 (out_cache_loc is zeros). finalize() backs the rest.
            self.ensure_prefix(self.page_size)

        for t in self.tensors:
            assert (
                t.is_cuda and t.device.index == self.device_id
            ), f"post-capture KV buffer landed on {t.device}, expected cuda:{self.device_id}"

    # -- backing --------------------------------------------------------------

    @staticmethod
    def _check_span(spec: _BufferSpec, span: int) -> int:
        """Return ``span`` if it fits ``[0, reserved_span]``; raise otherwise."""
        span = int(span)
        if not (0 <= span <= spec.reserved_span):
            raise ValueError(
                f"buffer {spec.desc.name!r}: span {span} outside "
                f"[0, {spec.reserved_span}] (reserved tensor bytes)"
            )
        return span

    def _back_spans(self, span_bytes: Sequence[int]) -> None:
        """Back each buffer to (at least) ``span_bytes[i]``. An out-of-reservation
        span is a descriptor bug: raise before committing anything, never clamp."""
        if self._arena is None:
            raise RuntimeError("backing after close / before construction")
        for spec, span in zip(self._specs, span_bytes):
            self._check_span(spec, span)
        gran = self._arena.granularity
        for spec, span in zip(self._specs, span_bytes):
            want = align_up(
                int(span), gran
            )  # <= aligned_reserved since span <= reserved
            if want > spec.backed_to:
                self._arena.commit_range(spec.offset, want)
                spec.backed_to = want

    def ensure_prefix(self, num_tokens: int) -> None:
        """Ensure the first ``num_tokens`` slots of every buffer are physically backed."""
        self._back_spans(
            [s.desc.prefix_span_bytes(num_tokens, self.page_size) for s in self._specs]
        )

    def finalize(self, final_num_tokens: int) -> None:
        """Back each buffer's final advertised span; set the final serving capacity."""
        final = int(final_num_tokens)
        if not (self.page_size <= final <= self._reserved_num_tokens):
            raise ValueError(
                f"final_num_tokens={final} must satisfy page_size="
                f"{self.page_size} <= final <= reserved={self._reserved_num_tokens}"
            )
        self._back_spans(
            [s.desc.final_span_bytes(final, self.page_size) for s in self._specs]
        )
        self._final_num_tokens = final

    # -- accessors / teardown -------------------------------------------------

    @property
    def backed_bytes(self) -> int:
        return self._arena.backed_bytes if self._arena is not None else 0

    def close(self) -> None:
        self.tensors = []
        self._specs = []
        if self._arena is not None:
            self._arena.close()
            self._arena = None
