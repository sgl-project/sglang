"""Scheduler-owned staging storage shared by successive Mooncake P/D managers."""

from __future__ import annotations

import logging

from sglang.srt.disaggregation.common.staging_buffer import (
    StagingAllocator,
    StagingBuffer,
    staging_grid_tokens,
)
from sglang.srt.disaggregation.common.staging_handler import _get_custom_mem_pool

logger = logging.getLogger(__name__)


class MooncakeStagingResource:
    def __init__(
        self,
        engine,
        kv_args,
        chunked_prefill_size: int | None,
        count: int,
        decode_bytes: int,
    ):
        self.engine = engine
        self.gpu_id = kv_args.gpu_id
        self.device = f"cuda:{self.gpu_id}"
        self.layout = self._layout(kv_args, chunked_prefill_size, count, decode_bytes)
        chunked_prefill_size = self.layout[3]  # Normalized staging grid width.
        self.prefill_buffer_bytes = (
            chunked_prefill_size // kv_args.page_size * sum(kv_args.kv_item_lens)
        )
        self.prefill_stride = (self.prefill_buffer_bytes + 255) // 256 * 256
        self.count = count
        self.decode_bytes = decode_bytes
        self.capacity = max(count * self.prefill_stride, decode_bytes)
        self.backing = None
        self.prefill_buffers = []
        self.decode_allocator = None
        self.active_role = None
        self.unsafe_reason = None
        self._registration_attempted = False
        self._initialized = False
        self._closed = False

    @staticmethod
    def _layout(kv_args, chunked_prefill_size, count, decode_bytes):
        chunked_prefill_size = chunked_prefill_size or 8192
        if (
            kv_args.page_size <= 0
            or chunked_prefill_size <= 0
            or chunked_prefill_size % kv_args.page_size
            or count <= 0
            or decode_bytes <= 0
            or sum(kv_args.kv_item_lens) <= 0
        ):
            raise ValueError("Invalid shared staging geometry or capacity")
        return (
            kv_args.gpu_id,
            kv_args.page_size,
            tuple(kv_args.kv_item_lens),
            staging_grid_tokens(chunked_prefill_size, kv_args.page_size),
            count,
            decode_bytes,
        )

    def validate_layout(self, kv_args, chunked_prefill_size, count, decode_bytes):
        # Role switching fixes model/TP/KV geometry and requires PP=1 with no
        # speculation or host receive, so P and D must produce the same layout.
        # A mismatch during rebuild is a configuration/construction bug, not
        # transient busy state that can be resolved by draining requests.
        if self.layout != self._layout(
            kv_args, chunked_prefill_size, count, decode_bytes
        ):
            raise RuntimeError("Shared staging layout changed; restart required")

    def initialize(self):
        if self.backing is not None or self._closed:
            raise RuntimeError(
                "Shared staging resource is already initialized or closed"
            )
        prefill_bytes = self.count * self.prefill_stride
        logger.warning(
            "Allocating shared Mooncake staging on GPU %d: capacity=%.2f MiB, "
            "P=%.2f MiB, D=%.2f MiB, capacity determined by %s. "
            "This allocation is not deducted from KV sizing; reserve runtime "
            "headroom (for example by reducing mem_fraction_static).",
            self.gpu_id,
            self.capacity / (1024 * 1024),
            prefill_bytes / (1024 * 1024),
            self.decode_bytes / (1024 * 1024),
            "prefill" if prefill_bytes >= self.decode_bytes else "decode",
        )
        custom_mem_pool, _ = _get_custom_mem_pool(self.device)
        self.backing = StagingBuffer(
            self.capacity, self.device, self.gpu_id, custom_mem_pool
        )
        # Keep ownership even if the transport cannot confirm registration.
        self._registration_attempted = True
        ret = self.engine.batch_register([self.backing.get_ptr()], [self.capacity])
        if ret != 0:
            raise RuntimeError(f"Shared staging registration failed (ret={ret})")
        self.prefill_buffers = [
            StagingBuffer.from_tensor(
                self.backing.buffer[
                    i * self.prefill_stride : i * self.prefill_stride
                    + self.prefill_buffer_bytes
                ],
                self.gpu_id,
            )
            for i in range(self.count)
        ]
        self.decode_allocator = StagingAllocator(
            self.decode_bytes,
            self.device,
            self.gpu_id,
            buffer=StagingBuffer.from_tensor(
                self.backing.buffer[: self.decode_bytes], self.gpu_id
            ),
        )
        self._initialized = True
        logger.info(
            "Shared Mooncake staging registered: ptr=%#x capacity=%d P=%d D=%d",
            self.backing.get_ptr(),
            self.capacity,
            self.count * self.prefill_stride,
            self.decode_bytes,
        )

    def poison(self, reason: str):
        if self.unsafe_reason is None:
            self.unsafe_reason = reason
            logger.error(
                "Shared Mooncake staging on GPU %d (role=%s) is unsafe; "
                "role switching requires restart: %s",
                self.gpu_id,
                self.active_role,
                reason,
            )

    def check_idle(self):
        if self.unsafe_reason is not None:
            return f"staging transfer not proven drained; restart required: {self.unsafe_reason}"
        allocator = self.decode_allocator
        if allocator is not None:
            with allocator.lock:
                if allocator.allocations or allocator.alloc_order:
                    return "staging allocations are still active"
        return None

    def acquire(self, role: str):
        if role not in ("prefill", "decode"):
            raise ValueError(f"Invalid staging role: {role}")
        if not self._initialized or self._closed or self.active_role is not None:
            raise RuntimeError(
                "Shared staging resource is unavailable or already borrowed"
            )
        if reason := self.check_idle():
            raise RuntimeError(reason)
        self.active_role = role

    def synchronize(self):
        for buf in self.prefill_buffers:
            if buf._gather_stream is not None:
                buf._gather_stream.synchronize()
        if self.decode_allocator is not None:
            stream = self.decode_allocator._scatter_stream
            if stream is not None:
                stream.synchronize()

    def release(self, role: str):
        if self.active_role != role:
            raise RuntimeError("Shared staging role does not own the resource")
        if reason := self.check_idle():
            raise RuntimeError(reason)
        self.synchronize()
        self.active_role = None

    def close(self):
        if self._closed:
            return
        if self.active_role is not None:
            raise RuntimeError("Stop the staging borrower before closing its storage")
        if reason := self.check_idle():
            raise RuntimeError(reason)
        self.synchronize()
        if self._registration_attempted:
            ret = self.engine.batch_deregister([self.backing.get_ptr()])
            if ret != 0:
                raise RuntimeError(f"Shared staging deregistration failed (ret={ret})")
            self._registration_attempted = False
        self.prefill_buffers = []
        self.decode_allocator = None
        self.backing = None
        self._closed = True
