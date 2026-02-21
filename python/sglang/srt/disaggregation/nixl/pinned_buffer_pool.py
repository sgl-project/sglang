"""
Unified pinned buffer pool for NIXL KV transfers.

This module provides a per-GPU singleton pool that shares a single pinned buffer
across all NixlKVManager instances on the same GPU, avoiding double allocation
when a decode node does both receiving (from prefill) and sending (for decode->decode
migration).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class PinnedBufferPool:
    """
    Per-GPU singleton that provides a shared pinned buffer for KV transfers.
    Both receiver (prefill->decode) and sender (decode->decode migration) use this.

    Pre-allocates the full buffer at creation to avoid runtime latency spikes.
    Uses range-based allocation for variable-sized concurrent transfers.
    """

    _instances: Dict[int, "PinnedBufferPool"] = {}  # gpu_id -> pool
    _lock = threading.Lock()

    @classmethod
    def get_or_create(
        cls,
        gpu_id: int,
        dtype: torch.dtype,
        total_size_bytes: int,
    ) -> "PinnedBufferPool":
        """Get existing pool for this GPU or create one (allocates immediately)."""
        with cls._lock:
            if gpu_id not in cls._instances:
                cls._instances[gpu_id] = cls(gpu_id, dtype, total_size_bytes)
            return cls._instances[gpu_id]

    @classmethod
    def clear_instances(cls):
        """Clear all pool instances. Used for testing."""
        with cls._lock:
            cls._instances.clear()

    def __init__(self, gpu_id: int, dtype: torch.dtype, total_size_bytes: int):
        self.gpu_id = gpu_id
        self.dtype = dtype
        self.total_size_bytes = total_size_bytes

        # Pre-allocate the full buffer NOW
        elem_size = torch.tensor([], dtype=self.dtype).element_size()
        num_elements = total_size_bytes // elem_size
        logger.info(
            f"[PinnedBufferPool] Pre-allocating {total_size_bytes / 1e9:.2f}GB "
            f"pinned buffer for GPU {gpu_id}"
        )
        self._buffer = torch.empty(num_elements, dtype=self.dtype, pin_memory=True)
        logger.info(f"[PinnedBufferPool] Allocation complete")

        # Range tracking: list of (start, end) tuples for allocated regions
        # Uses simple first-fit allocation
        self._allocated_ranges: List[Tuple[int, int]] = []  # [(start, end), ...]
        self._range_lock = threading.Lock()
        self._range_available = threading.Condition(self._range_lock)

        # Track NIXL registrations per agent (each agent needs its own registration)
        self._nixl_descs_by_agent: Dict[str, Any] = {}
        self._warned_full = False

    def allocate(
        self, size_bytes: int, timeout: Optional[float] = None
    ) -> Tuple[int, torch.Tensor]:
        """
        Allocate a contiguous region of the given size.
        Blocks if no space available until space is released.

        Args:
            size_bytes: Number of bytes to allocate
            timeout: Optional timeout in seconds. None (default) waits indefinitely.
                     Only use finite timeout for testing.

        Returns: (offset_bytes, buffer_view)
        """
        # Align to 256 bytes for better memory access
        aligned_size = ((size_bytes + 255) // 256) * 256
        log_interval = 10.0  # Log status every 10 seconds while waiting

        with self._range_available:
            deadline = time.time() + timeout if timeout is not None else None
            last_log_time = 0.0

            while True:
                # Try to find a free region (first-fit)
                offset = self._find_free_region(aligned_size)
                if offset is not None:
                    # Mark as allocated
                    self._allocated_ranges.append((offset, offset + aligned_size))
                    self._allocated_ranges.sort()  # Keep sorted for efficient search
                    self._warned_full = False

                    # Return view into buffer
                    elem_size = self._buffer.element_size()
                    start_elem = offset // elem_size
                    end_elem = start_elem + (aligned_size // elem_size)
                    return offset, self._buffer[start_elem:end_elem]

                # No space - log status periodically
                now = time.time()
                if now - last_log_time >= log_interval:
                    allocated_bytes = sum(e - s for s, e in self._allocated_ranges)
                    logger.warning(
                        f"[PinnedBufferPool] GPU {self.gpu_id}: Waiting for space. "
                        f"Need {size_bytes / 1e6:.1f}MB, "
                        f"buffer {allocated_bytes / 1e6:.1f}/{self.total_size_bytes / 1e6:.1f}MB used, "
                        f"{len(self._allocated_ranges)} active allocations. "
                        f"Consider increasing --nixl-cpu-buffer-size-gb if this persists."
                    )
                    last_log_time = now

                # Check timeout (only used for testing)
                if deadline is not None:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        raise RuntimeError(
                            f"[PinnedBufferPool] Timeout waiting for pinned buffer space. "
                            f"Needed {size_bytes} bytes ({size_bytes / 1e6:.2f}MB), "
                            f"total buffer {self.total_size_bytes} bytes ({self.total_size_bytes / 1e9:.2f}GB), "
                            f"allocated ranges: {len(self._allocated_ranges)}. "
                            f"Consider increasing --nixl-cpu-buffer-size-gb."
                        )
                    wait_time = min(remaining, log_interval)
                else:
                    wait_time = log_interval

                self._range_available.wait(timeout=wait_time)

    def _find_free_region(self, size_bytes: int) -> Optional[int]:
        """Find first free region that can fit size_bytes. Returns offset or None."""
        if not self._allocated_ranges:
            # Buffer is empty - use start
            if size_bytes <= self.total_size_bytes:
                return 0
            return None

        # Check gap before first allocation
        if self._allocated_ranges[0][0] >= size_bytes:
            return 0

        # Check gaps between allocations
        for i in range(len(self._allocated_ranges) - 1):
            gap_start = self._allocated_ranges[i][1]
            gap_end = self._allocated_ranges[i + 1][0]
            if gap_end - gap_start >= size_bytes:
                return gap_start

        # Check gap after last allocation
        last_end = self._allocated_ranges[-1][1]
        if self.total_size_bytes - last_end >= size_bytes:
            return last_end

        return None

    def release(self, offset: int):
        """Release a previously allocated region."""
        with self._range_available:
            # Find and remove the range starting at this offset
            original_len = len(self._allocated_ranges)
            self._allocated_ranges = [
                (s, e) for s, e in self._allocated_ranges if s != offset
            ]
            if len(self._allocated_ranges) == original_len:
                logger.warning(
                    f"[PinnedBufferPool] Attempted to release unknown offset {offset}"
                )
            self._range_available.notify_all()

    def get_buffer_info(self) -> Tuple[int, int]:
        """Return (data_ptr, nbytes) for NIXL registration."""
        return (self._buffer.data_ptr(), self._buffer.nbytes)

    def register_with_nixl(self, agent) -> Any:
        """Register full buffer with NIXL agent.

        Each NIXL agent needs its own registration, even for the same buffer.
        This is because NIXL descriptors are agent-specific.
        """
        agent_name = agent.name
        if agent_name in self._nixl_descs_by_agent:
            logger.debug(
                f"[PinnedBufferPool] Buffer already registered with agent {agent_name}"
            )
            return self._nixl_descs_by_agent[agent_name]

        addr = [(self._buffer.data_ptr(), self._buffer.nbytes, 0, "")]
        descs = agent.register_memory(addr, "DRAM")
        if not descs:
            raise Exception(
                f"[PinnedBufferPool] NIXL memory registration failed for pinned buffer "
                f"with agent {agent_name}"
            )
        self._nixl_descs_by_agent[agent_name] = descs
        logger.info(
            f"[PinnedBufferPool] Registered pinned buffer with NIXL agent {agent_name}: "
            f"{self._buffer.nbytes / 1e9:.2f}GB"
        )
        return descs

    @property
    def buffer(self) -> torch.Tensor:
        """Get the underlying buffer tensor."""
        return self._buffer

    def get_stats(self) -> Dict[str, Any]:
        """Get allocation statistics for monitoring."""
        with self._range_lock:
            allocated_bytes = sum(e - s for s, e in self._allocated_ranges)
            return {
                "gpu_id": self.gpu_id,
                "total_bytes": self.total_size_bytes,
                "allocated_bytes": allocated_bytes,
                "free_bytes": self.total_size_bytes - allocated_bytes,
                "num_allocations": len(self._allocated_ranges),
                "utilization": allocated_bytes / self.total_size_bytes
                if self.total_size_bytes > 0
                else 0.0,
            }
