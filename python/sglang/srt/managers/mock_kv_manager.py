"""Content-addressed store for VLCache image-KV reuse (Stage B).

VLCache reuses the *attention* keys/values of a repeated image so the language
model does not recompute them. This module is the store those K/V (and, if
desired, ViT embeddings) live in, keyed by a content id (uid) derived from the
image's content hash.

This is a **reference ("mock") implementation**: it holds tensors in host (CPU)
memory in an ``OrderedDict`` with LRU eviction. It is intentionally simple and
single-process. A production deployment swaps it for a GPU-resident / RDMA-backed
store (e.g. a Mooncake-style global-address store) behind the same small
interface -- ``write_kv`` / ``get_kv`` / ``release`` / ``release_all`` / ``in`` --
so no caller changes. The ``MockGAHandle`` / ``MockFlag`` types mirror the shape
of such a backend's handle and async-completion primitives so the swap is
mechanical.

Interface contract (kept stable across backends):
  - ``write_kv(tensor, uid, non_blocking=False)`` store a [num_tokens, ...] tensor
    under ``uid``. With ``non_blocking`` returns ``(uid, flag)``; the flag signals
    write completion. Evicts LRU entries past ``capacity``.
  - ``get_kv(target_tensor, uid, non_blocking=False)`` copy the stored tensor for
    ``uid`` into ``target_tensor`` (shape-checked). With ``non_blocking`` returns a
    completion flag, else ``None``.
  - ``uid in manager`` membership test (does the store hold ``uid``).
  - ``release(uid)`` / ``release_all()`` drop entries.
"""

from __future__ import annotations

import collections
import logging
import os
import threading
from typing import Dict, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)


class MockFlag:
    """Mock async-completion flag mirroring a real backend's write/read handle.

    A real (RDMA/remote) store returns a handle the caller can ``wait()`` on to
    know the transfer finished. Here the transfer is a synchronous copy that has
    already completed by the time the flag is returned, so the flag is always
    ready. (An earlier version spawned a background thread that slept 1ms per
    ``set()`` to "simulate" transfer latency; with ~2*num_layers stores per prefill
    that was hundreds of thread spawns per request, contending for the GIL and
    dominating TTFT on cache-miss requests -- removed.)
    """

    def __init__(self) -> None:
        self.event = threading.Event()
        self.event.set()

    def set(self) -> None:
        self.event.set()

    def wait(self) -> None:
        self.event.wait()

    def is_set(self) -> bool:
        return self.event.is_set()


class MockGAHandle:
    """Mock global-address handle describing one stored tensor's block layout.

    Mirrors the metadata a real global-address (GA) store hands back: how many
    blocks (tokens) and bytes-per-block the entry occupies. Used here only to
    shape-check reads against writes.
    """

    def __init__(self, num_blocks: int, bytes_per_block: int) -> None:
        self.num_blocks = num_blocks
        self.bytes_per_block = bytes_per_block
        self.total_bytes = num_blocks * bytes_per_block

    def get_num_blocks(self) -> int:
        return self.num_blocks

    def get_bytes_per_block(self) -> int:
        return self.bytes_per_block


class MockKVManager:
    """Reference host-memory K/V store with LRU eviction, keyed by uid.

    Args:
        capacity: Maximum number of entries retained. Writing past this evicts the
            least-recently-used entry (a bounded footprint -- the memory-stability
            knob for Risk 1; too low hurts hit rate, too high grows host memory).
    """

    def __init__(self, capacity: int = 10000, keep_on_device: bool = True) -> None:
        self.lock = threading.Lock()
        # uid -> layout handle (for shape validation), LRU-ordered.
        self.uid_to_handle: "collections.OrderedDict[str, MockGAHandle]" = collections.OrderedDict()
        # uid -> the stored tensor.
        self.uid_to_tensor: Dict[str, torch.Tensor] = {}
        self.capacity = capacity
        # Freelist of reusable backing buffers keyed by (shape, dtype, device-str).
        # write_kv copies into a recycled buffer instead of allocating a fresh one via
        # .clone(): on a workload of many distinct images each store would otherwise
        # trigger a cudaMalloc (which synchronizes the device), and that malloc -- NOT
        # the copy -- dominated cache-miss TTFT (~0.26ms/store * 2*num_layers). A real
        # global-address store DMAs into pre-registered memory with no per-write malloc;
        # recycling buffers mirrors that steady-state behavior.
        self._free_buffers: Dict[tuple, list] = collections.defaultdict(list)
        # When True, store tensors on their original (GPU) device so write/read are
        # device-to-device copies (async, stream-ordered) rather than D2H/H2D over
        # PCIe. A CPU-resident store forces a *synchronous* pageable H2D copy on every
        # get_kv -- ~num_layers blocking copies per request -- which dominates TTFT.
        # A production global-address store is likewise GPU/RDMA-resident; keeping the
        # reference on-device mirrors that and makes the benchmark representative.
        self.keep_on_device = keep_on_device

    def __contains__(self, uid: str) -> bool:
        with self.lock:
            return uid in self.uid_to_handle

    def write_kv(
        self, tensor: torch.Tensor, uid: str, non_blocking: bool = False
    ) -> Union[str, Tuple[str, MockFlag]]:
        """Store ``tensor`` under ``uid`` (moved to CPU), evicting LRU past capacity.

        Returns ``uid`` (blocking) or ``(uid, flag)`` (non-blocking) where ``flag``
        signals write completion.
        """
        if tensor.device.type == "cuda" and not self.keep_on_device:
            tensor = tensor.contiguous().cpu()
        num_tokens = tensor.shape[0]
        elements_per_token = tensor.numel() // num_tokens
        bytes_per_token = tensor.element_size() * elements_per_token
        handle = MockGAHandle(num_tokens, bytes_per_token)
        key = (tuple(tensor.shape), tensor.dtype, str(tensor.device))
        with self.lock:
            # Reuse a recycled backing buffer of the same shape if available; only
            # allocate (empty_like) when the freelist for this shape is empty. copy_ is
            # a stream-ordered D2D copy -- no cudaMalloc, no device sync.
            free = self._free_buffers[key]
            if free:
                buf = free.pop()
                buf.copy_(tensor)
            else:
                buf = tensor.detach().clone()
            self.uid_to_handle[uid] = handle
            self.uid_to_handle.move_to_end(uid)
            self.uid_to_tensor[uid] = buf
            if len(self.uid_to_handle) > self.capacity:
                evicted_uid, _ = self.uid_to_handle.popitem(last=False)
                evicted = self.uid_to_tensor.pop(evicted_uid, None)
                if evicted is not None:
                    # Return the evicted buffer to the freelist for reuse.
                    ekey = (tuple(evicted.shape), evicted.dtype, str(evicted.device))
                    self._free_buffers[ekey].append(evicted)
                logger.debug("MockKVManager evicted uid=%s (capacity=%d)", evicted_uid, self.capacity)
        if non_blocking:
            flag = MockFlag()
            flag.set()
            return uid, flag
        return uid

    def get_kv(
        self, target_tensor: torch.Tensor, uid: str, non_blocking: bool = False
    ) -> Optional[MockFlag]:
        """Copy the stored tensor for ``uid`` into ``target_tensor`` (shape-checked).

        Returns a completion flag (non-blocking) or ``None`` (blocking). Raises
        ``KeyError`` if ``uid`` is absent and ``ValueError`` on shape mismatch --
        both fail loudly rather than silently returning stale/wrong data.
        """
        with self.lock:
            if uid not in self.uid_to_handle:
                raise KeyError(f"uid {uid} not found in MockKVManager")
            self.uid_to_handle.move_to_end(uid)  # LRU touch
            handle = self.uid_to_handle[uid]
            source_tensor = self.uid_to_tensor[uid]
        expected_shape = (handle.get_num_blocks(),) + tuple(target_tensor.shape[1:])
        if tuple(target_tensor.shape) != expected_shape:
            raise ValueError(
                f"get_kv target shape {tuple(target_tensor.shape)} != stored shape {expected_shape} for uid {uid}"
            )
        target_tensor.copy_(source_tensor)
        if non_blocking:
            flag = MockFlag()
            flag.set()
            return flag
        return None

    def release(self, uid: str) -> None:
        """Drop a single entry if present."""
        with self.lock:
            self.uid_to_handle.pop(uid, None)
            self.uid_to_tensor.pop(uid, None)

    def release_all(self) -> None:
        """Drop all entries."""
        with self.lock:
            self.uid_to_handle.clear()
            self.uid_to_tensor.clear()


# Process-global store instance (mirrors the fork; a production store would be
# constructed with a memory-bounded capacity and possibly a GPU/RDMA backend).
# VLCACHE_CAPACITY overrides the entry cap: with a bounded working set the store
# reaches steady state (evict -> freelist reuse -> no per-store cudaMalloc), which is
# the representative regime; the default 10000 only matters for pathological
# every-request-distinct-image workloads.
_capacity = int(os.environ.get("VLCACHE_CAPACITY", "10000"))
mock_kv_manager = MockKVManager(capacity=_capacity)
