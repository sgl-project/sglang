import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from sglang.srt.mem_cache.memory_pool_host import HostKVCache

logger = logging.getLogger(__name__)


@dataclass
class _NsaStagingSlot:
    name: str
    tensor: Optional[torch.Tensor] = None
    base_ptr: Optional[int] = None
    capacity_pages: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class NsaStagingBufferManager:
    def __init__(self, store: Any, default_pages: int):
        self._store = store
        self._default_pages = default_pages
        self._slots = {
            "get": _NsaStagingSlot(name="get"),
            "set": _NsaStagingSlot(name="set"),
        }
        self.enabled = False

    @staticmethod
    def supports(mem_pool_host: HostKVCache) -> bool:
        return all(
            hasattr(mem_pool_host, attr)
            for attr in (
                "layer_num",
                "indexer_page_stride_size",
                "indexer_dtype",
                "device",
            )
        )

    def initialize(self, mem_pool_host: HostKVCache) -> None:
        self.shutdown()
        if not self.supports(mem_pool_host):
            self.enabled = False
            return

        # TODO: Remove reusable staging once NSA indexer has a native
        # page_first/page_first_direct-like layout and can use direct zero-copy.
        try:
            for slot_name in ("get", "set"):
                slot = self._slots[slot_name]
                with slot.lock:
                    self._ensure_capacity_locked(
                        slot=slot,
                        min_pages=self._default_pages,
                        mem_pool_host=mem_pool_host,
                    )
            self.enabled = True
        except Exception as e:
            logger.warning(
                "Failed to pre-register NSA staging buffers, falling back to "
                "per-operation registration: %s",
                e,
            )
            self.shutdown()

    @contextmanager
    def borrow(
        self, slot_name: str, min_pages: int, mem_pool_host: HostKVCache
    ) -> torch.Tensor:
        slot = self._slots[slot_name]
        with slot.lock:
            tensor = self._ensure_capacity_locked(
                slot=slot, min_pages=min_pages, mem_pool_host=mem_pool_host
            )
            yield tensor[:min_pages]

    def disable_with_warning(self, message: str, error: Exception):
        logger.warning("%s: %s", message, error)
        self.shutdown()

    def shutdown(self):
        for slot in self._slots.values():
            with slot.lock:
                self._release_slot_locked(slot)
        self.enabled = False

    def _register_tensor(self, tensor: torch.Tensor) -> int:
        base_ptr = tensor.data_ptr()
        total_size = tensor.numel() * tensor.element_size()
        ret_code = self._store.register_buffer(base_ptr, total_size)
        if ret_code:
            raise RuntimeError(
                "Failed to register reusable NSA staging buffer, "
                f"error code: {ret_code}"
            )
        return base_ptr

    def _ensure_capacity_locked(
        self, slot: _NsaStagingSlot, min_pages: int, mem_pool_host: HostKVCache
    ) -> torch.Tensor:
        if slot.tensor is not None and slot.capacity_pages >= min_pages:
            return slot.tensor

        old_tensor = slot.tensor
        old_base_ptr = slot.base_ptr
        new_capacity = max(
            min_pages,
            self._default_pages,
            slot.capacity_pages * 2 if slot.capacity_pages > 0 else 0,
        )
        new_tensor = torch.empty(
            (
                new_capacity,
                mem_pool_host.layer_num,
                mem_pool_host.indexer_page_stride_size,
            ),
            dtype=mem_pool_host.indexer_dtype,
            device=mem_pool_host.device,
        )
        new_base_ptr = self._register_tensor(new_tensor)
        slot.tensor = new_tensor
        slot.base_ptr = new_base_ptr
        slot.capacity_pages = new_capacity
        if old_base_ptr is not None:
            try:
                self._store.unregister_buffer(old_base_ptr)
            except Exception as e:
                logger.warning(
                    "Failed to unregister old %s staging buffer during grow: %s",
                    slot.name,
                    e,
                )
        del old_tensor
        return new_tensor

    def _release_slot_locked(self, slot: _NsaStagingSlot):
        if slot.base_ptr is not None:
            try:
                self._store.unregister_buffer(slot.base_ptr)
            except Exception as e:
                logger.warning(
                    "Failed to unregister %s staging buffer: %s", slot.name, e
                )
        slot.tensor = None
        slot.base_ptr = None
        slot.capacity_pages = 0
