# SPDX-License-Identifier: Apache-2.0
"""HiCache host mirrors of the MiniMax M3.1 NVFP4 main and index page rows.

Each device sub-pool stores one ``row_bytes`` row per (layer, page); its host
mirror keeps rows of the same width, so one page of one layer is a single
contiguous copy on both sides. Main and index share host page indices.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
import torch

from sglang.srt.mem_cache.device_pool_info import PageRowBufferInfo
from sglang.srt.mem_cache.pool_host.base import HostKVCache, host_memory_budget_bytes
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    get_allocator_from_storage,
)
from sglang.srt.utils import is_cuda, is_hip

_is_cuda = is_cuda()
_is_hip = is_hip()
if _is_cuda:
    from sgl_kernel.kvcacheio import transfer_embedding_ranges_direct

logger = logging.getLogger(__name__)

_LAYOUTS = ("layer_first", "page_first", "page_first_direct")
# Bounded batches let the copy engine start while later ones are built.
_DIRECT_BATCH_ENTRIES = 4096


def _merge_ranges(src_rows: np.ndarray, dst_rows: np.ndarray):
    """Coalesce row pairs that are consecutive on both sides into runs."""
    if src_rows.size == 0:
        return [], [], []
    breaks = np.flatnonzero((np.diff(src_rows) != 1) | (np.diff(dst_rows) != 1)) + 1
    starts = np.concatenate(([0], breaks))
    lengths = np.diff(np.concatenate((starts, [src_rows.size])))
    return src_rows[starts].tolist(), dst_rows[starts].tolist(), lengths.tolist()


class MiniMaxM3Nvfp4IndexKPoolHost(HostKVCache):
    """HiCache sidecar mirroring the packed index K page rows."""

    storage_objects_per_page = 1

    def __init__(
        self,
        buffer_info: PageRowBufferInfo,
        anchor_host: HostKVCache,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        *,
        start_layer: int,
        end_layer: int,
    ):
        if layout not in _LAYOUTS:
            raise ValueError(
                "MiniMax NVFP4 HiCache supports layer_first, page_first, "
                f"and page_first_direct layouts, got {layout!r}."
            )
        self.page_size = anchor_host.page_size
        if self.page_size != buffer_info.page_size:
            raise ValueError(
                "MiniMax NVFP4 host/device page sizes must match, got "
                f"{self.page_size} and {buffer_info.page_size}."
            )
        self.layout = layout
        self.pin_memory = pin_memory
        self.device = device
        self.allocator = get_allocator_from_storage(allocator_type)
        self.dtype = torch.uint8
        self.start_layer = start_layer
        self.end_layer = end_layer
        self._configure_device_rows(buffer_info)
        self.size = anchor_host.size
        self.page_num = anchor_host.page_num
        self.size_per_token = self.get_size_per_token()

        requested_bytes = self.page_num * self.layer_num * self.row_bytes
        available_bytes = host_memory_budget_bytes(requested_bytes)
        if requested_bytes > available_bytes:
            raise ValueError(
                "Not enough host memory for MiniMax NVFP4 index cache. "
                f"Requesting {requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free."
            )
        logger.info(
            f"Allocating {requested_bytes / 1e9:.2f} GB host memory for MiniMax "
            f"NVFP4 index cache (layout={layout})."
        )

        self.kv_buffer = self.init_kv_buffer()
        self.fd = getattr(self.allocator, "fd", None)
        self.can_use_jit = False
        self.can_use_write_back_jit = False
        self.lock = threading.RLock()
        self.clear()
        self._init_page_row_transfer()

    # ----- layout ------------------------------------------------------------

    def _configure_device_rows(self, buffer_info: PageRowBufferInfo) -> None:
        buffer_info.validate()
        self.target_layer_num = self.layer_num = len(buffer_info.rows)
        self.row_bytes = buffer_info.row_bytes
        self.device_rows = buffer_info.rows
        self.device_slab = buffer_info.slab
        self.device_num_pages = buffer_info.num_pages
        self.device_slots = buffer_info.num_pages * buffer_info.page_size

    @property
    def page_bytes(self) -> int:
        """Host bytes of one page across all layers."""
        return self.layer_num * self.row_bytes

    def get_size_per_token(self):
        return self.page_bytes // self.page_size

    def get_ksize_per_token(self):
        return self.get_size_per_token()

    def init_kv_buffer(self):
        if self.layout == "layer_first":
            dims = (self.layer_num, self.page_num, self.row_bytes)
        else:
            dims = (self.page_num, self.layer_num, 1, self.row_bytes)
        alloc_func = ALLOC_MEMORY_FUNCS[self.device_slab.device.type]
        return alloc_func(
            dims,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
        )

    def get_hybrid_pool_buffer(self):
        return [self.kv_buffer]

    def _host_layer_pages(self, layer_id: int) -> torch.Tensor:
        """``[page_num, row_bytes]`` host rows of one layer (strided for page-first)."""
        if self.layout == "layer_first":
            return self.kv_buffer[layer_id]
        return self.kv_buffer[:, layer_id, 0]

    def _host_row_ids(self, layers: np.ndarray, pages: np.ndarray) -> np.ndarray:
        """Row ids into ``kv_buffer.view(-1, row_bytes)``, layer-major."""
        layers, pages = layers[:, None], pages[None, :]
        if self.layout == "layer_first":
            return (layers * self.page_num + pages).reshape(-1)
        return (pages * self.layer_num + layers).reshape(-1)

    # ----- transfers -----------------------------------------------------------

    def _init_page_row_transfer(self) -> None:
        self._page_row_tables = None
        device = self.device_rows[0].device
        if not (_is_cuda or _is_hip) or device.type != "cuda" or not self.pin_memory:
            return
        from sglang.kernels.ops.kvcache.nvfp4_hicache import warmup_page_row_transfer

        # The kernel IO backend has no fallback; fail at construction, not mid-serving.
        warmup_page_row_transfer()
        host_table = torch.tensor(
            [self._host_layer_pages(i).data_ptr() for i in range(self.layer_num)],
            dtype=torch.uint64,
            device=device,
        )
        device_table = torch.tensor(
            [rows.data_ptr() for rows in self.device_rows],
            dtype=torch.uint64,
            device=device,
        )
        # Pool buffers are stable for this host pool's lifetime.
        self._page_row_tables = (host_table, device_table)
        self._page_row_layer_tables = [
            (host_table[i : i + 1], device_table[i : i + 1])
            for i in range(self.layer_num)
        ]
        self._host_stride = self._host_layer_pages(0).stride(0)

    def _copy_page_rows(self, tables, host_pages, device_pages, *, load):
        from sglang.kernels.ops.kvcache.nvfp4_hicache import transfer_page_rows

        transfer_page_rows(
            *tables,
            host_pages,
            device_pages,
            host_stride=self._host_stride,
            device_stride=self.row_bytes,
            row_bytes=self.row_bytes,
            load=load,
        )

    def _page_indices(self, host_indices, device_indices):
        if (
            (_is_cuda or _is_hip)
            and self.page_size == 128
            and host_indices.is_cuda
            and host_indices.device == device_indices.device
        ):
            from sglang.kernels.ops.kvcache.nvfp4_hicache import prepare_nvfp4_pages

            return prepare_nvfp4_pages(
                host_indices, device_indices, self.size, self.device_slots
            )
        if (
            host_indices.ndim != 1
            or device_indices.ndim != 1
            or host_indices.numel() != device_indices.numel()
            or host_indices.numel() % self.page_size
        ):
            raise ValueError(
                "MiniMax NVFP4 HiCache transfers require equal, full-page index lists."
            )
        pages = []
        for indices, capacity in (
            (host_indices, self.size),
            (device_indices, self.device_slots),
        ):
            if indices.dtype not in (torch.int32, torch.int64):
                raise ValueError("MiniMax NVFP4 HiCache requires integer indices.")
            rows = indices.reshape(-1, self.page_size)
            starts = rows[:, 0]
            valid = (
                (starts % self.page_size == 0).all()
                & (
                    rows
                    == starts[:, None]
                    + torch.arange(self.page_size, device=indices.device)
                ).all()
                & (indices >= 0).all()
                & (indices < capacity).all()
            )
            message = (
                "MiniMax NVFP4 HiCache requires aligned, ordered, in-range full pages."
            )
            if indices.is_cuda:
                torch._assert_async(valid, message)
            elif not valid.item():
                raise ValueError(message)
            pages.append((starts // self.page_size).contiguous())
        return tuple(pages)

    def _copy_layer_pages(
        self, layer_id, host_pages, device_pages, io_backend, *, load
    ):
        if io_backend not in ("kernel", "direct"):
            raise ValueError(f"Unsupported IO backend: {io_backend}")
        if host_pages.numel() == 0:
            return
        device = self.device_rows[layer_id]
        if device.is_cuda:
            if not self.pin_memory:
                raise ValueError("MiniMax NVFP4 HiCache requires pinned host memory.")
            if io_backend == "kernel":
                self._copy_page_rows(
                    self._page_row_layer_tables[layer_id],
                    host_pages,
                    device_pages,
                    load=load,
                )
                return
            self._copy_layer_pages_direct(layer_id, host_pages, device_pages, load=load)
            return
        # CPU reference path.
        host = self._host_layer_pages(layer_id)
        dst, src = (device, host) if load else (host, device)
        dst_pages, src_pages = (
            (device_pages, host_pages) if load else (host_pages, device_pages)
        )
        rows = src.index_select(0, src_pages.to(src.device)).to(dst.device)
        dst.index_copy_(0, dst_pages.to(dst.device), rows)

    def _copy_layer_pages_direct(self, layer_id, host_pages, device_pages, *, load):
        hp, dp = host_pages.numpy(), device_pages.numpy()
        host_rows = self._host_row_ids(np.array([layer_id]), hp)
        host_starts, device_starts, lengths = _merge_ranges(host_rows, dp)
        host = self.kv_buffer.view(-1, self.row_bytes)
        device = self.device_rows[layer_id]
        transfer_embedding_ranges_direct(
            host if load else device,
            device if load else host,
            host_starts if load else device_starts,
            device_starts if load else host_starts,
            lengths,
        )

    def _transfer_page_indices(self, host_indices, device_indices, io_backend):
        if io_backend not in ("kernel", "direct"):
            raise ValueError(f"Unsupported IO backend: {io_backend}")
        if io_backend == "direct":
            # Prepare CPU range metadata once before enqueueing any DMA.
            host_indices, device_indices = host_indices.cpu(), device_indices.cpu()
        return self._page_indices(host_indices, device_indices)

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft=False,
    ):
        self.prepare_layer_loader(
            device_pool, host_indices, device_indices, io_backend
        )(layer_id, is_draft=is_draft)

    def prepare_transfer_indices(self, host_indices, device_indices, io_backend):
        # Translate to pages once per L2 submission; the physical copies run once
        # per layer and would otherwise repeat the validation kernels each time.
        host_pages, device_pages = self._transfer_page_indices(
            host_indices, device_indices, io_backend
        )
        if io_backend == "kernel" and self._page_row_tables is not None:
            device = self._page_row_tables[0].device
            host_pages, device_pages = host_pages.to(device), device_pages.to(device)
        return host_pages, device_pages

    def load_to_device_per_layer_physical(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft=False,
    ):
        # Indices are the pages prepare_transfer_indices returned.
        self._load_layer_pages(
            host_indices, device_indices, layer_id, io_backend, is_draft=is_draft
        )

    def _load_layer_pages(
        self, host_pages, device_pages, layer_id, io_backend, *, is_draft
    ):
        if is_draft:
            raise ValueError("M3.1 NVFP4 HiCache supports only the base model")
        if not 0 <= layer_id < self.layer_num:
            raise ValueError("M3.1 NVFP4 HiCache layer is out of range")
        self._copy_layer_pages(
            layer_id, host_pages, device_pages, io_backend, load=True
        )

    def prepare_layer_loader(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        host_pages, device_pages = self.prepare_transfer_indices(
            host_indices, device_indices, io_backend
        )

        def load_layer(layer_id, *, is_draft=False):
            self._load_layer_pages(
                host_pages, device_pages, layer_id, io_backend, is_draft=is_draft
            )

        return load_layer

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        self.backup_from_device_all_layer_physical(
            device_pool,
            *self.prepare_transfer_indices(host_indices, device_indices, io_backend),
            io_backend,
        )

    def backup_from_device_all_layer_physical(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        # Indices are the pages prepare_transfer_indices returned.
        host_pages, device_pages = host_indices, device_indices
        if host_pages.numel() == 0:
            return
        if self.device_rows[0].is_cuda:
            if not self.pin_memory:
                raise ValueError("MiniMax NVFP4 HiCache requires pinned host memory.")
            if io_backend == "kernel":
                self._copy_page_rows(
                    self._page_row_tables, host_pages, device_pages, load=False
                )
                return
            if io_backend == "direct":
                self._backup_all_layers_direct(host_pages, device_pages)
                return
        for layer_id in range(self.layer_num):
            self._copy_layer_pages(
                layer_id, host_pages, device_pages, io_backend, load=False
            )

    def _backup_all_layers_direct(self, host_pages, device_pages):
        # One copy-engine plan over the whole device slab.
        hp, dp = host_pages.numpy(), device_pages.numpy()
        layers = np.arange(self.layer_num)
        src = (layers[:, None] * self.device_num_pages + dp[None, :]).reshape(-1)
        dst = self._host_row_ids(layers, hp)
        src_starts, dst_starts, lengths = _merge_ranges(src, dst)
        device = self.device_slab.view(-1, self.row_bytes)
        host = self.kv_buffer.view(-1, self.row_bytes)
        for i in range(0, len(lengths), _DIRECT_BATCH_ENTRIES):
            end = i + _DIRECT_BATCH_ENTRIES
            transfer_embedding_ranges_direct(
                device, host, src_starts[i:end], dst_starts[i:end], lengths[i:end]
            )

    # ----- flat pages ------------------------------------------------------------

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        page_idx = int(index) // self.page_size
        if self.layout == "layer_first":
            data_page = self.kv_buffer[:, page_idx]
        else:
            data_page = self.kv_buffer[page_idx, :, 0]
        return data_page.flatten() if flat else data_page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            self.page_bytes,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        )

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        page_idx = int(index) // self.page_size
        rows = data_page.reshape(self.layer_num, self.row_bytes)
        if self.layout == "layer_first":
            self.kv_buffer[:, page_idx] = rows
        else:
            self.kv_buffer[page_idx, :, 0] = rows

    def get_page_buffer_meta(self, indices):
        if self.layout not in ("page_first", "page_first_direct"):
            raise ValueError(f"Unsupported zero-copy layout: {self.layout}")
        if len(indices) % self.page_size != 0:
            raise ValueError("MiniMax NVFP4 page metadata requires full pages.")
        object_rows = np.array_split(
            np.arange(self.layer_num), self.storage_objects_per_page
        )
        object_ranges = [
            (int(rows[0]) * self.row_bytes, len(rows) * self.row_bytes)
            for rows in object_rows
        ]
        base_ptr = self.kv_buffer.data_ptr()
        ptrs, sizes = [], []
        for i in range(0, len(indices), self.page_size):
            page_ptr = base_ptr + int(indices[i]) // self.page_size * self.page_bytes
            for offset, size in object_ranges:
                ptrs.append(page_ptr + offset)
                sizes.append(size)
        return ptrs, sizes

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        if self.layout not in ("page_first", "page_first_direct"):
            return False
        return (
            self.kv_buffer.data_ptr() % page_size_bytes == 0
            and self.page_bytes % page_size_bytes == 0
        )


class MiniMaxM3Nvfp4MainKVPoolHost(MiniMaxM3Nvfp4IndexKPoolHost):
    """HiCache host mirror of the MSA-v1 main K/V page rows."""

    storage_objects_per_page = 2

    def __init__(
        self,
        buffer_info: PageRowBufferInfo,
        host_to_device_ratio: float,
        host_size: float,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        *,
        device_capacity_tokens: int,
        start_layer: int,
        end_layer: int,
        pool_label: str = "kv",
    ):
        if layout not in _LAYOUTS:
            raise ValueError(f"Unsupported MiniMax NVFP4 host layout: {layout}")
        if page_size != buffer_info.page_size:
            raise ValueError(
                "MSA-v1 NVFP4 main HiCache requires matching page sizes, got "
                f"{page_size} and {buffer_info.page_size}."
            )
        self._configure_device_rows(buffer_info)
        HostKVCache.__init__(
            self,
            dtype=torch.uint8,
            device_capacity_tokens=device_capacity_tokens,
            size_per_token=self.layer_num * self.row_bytes // page_size,
            start_layer=start_layer,
            end_layer=end_layer,
            host_to_device_ratio=host_to_device_ratio,
            host_size=host_size,
            page_size=page_size,
            layout=layout,
            pin_memory=pin_memory,
            device=device,
            allocator_type=allocator_type,
            pool_label=pool_label,
        )
        self.can_use_jit = False
        self._init_page_row_transfer()

    def get_ksize_per_token(self):
        return self.get_size_per_token() // 2
