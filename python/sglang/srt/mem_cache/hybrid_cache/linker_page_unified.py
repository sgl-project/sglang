# SPDX-License-Identifier: Apache-2.0
"""Bounded device buffers for direct-linker MHA head-group transfers."""

from contextlib import contextmanager

import torch

from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import DevicePoolEntry

# Each worker owns its buffer until its blocking store I/O finishes.
_STAGING_BYTES = 64 * 1024 * 1024
_MAX_STAGING_PAGES = 64


class LinkerPageUnifiedStager:
    """Pack NHD device rows into canonical page blocks, and restore them."""

    def __init__(self, entry: DevicePoolEntry, head_group_num: int):
        if (
            len(entry.components) != 1
            or not entry.packed
            or entry._row_span != entry.page_size
            or head_group_num < 2
        ):
            raise ValueError("Head-group staging requires one packed MHA pool.")
        layers = sorted(entry.layer_mapping)
        if not layers or any(
            isinstance(entry.layer_mapping[layer], int)
            or len(entry.layer_mapping[layer]) != 2
            for layer in layers
        ):
            raise ValueError(
                "Head-group staging requires one K and V buffer per layer."
            )
        self._buffers = {}
        for layer in layers:
            buffers = [
                entry.components[0][index] for index in entry.layer_mapping[layer]
            ]
            self._buffers[layer] = [
                buffer.view(torch.uint8).view(buffer.shape[0], -1) for buffer in buffers
            ]
        first = self._buffers[layers[0]][0]
        self.device = first.device
        self._row_bytes = first.shape[1]
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError(f"Head-group staging does not support {self.device.type}.")
        if any(
            buffer.device != self.device or buffer.shape[1] != self._row_bytes
            for buffers in self._buffers.values()
            for buffer in buffers
        ):
            raise ValueError("Head-group staging requires uniform K and V row widths.")
        self._page_size = entry.page_size
        self._head_group_num = head_group_num
        if self._row_bytes % head_group_num:
            raise ValueError("KV rows must divide evenly into head groups.")
        self._group_bytes = self._row_bytes // head_group_num
        page_bytes = len(layers) * 2 * entry.page_size * self._row_bytes
        self.capacity = max(1, min(_MAX_STAGING_PAGES, _STAGING_BYTES // page_bytes))
        self._stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )
        self._staging_layers = {layer: index for index, layer in enumerate(layers)}
        with self._staging_stream():
            self._token_offsets = torch.arange(entry.page_size, device=self.device)
            self._storage = torch.empty(
                self.capacity,
                head_group_num,
                len(layers),
                2,
                entry.page_size,
                self._group_bytes,
                dtype=torch.uint8,
                device=self.device,
            )
        self.entry = DevicePoolEntry(
            name=entry.name,
            indices_from_pool=entry.indices_from_pool,
            device_pool=entry.device_pool,
            components=[
                [self._storage[:, group, layer] for layer in range(len(layers))]
                for group in range(head_group_num)
            ],
            layer_mapping=self._staging_layers,
            page_size=entry.page_size,
            rows_are_pages=True,
            packed=False,
            tp_replicated=entry.tp_replicated,
        )

    @contextmanager
    def _staging_stream(self):
        if self._stream is None:
            yield
            return
        with torch.cuda.stream(self._stream):
            try:
                yield
            finally:
                # Neither the transport nor a completed load may observe an
                # outstanding GPU copy, including after a failed scatter.
                self._stream.synchronize()

    def _indices(self, locations):
        if not 0 < len(locations) <= self.capacity:
            raise ValueError("Head-group staging window exceeds the staging capacity.")
        starts = torch.as_tensor(locations, device=self.device, dtype=torch.int64)
        return (starts[:, None] + self._token_offsets).flatten()

    def pack(self, locations) -> None:
        """Gather a window of device page starts into the staging entry."""
        with self._staging_stream():
            indices = self._indices(locations)
            pages = len(locations)
            for layer, sources in self._buffers.items():
                stage_layer = self._staging_layers[layer]
                for component, source in enumerate(sources):
                    rows = source.index_select(0, indices).view(
                        pages, self._page_size, self._head_group_num, self._group_bytes
                    )
                    self._storage[:pages, :, stage_layer, component].copy_(
                        rows.transpose(1, 2)
                    )

    def unpack(self, locations, layers) -> None:
        """Scatter the received layer ranges to device rows, then wait for them."""
        with self._staging_stream():
            indices = self._indices(locations)
            pages = len(locations)
            for layer in layers:
                stage_layer = self._staging_layers[layer]
                for component, dest in enumerate(self._buffers[layer]):
                    rows = self._storage[:pages, :, stage_layer, component]
                    dest.index_copy_(
                        0, indices, rows.transpose(1, 2).reshape(-1, self._row_bytes)
                    )
