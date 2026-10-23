"""Bounded host-table prefetch; collectives stay on the consuming stream."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import msgspec
import torch

if TYPE_CHECKING:
    from sglang.srt.layers.engram import Engram
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class EngramPrefetchLayout(msgspec.Struct, frozen=True):
    """Rank-major hash segments, split at the same local token offset."""

    counts: tuple[int, ...]
    local_segment: int
    local_rows: int
    chunk_size: int

    @property
    def num_chunks(self) -> int:
        return (max(self.counts, default=0) + self.chunk_size - 1) // self.chunk_size

    def block_sizes(self, block: int) -> tuple[int, ...]:
        start = block * self.chunk_size
        return tuple(min(self.chunk_size, max(0, n - start)) for n in self.counts)

    def global_slices(self, block: int) -> tuple[slice, ...]:
        start, offset, slices = block * self.chunk_size, 0, []
        for count, size in zip(self.counts, self.block_sizes(block)):
            lo = offset + min(start, count)
            slices.append(slice(lo, lo + size))
            offset += count
        return tuple(slices)

    def local_slice(self, block: int) -> slice:
        count = min(self.counts[self.local_segment], self.local_rows)
        start = min(block * self.chunk_size, count)
        return slice(start, min(start + self.chunk_size, count))


class EngramPrefetcher:
    """Two current-layer chunks plus one first chunk per look-ahead layer.

    A model owns the buffers for its lifetime. Python bookkeeping is rebuilt per
    eager forward (or once at graph capture); graph replay uses the captured
    stream/event dependencies and the same buffer addresses.
    """

    def __init__(self, layers: list[Engram], chunk_size: int, prefetch_layers: int):
        from sglang.srt.runtime_context import get_exec, get_parallel, get_stream

        if chunk_size <= 0 or chunk_size % 128:
            raise ValueError(
                "Engram prefetch chunk size must be a positive multiple of 128"
            )
        if prefetch_layers < 0:
            raise ValueError("Engram prefetch layers must be nonnegative")
        if not layers or any(layer.embed.host_table is None for layer in layers):
            raise ValueError("Engram prefetch requires host-resident tables")
        reference = layers[0]
        if reference.q_weight.device.type != "cuda" or torch.version.hip:
            raise ValueError("Engram prefetch requires CUDA")
        self.embeddings = {layer.layer_hash_index: layer.embed for layer in layers}
        self.layer_ids = tuple(self.embeddings)
        self.chunk_size = chunk_size
        self.prefetch_layers = min(prefetch_layers, len(layers) - 1)
        self.shared = reference.embed._shared
        if not self.shared and get_exec().moe.elastic_ep_backend is not None:
            raise ValueError("Engram per-rank prefetch does not support elastic EP")
        parallel = get_parallel()
        segments = (
            1 if self.shared else max(parallel.attn_dp_size, parallel.attn_cp_size)
        )
        self.stream = get_stream("engram_prefetch")
        self.buffers = [
            torch.empty(
                segments * chunk_size,
                reference.wkv.input_size // reference.embed.dim,
                reference.embed.dim,
                dtype=torch.bfloat16,
                device=reference.q_weight.device,
            )
            for _ in range(self.prefetch_layers + 2)
        ]
        self.ready = [torch.cuda.Event() for _ in self.buffers]
        self.released = [torch.cuda.Event() for _ in self.buffers]
        for event in self.released:
            event.record()
        self.pending: dict[tuple[int, int], int] = {}
        self.free = list(range(len(self.buffers)))
        self.hashes = None
        self.layout = None
        self.order = ()
        self.position = 0

    @property
    def buffer_bytes(self) -> int:
        return sum(buffer.numel() * buffer.element_size() for buffer in self.buffers)

    @property
    def num_chunks(self) -> int:
        return 0 if self.layout is None else self.layout.num_chunks

    def begin(
        self,
        hash_ids: torch.Tensor,
        forward_batch: ForwardBatch | None,
        *,
        cp_all_tokens: bool = False,
        layer_ids: tuple[int, ...] | None = None,
    ) -> None:
        from sglang.srt.layers.dp_attention import (
            attn_cp_all_gather_into_tensor,
            dp_gather_replicate,
            dp_slot_in,
            get_dp_global_num_tokens,
            get_global_dp_buffer_len,
        )
        from sglang.srt.runtime_context import get_parallel

        self.finish()
        self.order = self.layer_ids if layer_ids is None else layer_ids
        if not self.order:
            return
        parallel = get_parallel()
        rows, num_layers, heads = hash_ids.shape
        counts, rank = (rows,), 0
        if not self.shared and parallel.tp_size > 1:
            flat = hash_ids.flatten(1).contiguous()
            if cp_all_tokens:
                counts = (rows,) * parallel.attn_cp_size
                rank = parallel.attn_cp_rank
                gathered = flat.new_empty((sum(counts), flat.shape[1]))
                attn_cp_all_gather_into_tensor(gathered, flat)
                hash_ids = gathered.view(-1, num_layers, heads)
            elif parallel.attn_dp_size > 1:
                assert forward_batch is not None
                counts = tuple(get_dp_global_num_tokens())
                assert sum(counts) == get_global_dp_buffer_len()
                rank = dp_slot_in(counts)
                gathered = flat.new_empty((sum(counts), flat.shape[1]))
                # MAX_LEN gather can zero the local padding in place.
                dp_gather_replicate(gathered, flat.clone(), forward_batch)
                hash_ids = gathered.view(-1, num_layers, heads)
        self.layout = EngramPrefetchLayout(counts, rank, rows, self.chunk_size)
        assert (
            max(self.layout.block_sizes(0), default=0) * len(counts)
            <= self.buffers[0].shape[0]
        )
        self.hashes = hash_ids
        self.hashes.record_stream(self.stream)
        self.position = 0
        current = torch.cuda.current_stream()
        self.stream.wait_stream(current)
        # begin() joins the previous producer and all old consumers precede us
        # on the compute stream. Re-record inside capture: waiting on a warmup
        # event from outside this graph would violate capture isolation.
        for event in self.released:
            event.record(current)
        self._prime()

    def _submit(self, layer: int, block: int) -> None:
        if block >= self.num_chunks or (layer, block) in self.pending:
            return
        slot = self.free.pop(0)
        self.pending[layer, block] = slot
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(self.released[slot])
            parts = [
                self.hashes[rows, layer] for rows in self.layout.global_slices(block)
            ]
            indices = parts[0] if len(parts) == 1 else torch.cat(parts)
            out = self.buffers[slot][: indices.shape[0]]
            self.embeddings[layer].gather_local_rows(indices, out)
            self.ready[slot].record(self.stream)

    def _prime(self) -> None:
        if self.position >= len(self.order):
            return
        layer = self.order[self.position]
        self._submit(layer, 0)
        self._submit(layer, 1)
        for ahead in self.order[
            self.position + 1 : self.position + 1 + self.prefetch_layers
        ]:
            self._submit(ahead, 0)

    def _combine(self, layer: int, block: int, slot: int) -> torch.Tensor:
        from sglang.srt.layers.dp_attention import is_dp_gatherv_active
        from sglang.srt.runtime_context import get_parallel

        sizes = self.layout.block_sizes(block)
        values = self.buffers[slot][: sum(sizes)]
        parallel = get_parallel()
        if not self.shared and parallel.tp_size > 1 and values.shape[0]:
            if parallel.tp_size == parallel.attn_dp_size:
                local = values.new_empty(
                    (sizes[self.layout.local_segment], *values.shape[1:])
                )
                if len(set(sizes)) == 1:
                    parallel.tp_group.reduce_scatter_tensor(local, values)
                    return local
                if is_dp_gatherv_active():
                    parallel.tp_group.reduce_scatterv(
                        values, output=local, sizes=list(sizes)
                    )
                    return local
            values = self.embeddings[layer]._reduce_owned_rows(values)
        start = sum(sizes[: self.layout.local_segment])
        return values[start : start + sizes[self.layout.local_segment]]

    @contextmanager
    def chunk(self, layer: int, block: int):
        slot = self.pending[layer, block]
        current = torch.cuda.current_stream()
        current.wait_event(self.ready[slot])
        try:
            rows = self.layout.local_slice(block)
            values = self._combine(layer, block, slot)
            # A DP gather slot may reserve more rows than this rank has
            # activations. Keep collective sizes uniform, then trim locally.
            yield rows, values[: rows.stop - rows.start]
        except BaseException:
            self.released[slot].record(current)
            self.finish()
            raise
        else:
            self.released[slot].record(current)
            del self.pending[layer, block]
            self.free.append(slot)
            self._submit(layer, block + 2)

    def end_layer(self, layer: int) -> None:
        assert self.order[self.position] == layer
        self.position += 1
        if self.position == len(self.order):
            self.finish()
        else:
            self._prime()

    def finish(self) -> None:
        """Join outstanding reads without a host/device synchronize."""
        if self.hashes is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        self.hashes = None
        self.layout = None
        self.order = ()
        self.position = 0
        self.pending.clear()
        self.free = list(range(len(self.buffers)))
