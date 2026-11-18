# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Fixed-capacity metadata for DSV4 prefill inside breakable NPU graphs.

A captured segment records kernels that read attention / compressor metadata
by tensor address.  Capture therefore allocates one capacity-sized metadata
object per token bucket and replay refreshes its contents in place.  Nothing
in this module may replace a tensor object after capture.

The two model generations have different storage protocols and therefore get
separate metadata classes over one shared lifecycle:

* ``Dsv4PrefillGraphMetadata`` -- ratio 4 / 128: c4/c128 slots, explicit
  per-token state tables, the four compressor/indexer dispatch tables.
* ``Dsv41PrefillGraphMetadata`` -- ratio 1 / 2: per-token ratio-1 rows,
  compacted ratio-2 rows, the cycle request-bank table, explicit window /
  compressed top-k lengths and the QLI / QSLI / candidate dispatch tables.
  The fused compressor does not use the V4 explicit state table.

``_Dsv4PrefillGraphMetadataBase`` owns only operations whose semantics are
identical for both generations: capacity allocation of the shared sequence
scalars, the capture/replay fill entry point, per-layer top-k staging,
in-place kernel metadata copying and the common bindings.  Protocol-specific
layout, sentinels and lifecycle stay in the concrete class.

Only one request per replay is admitted for now (``MAX_PREFILL_GRAPH_REQS``).
Multi-request replay needs the fused indexer and sparse-attention metadata
operators to accept zero-length request slots; that is not verified yet, so
the runner's admission sends those batches to the eager path.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Dict, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
        ForwardMetadata,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# One request.  Raising this requires the fused indexer / sparse-attention
# metadata operators to accept zero-length request slots (unverified).
MAX_PREFILL_GRAPH_REQS = 1

_KERNEL_METADATA_KEYS = (
    "c1a_metadata",
    "c4a_metadata",
    "c128a_metadata",
    "li_quant_metadata",
)
# Arbitrary floor; the live metadata ops emit far fewer than this.
_MIN_KERNEL_METADATA_INTS = 1024

# The V4.1 writer, indexer prologue and compressed top-k lengths use -1 as
# the skip sentinel (unlike V4's legacy 0 sentinel).
_DSV41_SLOT_SKIP = -1


def compress_capacity(num_tokens: int, ratio: int, max_reqs: int) -> int:
    """Output rows the fused compressor reserves for one token bucket."""
    return min(num_tokens, num_tokens // ratio + max_reqs)


def _copy_1d(dst: torch.Tensor, src: Optional[torch.Tensor]) -> None:
    """Copy a live vector, zero-filling the capacity tail."""
    if src is None:
        dst.zero_()
        return
    flat = src.reshape(-1)
    n = min(flat.numel(), dst.numel())
    dst[:n].copy_(flat[:n].to(dtype=dst.dtype))
    if n < dst.numel():
        dst[n:].zero_()


def _copy_cu(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Copy a non-decreasing cu and repeat its tail for spare request slots."""
    flat = src.reshape(-1)
    n = min(flat.numel(), dst.numel())
    dst[:n].copy_(flat[:n].to(dtype=dst.dtype))
    if n < dst.numel():
        # Spare slots are zero-length requests, not a second copy of request 0.
        dst[n:].copy_(flat[n - 1].expand(dst.numel() - n))


def _copy_table(dst: torch.Tensor, src: torch.Tensor, name: str) -> None:
    dst.zero_()
    if src.shape[0] > dst.shape[0] or src.shape[1] > dst.shape[1]:
        raise RuntimeError(
            f"DSV4 prefill graph {name} overflow: live={tuple(src.shape)} "
            f"captured={tuple(dst.shape)}"
        )
    rows, cols = src.shape
    dst[:rows, :cols].copy_(src[:rows, :cols].to(dtype=dst.dtype))


class _Dsv4PrefillGraphMetadataBase(abc.ABC):
    """Lifecycle shared by the generation-specific prefill metadata.

    Subclasses own their protocol's tensors and provide allocation, row fill,
    idle neutralization and binding.  The template methods here only run the
    steps that are identical for V4 and V4.1.
    """

    metadata: ForwardMetadata
    num_tokens: int
    max_reqs: int
    actual_q: torch.Tensor
    actual_q_pa: torch.Tensor
    actual_kv: torch.Tensor
    seqused: torch.Tensor
    start_pos: torch.Tensor
    kernel_metadata: Dict[str, torch.Tensor]
    topk_by_layer: Dict[int, torch.Tensor]

    def __init__(self) -> None:
        self.metadata = None
        self.num_tokens = 0
        self.max_reqs = 0
        self.topk_by_layer = {}

    @classmethod
    def allocate(
        cls,
        *,
        backend,
        seed: ForwardMetadata,
        num_tokens: int,
    ) -> _Dsv4PrefillGraphMetadataBase:
        """Allocate one capacity-sized metadata object for a capture bucket.

        The caller selects the concrete class from the model generation; this
        template only fixes the shared lifecycle and returns the container the
        runner stashes per shape key.
        """
        state = cls()
        state.metadata = seed
        state.num_tokens = num_tokens
        state.max_reqs = MAX_PREFILL_GRAPH_REQS
        state._allocate_sequence_scalars(backend)
        state._allocate_fields(backend=backend, seed=seed)
        state.fill(backend=backend, live=seed, forward_batch=None)
        state._bind(seed)
        return state

    def refresh(
        self,
        *,
        backend,
        live: ForwardMetadata,
        forward_batch: ForwardBatch,
    ) -> None:
        self.fill(backend=backend, live=live, forward_batch=forward_batch)

    def stage_topk(self, layer_id: int, topk_indices: torch.Tensor) -> torch.Tensor:
        """Keep the capture-time output object so replay writes into it.

        The indexer allocates its output inside the captured region, so the
        storage already belongs to the graph pool.  Per-source outputs cannot
        share the single ``forward_metadata.c*_topk_indices`` attribute: it
        only remembers the last layer assigned during capture.
        """
        self.topk_by_layer[layer_id] = topk_indices
        return topk_indices

    def fill(
        self,
        *,
        backend,
        live: ForwardMetadata,
        forward_batch: Optional[ForwardBatch],
    ) -> None:
        if forward_batch is not None:
            real_tokens = forward_batch.global_num_token_non_padded_cpu
            if forward_batch.forward_mode.is_idle() or real_tokens == 0:
                self._fill_idle(backend)
                return
        self._copy_sequence_scalars(live)
        self._fill_rows(backend=backend, live=live, forward_batch=forward_batch)

    def _allocate_sequence_scalars(self, backend) -> None:
        max_reqs = self.max_reqs
        int32 = {"dtype": torch.int32, "device": backend.device}
        self.actual_q = torch.zeros(max_reqs, **int32)
        self.actual_q_pa = torch.zeros(max_reqs + 1, **int32)
        self.actual_kv = torch.zeros(max_reqs, **int32)
        self.seqused = torch.zeros(max_reqs, **int32)
        self.start_pos = torch.zeros(max_reqs, **int32)

    def _copy_sequence_scalars(self, live: ForwardMetadata) -> None:
        _copy_cu(self.actual_q_pa, live.actual_seq_lengths_q_pa)
        _copy_cu(self.actual_q, live.actual_seq_lengths_q)
        _copy_1d(self.actual_kv, live.actual_seq_lengths_kv)
        _copy_1d(self.seqused, live.seqused)
        _copy_1d(self.start_pos, live.start_pos)

    def _copy_kernel_metadata(self, source, *, strict: bool = False) -> None:
        if strict and set(source) != set(self.kernel_metadata):
            raise RuntimeError(
                "DSV4 prefill graph kernel metadata keys changed: "
                f"captured={sorted(self.kernel_metadata)} "
                f"replay={sorted(source)}"
            )
        for key, dst in self.kernel_metadata.items():
            src = source.get(key)
            if src is None:
                if strict:
                    raise RuntimeError(
                        f"DSV4 prefill graph is missing kernel metadata {key}"
                    )
                dst.zero_()
                continue
            if src.numel() > dst.numel():
                raise RuntimeError(
                    f"DSV4 prefill graph {key} overflow: "
                    f"live={src.numel()} captured={dst.numel()}"
                )
            dst[: src.numel()].copy_(src.to(dtype=dst.dtype))
            if src.numel() < dst.numel():
                dst[src.numel() :].zero_()

    def _bind_common(self, metadata: ForwardMetadata) -> None:
        metadata.actual_seq_lengths_q = self.actual_q
        metadata.actual_seq_lengths_q_pa = self.actual_q_pa
        metadata.actual_seq_lengths_kv = self.actual_kv
        metadata.seqused = self.seqused
        metadata.start_pos = self.start_pos
        metadata.kernel_metadata = self.kernel_metadata
        metadata.dsv4_prefill_graph_state = self

    @abc.abstractmethod
    def _allocate_fields(self, *, backend, seed: ForwardMetadata) -> None:
        """Allocate this generation's tensors after the shared scalars."""

    @abc.abstractmethod
    def _fill_rows(
        self,
        *,
        backend,
        live: ForwardMetadata,
        forward_batch: Optional[ForwardBatch],
    ) -> None:
        """Refresh this generation's rows, tables and dispatch metadata."""

    @abc.abstractmethod
    def _fill_idle(self, backend) -> None:
        """Neutralize this generation's state for an idle replay."""

    @abc.abstractmethod
    def _bind(self, metadata: ForwardMetadata) -> None:
        """Bind this generation's fixed tensors on the capture metadata."""


class Dsv4PrefillGraphMetadata(_Dsv4PrefillGraphMetadataBase):
    """V4 (ratio 4 / 128) fixed metadata read by one captured prefill bucket."""

    ratios: list[int]
    positions: Dict[int, torch.Tensor]
    locs: Dict[int, torch.Tensor]
    state_tables: Dict[int, torch.Tensor]
    dummy_state: Dict[int, int]
    swa_page_table: torch.Tensor
    c4_page_table: Optional[torch.Tensor]
    c128_page_table: Optional[torch.Tensor]

    def __init__(self) -> None:
        super().__init__()
        self.ratios = []
        self.positions = {}
        self.locs = {}
        self.state_tables = {}
        self.dummy_state = {}

    def _allocate_fields(self, *, backend, seed: ForwardMetadata) -> None:
        self.ratios = [
            ratio for ratio in backend._dsv4_unique_compress_ratios if ratio in (4, 128)
        ]

        device = backend.device
        max_reqs = self.max_reqs
        int32 = {"dtype": torch.int32, "device": device}

        for ratio in self.ratios:
            capacity = compress_capacity(self.num_tokens, ratio, max_reqs)
            self.positions[ratio] = torch.zeros(
                capacity, dtype=torch.int64, device=device
            )
            # Capture writes compressed KV / state into reserved slots. Replay
            # fills the live prefix; the zero tail keeps its writes isolated.
            self.locs[ratio] = torch.zeros(capacity, dtype=torch.int64, device=device)

        page_size = backend.page_size
        context_len = backend.max_context_len
        swa_cols = max(1, (context_len + page_size - 1) // page_size)
        self.swa_page_table = torch.zeros((max_reqs, swa_cols), **int32)
        self.c4_page_table = None
        self.c128_page_table = None
        if backend._dsv4_has_c4:
            # The c4 builder steps the original token axis by page_size, so its
            # column count matches the SWA page table, not the block count.
            self.c4_page_table = torch.zeros((max_reqs, swa_cols), **int32)
        if backend._dsv4_has_c128:
            c128_page_size = backend.req_to_token_pool.c128_page_size
            groups = max(
                1, ((context_len // 128) + c128_page_size - 1) // c128_page_size
            )
            self.c128_page_table = torch.zeros((max_reqs, groups), **int32)

        metadata_ints = _MIN_KERNEL_METADATA_INTS
        for tensor in seed.kernel_metadata.values():
            metadata_ints = max(metadata_ints, tensor.numel())
        self.kernel_metadata = {
            key: torch.zeros(metadata_ints, **int32) for key in _KERNEL_METADATA_KEYS
        }

        for ratio in self.ratios:
            pool = backend._dsv4_state_pools_by_ratio[ratio]
            history = (2 if ratio == 4 else 1) * ratio
            self.dummy_state[ratio] = int(pool.dummy_state_loc)
            self.state_tables[ratio] = torch.full(
                (max_reqs, history + self.num_tokens),
                self.dummy_state[ratio],
                **int32,
            )

    def _fill_rows(
        self,
        *,
        backend,
        live: ForwardMetadata,
        forward_batch: Optional[ForwardBatch],
    ) -> None:
        capture = forward_batch is None
        for ratio in self.ratios:
            _copy_1d(
                self.positions[ratio],
                getattr(live, f"positions_cmp_padding_c{ratio}"),
            )
            _copy_1d(
                self.locs[ratio],
                None if capture else getattr(live, f"c{ratio}_loc"),
            )

        _copy_table(self.swa_page_table, live.swa_page_table, "swa_page_table")
        if self.c4_page_table is not None:
            _copy_table(self.c4_page_table, live.c4_page_table, "c4_page_table")
        if self.c128_page_table is not None:
            _copy_table(self.c128_page_table, live.c128_page_table, "c128_page_table")

        self._copy_kernel_metadata(live.kernel_metadata)
        if not capture:
            self._refresh_state_tables(backend, forward_batch)

    def _fill_idle(self, backend) -> None:
        # Keep captured kernels legal on an idle DP rank without touching
        # request state: one dummy query, zero valid tokens, page 0.
        self.actual_q.fill_(1)
        self.actual_q_pa.zero_()
        self.actual_q_pa[-1:].fill_(1)
        self.actual_kv.fill_(1)
        self.seqused.zero_()
        self.start_pos.zero_()
        for ratio in self.ratios:
            self.positions[ratio].zero_()
            self.locs[ratio].zero_()
        self.swa_page_table.zero_()
        if self.c4_page_table is not None:
            self.c4_page_table.zero_()
        if self.c128_page_table is not None:
            self.c128_page_table.zero_()
        for ratio, table in self.state_tables.items():
            table.fill_(self.dummy_state[ratio])
        rebuilt = backend._kernel_metadata_from_parts(
            bs=self.max_reqs,
            actual_seq_lengths_q_pa=self.actual_q_pa,
            actual_seq_lengths_kv=self.actual_kv,
        )
        self._copy_kernel_metadata(rebuilt)

    def _refresh_state_tables(self, backend, forward_batch: ForwardBatch) -> None:
        batch_size = forward_batch.batch_size
        if batch_size > self.max_reqs:
            raise RuntimeError(
                "DSV4 prefill graph state table got "
                f"{batch_size} requests, captured capacity is {self.max_reqs}"
            )
        for ratio, dst in self.state_tables.items():
            table = backend._build_dsv4_prefill_graph_state_table(
                ratio=ratio,
                req_pool_indices=forward_batch.req_pool_indices,
                start_pos=self.start_pos[:batch_size],
                cu_seqlens=self.actual_q_pa[: batch_size + 1],
                seqused=self.seqused[:batch_size],
                max_input_capacity=self.num_tokens,
            )
            dst.fill_(self.dummy_state[ratio])
            rows, cols = table.shape
            if cols > dst.shape[1]:
                raise RuntimeError(
                    "DSV4 prefill graph state table overflow: "
                    f"live={cols} captured={dst.shape[1]} ratio={ratio}"
                )
            dst[:rows, :cols].copy_(table.to(dtype=dst.dtype))

    def _bind(self, metadata: ForwardMetadata) -> None:
        self._bind_common(metadata)

        for ratio in self.ratios:
            setattr(metadata, f"positions_cmp_padding_c{ratio}", self.positions[ratio])
            setattr(metadata, f"c{ratio}_loc", self.locs[ratio])

        metadata.swa_page_table = self.swa_page_table
        if self.c4_page_table is not None:
            metadata.c4_page_table = self.c4_page_table
        if self.c128_page_table is not None:
            metadata.c128_page_table = self.c128_page_table

        # stage_topk binds the captured indexer output for every c4 layer.
        metadata.c4_topk_indices = None
        metadata.dsv4_explicit_state_block_tables = self.state_tables
        # The eager builder reads this as the largest per-request chunk; with
        # one request per graph it equals the bucket size.
        metadata.dsv4_max_input_capacity = self.num_tokens


class Dsv41PrefillGraphMetadata(_Dsv4PrefillGraphMetadataBase):
    """V4.1 (ratio 1 / 2) fixed metadata read by one captured prefill bucket."""

    ratios: list[int]
    positions: Dict[int, torch.Tensor]
    locs: Dict[int, torch.Tensor]
    cmp_topk_length: Dict[int, torch.Tensor]
    index_page_tables: Dict[int, torch.Tensor]
    live_mask: torch.Tensor
    positions_token: torch.Tensor
    out_loc_scratch: torch.Tensor
    swa_loc: torch.Tensor
    win_sparse_indices: torch.Tensor
    win_topk_length: torch.Tensor
    full_block_table: torch.Tensor
    swa_page_table: torch.Tensor
    swa_shared_with_full: bool
    cycle_state_block_table: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.ratios = []
        self.positions = {}
        self.locs = {}
        self.cmp_topk_length = {}
        self.index_page_tables = {}
        self.swa_shared_with_full = False

    def _allocate_fields(self, *, backend, seed: ForwardMetadata) -> None:
        self.ratios = list(backend._dsv41_low_ratios())
        if not self.ratios:
            raise RuntimeError(
                "DSV4.1 prefill graph requires at least one ratio-1/2 layer"
            )

        device = backend.device
        max_reqs = self.max_reqs
        num_tokens = self.num_tokens
        int32 = {"dtype": torch.int32, "device": device}
        int64 = {"dtype": torch.int64, "device": device}

        self.live_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
        self.positions_token = torch.zeros(num_tokens, **int64)
        self.out_loc_scratch = torch.zeros(num_tokens, **int64)
        self.swa_loc = torch.zeros(num_tokens, **int64)

        window = max(1, int(backend._dsv4_sliding_window_size))
        self.win_sparse_indices = torch.zeros((num_tokens, 1, window), **int32)
        self.win_topk_length = torch.zeros((num_tokens, 1), **int32)
        for ratio in self.ratios:
            capacity = compress_capacity(num_tokens, ratio, max_reqs)
            self.positions[ratio] = torch.zeros(capacity, **int64)
            self.locs[ratio] = torch.full((capacity,), _DSV41_SLOT_SKIP, **int32)
            self.cmp_topk_length[ratio] = torch.zeros((num_tokens, 1), **int32)

        page_size = int(backend.page_size)
        # The eager extend block table steps the whole request-pool row by
        # page_size, so its width comes from the pool row, not from the model
        # context alone (the pool carries extra rows).
        pool_row = int(backend.req_to_token_pool.row_width)
        page_cols = max(1, (pool_row + page_size - 1) // page_size)
        self.full_block_table = torch.zeros((max_reqs, page_cols), **int32)
        if seed.swa_page_table is seed.block_tables:
            self.swa_page_table = self.full_block_table
            self.swa_shared_with_full = True
        else:
            self.swa_page_table = torch.zeros((max_reqs, page_cols), **int32)

        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DSV41_INDEX_PAGE_SIZE,
        )

        for ratio in self.ratios:
            stride = (page_size // ratio) // DSV41_INDEX_PAGE_SIZE
            if stride < 1 or (page_size // ratio) % DSV41_INDEX_PAGE_SIZE:
                raise RuntimeError(
                    "DSV4.1 index page expansion needs page_size // ratio to be "
                    f"a positive multiple of {DSV41_INDEX_PAGE_SIZE}; got "
                    f"page_size={page_size} ratio={ratio}"
                )
            self.index_page_tables[ratio] = torch.zeros(
                (max_reqs, page_cols * stride), **int32
            )

        self.cycle_state_block_table = torch.zeros(max_reqs, **int32)

        keys = tuple(seed.kernel_metadata)
        if not keys:
            raise RuntimeError(
                "DSV4.1 prefill graph capture has no kernel metadata; the eager "
                "V4.1 metadata builder must run before allocation"
            )
        metadata_ints = _MIN_KERNEL_METADATA_INTS
        for tensor in seed.kernel_metadata.values():
            metadata_ints = max(metadata_ints, tensor.numel())
        self.kernel_metadata = {
            key: torch.zeros(metadata_ints, **int32) for key in keys
        }

    def _fill_rows(
        self,
        *,
        backend,
        live: ForwardMetadata,
        forward_batch: Optional[ForwardBatch],
    ) -> None:
        capture = forward_batch is None
        if capture:
            self.positions_token.zero_()
            self.out_loc_scratch.zero_()
        else:
            positions = forward_batch.positions.reshape(-1)
            if positions.numel() > self.num_tokens:
                raise RuntimeError(
                    "DSV4.1 prefill graph token overflow: live positions "
                    f"{positions.numel()} > captured bucket {self.num_tokens}"
                )
            n = min(int(positions.numel()), self.num_tokens)
            self.positions_token.zero_()
            if n:
                self.positions_token[:n].copy_(positions[:n].to(torch.int64))
            out_loc = forward_batch.out_cache_loc.reshape(-1)
            m = min(int(out_loc.numel()), self.num_tokens)
            self.out_loc_scratch.zero_()
            if m:
                self.out_loc_scratch[:m].copy_(out_loc[:m].to(torch.int64))

        # One live row per real token. Padding rows sit after cu[-1] and are
        # neutralized by every per-row length below.
        self.live_mask.copy_(
            torch.arange(self.num_tokens, device=self.live_mask.device)
            < self.actual_q_pa[-1].to(torch.int64)
        )

        self.swa_loc.copy_(
            backend.token_to_kv_pool.translate_loc_from_full_to_swa(
                self.out_loc_scratch
            ).to(torch.int64)
        )

        for ratio in self.ratios:
            backend._dsv41_fill_ratio_rows(
                ratio=ratio,
                positions=self.positions_token,
                out_loc=self.out_loc_scratch,
                live_mask=self.live_mask,
                pos_dst=self.positions[ratio],
                loc_dst=self.locs[ratio],
            )
        backend._dsv41_fill_window_buffers(
            positions=self.positions_token,
            live_mask=self.live_mask,
            indices=self.win_sparse_indices,
            lengths=self.win_topk_length,
        )
        for ratio in self.ratios:
            backend._dsv41_fill_cmp_lengths(
                positions=self.positions_token,
                live_mask=self.live_mask,
                dst=self.cmp_topk_length[ratio],
                ratio=ratio,
            )

        _copy_table(self.full_block_table, live.block_tables, "full_block_table")
        if not self.swa_shared_with_full:
            _copy_table(self.swa_page_table, live.swa_page_table, "swa_page_table")
        self._refresh_index_page_tables(backend)

        _copy_1d(
            self.cycle_state_block_table,
            live.dsv4_cycle_state_block_table[: self.max_reqs],
        )

        dispatch = backend._dsv41_attention_dispatch_metadata(
            cu_seqlens_q=self.actual_q_pa,
            win_lengths=self.win_topk_length,
            cmp_lengths_by_ratio={
                ratio: self.cmp_topk_length[ratio] for ratio in self.ratios
            },
        ) | backend._dsv41_index_dispatch_metadata(
            cu_seqlens_q=self.actual_q_pa,
            kv_lens=self.actual_kv,
            positions=self.positions_token,
            batch_size=self.max_reqs,
        )
        self._copy_kernel_metadata(dispatch, strict=True)

    def _refresh_index_page_tables(self, backend) -> None:
        from sglang.srt.layers.attention.dsv4.metadata import (
            expand_index_page_table,
        )
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DSV41_INDEX_PAGE_SIZE,
        )

        for ratio, dst in self.index_page_tables.items():
            expanded = expand_index_page_table(
                self.full_block_table,
                full_page_size=int(backend.page_size),
                compress_ratio=ratio,
                index_page_size=DSV41_INDEX_PAGE_SIZE,
            )
            _copy_table(dst, expanded, f"c{ratio}_index_page_table")

    def _fill_idle(self, backend) -> None:
        # Defensive only: the first delivery sends an all-idle DP group eager.
        # Keep every row legal without touching a real request bank or page.
        self.actual_q.fill_(1)
        self.actual_q_pa.zero_()
        self.actual_q_pa[-1:].fill_(1)
        self.actual_kv.fill_(1)
        self.seqused.zero_()
        self.start_pos.zero_()
        self.live_mask.zero_()
        self.positions_token.zero_()
        self.out_loc_scratch.zero_()
        self.swa_loc.zero_()
        for ratio in self.ratios:
            self.positions[ratio].zero_()
            self.locs[ratio].fill_(_DSV41_SLOT_SKIP)
            self.cmp_topk_length[ratio].zero_()
        self.win_sparse_indices.fill_(-1)
        self.win_sparse_indices[:, :, 0] = 0
        self.win_topk_length.zero_()
        self.full_block_table.zero_()
        if not self.swa_shared_with_full:
            self.swa_page_table.zero_()
        for dst in self.index_page_tables.values():
            dst.zero_()
        self.cycle_state_block_table.zero_()

        dispatch = backend._dsv41_attention_dispatch_metadata(
            cu_seqlens_q=self.actual_q_pa,
            win_lengths=self.win_topk_length,
            cmp_lengths_by_ratio={
                ratio: self.cmp_topk_length[ratio] for ratio in self.ratios
            },
        ) | backend._dsv41_index_dispatch_metadata(
            cu_seqlens_q=self.actual_q_pa,
            kv_lens=self.actual_kv,
            positions=self.positions_token,
            batch_size=self.max_reqs,
        )
        self._copy_kernel_metadata(dispatch, strict=True)

    def _bind(self, metadata: ForwardMetadata) -> None:
        self._bind_common(metadata)
        metadata.swa_loc = self.swa_loc

        metadata.block_tables = self.full_block_table
        metadata.swa_page_table = self.swa_page_table
        for ratio in self.ratios:
            setattr(metadata, f"positions_cmp_padding_c{ratio}", self.positions[ratio])
            setattr(metadata, f"c{ratio}_loc", self.locs[ratio])
            # Compressed page j is full page j: the full page table doubles as
            # the compressed page table.
            setattr(metadata, f"c{ratio}_page_table", self.full_block_table)
            setattr(
                metadata,
                f"c{ratio}_index_page_table",
                self.index_page_tables[ratio],
            )
            setattr(
                metadata,
                f"dsv41_cmp_topk_length_c{ratio}",
                self.cmp_topk_length[ratio],
            )

        metadata.dsv41_win_sparse_indices = self.win_sparse_indices
        metadata.dsv41_win_topk_length = self.win_topk_length
        metadata.dsv41_live_mask_buf = self.live_mask
        metadata.dsv4_cycle_state_block_table = self.cycle_state_block_table
        metadata.dsv4_explicit_state_block_tables = {}
        metadata.dsv4_max_input_capacity = self.num_tokens
        # The per-layer source outputs are graph-pool tensors bound through
        # topk_by_layer; the shared attributes must not masquerade as current.
        metadata.c1_topk_indices = None
        metadata.c2_topk_indices = None
        metadata.dsv41_candidate_blocks = None
