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

Only one request per replay is admitted for now (``MAX_PREFILL_GRAPH_REQS``).
Multi-request replay needs the fused indexer and sparse-attention metadata
operators to accept zero-length request slots; that is not verified yet, so
the runner's admission sends those batches to the eager path.
"""

from __future__ import annotations

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


class Dsv4PrefillGraphMetadata:
    """Stable, capacity-sized metadata read by one captured prefill bucket."""

    metadata: "ForwardMetadata"

    def __init__(self) -> None:
        self.metadata = None
        self.num_tokens = 0
        self.max_reqs = 0
        self.ratios: list[int] = []
        self.positions: Dict[int, torch.Tensor] = {}
        self.locs: Dict[int, torch.Tensor] = {}
        self.state_tables: Dict[int, torch.Tensor] = {}
        self.dummy_state: Dict[int, int] = {}
        self.topk_by_layer: Dict[int, torch.Tensor] = {}

    @classmethod
    def allocate(
        cls,
        *,
        backend,
        seed: "ForwardMetadata",
        num_tokens: int,
    ) -> "Dsv4PrefillGraphMetadata":
        state = cls()
        state.metadata = seed
        state.num_tokens = num_tokens
        state.max_reqs = MAX_PREFILL_GRAPH_REQS
        state.ratios = [
            ratio for ratio in backend._dsv4_unique_compress_ratios if ratio in (4, 128)
        ]

        device = backend.device
        max_reqs = state.max_reqs
        int32 = {"dtype": torch.int32, "device": device}

        state.actual_q = torch.zeros(max_reqs, **int32)
        state.actual_q_pa = torch.zeros(max_reqs + 1, **int32)
        state.actual_kv = torch.zeros(max_reqs, **int32)
        state.seqused = torch.zeros(max_reqs, **int32)
        state.start_pos = torch.zeros(max_reqs, **int32)

        for ratio in state.ratios:
            capacity = compress_capacity(num_tokens, ratio, max_reqs)
            state.positions[ratio] = torch.zeros(
                capacity, dtype=torch.int64, device=device
            )
            # Capture writes compressed KV / state into reserved slots. Replay
            # fills the live prefix; the zero tail keeps its writes isolated.
            state.locs[ratio] = torch.zeros(capacity, dtype=torch.int64, device=device)

        page_size = backend.page_size
        context_len = backend.max_context_len
        swa_cols = max(1, (context_len + page_size - 1) // page_size)
        state.swa_page_table = torch.zeros((max_reqs, swa_cols), **int32)
        state.c4_page_table = None
        state.c128_page_table = None
        if backend._dsv4_has_c4:
            # The c4 builder steps the original token axis by page_size, so its
            # column count matches the SWA page table, not the block count.
            state.c4_page_table = torch.zeros((max_reqs, swa_cols), **int32)
        if backend._dsv4_has_c128:
            c128_page_size = backend.req_to_token_pool.c128_page_size
            groups = max(
                1, ((context_len // 128) + c128_page_size - 1) // c128_page_size
            )
            state.c128_page_table = torch.zeros((max_reqs, groups), **int32)

        metadata_ints = _MIN_KERNEL_METADATA_INTS
        for tensor in seed.kernel_metadata.values():
            metadata_ints = max(metadata_ints, tensor.numel())
        state.kernel_metadata = {
            key: torch.zeros(metadata_ints, **int32) for key in _KERNEL_METADATA_KEYS
        }

        for ratio in state.ratios:
            pool = backend._dsv4_state_pools_by_ratio[ratio]
            history = (2 if ratio == 4 else 1) * ratio
            state.dummy_state[ratio] = int(pool.dummy_state_loc)
            state.state_tables[ratio] = torch.full(
                (max_reqs, history + num_tokens),
                state.dummy_state[ratio],
                **int32,
            )

        # forward_batch is None during capture; the capture values come from
        # the eager metadata alone and must not touch live request slots.
        state.fill(backend=backend, live=seed, forward_batch=None)
        state._bind(seed)
        return state

    def refresh(
        self,
        *,
        backend,
        live: "ForwardMetadata",
        forward_batch: "ForwardBatch",
    ) -> None:
        self.fill(backend=backend, live=live, forward_batch=forward_batch)

    def stage_topk(self, layer_id: int, topk_indices: torch.Tensor) -> torch.Tensor:
        """Keep the capture-time output object so replay writes into it.

        The indexer allocates its output inside the captured region, so the
        storage already belongs to the graph pool.  The eager attention break
        runs at replay and has to read *this layer's* buffer; the shared
        ``forward_metadata.c4_topk_indices`` attribute only remembers the last
        layer assigned during capture.
        """
        self.topk_by_layer[layer_id] = topk_indices
        return topk_indices

    def fill(
        self,
        *,
        backend,
        live: "ForwardMetadata",
        forward_batch: Optional["ForwardBatch"],
    ) -> None:
        capture = forward_batch is None
        real_tokens = None
        if forward_batch is not None:
            real_tokens = forward_batch.global_num_token_non_padded_cpu
        idle = forward_batch is not None and (
            forward_batch.forward_mode.is_idle() or real_tokens == 0
        )
        if idle:
            self._fill_idle(backend)
            return

        _copy_cu(self.actual_q_pa, live.actual_seq_lengths_q_pa)
        _copy_cu(self.actual_q, live.actual_seq_lengths_q)
        _copy_1d(self.actual_kv, live.actual_seq_lengths_kv)
        _copy_1d(self.seqused, live.seqused)
        _copy_1d(self.start_pos, live.start_pos)

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

    def _refresh_state_tables(self, backend, forward_batch: "ForwardBatch") -> None:
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

    def _copy_kernel_metadata(self, source) -> None:
        for key, dst in self.kernel_metadata.items():
            src = source.get(key)
            if src is None:
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

    def _bind(self, metadata: "ForwardMetadata") -> None:
        metadata.actual_seq_lengths_q = self.actual_q
        metadata.actual_seq_lengths_q_pa = self.actual_q_pa
        metadata.actual_seq_lengths_kv = self.actual_kv
        metadata.seqused = self.seqused
        metadata.start_pos = self.start_pos

        for ratio in self.ratios:
            setattr(metadata, f"positions_cmp_padding_c{ratio}", self.positions[ratio])
            setattr(metadata, f"c{ratio}_loc", self.locs[ratio])

        metadata.swa_page_table = self.swa_page_table
        if self.c4_page_table is not None:
            metadata.c4_page_table = self.c4_page_table
        if self.c128_page_table is not None:
            metadata.c128_page_table = self.c128_page_table

        metadata.kernel_metadata = self.kernel_metadata
        # stage_topk binds the captured indexer output for every c4 layer.
        metadata.c4_topk_indices = None
        metadata.dsv4_explicit_state_block_tables = self.state_tables
        # The eager builder reads this as the largest per-request chunk; with
        # one request per graph it equals the bucket size.
        metadata.dsv4_max_input_capacity = self.num_tokens
        metadata.dsv4_prefill_graph_state = self
