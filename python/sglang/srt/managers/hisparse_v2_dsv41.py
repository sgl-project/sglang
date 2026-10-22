"""HiSparse V2 over DeepSeek-V4.1's compressed KV and the HiCache host tier.

The ratio-1/2 KV of an admitted request's tree prefix may be demoted to the HiCache
host pools. Its indexer keys stay on GPU: the ratio-1/2 index pools carry a second
region addressed by host page, filled from the device page on demotion, so the
indexer keeps scoring the full history. Every index-source layer then swaps the
selected demoted rows into hot slots the request owns; resident rows are read in
place.

The tree holds an admitted request's prompt only up to one sliding window short of
its end, so the SWA window decode reads stays request-owned and on GPU.
"""

from __future__ import annotations

import logging
import os
from typing import NamedTuple

import torch

from sglang.kernels.ops.kvcache.hisparse import load_cache_to_device_buffer_v41
from sglang.srt.managers.hisparse_v2_coordinator import HiSparseV2Coordinator
from sglang.srt.mem_cache.hicache_storage import PoolName

logger = logging.getLogger(__name__)

_HOST_POOL_NAMES = {1: PoolName.DEEPSEEK_V4_C1, 2: PoolName.DEEPSEEK_V4_C2}
# Debug: the swap-in kernel re-reads every demoted selection against its host row.
_VERIFY = os.environ.get("SGLANG_HISPARSE_V2_VERIFY", "0") == "1"


class _SwapLayer(NamedTuple):
    layer_id: int
    ratio: int
    pool_layer: int
    # Ratio-1 index layers share the single ratio-1 pool layer, so each takes its
    # own run of the request's temporary slots.
    slot_run: int
    device_buffer: torch.Tensor
    slots_per_page: int
    data_bytes: int
    scale_bytes: int


class HiSparseV2DSV41Cache:
    @torch.inference_mode(False)
    def __init__(
        self,
        *,
        req_to_token_pool,
        token_to_kv_pool,
        index_source_layer_ids,
        top_k: int,
        buffer_tokens: int,
        block_size: int,
        device: str,
    ):
        pool = token_to_kv_pool
        self.pool = pool
        self.req_to_token = req_to_token_pool.req_to_token
        self.page_size = pool.page_size
        self.top_k = top_k
        self.buffer_tokens = buffer_tokens
        self.block_size = block_size
        self.device = device
        if pool.hisparse_v2_host_tokens <= 0:
            raise ValueError("HiSparse V2 needs the expanded ratio-1/2 index pools")
        if buffer_tokens < top_k or buffer_tokens > 32767:
            raise ValueError("HiSparse V2 device_buffer_size must be in [top_k, 32767]")
        if block_size % 64:
            raise ValueError("HiSparse V2 swap_in_block_size must be a multiple of 64")
        self.device_tokens = pool.full_size
        self.base_full_pages = pool.hisparse_v2_base_full_pages
        self.index_full_pages = (
            self.base_full_pages + pool.hisparse_v2_host_tokens // self.page_size
        )

        layers: list[_SwapLayer] = []
        num_ratio1 = 0
        for layer_id in index_source_layer_ids:
            if not pool._stage_start <= layer_id < pool._stage_end:
                continue
            ratio = pool.compression_ratios[layer_id]
            if ratio not in (1, 2):
                raise ValueError(
                    f"HiSparse V2 index layer {layer_id} has ratio {ratio}"
                )
            kv_pool = pool.kv_pools[ratio]
            layout = kv_pool.kv_layout
            if layout.page_align != layout.data_bytes:
                raise ValueError(f"HiSparse V2 does not support the {layout} layout")
            pool_layer = pool.layer_mapping[layer_id].compress_layer_id
            layers.append(
                _SwapLayer(
                    layer_id=layer_id,
                    ratio=ratio,
                    pool_layer=pool_layer,
                    slot_run=num_ratio1 if ratio == 1 else 0,
                    device_buffer=kv_pool.kv_buffer[pool_layer],
                    slots_per_page=kv_pool.page_size,
                    data_bytes=layout.data_bytes,
                    scale_bytes=layout.scale_bytes,
                )
            )
            num_ratio1 += ratio == 1
        if not layers:
            raise ValueError("HiSparse V2 found no ratio-1/2 index-source layer")
        self.swap_layers = tuple(layers)
        self.swap_index = {spec.layer_id: k for k, spec in enumerate(layers)}
        has_ratio2 = any(spec.ratio == 2 for spec in layers)
        needed = max(num_ratio1 * buffer_tokens, 2 * buffer_tokens if has_ratio2 else 0)
        self.reserved_buffer_tokens = -(-needed // self.page_size) * self.page_size

        rows, context = self.req_to_token.shape
        num_layers = len(layers)
        self.host_locs = torch.full(
            (rows, context), -1, dtype=torch.int32, device=device
        )
        self.protected_lens = torch.zeros(rows, dtype=torch.int64, device=device)
        self.hot_slots = torch.full(
            (num_layers, rows, buffer_tokens), -1, dtype=torch.int32, device=device
        )
        self.residency = torch.full(
            (num_layers, rows, buffer_tokens), -1, dtype=torch.int32, device=device
        )
        self._lru_init = torch.arange(buffer_tokens, dtype=torch.int16, device=device)
        self.lru = self._lru_init.expand(num_layers, rows, -1).clone()
        self.page_cols = (
            torch.arange(
                -(-context // self.page_size), dtype=torch.int64, device=device
            )
            * self.page_size
        )
        self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=device)
        # [positions neither tier holds, host rows swapped in, copies skipped for a
        # hot slot outside the pool]
        self.stats = torch.zeros(3, dtype=torch.int64, device=device)
        # HiCache attaches after decode graphs are captured.
        self.host_bindings = torch.zeros(
            (num_layers, 2), dtype=torch.int64, device=device
        )
        self.host_pool = None

    def attach_host_pools(self, host_pool_group) -> None:
        from sgl_kernel.kvcacheio import get_device_accessible_ptr

        self.host_pool = host_pool_group.get_pool(PoolName.KV)
        host_pages = self.host_pool.size // self.page_size
        if self.base_full_pages + host_pages > self.index_full_pages:
            raise ValueError(
                "HiSparse V2 index pools do not cover the HiCache host pool: "
                f"{self.base_full_pages} + {host_pages} > {self.index_full_pages} pages"
            )
        device_index = torch.device(self.device).index
        if device_index is None:
            device_index = torch.cuda.current_device()
        bindings = []
        for spec in self.swap_layers:
            host = host_pool_group.get_pool(_HOST_POOL_NAMES[spec.ratio])
            page_bytes = (
                spec.device_buffer.stride(0) * spec.device_buffer.element_size()
            )
            if host.layout != "page_first" or host.kv_buffer.dim() != 3:
                raise ValueError("HiSparse V2 needs the page_first HiCache layout")
            if host.item_bytes != page_bytes or host.num_host_pages != host_pages:
                raise ValueError(
                    f"HiSparse V2 host pool {host.pool_name} does not mirror its "
                    "device pages"
                )
            buffer = host.kv_buffer.view(-1)
            for probe in (buffer[:1], buffer[-1:]):
                if get_device_accessible_ptr(probe, device_index) != probe.data_ptr():
                    raise ValueError(
                        "HiSparse V2 reads host rows at their host address, but "
                        f"{host.pool_name} maps to a different device address"
                    )
            bindings.append(
                (
                    buffer.data_ptr() + spec.pool_layer * host.item_bytes,
                    host.layer_num * host.item_bytes,
                )
            )
        self.host_bindings.copy_(
            torch.tensor(bindings, dtype=torch.int64, device=self.device)
        )

    def prepare_forward(self, forward_batch) -> None:
        if torch.cuda.is_current_stream_capturing():
            return
        self.num_real_reqs.fill_(
            0 if forward_batch.forward_mode.is_idle() else forward_batch.batch_size
        )

    def assign_temp_locations(self, req_id: int, temp: torch.Tensor) -> None:
        temp = temp.to(device=self.device, dtype=torch.int64)
        ratio2_slots = temp[0::2] // 2
        for k, spec in enumerate(self.swap_layers):
            if spec.ratio == 1:
                start = spec.slot_run * self.buffer_tokens
                slots = temp[start : start + self.buffer_tokens]
            else:
                slots = ratio2_slots[: self.buffer_tokens]
            self.hot_slots[k, req_id] = slots.to(torch.int32)

    def preserve_index(self, device_tokens: torch.Tensor, host_tokens: torch.Tensor):
        source = device_tokens[:: self.page_size] // self.page_size
        destination = (
            self.base_full_pages + host_tokens[:: self.page_size] // self.page_size
        )
        # Host locks keep these rows alive for every request sharing the prefix.
        for ratio, index_pool in self.pool.index_pools.items():
            if ratio not in (1, 2):
                continue
            per_page = self.page_size // ratio // index_pool.page_size
            offsets = torch.arange(per_page, device=source.device)
            src = (source[:, None] * per_page + offsets).flatten()
            dst = (destination[:, None] * per_page + offsets).flatten()
            for buffer in index_pool.contiguous_page_row_buffers():
                buffer[dst] = buffer[src]

    def release(self, req_id: int) -> None:
        self.protected_lens[req_id] = 0
        # Demotion left -1 in the row; a later request in this slot reads past its
        # length through page tables, which must stay inside the pools.
        self.req_to_token[req_id].clamp_(min=0)
        self.host_locs[req_id].fill_(-1)
        self.hot_slots[:, req_id].fill_(-1)
        self.residency[:, req_id].fill_(-1)
        self.lru[:, req_id].copy_(self._lru_init)

    def patch_page_table(
        self, page_table: torch.Tensor, req_pool_indices: torch.Tensor
    ) -> None:
        """Point the demoted pages of a full-page table at the host region of the
        index pools; graph-recordable."""
        cols = self.page_cols[: page_table.shape[1]]
        hosts = self.host_locs[req_pool_indices.long()[:, None], cols[None, :]]
        host_pages = hosts // self.page_size + self.base_full_pages
        # Columns past a request's length may still read a demoted -1.
        page_table.copy_(torch.where(hosts >= 0, host_pages, page_table.clamp(min=0)))

    def swap_in(
        self,
        *,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        raw_indices: torch.Tensor,
        page_indices: torch.Tensor,
    ) -> None:
        k = self.swap_index[layer_id]
        spec = self.swap_layers[k]
        load_cache_to_device_buffer_v41(
            top_k_tokens=raw_indices,
            compressed_seq_lens=compressed_seq_lens,
            req_pool_indices=req_pool_indices,
            req_to_token=self.req_to_token,
            host_locs=self.host_locs,
            hot_slots=self.hot_slots[k],
            residency=self.residency[k],
            lru_slots=self.lru[k],
            host_binding=self.host_bindings[k],
            device_buffer=spec.device_buffer,
            top_k_device_locs=page_indices,
            num_real_reqs=self.num_real_reqs,
            stats=self.stats,
            num_top_k=self.top_k,
            compress_ratio=spec.ratio,
            data_bytes=spec.data_bytes,
            scale_bytes=spec.scale_bytes,
            page_size=spec.slots_per_page,
            block_size=self.block_size,
            verify=_VERIFY,
        )


class _FullSlotAllocator:
    """Hot slots need ratio-1/2 rows only, so they take no SWA peers."""

    def __init__(self, allocator):
        self.allocator = allocator

    def alloc(self, need_size: int):
        return self.allocator.full_attn_allocator.alloc(need_size)

    def free(self, indices: torch.Tensor) -> None:
        self.allocator.free_full(indices)


class HiSparseV2DSV41Coordinator(HiSparseV2Coordinator):
    def __init__(self, *, cache: HiSparseV2DSV41Cache, allocator, sliding_window: int):
        super().__init__(cache=cache, allocator=_FullSlotAllocator(allocator))
        self.sliding_window = sliding_window
        self._finish_count = 0
        self._demote_events = 0
        self._demoted_tokens = 0
        # Per reserved request: device tokens its own tail and decode output need.
        self._growth: dict[str, int] = {}
        # Per reservation still in prefill: tokens its locked tree path holds on device.
        self._prefilling: dict[str, int] = {}

    def attach_tree(self, tree_cache, *, draft_caches=()) -> None:
        from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore

        if not isinstance(tree_cache.tree_core, UnifiedTreeCore):
            raise ValueError("HiSparse V2 requires the Python unified tree core")
        if any(cache is not None for cache in draft_caches):
            raise ValueError(
                "HiSparse V2 on DeepSeek-V4.1 does not support speculative decoding"
            )
        self.tree_cache = tree_cache
        self.cache.attach_host_pools(tree_cache.host_pool_group)
        tree_cache.hisparse_v2_coordinator = self
        tree_cache.tree_core.hisparse_v2_coordinator = self
        logger.info(
            "HiSparse V2 (DeepSeek-V4.1): %d swap layers, %d hot rows each, "
            "%d temporary tokens per request, %d device + %d host tokens",
            len(self.cache.swap_layers),
            self.cache.buffer_tokens,
            self.cache.reserved_buffer_tokens,
            self.cache.device_tokens,
            self.cache.host_pool.size,
        )

    def checkpoint_up_to(self, req, up_to: int) -> int:
        # The SWA rows of the window decode reads are found through req_to_token, so
        # that window must stay request-owned rather than demotable tree pages.
        return max(req.kv.cache_protected_len, up_to - self.sliding_window)

    def temp_tokens_required(self, req, *, max_new_tokens: int) -> int:
        if req.rid in self.pending:
            return 0
        prefix_len = len(req.full_untruncated_fill_ids)
        if prefix_len < self.cache.reserved_buffer_tokens or (
            prefix_len
            + self.cache.reserved_buffer_tokens
            + max_new_tokens
            + self.cache.page_size
            >= self.cache.device_tokens
        ):
            return 0
        return self.cache.reserved_buffer_tokens

    def reserve_prefill(self, req, *, max_new_tokens: int) -> bool:
        needed = self.temp_tokens_required(req, max_new_tokens=max_new_tokens)
        if not needed:
            return True
        page = self.cache.page_size
        fill_len = len(req.full_untruncated_fill_ids)
        prefix_len = fill_len // page * page
        # The tail past the tree prefix and every output token stay on device.
        growth = -(-(fill_len + max_new_tokens) // page) * page - (
            max(0, fill_len - self.sliding_window) // page * page
        )
        temps = self.reserved_temp_tokens() + needed
        growths = sum(self._growth.values()) + growth
        device = int(self.cache.device_tokens * 0.95)
        # A request in chunked prefill locks its whole tree path on device until it
        # is admitted, so only admitted prefixes can make room by demotion.
        if temps + growths + sum(self._prefilling.values()) + fill_len > device:
            return False
        claimed = {
            node.id: len(node.key)
            for state in self.requests.values()
            for node in state.nodes
        }
        prefixes = (
            sum(claimed.values())
            + sum(p.prefix_len for p in self.pending.values())
            + prefix_len
        )
        if prefixes + temps + growths > self.cache.host_pool.size + device:
            return False
        from sglang.srt.managers.hisparse_v2_coordinator import _PrefillReservation
        from sglang.srt.mem_cache.common import evict_from_tree_cache

        evict_from_tree_cache(self.tree_cache, needed)
        temp = self.allocator.alloc(needed)
        if temp is None:
            return False
        self.pending[req.rid] = _PrefillReservation(temp, prefix_len)
        self._growth[req.rid] = growth
        self._prefilling[req.rid] = fill_len
        return True

    def admit_request(self, req) -> bool:
        self._prefilling.pop(req.rid, None)
        admitted = super().admit_request(req)
        if not admitted:
            self._growth.pop(req.rid, None)
        return admitted

    def on_node_evicted(self, node) -> None:
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

        if not self.requests:
            return
        component = node.component_data[ComponentType.FULL]
        if component.value is None:
            return
        affected = [
            (req_id, state)
            for req_id, state in self.requests.items()
            if any(prefix_node.id == node.id for prefix_node in state.nodes)
        ]
        if not affected:
            return
        if component.host_value is None:
            raise RuntimeError(
                "HiSparse V2 cannot discard an active prefix without backup"
            )
        self._wait_for_forward()
        device_rows, order = component.value.to(self.cache.device).long().sort()
        host_rows = component.host_value.to(self.cache.device).long()[order]
        for _, state in affected:
            if node.id not in state.host_locks:
                state.host_locks[node.id] = self.tree_cache.inc_host_lock_ref(
                    node.id
                ).to_dec_params()
        self.cache.preserve_index(device_rows, host_rows)
        for req_id, state in affected:
            row = self.cache.req_to_token[req_id, : state.prefix_len]
            lookup = torch.searchsorted(device_rows, row.long()).clamp(
                max=device_rows.numel() - 1
            )
            positions = ((row >= 0) & (device_rows[lookup] == row)).nonzero().flatten()
            if not positions.numel():
                continue
            # Publish the host rows before the allocator can hand the device rows out.
            self.cache.host_locs[req_id, positions] = host_rows[lookup[positions]].to(
                torch.int32
            )
            row[positions] = -1
            self._demoted_tokens += positions.numel()
        self._demote_events += 1
        if self._demote_events == 1 or self._demote_events % 256 == 0:
            logger.info(
                "HiSparse V2: demotion #%d moved node %s (%d tokens) to host for "
                "%d active requests; %d request tokens demoted so far",
                self._demote_events,
                node.id,
                device_rows.numel(),
                len(affected),
                self._demoted_tokens,
            )

    def request_finished(self, req) -> None:
        self._growth.pop(req.rid, None)
        self._prefilling.pop(req.rid, None)
        was_active = req.kv.req_pool_idx in self.requests
        super().request_finished(req)
        if not was_active:
            return
        self._finish_count += 1
        if self._finish_count % 64 == 0:
            missing, swapped, skipped = self.cache.stats.tolist()
            if missing:
                logger.error(
                    "HiSparse V2 selected %d rows held by neither tier", missing
                )
            if skipped:
                logger.error(
                    "HiSparse V2 skipped %d swap-ins with a hot slot outside the pool",
                    skipped,
                )
            logger.debug(
                "HiSparse V2: %d admitted requests finished, %d host rows swapped in",
                self._finish_count,
                swapped,
            )


def build_hisparse_v2_dsv41(*, model_runner):
    """The DeepSeek-V4.1 cache, and its coordinator on the target worker."""
    from sglang.srt.mem_cache.sparsity.hisparse_v2_config import (
        parse_hisparse_v2_config,
    )

    hf_config = model_runner.model_config.hf_text_config
    config = parse_hisparse_v2_config(model_top_k=hf_config.index_topk)
    cache = HiSparseV2DSV41Cache(
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool=model_runner.token_to_kv_pool,
        index_source_layer_ids=hf_config.index_source_layer_ids,
        top_k=config.top_k,
        buffer_tokens=config.buffer_tokens,
        block_size=config.block_size,
        device=model_runner.device,
    )
    coordinator = HiSparseV2DSV41Coordinator(
        cache=cache,
        allocator=model_runner.token_to_kv_pool_allocator,
        sliding_window=model_runner.token_to_kv_pool.sliding_window,
    )
    return cache, coordinator
