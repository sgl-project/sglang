"""Eager DFlash cache with independent, bounded request slots.

Full-attention pages belong to a request for its lifetime. Window layers use
request-private page rings. This initial storage adapter deliberately reserves
full capacity per slot; it does not pretend that inactive slots are free HBM.
The target allocator never hands out or frees any of these physical addresses.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class DraftSlotLayout:
    slots: int
    context_length: int
    page_size: int
    block_size: int
    window: int | None = None

    def __post_init__(self):
        if min(self.slots, self.context_length, self.page_size, self.block_size) <= 0:
            raise ValueError("Draft cache dimensions must be positive")
        if self.window is not None and self.window <= 0:
            raise ValueError("Draft window must be positive")

    @property
    def stride(self):
        return (
            ceil((self.context_length + 2 * self.block_size) / self.page_size)
            * self.page_size
        )

    @property
    def window_stride(self):
        if self.window is None:
            return self.stride
        return (
            ceil((self.window - 1 + 2 * self.block_size) / self.page_size) + 1
        ) * self.page_size

    @property
    def full_capacity(self):
        return self.slots * self.stride

    @property
    def window_capacity(self):
        return self.slots * self.window_stride

    def full_loc(self, slot: int, position: int):
        if not 1 <= slot <= self.slots or not 0 <= position < self.stride:
            raise ValueError("Draft slot/position outside its reservation")
        return self.page_size + (slot - 1) * self.stride + position

    def window_loc(self, slot: int, position: int):
        self.full_loc(slot, position)
        return (
            self.page_size
            + (slot - 1) * self.window_stride
            + position % self.window_stride
        )

    def reservation_bytes(self, *, row_bytes, full_layers, window_layers):
        """Fixed storage, including inactive slots and sentinel pages."""
        kv = (self.full_capacity + self.page_size) * full_layers * row_bytes
        if window_layers:
            kv += (self.window_capacity + self.page_size) * window_layers * row_bytes
        tables = (self.slots + 1) * self.stride * 4
        mapping = (self.full_capacity + self.page_size + 1) * 8 if window_layers else 0
        pages = (self.full_capacity + self.window_capacity) // self.page_size
        # Free/release/sort page-id arrays, plus construction's transient map.
        return kv + tables + 2 * mapping + pages * 24


def independent_draft_reservation_bytes(kvc):
    from sglang.srt.mem_cache.draft_kv_budget import resolve_dflash_draft_cell_size
    from sglang.srt.runtime_context import (
        get_schedule,
        get_spec,
    )

    spec, schedule = get_spec(), get_schedule()
    requested = schedule.max_running_requests
    if requested is None:
        raise ValueError(
            "Independent draft storage requires an explicit --max-running-requests"
        )
    owners = 1
    requests = max(1, requested // kvc.attn_dp_size)
    config = kvc.spec_aux_config.draft_model_config
    mixed = spec.speculative_draft_kv_layout == "hybrid"
    types = getattr(config.hf_config, "layer_types", None)
    if types is None:
        if mixed:
            raise ValueError(
                "Hybrid draft storage requires explicit native layer types"
            )
        types = ["full_attention"] * int(config.hf_config.num_hidden_layers)
    types = list(types)
    if mixed and set(types) - {"full_attention", "sliding_attention"}:
        raise ValueError("Unrecognized native draft layer type")
    full = types.count("full_attention") if mixed else len(types)
    windows = len(types) - full
    layout = DraftSlotLayout(
        slots=requests * owners,
        context_length=kvc.model_config.context_len,
        page_size=kvc.page_size,
        block_size=spec.speculative_num_draft_tokens,
        window=int(config.hf_config.sliding_window) if mixed else None,
    )
    row = resolve_dflash_draft_cell_size(
        draft_model_config=config,
        draft_num_layers=1,
        attn_tp_size=kvc.draft_attn_tp_size,
    )
    if row is None:
        raise ValueError("Cannot price independent draft KV geometry")
    return layout.reservation_bytes(
        row_bytes=row, full_layers=full, window_layers=windows
    )


class IndependentDraftCache:
    """Owns slot generations and maps logical positions to private storage.

    Uses ReqToTokenPool's aux-cache lifecycle so finish, abort and retraction
    invalidate coverage before the target slot can be reused. Graphs, overlap,
    prefix caches are rejected; PD requires the explicit typed transfer lane.
    """

    def __init__(
        self,
        *,
        layout,
        target_requests,
        requests,
        allocator,
        device,
        local_slot_offset=0,
    ):
        import torch

        self.layout = layout
        self.target_requests = target_requests
        self.requests = requests
        self.allocator = allocator
        self.device = device
        self.local_slot_offset = local_slot_offset
        self.dense_capacity = layout.full_capacity
        self.allocated_lens = [0] * (layout.slots + 1)
        self.generations = [-1] * (layout.slots + 1)
        table = requests.req_to_token
        # Bound construction temporaries even with a 1M declared context.
        for slot in range(1, layout.slots + 1):
            for start in range(0, layout.stride, 32768):
                stop = min(start + 32768, layout.stride)
                positions = torch.arange(start, stop, device=device, dtype=torch.int64)
                full = layout.page_size + (slot - 1) * layout.stride + positions
                table[slot, start:stop] = full.to(torch.int32)
                if layout.window is not None:
                    window = (
                        layout.page_size
                        + (slot - 1) * layout.window_stride
                        + positions.remainder(layout.window_stride)
                    )
                    allocator.full_to_swa_index_mapping[full] = window

    def alloc_to_lengths(self, *, req_pool_indices_cpu, target_seq_lens_cpu):
        for slot, length in zip(
            req_pool_indices_cpu.tolist(), target_seq_lens_cpu.tolist()
        ):
            target_slot = slot
            slot += self.local_slot_offset
            if (
                not 1 <= slot <= self.layout.slots
                or not 0 <= length <= self.layout.stride
            ):
                raise RuntimeError("Independent draft cache capacity exceeded")
            generation = int(self.target_requests.req_generation[target_slot])
            if self.generations[slot] != generation:
                if self.allocated_lens[slot]:
                    raise RuntimeError("Draft slot reused without lifecycle release")
                self.generations[slot] = generation
            self.allocated_lens[slot] = max(self.allocated_lens[slot], length)

    def free(self, slot):
        callback = getattr(self, "release_callback", None)
        if callback is not None:
            callback(slot)
        slot += self.local_slot_offset
        self.allocated_lens[slot] = 0
        self.generations[slot] = -1

    def clear(self):
        self.allocated_lens[:] = [0] * len(self.allocated_lens)
        self.generations[:] = [-1] * len(self.generations)

    def reset_allocator(self):
        # Fixed slot addresses do not participate in the target free lists.
        self.clear()

    def locations(self, req_pool_indices, positions, lengths):
        import torch

        owners = torch.repeat_interleave(
            req_pool_indices.to(torch.int64), lengths.to(torch.int64)
        )
        full = self.layout.page_size + (owners - 1) * self.layout.stride + positions
        window = None
        keep = None
        if self.layout.window is not None:
            window = (
                self.layout.page_size
                + (owners - 1) * self.layout.window_stride
                + positions.remainder(self.layout.window_stride)
            )
            # A prefill chunk may be much longer than the ring. Never scatter
            # multiple old tokens concurrently to the same live ring row.
            ends = torch.empty_like(lengths, dtype=torch.int64)
            offset = 0
            for i, length in enumerate(lengths.cpu().tolist()):
                ends[i] = positions[offset + length - 1] + 1 if length else 0
                offset += length
            keep = (
                positions
                >= torch.repeat_interleave(ends, lengths.to(torch.int64))
                - self.layout.window_stride
            )
        return full, window, keep


def allocate_independent_draft_cache(worker, target_requests, *, mixed):
    """Build a runner's storage without invoking the target's allocator."""

    from sglang.srt.mem_cache.allocator import (
        PagedTokenToKVPoolAllocator,
        TokenToKVPoolAllocator,
    )
    from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
    from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig

    allowed = (ReqToTokenPool,)
    if (
        type(target_requests) not in allowed
        or target_requests._aux_cache is not None
        or target_requests.pre_alloc_size
    ):
        raise ValueError(
            "Independent DFlash needs an unclaimed standard request-pool lifecycle"
        )
    runner = worker.draft_model_runner
    target_capacity = worker.model_runner.effective_logical_max_total_num_tokens
    layers = list(worker.draft_model.layers)
    from sglang.srt.models.dflash import DFlashAttention

    if runner.use_mla_backend or any(
        type(layer.self_attn) is not DFlashAttention for layer in layers
    ):
        raise ValueError(
            "Independent DFLASH cache currently supports the standard GQA/MHA draft only"
        )
    windows = {
        int(layer.self_attn.sliding_window_size) + 1
        for layer in layers
        if int(layer.self_attn.sliding_window_size) >= 0
    }
    full_layers = [
        layer.self_attn.attn.layer_id
        for layer in layers
        if int(layer.self_attn.sliding_window_size) < 0
    ]
    window_layers = [
        layer.self_attn.attn.layer_id
        for layer in layers
        if int(layer.self_attn.sliding_window_size) >= 0
    ]
    if mixed and (len(windows) != 1 or not full_layers or not window_layers):
        raise ValueError(
            "Hybrid DFlash requires a single native window plus full-attention layers"
        )
    owners = 1
    layout = DraftSlotLayout(
        slots=target_requests.size * owners,
        context_length=worker.model_runner.model_config.context_len,
        page_size=worker.page_size,
        block_size=worker.block_size,
        window=next(iter(windows)) if mixed else None,
    )
    requests = ReqToTokenPool(
        layout.slots, layout.stride, worker.device, enable_memory_saver=False
    )
    attention = layers[0].self_attn
    common = dict(
        page_size=layout.page_size,
        dtype=runner.kv_cache_dtype,
        head_num=attention.num_kv_heads,
        head_dim=attention.head_dim,
        device=worker.device,
        enable_memory_saver=False,
    )
    if mixed:
        pool = SWAKVPool(
            size=layout.full_capacity,
            size_swa=layout.window_capacity,
            swa_attention_layer_ids=window_layers,
            full_attention_layer_ids=full_layers,
            **common,
        )
        allocator = SWATokenToKVPoolAllocator(
            layout.full_capacity,
            layout.window_capacity,
            layout.page_size,
            runner.kv_cache_dtype,
            worker.device,
            pool,
            False,
        )
        runner.is_hybrid_swa = True
        runner.model_config.is_hybrid_swa = True
        runner.model_config.swa_attention_layer_ids = window_layers
        runner.model_config.full_attention_layer_ids = full_layers
        runner.sliding_window_size = layout.window - 1
    else:
        pool = MHATokenToKVPool(
            size=layout.full_capacity, layer_num=len(layers), **common
        )
        allocator = (
            TokenToKVPoolAllocator(
                layout.full_capacity, runner.kv_cache_dtype, worker.device, pool, False
            )
            if layout.page_size == 1
            else PagedTokenToKVPoolAllocator(
                layout.full_capacity,
                layout.page_size,
                runner.kv_cache_dtype,
                worker.device,
                pool,
                False,
            )
        )
    cache = IndependentDraftCache(
        layout=layout,
        target_requests=target_requests,
        requests=requests,
        allocator=allocator,
        device=worker.device,
        local_slot_offset=(0),
    )
    # Every request slot already reserves its entire model context in this
    # private slab. Report capacity in the target allocator's token units:
    # its free pages and the scheduler's conservation total must agree, even
    # when that allocator includes more headroom than one draft slot needs.
    cache.dense_capacity = target_capacity
    target_requests.attach_aux_cache(cache)
    runner.req_to_token_pool = requests
    runner.token_to_kv_pool = pool
    runner.token_to_kv_pool_allocator = allocator
    runner.max_total_num_tokens = layout.full_capacity
    runner.max_running_requests = layout.slots
    runner.memory_pool_config = MemoryPoolConfig(
        max_total_num_tokens=layout.full_capacity,
        max_running_requests=layout.slots,
        full_max_total_num_tokens=layout.full_capacity,
        swa_max_total_num_tokens=layout.window_capacity if mixed else None,
    )
    if mixed:
        runner.full_max_total_num_tokens = layout.full_capacity
        runner.swa_max_total_num_tokens = layout.window_capacity
        import msgspec

        runner.layer_info = msgspec.structs.replace(
            runner.layer_info,
            full_attention_layer_ids=full_layers,
            swa_attention_layer_ids=window_layers,
        )
    worker._draft_worker.req_to_token_pool = requests
    worker._draft_worker.token_to_kv_pool_allocator = allocator
    # Properties and eager/backend initialization still use the configurator's
    # logical-capacity contract, even though this adapter builds the pool itself.
    runner.init_kv_cache_configurator(
        draft_parallel_layout=worker.draft_parallel_layout
    )
    runner._init_post_memory_pool_components()
    return cache
