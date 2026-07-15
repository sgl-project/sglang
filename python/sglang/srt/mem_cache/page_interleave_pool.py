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

"""KV pools striped at logical-page granularity across a shard group.

Companion to ``DESIGN_kv_sharding_logical_page.md`` /
``PLAN_kv_shard_logical_page_impl.md``. Per-layer pool tensors keep the stock
shape and size — identical rows on every rank; only the written rows differ:
each rank persists the rows it owns under the pure placement bijection
(``mem_cache/page_interleave.py``).

Central consequence for the forward pass: during a sharded prefill, extend
attention never reads the KV pool — a rank's pool holds only its stripe of
the prefix *and* of the current chunk. Attention reads a per-layer assembled
scratch slot laid out ``[prefix region | chunk region | trash page]``:

- The prefix is NCCL-allgathered from the shard group into the slot one layer
  ahead, on a dedicated side stream with a dedicated ``PyNcclCommunicator``
  (the ``dsa_cache_layer_split.py`` broadcast template). The §5 padding rule
  of the design doc makes it a *regular* allgather: every rank contributes
  exactly ``prefix_groups * page_size`` rows (its page of every logical
  group, at the same local rows on every rank), so the owner-major allgather
  output needs no reorder pass.
- The chunk region is staged locally on the compute stream where the write
  path already holds the full chunk (GQA CP allgathers K/V before the pool
  write; MLA latent is replicated across the shard group at write time).
- The trash page absorbs padded/dummy locations (logical group 0 — the
  reserved padded page — and any location outside the current batch's plan).

Scratch addressing goes through a per-batch ``logical group -> plan
position`` lookup (``_group_pos``, one int32 per logical group, rebuilt in
``begin_shard_extend``): purely local, mirrored by construction, and valid
for any consumer index vector (attention page tables, MLA prefix
``kv_indices``), not just whole-prefix position arithmetic.

Layer-ahead pipelining needs no model changes: acquiring layer ``l``'s slot
for reading kicks layer ``l+1``'s gather. Reads happen in SPMD lockstep on
every rank, so the collective order is symmetric by construction.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.memory_pool import (
    GPU_MEMORY_TYPE_KV_CACHE,
    MHATokenToKVPool,
    MLATokenToKVPool,
    RadixAttention,
    unwrap_write_loc,
)
from sglang.srt.mem_cache.page_interleave import (
    PageInterleavePlacement,
    PageShardSpec,
)
from sglang.srt.mem_cache.utils import (
    get_mla_kv_buffer_triton,
    set_mla_kv_buffer_triton,
)

if TYPE_CHECKING:
    from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
    from sglang.srt.distributed.parallel_state import GroupCoordinator

logger = logging.getLogger(__name__)


class _ScratchSlot:
    """One half of the double buffer: scratch tensors + ready event.

    ``resident_key`` identifies what the slot currently holds — ``(layer_id,
    epoch)`` — so repeated reads of the same layer skip the re-gather and a
    new batch (epoch bump) retires stale residency without any invalidation
    walk.
    """

    def __init__(self, tensors: Dict[str, torch.Tensor], device_module):
        self.tensors = tensors
        self.ready = device_module.Event()
        self.resident_key: Optional[Tuple[int, int]] = None


class PageInterleaveKVPoolMixin:
    """Shard-generic state and mechanics; mixed into concrete pools below.

    Subclasses call ``_init_page_shard_state`` after the base pool has created
    its buffers, and implement ``_scratch_tensor_specs`` (per-slot tensors) and
    ``_gather_pairs`` (pool buffer -> scratch tensor pairs of one layer).
    """

    # ---- init ---------------------------------------------------------------

    def _init_page_shard_state(
        self, shard_spec: PageShardSpec, shard_group: GroupCoordinator
    ):
        from sglang.srt.distributed.device_communicators.pynccl import (
            PyNcclCommunicator,
        )

        spec = shard_spec
        assert spec.shard_size > 1, "page-interleave sharding needs shard_size > 1"
        assert spec.shard_size == shard_group.world_size
        assert spec.shard_rank == shard_group.rank_in_group
        assert spec.page_size == self.page_size
        gs = spec.logical_page_size
        assert spec.max_prefix_tokens % gs == 0
        assert spec.chunk_tokens % gs == 0

        self.shard_spec = spec
        self.placement = PageInterleavePlacement(spec)
        self.shard_rank = spec.shard_rank
        self.shard_size = spec.shard_size
        self.device_module = torch.get_device_module(self.device)

        # Dedicated communicator + stream so the layer-ahead gathers never
        # interleave with the group's main collectives (the layer-split
        # template: dsa_cache_layer_split.py::_init_layer_broadcast_comm).
        self.kv_gather_comm: PyNcclCommunicator = PyNcclCommunicator(
            group=shard_group.cpu_group, device=shard_group.device
        )
        self.kv_gather_stream = self.device_module.Stream()

        # Scratch: [prefix | chunk | trash page], double-buffered.
        scratch_rows = spec.max_prefix_tokens + spec.chunk_tokens + spec.page_size
        self._chunk_base = spec.max_prefix_tokens
        self._trash_base = spec.max_prefix_tokens + spec.chunk_tokens
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                self._slots = [
                    _ScratchSlot(
                        self._scratch_tensor_specs(scratch_rows), self.device_module
                    )
                    for _ in range(2)
                ]
                # Per-batch plan position of every logical group; -1 = not in
                # the current batch (translated to the trash page). Logical
                # group ids run [0, size/ps + 1] — the same range as the
                # physical pool's pages incl. the reserved padded page.
                self._group_pos = torch.full(
                    (self.size // spec.page_size + 2,),
                    -1,
                    dtype=torch.int32,
                    device=self.device,
                )

        self._epoch = 0
        self._shard_extend_active = False
        self._prefix_active = False
        self._n_prefix_groups = 0
        self._send_rows: Optional[torch.Tensor] = None
        self._write_plan_key = None
        self._write_plan: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._translate_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        self._translate_cache_epoch = -1

        logger.info(
            "Page-interleave KV sharding enabled: shard_rank=%d shard_size=%d "
            "page_size=%d scratch_rows=%d x2",
            self.shard_rank,
            self.shard_size,
            spec.page_size,
            scratch_rows,
        )

    # ---- subclass hooks -------------------------------------------------------

    def _scratch_tensor_specs(self, rows: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def _gather_pairs(self, local_layer: int) -> List[Tuple[torch.Tensor, str]]:
        """(pool buffer of ``local_layer``, scratch tensor name) pairs."""
        raise NotImplementedError

    # ---- per-batch plan -------------------------------------------------------

    def begin_shard_extend(
        self,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        prefix_lens_cpu: List[int],
        seq_lens_cpu: List[int],
    ) -> None:
        """Capture the batch's gather plan and kick the first layer's gather.

        Called once per scheduled extend batch, at attention-metadata build
        time (after ``req_to_token`` holds the batch's allocation). All inputs
        are mirrored across ranks, so every rank derives the identical plan.
        """
        spec = self.shard_spec
        ps, gs = spec.page_size, spec.logical_page_size

        prefix_group_list = []
        chunk_group_list = []
        for i in range(len(prefix_lens_cpu)):
            prefix_len, seq_len = int(prefix_lens_cpu[i]), int(seq_lens_cpu[i])
            assert prefix_len % ps == 0, (
                f"sharded prefill requires physical-page-aligned prefixes "
                f"(the radix-tree match quantum), got prefix_len={prefix_len}, "
                f"page_size={ps}"
            )
            row = req_to_token[req_pool_indices[i]]
            # Sample every ps-th position, then collapse runs. Stride-gs
            # sampling would be wrong on both sides once fresh-group adoption
            # exists: an adoption seam makes one gs-window of positions span
            # TWO groups (the old boundary group, then the adopted group at
            # the position-congruent offset), and a cached prefix can carry a
            # seam per past turn boundary — any group with no sample point
            # would silently translate to the trash page. Every group's
            # positions in a request form ONE contiguous run starting at a
            # ps-aligned position with ps-multiple length (I3 + the ps tree
            # quantum), so ps-stride sampling hits every run and
            # unique_consecutive keeps one entry per group, in plan order.
            if prefix_len > 0:
                prefix_group_list.append(
                    torch.unique_consecutive(row[:prefix_len:ps].long() // gs)
                )
            if seq_len > prefix_len:
                chunk_group_list.append(
                    torch.unique_consecutive(row[prefix_len:seq_len:ps].long() // gs)
                )

        def _cat(parts):
            if not parts:
                return torch.empty((0,), dtype=torch.int64, device=self.device)
            return torch.cat(parts)

        prefix_groups = _cat(prefix_group_list)
        chunk_groups = _cat(chunk_group_list)
        n_prefix = prefix_groups.numel()
        n_chunk = chunk_groups.numel()
        assert n_prefix * gs <= spec.max_prefix_tokens, (
            f"prefix ({n_prefix * gs} tokens) exceeds the scratch prefix "
            f"capacity ({spec.max_prefix_tokens}); a reused chain carries "
            f"more mid-group turn boundaries than provisioned — relaunch "
            f"with a higher SGLANG_KV_SHARD_MAX_PREFIX_TURNS"
        )
        assert n_chunk * gs <= spec.chunk_tokens, (
            f"chunk ({n_chunk * gs} tokens) exceeds the scratch chunk "
            f"capacity ({spec.chunk_tokens})"
        )

        self._group_pos.fill_(-1)
        if n_prefix:
            self._group_pos[prefix_groups] = torch.arange(
                n_prefix, dtype=torch.int32, device=self.device
            )
        if n_chunk:
            self._group_pos[chunk_groups] = torch.arange(
                n_prefix, n_prefix + n_chunk, dtype=torch.int32, device=self.device
            )

        self._epoch += 1
        self._n_prefix_groups = n_prefix
        self._prefix_active = n_prefix > 0
        self._shard_extend_active = True
        self._write_plan_key = None
        self._write_plan = None
        if self._prefix_active:
            # Every rank's stripe of logical group Q sits at the same local
            # rows [Q*ps, (Q+1)*ps) (symmetric allocation), so the send-row
            # vector is identical on all ranks.
            self._send_rows = (
                prefix_groups[:, None] * ps
                + torch.arange(ps, dtype=torch.int64, device=self.device)
            ).reshape(-1)
            self._prefetch_layer(self.start_layer)
        else:
            self._send_rows = None

    def end_shard_extend(self) -> None:
        """Mark no sharded extend in flight (non-extend forward modes)."""
        self._shard_extend_active = False

    # ---- translation ----------------------------------------------------------

    def translate_loc_to_scratch(self, loc: torch.Tensor) -> torch.Tensor:
        """Logical token slots -> rows of the current batch's scratch slots.

        Prefix groups land owner-major (rank r's pages contiguous at
        ``r * block``, in plan order); chunk groups land in sequence order in
        the chunk region; anything outside the plan (padded locations, the
        reserved group 0) lands in the trash page.
        """
        spec = self.shard_spec
        ps, gs = spec.page_size, spec.logical_page_size
        loc64 = loc.long()
        j = self._group_pos[loc64 // gs].long()
        in_group = loc64 % gs
        n_prefix = self._n_prefix_groups
        block = n_prefix * ps
        prefix_row = (in_group // ps) * block + j * ps + loc64 % ps
        chunk_row = self._chunk_base + (j - n_prefix) * gs + in_group
        row = torch.where(j >= n_prefix, chunk_row, prefix_row)
        return torch.where(j < 0, self._trash_base + loc64 % ps, row)

    # ---- the layer-ahead gather -------------------------------------------------

    def _prefetch_layer(self, layer_id: int) -> None:
        """Kick the allgather assembling ``layer_id``'s prefix into its slot,
        on the gather stream. Idempotent per ``(layer_id, epoch)``."""
        local_layer = layer_id - self.start_layer
        if local_layer >= self.layer_num:
            return
        slot = self._slots[layer_id % 2]
        key = (layer_id, self._epoch)
        if slot.resident_key == key:
            return
        block = self._n_prefix_groups * self.shard_spec.page_size
        # Order the gather after all prior compute-stream work: the previous
        # tenant's reads (attention of layer_id - 2) and the pool writes that
        # produced the prefix rows.
        self.kv_gather_stream.wait_stream(self.device_module.current_stream())
        with self.device_module.stream(self.kv_gather_stream):
            for pool_buf, name in self._gather_pairs(local_layer):
                scratch = slot.tensors[name]
                send = scratch[self.shard_rank * block : (self.shard_rank + 1) * block]
                torch.index_select(pool_buf, 0, self._send_rows, out=send)
                # In-place regular allgather: send is exactly the rank's block
                # of the output, every rank contributes `block` rows.
                with self.kv_gather_comm.change_state(enable=True):
                    self.kv_gather_comm.all_gather(
                        scratch[: self.shard_size * block], send
                    )
            slot.ready.record(self.kv_gather_stream)
        slot.resident_key = key

    def _acquire_slot_for_read(self, layer_id: int) -> _ScratchSlot:
        """Block the compute stream on the slot's ready event (a no-op when
        the layer-ahead gather already landed) and kick the next layer's
        gather. A residency miss is a caller bug — the batch plan captured by
        ``begin_shard_extend`` must have prefetched this layer."""
        assert self._shard_extend_active, (
            "sharded pool read outside an active sharded extend batch "
            "(begin_shard_extend not called?)"
        )
        slot = self._slots[layer_id % 2]
        if self._prefix_active:
            assert slot.resident_key == (layer_id, self._epoch), (
                f"prefix scratch miss for layer {layer_id} "
                f"(resident={slot.resident_key}, epoch={self._epoch})"
            )
            self.device_module.current_stream().wait_event(slot.ready)
            self._prefetch_layer(layer_id + 1)
        return slot

    def _translate_loc_cached(self, loc: torch.Tensor) -> torch.Tensor:
        """Per-batch memoized ``translate_loc_to_scratch`` for the per-layer
        callers: the same loc tensors (``out_cache_loc`` at every layer's
        ``set_kv_buffer``; the prefix ``kv_indices`` at every layer's
        ``get_mla_kv_buffer``) arrive at all layers of a forward, and the
        plan is frozen per epoch — so each distinct loc tensor is translated
        once per batch instead of once per layer (~18 elementwise launches
        per call; the ``_get_write_plan`` precedent below). The dict holds
        one entry per distinct loc tensor of the batch and is dropped on
        epoch change."""
        if self._translate_cache_epoch != self._epoch:
            self._translate_cache_epoch = self._epoch
            self._translate_cache = {}
        key = (loc.data_ptr(), loc.numel())
        rows = self._translate_cache.get(key)
        if rows is None:
            rows = self.translate_loc_to_scratch(loc)
            self._translate_cache[key] = rows
        return rows

    # ---- write plan (owner filter), cached per (loc tensor, epoch) --------------

    def _get_write_plan(self, loc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(owned positions into ``loc``, their local physical rows).

        The same ``out_cache_loc`` tensor is passed for every layer of a
        forward, so the owner filter is computed once per batch, not per
        layer.
        """
        key = (loc.data_ptr(), loc.numel(), self._epoch)
        if self._write_plan_key != key:
            owned_idx = torch.nonzero(
                self.placement.local_mask(loc, self.shard_rank)
            ).squeeze(1)
            local_rows = self.placement.local_index(loc[owned_idx])
            self._write_plan_key = key
            self._write_plan = (owned_idx, local_rows)
        return self._write_plan


class PageInterleaveMHATokenToKVPool(PageInterleaveKVPoolMixin, MHATokenToKVPool):
    """MHA/GQA pool striped across the attention CP group.

    The write path relies on prefill CP's existing full-chunk allgather:
    ``cp_allgather_and_save_kv_cache`` hands ``set_kv_buffer`` the complete
    chunk's K/V with the full (unsplit) ``out_cache_loc`` on every rank; each
    rank persists only its stripe and stages the full chunk into the current
    layer's scratch chunk region (extend attention reads prefix *and* chunk
    through the scratch page table).
    """

    def __init__(
        self,
        *args,
        shard_spec: PageShardSpec,
        shard_group: GroupCoordinator,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert not self.use_hnd, "KV sharding supports the NHD layout only"
        assert (
            self.kv_cache_layout == "nhd"
        ), f"KV sharding supports the NHD layout only, got {self.kv_cache_layout}"
        assert not self.post_capture_active
        self._init_page_shard_state(shard_spec, shard_group)

    def _scratch_tensor_specs(self, rows: int) -> Dict[str, torch.Tensor]:
        return {
            "k": torch.zeros(
                (rows, self.head_num, self.head_dim),
                dtype=self.store_dtype,
                device=self.device,
            ),
            "v": torch.zeros(
                (rows, self.head_num, self.v_head_dim),
                dtype=self.store_dtype,
                device=self.device,
            ),
        }

    def _gather_pairs(self, local_layer: int):
        return [
            (self.k_buffer[local_layer], "k"),
            (self.v_buffer[local_layer], "v"),
        ]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
        dcp_kv_mask: Optional[torch.Tensor] = None,
    ):
        assert dcp_kv_mask is None, "DCP is mutually exclusive with KV sharding"
        loc, _, _ = unwrap_write_loc(loc_info)
        layer_id = (
            layer_id_override if layer_id_override is not None else layer.layer_id
        )

        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k = cache_k / k_scale
            if v_scale is not None:
                cache_v = cache_v / v_scale
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)
        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        if self._shard_extend_active:
            # Stage the full chunk into this layer's slot on the compute
            # stream: ordered before this layer's attention read for free and
            # disjoint from the gather stream, which only writes the prefix
            # region. Padded locations translate to the trash page.
            slot = self._slots[layer_id % 2]
            rows = self._translate_loc_cached(loc)
            slot.tensors["k"][rows] = cache_k
            slot.tensors["v"][rows] = cache_v

        owned_idx, local_rows = self._get_write_plan(loc)
        if owned_idx.numel() > 0:
            self._store_kv_layer(
                layer_id - self.start_layer,
                local_rows,
                cache_k.index_select(0, owned_idx),
                cache_v.index_select(0, owned_idx),
            )

    def get_key_buffer(self, layer_id: int):
        if self._shard_extend_active:
            slot = self._acquire_slot_for_read(layer_id)
            k = slot.tensors["k"]
            return k.view(self.dtype) if self.store_dtype != self.dtype else k
        return super().get_key_buffer(layer_id)

    def get_value_buffer(self, layer_id: int):
        if self._shard_extend_active:
            slot = self._acquire_slot_for_read(layer_id)
            v = slot.tensors["v"]
            return v.view(self.dtype) if self.store_dtype != self.dtype else v
        return super().get_value_buffer(layer_id)


class PageInterleaveMLATokenToKVPool(PageInterleaveKVPoolMixin, MLATokenToKVPool):
    """MLA latent pool striped across its shard group (the attn-CP group when
    prefill CP is active, the attn-TP group otherwise — see
    ``page_interleave.get_kv_shard_group``).

    The latent KV reaching the write path is identical on every shard-group
    rank (``ReplicatedLinear`` projection across attn-TP; the CP allgather of
    ``rebuild_cp_kv_cache`` across attn-CP), so the write filter needs no
    compute-time communication. Read consumers, all served from the assembled
    scratch: the chunked-prefix MHA path fetches prefix rows through
    ``get_mla_kv_buffer``; the absorbed-MLA and one-shot paths read
    ``get_key_buffer``/``get_value_buffer`` through the translated page
    table. The current chunk is staged into the slot at write time so the
    page-table consumers cover ``[prefix | chunk]`` uniformly.
    """

    def __init__(
        self,
        *args,
        shard_spec: PageShardSpec,
        shard_group: GroupCoordinator,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert not self.use_dsa, "DSA models are not supported by KV sharding yet"
        self._init_page_shard_state(shard_spec, shard_group)

    def _scratch_tensor_specs(self, rows: int) -> Dict[str, torch.Tensor]:
        return {
            "kv": torch.zeros(
                (rows, 1, self.kv_cache_dim),
                dtype=self.store_dtype,
                device=self.device,
            ),
        }

    def _gather_pairs(self, local_layer: int):
        return [(self.kv_buffer[local_layer], "kv")]

    def _scratch_kv(self, slot: _ScratchSlot) -> torch.Tensor:
        kv = slot.tensors["kv"]
        if self.store_dtype != self.dtype:
            return kv.view(self.dtype)
        return kv

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        loc, _, _ = unwrap_write_loc(loc_info)
        if self._shard_extend_active:
            # Stage the full chunk into this layer's slot (compute stream):
            # the absorbed / one-shot readers cover the current chunk through
            # the translated page table too.
            slot = self._slots[layer.layer_id % 2]
            rows = self._translate_loc_cached(loc)
            staged_k = cache_k
            if staged_k.dtype != self.dtype:
                staged_k = staged_k.to(self.dtype)
            self._scratch_kv(slot)[rows] = staged_k
        owned_idx, local_rows = self._get_write_plan(loc)
        if owned_idx.numel() == 0:
            return
        super().set_kv_buffer(
            layer,
            local_rows,
            cache_k.index_select(0, owned_idx),
            cache_v,
        )

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        if self._shard_extend_active:
            slot = self._slots[layer.layer_id % 2]
            rows = self._translate_loc_cached(loc)
            staged_nope, staged_rope = cache_k_nope, cache_k_rope
            if staged_nope.dtype != self.dtype:
                staged_nope = staged_nope.to(self.dtype)
                staged_rope = staged_rope.to(self.dtype)
            if self.store_dtype != self.dtype:
                staged_nope = staged_nope.view(self.store_dtype)
                staged_rope = staged_rope.view(self.store_dtype)
            set_mla_kv_buffer_triton(slot.tensors["kv"], rows, staged_nope, staged_rope)
        owned_idx, local_rows = self._get_write_plan(loc)
        if owned_idx.numel() == 0:
            return
        super().set_mla_kv_buffer(
            layer,
            local_rows,
            cache_k_nope.index_select(0, owned_idx),
            cache_k_rope.index_select(0, owned_idx),
        )

    def get_kv_buffer_shape(self):
        # Shape probes (e.g. the eager runner's DCP-metadata prep) must not
        # route through the attention getters below — they may run before
        # this batch's metadata build while the previous batch's shard-extend
        # flag is still set.
        k = self.kv_buffer[0]
        return k.shape, k[..., : self.kv_lora_rank].shape

    def get_key_buffer(self, layer_id: int):
        # During a sharded extend the pool holds only this rank's stripe;
        # page-table readers (absorbed MLA, incl. the CP zigzag wrapper) get
        # the assembled scratch — metadata.page_table is already translated
        # to scratch rows.
        if self._shard_extend_active:
            return self._scratch_kv(self._acquire_slot_for_read(layer_id))
        return super().get_key_buffer(layer_id)

    def get_value_buffer(self, layer_id: int):
        if self._shard_extend_active:
            kv = self._scratch_kv(self._acquire_slot_for_read(layer_id))
            return kv[..., : self.kv_lora_rank]
        return super().get_value_buffer(layer_id)

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        slot = self._acquire_slot_for_read(layer.layer_id)
        rows = self._translate_loc_cached(loc)
        kv_buffer = slot.tensors["kv"]
        if self.store_dtype != self.dtype:
            kv_buffer = kv_buffer.view(self.dtype)
        dst_dtype = dst_dtype or self.dtype
        cache_k_nope = torch.empty(
            (loc.shape[0], 1, self.kv_lora_rank),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        cache_k_rope = torch.empty(
            (loc.shape[0], 1, self.qk_rope_head_dim),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        get_mla_kv_buffer_triton(kv_buffer, rows, cache_k_nope, cache_k_rope)
        return cache_k_nope, cache_k_rope
