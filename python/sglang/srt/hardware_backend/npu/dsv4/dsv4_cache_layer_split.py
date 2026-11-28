# SPDX-License-Identifier: Apache-2.0
"""Layer-sharded NPU DSV4 KV cache pool for context-parallel prefill.

Splits the DSV4 multi-pool KV cache across context-parallel (CP) ranks
so each rank only materializes the layers it owns, reducing per-rank
KV memory.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool import (
    DSV4NPUTokenToKVPool,
)
from sglang.srt.hardware_backend.npu.layer_split_mixin import NPULayerSplitMixin
from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
from sglang.srt.layers.cp.utils import get_layer_shard_range
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4SingleKVPool

logger = logging.getLogger(__name__)

# Broadcast kinds. ``index`` carries the c4-indexer K + scale pair (the
# indexer shares the C4 page space).
KIND_SWA = "swa"
KIND_C4 = "c4"
KIND_C128 = "c128"
KIND_INDEX = "index"


def compute_broadcast_pages(
    live_pages: torch.Tensor,
    write_loc: torch.Tensor,
    page_size: int,
) -> Optional[torch.Tensor]:
    """Live pages that this forward's writes cannot fully cover.

    A page is "fully covered" when every slot in it appears in ``write_loc`` —
    the layer's set then writes all its rows and no broadcast is needed for
    it. The remaining pages must come from the owner-broadcast: their
    non-new rows are immutable (owner's pool content this forward never
    overwrites) and their new rows get patched by the set.
    """
    if write_loc is None or live_pages is None or live_pages.numel() == 0:
        return None
    live = live_pages.reshape(-1).long()
    loc_pages = write_loc.reshape(-1).long() // page_size
    counts = torch.bincount(loc_pages, minlength=int(live.max().item()) + 1)
    fully_covered = counts[live] == page_size
    return live[~fully_covered]


class LayerSplitDSV4NPUTokenToKVPool(NPULayerSplitMixin, DSV4NPUTokenToKVPool):
    """NPU DSV4 KV pool that shards layers across CP ranks with owner-broadcast reads.

    Each sub-pool allocates buffers only for the stage layers this rank owns;
    ``layer_mapping`` stays global so routing is unchanged, and the per-ratio
    owned-slot tables below rebase ``compress_layer_id`` onto the dense owned
    buffers.
    """

    def __init__(
        self,
        *args,
        layer_shard_rank: int,
        layer_shard_size: int,
        **kwargs,
    ):
        self._init_layer_split(layer_shard_rank, layer_shard_size)
        ratios = kwargs.get("compression_ratios")
        assert ratios is not None, "compression_ratios is required"
        unsupported = sorted({r for r in ratios if r not in (0, 4, 128)})
        assert not unsupported, (
            "NPU DSV4 layer-split only supports compression ratios 0/4/128 "
            f"(the V4 family); got ratios {unsupported}. The V4.1 ratio-1/2 "
            "compressed pools are not supported on NPU yet."
        )
        start_layer = kwargs.get("start_layer")
        end_layer = kwargs.get("end_layer")
        if (
            start_layer is not None
            and end_layer is not None
            and len(ratios) >= end_layer
        ):
            self._ls_stage_start = start_layer
            self._ls_stage_end = end_layer
        else:
            self._ls_stage_start = 0
            self._ls_stage_end = len(ratios)
        stage_layer_num = self._ls_stage_end - self._ls_stage_start
        self._ls_owned_start, self._ls_owned_end = get_layer_shard_range(
            layer_shard_rank, layer_shard_size, stage_layer_num
        )
        # Global id of the first owned layer
        self.layer_shard_start = self._ls_stage_start + self._ls_owned_start
        super().__init__(*args, **kwargs)
        self._log_layer_shard_plan("DSV4")
        self._log_actual_kv_bytes()
        self._build_owned_slot_maps()
        self._create_remote_scratches()
        # Live page sets of the current forward
        self._live_pages: Dict[str, Optional[torch.Tensor]] = {}

    def _log_actual_kv_bytes(self) -> None:
        """The actual KV bytes this rank allocated, all families summed."""
        try:
            swa_bytes = sum(b.nbytes for b in self.swa_kv_pool.kv_buffer)
            c4_bytes = (
                sum(b.nbytes for b in self.c4_kv_pool.kv_buffer)
                if self.c4_kv_pool is not None
                else 0
            )
            c128_bytes = (
                sum(b.nbytes for b in self.c128_kv_pool.kv_buffer)
                if self.c128_kv_pool is not None
                else 0
            )
            index_bytes = 0
            for pool in self.index_pools.values():
                index_bytes += sum(b.nbytes for b in pool.index_k_buffer)
                index_bytes += sum(b.nbytes for b in pool.index_scale_buffer)
            state_bytes = 0
            for pools in (
                self.compress_state_pools,
                self.indexer_compress_state_pools,
            ):
                for pool in pools:
                    buf = getattr(pool, "kv_score_buffer", None)
                    score = getattr(buf, "kv_score", None)
                    if score is not None:
                        state_bytes += score.nbytes
            gb = 1 << 30
            logger.info(
                "DSV4 KV cache [layer-split rank %d/%d owns global layers"
                " [%d, %d) (%d/%d)]: actual %.2f GB (swa %.2f, c4 %.2f,"
                " c128 %.2f, index %.2f, state %.2f)",
                self.layer_shard_rank,
                self.layer_shard_size,
                self.layer_shard_start,
                self._ls_stage_start + self._ls_owned_end,
                self._ls_owned_end - self._ls_owned_start,
                self._ls_stage_end - self._ls_stage_start,
                (swa_bytes + c4_bytes + c128_bytes + index_bytes + state_bytes) / gb,
                swa_bytes / gb,
                c4_bytes / gb,
                c128_bytes / gb,
                index_bytes / gb,
                state_bytes / gb,
            )
        except Exception:
            logger.warning(
                "DSV4 KV cache: failed to log actual allocation bytes",
                exc_info=True,
            )

    def _build_owned_slot_maps(self):
        """Dense owned-slot tables per compressed pool family."""
        owned_stage_ids = range(
            self._ls_stage_start + self._ls_owned_start,
            self._ls_stage_start + self._ls_owned_end,
        )
        self._ls_compress_slots: Dict[int, Dict[int, int]] = {4: {}, 128: {}}
        counters: dict[int, int] = {4: 0, 128: 0}
        for global_id in owned_stage_ids:
            ratio = self.compression_ratios[global_id]
            if ratio == 0:
                continue
            # compress_layer_id of a c4/c128 layer = number of same-ratio
            # stage layers before it.
            cl_id = sum(
                1
                for gid in range(self._ls_stage_start, global_id)
                if self.compression_ratios[gid] == ratio
            )
            self._ls_compress_slots[ratio][cl_id] = counters[ratio]
            counters[ratio] += 1

    def _ls_owned_stage_ids(self) -> List[int]:
        """Global ids of the stage layers this rank owns, ascending."""
        return list(
            range(
                self._ls_stage_start + self._ls_owned_start,
                self._ls_stage_start + self._ls_owned_end,
            )
        )

    def get_kv_transfer_layer_span(self, num_entries: int) -> int:
        """Layer count this rank's contiguous buf entries cover."""
        owned_ids = self._ls_owned_stage_ids()
        owned_c4 = sum(self.compression_ratios[l] == 4 for l in owned_ids)
        assert num_entries == 3 * owned_c4, (
            f"layer-split transfer reported {num_entries} entries; expected "
            f"3 buffer groups x {owned_c4} owned ratio-4 layers"
        )
        return self._ls_owned_end - self._ls_owned_start

    def swa_layer_ids(self) -> List[int]:
        """Global ids of the layers whose SWA buffer lives on this rank."""
        return self._ls_owned_stage_ids()

    def get_c128_layer_ids(self) -> List[int]:
        """Owned c128 layer ids in buffer order."""
        if self.c128_kv_pool is None:
            return []
        return [
            layer_id
            for layer_id in self._ls_owned_stage_ids()
            if self.compression_ratios[layer_id] == 128
        ]

    def get_kv_layer_ids(self) -> List[int]:
        """Owned c4 entries' global ids in registration order."""
        if self.c4_kv_pool is None:
            return []
        owned_c4 = [
            layer_id
            for layer_id in self._ls_owned_stage_ids()
            if self.compression_ratios[layer_id] == 4
        ]
        indexer_pool = self._indexer_pool(4)
        ids: List[int] = []
        for buffers in (
            self.c4_kv_pool.kv_buffer,
            indexer_pool.index_k_buffer,
            indexer_pool.index_scale_buffer,
        ):
            ids.extend(layer_id for layer_id, _ in zip(owned_c4, buffers))
        return ids

    def get_device_pool_infos(self) -> tuple:
        """Describe the owned owner stores; sources shrink to owned layers."""
        owned = set(self._ls_owned_stage_ids())
        saved_sources = self.sources_by_ratio
        self.sources_by_ratio = {
            ratio: [l for l in sources if l in owned]
            for ratio, sources in saved_sources.items()
        }
        try:
            return super().get_device_pool_infos()
        finally:
            self.sources_by_ratio = saved_sources

    def _split_owned_stage_local(self, layer_id: int) -> int:
        """Stage-local id of an owned layer, rebased onto the owned range."""
        local = layer_id - self._ls_stage_start
        assert self._ls_owned_start <= local < self._ls_owned_end, (
            f"layer {layer_id} (local {local}) is not owned by shard rank "
            f"{self.layer_shard_rank}"
        )
        return local - self._ls_owned_start

    def _swa_local_layer_id(self, layer_id: int) -> int:
        """Rebase onto this rank's SWA buffer (the split allocates SWA
        buffers for owned layers only)."""
        return self._split_owned_stage_local(layer_id)

    def _owned_compress_slot(self, layer_id: int) -> Tuple[int, int]:
        item = self.layer_mapping[layer_id]
        ratio = item.compress_ratio
        assert ratio in self._ls_compress_slots, ratio
        return ratio, self._ls_compress_slots[ratio][item.compress_layer_id]

    def _split_layer_kinds(self, layer_id: int) -> Tuple[str, ...]:
        ratio = self.compression_ratios[layer_id]
        if ratio == 0:
            return (KIND_SWA,)
        if ratio == 4:
            return (KIND_SWA, KIND_C4, KIND_INDEX)
        if ratio == 128:
            return (KIND_SWA, KIND_C128)
        raise ValueError(
            f"DSV4 layer-split does not support compress_ratio={ratio} "
            f"(layer {layer_id})"
        )

    def _make_kv_pool(
        self,
        *,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        global_page_size: int,
        cls: type = DeepSeekV4SingleKVPool,
        kv_layout: KVLayout = KVLayout.V4,
    ):
        is_c4_pool = page_size * 4 == global_page_size
        is_c128_pool = page_size * 128 == global_page_size
        if is_c4_pool:
            ratio = 4
        elif is_c128_pool:
            ratio = 128
        elif page_size == global_page_size:
            ratio = 0
        else:
            raise ValueError(
                "DSV4 layer-split does not support low-ratio compressed pools "
                f"(page_size={page_size}, global={global_page_size})"
            )
        if ratio == 0:
            owned_num = self._ls_owned_end - self._ls_owned_start
        else:
            owned_num = sum(
                1
                for gid in range(
                    self._ls_stage_start + self._ls_owned_start,
                    self._ls_stage_start + self._ls_owned_end,
                )
                if self.compression_ratios[gid] == ratio
            )
        if owned_num != layer_num:
            logger.info(
                "DSV4 layer-split: shrinking ratio-%s pool from %d to %d owned "
                "layers (shard rank %d/%d)",
                ratio,
                layer_num,
                owned_num,
                self.layer_shard_rank,
                self.layer_shard_size,
            )
        return super()._make_kv_pool(
            size=size,
            page_size=page_size,
            dtype=dtype,
            layer_num=owned_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            global_page_size=global_page_size,
            cls=cls,
            kv_layout=kv_layout,
        )

    def _make_indexer_pool(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        index_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        global_page_size: Optional[int] = None,
    ):
        ratio = global_page_size // page_size
        assert ratio in (4, 128), f"indexer pool ratio {ratio} unsupported"
        owned_num = sum(
            1
            for gid in range(
                self._ls_stage_start + self._ls_owned_start,
                self._ls_stage_start + self._ls_owned_end,
            )
            if self.compression_ratios[gid] == ratio
        )
        return super()._make_indexer_pool(
            size,
            page_size,
            dtype,
            index_head_dim,
            owned_num,
            device,
            enable_memory_saver,
            global_page_size=global_page_size,
        )

    def _create_remote_scratches(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            remote_swa = None
            if self.swa_kv_pool is not None:
                remote_swa = self._make_single_pool_scratch(self.swa_kv_pool)
            remote_c4 = remote_c128 = None
            if self.c4_kv_pool is not None:
                remote_c4 = self._make_single_pool_scratch(self.c4_kv_pool)
            if self.c128_kv_pool is not None:
                remote_c128 = self._make_single_pool_scratch(self.c128_kv_pool)
            remote_index_k = remote_index_scale = None
            index_pool = self.index_pools.get(4)
            if index_pool is not None:
                remote_index_k, remote_index_scale = self._make_indexer_scratch(
                    index_pool
                )
        self._register_split_kind(KIND_SWA, remote_swa)
        self._register_split_kind(KIND_C4, remote_c4)
        self._register_split_kind(KIND_C128, remote_c128)
        self._register_split_kind(KIND_INDEX, remote_index_k)
        self._remote_index_scale_scratch = remote_index_scale

    def _make_single_pool_scratch(self, pool) -> torch.Tensor:
        """One layer's PA_ND buffer for ``pool`` (works even when the pool
        shrank to zero owned layers on this rank)."""
        assert pool.store_dtype == torch.bfloat16, (
            "DSV4 layer-split requires the NPU bf16 PA_ND pool"
        )
        if is_npu_arch35():
            kv_dim = pool.arch35_packed_kv_dim
            kv_dtype = torch.float8_e4m3fn
        else:
            kv_dim = pool.qk_nope_head_dim + pool.qk_rope_head_dim
            kv_dtype = torch.bfloat16
        npu_num_pages = (pool.size + pool.kernel_page_size + 1) // pool.kernel_page_size
        return torch.zeros(
            npu_num_pages,
            pool.kernel_page_size,
            1,
            kv_dim,
            dtype=kv_dtype,
            device=pool.device,
        )

    def _make_indexer_scratch(self, index_pool) -> Tuple[torch.Tensor, torch.Tensor]:
        kp = index_pool._kernel_page_size
        npu_num_pages = (index_pool.size + kp + 1) // kp
        if is_npu_arch35():
            k_dtype, scale_dtype = torch.float8_e4m3fn, torch.float32
        else:
            k_dtype, scale_dtype = torch.int8, torch.float16
        d = index_pool.index_head_dim
        remote_index_k = torch.zeros(
            npu_num_pages, kp, 1, d, dtype=k_dtype, device=index_pool.device
        )
        remote_index_scale = torch.zeros(
            npu_num_pages, kp, 1, 1, dtype=scale_dtype, device=index_pool.device
        )
        return remote_index_k, remote_index_scale

    def set_live_pages(
        self,
        pages_by_kind: Dict[str, torch.Tensor],
        broadcast_pages_by_kind: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Register the sorted-unique live page ids per page space for the
        current forward. Must be called on every CP rank with identical sets
        before the first non-owned layer read.
        """
        self._live_pages = {
            kind: (pages if pages is None else torch.unique(pages.reshape(-1)))
            for kind, pages in pages_by_kind.items()
        }
        self._broadcast_pages = (
            {
                kind: (None if pages is None else torch.unique(pages.reshape(-1)))
                for kind, pages in broadcast_pages_by_kind.items()
            }
            if broadcast_pages_by_kind is not None
            else {}
        )

    def _split_pages_for(self, kind: str) -> Optional[torch.Tensor]:
        if kind == KIND_INDEX:
            kind = KIND_C4  # indexer shares the C4 page space
        return self._live_pages.get(kind)

    def _broadcast_pages_for(self, kind: str) -> Optional[torch.Tensor]:
        """The page subset a kick delivers (falls back to the live set)."""
        if kind == KIND_INDEX:
            kind = KIND_C4  # indexer shares the C4 page space
        subset = self._broadcast_pages.get(kind)
        return subset if subset is not None else self._split_pages_for(kind)

    def prefetch_layer(self, layer_id: int) -> None:
        """Kick one-layer-ahead owner-broadcasts for every kind of ``layer_id``."""
        for kind in self._split_layer_kinds(layer_id):
            pages = self._broadcast_pages_for(kind)
            if pages is None or pages.numel() == 0:
                continue
            self._prefetch_kind(
                kind,
                layer_id,
                index=pages,
                layer_transfer_counter=self.layer_transfer_counter,
                layer_transfer_idx=layer_id - self.start_layer,
            )

    def maybe_prefetch_next_layer(self, next_layer_id: int, forward_batch) -> None:
        """Gate + kick the one-layer-ahead broadcast for ``next_layer_id``.

        Skipped for prefix-free batches: without a cached prefix the CP
        gather already produces whole live rows, the layer's set seeds its
        scratch locally under ``cp_gathered_full`` and every read hits the
        registered copy, so the kick's payload would be discarded in full.
        """
        if not forward_batch.forward_mode.is_extend():
            return
        if next_layer_id >= self._ls_stage_end:
            return
        prefix_lens = getattr(forward_batch, "extend_prefix_lens_cpu", None)
        if prefix_lens is not None and all(int(n) == 0 for n in prefix_lens):
            return
        self.prefetch_layer(next_layer_id)

    def _split_owner_buffer(self, kind: str, layer_id: int) -> torch.Tensor:
        if kind == KIND_SWA:
            return self.swa_kv_pool.kv_buffer[self._swa_local_layer_id(layer_id)]
        if kind == KIND_C4:
            ratio, slot = self._owned_compress_slot(layer_id)
            assert ratio == 4
            return self.c4_kv_pool.kv_buffer[slot]
        if kind == KIND_C128:
            ratio, slot = self._owned_compress_slot(layer_id)
            assert ratio == 128
            return self.c128_kv_pool.kv_buffer[slot]
        assert kind == KIND_INDEX, kind
        ratio, slot = self._owned_compress_slot(layer_id)
        assert ratio == 4
        return self._indexer_pool(4).get_index_k(slot)

    def _split_broadcast_units(
        self, kind: str, layer_id: int, index
    ) -> List[Tuple[Optional[torch.Tensor], torch.Tensor]]:
        # ``index`` carries the kick's broadcast subset; the synchronous
        # fallback materializes the full live set.
        pages = index if index is not None else self._split_pages_for(kind)
        assert pages is not None, (
            f"live pages for kind {kind!r} not injected; the DSV4 attention "
            "backend must call set_live_pages() before the first non-owned "
            "layer read under layer split"
        )
        is_owner = self._is_layer_owned(layer_id)

        def _empty(scratch: torch.Tensor) -> torch.Tensor:
            return torch.empty(
                (pages.numel(),) + scratch.shape[1:],
                dtype=scratch.dtype,
                device=scratch.device,
            )

        if kind == KIND_INDEX:
            # Two units: indexer K + scale, both addressed in the C4 page
            # space with the same live-page set.
            if is_owner:
                index_pool = self._indexer_pool(4)
                _, slot = self._owned_compress_slot(layer_id)
                return [
                    (index_pool.get_index_k(slot)[pages], None),
                    (index_pool.get_index_scale(slot)[pages], None),
                ]
            return [
                (None, _empty(self._split_scratch(KIND_INDEX))),
                (None, _empty(self._remote_index_scale_scratch)),
            ]
        if is_owner:
            return [(self._split_owner_buffer(kind, layer_id)[pages], None)]
        return [(None, _empty(self._split_scratch(kind)))]

    def _split_unpack_after_broadcast(
        self, kind: str, layer_id: int, index, packed: List[torch.Tensor]
    ) -> None:
        # Same subset resolution as _split_broadcast_units: the kick's pending
        # unpack must scatter into exactly the pages the broadcast delivered.
        pages = index if index is not None else self._split_pages_for(kind)
        if kind == KIND_INDEX:
            scratch_k = self._split_scratch(KIND_INDEX)
            scratch_k[pages] = packed[0]
            self._remote_index_scale_scratch[pages] = packed[1]
            return
        scratch = self._split_scratch(kind)
        scratch[pages] = packed[0]

    def get_swa_buffer(
        self, layer_id: int, loc: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        self._ensure_remote_kind(
            KIND_SWA, layer_id, layer_transfer_counter=self.layer_transfer_counter
        )
        if self._is_layer_owned(layer_id):
            return super().get_swa_buffer(layer_id)
        kv = self._split_scratch(KIND_SWA)
        if loc is not None:
            kv = kv.flatten(0, 1)[loc]
        return kv

    def get_compress_buffer(
        self,
        layer_id: int,
        from_indexer: bool = False,
        loc: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        item = self.layer_mapping[layer_id]
        if item.compress_ratio == 0:
            return None
        kind = (
            KIND_INDEX
            if from_indexer
            else (KIND_C4 if item.compress_ratio == 4 else KIND_C128)
        )
        self._ensure_remote_kind(
            kind, layer_id, layer_transfer_counter=self.layer_transfer_counter
        )
        if self._is_layer_owned(layer_id):
            if from_indexer:
                _, slot = self._owned_compress_slot(layer_id)
                kv = self._indexer_pool(item.compress_ratio).get_index_k(slot)
            else:
                compress_pool = item.compress_kv_pool
                _, slot = self._owned_compress_slot(layer_id)
                kv = compress_pool.kv_buffer[slot]
            if loc is not None:
                kv = kv.flatten(0, 1)[loc]
            return kv
        scratch = self._split_scratch(kind)
        if loc is not None:
            scratch = scratch.flatten(0, 1)[loc]
        return scratch

    def get_compress_dequant_scale_buffer(
        self,
        layer_id: int,
        from_indexer: bool,
    ) -> torch.Tensor:
        assert from_indexer, "only indexer compress pool has dequant scale"
        self._ensure_remote_kind(
            KIND_INDEX, layer_id, layer_transfer_counter=self.layer_transfer_counter
        )
        if self._is_layer_owned(layer_id):
            _, slot = self._owned_compress_slot(layer_id)
            return self._indexer_pool(
                self.compression_ratios[layer_id]
            ).get_index_scale(slot)
        return self._remote_index_scale_scratch

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        item = self.layer_mapping[layer_id]
        ratio = item.compress_ratio
        if ratio == 0:
            return self.get_swa_buffer(layer_id)
        kind = KIND_C4 if ratio == 4 else KIND_C128
        self._ensure_remote_kind(
            kind, layer_id, layer_transfer_counter=self.layer_transfer_counter
        )
        if self._is_layer_owned(layer_id):
            _, slot = self._owned_compress_slot(layer_id)
            pool = self.c4_kv_pool if ratio == 4 else self.c128_kv_pool
            return pool.kv_buffer[slot]
        return self._split_scratch(kind)

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        # V4 uses MQA / latent attention — the K buffer doubles as V.
        return self.get_key_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        buf = self.get_key_buffer(layer_id)
        return buf, buf

    def set_swa_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache: torch.Tensor,
        cp_gathered_full: bool = False,
    ) -> None:
        st = self._split_state(KIND_SWA)
        prefix_broadcast_pending = (
            st.pending_remote_broadcast and st.pending_remote_layer_id == layer_id
        )
        had_remote_copy = st.remote_layer_id == layer_id
        if st.pending_remote_layer_id == layer_id:
            self._finalize_pending_kind(KIND_SWA, set_remote_layer_id=False)
        self._invalidate_kind_cache(KIND_SWA, layer_id)
        if cp_gathered_full or prefix_broadcast_pending or had_remote_copy:
            if self._is_layer_owned(layer_id):
                super().set_swa_buffer(layer_id, loc, cache)
            else:
                self._seed_swa_scratch(loc, cache)
            st.remote_layer_id = layer_id
            return
        if not self._is_layer_owned(layer_id):
            return
        super().set_swa_buffer(layer_id, loc, cache)

    def _seed_swa_scratch(self, loc: torch.Tensor, cache: torch.Tensor) -> None:
        """Scatter the CP-gathered rows into the SWA scratch (patch-write)."""
        scratch = self._split_scratch(KIND_SWA)
        if is_npu_arch35():
            self._write_arch35_packed_kv(buf=scratch, loc=loc, cache=cache)
            return
        self._scatter_rows(scratch, loc, cache)

    def _scatter_rows(
        self, buf: torch.Tensor, loc: torch.Tensor, rows: torch.Tensor
    ) -> None:
        """Index_put ``rows`` at token slots ``loc`` of a single-layer
        page-space buffer (the scratch shares the pool's per-layer layout)."""
        buf_flat = buf.flatten(0, 1)
        rows = rows.to(buf_flat.dtype)
        if rows.ndim < buf_flat.ndim:
            # Rows may arrive with the leading token axis only (indexer scale
            # is [T]) or one axis short of the buffer (indexer K is [T, D]
            # into [T, 1, D]); expand onto the buffer's trailing dims.
            rows = rows.reshape(rows.shape[0], *buf_flat.shape[1:])
        buf_flat[loc] = rows

    def set_compress_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        kv: torch.Tensor,
        kv_scale: Optional[torch.Tensor],
        from_indexer: bool,
        cp_gathered_full: bool = False,
    ) -> None:
        kind = (
            KIND_INDEX
            if from_indexer
            else (
                KIND_C4
                if self.layer_mapping[layer_id].compress_ratio == 4
                else KIND_C128
            )
        )
        st = self._split_state(kind)
        prefix_broadcast_pending = (
            st.pending_remote_broadcast and st.pending_remote_layer_id == layer_id
        )
        had_remote_copy = st.remote_layer_id == layer_id
        if st.pending_remote_layer_id == layer_id:
            self._finalize_pending_kind(kind, set_remote_layer_id=False)
        self._invalidate_kind_cache(kind, layer_id)
        if cp_gathered_full or prefix_broadcast_pending or had_remote_copy:
            if self._is_layer_owned(layer_id):
                self._write_owned_compress_buffer(
                    layer_id, loc, kv, kv_scale, from_indexer
                )
            else:
                self._seed_compress_scratch(kind, loc, kv, kv_scale, from_indexer)
            st.remote_layer_id = layer_id
            return
        if not self._is_layer_owned(layer_id):
            return
        self._write_owned_compress_buffer(layer_id, loc, kv, kv_scale, from_indexer)

    def _write_owned_compress_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        kv: torch.Tensor,
        kv_scale: Optional[torch.Tensor],
        from_indexer: bool,
    ) -> None:
        item = self.layer_mapping[layer_id]
        ratio = item.compress_ratio
        _, slot = self._owned_compress_slot(layer_id)
        device_type = kv.device.type
        if from_indexer:
            indexer_pool = self._indexer_pool(ratio)
            if device_type == "npu":
                assert indexer_pool.has_npu_storage, (
                    "NPU index buffers not allocated — pool was init'd on CUDA?"
                )
                indexer_pool.set_index_k_scale(slot, loc, kv, kv_scale)
                return
            if kv_scale is None:
                indexer_pool.set_index_fused(slot, loc, kv)
                return
            indexer_pool.set_index_k_scale_buffer(slot, loc, kv, kv_scale)
            return
        compress_pool = self.c4_kv_pool if ratio == 4 else self.c128_kv_pool
        if device_type == "npu":
            buf = compress_pool.kv_buffer[slot]
            if is_npu_arch35():
                self._write_arch35_packed_kv(buf=buf, loc=loc, cache=kv)
                return
            buf_flat = buf.flatten(0, 1)
            kv_view = kv.to(buf_flat.dtype)
            if kv_view.ndim == buf_flat.ndim - 1:
                kv_view = kv_view.unsqueeze(1)
            buf_flat[loc] = kv_view
            return
        compress_pool.set_key_buffer_fused(slot, loc, kv)

    def _seed_compress_scratch(
        self,
        kind: str,
        loc: torch.Tensor,
        kv: torch.Tensor,
        kv_scale: Optional[torch.Tensor],
        from_indexer: bool,
    ) -> None:
        """Scatter the CP-gathered rows into the scratch (patch-write)."""
        if from_indexer:
            self._scatter_rows(self._split_scratch(KIND_INDEX), loc, kv)
            if kv_scale is not None:
                self._scatter_rows(self._remote_index_scale_scratch, loc, kv_scale)
            return
        scratch = self._split_scratch(kind)
        if is_npu_arch35():
            self._write_arch35_packed_kv(buf=scratch, loc=loc, cache=kv)
            return
        self._scatter_rows(scratch, loc, kv)
