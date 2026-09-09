"""Layer-sharded DSV4 KV pool for NPU prefill context parallelism.

``LayerSplitDSV4NPUTokenToKVPool`` splits the DeepSeek-V4 NPU KV/indexer cache
layers across context-parallel (CP) ranks so each rank only materializes the
layers it owns. A rank reading a layer owned by another CP rank pulls the
forward's selected pages via one owner broadcast into a per-family remote
scratch; the zbal process-group backend, whose broadcast is unproven, falls
back to chunked all-gather.

Compress-state pools are not sharded: the compressor runs on every rank for
every layer, so every rank holds identical per-layer state.

Enabled only for DSV4 PD prefill workers under prefill-CP (see
``sglang.srt.layers.cp.utils.is_glm_dsa_cache_layer_split_enabled``).
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.dsv4.dsv4_layer_split_plan import (
    DSV4LayerShardPlan,
)
from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
from sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool import (
    DeepSeekV4SingleKVPool,
    DSV4NPUTokenToKVPool,
    NPUDeepSeekV4IndexerPool,
    NPUDeepSeekV4SingleKVPool,
)
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)


def _num_pages(size: int, page_size: int) -> int:
    """Physical pages covering ``size`` tokens at ``page_size`` tokens/page."""
    return (size + page_size + 1) // page_size


# Collectives stage <=N-byte chunks through a fresh staging tensor; pool-resident
# multi-MB operands (ZBAL/VMM) are the operand class that corrupts on this stack.
_LS_CHUNK_BYTES = envs.SGLANG_DSV4_LS_CHUNK_BYTES.get()
assert _LS_CHUNK_BYTES > 0, "SGLANG_DSV4_LS_CHUNK_BYTES must be positive"


class LayerSplitNPUDeepSeekV4SingleKVPool(NPUDeepSeekV4SingleKVPool):
    """NPU bf16 KV pool allocating full buffers for owned layers only.

    Non-owned layers get 0-row placeholders so ``kv_buffer`` stays index
    aligned; their content is materialized on read via the parent pool's
    owner all-gather scratch buffer.
    """

    def __init__(self, *args, layer_owned_fn: Callable[[int], bool], **kwargs):
        self._layer_owned_fn = layer_owned_fn
        super().__init__(*args, **kwargs)
        # 0-row non-owned layers are only expressible in the PA_ND bf16 layout.
        assert self.store_dtype == torch.bfloat16, (
            f"layer split requires a bf16 KV cache, got {self.store_dtype}"
        )

    def _num_pages_for(self, local_layer_idx: int) -> int:
        full_pages = _num_pages(self.size, self.kernel_page_size)
        return full_pages if self._layer_owned_fn(local_layer_idx) else 0


class LayerSplitNPUDeepSeekV4IndexerPool(NPUDeepSeekV4IndexerPool):
    """NPU c4-indexer pool allocating owned layers only (packed CUDA buffer,
    int8 K and fp16 scale all follow ownership)."""

    def __init__(self, *args, layer_owned_fn: Callable[[int], bool], **kwargs):
        self._layer_owned_fn = layer_owned_fn
        super().__init__(*args, **kwargs)

    def _num_pages_for(self, local_layer_idx: int) -> int:
        full_pages = _num_pages(self.size, self._kernel_page_size)
        return full_pages if self._layer_owned_fn(local_layer_idx) else 0


# Buffer families served through owner all-gather scratch copies. index_k and
# index_scale exist only on c4 layers.
_REMOTE_FAMILIES = ("swa", "c4", "c128", "index_k", "index_scale")


class LayerSplitDSV4NPUTokenToKVPool(DSV4NPUTokenToKVPool):
    """DSV4 NPU KV pool that shards layers across CP ranks.

    Reads of non-owned layers are served from a per-family remote scratch
    buffer filled by an owner broadcast (chunked all-gather under zbal);
    writes only land on owner ranks and invalidate the local scratch copy so
    the next read re-gathers.
    """

    def __init__(self, *args, layer_shard_rank: int, layer_shard_size: int, **kwargs):
        assert (
            layer_shard_rank is not None and layer_shard_size > 1
        ), "LayerSplitDSV4NPUTokenToKVPool requires layer_shard_size > 1"
        self.layer_shard_rank = layer_shard_rank
        self.layer_shard_size = layer_shard_size
        self.layer_shard_enabled = True
        # Built on the first _make_kv_pool call inside super().__init__: the
        # plan needs layer_num / ratios / stage range the base sets before it.
        self._shard_plan: Optional[DSV4LayerShardPlan] = None
        super().__init__(*args, **kwargs)
        assert (
            not self._unified_kv
        ), "Layer split does not support the unified-KV layout yet"
        self._init_remote_buffers()
        plan = self._get_shard_plan()
        # Global (absolute) layer range owned by this rank, read by the PD
        # bootstrap (disaggregation/prefill.py) to advertise the shard window.
        self.layer_shard_start = plan.shard_start
        self.layer_shard_end = plan.shard_end
        logger.info(
            "DSV4 layer shard plan (continuous): layer_num=%d, shard_size=%d, "
            "rank=%d, global=[%d,%d), owned c4 range=%s, owned c128 range=%s, "
            "partitions=%s",
            self.layer_num,
            self.layer_shard_size,
            self.layer_shard_rank,
            plan.shard_start,
            plan.shard_end,
            plan.owned_bucket_range("c4"),
            plan.owned_bucket_range("c128"),
            plan.partition_summary(),
        )

    # ---- ownership plan ----------------------------------------------------

    def _get_shard_plan(self) -> DSV4LayerShardPlan:
        if self._shard_plan is None:
            self._shard_plan = DSV4LayerShardPlan(
                rank=self.layer_shard_rank,
                shard_size=self.layer_shard_size,
                num_layers=self.layer_num,
                stage_start=self._stage_start,
                ratios=self.compression_ratios[self._stage_start : self._stage_end],
            )
        return self._shard_plan

    def _is_layer_owned(self, layer_id: int) -> bool:
        return self._get_shard_plan().is_layer_owned(layer_id)

    def _layer_owner_rank(self, layer_id: int) -> int:
        return self._get_shard_plan().owner_rank(layer_id)

    def _owned_fn_for_bucket(self, bucket: str) -> Callable[[int], bool]:
        # Bucket ids are monotonic in the stage-local id, so ownership is a range.
        plan = self._get_shard_plan()
        start, end = plan.owned_bucket_range(bucket)
        return lambda local_idx: start <= local_idx < end

    # ---- sub-pool factories ------------------------------------------------

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
    ) -> LayerSplitNPUDeepSeekV4SingleKVPool:
        assert cls is DeepSeekV4SingleKVPool, (
            "enable_hisparse is incompatible with --enable-dsa-cache-layer-split "
            f"(got c4 pool class {cls.__name__})."
        )
        # Full/SWA use the global page size, C4 its native page, C128 its own;
        # mirrors the NPU base _make_kv_pool.
        if page_size * 4 == global_page_size:
            bucket, kernel_page_size = "c4", page_size
        elif page_size * 128 == global_page_size:
            bucket, kernel_page_size = "c128", self.c128_page_size
        else:
            bucket, kernel_page_size = "swa", global_page_size
        return LayerSplitNPUDeepSeekV4SingleKVPool(
            size,
            page_size,
            dtype,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            kernel_page_size=kernel_page_size,
            layer_owned_fn=self._owned_fn_for_bucket(bucket),
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
    ) -> LayerSplitNPUDeepSeekV4IndexerPool:
        # Indexer shares C4 addresses and therefore uses the same native page.
        return LayerSplitNPUDeepSeekV4IndexerPool(
            size,
            page_size,
            dtype,
            index_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            kernel_page_size=page_size,
            layer_owned_fn=self._owned_fn_for_bucket("c4"),
        )

    # ---- remote scratch + owner all-gather --------------------------------

    def _init_remote_buffers(self) -> None:
        # One full-layer scratch per family, allocated on every rank: a rank
        # with no owned layer of a bucket still receives its gathers. The
        # broadcast staging has the same per-family shape and receives the
        # compacted selected pages; the all-gather fallback instead stages
        # through the shared 1MB tensor below.
        group = get_parallel().attn_cp_group
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            device = self.device

            def scratch(buffers: List[torch.Tensor], size: int, page_size: int) -> torch.Tensor:
                # Shape/dtype mirror the sub-pool's real per-layer buffer, so
                # A5 packed/indexer formats are picked up without special
                # cases. A 0-row non-owned placeholder keeps the full shape.
                buf = buffers[0]
                return torch.zeros(
                    _num_pages(size, page_size),
                    buf.shape[1],
                    buf.shape[2],
                    buf.shape[3],
                    dtype=buf.dtype,
                    device=device,
                )

            indexer = self.c4_indexer_kv_pool
            self._remote_buffers = {
                "swa": scratch(
                    self.swa_kv_pool.kv_buffer,
                    self.swa_kv_pool.size,
                    self.swa_kv_pool.kernel_page_size,
                ),
                "c4": scratch(
                    self.c4_kv_pool.kv_buffer,
                    self.c4_kv_pool.size,
                    self.c4_kv_pool.kernel_page_size,
                ),
                "c128": scratch(
                    self.c128_kv_pool.kv_buffer,
                    self.c128_kv_pool.size,
                    self.c128_kv_pool.kernel_page_size,
                ),
                "index_k": scratch(
                    indexer.index_k_buffer,
                    indexer.size,
                    indexer._kernel_page_size,
                ),
                "index_scale": scratch(
                    indexer.index_scale_buffer,
                    indexer.size,
                    indexer._kernel_page_size,
                ),
            }
            self._ls_staging = {
                family: torch.empty_like(buf) for family, buf in self._remote_buffers.items()
            }
        # Broadcast is bitwise-verified correct on hccl; the zbal interposed
        # process group falls back to chunked all-gather.
        self._use_broadcast = (
            torch.distributed.get_backend(group.device_group) != "zbal"
        )
        # Async reads: one in-flight launch per family on a side stream, keyed
        # (layer_id, event, work, mode, selected); consumed before the read.
        self._use_async = self._use_broadcast and envs.SGLANG_DSV4_LS_ASYNC_READ.get()
        self._async_slots: Dict[str, Optional[tuple]] = {
            family: None for family in _REMOTE_FAMILIES
        }
        self._comm_streams: Dict[str, torch.npu.Stream] = {}
        # family -> layer_id of the last materialized remote copy; reset per
        # forward by begin_forward_staging.
        self._remote_layer_cache: Dict[str, Optional[int]] = {
            family: None for family in _REMOTE_FAMILIES
        }
        # Active-page union of the current forward per family, set by
        # begin_forward_staging; reads fall back to whole-layer gathers before it.
        self._staging_pages: Dict[str, torch.Tensor] = {}
        # family -> page-id -> staging row for the current forward's compact
        # remote copies; None without a compact plan.
        self._row_maps: Dict[str, Optional[torch.Tensor]] = {
            family: None for family in _REMOTE_FAMILIES
        }
        # Shared chunk-staging operand; int8 is the smallest family dtype, so
        # byte capacity == element capacity.
        self._staging = torch.zeros(
            _LS_CHUNK_BYTES, dtype=torch.int8, device=self.device
        )

    # index_k / index_scale share the c4 page-id space and the c4 plan.

    def begin_forward_staging(self, tables: Dict[str, Optional[torch.Tensor]]) -> None:
        """Record the forward's active pages per family as a CP-group union;
        compact remote copies keep the pages in staging rows."""
        group = get_parallel().attn_cp_group
        self._row_maps = {family: None for family in _REMOTE_FAMILIES}
        for family, table in tables.items():
            remote = self._remote_buffers.get(family)
            if remote is None or table is None or table.numel() == 0:
                self._staging_pages.pop(family, None)
                continue
            flat = table.reshape(-1).to(torch.long)
            in_range = (flat >= 0) & (flat < remote.shape[0])
            mask = torch.zeros(remote.shape[0], dtype=torch.int32, device=flat.device)
            mask.index_add_(
                0,
                flat[in_range],
                torch.ones(
                    int(in_range.sum()), dtype=torch.int32, device=flat.device
                ),
            )
            torch.distributed.all_reduce(mask, group=group.device_group)
            selected = torch.nonzero(mask, as_tuple=False).flatten()
            self._staging_pages[family] = selected
            row = torch.full(
                (remote.shape[0],), -1, dtype=torch.int32, device=selected.device
            )
            row[selected] = torch.arange(
                selected.numel(), dtype=torch.int32, device=selected.device
            )
            self._row_maps[family] = row
        # index_k / index_scale share the c4 page-id space: mirror the c4
        # plan under their own names so every family resolves directly.
        if "c4" in self._staging_pages:
            self._staging_pages["index_k"] = self._staging_pages["c4"]
            self._staging_pages["index_scale"] = self._staging_pages["c4"]
            self._row_maps["index_k"] = self._row_maps["c4"]
            self._row_maps["index_scale"] = self._row_maps["c4"]
        # A previous forward's remote-read cache would suppress this
        # forward's transfers.
        self._remote_layer_cache = {family: None for family in _REMOTE_FAMILIES}

    def page_table_for_read(
        self, family: str, layer_id: int, table: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Table matching the buffer get_*_buffer returned: original page ids
        for owned layers and whole-layer copies, staging rows for compact ones."""
        if (
            table is None
            or self._is_layer_owned(layer_id)
            or not self._is_compact(family)
        ):
            return table
        row_map = self._row_maps.get(family)
        if row_map is None:
            return table
        pages = row_map.shape[0]
        flat = table.reshape(-1)
        rows = row_map[flat.clamp(0, pages - 1)].reshape(table.shape)
        in_plan = ((flat >= 0) & (flat < pages)).reshape(table.shape)
        return torch.where(in_plan, rows, table).to(table.dtype)

    def _read_layer_buffer(self, family: str, layer_id: int) -> torch.Tensor:
        """This rank's buffer for ``family``/``layer_id``, remote layers included.
        The scratch holds one layer: consume before the next read, in lockstep."""
        local = self._local_family_buffer(family, layer_id)
        remote = self._remote_buffers[family]
        target = self._compact_read_target(family, local, remote)
        if self._remote_layer_cache[family] == layer_id:
            return target

        # A pending launch for this family is consumed here; when it was for
        # this layer the payload is already in place and no sync transfer runs.
        if self._consume_async(family) == layer_id:
            self._remote_layer_cache[family] = layer_id
            return target

        # attn_cp_group must have no other submitters while layer split is on:
        # HCCL pairs this group's collectives by submission order only.
        group = get_parallel().attn_cp_group
        selected = self._staging_pages.get(family)
        if self._use_broadcast:
            if self._is_compact(family):
                self._broadcast_selected(
                    family, local, remote, selected, layer_id, group
                )
            else:
                self._broadcast_whole_layer(local, remote, layer_id, group)
        else:
            if self._is_compact(family):
                self._launch_allgather_selected(
                    family, local, remote, selected, layer_id, group
                )
            else:
                self._read_via_allgather_chunks(family, layer_id, local, remote, group)
        self._remote_layer_cache[family] = layer_id
        return target

    def _is_compact(self, family: str) -> bool:
        """Whether this family's current transfer is selected-pages compact."""
        selected = self._staging_pages.get(family)
        return (
            selected is not None
            and selected.numel() < self._remote_buffers[family].shape[0]
        )

    def _compact_read_target(self, family: str, local, remote) -> torch.Tensor:
        """Buffer a get_*_buffer caller must read: owned layers read locally;
        compact remote copies stay in the staging rows they were received in."""
        if local is not None:
            return local
        if self._is_compact(family):
            return self._ls_staging[family][: int(self._staging_pages[family].numel())]
        return remote

    def _launch_async_read(self, family: str, layer_id: int) -> None:
        """Launch this family/layer transfer on the comm stream right after
        the owner-side write; the read consumes the recorded event."""
        if not self._use_async:
            return
        self._consume_async(family)
        local = self._local_family_buffer(family, layer_id)
        remote = self._remote_buffers[family]
        selected = self._staging_pages.get(family)
        compact = self._is_compact(family)
        group = get_parallel().attn_cp_group
        src = group.ranks[self._layer_owner_rank(layer_id)]
        stream = self._comm_streams.get(family)
        if stream is None:
            stream = self._comm_streams[family] = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        work = None
        with torch.npu.stream(stream):
            if compact:
                staging = self._ls_staging[family]
                k = int(selected.numel())
                if local is not None:
                    staging[:k].copy_(local.index_select(0, selected))
                work = torch.distributed.broadcast(
                    staging[:k], src=src, group=group.device_group, async_op=True
                )
            else:
                tensor = local if local is not None else remote
                work = torch.distributed.broadcast(
                    tensor, src=src, group=group.device_group, async_op=True
                )
            event = torch.npu.Event()
            event.record(torch.npu.current_stream())
        self._async_slots[family] = (layer_id, event, work)

    def _consume_async(self, family: str) -> Optional[int]:
        """Wait out the family's pending transfer; the compact payload already
        sits in the staging rows consumers read via page_table_for_read."""
        slot = self._async_slots.pop(family, None)
        if slot is None:
            return None
        layer_id, event, _work = slot
        torch.npu.current_stream().wait_event(event)
        return layer_id

    def _broadcast_selected(
        self,
        family: str,
        local: Optional[torch.Tensor],
        remote: torch.Tensor,
        selected: torch.Tensor,
        layer_id: int,
        group,
    ) -> None:
        """Owner compacts the selected pages into the staging buffer; one
        broadcast delivers them and consumers read the compact rows directly."""
        staging = self._ls_staging[family]
        k = int(selected.numel())
        if local is not None:
            staging[:k].copy_(local.index_select(0, selected))
        torch.distributed.broadcast(
            staging[:k], src=group.ranks[self._layer_owner_rank(layer_id)],
            group=group.device_group,
        )

    def _broadcast_whole_layer(
        self,
        local: Optional[torch.Tensor],
        remote: torch.Tensor,
        layer_id: int,
        group,
    ) -> None:
        """Owner's whole layer buffer broadcast into the receiver's scratch."""
        tensor = local if local is not None else remote
        torch.distributed.broadcast(
            tensor, src=group.ranks[self._layer_owner_rank(layer_id)],
            group=group.device_group,
        )

    def _read_via_allgather_chunks(
        self,
        family: str,
        layer_id: int,
        local: Optional[torch.Tensor],
        remote: torch.Tensor,
        group,
    ) -> None:
        """Chunked all-gather fallback: the non-owner copies the owner's slot
        per chunk into the scratch."""
        flat_src = local.reshape(-1) if local is not None else None
        flat_dst = remote.reshape(-1)
        elem = flat_dst.element_size()
        total = flat_dst.numel()
        step = max(1, _LS_CHUNK_BYTES // elem)
        owner_slot = self._layer_owner_rank(layer_id)
        world = group.world_size
        for start in range(0, total, step):
            end = min(start + step, total)
            width = end - start
            send = self._staging[: width * elem].view(flat_dst.dtype)
            if flat_src is not None:
                send.copy_(flat_src[start:end])
            gathered = torch.empty(
                (world, width), dtype=flat_dst.dtype, device=flat_dst.device
            )
            group.all_gather_into_tensor(gathered, send)
            if flat_src is None:
                flat_dst[start:end].copy_(gathered[owner_slot])

    def _launch_allgather_selected(
        self,
        family: str,
        local: Optional[torch.Tensor],
        remote: torch.Tensor,
        selected: torch.Tensor,
        layer_id: int,
        group,
    ) -> None:
        """All-gather the selected active pages into the staging rows consumers
        read via page_table_for_read."""
        pages = remote.shape[0]
        row_elems = remote[0].numel()
        elem = remote.element_size()
        step_pages = max(1, _LS_CHUNK_BYTES // (row_elems * elem))
        owner_slot = self._layer_owner_rank(layer_id)
        flat_src = None if local is None else local.reshape(pages, -1)
        dst = self._ls_staging[family].reshape(pages, -1)
        offset = 0
        for sel in selected.split(step_pages):
            n = int(sel.numel())
            chunk = dst[offset : offset + n]
            if flat_src is not None:
                chunk.copy_(flat_src.index_select(0, sel))
            gathered = torch.empty(
                (group.world_size, n, row_elems),
                dtype=remote.dtype,
                device=remote.device,
            )
            group.all_gather_into_tensor(gathered, chunk)
            if flat_src is None:
                chunk.copy_(gathered[owner_slot])
            offset += n

    def _invalidate_family(self, families: Tuple[str, ...], layer_id: int) -> None:
        for family in families:
            if self._remote_layer_cache[family] == layer_id:
                self._remote_layer_cache[family] = None

    def _local_family_buffer(
        self, family: str, layer_id: int
    ) -> Optional[torch.Tensor]:
        """This rank's own buffer for ``family``/``layer_id``, or None when the
        layer belongs to another CP rank."""
        if not self._is_layer_owned(layer_id):
            return None
        if family == "swa":
            return self.swa_kv_pool.kv_buffer[layer_id]
        item = self.layer_mapping[layer_id]
        if family == "c4":
            assert item.compress_ratio == 4
            return self.c4_kv_pool.kv_buffer[item.compress_layer_id]
        if family == "c128":
            assert item.compress_ratio == 128
            return self.c128_kv_pool.kv_buffer[item.compress_layer_id]
        if family == "index_k":
            assert item.compress_ratio == 4
            return self.c4_indexer_kv_pool.index_k_buffer[item.compress_layer_id]
        assert family == "index_scale" and item.compress_ratio == 4
        return self.c4_indexer_kv_pool.index_scale_buffer[item.compress_layer_id]

    # ---- KV reads: owner all-gather -----------------------------------------

    def get_swa_buffer(
        self, layer_id: int, loc: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        kv = self._read_layer_buffer("swa", layer_id)
        if loc is not None:
            if not self._is_layer_owned(layer_id) and self._is_compact("swa"):
                loc = self._remap_flat_loc("swa", kv, loc)
            kv = kv.flatten(0, 1)[loc]
        return kv

    def _remap_flat_loc(
        self, family: str, kv: torch.Tensor, loc: torch.Tensor
    ) -> torch.Tensor:
        # loc is page-major over (pages, page_size); translate page ids to rows.
        page_size = kv.shape[1]
        row_map = self._row_maps[family]
        page = loc // page_size
        rows = row_map[page.clamp(0, row_map.shape[0] - 1)]
        return torch.where(
            rows >= 0, rows * page_size + loc % page_size, loc
        ).to(loc.dtype)

    def get_compress_buffer(
        self,
        layer_id: int,
        from_indexer: bool = False,
        loc: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        item = self.layer_mapping[layer_id]
        if item.compress_ratio == 0:
            return None
        if from_indexer:
            assert item.compress_ratio == 4, "indexer only on c4 layers"
            family, kv = "index_k", self._read_layer_buffer("index_k", layer_id)
        elif item.compress_ratio == 4:
            family, kv = "c4", self._read_layer_buffer("c4", layer_id)
        else:
            family, kv = "c128", self._read_layer_buffer("c128", layer_id)
        if loc is not None:
            if not self._is_layer_owned(layer_id) and self._is_compact(family):
                loc = self._remap_flat_loc(family, kv, loc)
            kv = kv.flatten(0, 1)[loc]
        return kv

    def get_compress_dequant_scale_buffer(
        self, layer_id: int, from_indexer: bool
    ) -> torch.Tensor:
        assert from_indexer, "only indexer compress pool has dequant scale"
        return self._read_layer_buffer("index_scale", layer_id)

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        item = self.layer_mapping[layer_id]
        if item.compress_ratio == 0:
            return self._read_layer_buffer("swa", layer_id)
        if item.compress_ratio == 4:
            return self._read_layer_buffer("c4", layer_id)
        return self._read_layer_buffer("c128", layer_id)

    def get_swa_raw_buffer(self, layer_id: int) -> torch.Tensor:
        return self._read_layer_buffer("swa", layer_id)

    # ---- KV writes: owned-only, invalidate remote copies --------------------

    def set_swa_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache: torch.Tensor,
    ) -> None:
        self._invalidate_family(("swa",), layer_id)
        if self._is_layer_owned(layer_id):
            super().set_swa_buffer(layer_id, loc, cache)
        self._launch_async_read("swa", layer_id)

    def set_compress_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        kv: torch.Tensor,
        kv_scale: Optional[torch.Tensor],
        from_indexer: bool,
    ) -> None:
        ratio = self.layer_mapping[layer_id].compress_ratio
        # Launch only the families this call wrote; the c4 KV broadcast belongs
        # to the attention-side write, not the indexer's.
        if from_indexer:
            families = ("index_k", "index_scale")
        elif ratio == 4:
            families = ("c4",)
        elif ratio == 128:
            families = ("c128",)
        else:
            families = ()
        if families:
            self._invalidate_family(families, layer_id)
        if self._is_layer_owned(layer_id):
            super().set_compress_buffer(layer_id, loc, kv, kv_scale, from_indexer)
        for family in families:
            self._launch_async_read(family, layer_id)

    # ---- PD transfer: report owned layers only ------------------------------

    def get_contiguous_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        """Main PD buffers: [c4 KV, index K, index scale], each section sliced
        to this rank's owned c4 layers."""
        buffers = (
            self._owned_bucket_buffers(self.c4_kv_pool.kv_buffer, "c4")
            + self._owned_bucket_buffers(self.c4_indexer_kv_pool.index_k_buffer, "c4")
            + self._owned_bucket_buffers(
                self.c4_indexer_kv_pool.index_scale_buffer, "c4"
            )
        )
        return (
            [buf.data_ptr() for buf in buffers],
            [buf.nbytes for buf in buffers],
            [buf[0].nbytes for buf in buffers],
        )

    def get_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        """SWA component (owned only): SWA KV + c4 attn/indexer states."""
        plan = self._get_shard_plan()
        swa_start, swa_end = plan.owned_stage_local_range()
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []
        for buf in self.swa_kv_pool.kv_buffer[swa_start:swa_end]:
            data_ptrs.append(buf.data_ptr())
            data_lens.append(buf.nbytes)
            item_lens.append(buf[0].nbytes)
        for pools in (self.compress_state_pools, self.indexer_compress_state_pools):
            # compress_state_pools is absolute-layer-indexed; layer split
            # requires pp_size == 1, so stage-local ids coincide. On A5
            # (CYCLE cache_mode) C4 state ships as its own DSV4_C4_STATE
            # component instead.
            if is_npu_arch35():
                continue
            for idx in plan.owned_stage_local_ids("c4"):
                state = pools[idx].kv_score_buffer.kv_score
                data_ptrs.append(state.data_ptr())
                data_lens.append(state.nbytes)
                item_lens.append(state[0].nbytes * pools[idx].ring_size)
        return data_ptrs, data_lens, item_lens

    def get_c128_kv_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        buffers = self._owned_bucket_buffers(self.c128_kv_pool.kv_buffer, "c128")
        return (
            [buf.data_ptr() for buf in buffers],
            [buf.nbytes for buf in buffers],
            [buf[0].nbytes for buf in buffers],
        )

    def get_c128_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []
        for idx in self._get_shard_plan().owned_stage_local_ids("c128"):
            pool = self.compress_state_pools[idx]
            state = pool.kv_score_buffer.kv_score
            data_ptrs.append(state.data_ptr())
            data_lens.append(state.nbytes)
            item_lens.append(state[0].nbytes * pool.ring_size)
        return data_ptrs, data_lens, item_lens

    def _owned_bucket_buffers(
        self, buffers: List[torch.Tensor], bucket: str
    ) -> List[torch.Tensor]:
        start, end = self._get_shard_plan().owned_bucket_range(bucket)
        return buffers[start:end]

    # ---- PD transfer: layer ids parallel to the owned buf lists -------------

    def _owned_global_bucket_ids(self, bucket: str) -> List[int]:
        plan = self._get_shard_plan()
        return [self._stage_start + i for i in plan.owned_stage_local_ids(bucket)]

    def get_kv_layer_ids(self) -> List[int]:
        return self._owned_global_bucket_ids("c4") * 3

    def get_state_layer_ids(self) -> List[int]:
        plan = self._get_shard_plan()
        return (
            list(range(plan.shard_start, plan.shard_end))
            + self._owned_global_bucket_ids("c4") * 2
        )

    def get_c128_layer_ids(self) -> List[int]:
        return self._owned_global_bucket_ids("c128")
