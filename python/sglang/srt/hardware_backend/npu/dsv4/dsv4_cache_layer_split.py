"""Layer-sharded DSV4 KV pool for NPU prefill context parallelism.

``LayerSplitDSV4NPUTokenToKVPool`` splits the DeepSeek-V4 NPU KV/indexer cache
layers across context-parallel (CP) ranks so each rank only materializes the
layers it owns, cutting per-rank KV memory on PD prefill workers. When a rank
reads a layer owned by another CP rank, every rank of the CP group joins an
owner broadcast into a small per-family remote scratch buffer (the owner passes
its local buffer as the broadcast payload, so only non-owners copy).

Compress-state pools are not sharded: the compressor runs on every rank for
every layer (its internal CP all-gather requires the whole group), so every
rank holds identical per-layer state and only the KV/indexer buffers are
sharded.

Layer split is enabled only for DSV4 PD prefill workers under prefill-CP (see
``sglang.srt.layers.cp.utils.is_glm_dsa_cache_layer_split_enabled``).
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.hardware_backend.npu.dsv4.dsv4_layer_split_plan import (
    DSV4LayerShardPlan,
)
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


class LayerSplitNPUDeepSeekV4SingleKVPool(NPUDeepSeekV4SingleKVPool):
    """NPU bf16 KV pool allocating full buffers for owned layers only.

    Non-owned layers get 0-row placeholders so ``kv_buffer`` stays index
    aligned; their content is materialized on read via the parent pool's
    owner-broadcast scratch buffer.
    """

    def __init__(self, *args, layer_owned_fn: Callable[[int], bool], **kwargs):
        self._layer_owned_fn = layer_owned_fn
        super().__init__(*args, **kwargs)

    def _num_pages_for(self, local_layer_idx: int) -> int:
        full_pages = _num_pages(self.size, self.kernel_page_size)
        return full_pages if self._layer_owned_fn(local_layer_idx) else 0

    def create_buffer(self, *, num_pages: int):
        # The NPU base derives the page count from self.size; honor num_pages
        # so non-owned layers are 0-row, not narrowed full storage.
        assert self.store_dtype == torch.bfloat16, (
            "LayerSplitDSV4NPUTokenToKVPool requires a bf16 KV cache; other "
            "store dtypes fall back to a layout that cannot express 0-row "
            f"non-owned layers (got {self.store_dtype})."
        )
        kv_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.kv_cache_total_dim = kv_dim
        return torch.zeros(
            num_pages,
            self.kernel_page_size,
            1,
            kv_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.kv_buffer = [
                self.create_buffer(num_pages=self._num_pages_for(i))
                for i in range(self.layer_num)
            ]


class LayerSplitNPUDeepSeekV4IndexerPool(NPUDeepSeekV4IndexerPool):
    """NPU c4-indexer pool allocating owned layers only.

    The packed CUDA ``index_k_with_scale_buffer`` (NSA compat, unused for NPU
    reads) and the dedicated int8 K / fp16 scale buffers all follow ownership.
    """

    def __init__(self, *args, layer_owned_fn: Callable[[int], bool], **kwargs):
        self._layer_owned_fn = layer_owned_fn
        super().__init__(*args, **kwargs)

    def _owned_pages(self, local_layer_idx: int) -> int:
        full_pages = _num_pages(self.size, self._kernel_page_size)
        return full_pages if self._layer_owned_fn(local_layer_idx) else 0

    def _create_buffer(self):
        kp = self._kernel_page_size
        page_bytes = self.page_size * self.get_bytes_per_token()
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.index_k_with_scale_buffer = [
                torch.zeros(
                    self._owned_pages(i),
                    page_bytes,
                    dtype=self.index_k_with_scale_buffer_dtype,
                    device=self.device,
                )
                for i in range(self.layer_num)
            ]
            self.index_k_buffer = [
                torch.zeros(
                    self._owned_pages(i), kp, 1, self.index_head_dim,
                    dtype=torch.int8, device=self.device,
                )
                for i in range(self.layer_num)
            ]
            self.index_scale_buffer = [
                torch.zeros(
                    self._owned_pages(i), kp, 1, 1,
                    dtype=torch.float16, device=self.device,
                )
                for i in range(self.layer_num)
            ]


# Buffer families served through owner-broadcast scratch copies; index_k and
# index_scale are invalidated together by a compressor epilog write.
_REMOTE_FAMILIES = ("swa", "c4", "c128", "index_k", "index_scale")


class LayerSplitDSV4NPUTokenToKVPool(DSV4NPUTokenToKVPool):
    """DSV4 NPU KV pool that shards layers across CP ranks.

    Reads of non-owned layers are served from a per-family remote scratch
    buffer filled by an owner broadcast; writes only land on owner ranks and
    invalidate the local scratch copy so the next read re-broadcasts.
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
        plan = self._get_shard_plan()
        ids = plan.bucket_layer_ids(bucket)
        return lambda local_idx: plan.is_stage_local_owned(ids[local_idx])

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

    # ---- remote scratch + owner broadcast ----------------------------------

    def _init_remote_buffers(self) -> None:
        # One full-layer scratch per family, allocated on every rank: a rank
        # with no owned layer of a bucket still receives its broadcasts.
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            device = self.device
            kv_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
            dtype = self.swa_kv_pool.store_dtype

            def scratch(sub_pool, page_size: int, rows: int, cols: int, dt: torch.dtype):
                return torch.zeros(
                    _num_pages(sub_pool.size, page_size),
                    page_size,
                    rows,
                    cols,
                    dtype=dt,
                    device=device,
                )

            indexer = self.c4_indexer_kv_pool
            self._remote_buffers = {
                "swa": scratch(
                    self.swa_kv_pool, self.swa_kv_pool.kernel_page_size, 1, kv_dim, dtype
                ),
                "c4": scratch(
                    self.c4_kv_pool, self.c4_kv_pool.kernel_page_size, 1, kv_dim, dtype
                ),
                "c128": scratch(
                    self.c128_kv_pool,
                    self.c128_kv_pool.kernel_page_size,
                    1,
                    kv_dim,
                    dtype,
                ),
                "index_k": scratch(
                    indexer, indexer._kernel_page_size, 1, self.indexer_head_dim, torch.int8
                ),
                "index_scale": scratch(
                    indexer, indexer._kernel_page_size, 1, 1, torch.float16
                ),
            }
        # family -> layer_id of the last materialized remote copy.
        self._remote_layer_cache: Dict[str, Optional[int]] = {
            family: None for family in _REMOTE_FAMILIES
        }

    def _broadcast_read(self, family: str, layer_id: int) -> torch.Tensor:
        """Materialize ``family``'s buffer for ``layer_id`` via owner broadcast.

        All ranks issue these in the same order (a skipped call hangs the
        group) and invalidation rides the write calls every rank issues.
        """
        local = self._local_family_buffer(family, layer_id)
        remote = self._remote_buffers[family]
        if self._remote_layer_cache[family] == layer_id:
            return local if local is not None else remote

        target = local if local is not None else remote
        get_parallel().attn_cp_group.broadcast(
            target, src=self._layer_owner_rank(layer_id)
        )
        self._remote_layer_cache[family] = layer_id
        return target

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

    # ---- KV reads: owner-broadcast ------------------------------------------

    def get_swa_buffer(
        self, layer_id: int, loc: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        kv = self._broadcast_read("swa", layer_id)
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
        if from_indexer:
            assert item.compress_ratio == 4, "indexer only on c4 layers"
            kv = self._broadcast_read("index_k", layer_id)
        elif item.compress_ratio == 4:
            kv = self._broadcast_read("c4", layer_id)
        else:
            kv = self._broadcast_read("c128", layer_id)
        if loc is not None:
            kv = kv.flatten(0, 1)[loc]
        return kv

    def get_compress_dequant_scale_buffer(
        self, layer_id: int, from_indexer: bool
    ) -> torch.Tensor:
        assert from_indexer, "only indexer compress pool has dequant scale"
        return self._broadcast_read("index_scale", layer_id)

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        item = self.layer_mapping[layer_id]
        if item.compress_ratio == 0:
            return self._broadcast_read("swa", layer_id)
        if item.compress_ratio == 4:
            return self._broadcast_read("c4", layer_id)
        return self._broadcast_read("c128", layer_id)

    def get_swa_raw_buffer(self, layer_id: int) -> torch.Tensor:
        return self._broadcast_read("swa", layer_id)

    # ---- KV writes: owned-only, invalidate remote copies --------------------

    def set_swa_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache: torch.Tensor,
    ) -> None:
        self._invalidate_family(("swa",), layer_id)
        if not self._is_layer_owned(layer_id):
            return
        super().set_swa_buffer(layer_id, loc, cache)

    def set_compress_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        kv: torch.Tensor,
        kv_scale: Optional[torch.Tensor],
        from_indexer: bool,
    ) -> None:
        self._invalidate_family(("index_k", "index_scale"), layer_id)
        ratio = self.layer_mapping[layer_id].compress_ratio
        if ratio == 4:
            self._invalidate_family(("c4",), layer_id)
        elif ratio == 128:
            self._invalidate_family(("c128",), layer_id)
        if not self._is_layer_owned(layer_id):
            return
        super().set_compress_buffer(layer_id, loc, kv, kv_scale, from_indexer)

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
            # requires pp_size == 1, so stage-local ids coincide.
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
