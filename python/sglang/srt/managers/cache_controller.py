from __future__ import annotations

"""
Copyright 2023-2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import math
import threading
import time
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING, List, NamedTuple, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import (
    STORAGE_BATCH_SIZE,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolName,
    PoolTransfer,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool_host import HostKVCache

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    MLATokenToKVPool,
    MLATokenToKVPoolFP4,
)
from sglang.srt.utils import get_device_module, is_cuda

logger = logging.getLogger(__name__)

device_module = get_device_module()


# Storage backends compatible with MLA/NSA host-memory dedup. The dummy
# (non-rank-0) host pool has no local KV buffer, so dedup only engages for
# backends that never construct against / register / read that buffer.
# "file" (HiCacheFile) is built from config alone and stages via flat copies,
# so it coexists; the RDMA/registered backends (mooncake/eic/simm/hf3fs/nixl/
# aibrix) touch the host buffer at construct or register time, so dedup stays
# off for them and every rank keeps a full host pool (pre-dedup behavior).
_DEDUP_COMPATIBLE_STORAGE = frozenset({None, "", "file"})


def storage_supports_host_dedup(storage_backend) -> bool:
    """Whether MLA/NSA host-memory dedup can engage with this storage backend."""
    return storage_backend in _DEDUP_COMPATIBLE_STORAGE


class LayerLoadingEvent:
    def __init__(self, num_layers: int):
        self._num_layers = num_layers
        self.load_events = [device_module.Event() for _ in range(num_layers)]
        self.start_event = device_module.Event()  # start event on controller stream

    def complete(self, layer_index: int):
        assert 0 <= layer_index < self._num_layers
        self.load_events[layer_index].record()

    def wait(self, layer_index: int):
        device_module.current_stream().wait_event(self.load_events[layer_index])

    @property
    def finish_event(self):
        return self.load_events[-1]


class LayerDoneCounter:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        # extra producer and consumer counters for overlap mode
        self.num_counters = 3
        self.events = [LayerLoadingEvent(num_layers) for _ in range(self.num_counters)]
        self.producer_index = -1
        self.consumer_index = -1

    def update_producer(self):
        self.producer_index = (self.producer_index + 1) % self.num_counters
        assert self.events[
            self.producer_index
        ].finish_event.query(), (
            "Producer finish event should be ready before being reused."
        )
        return self.producer_index

    def set_consumer(self, index: int):
        self.consumer_index = index

    def wait_until(self, threshold: int):
        if self.consumer_index < 0:
            return
        self.events[self.consumer_index].wait(threshold)

    def reset(self):
        self.producer_index = -1
        self.consumer_index = -1


class CacheOperation:

    counter = 0

    def __init__(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        node_id: int,
        priority: Optional[int] = None,
    ):
        self.host_indices = host_indices
        self.device_indices = device_indices
        self.node_ids = [node_id]
        self.data = None

        self.id = CacheOperation.counter
        CacheOperation.counter += 1
        # default priority is the order of creation
        self.priority = priority if priority is not None else self.id

    @staticmethod
    def merge_ops(ops: List[CacheOperation]) -> CacheOperation:
        assert len(ops) > 0
        if len(ops) == 1:
            return ops[0]

        host_indices = torch.cat([op.host_indices for op in ops])
        device_indices = torch.cat([op.device_indices for op in ops])
        node_ids = []
        priority = min(op.priority for op in ops)
        for op in ops:
            node_ids.extend(op.node_ids)
        merged_op = CacheOperation(host_indices, device_indices, -1, priority)
        merged_op.node_ids = node_ids
        return merged_op

    def __lt__(self, other: CacheOperation):
        return self.priority < other.priority


class HiCacheAck(NamedTuple):
    start_event: device_module.Event
    finish_event: device_module.Event
    node_ids: List[int]


class TransferBuffer:
    """
    Overlapping buffer preparation and transfer operations to improve throughput.
    """

    def __init__(self, stop_event, buffer_count: int = 3) -> None:
        self.stop_event = stop_event
        self.buffers = Queue(maxsize=buffer_count)

    def full(self) -> bool:
        return self.buffers.full()

    def empty(self) -> bool:
        return self.buffers.empty()

    def put(self, item, block=True, timeout=1) -> None:
        while not self.stop_event.is_set():
            try:
                self.buffers.put(item, block=block, timeout=timeout)
                break
            except Full:
                if not block:
                    break
                continue
            except Exception as e:
                logger.error(e)

    def get(self, block=True, timeout=1) -> Optional[CacheOperation]:
        try:
            return self.buffers.get(block=block, timeout=timeout)
        except Empty:
            return None
        except Exception as e:
            logger.error(e)

    def clear(self):
        self.buffers.queue.clear()


class StorageOperation:
    counter = 0

    def __init__(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        self.host_indices = host_indices
        self.token_ids = token_ids
        self.last_hash = last_hash
        self.completed_tokens = 0
        self.hash_value = hash_value if hash_value is not None else []
        self.prefix_keys = prefix_keys

        self.id = StorageOperation.counter
        StorageOperation.counter += 1

    def __lt__(self, other: "StorageOperation"):
        return self.id < other.id


class PrefetchOperation(StorageOperation):
    def __init__(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        self.request_id = request_id

        self._lock = threading.Lock()
        self._terminated_flag = False
        self.start_time = time.monotonic()

        super().__init__(host_indices, token_ids, last_hash, prefix_keys=prefix_keys)

    def increment(self, num_tokens: int):
        with self._lock:
            if self._terminated_flag:
                return False
            self.completed_tokens += num_tokens
            return True

    def mark_terminate(self):
        with self._lock:
            self._terminated_flag = True

    def is_terminated(self) -> bool:
        return self._terminated_flag


class HiCacheController:

    def __init__(
        self,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        mem_pool_host: HostKVCache,
        page_size: int,
        tp_group: torch.distributed.ProcessGroup,
        load_cache_event: threading.Event,
        attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
        attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
        write_policy: str = "write_through_selective",
        io_backend: str = "",
        storage_backend: Optional[str] = None,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        pp_rank: int = 0,
        pp_size: int = 1,
        enable_storage_metrics: bool = False,
        mla_broadcast_state: Optional[dict] = None,
    ):
        self.tp_group = tp_group
        self.attn_cp_group = attn_cp_group
        self.attn_tp_group = attn_tp_group
        self.prefetch_sync_groups: List[torch.distributed.ProcessGroup] = []
        # Optional prebuilt rendezvous state from HiRadixCache (created BEFORE
        # the slow rank-0 host KV alloc, see prebuild_mla_broadcast_state for
        # the 10-min NCCL watchdog story). Consumed in this __init__ for the
        # MLA broadcast group, and in _create_prefetch_sync_groups for the
        # gloo prefetch groups. Cleared after consumption so any future
        # runtime detach→re-attach builds fresh.
        self._prebuilt_prefetch_sync_groups: Optional[
            List[torch.distributed.ProcessGroup]
        ] = (
            mla_broadcast_state.get("prefetch_sync_groups")
            if mla_broadcast_state is not None
            else None
        )
        self.mem_pool_device_allocator = token_to_kv_pool_allocator
        mem_pool_device = token_to_kv_pool_allocator.get_kvcache()
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        if isinstance(mem_pool_device, HybridLinearKVPool):
            mem_pool_device = mem_pool_device.full_kv_pool
        self.mem_pool_device = mem_pool_device
        self.mem_pool_host = mem_pool_host
        self.write_policy = write_policy
        self.page_size = page_size
        self.io_backend = io_backend
        self.enable_storage = False
        self.storage_backend = None
        self.storage_backend_type = None
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.enable_storage_metrics = enable_storage_metrics

        # Draft KV pool support (best-effort piggyback on target L2/L3 ops).
        self.has_draft = False
        self.mem_pool_device_draft = None
        self.mem_pool_host_draft = None
        self.draft_page_get_func = None
        self.draft_page_set_func = None

        # Default storage page IO functions (may be overridden by attach).
        self.page_get_func = self._generic_page_get
        self.page_set_func = self._generic_page_set

        # Dedicated stop event for storage background threads (prefetch/backup).
        # NOTE: Do NOT reuse `self.stop_event` here since it also guards core HiCache
        # transfer buffers (CPU<->GPU). We want to allow runtime attach/detach of
        # storage without stopping the whole controller.
        self.storage_stop_event = threading.Event()

        self.device = self.mem_pool_device.device
        self.layer_num = self.mem_pool_device.layer_num
        self.layer_done_counter = LayerDoneCounter(self.layer_num)
        self.mem_pool_device.register_layer_transfer_counter(self.layer_done_counter)

        if write_policy not in [
            "write_through",
            "write_through_selective",
            "write_back",
        ]:
            raise ValueError(f"Invalid write policy: {write_policy}")

        # self.write_queue = PriorityQueue[CacheOperation]()
        self.load_queue: List[CacheOperation] = []
        self.write_queue: List[CacheOperation] = []
        self.ack_load_queue: List[HiCacheAck] = []
        self.ack_write_queue: List[HiCacheAck] = []

        self.stop_event = threading.Event()
        self.write_buffer = TransferBuffer(self.stop_event)
        self.load_buffer = TransferBuffer(self.stop_event, buffer_count=10)

        self.write_stream = device_module.Stream()
        self.load_stream = device_module.Stream()

        # MLA KV is identical across TP ranks; rank 0 owns the host copy and
        # broadcasts loaded GPU pages to the other ranks.
        # FP4 is excluded: it keeps an extra per-rank scale buffer that this
        # broadcast does not replicate (MLATokenToKVPoolFP4 subclasses MLA).
        # Dedup only engages with a dedup-compatible storage backend (see
        # storage_supports_host_dedup): the non-rank-0 dummy pool has no host
        # buffer, which RDMA/registered L3 backends cannot tolerate. Must match
        # the dummy-pool decision in HiRadixCache / the hybrid assembler.
        self.is_mla = (
            isinstance(self.mem_pool_device, MLATokenToKVPool)
            and not isinstance(self.mem_pool_device, MLATokenToKVPoolFP4)
            and is_cuda()
            and storage_supports_host_dedup(storage_backend)
        )
        # DSA additionally stores a per-page indexer buffer that must be
        # broadcast alongside the MLA latent (DSATokenToKVPool subclasses MLA).
        self.is_dsa = isinstance(self.mem_pool_device, DSATokenToKVPool)
        self.mla_bcast_group = None
        if self.is_mla:
            if is_dp_attention_enabled():
                self._mla_tp_rank = get_attention_tp_rank()
                self._mla_tp_size = get_attention_tp_size()
            else:
                self._mla_tp_rank = get_tensor_model_parallel_rank()
                self._mla_tp_size = get_tensor_model_parallel_world_size()
            if self._mla_tp_size > 1:
                if mla_broadcast_state is not None:
                    # Pre-built by HiRadixCache BEFORE HostKVCache alloc, so
                    # the world all_gather_object inside
                    # create_custom_parallel_group did NOT race rank 0's
                    # multi-minute host KV pin against the 600s NCCL watchdog.
                    self.mla_bcast_group = mla_broadcast_state["mla_bcast_group"]
                    self._mla_bcast_src = mla_broadcast_state["mla_bcast_src"]
                    self._mla_bt_num_tokens = mla_broadcast_state["mla_bt_num_tokens"]
                    self._mla_bt = mla_broadcast_state["mla_bt"]
                    self._mla_idx_bufs = mla_broadcast_state["mla_idx_bufs"]
                    if self._mla_idx_bufs is not None:
                        self._mla_idx_elem = mla_broadcast_state["mla_idx_elem"]
                        self._mla_idx_bt = mla_broadcast_state["mla_idx_bt"]
                else:
                    self._init_mla_broadcast()
        else:
            self._mla_tp_rank = 0
            self._mla_tp_size = 1

        # If a storage backend is provided at startup, treat it as an implicit attach,
        # so init/runtime share the same lifecycle semantics and code paths.
        if storage_backend is not None:
            try:
                self.attach_storage_backend(
                    storage_backend=storage_backend,
                    prefetch_threshold=prefetch_threshold,
                    model_name=model_name,
                    storage_backend_extra_config=storage_backend_extra_config,
                )
            except ValueError as e:
                # Preserve the historical error shape on init for unknown backends.
                raise ValueError(f"Failed to create storage backend: {e}") from e

    @property
    def mla_broadcast_enabled(self) -> bool:
        return self.is_mla and self.mla_bcast_group is not None

    def _init_mla_broadcast(self) -> None:
        # Delegate to the static helper so the runtime detach→re-attach path
        # (which has no prebuilt state) and the early-prebuild path share one
        # construction code path.
        state = HiCacheController.prebuild_mla_broadcast_state(
            self.mem_pool_device,
            self.tp_group,
            self.attn_cp_group,
            self.attn_tp_group,
            self.layer_num,
            self.device,
            is_dsa=self.is_dsa,
            enable_storage=False,
        )
        self.mla_bcast_group = state["mla_bcast_group"]
        self._mla_bcast_src = state["mla_bcast_src"]
        self._mla_bt_num_tokens = state["mla_bt_num_tokens"]
        self._mla_bt = state["mla_bt"]
        self._mla_idx_bufs = state["mla_idx_bufs"]
        if self._mla_idx_bufs is not None:
            self._mla_idx_elem = state["mla_idx_elem"]
            self._mla_idx_bt = state["mla_idx_bt"]

    @staticmethod
    def prebuild_mla_broadcast_state(
        mem_pool_device,
        tp_group: torch.distributed.ProcessGroup,
        attn_cp_group: Optional[torch.distributed.ProcessGroup],
        attn_tp_group: Optional[torch.distributed.ProcessGroup],
        layer_num: int,
        device,
        *,
        is_dsa: bool,
        enable_storage: bool = False,
    ) -> dict:
        """Create every world-collective HiCacheController would issue at
        init time BEFORE HostKVCache is allocated.

        Why: HostKVCache.__init__ on rank 0 pins the full host KV (can exceed
        the 10-min NCCL watchdog at 800GB), while non-rank-0 attn-TP ranks
        hit the dummy fast path and race ahead to the next world-collective.
        Two collectives downstream would otherwise race rank 0's slow pin:
          (a) _init_mla_broadcast → create_custom_parallel_group(nccl) for
              the mla_bcast_group  (broadcast loaded KV across attn-TP);
          (b) attach_storage_backend → _create_prefetch_sync_groups →
              create_custom_parallel_group(gloo) for prefetch_sync_groups
              (only with --hicache-storage-backend; coordinates L3
              prefetch/eviction across ranks).
        Both go through an all_gather_object + new_group on the default-world
        NCCL comm, whose watchdog kills the proc at 600s.

        Calling this BEFORE HostKVCache means the rendezvouses happen while
        rank 0 has not started pinning yet — they complete in <1s on all
        ranks in lockstep, and rank 0 is then free to take its time pinning
        without racing any further collective.

        Returns a dict to be passed into HiCacheController(...) via
        mla_broadcast_state=...
            HiCacheController.__init__ uses the prebuilt mla_bcast_group;
            _create_prefetch_sync_groups uses the prebuilt
            prefetch_sync_groups if present.
        """
        from sglang.srt.distributed.parallel_state import create_custom_parallel_group

        base_group = tp_group
        if is_dp_attention_enabled() and attn_tp_group is not None:
            base_group = attn_tp_group
        group_ranks = torch.distributed.get_process_group_ranks(base_group)

        # Dedicated NCCL group so the load-time broadcast never interleaves
        # with the model-forward collectives.
        mla_bcast_group = create_custom_parallel_group(
            group_ranks=list(group_ranks), backend="nccl"
        )
        mla_bcast_src = group_ranks[0]

        # Staging buffer coalescing all layers of one token chunk into a
        # single broadcast; reused across loads and bounded by bt_num_tokens.
        mla_bt_num_tokens = 512
        mla_bt = torch.empty(
            layer_num * mla_bt_num_tokens * mem_pool_device.kv_cache_dim,
            dtype=mem_pool_device.kv_buffer[0].dtype,
            device=device,
        )
        # DSA also stores a per-page indexer buffer that must be broadcast.
        mla_idx_bufs = None
        mla_idx_elem = None
        mla_idx_bt = None
        if is_dsa:
            mla_idx_bufs = mem_pool_device.index_k_with_scale_buffer
            mla_idx_elem = math.prod(mla_idx_bufs[0].shape[1:]) or 1
            mla_idx_bt = torch.empty(
                layer_num * mla_bt_num_tokens * mla_idx_elem,
                dtype=mla_idx_bufs[0].dtype,
                device=device,
            )

        # Prefetch sync group rendezvouses (storage-backend path only). Mirror
        # _create_prefetch_sync_groups: CP+TP attn groups if either is set,
        # else the model-tp group. Dedupe by rank-set so a CP=1 case doesn't
        # double-build the same gloo group.
        #
        # Stays None when enable_storage=False, so that a later runtime
        # attach_storage_backend (with no prebuilt groups) still goes through
        # the normal inline build path in _create_prefetch_sync_groups instead
        # of consuming an empty prebuilt list and ending up with zero groups.
        prefetch_sync_groups: Optional[List[torch.distributed.ProcessGroup]] = None
        if enable_storage:
            prefetch_sync_groups = []
            seen_rank_sets = set()
            if attn_cp_group is not None or attn_tp_group is not None:
                base_groups = [attn_cp_group, attn_tp_group]
            else:
                base_groups = [tp_group]
            for group in base_groups:
                if group is None or torch.distributed.get_world_size(group=group) == 1:
                    continue
                ranks = tuple(torch.distributed.get_process_group_ranks(group))
                if ranks in seen_rank_sets:
                    continue
                seen_rank_sets.add(ranks)
                prefetch_sync_groups.append(
                    create_custom_parallel_group(
                        group_ranks=list(ranks), backend="gloo"
                    )
                )

        return {
            "mla_bcast_group": mla_bcast_group,
            "mla_bcast_src": mla_bcast_src,
            "mla_bt_num_tokens": mla_bt_num_tokens,
            "mla_bt": mla_bt,
            "mla_idx_bufs": mla_idx_bufs,
            "mla_idx_elem": mla_idx_elem,
            "mla_idx_bt": mla_idx_bt,
            "prefetch_sync_groups": prefetch_sync_groups,
        }

    @staticmethod
    def maybe_prebuild_mla_broadcast_state(
        kv_cache,
        tp_group: torch.distributed.ProcessGroup,
        attn_cp_group: Optional[torch.distributed.ProcessGroup],
        attn_tp_group: Optional[torch.distributed.ProcessGroup],
        storage_backend: Optional[str],
    ) -> Optional[dict]:
        """Gated convenience wrapper around prebuild_mla_broadcast_state.

        Returns None when no rendezvous is needed (MHA, FP4, non-cuda,
        mla_tp_size == 1, or a non-dedup-compatible storage backend), else
        the state dict to pass to HiCacheController(...) via
        mla_broadcast_state=...

        Gating must mirror HiCacheController.is_mla (minus the per-rank
        check) so we don't build a group on ranks the controller would
        then ignore — that would leak the NCCL group AND desync the
        rendezvous counter. Shared between HiRadixCache and the unified
        assembler strategies.
        """
        if not (
            isinstance(kv_cache, MLATokenToKVPool)
            and not isinstance(kv_cache, MLATokenToKVPoolFP4)
            and storage_supports_host_dedup(storage_backend)
        ):
            return None
        if not is_cuda():
            return None
        if is_dp_attention_enabled():
            mla_tp_size = get_attention_tp_size()
        else:
            mla_tp_size = get_tensor_model_parallel_world_size()
        if mla_tp_size <= 1:
            return None
        return HiCacheController.prebuild_mla_broadcast_state(
            kv_cache,
            tp_group,
            attn_cp_group,
            attn_tp_group,
            kv_cache.layer_num,
            kv_cache.device,
            is_dsa=isinstance(kv_cache, DSATokenToKVPool),
            enable_storage=storage_backend is not None,
        )

    def _destroy_mla_broadcast_group(self) -> None:
        if self.mla_bcast_group is not None:
            try:
                torch.distributed.destroy_process_group(self.mla_bcast_group)
            except Exception:
                pass
            self.mla_bcast_group = None

    def get_attn_cp_rank_and_size(self) -> tuple[int, int]:
        """Derive CP rank/size from the attn_cp process group."""
        if self.attn_cp_group is not None:
            return (
                torch.distributed.get_rank(group=self.attn_cp_group),
                torch.distributed.get_world_size(group=self.attn_cp_group),
            )
        return 0, 1

    def _create_prefetch_sync_groups(self) -> None:
        # If HiRadixCache prebuilt the gloo groups BEFORE the slow host KV
        # alloc (to avoid the 600s NCCL watchdog race — see
        # prebuild_mla_broadcast_state), reuse them and clear the slot so a
        # later runtime detach→re-attach builds fresh.
        if self._prebuilt_prefetch_sync_groups is not None:
            self.prefetch_sync_groups = self._prebuilt_prefetch_sync_groups
            self._prebuilt_prefetch_sync_groups = None
            return

        from sglang.srt.distributed.parallel_state import create_custom_parallel_group

        self.prefetch_sync_groups = []
        seen_rank_sets = set()

        if self.attn_cp_group is not None or self.attn_tp_group is not None:
            base_groups = [self.attn_cp_group, self.attn_tp_group]
        else:
            base_groups = [self.tp_group]

        for group in base_groups:
            if group is None or torch.distributed.get_world_size(group=group) == 1:
                continue
            group_ranks = tuple(torch.distributed.get_process_group_ranks(group))
            if group_ranks in seen_rank_sets:
                continue
            seen_rank_sets.add(group_ranks)
            self.prefetch_sync_groups.append(
                create_custom_parallel_group(
                    group_ranks=list(group_ranks), backend="gloo"
                )
            )

    def _destroy_prefetch_sync_groups(self) -> None:
        for group in self.prefetch_sync_groups:
            try:
                torch.distributed.destroy_process_group(group)
            except Exception:
                pass
        self.prefetch_sync_groups = []

    def _all_reduce_prefetch_groups(self, tensor: torch.Tensor, op) -> None:
        for group in self.prefetch_sync_groups:
            torch.distributed.all_reduce(tensor, op=op, group=group)

    def _start_storage_threads(self):
        """Start storage prefetch/backup threads and their queues.

        This is used by runtime attach, and also by reset when storage is enabled.
        """
        assert self.enable_storage
        assert not self.storage_stop_event.is_set()

        self.prefetch_thread = threading.Thread(
            target=self.prefetch_thread_func, daemon=True
        )
        self.backup_thread = threading.Thread(
            target=self.backup_thread_func, daemon=True
        )
        self.prefetch_queue = Queue()
        self.backup_queue = Queue()

        self.prefetch_revoke_queue: Queue[str] = Queue()
        self.ack_backup_queue: Queue[StorageOperation] = Queue()
        self.host_mem_release_queue: Queue[torch.Tensor] = Queue()

        self.prefetch_thread.start()
        self.backup_thread.start()

    def _stop_storage_threads(self):
        """Stop storage prefetch/backup threads and drain internal queues.

        Caller should ensure no in-flight requests.
        """
        # Always request stop. This is safe even when storage is already disabled,
        # and makes detach truly idempotent (previous partial detach may have left
        # threads alive).
        # NOTE: do NOT clear stop_event unless threads have fully stopped; otherwise
        # a still-alive thread may resume and touch released state.
        self.storage_stop_event.set()

        # Best-effort wakeups so threads exit promptly even if blocked on queues.
        try:
            if hasattr(self, "prefetch_queue"):
                self.prefetch_queue.put_nowait(None)
            if hasattr(self, "backup_queue"):
                self.backup_queue.put_nowait(None)
            if hasattr(self, "prefetch_buffer"):
                self.prefetch_buffer.put_nowait(None)
        except Exception:
            pass

        # Best-effort joins (threads are daemon, but join keeps state clean).
        threads = []
        if hasattr(self, "prefetch_thread"):
            threads.append(self.prefetch_thread)
        if hasattr(self, "backup_thread"):
            threads.append(self.backup_thread)
        if hasattr(self, "prefetch_io_aux_thread"):
            threads.append(self.prefetch_io_aux_thread)

        for t in threads:
            try:
                t.join(timeout=10)
            except Exception:
                pass

        alive = [t for t in threads if getattr(t, "is_alive", lambda: False)()]
        if alive:
            logger.error(
                "Failed to stop HiCache storage threads cleanly: %s",
                [getattr(t, "name", repr(t)) for t in alive],
            )
            raise RuntimeError("Failed to stop HiCache storage threads cleanly.")

    def attach_storage_backend(
        self,
        storage_backend: str,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
    ):
        """Attach (enable) storage backend at runtime.

        Requirement: no in-flight requests. This call is expected to run on the scheduler
        thread (control path), not concurrently with prefetch/backup.
        """
        if self.enable_storage:
            raise RuntimeError("Storage backend already attached.")

        # Reject backends that cannot coexist with the active MLA/NSA
        # host-memory dedup broadcast.
        #
        # When startup --hicache-storage-backend was unset (or "file"),
        # storage_supports_host_dedup(...) was True, so this controller
        # entered dedup mode on every attn-TP rank: rank 0 owns the host
        # copy and broadcasts loaded GPU pages to non-rank-0 ranks that
        # hold allocator-only dummy host pools (kv_buffer is None). The
        # RDMA/registered backends -- mooncake / eic / simm / hf3fs / nixl
        # / aibrix -- pin or register that buffer at construct or register
        # time and would dereference None on the dummy ranks.
        #
        # Reject on EVERY dedup participant (self.mla_broadcast_enabled is
        # True on rank 0 too because it is the broadcast source). Gating
        # on self.mem_pool_host._is_dummy alone would be rank-asymmetric:
        # rank 0's full pool would silently accept the attach while peers
        # raise, and because attach_storage_backend is fanned out via the
        # tokenizer with no rollback on partial failure, the server would
        # be left in a half-attached state.
        if self.mla_broadcast_enabled and not storage_supports_host_dedup(
            storage_backend
        ):
            raise RuntimeError(
                f"Cannot runtime-attach non-dedup-compatible storage backend "
                f"{storage_backend!r} while MLA/NSA host-memory dedup is "
                f"active: non-rank-0 attn-TP ranks hold dummy host pools "
                f"(kv_buffer=None) and this backend would dereference them. "
                f"Only None/''/'file' backends can attach later in dedup "
                f"mode. Restart the server with "
                f"--hicache-storage-backend={storage_backend} to use this "
                f"backend (every rank will then keep a full host pool)."
            )

        # Defensive: a previous partial detach may have flipped `enable_storage` but
        # left background threads alive. Attaching on top of them is unsafe.
        try:
            self._stop_storage_threads()
        except Exception as e:
            raise RuntimeError(
                "Cannot attach storage backend: previous detach did not stop storage threads cleanly."
            ) from e

        # Rollback-safe init: if creation fails, keep controller state consistent
        # for future attach attempts.
        self.storage_backend_type = storage_backend
        from sglang.srt.mem_cache.utils import get_hash_str

        self.get_hash_str = get_hash_str
        self.storage_config = self._generate_storage_config(
            model_name, storage_backend_extra_config
        )
        # for MLA models, only one rank needs to backup the KV cache
        self.backup_skip = (
            self.storage_config.is_mla_model
            # todo: load balancing
            and self.storage_config.tp_rank != 0
        )

        # Use storage backend factory for dynamic backend creation
        from sglang.srt.mem_cache.storage import StorageBackendFactory

        try:
            self.storage_backend = StorageBackendFactory.create_backend(
                storage_backend, self.storage_config, self.mem_pool_host
            )
            # A dummy (non-rank-0 dedup) host pool has no KV buffer to register;
            # backends that pin/register the buffer (Mooncake/SiMM/EIC) would
            # crash on the None buffer. Non-rank-0 never reads L3 anyway (see
            # _page_transfer), so skip registration for it.
            if getattr(self.mem_pool_host, "_is_dummy", False):
                logger.info(
                    "Skipping register_mem_pool_host on dummy (non-rank-0 dedup) "
                    "host pool with no KV buffer."
                )
            else:
                self.storage_backend.register_mem_pool_host(self.mem_pool_host)

            self.enable_storage = True
            # todo: threshold policy for prefetching
            self.prefetch_threshold = max(prefetch_threshold, self.page_size)
            self.prefetch_capacity_limit = max(
                0, int(0.8 * (self.mem_pool_host.size - self.mem_pool_device.size))
            )
            # tracking the number of tokens locked in prefetching, updated by the main scheduler thread
            self.prefetch_tokens_occupied = 0

            # Use dedicated gloo groups so storage prefetch sync is isolated
            # from other collectives and consistent across CPxTP participants.
            self._create_prefetch_sync_groups()

            # Select the get and set functions
            self.page_get_func = self._generic_page_get
            self.page_set_func = self._generic_page_set

            if (
                self.storage_backend_type
                in ["hf3fs", "mooncake", "eic", "nixl", "simm"]
            ) or (
                self.storage_backend_type == "dynamic"
                and bool(self.storage_config.extra_config.get("interface_v1", 0))
            ):
                self.page_get_func = self._page_get_zero_copy
                self.page_set_func = self._page_set_zero_copy

            self._maybe_register_draft_with_storage()

            # Ensure stop_event is clear before starting threads.
            self.storage_stop_event.clear()
            self._start_storage_threads()
        except Exception:
            # Best-effort cleanup for partial init.
            try:
                self._stop_storage_threads()
            except Exception:
                pass
            self._destroy_prefetch_sync_groups()
            try:
                if (
                    hasattr(self, "storage_backend")
                    and self.storage_backend is not None
                ):
                    if hasattr(self.storage_backend, "close"):
                        self.storage_backend.close()
            except Exception:
                pass
            self.storage_backend = None
            self.storage_backend_type = None
            self.enable_storage = False
            self.page_get_func = self._generic_page_get
            self.page_set_func = self._generic_page_set
            self.draft_page_get_func = None
            self.draft_page_set_func = None
            raise

    def detach_storage_backend(self):
        """Detach (disable) storage backend at runtime.

        Requirement: no in-flight requests. This will stop storage threads and release
        the backend instance (best-effort close).
        """
        # Idempotent cleanup: even if `enable_storage` is already False,
        # we may still have leftover resources (threads/backend/process group) from a
        # previous partial detach. We attempt cleanup whenever possible.
        try:
            self._stop_storage_threads()
        except Exception as e:
            # Do not proceed tearing down backend/process group if threads are not
            # fully stopped; otherwise still-alive threads may touch released state.
            # Caller can retry detach.
            logger.exception("Stop storage threads failed: %s", e)
            # IMPORTANT: Do not silently succeed. Upper layers rely on exceptions here
            # to avoid flipping `enable_storage` flags while threads are still alive.
            raise RuntimeError("Stop storage threads failed; detach aborted.") from e

        # Best-effort destroy process groups created for storage ops.
        self._destroy_prefetch_sync_groups()

        # Best-effort close (some backends rely on GC/destructor).
        try:
            if (
                hasattr(self, "storage_backend")
                and self.storage_backend is not None
                and hasattr(self.storage_backend, "close")
            ):
                self.storage_backend.close()
        except Exception:
            logger.exception("Failed to close storage backend cleanly.")

        self.storage_backend = None
        self.storage_backend_type = None
        self.enable_storage = False
        self.page_get_func = self._generic_page_get
        self.page_set_func = self._generic_page_set
        self.draft_page_get_func = None
        self.draft_page_set_func = None
        # Now it's safe to clear the stop event for future re-attach.
        self.storage_stop_event.clear()

    def _generate_storage_config(
        self,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
    ):
        if storage_backend_extra_config is None:
            storage_backend_extra_config = {}

        if is_dp_attention_enabled():
            self.tp_rank = get_attention_tp_rank()
            self.tp_size = get_attention_tp_size()
            self.dp_rank = get_attention_dp_rank()
        else:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
            self.dp_rank = 0

        # Currently, NPUMLATokenToKVPool is the subclass of MLATokenToKVPool.
        # DeepSeekV4TokenToKVPool has compressed MLA-style rank-replicated cache
        # data. storage only needs rank 0 to write it back.
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

        is_mla_model = isinstance(self.mem_pool_device, MLATokenToKVPool)
        is_compressed_mla_model = isinstance(
            self.mem_pool_device, DeepSeekV4TokenToKVPool
        )
        is_rank_replicated = is_mla_model or is_compressed_mla_model
        # Least Common Multiple among heterogeneous tp size
        tp_lcm_size = storage_backend_extra_config.pop("tp_lcm_size", None)
        should_split_heads = False

        if tp_lcm_size:
            assert (
                tp_lcm_size % self.tp_size == 0
            ), "tp_lcm_size must be divisible by tp_size."
            should_split_heads = (
                not is_rank_replicated
                and self.mem_pool_host.layout == "page_head"
                and tp_lcm_size > self.tp_size
            )

        attn_cp_rank, attn_cp_size = self.get_attn_cp_rank_and_size()

        return HiCacheStorageConfig(
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            attn_cp_rank=attn_cp_rank,
            attn_cp_size=attn_cp_size,
            # TODO(hzh): Rename is_mla_model to is_rank_replicated.
            is_mla_model=is_rank_replicated,
            enable_storage_metrics=self.enable_storage_metrics,
            is_page_first_layout=self.mem_pool_host.layout == "page_first",
            model_name=model_name,
            tp_lcm_size=tp_lcm_size,
            should_split_heads=should_split_heads,
            extra_config=storage_backend_extra_config,
        )

    def reset(self):
        self.stop_event.set()
        self.storage_stop_event.set()

        self.write_queue.clear()
        self.load_queue.clear()
        self.write_buffer.clear()
        self.load_buffer.clear()
        self.ack_write_queue.clear()
        self.ack_load_queue.clear()
        if self.enable_storage:
            self.prefetch_thread.join()
            self.backup_thread.join()
            self.prefetch_queue.queue.clear()
            self.backup_queue.queue.clear()
            self.prefetch_revoke_queue.queue.clear()
            self.ack_backup_queue.queue.clear()

        self.stop_event.clear()
        self.storage_stop_event.clear()

        if self.enable_storage:
            self.prefetch_thread = threading.Thread(
                target=self.prefetch_thread_func, daemon=True
            )
            self.backup_thread = threading.Thread(
                target=self.backup_thread_func, daemon=True
            )
            self.prefetch_thread.start()
            self.backup_thread.start()

    def write(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
    ) -> Optional[torch.Tensor]:
        """
        Back up KV caches from device memory to host memory.
        """
        host_indices = self.mem_pool_host.alloc(len(device_indices))
        if host_indices is None:
            return None
        self.write_queue.append(
            CacheOperation(host_indices, device_indices, node_id, priority)
        )
        self.start_writing()
        return host_indices

    def start_writing(self) -> None:
        if len(self.write_queue) == 0:
            return

        op = CacheOperation.merge_ops(self.write_queue)
        self.write_queue.clear()

        start_event = device_module.Event()
        finish_event = device_module.Event()

        if self.mla_broadcast_enabled and self._mla_tp_rank != 0:
            # Non-rank-0 ranks have a dummy host pool: skip D2H, just ack.
            start_event.record()
            finish_event.record()
            self.ack_write_queue.append(
                HiCacheAck(start_event, finish_event, op.node_ids)
            )
            return

        host_indices, device_indices = self.move_indices(
            op.host_indices, op.device_indices
        )

        start_event.record()
        with device_module.stream(self.write_stream):
            start_event.wait(self.write_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device, host_indices, device_indices, self.io_backend
            )
            if self.has_draft:
                self.mem_pool_host_draft.backup_from_device_all_layer(
                    self.mem_pool_device_draft,
                    host_indices,
                    device_indices,
                    self.io_backend,
                )
            finish_event.record()
            # NOTE: We must save the host indices and device indices here,
            # this is because we need to guarantee that these tensors are
            # still alive when the write stream is executing.
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_stream)

        self.ack_write_queue.append(HiCacheAck(start_event, finish_event, op.node_ids))

    def load(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
    ) -> Optional[torch.Tensor]:
        """
        Load KV caches from host memory to device memory.
        """
        device_indices = self.mem_pool_device_allocator.alloc(len(host_indices))
        if device_indices is None:
            return None
        self.load_queue.append(
            CacheOperation(host_indices, device_indices, node_id, priority)
        )
        return device_indices

    def move_indices(self, host_indices: torch.Tensor, device_indices: torch.Tensor):
        # move indices to GPU if using kernels, to host if using direct indexing
        if self.io_backend == "kernel":
            if not host_indices.is_cuda:
                host_indices = host_indices.to(self.device, non_blocking=True)
            return host_indices, device_indices
        elif self.io_backend == "direct":
            if self.mem_pool_host.layout == "layer_first":
                device_indices = device_indices.cpu()
                host_indices, idx = host_indices.sort()
                return host_indices, device_indices.index_select(0, idx)
            elif self.mem_pool_host.layout == "page_first_direct":
                return host_indices, device_indices.cpu()
            else:
                raise ValueError(
                    f"Unsupported layout {self.mem_pool_host.layout!r} for io backend 'direct'"
                )
        elif self.io_backend == "kernel_ascend":
            return host_indices, device_indices.cpu()
        else:
            raise ValueError(f"Unsupported io backend")

    def start_loading(self) -> int:
        if len(self.load_queue) == 0:
            return -1

        producer_id = self.layer_done_counter.update_producer()
        op = CacheOperation.merge_ops(self.load_queue)
        self.load_queue.clear()

        if self.mla_broadcast_enabled:
            return self._start_loading_mla(producer_id, op)

        host_indices, device_indices = self.move_indices(
            op.host_indices, op.device_indices
        )
        producer_event = self.layer_done_counter.events[producer_id]
        producer_event.start_event.record()

        with device_module.stream(self.load_stream):
            producer_event.start_event.wait(self.load_stream)
            for i in range(self.layer_num):
                self.mem_pool_host.load_to_device_per_layer(
                    self.mem_pool_device,
                    host_indices,
                    device_indices,
                    i,
                    self.io_backend,
                )
                if self.has_draft and i < self.mem_pool_host_draft.layer_num:
                    self.mem_pool_host_draft.load_to_device_per_layer(
                        self.mem_pool_device_draft,
                        host_indices,
                        device_indices,
                        i,
                        self.io_backend,
                    )
                producer_event.complete(i)
            # NOTE: We must save the host indices and device indices here,
            # this is because we need to guarantee that these tensors are
            # still alive when the load stream is executing.
            if host_indices.is_cuda:
                host_indices.record_stream(self.load_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.load_stream)

        self.ack_load_queue.append(
            HiCacheAck(
                start_event=producer_event.start_event,
                finish_event=producer_event.finish_event,
                node_ids=op.node_ids,
            )
        )
        return producer_id

    def _start_loading_mla(self, producer_id: int, op: CacheOperation) -> int:
        """Load MLA KV on rank 0, then broadcast it to the other TP ranks.

        H2D and broadcast are both enqueued on ``load_stream``: stream ordering
        guarantees rank 0's H2D lands before the broadcast reads the KV buffer,
        and the per-layer load events fire when the stream drains, so the normal
        ``loading_check`` ack path finalizes the load with no extra polling.
        """
        producer_event = self.layer_done_counter.events[producer_id]
        producer_event.start_event.record()

        with device_module.stream(self.load_stream):
            producer_event.start_event.wait(self.load_stream)
            if self._mla_tp_rank == 0:
                host_indices, device_indices = self.move_indices(
                    op.host_indices, op.device_indices
                )
                for i in range(self.layer_num):
                    self.mem_pool_host.load_to_device_per_layer(
                        self.mem_pool_device,
                        host_indices,
                        device_indices,
                        i,
                        self.io_backend,
                    )
                if host_indices.is_cuda:
                    host_indices.record_stream(self.load_stream)
                if device_indices.is_cuda:
                    device_indices.record_stream(self.load_stream)
                # The "direct" io backend may issue H2D off load_stream, so plain
                # stream ordering is not enough; fully land rank 0's H2D before
                # the broadcast reads the device KV buffer.
                self.load_stream.synchronize()

            self._broadcast_mla_kv(op.device_indices)
            for i in range(self.layer_num):
                producer_event.complete(i)

        self.ack_load_queue.append(
            HiCacheAck(
                start_event=producer_event.start_event,
                finish_event=producer_event.finish_event,
                node_ids=op.node_ids,
            )
        )
        return producer_id

    def _bcast_buf(self, buf_list, staging, target, elem) -> None:
        """Broadcast one per-layer buffer set from rank 0, in row chunks.

        ``target`` indexes dim 0 of each layer tensor (token indices for the KV
        latent, page indices for the DSA indexer). Must run on load_stream.
        """
        is_src = self._mla_tp_rank == 0
        n = target.shape[0]
        for start in range(0, n, self._mla_bt_num_tokens):
            cur = min(self._mla_bt_num_tokens, n - start)
            idx = target[start : start + cur]
            chunk = staging[: self.layer_num * cur * elem]
            if is_src:
                for layer_id in range(self.layer_num):
                    o = layer_id * cur * elem
                    chunk[o : o + cur * elem].copy_(buf_list[layer_id][idx].reshape(-1))
            torch.distributed.broadcast(
                chunk, src=self._mla_bcast_src, group=self.mla_bcast_group
            )
            if not is_src:
                for layer_id in range(self.layer_num):
                    o = layer_id * cur * elem
                    buf_list[layer_id][idx] = chunk[o : o + cur * elem].view(
                        buf_list[layer_id][idx].shape
                    )

    def _broadcast_mla_kv(self, device_indices: torch.Tensor) -> None:
        """Broadcast loaded KV (and DSA indexer) pages from rank 0 to peers.

        Coalesced layer-by-layer into reused staging buffers and sent in chunks,
        one NCCL broadcast per chunk over the dedicated group. Must run on the
        load stream.
        """
        indices = device_indices
        if not indices.is_cuda:
            indices = indices.to(self.device)
        if indices.is_cuda:
            indices.record_stream(self.load_stream)
        self._bcast_buf(
            self.mem_pool_device.kv_buffer,
            self._mla_bt,
            indices,
            self.mem_pool_device.kv_cache_dim,
        )
        if self._mla_idx_bufs is not None:
            page_size = self.mem_pool_device.page_size
            page_idx = (
                torch.unique(torch.div(indices, page_size, rounding_mode="floor"))
                if page_size > 1
                else indices
            )
            if page_idx.is_cuda:
                page_idx.record_stream(self.load_stream)
            self._bcast_buf(
                self._mla_idx_bufs, self._mla_idx_bt, page_idx, self._mla_idx_elem
            )

    def evict_device(self, device_indices: torch.Tensor) -> int:
        self.mem_pool_device_allocator.free(device_indices)
        return len(device_indices)

    def evict_host(self, host_indices: torch.Tensor, backup_only: bool = True) -> int:
        if not backup_only:
            raise ValueError("Other eviction policies are not supported yet.")

        self.mem_pool_host.free(host_indices)
        return len(host_indices)

    def set_draft_kv_pool(self, draft_device_pool, draft_host_pool) -> None:
        """Register draft KV pools so L2/L3 ops piggyback draft transfers."""
        if self.mla_broadcast_enabled:
            raise NotImplementedError(
                "Draft KV pools are not supported together with the MLA host "
                "memory dedup broadcast. Disable hierarchical cache for the "
                "draft model or run MLA without TP>1 dedup."
            )
        self.has_draft = True
        self.mem_pool_device_draft = draft_device_pool
        self.mem_pool_host_draft = draft_host_pool
        logger.info(
            "HiCache draft KV registered: %s (host %d slots)",
            type(draft_device_pool).__name__,
            draft_host_pool.size,
        )

        # If storage is already attached, wire up the draft I/O path now.
        # Otherwise this will be deferred until attach_storage_backend().
        self._maybe_register_draft_with_storage()

    def _maybe_register_draft_with_storage(self) -> None:
        """Pick the draft L3 IO implementation."""
        self.draft_page_get_func = None
        self.draft_page_set_func = None
        if not self.has_draft or not self.enable_storage:
            return

        backend = self.storage_backend_type

        # Multi-pool zero-copy backends.
        if backend == "mooncake":
            if self.storage_config.should_split_heads:
                logger.warning(
                    "HiCache draft L3 disabled: should_split_heads not yet "
                    "supported on the mooncake v2 path."
                )
                return
            self.storage_backend.register_mem_host_pool_v2(
                self.mem_pool_host_draft, PoolName.DRAFT
            )
            self.draft_page_get_func = self._draft_page_get_v2
            self.draft_page_set_func = self._draft_page_set_v2
            return

        # TODO: support "hf3fs", "eic", "nixl", "simm"
        if backend in {"hf3fs", "eic", "nixl", "simm"}:
            logger.warning(
                "HiCache draft L3 disabled: backend %s does not yet support "
                "draft pool registration.",
                backend,
            )
            return

        # Generic backends.
        self.draft_page_get_func = self._draft_page_get_generic
        self.draft_page_set_func = self._draft_page_set_generic

    def prefetch(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ) -> PrefetchOperation:
        """
        Prefetch KV caches from storage backend to host memory.
        """
        operation = PrefetchOperation(
            request_id, host_indices, new_input_tokens, last_hash, prefix_keys
        )
        self.prefetch_queue.put(operation)
        return operation

    def terminate_prefetch(self, operation):
        operation.mark_terminate()
        return operation.completed_tokens, operation.hash_value

    def append_host_mem_release(self, host_indices: torch.Tensor):
        if host_indices.numel() == 0:
            return
        pages = host_indices.split(self.mem_pool_host.page_size)
        for page in pages:
            self.host_mem_release_queue.put(page)

    def _page_get_zero_copy(
        self, operation, hash_values, host_indices, extra_info=None
    ):
        results = self.storage_backend.batch_get_v1(
            hash_values, host_indices, extra_info
        )
        inc = 0
        for i in range(len(hash_values)):
            if not results[i]:
                logger.warning(
                    f"Prefetch operation {operation.request_id} failed to retrieve page {hash_values[i]}."
                )
                break
            inc += self.page_size
        operation.increment(inc)

    # todo: deprecate
    def _generic_page_get(self, operation, hash_values, host_indices, extra_info=None):
        dummy_page_dst = [
            self.mem_pool_host.get_dummy_flat_data_page() for _ in hash_values
        ]
        page_data = self.storage_backend.batch_get(hash_values, dummy_page_dst)
        if page_data is None:
            return
        for i in range(len(hash_values)):
            if page_data[i] is None:
                logger.warning(
                    f"Prefetch operation {operation.request_id} failed to retrieve page {hash_values[i]}."
                )
                break
            # Must set the data before increasing the completed tokens.
            # Otherwise this page may be read before being set.
            self.mem_pool_host.set_from_flat_data_page(
                host_indices[i * self.page_size],
                page_data[i],
            )
            if not operation.increment(self.page_size):
                break  # Operation terminated by controller

    def _page_transfer(self, operation):
        # MLA dedup: non-rank-0 ranks have a dummy host pool (no kv_buffer), so
        # only rank 0 reads L3 into host; the other ranks receive the data via
        # the load-time broadcast. Mark the prefetch complete here so the
        # cross-rank accounting (already MIN-synced in _storage_hit_query) and
        # host-slot bookkeeping stay consistent without touching the dummy pool.
        # (Backup is already rank-0-only via self.backup_skip.)
        if self.mla_broadcast_enabled and self._mla_tp_rank != 0:
            operation.completed_tokens += len(operation.hash_value) * self.page_size
            return
        # Transfer batch by batch
        prefix_keys = operation.prefix_keys
        for i in range(0, len(operation.hash_value), STORAGE_BATCH_SIZE):
            batch_hashes = operation.hash_value[i : i + STORAGE_BATCH_SIZE]
            batch_host_indices = operation.host_indices[
                i * self.page_size : (i + len(batch_hashes)) * self.page_size
            ]

            # Best-effort draft L3 read before publishing target completion.
            # Otherwise wait_complete can race and load back target KV before
            # draft KV reaches host memory.
            if self.has_draft:
                self._draft_page_get(batch_hashes, batch_host_indices)

            prev_completed_tokens = operation.completed_tokens
            # Get one batch token, and update the completed_tokens if succeed
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            self.page_get_func(operation, batch_hashes, batch_host_indices, extra_info)
            # Check termination
            if (
                operation.completed_tokens
                != prev_completed_tokens + len(batch_hashes) * self.page_size
            ):
                operation.mark_terminate()
                break  # Some operations fail or operation terminated by controller

            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes

    def prefetch_io_aux_func(self):
        """
        Auxiliary function conducting IO operations for prefetching.
        """
        while not self.storage_stop_event.is_set():
            try:
                operation = self.prefetch_buffer.get(block=True, timeout=1)
                if operation is None:
                    continue
                self._page_transfer(operation)
                # operation terminated by controller, release pre-allocated memory
                self.append_host_mem_release(
                    operation.host_indices[operation.completed_tokens :]
                )
            except Empty:
                continue

    def prefetch_rate_limited(self) -> bool:
        """
        Rate limit the prefetching operations to avoid overwhelming the storage backend.
        """
        # cancel prefetch if too much memory is occupied
        if self.prefetch_tokens_occupied >= self.prefetch_capacity_limit:
            return True
        # todo: more sophisticated rate limiting based on storage backend performance
        return False

    def _storage_hit_query(self, operation) -> tuple[list[str], int]:
        last_hash = operation.last_hash
        tokens_to_fetch = operation.token_ids
        prefix_keys = operation.prefix_keys.copy() if operation.prefix_keys else None

        storage_query_count = 0
        hash_value = []

        for start in range(
            0, len(tokens_to_fetch), self.page_size * STORAGE_BATCH_SIZE
        ):
            end = min(start + self.page_size * STORAGE_BATCH_SIZE, len(tokens_to_fetch))
            batch_tokens = tokens_to_fetch[start:end]
            batch_hashes = []
            for i in range(0, len(batch_tokens), self.page_size):
                last_hash = self.get_hash_str(
                    batch_tokens[i : i + self.page_size], last_hash
                )
                batch_hashes.append(last_hash)
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            hit_page_num = self.storage_backend.batch_exists(batch_hashes, extra_info)
            hash_value.extend(batch_hashes[:hit_page_num])
            storage_query_count += hit_page_num * self.page_size
            if hit_page_num < len(batch_hashes):
                break
            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes

        return hash_value, storage_query_count

    def prefetch_thread_func(self):
        """
        Manage prefetching operations from storage backend to host memory.
        """
        self.prefetch_buffer = Queue()
        self.prefetch_io_aux_thread = threading.Thread(
            target=self.prefetch_io_aux_func, daemon=True
        )
        self.prefetch_io_aux_thread.start()
        while (not self.storage_stop_event.is_set()) or not self.prefetch_queue.empty():
            try:
                operation = self.prefetch_queue.get(block=True, timeout=1)
                if operation is None:
                    continue
                hash_value, storage_hit_count = self._storage_hit_query(operation)
                storage_hit_count_tensor = torch.tensor(
                    storage_hit_count, dtype=torch.int
                )
                self._all_reduce_prefetch_groups(
                    storage_hit_count_tensor, torch.distributed.ReduceOp.MIN
                )
                storage_hit_count = storage_hit_count_tensor.item()

                if storage_hit_count < self.prefetch_threshold:
                    # not to prefetch if not enough benefits
                    self.prefetch_revoke_queue.put(operation.request_id)
                    self.append_host_mem_release(operation.host_indices)
                    logger.debug(
                        f"Revoking prefetch for request {operation.request_id} due to insufficient hits ({storage_hit_count})."
                    )
                else:
                    operation.hash_value = hash_value[
                        : (storage_hit_count // self.page_size)
                    ]
                    # free the pre-allocated memory for pages that are not hit
                    self.append_host_mem_release(
                        operation.host_indices[storage_hit_count:]
                    )
                    operation.host_indices = operation.host_indices[:storage_hit_count]
                    logger.debug(
                        f"Prefetching {len(operation.hash_value)} pages for request {operation.request_id}."
                    )
                    self.prefetch_buffer.put(operation)

            except Empty:
                continue

    def write_storage(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
    ) -> int:
        """
        Write KV caches from host memory to storage backend.
        """
        operation = StorageOperation(
            host_indices, token_ids, hash_value=hash_value, prefix_keys=prefix_keys
        )
        self.backup_queue.put(operation)
        return operation.id

    # todo: deprecate
    def _generic_page_set(self, hash_values, host_indices, extra_info=None) -> bool:
        data = [
            self.mem_pool_host.get_data_page(host_indices[i * self.page_size])
            for i in range(len(hash_values))
        ]
        return self.storage_backend.batch_set(hash_values, data)

    def _page_set_zero_copy(self, hash_values, host_indices, extra_info=None) -> bool:
        return all(
            self.storage_backend.batch_set_v1(hash_values, host_indices, extra_info)
        )

    def _draft_page_set(self, hash_values, host_indices) -> None:
        """Best-effort write draft KV pages to L3 alongside the target backup."""
        if self.draft_page_set_func is None:
            return
        try:
            self.draft_page_set_func(hash_values, host_indices)
        except Exception:
            logger.debug(
                "Draft L3 write failed (best-effort), skipping.", exc_info=True
            )

    def _draft_page_get(self, hash_values, host_indices) -> None:
        """Best-effort read draft KV pages from L3 (mirrors `_draft_page_set`)."""
        if self.draft_page_get_func is None:
            return
        try:
            self.draft_page_get_func(hash_values, host_indices)
        except Exception:
            logger.debug("Draft L3 read failed (best-effort), skipping.", exc_info=True)

    def _draft_page_set_v2(self, hash_values, host_indices) -> None:
        self.storage_backend.batch_set_v2(
            [
                PoolTransfer(
                    name=PoolName.DRAFT,
                    host_indices=host_indices,
                    keys=list(hash_values),
                )
            ]
        )

    def _draft_page_get_v2(self, hash_values, host_indices) -> None:
        self.storage_backend.batch_get_v2(
            [
                PoolTransfer(
                    name=PoolName.DRAFT,
                    host_indices=host_indices,
                    keys=list(hash_values),
                )
            ]
        )

    def _draft_page_set_generic(self, hash_values, host_indices) -> None:
        # `{hash}.draft` mirrors HiCacheStorage._get_component_key's
        # `{key}.{pool_name}` convention so target/draft pages never collide.
        draft_keys = [f"{h}.{PoolName.DRAFT}" for h in hash_values]
        draft_data = [
            self.mem_pool_host_draft.get_data_page(host_indices[i * self.page_size])
            for i in range(len(draft_keys))
        ]
        self.storage_backend.batch_set(draft_keys, draft_data)

    def _draft_page_get_generic(self, hash_values, host_indices) -> None:
        draft_keys = [f"{h}.{PoolName.DRAFT}" for h in hash_values]
        draft_dummy = [
            self.mem_pool_host_draft.get_dummy_flat_data_page() for _ in draft_keys
        ]
        draft_pages = self.storage_backend.batch_get(draft_keys, draft_dummy)
        if draft_pages is None:
            return
        for i, p in enumerate(draft_pages):
            if p is not None:
                self.mem_pool_host_draft.set_from_flat_data_page(
                    host_indices[i * self.page_size], p
                )

    # Backup batch by batch
    def _page_backup(self, operation):
        # Backup batch by batch
        prefix_keys = operation.prefix_keys
        for i in range(0, len(operation.hash_value), STORAGE_BATCH_SIZE):
            batch_hashes = operation.hash_value[i : i + STORAGE_BATCH_SIZE]
            batch_host_indices = operation.host_indices[
                i * self.page_size : (i + len(batch_hashes)) * self.page_size
            ]
            # Set one batch token, and record if success.
            # todo: allow partial success
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            success = self.page_set_func(batch_hashes, batch_host_indices, extra_info)
            if not success:
                logger.warning(
                    f"Write page to storage: {len(batch_hashes)} pages failed."
                )
                break

            # Best-effort draft L3 write alongside target.
            if self.has_draft:
                self._draft_page_set(batch_hashes, batch_host_indices)

            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes
            operation.completed_tokens += self.page_size * len(batch_hashes)

    def backup_thread_func(self):
        """
        Manage backup operations from host memory to storage backend.
        """
        while not self.storage_stop_event.is_set():
            try:
                operation = self.backup_queue.get(block=True, timeout=1)
                if operation is None:
                    continue

                if not self.backup_skip:
                    self._page_backup(operation)
                self.ack_backup_queue.put(operation)

            except Empty:
                continue
