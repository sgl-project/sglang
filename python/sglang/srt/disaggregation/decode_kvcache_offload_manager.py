from __future__ import annotations

import json
import logging
import threading
import time
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.disaggregation.kv_events import OffloadedState
from sglang.srt.environ import envs
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import free_req_slot
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    build_hybrid_swa_stack,
    build_kv_host_pool,
)
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.unified_memory_pool import UnifiedSWAKVPool
from sglang.srt.mem_cache.utils import storage_namespace_seed
from sglang.srt.runtime_context import (
    get_memory,
    get_schedule,
    get_serving,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


class DecodeKVCacheOffloadManager:
    """Manage decode-side KV cache offloading lifecycle and operations."""

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tp_group: torch.distributed.ProcessGroup,
        tree_cache: BasePrefixCache,
    ) -> None:
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = get_schedule().page_size
        self.request_counter = 0
        self.tree_cache = tree_cache
        env_stride = envs.SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE.get()
        if env_stride is None or env_stride <= 0:
            self.offload_stride = self.page_size
        else:
            self.offload_stride = max(
                self.page_size, (env_stride // self.page_size) * self.page_size
            )
        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        hicache_storage_backend_extra_config = {}
        if get_memory().hicache_storage_backend_extra_config:
            try:
                hicache_storage_backend_extra_config = json.loads(
                    get_memory().hicache_storage_backend_extra_config
                )
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid hicache storage backend extra config JSON: {e}"
                )

        self.kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        if isinstance(self.kv_cache, UnifiedSWAKVPool):
            full_mapping = {
                gid: lid
                for gid, (lid, is_swa) in self.kv_cache.layers_mapping.items()
                if not is_swa
            }
            swa_mapping = {
                gid: lid
                for gid, (lid, is_swa) in self.kv_cache.layers_mapping.items()
                if is_swa
            }
            params = CacheInitParams(
                disable=False,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                page_size=self.page_size,
                tp_cache_group=tp_group,
            )
            self.decode_host_mem_pool, self.cache_controller = build_hybrid_swa_stack(
                params=params,
                full_kv_pool=self.kv_cache.full_kv_pool,
                swa_kv_pool=self.kv_cache.swa_kv_pool,
                full_layer_mapping=full_mapping,
                swa_layer_mapping=swa_mapping,
                load_cache_event=threading.Event(),
                storage_backend=get_memory().hicache_storage_backend,
                use_mla=False,
                model_name=get_serving().served_model_name,
                storage_backend_extra_config=hicache_storage_backend_extra_config,
            )
        else:
            if not isinstance(self.kv_cache, (MHATokenToKVPool, MLATokenToKVPool)):
                raise ValueError("Unsupported KV cache type for decode offload")
            self.decode_host_mem_pool = build_kv_host_pool(
                kv_pool=self.kv_cache,
                page_size=self.page_size,
                use_mla=isinstance(self.kv_cache, MLATokenToKVPool),
            )
            self.cache_controller = HiCacheController(
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                mem_pool_host=self.decode_host_mem_pool,
                page_size=self.page_size,
                tp_group=tp_group,
                io_backend=get_memory().hicache_io_backend,
                load_cache_event=threading.Event(),
                storage_backend=get_memory().hicache_storage_backend,
                model_name=get_serving().served_model_name,
                storage_backend_extra_config=hicache_storage_backend_extra_config,
            )

        # ack_id -> (rid, host_indices, incremental_tokens, page_hashes,
        # start_time, pool_transfers); the request's progress is on the
        # request itself (``req.disagg.offload``).
        self.ongoing_offload = {}
        self.ongoing_backup = {}
        logger.info("Enable offload kv cache for decode side")

    def release_host_resources(self) -> None:
        self.decode_host_mem_pool.destroy()

    def has_inflight_offload(self, req: Req) -> bool:
        """Whether a copy of the request's KV has not landed yet."""
        state = req.disagg.offload
        return state is not None and any(
            ack_id in self.ongoing_offload for ack_id in state.ack_ids
        )

    def _prefill_offloaded_len(self, req: Req) -> int:
        # Page-aligned prompt length; the prefill instance offloaded this part.
        return len(req.origin_input_ids) // self.page_size * self.page_size

    def has_inflight_device_transfer(self) -> bool:
        """Whether a device-to-host copy may still read unified KV pages."""
        return bool(self.ongoing_offload)

    def _resolve_offload_transfers(
        self, virtual_indices: torch.Tensor
    ) -> tuple[torch.Tensor, list[PoolTransfer]]:
        full_indices = virtual_indices
        if not isinstance(self.kv_cache, UnifiedSWAKVPool):
            return full_indices, []

        swa_indices = self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
            virtual_indices
        )
        # Page-aligned chunks have whole-page bindings; zero denotes the sink page.
        live_swa_indices = swa_indices[swa_indices > 0].to(torch.int64)
        if live_swa_indices.numel() == 0:
            return full_indices, []
        return full_indices, [
            PoolTransfer(
                name=PoolName.SWA,
                device_indices=live_swa_indices,
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )
        ]

    def offload_kv_cache(self, req) -> bool:
        """Offload incremental KV cache for decode side."""

        if self.cache_controller is None or self.decode_host_mem_pool is None:
            return False

        if req.kv.req_pool_idx == -1 or len(req.output_ids) == 0:
            return False

        token_indices = self.req_to_token_pool.req_to_token[req.kv.req_pool_idx]
        if token_indices.dim() == 0 or token_indices.numel() == 0:
            return False

        # Prefill side offloads page-aligned origin_input_ids, decode side offloads the incremental part
        all_tokens = req.origin_input_ids + req.output_ids[:-1]
        prefill_offloaded_len = self._prefill_offloaded_len(req)
        state = req.disagg.offload
        if state is None:
            prefill_hashes = self._compute_prefix_hash(
                req, req.origin_input_ids[:prefill_offloaded_len]
            )
            last_prefill_hash = (
                prefill_hashes[-1] if prefill_offloaded_len > 0 else None
            )
            state = OffloadedState(last_hash=last_prefill_hash)
            req.disagg.offload = state
        incremental_total = len(all_tokens) - prefill_offloaded_len
        incremental_new = incremental_total - state.inc_len
        incremental_aligned_len = (
            incremental_new // self.offload_stride * self.offload_stride
        )

        if incremental_aligned_len == 0:
            return False

        # Extract incremental tokens and indices for the newly available chunk
        start = prefill_offloaded_len + state.inc_len
        end = start + incremental_aligned_len
        incremental_tokens = all_tokens[start:end]
        incremental_indices = token_indices[start:end].to(torch.int64)
        device_indices, pool_transfers = self._resolve_offload_transfers(
            incremental_indices
        )

        # Prefill-aligned GPU slots are freed at request finish in
        # _release_finished_req, NOT here. The decoding request
        # continues to attend to those slots via req_to_token; freeing
        # them mid-decode races with concurrent admission, which can
        # reuse the slots and produce cross-pollinated KV reads.

        # Asynchronously offload incremental KV cache from device to host
        self.request_counter += 1
        ack_id = self.request_counter
        host_indices = self.cache_controller.write(
            device_indices=device_indices,
            node_id=ack_id,
            extra_pools=pool_transfers,
        )
        if host_indices is None:
            logger.error(f"Not enough host memory for request {req.rid}")
            return False

        # The chain advances in submit order, the order the copies land in.
        page_hashes = self._compute_prefix_hash(
            req, incremental_tokens, state.last_hash
        )
        if len(page_hashes) > 0:
            state.last_hash = page_hashes[-1]
        state.ack_ids = [a for a in state.ack_ids if a in self.ongoing_offload]
        state.ack_ids.append(ack_id)
        self.ongoing_offload[ack_id] = (
            req.rid,
            host_indices,
            incremental_tokens,
            page_hashes,
            time.time(),
            pool_transfers,
        )
        state.inc_len += incremental_aligned_len
        return True

    def check_offload_progress(self, releasing: List[Req]) -> List[Req]:
        """Check the progress of offload from device to host and backup from
        host to storage. A request of ``releasing`` whose last copy landed has
        its KV released; returns those requests."""
        cc = self.cache_controller

        qsizes = torch.tensor(
            [
                len(cc.ack_write_queue),
                cc.ack_backup_queue.qsize(),
            ],
            dtype=torch.int,
        )
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                qsizes, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )

        n_write, n_backup = map(int, qsizes.tolist())
        released = self._check_offload_progress(n_write, releasing)
        self._check_backup_progress(n_backup)
        return released

    def refuse_retraction_while_copying(self):
        """Refuse retraction on every rank if any rank still tracks a D2H copy;
        the caller first checks the offload progress."""
        cc = self.cache_controller
        # Retraction frees the device KV these copies read. Storage backups read only
        # host memory, so they do not count.
        pending = torch.tensor(
            [bool(cc.ack_write_queue or self.ongoing_offload)],
            dtype=torch.int,
        )
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                pending, op=torch.distributed.ReduceOp.MAX, group=self.tp_group
            )
        if pending.item():
            raise RuntimeError(
                "Decode KV offload copies are still pending on some rank; "
                f"refusing to retract (local: {len(cc.ack_write_queue)} acks, "
                f"{len(self.ongoing_offload)} copies, "
                f"{len({rid for rid, *_ in self.ongoing_offload.values()})} requests)."
            )

    def _check_offload_progress(
        self, finish_count: int, releasing: List[Req]
    ) -> List[Req]:
        """Check the progress of offload from device to host."""
        released = []
        while finish_count > 0:
            ack = self.cache_controller.ack_write_queue.pop(0)
            ack.finish_event.synchronize()
            for ack_id in ack.node_ids:
                (
                    rid,
                    host_indices,
                    incremental_tokens,
                    page_hashes,
                    start_time,
                    pool_transfers,
                ) = self.ongoing_offload.pop(ack_id)

                self._trigger_backup(
                    rid,
                    host_indices,
                    incremental_tokens,
                    page_hashes,
                    start_time,
                    pool_transfers,
                )

                # A request still decoding is not in releasing; it releases
                # at finish, or moves there while a copy is in flight.
                req = self._find_releasing(releasing, rid, ack_id)
                if req is not None and not self.has_inflight_offload(req):
                    self._release_finished_req(req)
                    released.append(req)
            finish_count -= 1
        return released

    @staticmethod
    def _find_releasing(releasing: List[Req], rid: str, ack_id: int) -> Optional[Req]:
        """The releasing request that submitted the copy; a reused rid can
        name another request, so the ack id decides."""
        for req in releasing:
            state = req.disagg.offload
            if req.rid == rid and state is not None and ack_id in state.ack_ids:
                return req
        return None

    def _release_finished_req(self, req: Req):
        # Defensive guard: ReqToTokenPool.free sets req_pool_idx to None,
        # so a previously-released request must be skipped here to avoid
        # non-idempotent side effects (double free, double unlock).
        if req.kv.req_pool_idx is None or req.kv.req_pool_idx == -1:
            return

        # Released only at request finish; a mid-decode free races with
        # concurrent admission over live slots. The prefix the prealloc match
        # took from the tree stays with the tree: not freed, only unlocked.
        self.tree_cache.free_kv_row(
            req.kv, [(req.kv.cache_protected_len, req.kv.kv_allocated_len)]
        )
        self.tree_cache.unlock(req.lock)
        req.lock = None
        free_req_slot(self.req_to_token_pool, req)
        req.kv.mark_kv_released()
        req.disagg.offload = None

    def _check_backup_progress(self, finish_count):
        """Check the progress of backup from host to storage."""
        for _ in range(finish_count):
            storage_operation = self.cache_controller.ack_backup_queue.get()
            ack_id = storage_operation.id
            req_id, host_indices, start_time, pool_transfers = self.ongoing_backup.pop(
                ack_id
            )

            # Release host memory
            for transfer in pool_transfers:
                self.decode_host_mem_pool.get_pool(transfer.name).free(
                    transfer.host_indices
                )
            self.decode_host_mem_pool.free(host_indices)

            logger.debug(
                f"Finished backup request {req_id}, free host memory, len:{len(host_indices)}, cost time:{time.time() - start_time:.2f} seconds."
            )

    def _trigger_backup(
        self,
        rid,
        host_indices,
        incremental_tokens,
        page_hashes,
        start_time,
        pool_transfers=(),
    ):
        """Trigger async backup from host to storage."""
        for transfer in pool_transfers:
            page_count = transfer.device_indices.numel() // self.page_size
            transfer.keys = page_hashes[-page_count:]
        ack_id = self.cache_controller.write_storage(
            host_indices,
            incremental_tokens,
            hash_value=page_hashes,
            **({"extra_pools": pool_transfers} if pool_transfers else {}),
        )
        self.ongoing_backup[ack_id] = (
            rid,
            host_indices,
            start_time,
            pool_transfers,
        )

    def _compute_prefix_hash(self, req: Req, tokens, prior_hash=""):
        """Match prefill storage hashes."""
        page_hashes = []
        last_hash = prior_hash or storage_namespace_seed(req.extra_key, req.cache_salt)
        for offset in range(0, len(tokens), self.page_size):
            page_tokens = tokens[offset : offset + self.page_size]
            last_hash = self.cache_controller.get_hash_str(page_tokens, last_hash)
            page_hashes.append(last_hash)
        return page_hashes

    def finalize_release_on_finish(self, req: Req):
        """Free any remaining tail KV that was not offloaded due to non-aligned length."""
        if self.has_inflight_offload(req):
            return
        self._release_finished_req(req)
