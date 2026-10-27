from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Optional
from weakref import WeakKeyDictionary as WeakKeyDict

import torch

from sglang.srt.disaggregation.kv_events import OffloadedState
from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.unified_cache.components import (
    CacheTransferPhase,
    ComponentType,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.mem_cache.utils import storage_namespace_seed
from sglang.srt.runtime_context import (
    get_exec,
    get_schedule,
    mamba_track_grid,
    max_speculative_num_draft_tokens,
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
        tree_cache: UnifiedRadixCache,
    ) -> None:
        if (
            not isinstance(tree_cache, UnifiedRadixCache)
            or tree_cache.cache_controller is None
        ):
            raise ValueError(
                "Decode KV offload writes through the tree cache's HiCache host "
                f"stack, which {type(tree_cache).__name__} does not attach (a "
                "pure-SWA model's prefill has no L3 path to read it from either)."
            )
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = get_schedule().page_size
        self.request_counter = 0
        self.tree_cache = tree_cache
        self.cache_controller = tree_cache.cache_controller
        self.host_pool_group = tree_cache.host_pool_group
        self.writes_swa = PoolName.SWA in self.host_pool_group.entry_map
        self.writes_mamba = PoolName.MAMBA in self.host_pool_group.entry_map

        env_stride = envs.SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE.get()
        if env_stride is None or env_stride <= 0:
            self.offload_stride = self.page_size
        else:
            self.offload_stride = max(
                self.page_size, (env_stride // self.page_size) * self.page_size
            )
        if self.writes_mamba:
            assert get_exec().mamba.enable_mamba_extra_buffer, (
                "Decode KV offload of a Mamba model needs Mamba state tracking"
            )
            track_grid = mamba_track_grid(self.page_size)
            self.offload_stride = math.lcm(self.offload_stride, track_grid)
            num_draft_tokens = max_speculative_num_draft_tokens() or 1
            assert track_grid > num_draft_tokens, (
                f"Mamba track grid {track_grid} must exceed the "
                f"{num_draft_tokens} tokens one decode step can commit"
            )
        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        self.ongoing_offload = {}
        self.ongoing_backup = {}
        # Keyed by Req identity (rids can be reused while a D2H copy is still
        # in flight); weak keys so a dropped Req is never pinned here.
        self.offloaded_state: WeakKeyDict[Req, OffloadedState] = WeakKeyDict()
        self.offload_inflight: WeakKeyDict[Req, int] = WeakKeyDict()
        self.chunks_without_mamba_state = 0
        logger.info(
            "Enable offload kv cache for decode side (stride %d tokens)",
            self.offload_stride,
        )

    def _mark_offload_started(self, req: Req):
        self.offload_inflight[req] = self.offload_inflight.get(req, 0) + 1

    def _mark_offload_finished(self, req: Req):
        count = self.offload_inflight.get(req, 0)
        if count <= 1:
            self.offload_inflight.pop(req, None)
        else:
            self.offload_inflight[req] = count - 1

    def _has_inflight_offload(self, req: Req):
        return self.offload_inflight.get(req, 0) > 0

    def _prefill_offloaded_len(self, req: Req) -> int:
        # Page-aligned prompt length; the prefill instance offloaded this part.
        return len(req.origin_input_ids) // self.page_size * self.page_size

    def has_inflight_device_transfer(self) -> bool:
        """Whether a device-to-host copy may still read unified KV pages."""
        return bool(self.ongoing_offload)

    def _chunk_transfers(
        self, req: Req, device_indices: torch.Tensor, end: int
    ) -> tuple[list[PoolTransfer], Optional[torch.Tensor]]:
        component_transfers: dict[ComponentType, list[PoolTransfer]] = {}
        if self.writes_swa:
            swa_indices = (
                self.token_to_kv_pool_allocator.translate_swa_indices_for_transfer(
                    device_indices
                )
            )
            # Page-aligned chunks have whole-page bindings; zero denotes the sink page.
            live_swa_indices = swa_indices[swa_indices > 0].to(torch.int64)
            assert live_swa_indices.numel() % self.page_size == 0
            if live_swa_indices.numel() > 0:
                component_transfers[ComponentType.SWA] = [
                    PoolTransfer(
                        name=PoolName.SWA,
                        device_indices=live_swa_indices,
                        hit_policy=PoolHitPolicy.TRAILING_PAGES,
                    )
                ]
        new_mamba_slot = None
        if self.writes_mamba:
            mamba_slots = self._reserve_mamba_state(req, end)
            if mamba_slots is not None:
                tracked_slot, new_mamba_slot = mamba_slots
                component_transfers[ComponentType.MAMBA] = [
                    PoolTransfer(
                        name=PoolName.MAMBA,
                        device_indices=tracked_slot,
                        hit_policy=PoolHitPolicy.TRAILING_PAGES,
                    )
                ]
        transfers = [t for ts in component_transfers.values() for t in ts]
        transfers.extend(
            self.tree_cache._build_sidecar_transfers(
                CacheTransferPhase.BACKUP_HOST,
                PoolTransfer(name=PoolName.KV, device_indices=device_indices),
                component_transfers,
            )
        )
        return transfers, new_mamba_slot

    def _reserve_mamba_state(
        self, req: Req, end: int
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if req.kv.mamba_last_track_seqlen != end:
            self._log_chunk_without_mamba_state(
                f"tracked at {req.kv.mamba_last_track_seqlen}, chunk ends at {end}"
            )
            return None
        pool = self.req_to_token_pool
        new_slot = pool.mamba_allocator.alloc(1)
        if new_slot is None:
            self._log_chunk_without_mamba_state("no free Mamba slot")
            return None
        # A copy: donation overwrites this buffer entry with new_slot.
        tracked_slot = (
            req.kv.mamba_ping_pong_track_buffer[pool.get_mamba_ping_pong_keep_idx(req)]
            .unsqueeze(-1)
            .clone()
        )
        return tracked_slot, new_slot

    def _donate_mamba_state(self, req: Req, new_slot: torch.Tensor) -> None:
        self.req_to_token_pool.donate_mamba_ping_pong_slot(req, new_slot)
        req.kv.mamba_last_track_seqlen = None
        req.kv.mamba_prev_track_seqlen = None

    def _log_chunk_without_mamba_state(self, reason: str) -> None:
        self.chunks_without_mamba_state += 1
        count = self.chunks_without_mamba_state
        if count & (count - 1) == 0:
            logger.warning(
                "Decode KV offload wrote %d chunk(s) without a Mamba state (%s)",
                count,
                reason,
            )

    def offload_kv_cache(self, req) -> bool:
        """Offload incremental KV cache for decode side."""

        if req.kv.req_pool_idx == -1 or len(req.output_ids) == 0:
            return False

        token_indices = self.req_to_token_pool.req_to_token[req.kv.req_pool_idx]
        if token_indices.dim() == 0 or token_indices.numel() == 0:
            return False

        # Prefill side offloads page-aligned origin_input_ids, decode side offloads the incremental part
        all_tokens = req.origin_input_ids + req.output_ids[:-1]
        prefill_offloaded_len = self._prefill_offloaded_len(req)
        state = self.offloaded_state.get(req)
        if state is None:
            prefill_hashes = self._compute_prefix_hash(
                req, req.origin_input_ids[:prefill_offloaded_len]
            )
            last_prefill_hash = (
                prefill_hashes[-1] if prefill_offloaded_len > 0 else None
            )
            state = OffloadedState(last_hash=last_prefill_hash)
            self.offloaded_state[req] = state
        start = prefill_offloaded_len + state.inc_len
        end = len(all_tokens) // self.offload_stride * self.offload_stride
        if end <= start:
            return False

        incremental_tokens = all_tokens[start:end]
        device_indices = token_indices[start:end].to(torch.int64)
        pool_transfers, new_mamba_slot = self._chunk_transfers(req, device_indices, end)

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
            if new_mamba_slot is not None:
                self.req_to_token_pool.mamba_allocator.free(new_mamba_slot)
            return False
        if new_mamba_slot is not None:
            self._donate_mamba_state(req, new_mamba_slot)

        self._mark_offload_started(req)
        self.ongoing_offload[ack_id] = (
            req,
            host_indices,
            incremental_tokens,
            time.time(),
            pool_transfers,
        )
        state.inc_len = end - prefill_offloaded_len
        return True

    def check_offload_progress(self):
        """Check the progress of offload from device to host and backup from host to storage."""
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
        self._check_offload_progress(n_write)
        self._check_backup_progress(n_backup)

    def drain_before_retraction(self):
        """Drain offloads, then refuse retraction on every rank if any rank still tracks a D2H copy."""
        self.check_offload_progress()
        cc = self.cache_controller
        # Retraction frees the device KV these copies read. Storage backups read only
        # host memory, so they do not count.
        pending = torch.tensor(
            [bool(cc.ack_write_queue or self.ongoing_offload or self.offload_inflight)],
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
                f"{len(self.offload_inflight)} requests)."
            )

    def _check_offload_progress(self, finish_count):
        """Check the progress of offload from device to host."""
        while finish_count > 0:
            ack = self.cache_controller.ack_write_queue.pop(0)
            ack.finish_event.synchronize()
            for ack_id in ack.node_ids:
                (
                    req,
                    host_indices,
                    incremental_tokens,
                    start_time,
                    pool_transfers,
                ) = self.ongoing_offload.pop(ack_id)

                self._mark_offload_finished(req)
                self._free_mamba_states(pool_transfers)
                prior_hash = (
                    self.offloaded_state[req].last_hash
                    if req in self.offloaded_state
                    else None
                )
                last_hash = self._trigger_backup(
                    req,
                    host_indices,
                    incremental_tokens,
                    start_time,
                    prior_hash,
                    pool_transfers,
                )
                if req in self.offloaded_state:
                    self.offloaded_state[req].last_hash = last_hash

                if req.finished() and not self._has_inflight_offload(req):
                    self._release_finished_req(req)
            finish_count -= 1

    def _free_mamba_states(self, pool_transfers: list[PoolTransfer]) -> None:
        for transfer in pool_transfers:
            if transfer.name == PoolName.MAMBA:
                self.req_to_token_pool.mamba_allocator.free(transfer.device_indices)

    def _release_finished_req(self, req: Req):
        # Defensive guard: ReqToTokenPool.free sets req_pool_idx to None,
        # so a previously-released request must be skipped here to avoid
        # non-idempotent side effects (double free, double unlock).
        if not req.kv.holds_kv:
            return

        # Released only at request finish; a mid-decode free races with
        # concurrent admission over live slots.
        release_kv_cache(req, self.tree_cache, checkpoint=False)
        self.offloaded_state.pop(req, None)

    def _check_backup_progress(self, finish_count):
        """Check the progress of backup from host to storage."""
        for _ in range(finish_count):
            storage_operation = self.cache_controller.ack_backup_queue.get()
            ack_id = storage_operation.id
            req_id, host_indices, start_time, pool_transfers = self.ongoing_backup.pop(
                ack_id
            )

            # Release host memory; a sidecar shares its source pool's indices.
            for transfer in pool_transfers:
                if transfer.indices_from_pool is None:
                    self.host_pool_group.free(transfer.host_indices, pool=transfer.name)
            self.host_pool_group.free(host_indices)
            if self.tree_cache.storage_metrics_collector is not None:
                self.tree_cache.storage_metrics_collector.log_backuped_tokens(
                    storage_operation.completed_tokens
                )

            logger.debug(
                f"Finished backup request {req_id}, free host memory, len:{len(host_indices)}, cost time:{time.time() - start_time:.2f} seconds."
            )

    def _trigger_backup(
        self,
        req,
        host_indices,
        incremental_tokens,
        start_time,
        prior_hash,
        pool_transfers=(),
    ):
        """Trigger async backup from host to storage."""
        page_hashes = self._compute_prefix_hash(req, incremental_tokens, prior_hash)
        keys_by_pool = {PoolName.KV: page_hashes}
        for transfer in pool_transfers:
            if transfer.indices_from_pool is not None:
                continue
            if transfer.name == PoolName.MAMBA:
                transfer.keys = page_hashes[-1:]
            else:
                page_count = transfer.device_indices.numel() // self.page_size
                transfer.keys = page_hashes[-page_count:]
            keys_by_pool[transfer.name] = transfer.keys
        for transfer in pool_transfers:
            if transfer.indices_from_pool is not None:
                transfer.keys = keys_by_pool[transfer.indices_from_pool]
        ack_id = self.cache_controller.write_storage(
            host_indices,
            incremental_tokens,
            hash_value=page_hashes,
            **({"extra_pools": pool_transfers} if pool_transfers else {}),
        )
        self.ongoing_backup[ack_id] = (
            req.rid,
            host_indices,
            start_time,
            pool_transfers,
        )
        return page_hashes[-1] if len(page_hashes) > 0 else prior_hash

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
        if self._has_inflight_offload(req):
            return
        self._release_finished_req(req)
