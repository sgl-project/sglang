from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.shared_hicache.manager import (
    SharedHiCacheManager,
    SharedHiCacheResult,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

_SHARED_HICACHE_PREPARE_LOOKAHEAD_EXTRA_REQS = 1
_SHARED_HICACHE_STATUS_FAILED = 0
_SHARED_HICACHE_STATUS_SKIP = 1
_SHARED_HICACHE_STATUS_PENDING = 2
_SHARED_HICACHE_STATUS_READY = 3


class SharedHiCacheSchedulerMixin:
    def init_shared_hicache(self) -> None:
        self.shared_hicache_manager = SharedHiCacheManager.from_scheduler(self)

    def _shared_hicache_schedule_candidates(
        self, waiting_queue: list["Req"], running_bs: int
    ) -> list["Req"]:
        if not waiting_queue:
            return []

        capacity = int(self.get_num_allocatable_reqs(running_bs))
        prefill_max_requests = getattr(
            getattr(self, "server_args", None), "prefill_max_requests", None
        )
        if prefill_max_requests is not None:
            capacity = min(capacity, max(0, int(prefill_max_requests)))

        if capacity <= 0:
            if getattr(self, "chunked_req", None) is None and not getattr(
                self, "enable_priority_preemption", False
            ):
                return []
            capacity = 1

        limit = capacity + _SHARED_HICACHE_PREPARE_LOOKAHEAD_EXTRA_REQS
        return waiting_queue[: min(len(waiting_queue), limit)]

    def _prepare_shared_hicache_for_schedule_batch(self, reqs: list["Req"]) -> set[str]:
        shared_hicache_manager = getattr(self, "shared_hicache_manager", None)
        if shared_hicache_manager is None or not reqs:
            return set()

        for req in reqs:
            req.shared_hicache_max_prefix_len = None

        probe_statuses: list[int] = []
        local_prefix_lens: list[int] = []
        pending_rids: set[str] = set()

        for req in reqs:
            if not shared_hicache_manager.has_reuse_plan(req):
                probe_statuses.append(_SHARED_HICACHE_STATUS_SKIP)
                local_prefix_lens.append(0)
                continue
            try:
                # Probe the current local prefix without taking COW allocations.
                req.init_next_round_input(self.tree_cache, cow_mamba=False)
                local_prefix_len = len(req.prefix_indices) + int(
                    getattr(req, "host_hit_length", 0) or 0
                )
                probe_status = _SHARED_HICACHE_STATUS_READY
            except Exception:
                logger.exception(
                    "SharedHiCache failed while probing local prefix for rid=%s; "
                    "falling back to local prefill on all TP ranks",
                    req.rid,
                )
                local_prefix_len = 0
                probe_status = _SHARED_HICACHE_STATUS_FAILED
            probe_statuses.append(probe_status)
            local_prefix_lens.append(local_prefix_len)

        local_reqs: list[tuple["Req", int]] = []
        reduced_probe_statuses = self._sync_shared_hicache_status_min_batch(
            probe_statuses
        )
        for req, local_status, reduced_status, local_prefix_len in zip(
            reqs, probe_statuses, reduced_probe_statuses, local_prefix_lens
        ):
            if reduced_status == _SHARED_HICACHE_STATUS_FAILED:
                req.shared_hicache_plan = None
                self._release_shared_hicache_request(req.rid)
                continue
            if reduced_status == _SHARED_HICACHE_STATUS_SKIP:
                if local_status == _SHARED_HICACHE_STATUS_READY:
                    logger.warning(
                        "SharedHiCache plan availability diverged across TP ranks for rid=%s; "
                        "falling back to local prefill",
                        req.rid,
                    )
                req.shared_hicache_plan = None
                self._release_shared_hicache_request(req.rid)
                continue
            local_reqs.append((req, local_prefix_len))

        if not local_reqs:
            return pending_rids

        common_local_prefix_lens = self._sync_shared_hicache_int_min_batch(
            [local_prefix_len for _, local_prefix_len in local_reqs]
        )

        prepared_reqs: list[tuple["Req", SharedHiCacheResult | None, int]] = []
        prepare_statuses: list[int] = []

        for (req, _), common_local_prefix_len in zip(
            local_reqs, common_local_prefix_lens
        ):
            result = None
            try:
                req.shared_hicache_max_prefix_len = common_local_prefix_len
                # Recompute using the TP-common local prefix before side-effecting
                # SharedHiCache prepare, so all ranks fetch/insert the same suffix.
                req.init_next_round_input(self.tree_cache, cow_mamba=False)
                result = shared_hicache_manager.prepare_reuse(req)
                prepare_status = (
                    _SHARED_HICACHE_STATUS_PENDING
                    if result.pending
                    else _SHARED_HICACHE_STATUS_READY
                )
            except Exception:
                logger.exception(
                    "SharedHiCache failed for rid=%s; falling back to local prefill on all TP ranks",
                    req.rid,
                )
                prepare_status = _SHARED_HICACHE_STATUS_FAILED

            prepared_reqs.append((req, result, common_local_prefix_len))
            prepare_statuses.append(prepare_status)

        reduced_prepare_statuses = self._sync_shared_hicache_status_min_batch(
            prepare_statuses
        )
        ready_reqs = []
        local_prefix_lens = []
        for (req, result, local_prefix_len), reduced_status in zip(
            prepared_reqs, reduced_prepare_statuses
        ):
            if reduced_status == _SHARED_HICACHE_STATUS_FAILED:
                req.shared_hicache_plan = None
                self._release_shared_hicache_request(req.rid)
                continue
            if reduced_status == _SHARED_HICACHE_STATUS_PENDING:
                pending_rids.add(str(req.rid))
                continue
            ready_reqs.append(req)
            shared_prefix_len = int(getattr(result, "prefix_len", 0))
            local_prefix_lens.append(max(local_prefix_len, shared_prefix_len))

        common_prefix_lens = self._sync_shared_hicache_int_min_batch(local_prefix_lens)
        for req, local_prefix_len, common_prefix_len in zip(
            ready_reqs, local_prefix_lens, common_prefix_lens
        ):
            req.shared_hicache_max_prefix_len = common_prefix_len
            if common_prefix_len != local_prefix_len:
                logger.debug(
                    "SharedHiCache clamped TP prefix for rid=%s local=%d common=%d",
                    req.rid,
                    local_prefix_len,
                    common_prefix_len,
                )

        return pending_rids

    def _sync_shared_hicache_status_min_batch(self, values: list[int]) -> list[int]:
        # Match disagg polling: lower status is less advanced and dominates.
        return self._sync_shared_hicache_int_min_batch(values)

    def _sync_shared_hicache_int_min_batch(self, values: list[int]) -> list[int]:
        if not values:
            return []
        group_size, group = self._shared_hicache_sync_group()
        if group_size <= 1:
            return [int(value) for value in values]

        tensor = torch.tensor([int(value) for value in values], dtype=torch.int64)
        torch.distributed.all_reduce(
            tensor,
            op=torch.distributed.ReduceOp.MIN,
            group=group,
        )
        return [int(value) for value in tensor.tolist()]

    def _shared_hicache_sync_group(self):
        ps = getattr(self, "ps", None)
        group_size = int(getattr(ps, "attn_tp_size", getattr(ps, "tp_size", 1)))
        group = getattr(self, "attn_tp_cpu_group", None)
        if group is None:
            group = getattr(self, "tp_cpu_group", None)
        return group_size, group

    def _init_next_round_input_with_shared_hicache_tp_sync(self, req: "Req") -> None:
        req.init_next_round_input(self.tree_cache)
        req.shared_hicache_max_prefix_len = None

    def _release_shared_hicache_request(self, rid: str) -> None:
        shared_hicache_manager = getattr(self, "shared_hicache_manager", None)
        if shared_hicache_manager is not None:
            shared_hicache_manager.release_request(rid)
