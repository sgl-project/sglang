from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.shared_hicache.manager import SharedHiCacheManager

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


class SharedHiCacheSchedulerMixin:
    def init_shared_hicache(self) -> None:
        self.shared_hicache_manager = SharedHiCacheManager.from_scheduler(self)

    def _prepare_shared_hicache_for_schedule(self, req: "Req") -> bool:
        shared_hicache_manager = getattr(self, "shared_hicache_manager", None)
        if shared_hicache_manager is None:
            return True

        req.shared_hicache_max_prefix_len = None
        has_reuse_plan = shared_hicache_manager.has_reuse_plan(req)
        any_rank_has_plan, all_ranks_have_plan = self._sync_shared_hicache_bool(
            has_reuse_plan
        )
        if not any_rank_has_plan:
            return True
        if not all_ranks_have_plan:
            logger.warning(
                "SharedHiCache plan availability diverged across TP ranks for rid=%s; "
                "falling back to local prefill",
                req.rid,
            )
            req.shared_hicache_plan = None
            self._release_shared_hicache_request(req.rid)
            return True

        pending = False
        result = None
        req.shared_hicache_max_prefix_len = 0
        try:
            # Probe the current local prefix without taking COW allocations.
            # The final schedule path below recomputes the prefix after remote pages land.
            req.init_next_round_input(self.tree_cache, cow_mamba=False)
            result = shared_hicache_manager.prepare_reuse(req)
            pending = result.pending
        except Exception:
            logger.exception(
                "SharedHiCache failed for rid=%s; continuing with local prefill",
                req.rid,
            )
            req.shared_hicache_plan = None
            self._release_shared_hicache_request(req.rid)
            pending = False

        any_rank_pending, _ = self._sync_shared_hicache_bool(pending)
        if any_rank_pending:
            return False
        if result is None:
            return True

        local_prefix_len = int(getattr(result, "prefix_len", 0))
        common_prefix_len = self._sync_shared_hicache_int_min(local_prefix_len)
        req.shared_hicache_max_prefix_len = common_prefix_len
        if common_prefix_len != local_prefix_len:
            logger.debug(
                "SharedHiCache clamped TP prefix for rid=%s local=%d common=%d",
                req.rid,
                local_prefix_len,
                common_prefix_len,
            )
        return True

    def _sync_shared_hicache_bool(self, value: bool) -> tuple[bool, bool]:
        group_size = self.ps.tp_size
        group = self.tp_cpu_group
        if group_size <= 1:
            return value, value

        flag = torch.tensor([1 if value else 0], dtype=torch.int32)
        torch.distributed.all_reduce(
            flag,
            op=torch.distributed.ReduceOp.SUM,
            group=group,
        )
        count = int(flag.item())
        return count > 0, count == group_size

    def _sync_shared_hicache_int_min(self, value: int) -> int:
        group_size = self.ps.tp_size
        group = self.tp_cpu_group
        if group_size <= 1:
            return int(value)

        tensor = torch.tensor([int(value)], dtype=torch.int64)
        torch.distributed.all_reduce(
            tensor,
            op=torch.distributed.ReduceOp.MIN,
            group=group,
        )
        return int(tensor.item())

    def _shared_hicache_tp_sync_enabled(self) -> bool:
        if getattr(self, "shared_hicache_manager", None) is None:
            return False
        return self.ps.tp_size > 1

    def _init_next_round_input_with_shared_hicache_tp_sync(self, req: "Req") -> None:
        requested_cap = req.shared_hicache_max_prefix_len
        if not self._shared_hicache_tp_sync_enabled() or requested_cap is None:
            req.init_next_round_input(self.tree_cache)
            req.shared_hicache_max_prefix_len = None
            return

        req.init_next_round_input(self.tree_cache, cow_mamba=False)
        common_prefix_len = self._sync_shared_hicache_int_min(len(req.prefix_indices))
        common_prefix_len = min(common_prefix_len, int(requested_cap))

        req.shared_hicache_max_prefix_len = common_prefix_len
        req.init_next_round_input(self.tree_cache)
        req.shared_hicache_max_prefix_len = None

    def _release_shared_hicache_request(self, rid: str) -> None:
        shared_hicache_manager = getattr(self, "shared_hicache_manager", None)
        if shared_hicache_manager is not None:
            shared_hicache_manager.release_request(rid)
