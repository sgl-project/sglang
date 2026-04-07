import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.speculative.spectre.spectre_protocol import SpecType

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

logger = logging.getLogger(__name__)


class SpectreKVRollbacker:
    def __init__(
        self,
        token_to_kv_pool_allocator: "TokenToKVPoolAllocator",
        req_to_token_pool: "ReqToTokenPool",
        tree_cache: "BasePrefixCache",
        page_size: int = 1,
        promote_interval: int = 50,
        num_draft_tokens: int = 5,
        tp_rank: int = 0,
    ) -> None:
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.token_to_kv_pool = token_to_kv_pool_allocator
        self.req_to_token_pool = req_to_token_pool
        self.tree_cache = tree_cache
        self.page_size = page_size
        self.tp_rank = tp_rank

    def get_prefix_len(self, req: "Req") -> int:
        if hasattr(req, "prefix_indices") and req.prefix_indices is not None:
            if isinstance(req.prefix_indices, torch.Tensor):
                return len(req.prefix_indices)
            elif isinstance(req.prefix_indices, list):
                return len(req.prefix_indices)
        return 0

    def can_local_rollback(self, req: "Req", fork_point: int) -> bool:
        if self.page_size > 1:
            return False

        prefix_len = self.get_prefix_len(req)
        return fork_point >= prefix_len

    def rollback(
        self, req: "Req", fork_point: int, current_kv_len: Optional[int] = None
    ) -> bool:
        if current_kv_len is None:
            input_ids = getattr(req, "origin_input_ids", [])
            output_ids = getattr(req, "output_ids", [])
            input_len = len(input_ids) if input_ids is not None else 0
            output_len = len(output_ids) if output_ids is not None else 0
            total_len = input_len + output_len
            current_kv_len = max(0, total_len - 1)

        if self.can_local_rollback(req, fork_point):
            return self.local_rollback(req, fork_point, current_kv_len)
        else:
            return False

    def local_rollback(
        self,
        req: "Req",
        fork_point: int,
        current_kv_len: int,
    ) -> bool:
        if self.page_size > 1:
            if self.tp_rank == 0:
                logger.warning(
                    f"[SpectreKVRollbacker] local_rollback called with page_size={self.page_size}, "
                    "this is not safe! Use re-prefill instead."
                )
            return False

        if req.req_pool_idx is None:
            return False

        if fork_point >= current_kv_len:
            return False

        prefix_len = self.get_prefix_len(req)
        if fork_point < prefix_len:
            if self.tp_rank == 0:
                logger.error(
                    f"[SpectreKVRollbacker] local_rollback called with fork_point={fork_point} < "
                    f"prefix_len={prefix_len}, this would free RadixCache indices! Aborting."
                )
            return False

        try:
            max_len = self.req_to_token_pool.req_to_token.shape[1]
            start = min(fork_point, max_len)
            end = min(current_kv_len, max_len)

            if start >= end:
                if self.tp_rank == 0:
                    logger.warning(
                        f"[SpectreKVRollbacker] Invalid rollback range for {req.rid}: "
                        f"start={start}, end={end}, max_len={max_len}"
                    )
                return False

            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, start:end
            ]

            self.token_to_kv_pool_allocator.free(kv_indices)
            old_committed = req.kv_committed_len
            old_allocated = req.kv_allocated_len

            req.kv_committed_len = fork_point
            req.kv_allocated_len = fork_point

            if self.tp_rank == 0:
                logger.debug(
                    f"[SpectreKVRollbacker] Local rollback for {req.rid}: "
                    f"freed [{fork_point}, {current_kv_len}), "
                    f"kv_committed: {old_committed} -> {req.kv_committed_len}, "
                    f"kv_allocated: {old_allocated} -> {req.kv_allocated_len}, "
                    f"prefix_len={prefix_len}"
                )
            return True
        except Exception as e:
            if self.tp_rank == 0:
                logger.warning(
                    f"[SpectreKVRollbacker] Failed to local_rollback for {req.rid}: {e}"
                )
            return False

    def release_all_kv_for_finished_req(self, req: "Req") -> None:
        if req.req_pool_idx is None:
            return

        kv_len = req.kv_committed_len
        req.fill_ids = (req.origin_input_ids + req.output_ids)[:kv_len]

        release_kv_cache(req, self.tree_cache)

        if self.tp_rank == 0:
            logger.debug(
                f"[SpectreKVRollbacker] Released all KV for finished request {req.rid}"
            )

    def release_all_kv_for_reprefill_req(self, req: "Req") -> None:
        if req.req_pool_idx is None:
            return

        kv_len = req.kv_committed_len
        req.fill_ids = (req.origin_input_ids + req.output_ids)[:kv_len]

        release_kv_cache(req, self.tree_cache)

        if self.tp_rank == 0:
            logger.debug(
                f"[SpectreKVRollbacker] Released all KV for re-prefill {req.rid}"
            )

    def release_all_kv_for_finish(self, req: "Req") -> None:
        self.release_all_kv_for_finished_req(req)

    def release_all_kv_for_reprefill(self, req: "Req") -> None:
        self.release_all_kv_for_reprefill_req(req)

    def release_all_kv(self, req: "Req") -> None:
        self.release_all_kv_for_finished_req(req)

    def compute_unstable_tokens(self, req: "Req") -> int:
        if req is None or getattr(req, "req_pool_idx", None) is None:
            return 0

        if not hasattr(req, "spec_type") or req.spec_type != SpecType.DRAFT_REQUEST:
            return 0

        total_len = len(getattr(req, "origin_input_ids", [])) + len(
            getattr(req, "output_ids", [])
        )
        current_kv_len = max(0, total_len - 1)
        prefix_len = self.get_prefix_len(req)

        return max(0, current_kv_len - prefix_len)

    def compute_total_unstable_tokens(
        self,
        paused_reqs: list,
        running_batch_reqs: list,
        waiting_queue: list,
    ) -> int:
        total = 0
        seen_rids = set()

        for req in paused_reqs:
            if req.rid not in seen_rids:
                total += self.compute_unstable_tokens(req)
                seen_rids.add(req.rid)

        for req in running_batch_reqs:
            if req.rid not in seen_rids:
                total += self.compute_unstable_tokens(req)
                seen_rids.add(req.rid)

        for req in waiting_queue:
            if req.rid not in seen_rids:
                total += self.compute_unstable_tokens(req)
                seen_rids.add(req.rid)

        return total

    def get_memory_stats(self, req: "Req") -> dict:
        if req is None:
            return {"rid": "unknown", "status": "no_request"}

        total_len = len(getattr(req, "origin_input_ids", [])) + len(
            getattr(req, "output_ids", [])
        )
        current_kv_len = max(0, total_len - 1)
        prefix_len = self.get_prefix_len(req)
        has_kv = getattr(req, "req_pool_idx", None) is not None

        return {
            "rid": req.rid,
            "total_len": total_len,
            "current_kv_len": current_kv_len,
            "prefix_len": prefix_len,
            "unstable": max(0, current_kv_len - prefix_len),
            "has_kv": has_kv,
            "strategy": "local_rollback",
        }
