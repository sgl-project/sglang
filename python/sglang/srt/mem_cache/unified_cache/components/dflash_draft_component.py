"""Device-only radix content component for compact Qwen DFlash."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

import msgspec
import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.unified_cache.cache_action import FreeComponentDeviceSlot
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    ComponentType,
    EvictLayer,
    LRURefreshPhase,
    TreeComponent,
    next_component_uuid,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_cache.cache_action import (
        CacheAction,
        ComponentAction,
    )
    from sglang.srt.mem_cache.unified_radix_cache import (
        NodeId,
        UnifiedRadixCache,
        UnifiedTreeNode,
    )
    from sglang.srt.speculative.dflash_draft_content_allocator import (
        DFlashDraftContentAllocator,
    )

logger = logging.getLogger(__name__)


class DFlashDraftMatchPlan(msgspec.Struct, frozen=True):
    source_rows: torch.Tensor
    node_id: int
    boundary_uuid: int
    matched_tokens: int


class DFlashDraftPendingPublish(msgspec.Struct, frozen=True):
    start_seqlen: int
    rows: torch.Tensor


class DFlashDraftComponent(TreeComponent):
    """Canonical committed draft KV stored alongside the target radix tree."""

    component_type = ComponentType.DRAFT
    _MATCH_UUID_KEY = "dflash_match_uuid"

    def __init__(self, cache: UnifiedRadixCache, params: CacheInitParams):
        super().__init__(cache, params)
        if params.enable_session_radix_cache:
            raise ValueError("Qwen DFlash radix sidecar does not support sessions")
        if params.dflash_draft_content_allocator is None:
            raise ValueError("DFlashDraftComponent requires a content allocator")
        if not params.dflash_draft_window_size or params.dflash_draft_window_size <= 0:
            raise ValueError("DFlashDraftComponent requires a positive draft window")
        self.allocator: DFlashDraftContentAllocator = (
            params.dflash_draft_content_allocator
        )
        self.window_size = int(params.dflash_draft_window_size)
        self._match_plans: dict[str, DFlashDraftMatchPlan] = {}
        self._pending_publishes: dict[str, DFlashDraftPendingPublish] = {}
        self._evict_device_cursor = None
        self._published_tokens = 0
        self._evicted_tokens = 0

    def reset_component_state(self) -> None:
        self._match_plans.clear()
        self._pending_publishes.clear()
        self._evict_device_cursor = None
        self._published_tokens = 0
        self._evicted_tokens = 0
        self.allocator.clear()

    def reset_session_state(self) -> None:
        super().reset_session_state()

    def _dec_session_coverage(self, session_id: str, leaf: UnifiedTreeNode) -> None:
        raise AssertionError("DFlash draft sessions are disabled")

    def _advance_session_coverage(
        self,
        session_id: str,
        leaf: UnifiedTreeNode,
        old_ancestor: Optional[UnifiedTreeNode],
    ) -> None:
        raise AssertionError("DFlash draft sessions are disabled")

    def _recede_session_coverage(
        self,
        session_id: str,
        leaf: UnifiedTreeNode,
        fallback: Optional[UnifiedTreeNode],
    ) -> None:
        raise AssertionError("DFlash draft sessions are disabled")

    def refresh_lru(
        self,
        phase: LRURefreshPhase,
        node: UnifiedTreeNode,
        root_node: UnifiedTreeNode,
    ) -> None:
        if phase is LRURefreshPhase.WALKDOWN:
            return
        self.tree_core.lru_lists[
            self.component_type
        ].reset_node_and_window_ancestors_mru(
            node,
            root_node,
            self.window_size + self.tree_core.page_size,
            self.node_has_component_data,
        )

    def create_match_validator(
        self, match_device_only: bool = False
    ) -> Callable[[UnifiedTreeNode], bool]:
        state = {"covered": float("inf")}
        ct = self.component_type

        def validator(node: UnifiedTreeNode) -> bool:
            value = node.component_data[ct].value
            if value is None:
                state["covered"] = 0
                return False
            state["covered"] += len(value)
            return state["covered"] >= self.window_size

        return validator

    def finalize_match_result_in_cache(
        self, params: MatchPrefixParams, result: MatchResult
    ) -> MatchResult:
        req = params.req
        rid = getattr(req, "rid", None) if req is not None else None
        if rid is None:
            return result
        self.release_match_pin(rid)
        if params.pin_dflash_restore and not params.require_dflash_draft:
            raise RuntimeError("DFlash restore pin requires DRAFT coverage validation")
        if not params.pin_dflash_restore:
            return result
        if result.device_indices.numel() == 0:
            return result
        required_tokens = min(self.window_size, int(result.device_indices.numel()))
        self._pin_match(rid, result.best_match_node, required_tokens)
        return result._replace(dflash_match_rid=rid)

    def _pin_match(
        self, rid: str, node_id: NodeId, required_tokens: int
    ) -> DFlashDraftMatchPlan:
        root = self.tree_core.root_node
        node = self.tree_core.node_by_id(node_id)
        chunks: list[torch.Tensor] = []
        locked_nodes: list[UnifiedTreeNode] = []
        covered = 0
        while node is not root and covered < required_tokens:
            cd = node.component_data[self.component_type]
            if cd.value is None:
                raise RuntimeError("DFlash match lost content before it was pinned")
            chunks.append(cd.value)
            locked_nodes.append(node)
            covered += len(cd.value)
            node = node.parent
        if covered < required_tokens:
            raise RuntimeError("DFlash match has insufficient trailing coverage")

        boundary = locked_nodes[-1]
        boundary_cd = boundary.component_data[self.component_type]
        boundary_uuid = boundary_cd.metadata.get(self._MATCH_UUID_KEY)
        create_boundary_uuid = boundary_uuid is None
        if create_boundary_uuid:
            boundary_uuid = next_component_uuid()

        chunks.reverse()
        source_rows = torch.cat(chunks)[-required_tokens:].clone()
        self.allocator.assert_allocated(source_rows)
        plan = DFlashDraftMatchPlan(
            source_rows=source_rows,
            node_id=int(node_id),
            boundary_uuid=int(boundary_uuid),
            matched_tokens=required_tokens,
        )
        acquired_nodes: list[UnifiedTreeNode] = []
        boundary_uuid_published = False
        try:
            for locked in locked_nodes:
                cd = locked.component_data[self.component_type]
                if cd.lock_ref == 0:
                    size = len(cd.value)
                    self.tree_core.component_evictable_size_[
                        self.component_type
                    ] -= size
                    self.tree_core.component_protected_size_[
                        self.component_type
                    ] += size
                cd.lock_ref += 1
                acquired_nodes.append(locked)
                self.tree_core._update_evictable_leaf_sets(locked)

            if create_boundary_uuid:
                boundary_cd.metadata[self._MATCH_UUID_KEY] = boundary_uuid
                boundary_uuid_published = True
            self._match_plans[rid] = plan
        except BaseException:
            if (
                boundary_uuid_published
                and boundary_cd.metadata.get(self._MATCH_UUID_KEY) == boundary_uuid
            ):
                boundary_cd.metadata.pop(self._MATCH_UUID_KEY)
            for locked in reversed(acquired_nodes):
                cd = locked.component_data[self.component_type]
                cd.lock_ref -= 1
                if cd.lock_ref == 0:
                    size = len(cd.value)
                    self.tree_core.component_protected_size_[
                        self.component_type
                    ] -= size
                    self.tree_core.component_evictable_size_[
                        self.component_type
                    ] += size
                self.tree_core._update_evictable_leaf_sets(locked)
            raise
        return plan

    def get_match_plan(self, rid: str) -> Optional[DFlashDraftMatchPlan]:
        return self._match_plans.get(rid)

    def stage_pending_publish(
        self, rid: str, start_seqlen: int, rows: torch.Tensor
    ) -> None:
        if rid in self._pending_publishes:
            raise RuntimeError(f"DFlash request {rid!r} already has pending content")
        if start_seqlen < 0 or rows.numel() == 0:
            raise ValueError(
                "DFlash pending publish requires a nonnegative start and rows"
            )
        self.allocator.claim_lease(rows)
        try:
            self._pending_publishes[rid] = DFlashDraftPendingPublish(
                start_seqlen=int(start_seqlen), rows=rows
            )
        except BaseException:
            # Once claim_lease succeeds the worker no longer owns a lease. Keep
            # stage atomic by returning the rows here if publishing cannot finish.
            self.allocator.free(rows)
            raise

    def free_unstaged_content(self, rows: torch.Tensor) -> None:
        self.allocator.free_lease(rows)

    def release_match_pin(self, rid: str) -> bool:
        plan = self._match_plans.pop(rid, None)
        if plan is None:
            return False
        root = self.tree_core.root_node
        node = self.tree_core.node_by_id(plan.node_id)
        while node is not root:
            cd = node.component_data[self.component_type]
            if cd.value is not None and cd.lock_ref > 0:
                cd.lock_ref -= 1
                if cd.lock_ref == 0:
                    size = len(cd.value)
                    self.tree_core.component_protected_size_[
                        self.component_type
                    ] -= size
                    self.tree_core.component_evictable_size_[
                        self.component_type
                    ] += size
                self.tree_core._update_evictable_leaf_sets(node)
            if cd.metadata.get(self._MATCH_UUID_KEY) == plan.boundary_uuid:
                return True
            node = node.parent
        raise RuntimeError("DFlash match pin boundary disappeared before release")

    def release_request_state(self, rid: str) -> None:
        try:
            self.release_match_pin(rid)
        finally:
            # Abort/teardown must not leak pending content even if a corrupted
            # pin boundary makes release_match_pin raise.
            pending = self._pending_publishes.pop(rid, None)
            if pending is not None:
                self.allocator.free(pending.rows)

    def discard_pending_publish(self, rid: str) -> bool:
        pending = self._pending_publishes.pop(rid, None)
        if pending is None:
            return False
        self.allocator.free(pending.rows)
        return True

    @staticmethod
    def _incoming_span(
        params: InsertParams, span_start: int, span_len: int
    ) -> tuple[int, int, int, torch.Tensor] | None:
        if params.draft_value is None:
            return None
        publish_start = int(params.draft_start_seqlen)
        publish_end = publish_start + len(params.draft_value)
        start = max(span_start, publish_start)
        end = min(span_start + span_len, publish_end)
        if start >= end:
            return None
        value_start = start - publish_start
        return (
            start - span_start,
            end - start,
            value_start,
            params.draft_value[value_start : value_start + end - start],
        )

    @staticmethod
    def _mark_processed(params: InsertParams, start: int, count: int) -> None:
        if params.draft_processed is None:
            raise RuntimeError("DFlash draft insert requires a processed bitmap")
        if len(params.draft_processed) != len(params.draft_value):
            raise RuntimeError("DFlash draft processed bitmap length mismatch")
        processed = params.draft_processed[start : start + count]
        if bool(torch.any(processed)):
            raise RuntimeError("DFlash draft rows were processed more than once")
        processed[:] = True

    def _attach_span(
        self,
        node: UnifiedTreeNode,
        span_start: int,
        params: InsertParams,
        cache_actions: list[CacheAction | ComponentAction],
    ) -> None:
        incoming = self._incoming_span(params, span_start, len(node.key))
        if incoming is None:
            return
        start_offset, count, value_start, rows = incoming
        if start_offset:
            _, action = self.tree_core._split_node(node.key, node, start_offset)
            if action is not None:
                cache_actions.append(action)
            span_start += start_offset
        target = node
        if count < len(node.key):
            target, action = self.tree_core._split_node(node.key, node, count)
            if action is not None:
                cache_actions.append(action)

        cd = target.component_data[self.component_type]
        if cd.value is None:
            cd.value = rows.clone()
            self.tree_core.component_evictable_size_[self.component_type] += len(rows)
            self._published_tokens += len(rows)
            logger.debug(
                "DFLASH_RADIX published node_id=%s span_start=%d tokens=%d "
                "published_total=%d",
                target.id,
                span_start,
                len(rows),
                self._published_tokens,
            )
            lru = self.tree_core.lru_lists[self.component_type]
            if not lru.in_list(target):
                lru.insert_mru(target)
        else:
            cache_actions.append(
                FreeComponentDeviceSlot(
                    indices=[rows], component_type=self.component_type
                )
            )
        self._mark_processed(params, value_start, count)

    def update_component_on_insert_overlap(
        self,
        node: UnifiedTreeNode,
        prefix_len: int,
        total_prefix_len: int,
        value_slice: torch.Tensor,
        params: InsertParams,
        cache_actions: list[CacheAction | ComponentAction],
    ) -> int:
        self._attach_span(node, total_prefix_len, params, cache_actions)
        return prefix_len

    def commit_insert_component_data(
        self,
        node: UnifiedTreeNode,
        is_new_leaf: bool,
        params: InsertParams,
        result: InsertResult,
        cache_actions: list[CacheAction | ComponentAction],
    ) -> None:
        if is_new_leaf:
            self._attach_span(node, result.prefix_len, params, cache_actions)

    def cleanup_pending_rows(self, params: InsertParams) -> None:
        if params.draft_value is None:
            return
        if params.draft_processed is None:
            self.allocator.free(params.draft_value)
            params.draft_value = None
            return
        pending = params.draft_value[~params.draft_processed.to(torch.bool)]
        self.allocator.free(pending)
        params.draft_processed[:] = True

    def cleanup_after_caching_req(
        self,
        req: Req,
        is_finished: bool,
        insert_result: Optional[InsertResult] = None,
        insert_params: Optional[InsertParams] = None,
    ) -> None:
        if is_finished:
            # is_insert=False (skip-radix, abort, or teardown) bypasses prepare,
            # so a staged publish can still be owned by the request here.
            self.release_request_state(req.rid)
        if insert_params is not None:
            self.cleanup_pending_rows(insert_params)

    def prepare_for_caching_req(
        self,
        req: Req,
        insert_params: InsertParams,
        token_ids_len: int,
        is_finished: bool,
    ) -> Optional[int]:
        pending = self._pending_publishes.pop(req.rid, None)
        if pending is None:
            return None
        end_seqlen = pending.start_seqlen + len(pending.rows)
        if end_seqlen > token_ids_len:
            self.allocator.free(pending.rows)
            raise RuntimeError(
                "DFlash pending content exceeds the cacheable request prefix: "
                f"rid={req.rid!r}, end={end_seqlen}, token_ids_len={token_ids_len}"
            )
        insert_params.draft_start_seqlen = pending.start_seqlen
        insert_params.draft_value = pending.rows
        insert_params.draft_processed = torch.zeros(
            len(pending.rows), dtype=torch.bool, device="cpu"
        )
        return None

    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ) -> None:
        ct = self.component_type
        parent_cd = new_parent.component_data[ct]
        child_cd = child.component_data[ct]
        parent_cd.lock_ref = child_cd.lock_ref
        parent_cd.session_ref = child_cd.session_ref
        value = child_cd.value
        if value is not None:
            split_len = len(new_parent.key)
            parent_cd.value = value[:split_len].clone()
            child_cd.value = value[split_len:].clone()
        uuid = child_cd.metadata.pop(self._MATCH_UUID_KEY, None)
        if uuid is not None:
            parent_cd.metadata[self._MATCH_UUID_KEY] = uuid

    def evict_component(
        self,
        node: UnifiedTreeNode,
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:
        cd = node.component_data[self.component_type]
        if EvictLayer.HOST in target and cd.host_value is not None:
            raise AssertionError("DFlash draft sidecar must remain device-only")
        if EvictLayer.DEVICE not in target or cd.value is None:
            return 0, 0
        if cd.lock_ref:
            raise RuntimeError("attempted to evict pinned DFlash draft content")
        value = cd.value
        cd.value = None
        device_frees[self.component_type].append(value)
        self.tree_core.component_evictable_size_[self.component_type] -= len(value)
        self._evicted_tokens += len(value)
        logger.debug(
            "DFLASH_RADIX evicted node_id=%s tokens=%d evicted_total=%d",
            node.id,
            len(value),
            self._evicted_tokens,
        )
        return len(value), 0

    def eviction_priority(self, is_leaf: bool) -> int:
        return 0

    def _evict_device_start(self, request_cnt: int) -> None:
        self._evict_device_request_cnt = int(request_cnt)
        self._evict_device_cursor = self.tree_core.lru_lists[
            self.component_type
        ].get_lru_no_lock()

    def _evict_device_next_node(
        self,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> Optional[NodeId]:
        ct = self.component_type
        lru = self.tree_core.lru_lists[ct]
        node = self._evict_device_cursor
        if tracker[ct] >= self._evict_device_request_cnt or node is None:
            return None
        previous = lru.get_prev_no_lock(node)
        self.tree_core._evict_component_and_detach_lru(
            node,
            self,
            target=EvictLayer.DEVICE,
            tracker=tracker,
            device_frees=device_frees,
            host_frees=host_frees,
        )
        self.tree_core._update_evictable_leaf_sets(node)
        self._evict_device_cursor = previous
        # DRAFT eviction never returns a leaf for whole-node deletion.
        return None

    def _evict_device_end(self) -> None:
        self._evict_device_cursor = None

    def acquire_component_lock(
        self,
        node: UnifiedTreeNode,
        result: IncLockRefResult,
        lock_host: bool = False,
    ) -> IncLockRefResult:
        # Match finalization owns a short-lived, DRAFT-only restore pin.
        return result

    def release_component_lock(
        self,
        node: UnifiedTreeNode,
        params: Optional[DecLockRefParams],
        lock_host: bool = False,
    ) -> None:
        return None

    def free_host_values(self, host_values: list[torch.Tensor]) -> None:
        if host_values:
            raise AssertionError("DFlash draft sidecar has no host pool")

    def apply_component_action(self, action: ComponentAction) -> None:
        if isinstance(action, FreeComponentDeviceSlot):
            for indices in action.indices:
                self.allocator.free(indices)
            return
        raise AssertionError(
            f"DFlashDraftComponent: unhandled action {type(action).__name__}"
        )
