# SPDX-License-Identifier: Apache-2.0
"""Persistent-host cache control on Python radix copies."""

from __future__ import annotations

import hashlib
import threading
from array import array
from dataclasses import dataclass

import msgspec

from sglang.srt.managers.kv_control.contracts import (
    ContentSpan,
    ControlError,
    CopyTo,
    ExecutionUpdate,
    Location,
    Status,
    TargetOutcome,
    TokenRange,
    ResolvedTarget,
    resolve_prefix,
    Value,
    cache_compatibility,
    validated,
)
from sglang.srt.managers.kv_control.controller import ExecutionRequest
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.control_state import CopySelection


class ResidencyEntry(Value):
    branch_index: int
    descriptor: ContentSpan
    coverage_kind: str
    window_size: int | None
    lock_ref: int
    location: Location = Location(tier="gpu")


class ResidencySnapshot(Value):
    namespace: str
    engine_epoch: str
    cache_generation: int
    observed_at: float
    items: tuple[ResidencyEntry, ...]
    missing: tuple[TargetOutcome, ...]


class _StorageScope(Value):
    payload: CopyTo
    target: ResolvedTarget
    max_result_targets: int


@dataclass(frozen=True)
class _Row:
    selection: CopySelection
    branch: int
    prefix: tuple[int, ...]
    node_start: int
    node_end: int
    compatibility: object
    kind: str = "tokens"
    window: int | None = None


class UnifiedCacheExecutor:
    """One cache, namespace, epoch and owner thread with explicit capabilities.

    Cache reconstruction/reset invalidates this adapter. Construct a new adapter
    with the owner's new epoch after reset. Namespace changes also need a new
    cache configuration; a new control label cannot relabel existing KV.
    """

    def __init__(self, cache, *, engine_id: str, engine_epoch: str):
        self.cache = cache
        self.core = cache.tree_core
        self.control_state = getattr(self.core, "control_state", None)
        if self.control_state is None:
            raise ControlError("unsupported", "KV control is disabled")
        if threading.get_ident() != cache._kv_control_owner_thread:
            raise ControlError(
                "wrong_owner", "Bind cache control on the cache owner thread"
            )
        self.control_state.bind(engine_id, engine_epoch)
        self.config = self.control_state.config
        self.engine_id, self.engine_epoch = engine_id, engine_epoch
        self.layout_id = "python:{}:p{}:w{}".format(
            "+".join(ct.name.lower() for ct in self.core.component_types),
            self.core.page_size,
            cache._sliding_window_size or 0,
        )
        self.generation = self.control_state.generation
        self._thread = threading.get_ident()
        self.cache_salt_transform = None
        self.storage = None
        self.storage_adapter = None
        self.storage_reason = "Storage is not attached"
        self.storage_reads = None
        self.protection = None
        self.host_control = (
            self.core.enable_hicache and not self._transfer_unsupported_reason()
        )
        if self.host_control:
            self._refresh_storage_adapter()

    def _storage_prefix(self, work):
        self._check(work.target)
        scope = work.target
        end = scope.requested.end_token
        if scope.requested.start_token or end % self.core.page_size:
            raise ControlError(
                "unsupported", "Storage control requires a page-aligned whole prefix"
            )
        if end // self.core.page_size > self.config.max_query_nodes:
            raise ControlError(
                "capacity", "Transfer scope exceeds its page traversal budget"
            )
        return scope.content.token_ids[:end], self._compatibility(scope.content)

    def _load_pending(self, node):
        if node.load_back_pending_id is not None:
            return True
        # Write-through loads may only be tracked by their in-flight anchor.
        for anchor in self.cache.ongoing_load_back:
            current = self.core.node_by_id(anchor)
            while current is not None and current is not self.core.root_node:
                if current is node:
                    return True
                current = current.parent
        return False

    def _hold_host(self, holds, nodes):
        known = {node_id for node_id, _ in holds}
        for node in nodes:
            if (
                node.id not in known
                and node.component_data[ComponentType.FULL].host_value is not None
            ):
                receipt = self.cache.inc_host_lock_ref(node.id).to_dec_params()
                holds.append((node.id, receipt))

    def _release_host(self, holds):
        for node_id, receipt in reversed(holds):
            self.cache.dec_host_lock_ref(node_id, receipt)
        holds.clear()

    def _refresh_storage_adapter(self):
        from sglang.srt.mem_cache.unified_cache.storage_execution import (
            storage_execution_adapter,
        )

        adapter, reason = (None, "Storage is not attached")
        if self.host_control and self.cache.enable_storage:
            adapter, reason = storage_execution_adapter(self.cache)
        changed = self.storage_adapter is not None and (
            adapter is None
            or adapter.store is not self.storage_adapter.store
            or adapter.backend_namespace != self.storage_adapter.backend_namespace
        )
        if changed and self.storage is not None:
            if (
                self.storage.jobs
                or self.storage.interests
                or self.storage_reads.jobs
                or self.storage_reads.interests
                or self.storage_reads.demands
            ):
                raise ControlError(
                    "busy", "Storage control must drain before replacing its adapter"
                )
        self.storage_adapter, self.storage_reason = adapter, reason
        enabled = adapter is not None
        if enabled and self.storage is None:
            from sglang.srt.mem_cache.unified_cache.control_storage import (
                StorageCacheTransfers,
            )

            self.storage = StorageCacheTransfers(self)
            from sglang.srt.mem_cache.unified_cache.control_storage_reads import (
                StorageCacheReads,
            )

            self.storage_reads = StorageCacheReads(self)
        elif not enabled and self.storage is not None:
            if (
                self.storage.jobs
                or self.storage.interests
                or self.storage_reads.jobs
                or self.storage_reads.interests
                or self.storage_reads.demands
            ):
                raise ControlError(
                    "busy", "Storage control must drain before detaching"
                )
            self.storage = None
            self.storage_reads = None

    def _transfer_unsupported_reason(self):
        if self.core.component_types != (ComponentType.FULL,):
            return "Host control transfers require the built-in FULL component"
        if (
            getattr(self.cache, "buffer_pipeline", None) is not None
            or self.cache.sidecar_pool_specs
        ):
            return "Buffer/draft/sidecar transfers are not integrated"
        return ""

    def _check(self, target):
        if threading.get_ident() != self._thread:
            raise ControlError(
                "wrong_owner", "Cache control must run on its owning thread"
            )
        if (
            target.engine_epoch != self.engine_epoch
            or self.control_state.generation != self.generation
        ):
            raise ControlError("stale_epoch", "Cache control generation changed")
        if target.namespace != self.config.namespace:
            raise ControlError("incompatible_namespace", "Cache namespace differs")
        reason = self.cache._kv_control_unsupported_reason
        if self.core.enable_hicache:
            reason = reason or self._transfer_unsupported_reason()
        if reason or self.core.enable_external_cache_linker:
            raise ControlError(
                "unsupported",
                reason or "Direct linker control is not integrated",
            )

    def capabilities(self):
        self._refresh_storage_adapter()
        reason = self.cache._kv_control_unsupported_reason
        if self.core.enable_hicache:
            reason = reason or self._transfer_unsupported_reason()
        if self.control_state.generation != self.generation:
            reason = "Cache generation changed; construct a new epoch-bound executor"
        supported = not reason and not self.core.enable_external_cache_linker
        return {
            "namespace": self.config.namespace,
            "pin_protections": tuple(
                {
                    "tier": "host",
                    "condition": condition,
                    "owner": "engine",
                    "scope": "complete_full_prefix_at_existing_boundary",
                }
                for condition in ("eviction_exclusion", "eviction_priority")
            )
            if supported and self.protection is not None
            else (),
            "engine_epoch": self.engine_epoch,
            "cache_generation": self.generation,
            "tree_core": "python",
            "layout_id": self.layout_id,
            "actions": ("sglang.copy_to", "sglang.move_to")
            if supported and self.host_control
            else (),
            "components": tuple(ct.name.lower() for ct in self.core.component_types)
            if supported
            else (),
            "tier": "gpu",
            "directions": (
                {
                    "copy_to": self.storage_adapter.copy_paths
                    if self.storage_adapter
                    else (),
                    "move_to": self.storage_adapter.move_paths
                    if self.storage_adapter
                    else (),
                    "storage_backend": self.storage_adapter.backend
                    if self.storage_adapter
                    else None,
                    "backend_namespace": self.storage_adapter.backend_namespace
                    if self.storage_adapter
                    else None,
                    "storage_targets": self.storage_adapter is not None,
                    "storage_profile": self.storage_adapter.profile
                    if self.storage_adapter
                    else None,
                    "storage_adapter_reason": self.storage_reason,
                    "storage_scope": "single_complete_token_prefix"
                    if self.storage is not None
                    else None,
                    "granularity": "whole_token_prefix_at_existing_radix_boundary",
                    "page_alignment": self.core.page_size,
                    "max_scope_pages": self.config.max_query_nodes,
                }
                if supported and self.host_control
                else {}
            ),
            "reason": reason or ("" if supported else "Linker is unsupported"),
        }

    def storage_targets(self, target, location):
        """Trusted owner query: key descriptions, without a network existence check."""
        from sglang.srt.managers.kv_control.contracts import (
            Prefetch,
            PublicLocation,
            validate_public,
            validate_target,
        )
        from sglang.srt.managers.kv_control.public import normalize_public
        from sglang.srt.mem_cache.utils import get_storage_hash_str

        validate_target(target)
        validate_public(
            Prefetch(
                target=target, source=location, destination=PublicLocation(tier="host")
            )
        )
        self._refresh_storage_adapter()
        if self.storage_adapter is None:
            raise ControlError("unsupported", self.storage_reason)
        payload = normalize_public(
            Prefetch(
                target=target, source=location, destination=PublicLocation(tier="host")
            ),
            self.capabilities(),
        )
        service = self.cache.kv_control_service
        scope = resolve_prefix(
            payload.target,
            service.control.context("storage_targets"),
            service.config.limits,
        )
        self._check(scope)
        # The shared path validator checks complete-prefix geometry and budgets;
        # no operation, claim or transfer interest is created by this query.
        projection = _StorageScope(
            payload=payload,
            target=scope,
            max_result_targets=self.config.max_result_targets,
        )
        tokens, compatibility = self.storage_reads._scope(projection)
        hashes = get_storage_hash_str(
            RadixKey(
                array("q", tokens),
                extra_key=compatibility.extra_key,
                cache_salt=compatibility.cache_salt,
            ),
            page_size=self.core.page_size,
        )
        pages = self.storage_adapter.describe_pages(hashes)
        return dict(
            description_version="1.0",
            namespace=scope.namespace,
            compatibility=cache_compatibility(scope.content),
            key_salt_encoding="sglang.kv.weight.v1"
            if service.namespace_keys
            else "owner_assertion",
            page_size=self.core.page_size,
            location=self.storage_adapter.location(),
            existence="unknown",
            pages=tuple(
                dict(
                    start_token=i * self.core.page_size,
                    end_token=(i + 1) * self.core.page_size,
                    component="full",
                    storage_digest=h,
                    **page,
                )
                for i, (h, page) in enumerate(zip(hashes, pages))
            ),
        )

    def _compatibility(self, content):
        compatibility = cache_compatibility(content)
        if compatibility.extra_key is not None or compatibility.is_multimodal:
            raise ControlError(
                "unsupported", "LoRA/extra-key and multimodal targets are unsupported"
            )
        if compatibility.layout_id not in (None, self.layout_id):
            raise ControlError("unsupported", "Unsupported cache layout")
        return msgspec.structs.replace(
            compatibility,
            layout_id=self.layout_id,
            cache_salt=self.cache_salt_transform(compatibility.cache_salt)
            if self.cache_salt_transform is not None
            else compatibility.cache_salt,
        )

    def _walk(self, tokens, compatibility):
        key = RadixKey(
            array("q", tokens),
            extra_key=compatibility.extra_key,
            cache_salt=compatibility.cache_salt,
        )
        node, offset, path = self.core.root_node, 0, []
        while offset + self.core.page_size <= len(key):
            child = node.children.get(key.child_key_at(offset, self.core.page_size))
            if child is None:
                break
            if child.key.is_bigram:
                raise ControlError("unsupported", "Bigram radix content is unsupported")
            matched = child.key.match_at(key, offset, self.core.page_size)
            if not matched:
                break
            path.append((child, offset, offset + len(child.key), offset + matched))
            if len(path) > self.config.max_query_nodes:
                raise ControlError("capacity", "Radix traversal exceeds query budget")
            offset += matched
            if matched != len(child.key):
                break
            node = child
        return path

    def _content_id(self, prefix, component, compatibility):
        body = (self.config.namespace, compatibility, component.name, prefix)
        return hashlib.sha256(msgspec.json.encode(body)).hexdigest()

    @staticmethod
    def _gap(branch, component, start, end, *, point=False, code="missing_copy"):
        return TargetOutcome(
            branch_index=branch,
            coverage=TokenRange(
                component=component.name.lower(), start_token=start, end_token=end
            ),
            coverage_kind="checkpoint" if point else "tokens",
            status=Status.MISSING if code == "missing_copy" else Status.UNSUPPORTED,
            location=Location(tier="gpu"),
            code=code,
        )

    def _select(
        self,
        prefix,
        compatibility,
        branch,
        start,
        end,
        components,
        *,
        tier="gpu",
    ):
        rows, missing = [], []
        path = self._walk(prefix[:end], compatibility)
        for component in components:
            if component == ComponentType.MAMBA:
                candidate = next(
                    ((node, lo, hi) for node, lo, hi, _ in path if hi == end), None
                )
                if (
                    candidate
                    and candidate[0].component_data[component].value is not None
                ):
                    node, lo, hi = candidate
                    if (
                        node.load_back_pending_id is not None
                        or node.write_through_pending_id is not None
                    ):
                        raise ControlError(
                            "unsupported",
                            "In-flight checkpoints need transfer integration",
                        )
                    if len(node.component_data[component].value) != 1:
                        raise ControlError(
                            "unsupported", "Unsupported checkpoint slot layout"
                        )
                    rows.append(
                        _Row(
                            CopySelection(node, component, end, end),
                            branch,
                            prefix,
                            lo,
                            hi,
                            compatibility,
                            "checkpoint",
                        )
                    )
                else:
                    missing.append(self._gap(branch, component, end, end, point=True))
                continue
            size = (
                self.cache._sliding_window_size
                if component == ComponentType.SWA
                else None
            )
            if size is not None:
                size = (
                    (size + self.core.page_size - 1)
                    // self.core.page_size
                    * self.core.page_size
                )
            selected_start = max(start, end - size) if size is not None else start
            cursor = selected_start
            for node, lo, hi, matched in path:
                left, right = max(selected_start, lo), min(end, matched)
                if left >= right:
                    continue
                if (
                    node.load_back_pending_id is not None
                    or node.write_through_pending_id is not None
                    or (
                        tier == "gpu" and self.host_control and self._load_pending(node)
                    )
                ):
                    raise ControlError(
                        "unsupported",
                        "In-flight cache copies need transfer integration",
                    )
                cd = node.component_data[component]
                value = cd.value if tier == "gpu" else cd.host_value
                if value is not None:
                    if len(value) != len(node.key):
                        raise ControlError(
                            "unsupported",
                            "Component has an unsupported physical span layout",
                        )
                    if cursor < left:
                        missing.append(self._gap(branch, component, cursor, left))
                    rows.append(
                        _Row(
                            CopySelection(node, component, left, right),
                            branch,
                            prefix,
                            lo,
                            hi,
                            compatibility,
                            "window" if size is not None else "tokens",
                            size,
                        )
                    )
                    cursor = right
            if cursor < end:
                missing.append(self._gap(branch, component, cursor, end))
        return rows, missing

    def _resolve(self, target, limit, *, tier="gpu"):
        self._check(target)
        if type(limit) is not int or not 1 <= limit <= self.config.max_result_targets:
            raise ControlError("capacity", "Invalid cache result budget")
        start, end = target.requested.start_token, target.requested.end_token
        page = self.core.page_size
        left, right = (start + page - 1) // page * page, end // page * page
        rows, missing = [], []
        if (left, right) != (start, end):
            if not target.allow_clip:
                raise ControlError("unsupported", "Target range is not page-aligned")
            if start < min(left, end):
                missing.append(
                    self._gap(
                        0, ComponentType.FULL, start, min(left, end), code="unaligned"
                    )
                )
            if max(right, left) < end:
                missing.append(
                    self._gap(
                        0, ComponentType.FULL, max(right, left), end, code="unaligned"
                    )
                )
        if left < right:
            rows, gaps = self._select(
                target.content.token_ids,
                self._compatibility(target.content),
                0,
                left,
                right,
                self.core.component_types,
                tier=tier,
            )
            missing.extend(gaps)
        if len(rows) + len(missing) > limit:
            raise ControlError("capacity", "Cache result scope exceeds budget")
        return rows, missing

    def _descriptor(self, row):
        selected = row.selection
        return ContentSpan(
            content_id=self._content_id(
                row.prefix[: selected.end], selected.component, row.compatibility
            ),
            component=selected.component.name.lower(),
            start_token=selected.start,
            end_token=selected.end,
            coverage_kind=row.kind,
            window_size=row.window,
        )

    def pin(self, *, ref, payload, scope):
        if threading.get_ident() != self._thread:
            raise ControlError("wrong_owner", "PIN must run on its cache owner")
        if (
            ref.context.engine_id != self.engine_id
            or ref.context.engine_epoch != self.engine_epoch
        ):
            raise ControlError("stale_epoch", "PIN owner identity differs")
        if self.protection is None:
            raise ControlError("unsupported", "L2 protection is not installed")
        return self.protection.start(ref=ref, payload=payload, scope=scope)

    def start(self, work: ExecutionRequest) -> ExecutionUpdate:
        try:
            self._refresh_storage_adapter()
            payload = work.payload
            for location in (payload.src, payload.dst):
                if location.backend_namespace is not None and (
                    location.tier != "storage"
                    or self.storage_adapter is None
                    or location.backend_namespace
                    != self.storage_adapter.backend_namespace
                ):
                    raise ControlError(
                        "incompatible_namespace", "Storage backend namespace differs"
                    )
            if (
                work.ref.context.engine_id != self.engine_id
                or work.ref.context.engine_epoch != self.engine_epoch
            ):
                raise ControlError("stale_epoch", "Executor owner identity differs")
            if not self.host_control:
                raise ControlError("unsupported", "Cache movement requires HiCache")
            if payload.dst.tier == "storage" and self.storage is not None:
                return self.storage.start(work)
            if payload.src.tier == "storage" and self.storage_reads is not None:
                return self.storage_reads.start(work)
            raise ControlError("unsupported", "Only L2 to/from storage is supported")
        except ControlError as error:
            status = {
                "unsupported": Status.UNSUPPORTED,
                "missing": Status.MISSING,
                "unavailable": Status.DEFERRED,
            }.get(error.code, Status.REJECTED)
            return ExecutionUpdate(
                status=status, quiescent=True, code=error.code, reason=str(error)
            )

    def poll(self, ref):
        if (
            self.storage_reads is not None
            and ref.operation_id in self.storage_reads.interests
        ):
            return self.storage_reads.poll(ref)
        if self.storage is not None and ref.operation_id in self.storage.interests:
            return self.storage.poll(ref)
        return None

    def begin_poll(self, *, limit=64):
        if self.storage is not None:
            self.storage.begin_poll(limit=limit)
        if self.storage_reads is not None:
            self.storage_reads.begin_poll(limit=limit)

    def cancel(self, ref):
        if (
            self.storage_reads is not None
            and ref.operation_id in self.storage_reads.interests
        ):
            return self.storage_reads.cancel(ref)
        if self.storage is not None and ref.operation_id in self.storage.interests:
            return self.storage.cancel(ref)
        return ExecutionUpdate(status=Status.CANCELLED, quiescent=True)

    def residency(
        self,
        target: ResolvedTarget,
        *,
        limit: int = 256,
        location: Location | None = None,
    ) -> ResidencySnapshot:
        location = validated(location or Location(tier="gpu"), Location)
        if (
            location.engine_id not in (None, self.engine_id)
            or location.backend is not None
            or location.tier not in ("gpu", "host")
        ):
            raise ControlError(
                "unsupported", "Residency inspection supports local GPU/host tiers"
            )
        if location.tier == "host" and not self.host_control:
            raise ControlError(
                "unsupported", "Host residency inspection is not configured"
            )
        location = msgspec.structs.replace(location, engine_id=self.engine_id)
        rows, missing = self._resolve(target, limit, tier=location.tier)
        return ResidencySnapshot(
            namespace=self.config.namespace,
            engine_epoch=self.engine_epoch,
            cache_generation=self.generation,
            observed_at=self.control_state.clock(),
            items=tuple(
                ResidencyEntry(
                    branch_index=row.branch,
                    descriptor=self._descriptor(row),
                    coverage_kind=row.kind,
                    window_size=row.window,
                    lock_ref=getattr(
                        row.selection.node.component_data[row.selection.component],
                        "lock_ref" if location.tier == "gpu" else "host_lock_ref",
                    ),
                    location=location,
                )
                for row in rows
            ),
            missing=tuple(
                msgspec.structs.replace(gap, location=location) for gap in missing
            ),
        )
