# SPDX-License-Identifier: Apache-2.0
"""Synchronous native priority execution against actual Python radix copies."""

from __future__ import annotations

import hashlib
import threading
from array import array
from dataclasses import dataclass, replace

import msgspec

from sglang.srt.managers.kv_control.contracts import (
    ContentSpan,
    ControlError,
    ExecutionUpdate,
    Location,
    SetPriority,
    Status,
    TargetOutcome,
    TokenRange,
    TokensTarget,
    Value,
    cache_compatibility,
)
from sglang.srt.managers.kv_control.controller import ExecutionRequest
from sglang.srt.managers.kv_control.index import ResolvedTarget
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.priority_claims import CopySelection


class ResidencyEntry(Value):
    branch_index: int
    descriptor: ContentSpan
    coverage_kind: str
    window_size: int | None
    effective_priority: int
    lock_ref: int


class ResidencySnapshot(Value):
    namespace: str
    engine_epoch: str
    cache_generation: int
    observed_at: float
    items: tuple[ResidencyEntry, ...]
    missing: tuple[TargetOutcome, ...]


@dataclass(frozen=True)
class _Row:
    selection: CopySelection
    branch: int
    prefix: tuple[int, ...]
    node_start: int
    node_end: int
    compatibility: object
    content_id: str | None = None
    kind: str = "tokens"
    window: int | None = None


class UnifiedCacheExecutor:
    """One cache, namespace, epoch and owner thread; no transfer implementation.

    Cache reconstruction/reset invalidates this adapter. Construct a new adapter
    with the owner's new epoch after reset. Namespace changes also need a new
    cache configuration; a new control label cannot relabel existing KV.
    """

    def __init__(self, cache, *, engine_id: str, engine_epoch: str):
        self.cache = cache
        self.core = cache.tree_core
        self.claims = getattr(self.core, "priority_claims", None)
        if self.claims is None:
            raise ControlError("unsupported", "Cache copy priorities are disabled")
        if threading.get_ident() != cache._kv_control_owner_thread:
            raise ControlError(
                "wrong_owner", "Bind cache control on the cache owner thread"
            )
        self.claims.bind(engine_id, engine_epoch)
        self.config = self.claims.config
        self.engine_id, self.engine_epoch = engine_id, engine_epoch
        self.layout_id = "python:{}:p{}:w{}".format(
            "+".join(ct.name.lower() for ct in self.core.component_types),
            self.core.page_size,
            cache._sliding_window_size or 0,
        )
        self.generation = self.claims.generation
        self._thread = threading.get_ident()

    def _check(self, target):
        if threading.get_ident() != self._thread:
            raise ControlError(
                "wrong_owner", "Cache control must run on its owning thread"
            )
        if (
            target.engine_epoch != self.engine_epoch
            or self.claims.generation != self.generation
        ):
            raise ControlError("stale_epoch", "Cache control generation changed")
        if target.namespace != self.config.namespace:
            raise ControlError("incompatible_namespace", "Cache namespace differs")
        reason = self.cache._kv_control_unsupported_reason
        if reason or self.core.enable_hicache or self.core.enable_external_cache_linker:
            raise ControlError(
                "unsupported",
                reason or "HiCache/linker priority execution is not integrated",
            )

    def capabilities(self):
        reason = self.cache._kv_control_unsupported_reason
        if self.claims.generation != self.generation:
            reason = "Cache generation changed; construct a new epoch-bound executor"
        supported = (
            not reason
            and not self.core.enable_hicache
            and not self.core.enable_external_cache_linker
        )
        return {
            "namespace": self.config.namespace,
            "engine_epoch": self.engine_epoch,
            "cache_generation": self.generation,
            "tree_core": "python",
            "layout_id": self.layout_id,
            "actions": ("sglang.set_priority",) if supported else (),
            "components": tuple(ct.name.lower() for ct in self.core.component_types)
            if supported
            else (),
            "tier": "gpu",
            "directions": (),
            "reason": reason or ("" if supported else "HiCache/linker is unsupported"),
        }

    def _compatibility(self, content):
        compatibility = cache_compatibility(content)
        if compatibility.extra_key is not None or compatibility.is_multimodal:
            raise ControlError(
                "unsupported", "LoRA/extra-key and multimodal targets are unsupported"
            )
        if compatibility.layout_id not in (None, self.layout_id):
            raise ControlError("unsupported", "Unsupported cache layout")
        return msgspec.structs.replace(compatibility, layout_id=self.layout_id)

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
        window=True,
        content_id=None,
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
                            content_id,
                            "checkpoint",
                        )
                    )
                else:
                    missing.append(self._gap(branch, component, end, end, point=True))
                continue
            size = (
                self.cache._sliding_window_size
                if component == ComponentType.SWA and window
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
                ):
                    raise ControlError(
                        "unsupported",
                        "In-flight cache copies need transfer integration",
                    )
                cd = node.component_data[component]
                if cd.value is not None:
                    if len(cd.value) != len(node.key):
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
                            content_id,
                            "window" if size is not None else "tokens",
                            size,
                        )
                    )
                    cursor = right
            if cursor < end:
                missing.append(self._gap(branch, component, cursor, end))
        return rows, missing

    def _descriptor_prefixes(self, content, compatibility):
        # Explicit descriptor queries are bounded. There is no background hash
        # index or per-page metadata for ordinary cache nodes.
        if len(self.core._node_arena) > self.config.max_query_nodes:
            raise ControlError("capacity", "Descriptor lookup exceeds node budget")
        wanted = {span.content_id: span for span in content.spans}
        found = {}
        stack = [(child, ()) for child in self.core.root_node.children.values()]
        while stack:
            node, prefix = stack.pop()
            if (
                node.key.cache_salt != compatibility.cache_salt
                or node.key.extra_key != compatibility.extra_key
                or node.key.is_bigram
            ):
                continue
            tokens = prefix + tuple(node.key.token_ids)
            if len(tokens) > 65536:
                raise ControlError("capacity", "Descriptor prefix exceeds token budget")
            for identity, span in wanted.items():
                if len(prefix) < span.end_token <= len(tokens):
                    ct = self._component(span.component)
                    if (
                        self._content_id(tokens[: span.end_token], ct, compatibility)
                        == identity
                    ):
                        found[identity] = tokens[: span.end_token]
            stack.extend((child, tokens) for child in node.children.values())
        return found

    def _component(self, name):
        component = {ct.name.lower(): ct for ct in self.core.component_types}.get(name)
        if component is None:
            raise ControlError("unsupported", "Component is not configured")
        return component

    def _resolve(self, target, limit):
        self._check(target)
        if type(limit) is not int or not 1 <= limit <= self.config.max_result_targets:
            raise ControlError("capacity", "Invalid cache result budget")
        rows, missing = [], []
        page = self.core.page_size
        for branch_index, branch in enumerate(target.branches):
            content = branch.content
            compatibility = self._compatibility(content)
            if isinstance(content, TokensTarget):
                start, end = branch.requested.start_token, branch.requested.end_token
                clipped_start, clipped_end = (
                    (start + page - 1) // page * page,
                    end // page * page,
                )
                if (clipped_start, clipped_end) != (start, end):
                    if not target.allow_clip:
                        raise ControlError(
                            "unsupported", "Target range is not page-aligned"
                        )
                    if start < min(clipped_start, end):
                        missing.append(
                            self._gap(
                                branch_index,
                                ComponentType.FULL,
                                start,
                                min(clipped_start, end),
                                code="unaligned",
                            )
                        )
                    if max(clipped_end, clipped_start) < end:
                        missing.append(
                            self._gap(
                                branch_index,
                                ComponentType.FULL,
                                max(clipped_end, clipped_start),
                                end,
                                code="unaligned",
                            )
                        )
                if clipped_start < clipped_end:
                    selected, gaps = self._select(
                        content.token_ids,
                        compatibility,
                        branch_index,
                        clipped_start,
                        clipped_end,
                        self.core.component_types,
                    )
                    rows.extend(selected)
                    missing.extend(gaps)
            else:
                for name in content.components:
                    self._component(name)
                prefixes = self._descriptor_prefixes(content, compatibility)
                for span in branch.spans:
                    component = self._component(span.component)
                    if (component == ComponentType.MAMBA) != (
                        span.coverage_kind == "checkpoint"
                    ):
                        raise ControlError(
                            "unsupported", "Component descriptor geometry differs"
                        )
                    if span.coverage_kind == "window":
                        expected = (
                            ((self.cache._sliding_window_size or 0) + page - 1)
                            // page
                            * page
                        )
                        if (
                            component != ComponentType.SWA
                            or span.window_size != expected
                        ):
                            raise ControlError(
                                "unsupported",
                                "Descriptor window differs from the cache layout",
                            )
                    if span.start_token % page or span.end_token % page:
                        raise ControlError(
                            "unsupported", "Descriptor is not page-aligned"
                        )
                    prefix = prefixes.get(span.content_id)
                    if prefix is None:
                        missing.append(
                            self._gap(
                                branch_index,
                                component,
                                span.start_token,
                                span.end_token,
                                point=span.coverage_kind == "checkpoint",
                            )
                        )
                    else:
                        selected, gaps = self._select(
                            prefix,
                            compatibility,
                            branch_index,
                            span.start_token,
                            span.end_token,
                            (component,),
                            window=False,
                            content_id=span.content_id,
                        )
                        rows.extend(
                            replace(
                                row, kind=span.coverage_kind, window=span.window_size
                            )
                            for row in selected
                        )
                        missing.extend(gaps)
                for gap in branch.missing:
                    missing.append(
                        self._gap(
                            branch_index,
                            self._component(gap.coverage.component),
                            gap.coverage.start_token,
                            gap.coverage.end_token,
                        )
                    )
                for span in branch.unaligned:
                    if not target.allow_clip:
                        raise ControlError(
                            "unsupported", "Exact range splits a logical descriptor"
                        )
                    missing.append(
                        self._gap(
                            branch_index,
                            self._component(span.component),
                            max(span.start_token, branch.requested.start_token),
                            min(span.end_token, branch.requested.end_token),
                            code="unaligned",
                        )
                    )
            if len(rows) + len(missing) > limit:
                raise ControlError("capacity", "Cache result scope exceeds budget")
        return rows, missing

    def _plan_splits(self, rows):
        cuts = {}
        for row in rows:
            if row.kind != "checkpoint":
                cuts.setdefault(row.selection.node.id, set()).update(
                    (row.selection.start, row.selection.end)
                )
        planned = []
        for row in rows:
            if row.kind == "checkpoint":
                planned.append(row)
                continue
            boundaries = (
                [row.selection.start]
                + sorted(
                    c
                    for c in cuts[row.selection.node.id]
                    if row.selection.start < c < row.selection.end
                )
                + [row.selection.end]
            )
            planned.extend(
                replace(
                    row,
                    selection=CopySelection(
                        row.selection.node, row.selection.component, a, b
                    ),
                )
                for a, b in zip(boundaries, boundaries[1:])
            )
        return planned, cuts

    def _split(self, rows, cuts):
        fragments = {}
        for row in rows:
            original = row.selection.node
            if original.id in fragments:
                continue
            current, start, parts = original, row.node_start, []
            for boundary in sorted(
                c
                for c in cuts.get(original.id, ())
                if row.node_start < c < row.node_end
            ):
                parent, action = self.core._split_node(
                    current.key, current, boundary - start
                )
                assert action is None, "control excludes in-flight cache transfers"
                parts.append((parent, start, boundary))
                start = boundary
            parts.append((current, start, row.node_end))
            fragments[original.id] = parts
        result = []
        for row in rows:
            for node, start, end in fragments[row.selection.node.id]:
                selected = row.selection
                if (row.kind == "checkpoint" and end == selected.end) or (
                    row.kind != "checkpoint"
                    and start <= selected.start
                    and selected.end <= end
                ):
                    result.append(
                        replace(
                            row,
                            selection=CopySelection(
                                node, selected.component, selected.start, selected.end
                            ),
                        )
                    )
                    break
        return result

    def _descriptor(self, row):
        selected = row.selection
        return ContentSpan(
            content_id=row.content_id
            or self._content_id(
                row.prefix[: selected.end], selected.component, row.compatibility
            ),
            component=selected.component.name.lower(),
            start_token=selected.start,
            end_token=selected.end,
            coverage_kind=row.kind,
            window_size=row.window,
        )

    def start(self, work: ExecutionRequest) -> ExecutionUpdate:
        try:
            if (
                work.ref.context.engine_id != self.engine_id
                or work.ref.context.engine_epoch != self.engine_epoch
            ):
                raise ControlError("stale_epoch", "Executor owner identity differs")
            if not isinstance(work.payload, SetPriority):
                raise ControlError(
                    "unsupported", "Cache movement has no executable adapter"
                )
            location = work.payload.location
            if (
                location.tier != "gpu"
                or location.backend is not None
                or location.engine_id not in (None, self.engine_id)
            ):
                raise ControlError(
                    "unsupported", "Only the local device cache priority is supported"
                )
            limit = min(self.config.max_result_targets, work.max_result_targets)
            rows, missing = self._resolve(work.target, limit)
            rows, cuts = self._plan_splits(rows)
            if len(rows) + len(missing) > limit:
                raise ControlError("capacity", "Split result scope exceeds budget")
            if not rows:
                unsupported = missing and all(
                    gap.status == Status.UNSUPPORTED for gap in missing
                )
                return ExecutionUpdate(
                    status=Status.UNSUPPORTED if unsupported else Status.MISSING,
                    quiescent=True,
                    code="unaligned" if unsupported else "missing_copy",
                    target_outcomes=tuple(missing),
                )
            self.claims.preflight(
                [row.selection for row in rows],
                work.payload,
                work.ref.context.caller_id,
            )
            # All scope, compatibility, result and claim budgets pass before
            # splitting or changing claims. The synchronous owner cannot race.
            rows = self._split(rows, cuts)
            changed = self.claims.apply(
                [row.selection for row in rows],
                work.payload,
                work.ref.context.caller_id,
            )
            step = (
                "priority_cleared" if work.payload.value is None else "priority_applied"
            )
            outcomes = []
            for row in rows:
                descriptor = self._descriptor(row)
                outcomes.append(
                    TargetOutcome(
                        branch_index=row.branch,
                        coverage=TokenRange(
                            component=descriptor.component,
                            start_token=descriptor.start_token,
                            end_token=descriptor.end_token,
                        ),
                        content_id=descriptor.content_id,
                        coverage_kind=row.kind,
                        window_size=row.window,
                        status=Status.COMPLETED,
                        location=Location(tier="gpu", engine_id=self.engine_id),
                        completed_steps=(step,),
                        code=""
                        if (row.selection.node.id, row.selection.component) in changed
                        else "revision_unchanged",
                    )
                )
            return ExecutionUpdate(
                status=Status.PARTIAL if missing else Status.COMPLETED,
                quiescent=True,
                completed_steps=(step,),
                target_outcomes=tuple(outcomes + missing),
            )
        except ControlError as error:
            status = (
                Status.UNSUPPORTED if error.code == "unsupported" else Status.REJECTED
            )
            return ExecutionUpdate(
                status=status, quiescent=True, code=error.code, reason=str(error)
            )

    def poll(self, ref):
        return None  # All supported priority work completes in start().

    def cancel(self, ref):
        return ExecutionUpdate(status=Status.CANCELLED, quiescent=True)

    def residency(
        self, target: ResolvedTarget, *, limit: int = 256
    ) -> ResidencySnapshot:
        rows, missing = self._resolve(target, limit)
        return ResidencySnapshot(
            namespace=self.config.namespace,
            engine_epoch=self.engine_epoch,
            cache_generation=self.generation,
            observed_at=self.claims.clock(),
            items=tuple(
                ResidencyEntry(
                    branch_index=row.branch,
                    descriptor=self._descriptor(row),
                    coverage_kind=row.kind,
                    window_size=row.window,
                    effective_priority=self.claims.effective(
                        row.selection.node, row.selection.component
                    )
                    or 0,
                    lock_ref=row.selection.node.component_data[
                        row.selection.component
                    ].lock_ref,
                )
                for row in rows
            ),
            missing=tuple(missing),
        )
