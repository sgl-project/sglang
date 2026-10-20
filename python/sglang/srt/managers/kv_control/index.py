# SPDX-License-Identifier: Apache-2.0
"""Bounded logical relationships. No method asserts physical residency."""

from __future__ import annotations

from dataclasses import dataclass

import msgspec

from sglang.srt.managers.kv_control.contracts import (
    Associate,
    ContentSpan,
    ContentTarget,
    ControlError,
    Identifier,
    Limits,
    LogicalChange,
    LogicalTarget,
    ParentPrefix,
    Position,
    SessionTarget,
    TaskIdentity,
    TokenRange,
    TokensTarget,
    Value,
    validate_target,
    validated,
)


class Continuation(Value):
    task: TaskIdentity
    continuation_id: str
    revision: int
    incarnation: int
    content: ContentTarget | TokensTarget
    parent_prefix: ParentPrefix | None = None


class RangeGap(Value):
    coverage: TokenRange
    reason: str


class ResolvedBranch(Value):
    continuation: Continuation | None
    content: ContentTarget | TokensTarget
    requested: TokenRange
    spans: tuple[ContentSpan, ...]
    unaligned: tuple[ContentSpan, ...]
    missing: tuple[RangeGap, ...]
    dependencies: tuple[TokenRange, ...]


class ResolvedTarget(Value):
    engine_epoch: str
    namespace: str
    index_generation: int
    allow_clip: bool
    branches: tuple[ResolvedBranch, ...]


class IndexCursor(Value):
    engine_epoch: Identifier
    namespace: Identifier
    generation: Position
    session_id: Identifier | None
    after: tuple[Identifier, Identifier]


class AssociationPage(Value):
    engine_epoch: str
    namespace: str
    generation: int
    items: tuple[Continuation, ...]
    session_revisions: tuple[tuple[str, int], ...]
    next_cursor: IndexCursor | None


@dataclass
class _Session:
    identity: TaskIdentity
    revision: int
    parent_id: str | None
    continuations: set[str]
    children: set[str]


def _length(content):
    return (
        len(content.token_ids)
        if isinstance(content, TokensTarget)
        else content.token_count
    )


def _sizes(content):
    return (
        (0, len(content.token_ids))
        if isinstance(content, TokensTarget)
        else (len(content.spans), 0)
    )


class LogicalIndex:
    """One owner-thread index with immutable, independently usable snapshots.

    Revision/incarnation numbers never repeat within this epoch. Failed updates
    change neither generations nor reverse memberships. Capacity is refused;
    history is only forgotten through explicit prune/finish operations.
    """

    def __init__(self, namespace: str, engine_epoch: str, *, limits: Limits = Limits()):
        self.namespace = validated(namespace, Identifier)
        self.engine_epoch = validated(engine_epoch, Identifier)
        self.limits = limits
        self.generation = 0
        self._sessions: dict[str, _Session] = {}
        self._continuations: dict[tuple[str, str], Continuation] = {}
        self._members: dict[ContentSpan, set[tuple[str, str]]] = {}
        self._spans = 0
        self._tokens = 0

    def get(self, session_id: str, continuation_id: str) -> Continuation:
        validated(session_id, Identifier)
        validated(continuation_id, Identifier)
        record = self._continuations.get((session_id, continuation_id))
        if record is None:
            raise ControlError("missing", "Continuation is not tracked")
        return record

    def session_revision(self, session_id: str) -> int:
        validated(session_id, Identifier)
        session = self._sessions.get(session_id)
        if session is None:
            raise ControlError("missing", "Session is not tracked")
        return session.revision

    @staticmethod
    def _expect(expected, actual):
        if type(expected) is not int or expected != actual:
            raise ControlError("stale_revision", f"Expected current revision {actual}")

    def _check_namespace(self, namespace):
        if namespace != self.namespace:
            raise ControlError("incompatible_namespace", "Target namespace differs")

    def associate(self, payload: Associate) -> Continuation:
        payload = validated(payload, Associate)
        content = payload.target
        validate_target(content)
        self._check_namespace(content.namespace)
        if (
            content.start_token != 0
            or content.end_token is not None
            or content.allow_clip
        ):
            raise ControlError(
                "invalid_payload", "Association requires a complete logical prefix"
            )
        if isinstance(content, ContentTarget):
            content = msgspec.structs.replace(
                content,
                spans=tuple(
                    sorted(
                        content.spans,
                        key=lambda s: (s.component, s.start_token, s.end_token),
                    )
                ),
            )
        key = payload.session_id, payload.continuation_id
        previous = self._continuations.get(key)
        session = self._sessions.get(payload.session_id)
        self._expect(payload.expected_revision, previous.revision if previous else 0)
        parent_id = session.parent_id if session else None
        if payload.task_parent is not None:
            parent_id = payload.task_parent.session_id
            if parent_id == payload.session_id:
                raise ControlError("conflict", "Task cannot parent itself")
            self._expect(
                payload.task_parent.expected_revision, self.session_revision(parent_id)
            )
            if session is not None and session.parent_id != parent_id:
                raise ControlError("conflict", "Task reparenting is unsupported")
        parent_prefix = payload.parent_prefix
        if previous is not None:
            if parent_prefix is not None and parent_prefix != previous.parent_prefix:
                raise ControlError("conflict", "Continuation parent cannot change")
            parent_prefix = previous.parent_prefix
            self._check_extension(previous.content, content)
        elif parent_prefix is not None:
            if (parent_prefix.session_id, parent_prefix.continuation_id) == key:
                raise ControlError("conflict", "Continuation cannot parent itself")
            parent = self.get(parent_prefix.session_id, parent_prefix.continuation_id)
            self._expect(parent_prefix.expected_revision, parent.revision)
            self._check_parent(parent.content, content, parent_prefix.end_token)
        old_spans, old_tokens = _sizes(previous.content) if previous else (0, 0)
        spans, tokens = _sizes(content)
        if (
            spans > self.limits.max_scope_spans
            or tokens > self.limits.max_scope_tokens
            or self._spans - old_spans + spans > self.limits.max_index_spans
            or self._tokens - old_tokens + tokens > self.limits.max_index_tokens
            or (session is None and len(self._sessions) >= self.limits.max_sessions)
            or (
                previous is None
                and len(self._continuations) >= self.limits.max_continuations
            )
        ):
            raise ControlError("capacity", "Logical index capacity exhausted")
        if previous is not None and previous.content == content:
            return previous
        # All validation is complete. Publish the replacement and counters together.
        self.generation += 1
        if session is None:
            session = _Session(
                TaskIdentity(
                    session_id=payload.session_id, incarnation=self.generation
                ),
                self.generation,
                parent_id,
                set(),
                set(),
            )
            self._sessions[payload.session_id] = session
            if parent_id is not None:
                self._sessions[parent_id].children.add(payload.session_id)
        record = Continuation(
            task=session.identity,
            continuation_id=payload.continuation_id,
            revision=self.generation,
            incarnation=previous.incarnation if previous else self.generation,
            content=content,
            parent_prefix=parent_prefix,
        )
        if previous:
            self._remove_members(key, previous.content)
        self._continuations[key] = record
        session.continuations.add(payload.continuation_id)
        if isinstance(content, ContentTarget):
            for span in content.spans:
                self._members.setdefault(span, set()).add(key)
        self._spans += spans - old_spans
        self._tokens += tokens - old_tokens
        self._touch(payload.session_id)
        return record

    @staticmethod
    def _check_extension(old, new):
        if type(old) is not type(new) or _length(new) < _length(old):
            raise ControlError("conflict", "Continuation identity is append-only")
        if isinstance(old, TokensTarget):
            if new.token_ids[: len(old.token_ids)] != old.token_ids:
                raise ControlError(
                    "conflict", "Continuation tokens cannot be rewritten"
                )
        elif old.components != new.components or not set(old.spans).issubset(new.spans):
            raise ControlError(
                "conflict", "Continuation descriptors cannot be rewritten"
            )

    def _check_parent(self, parent, child, end):
        if not 0 < end <= min(_length(parent), _length(child)):
            raise ControlError("missing", "Parent prefix boundary is not tracked")
        if type(parent) is not type(child):
            raise ControlError(
                "unsupported", "Cannot prove equality across descriptor formats"
            )
        if isinstance(parent, TokensTarget):
            if parent.token_ids[:end] != child.token_ids[:end]:
                raise ControlError(
                    "conflict", "Branch tokens differ from parent prefix"
                )
            return
        resolution = self._resolve_branch(parent, None, 0, end)
        if resolution.unaligned:
            raise ControlError("unaligned", "Parent boundary splits a descriptor")
        if resolution.missing:
            raise ControlError("missing", "Parent prefix has missing component history")
        if parent.components != child.components or not set(resolution.spans).issubset(
            child.spans
        ):
            raise ControlError("conflict", "Branch content differs from parent prefix")

    def _touch(self, session_id):
        while session_id is not None:
            session = self._sessions[session_id]
            session.revision = self.generation
            session_id = session.parent_id

    def members(self, span: ContentSpan) -> tuple[tuple[str, str], ...]:
        """Reverse lookup of an exact descriptor in this index's namespace."""
        span = validated(span, ContentSpan)
        return tuple(sorted(self._members.get(span, ())))

    def token_members(self, prefix: TokensTarget) -> tuple[tuple[str, str], ...]:
        """Explicit bounded query by token equality; stores no per-token index."""
        prefix = validated(prefix, TokensTarget)
        self._check_namespace(prefix.namespace)
        if prefix.start_token or prefix.end_token is not None or prefix.allow_clip:
            raise ControlError(
                "invalid_payload", "Reverse token lookup needs a full prefix"
            )
        return tuple(
            sorted(
                key
                for key, record in self._continuations.items()
                if isinstance(record.content, TokensTarget)
                and record.content.token_ids[: len(prefix.token_ids)]
                == prefix.token_ids
            )
        )

    def resolve(self, target: LogicalTarget) -> ResolvedTarget:
        target = validated(target, LogicalTarget)
        validate_target(target)
        self._check_namespace(target.namespace)
        if isinstance(target, SessionTarget):
            revision = self.session_revision(target.session_id)
            if target.continuation_id is None:
                records = tuple(
                    self.get(target.session_id, name)
                    for name in sorted(self._sessions[target.session_id].continuations)
                )
            else:
                records = (self.get(target.session_id, target.continuation_id),)
                revision = records[0].revision
            if target.expected_revision is not None:
                self._expect(target.expected_revision, revision)
            if not records:
                raise ControlError("missing", "Session has no tracked continuations")
            sources = tuple((record.content, record) for record in records)
        else:
            sources = ((target, None),)
        if (
            len(sources) > self.limits.max_branches
            or sum(_sizes(c)[0] for c, _ in sources) > self.limits.max_scope_spans
            or sum(_sizes(c)[1] for c, _ in sources) > self.limits.max_scope_tokens
        ):
            raise ControlError("capacity", "Resolved snapshot exceeds scope limit")
        branches = []
        for content, record in sources:
            end = target.end_token if target.end_token is not None else _length(content)
            if end <= target.start_token:
                raise ControlError("missing", "Range starts beyond tracked content")
            branches.append(
                self._resolve_branch(content, record, target.start_token, end)
            )
        return ResolvedTarget(
            engine_epoch=self.engine_epoch,
            namespace=self.namespace,
            index_generation=self.generation,
            allow_clip=target.allow_clip,
            branches=tuple(branches),
        )

    @staticmethod
    def _resolve_branch(content, record, start, end):
        missing, full, unaligned, dependencies = [], [], [], set()
        if isinstance(content, TokensTarget):
            if end > len(content.token_ids):
                missing.append(
                    RangeGap(
                        coverage=TokenRange(
                            start_token=max(start, len(content.token_ids)),
                            end_token=end,
                        ),
                        reason="untracked_tokens",
                    )
                )
        else:
            for component in content.components:
                cursor = start
                for span in sorted(
                    (s for s in content.spans if s.component == component),
                    key=lambda s: s.start_token,
                ):
                    left, right = max(start, span.start_token), min(end, span.end_token)
                    if left >= right:
                        continue
                    if left > cursor:
                        missing.append(
                            RangeGap(
                                coverage=TokenRange(
                                    component=component,
                                    start_token=cursor,
                                    end_token=left,
                                ),
                                reason="untracked_content",
                            )
                        )
                    cursor = max(cursor, right)
                    if span.start_token < start or span.end_token > end:
                        unaligned.append(span)
                    else:
                        full.append(span)
                        if span.dependency_start < span.start_token:
                            dependencies.add(
                                TokenRange(
                                    component=component,
                                    start_token=span.dependency_start,
                                    end_token=span.start_token,
                                )
                            )
                if cursor < end:
                    missing.append(
                        RangeGap(
                            coverage=TokenRange(
                                component=component, start_token=cursor, end_token=end
                            ),
                            reason="untracked_content",
                        )
                    )
        return ResolvedBranch(
            continuation=record,
            content=content,
            requested=TokenRange(start_token=start, end_token=end),
            spans=tuple(full),
            unaligned=tuple(unaligned),
            missing=tuple(missing),
            dependencies=tuple(
                sorted(
                    dependencies,
                    key=lambda r: (r.component, r.start_token, r.end_token),
                )
            ),
        )

    def query(
        self,
        *,
        session_id: str | None = None,
        cursor: IndexCursor | None = None,
        limit: int = 256,
    ) -> AssociationPage:
        if type(limit) is not int or not 1 <= limit <= self.limits.max_page_size:
            raise ControlError("invalid_cursor", "Page size exceeds limit")
        if session_id is not None:
            self.session_revision(session_id)
        if cursor is not None:
            cursor = validated(cursor, IndexCursor)
            if (
                cursor.engine_epoch != self.engine_epoch
                or cursor.namespace != self.namespace
            ):
                raise ControlError(
                    "stale_epoch", "Cursor belongs to another index generation"
                )
            if cursor.generation != self.generation:
                raise ControlError(
                    "stale_revision", "Index changed; restart pagination"
                )
            if (
                cursor.session_id != session_id
                or cursor.after not in self._continuations
            ):
                raise ControlError(
                    "invalid_cursor", "Cursor does not belong to this query"
                )
        keys = sorted(
            key
            for key in self._continuations
            if (session_id is None or key[0] == session_id)
            and (cursor is None or key > cursor.after)
        )
        selected = keys[:limit]
        next_cursor = (
            IndexCursor(
                engine_epoch=self.engine_epoch,
                namespace=self.namespace,
                generation=self.generation,
                session_id=session_id,
                after=selected[-1],
            )
            if len(keys) > limit
            else None
        )
        return AssociationPage(
            engine_epoch=self.engine_epoch,
            namespace=self.namespace,
            generation=self.generation,
            items=tuple(self._continuations[key] for key in selected),
            session_revisions=tuple(
                (name, self.session_revision(name))
                for name in sorted({key[0] for key in selected})
            ),
            next_cursor=next_cursor,
        )

    def finish_task(self, session_id: str, *, expected_revision: int) -> LogicalChange:
        self._expect(expected_revision, self.session_revision(session_id))
        todo, removed = [session_id], []
        while todo:
            name = todo.pop()
            removed.append(name)
            todo.extend(sorted(self._sessions[name].children, reverse=True))
        identities = tuple(self._sessions[name].identity for name in removed)
        parent_id = self._sessions[session_id].parent_id
        self.generation += 1
        if parent_id is not None:
            self._sessions[parent_id].children.remove(session_id)
            self._touch(parent_id)
        for name in removed:
            session = self._sessions.pop(name)
            for continuation_id in session.continuations:
                self._remove((name, continuation_id))
        return LogicalChange(index_generation=self.generation, removed_tasks=identities)

    def prune(
        self, session_id: str, continuation_id: str, *, expected_revision: int
    ) -> None:
        record = self.get(session_id, continuation_id)
        self._expect(expected_revision, record.revision)
        self.generation += 1
        self._remove((session_id, continuation_id))
        self._sessions[session_id].continuations.remove(continuation_id)
        self._touch(session_id)

    def _remove_members(self, key, content):
        if isinstance(content, ContentTarget):
            for span in content.spans:
                members = self._members[span]
                members.remove(key)
                if not members:
                    del self._members[span]

    def _remove(self, key):
        record = self._continuations.pop(key)
        self._remove_members(key, record.content)
        spans, tokens = _sizes(record.content)
        self._spans -= spans
        self._tokens -= tokens

    def capacity(self) -> dict[str, int]:
        return {
            "sessions": len(self._sessions),
            "continuations": len(self._continuations),
            "spans": self._spans,
            "tokens": self._tokens,
        }
