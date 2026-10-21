# SPDX-License-Identifier: Apache-2.0
"""Tool coordination proposes movement from bounded observations and estimates."""

from collections import OrderedDict
from typing import Annotated, Literal

import msgspec

from sglang.srt.managers.kv_control.contracts import (
    ControlError,
    CopyTo,
    Location,
    MoveTo,
    SessionTarget,
    TokensTarget,
    Value,
)
from sglang.srt.managers.kv_control.policy import KVPolicy, PolicyProposal


class ToolWaitOptions(Value):
    tier: Literal["host", "storage"] = "host"
    min_pressure: Annotated[float, msgspec.Meta(ge=0, le=1)] = 0.7
    min_pause_seconds: Annotated[float, msgspec.Meta(ge=0, le=86400)] = 0.5
    initial_estimated_transfer_seconds: Annotated[
        float, msgspec.Meta(ge=1e-6, le=60)
    ] = 0.1
    lead_seconds: Annotated[float, msgspec.Meta(ge=0, le=300)] = 0.1
    prefetch_on_admission: bool = True
    max_contexts: Annotated[int, msgspec.Meta(ge=1, le=4096)] = 128


class ToolWaitPolicy(KVPolicy):
    consumed_signals = frozenset({"tool_wait", "resume", "resume_prediction"})
    observation_categories = frozenset({"capacity"})
    movement_policy = True

    def __init__(self, options):
        self.options = options
        self.uses_guidance = options.prefetch_on_admission
        self._kept = OrderedDict()
        self.estimated_save_seconds = options.initial_estimated_transfer_seconds
        self.estimated_load_seconds = options.initial_estimated_transfer_seconds

    def consumes(self, payload):
        return super().consumes(payload) and isinstance(payload.target, SessionTarget)

    @staticmethod
    def target(scope):
        if len(scope.branches) != 1:
            raise ControlError(
                "unsupported", "Tool coordination requires one token-backed prefix"
            )
        branch = scope.branches[0]
        if not isinstance(branch.content, TokensTarget) or branch.requested.start_token:
            raise ControlError(
                "unsupported", "Tool coordination requires a complete token prefix"
            )
        return msgspec.structs.replace(
            branch.content,
            token_ids=branch.content.token_ids[: branch.requested.end_token],
            start_token=0,
            end_token=None,
            allow_clip=False,
        )

    def on_signal(self, signal, view):
        payload = signal.payload
        if payload.signal == "tool_wait":
            gpu = next(
                (p for p in view.pools if p.tier == "gpu" and p.component == "full"),
                None,
            )
            cost = max(
                self.options.min_pause_seconds,
                self.estimated_save_seconds + self.estimated_load_seconds,
            )
            worthwhile = (
                gpu is not None
                and gpu.total_tokens > 0
                and gpu.referenced_tokens
                >= self.options.min_pressure * gpu.total_tokens
                and payload.resume_after_seconds is not None
                and payload.resume_after_seconds > cost
            )
            if not worthwhile:
                self._kept[signal.owner_key] = signal.tasks
                self._kept.move_to_end(signal.owner_key)
                while len(self._kept) > self.options.max_contexts:
                    self._kept.popitem(last=False)
                return ()
            self._kept.pop(signal.owner_key, None)
            self.target(signal.scope)  # Validate this executable profile.
            return (
                PolicyProposal(
                    "save",
                    MoveTo(
                        target=payload.target,
                        src=Location(tier="gpu"),
                        dst=Location(tier=self.options.tier),
                    ),
                    signal.scope,
                ),
            )
        if signal.owner_key in self._kept or signal.previous_scope is None:
            return ()
        scope = signal.previous_scope
        target = self.target(scope)
        not_before = view.now
        if payload.signal == "resume_prediction":
            if payload.resume_after_seconds is None:
                return ()
            not_before += max(
                0.0,
                payload.resume_after_seconds
                - max(self.options.lead_seconds, self.estimated_load_seconds),
            )
        return (
            PolicyProposal(
                "restore",
                CopyTo(
                    target=target,
                    src=Location(tier=self.options.tier),
                    dst=Location(tier="gpu"),
                ),
                scope,
                not_before=not_before,
                prepare_for_admission=self.options.prefetch_on_admission,
            ),
        )

    def on_result(self, proposal, result, *, elapsed_seconds):
        if (
            elapsed_seconds <= 0
            or "destination_ready" not in result.update.completed_steps
        ):
            return
        attribute = (
            "estimated_save_seconds"
            if isinstance(proposal.action, MoveTo)
            else "estimated_load_seconds"
            if isinstance(proposal.action, CopyTo)
            else None
        )
        if attribute is not None:
            old = getattr(self, attribute)
            setattr(
                self,
                attribute,
                max(
                    self.options.initial_estimated_transfer_seconds,
                    elapsed_seconds,
                    0.75 * old + 0.25 * elapsed_seconds,
                ),
            )

    def cleanup(self, tasks):
        removed = frozenset(tasks)
        for key, owners in tuple(self._kept.items()):
            if removed.intersection(owners):
                self._kept.pop(key)
