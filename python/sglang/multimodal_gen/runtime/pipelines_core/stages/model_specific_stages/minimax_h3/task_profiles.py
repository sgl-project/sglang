# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 task profiles.

Data table driving request validation and plan resolution for the three v1
tasks (t2va / fl2va / ref2va). One row per task; stages and the request
projector consume rows instead of branching on task names.

Design summary: keyframes bind target geometry; references remain independent.
"""

from __future__ import annotations

from types import MappingProxyType

import msgspec

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_DEFAULT_BRANCHES,
)

MINIMAX_H3_CONDITION_ROLE_KEYFRAME = "keyframe"
MINIMAX_H3_CONDITION_ROLE_REFERENCE = "reference"

MINIMAX_H3_TASK_T2VA = "t2va"
MINIMAX_H3_TASK_FL2VA = "fl2va"
MINIMAX_H3_TASK_REF2VA = "ref2va"

# Public dispatcher contract.  SGLang only needs the stable task mapping.
MINIMAX_H3_TASK_PARTITIONS = MappingProxyType(
    {
        "t2va": "fl2va",
        "fl2va": "fl2va",
        "ref2va": "ref2va",
    }
)


def canonical_minimax_h3_task(task: str) -> str:
    """Normalize a public task name (no aliases are currently defined)."""

    return task


def partition_for_task(task: str) -> str:
    """Return the deployment partition serving *task*."""

    if not isinstance(task, str) or not task.strip():
        raise ValueError("MiniMax H3 task must be a non-empty string")
    normalized = task.strip().lower()
    try:
        return MINIMAX_H3_TASK_PARTITIONS[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported MiniMax H3 task {task!r}") from exc


MINIMAX_H3_FINITE_ASPECT_RATIOS = (
    "21:9",
    "16:9",
    "4:3",
    "1:1",
    "3:4",
    "9:16",
)

# ``fl2va`` remains the single public task name for all keyframe variants:
# first-frame-only, last-frame-only, and first+last.  Keep the accepted
# semantic signatures centralized so validation and every downstream sink can
# reject middle/reversed/stale payloads consistently.
MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES: tuple[tuple[int, ...], ...] = (
    (0,),
    (-1,),
    (0, -1),
)

# Canonical material-chain names are part of the direct runtime plan.
# Profile validation keeps only the wire-level allowlist here.
MINIMAX_H3_CANONICAL_MATERIAL_CHAINS = frozenset(
    {
        "image.target_canvas",
        "image.reference_preserve",
        "video.reference_preserve",
        "video_audio.reference_preserve",
        "audio",
    }
)


class MiniMaxH3ConditionRule(msgspec.Struct, frozen=True):
    """Per-(role, type) admission + routing rule."""

    role: str
    condition_type: str
    material_chain: str
    requires_frame_index: bool = False
    visual_tokenizer_encode: bool = False
    audio_tokenizer_encode: bool = False


class MiniMaxH3TaskProfile(msgspec.Struct, frozen=True):
    """One row of the task table."""

    task: str
    conditions_required: bool
    condition_rules: tuple[MiniMaxH3ConditionRule, ...]
    branches: tuple[dict, ...]
    default_flow_shift: float
    default_audio_flow_shift: float
    required_components: tuple[str, ...]
    aspect_ratio_forced_auto: bool
    geometry_source: str  # "explicit_target" | "first_keyframe" | "model_default"
    # For ref2va video, target may omit duration_seconds
    # when an audio reference is present — duration derives from the
    # reference audio probe and is then checked against the shared 4-15s range.
    duration_from_audio_reference: bool = False
    # Some deployments keep video references disabled while retaining the
    # routing rule for projection/contract compatibility.
    video_reference_supported: bool = True
    min_condition_count: int | None = None
    max_condition_count: int | None = None
    # Resolution policy for target.aspect_ratio="auto".  A concrete ratio
    # resolves immediately; None keeps geometry deferred to the named source.
    auto_aspect_ratio: str | None = None
    auto_geometry_source: str | None = None

    def rule_for(self, *, role: str, condition_type: str) -> MiniMaxH3ConditionRule:
        for rule in self.condition_rules:
            if rule.role == role and rule.condition_type == condition_type:
                return rule
        raise ValueError(
            f"task {self.task!r} does not allow condition "
            f"role={role!r} type={condition_type!r}"
        )


_BASE_COMPONENTS = (
    "processor",
    "text_encoder",
    "tokenizer",
    "transformer",
    "video_vae",
    "audio_vae",
)

_BRANCHES = tuple(dict(branch) for branch in MINIMAX_H3_DEFAULT_BRANCHES)
MINIMAX_H3_TASK_PROFILES: dict[str, MiniMaxH3TaskProfile] = {
    MINIMAX_H3_TASK_T2VA: MiniMaxH3TaskProfile(
        task=MINIMAX_H3_TASK_T2VA,
        conditions_required=False,
        condition_rules=(),
        branches=_BRANCHES,
        default_flow_shift=12.0,
        default_audio_flow_shift=3.0,
        required_components=_BASE_COMPONENTS,
        aspect_ratio_forced_auto=False,
        geometry_source="explicit_target",
        auto_aspect_ratio="16:9",
        auto_geometry_source="policy_default",
    ),
    MINIMAX_H3_TASK_FL2VA: MiniMaxH3TaskProfile(
        task=MINIMAX_H3_TASK_FL2VA,
        conditions_required=True,
        condition_rules=(
            MiniMaxH3ConditionRule(
                role=MINIMAX_H3_CONDITION_ROLE_KEYFRAME,
                condition_type="image",
                material_chain="image.target_canvas",
                requires_frame_index=True,
                visual_tokenizer_encode=True,
            ),
        ),
        branches=_BRANCHES,
        default_flow_shift=12.0,
        default_audio_flow_shift=3.0,
        required_components=_BASE_COMPONENTS,
        # FL follows the resolved target geometry projected by the caller.
        # Its first/last keyframes are prepared against that target canvas.
        aspect_ratio_forced_auto=False,
        geometry_source="explicit_target",
        min_condition_count=1,
        max_condition_count=2,
        auto_geometry_source="first_keyframe",
    ),
    MINIMAX_H3_TASK_REF2VA: MiniMaxH3TaskProfile(
        task=MINIMAX_H3_TASK_REF2VA,
        conditions_required=True,
        condition_rules=(
            MiniMaxH3ConditionRule(
                role=MINIMAX_H3_CONDITION_ROLE_REFERENCE,
                condition_type="image",
                material_chain="image.reference_preserve",
                visual_tokenizer_encode=True,
            ),
            MiniMaxH3ConditionRule(
                role=MINIMAX_H3_CONDITION_ROLE_REFERENCE,
                condition_type="video",
                material_chain="video.reference_preserve",
                visual_tokenizer_encode=True,
                # ref2va video: the reference video's ORIGINAL soundtrack is the
                # audio reference; -17 rows cover the full pre-truncation track.
                # Route the same material into the audio encoder.
                audio_tokenizer_encode=True,
            ),
            MiniMaxH3ConditionRule(
                role=MINIMAX_H3_CONDITION_ROLE_REFERENCE,
                condition_type="video_audio",
                material_chain="video_audio.reference_preserve",
                visual_tokenizer_encode=True,
                audio_tokenizer_encode=True,
            ),
            MiniMaxH3ConditionRule(
                role=MINIMAX_H3_CONDITION_ROLE_REFERENCE,
                condition_type="audio",
                material_chain="audio",
                audio_tokenizer_encode=True,
            ),
        ),
        branches=_BRANCHES,
        default_flow_shift=12.0,
        default_audio_flow_shift=3.0,
        required_components=_BASE_COMPONENTS,
        # ref2va video may use an explicit aspect ratio such as "7:4":
        # ref2va allows explicit aspect ratios; "auto" falls back to the
        # policy default (references never bind target geometry).
        aspect_ratio_forced_auto=False,
        geometry_source="explicit_target",
        duration_from_audio_reference=True,
        auto_aspect_ratio="16:9",
        auto_geometry_source="policy_default",
        # ref2va video uses a subject image plus a reference video with soundtrack.
        video_reference_supported=True,
    ),
}


def minimax_h3_task_profile(task: str) -> MiniMaxH3TaskProfile:
    profile = MINIMAX_H3_TASK_PROFILES.get(task)
    if profile is None:
        raise ValueError(
            f"unknown minimax_h3 task {task!r}; supported: "
            f"{sorted(MINIMAX_H3_TASK_PROFILES)}"
        )
    return profile


def _validate_profiles() -> None:
    for task, profile in MINIMAX_H3_TASK_PROFILES.items():
        if profile.task != task:
            raise ValueError(f"profile key/task mismatch: {task} vs {profile.task}")
        roles = {rule.role for rule in profile.condition_rules}
        if len(roles) > 1:
            raise ValueError(
                f"task {task}: condition roles must not mix, got {sorted(roles)}"
            )
        if profile.min_condition_count is not None and profile.min_condition_count <= 0:
            raise ValueError(f"task {task}: min_condition_count must be positive")
        if profile.max_condition_count is not None and profile.max_condition_count <= 0:
            raise ValueError(f"task {task}: max_condition_count must be positive")
        if (
            profile.min_condition_count is not None
            and profile.max_condition_count is not None
            and profile.min_condition_count > profile.max_condition_count
        ):
            raise ValueError(
                f"task {task}: min_condition_count exceeds max_condition_count"
            )
        for rule in profile.condition_rules:
            if rule.material_chain not in MINIMAX_H3_CANONICAL_MATERIAL_CHAINS:
                raise ValueError(
                    f"task {task}: unknown material chain {rule.material_chain!r}"
                )


_validate_profiles()

__all__ = [
    "MINIMAX_H3_CONDITION_ROLE_KEYFRAME",
    "MINIMAX_H3_FINITE_ASPECT_RATIOS",
    "MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES",
    "MINIMAX_H3_CONDITION_ROLE_REFERENCE",
    "MINIMAX_H3_CANONICAL_MATERIAL_CHAINS",
    "MINIMAX_H3_TASK_FL2VA",
    "MINIMAX_H3_TASK_PROFILES",
    "MINIMAX_H3_TASK_REF2VA",
    "MINIMAX_H3_TASK_T2VA",
    "MiniMaxH3TaskProfile",
    "canonical_minimax_h3_task",
    "minimax_h3_task_profile",
    "partition_for_task",
]
