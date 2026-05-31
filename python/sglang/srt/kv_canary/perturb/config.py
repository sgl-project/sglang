from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from sglang.srt.environ import envs
from sglang.srt.kv_canary.buffer_group import PoolKind


class TargetGroupKind(IntEnum):
    FULL = PoolKind.FULL.value
    SWA = PoolKind.SWA.value

    def __str__(self) -> str:
        return self.name.lower()


@dataclass(frozen=True, slots=True, kw_only=True)
class PerturbConfig:
    target_group_kind: TargetGroupKind | None
    warmup_steps: int

    @classmethod
    def from_env(cls) -> "PerturbConfig":
        return cls(
            target_group_kind=_parse_target_group_kind_from_env(
                raw=envs.SGLANG_KV_CANARY_PERTURB_TARGET_GROUP.get(),
            ),
            warmup_steps=envs.SGLANG_KV_CANARY_PERTURB_WARMUP_STEPS.get(),
        )


def _parse_target_group_kind_from_env(
    *,
    raw: str | None,
) -> TargetGroupKind | None:
    if raw is not None and raw.strip():
        return _parse_target_group_kind(raw)
    return None


def require_target_group_kind(
    *, target_group_kind: TargetGroupKind | None, perturb_name: str
) -> TargetGroupKind:
    if target_group_kind is None:
        raise ValueError(
            "SGLANG_KV_CANARY_PERTURB_TARGET_GROUP must be explicitly set to "
            f"'full' or 'swa' when {perturb_name} perturbation is enabled"
        )
    return target_group_kind


def _parse_target_group_kind(raw: str | None) -> TargetGroupKind:
    if raw is None or not raw.strip():
        raise ValueError(
            "SGLANG_KV_CANARY_PERTURB_TARGET_GROUP must be explicitly set to "
            "'full' or 'swa'"
        )

    value = raw.strip().lower()
    try:
        return TargetGroupKind[value.upper()]
    except KeyError:
        raise ValueError(
            "SGLANG_KV_CANARY_PERTURB_TARGET_GROUP must be one of 'full' / "
            f"'swa', got {raw!r}"
        ) from None
