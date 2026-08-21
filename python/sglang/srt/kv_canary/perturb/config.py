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
    req_to_token_prob: float
    real_kv_used_prob: float
    real_kv_unused_cache_prob: float
    real_kv_post_forward_prob: float
    target_group_kind: TargetGroupKind | None
    warmup_steps: int

    @classmethod
    def from_env(cls) -> PerturbConfig:
        real_kv_used_prob = envs.SGLANG_KV_CANARY_PERTURB_REAL_KV_USED_PROB.get()
        real_kv_unused_cache_prob = (
            envs.SGLANG_KV_CANARY_PERTURB_REAL_KV_UNUSED_CACHE_PROB.get()
        )
        real_kv_post_forward_prob = (
            envs.SGLANG_KV_CANARY_PERTURB_REAL_KV_POST_FORWARD_PROB.get()
        )
        return cls(
            req_to_token_prob=envs.SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_PROB.get(),
            real_kv_used_prob=real_kv_used_prob,
            real_kv_unused_cache_prob=real_kv_unused_cache_prob,
            real_kv_post_forward_prob=real_kv_post_forward_prob,
            target_group_kind=_parse_target_group_kind_from_env(
                raw=envs.SGLANG_KV_CANARY_PERTURB_TARGET_GROUP.get(),
                real_kv_used_prob=real_kv_used_prob,
                real_kv_unused_cache_prob=real_kv_unused_cache_prob,
                real_kv_post_forward_prob=real_kv_post_forward_prob,
            ),
            warmup_steps=envs.SGLANG_KV_CANARY_PERTURB_WARMUP_STEPS.get(),
        )


def _parse_target_group_kind_from_env(
    *,
    raw: str | None,
    real_kv_used_prob: float,
    real_kv_unused_cache_prob: float,
    real_kv_post_forward_prob: float,
) -> TargetGroupKind | None:
    if raw is not None and raw.strip():
        return _parse_target_group_kind(raw)
    if (
        real_kv_used_prob > 0.0
        or real_kv_unused_cache_prob > 0.0
        or real_kv_post_forward_prob > 0.0
    ):
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
