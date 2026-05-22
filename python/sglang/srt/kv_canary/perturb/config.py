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
    """Fault-injection knobs for the canary self-test path. Separate from CanaryConfig because
    production never touches these — only the canary's own e2e tests flip them on via env vars.

    Fields:
        req_to_token_prob: per-forward probability of trampling a random req_to_token entry to
            drive a violation. 0 = disabled.
        real_kv_used_prob: per-forward probability of flipping byte_offset=0 of an active req's
            currently-used KV slot. Detection routes through the per-forward HEAD/TAIL verify
            kernel (real_kv_hash violation). 0 = disabled.
        real_kv_unused_cache_prob: per-forward probability of flipping byte_offset=0 of a radix-
            cached but currently-unused (orphan) KV slot. Detection routes through sweep verify
            (per-forward never looks at this slot). 0 = disabled.
        real_kv_post_forward_prob: per-forward probability of flipping byte_offset=0 of a slot
            picked from forward_batch.out_cache_loc AFTER the TAIL kernel has captured its canary
            hash. Designed for PD disagg self-test: P-side runs the flip just before
            ``send_kv_chunk`` so D's first decode forward HEAD/TAIL kernel catches the
            real_kv_hash violation. 0 = disabled.
        target_group_kind: which CanaryBufferGroup to target with real_kv_used / real_kv_unused
            perturb. FULL / SWA exact-match the PoolKind name.
        warmup_steps: number of initial forward steps to gate off all perturb hooks. Prevents
            perturb from firing during sglang warmup, where a garbage write can trip a CUDA error
            before the canary's deferred D2H violation pump has a chance to log the canary_kind
            line.
    """

    req_to_token_prob: float
    real_kv_used_prob: float
    real_kv_unused_cache_prob: float
    real_kv_post_forward_prob: float
    target_group_kind: TargetGroupKind
    warmup_steps: int

    @classmethod
    def from_env(cls) -> "PerturbConfig":
        return cls(
            req_to_token_prob=envs.SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_PROB.get(),
            real_kv_used_prob=envs.SGLANG_KV_CANARY_PERTURB_REAL_KV_USED_PROB.get(),
            real_kv_unused_cache_prob=envs.SGLANG_KV_CANARY_PERTURB_REAL_KV_UNUSED_CACHE_PROB.get(),
            real_kv_post_forward_prob=envs.SGLANG_KV_CANARY_PERTURB_REAL_KV_POST_FORWARD_PROB.get(),
            target_group_kind=_parse_target_group_kind(
                envs.SGLANG_KV_CANARY_PERTURB_TARGET_GROUP.get()
            ),
            warmup_steps=envs.SGLANG_KV_CANARY_PERTURB_WARMUP_STEPS.get(),
        )


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
