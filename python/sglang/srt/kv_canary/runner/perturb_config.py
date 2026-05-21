from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sglang.srt.environ import envs


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
        target_group_kind: which CanaryBufferGroup to target with real_kv_used / real_kv_unused
            perturb. "full" / "swa" exact-match the PoolKind name; "any" picks at random among
            groups with non-empty real_kv_sources.
        real_kv_prob: DEPRECATED — retained for one commit to keep the old hook compatible until
            the follow-up commit splits perturb_real_kv into used / unused_cache.
        real_kv_require_orphan: DEPRECATED — see real_kv_prob.
        warmup_steps: number of initial forward steps to gate off all perturb hooks. Prevents
            perturb from firing during sglang warmup, where a garbage write can trip a CUDA error
            before the canary's deferred D2H violation pump has a chance to log the canary_kind
            line.
    """

    req_to_token_prob: float = 0.0
    real_kv_used_prob: float = 0.0
    real_kv_unused_cache_prob: float = 0.0
    target_group_kind: Literal["full", "swa", "any"] = "any"
    real_kv_prob: float = 0.0
    real_kv_require_orphan: bool = False
    warmup_steps: int = 50

    @classmethod
    def from_env(cls) -> "PerturbConfig":
        return cls(
            req_to_token_prob=envs.SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_PROB.get(),
            real_kv_used_prob=envs.SGLANG_KV_CANARY_PERTURB_REAL_KV_USED_PROB.get(),
            real_kv_unused_cache_prob=envs.SGLANG_KV_CANARY_PERTURB_REAL_KV_UNUSED_CACHE_PROB.get(),
            target_group_kind=_parse_target_group_kind(
                envs.SGLANG_KV_CANARY_PERTURB_TARGET_GROUP.get()
            ),
            real_kv_prob=envs.SGLANG_KV_CANARY_REAL_PERTURB_BYTES_PROB.get(),
            real_kv_require_orphan=envs.SGLANG_KV_CANARY_REAL_PERTURB_BYTES_REQUIRE_ORPHAN.get(),
            warmup_steps=envs.SGLANG_KV_CANARY_PERTURB_WARMUP_STEPS.get(),
        )


def _parse_target_group_kind(raw: str) -> Literal["full", "swa", "any"]:
    value = (raw or "any").strip().lower()
    if value not in ("full", "swa", "any"):
        raise ValueError(
            f"SGLANG_KV_CANARY_PERTURB_TARGET_GROUP must be one of 'full' / 'swa' / 'any', "
            f"got {raw!r}"
        )
    return value  # type: ignore[return-value]
