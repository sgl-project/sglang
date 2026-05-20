from __future__ import annotations

from dataclasses import dataclass

from sglang.srt.environ import envs


@dataclass(frozen=True, slots=True, kw_only=True)
class PerturbConfig:
    """Fault-injection knobs for the canary self-test path. Separate from CanaryConfig because
    production never touches these — only the canary's own e2e tests flip them on via env vars.

    Fields:
        req_to_token_prob: per-forward probability of trampling a random req_to_token entry to
            drive a violation. 0 = disabled.
        real_kv_prob: per-forward probability of flipping one byte of the real KV pool at an
            alive-but-not-verified-this-step slot. Only meaningful when real_kv_hash_mode != OFF.
        real_kv_require_orphan: when True the real-KV perturb hook skips this step if no
            radix-cache orphan is currently available (no fallback to running-req slots). Used by
            sweep-only self-tests that must guarantee any fired violation has kernel_kind=SWEEP_*.
    """

    req_to_token_prob: float = 0.0
    real_kv_prob: float = 0.0
    real_kv_require_orphan: bool = False

    @classmethod
    def from_env(cls) -> "PerturbConfig":
        return cls(
            req_to_token_prob=envs.SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_PROB.get(),
            real_kv_prob=envs.SGLANG_KV_CANARY_REAL_PERTURB_BYTES_PROB.get(),
            real_kv_require_orphan=envs.SGLANG_KV_CANARY_REAL_PERTURB_BYTES_REQUIRE_ORPHAN.get(),
        )
