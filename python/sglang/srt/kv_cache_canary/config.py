from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from sglang.jit_kernel.kv_cache_canary_verify import RealKvHashMode
from sglang.jit_kernel.kv_cache_canary_write import (
    CanaryPseudoMode as CanaryInputCheckMode,
)

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryConfig:
    """Top-level canary configuration. All knobs live here; nothing reads env vars deeper in the stack.

    Constructed once at server startup from CLI flags / env vars, then frozen and passed into
    install_canary(server_args, model_runner) once. Subsequent runtime never mutates it.

    Fields:
        mode: "off" | "on" | "raise". off = no canary installed; on = canary runs, violations are logged
            but do NOT raise (used for production observability + canary self-test perturb); raise =
            violations propagate to host as RuntimeError after the next D2H pump.
        ring_capacity: Violation ring capacity (rows in ViolationLog.violation_ring). Sized generously
            (default 1024); overflow only drops detail beyond row N, the monotonic counter still grows.
        sweep_every_n_steps: 0 disables sweep entirely; positive N means every N-th forward step the runner
            additionally walks all alive slots (running ∪ radix-orphan) and verifies them.
        real_kv_hash_mode: RealKvHashMode (OFF / BIT / ALL). Uniform across head/tail/sweep launches; if a
            workload wants per-launch granularity it bumps mode globally (BIT is cheap enough this is fine).
        input_check_mode: CanaryInputCheckMode (OFF / ON). ON = canary_write_step additionally compares
            forward_batch.input_ids[i] / positions[i] against caller-supplied expected_input_tokens[i] /
            expected_input_positions[i]; mismatch records a violation. ON is only useful when something
            else (e.g. mock_model.sampler.fill_expected_inputs) is feeding the expected_* placeholders
            per forward — canary itself knows no oracle.
        perturb_req_to_token_prob: For canary self-test only. 0 = disabled; positive (e.g. 1e-4) = each
            forward step has this probability of trampling a random req_to_token entry to drive a violation.
            Reads from SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN env var if not set explicitly.
        perturb_real_kv_prob: Same as above but trampling real KV bytes (only meaningful when
            real_kv_hash_mode != OFF). Reads from SGLANG_KV_CANARY_PERTURB_REAL_KV if not set.
        stats_print_every_n_steps: 0 disables periodic stats logging; positive N prints
            "canary protected N tokens, ran M sweep passes, K violations so far" every N forward steps.
        allreduce_violation_signal: True = end-of-step pump performs cross-rank allreduce on the local
            is_errored byte so all ranks raise in lockstep; False = each rank raises independently (faster
            but produces partial-failure logs across TP groups). Default True.
    """

    mode: Literal["off", "on", "raise"]
    ring_capacity: int = 1024
    sweep_every_n_steps: int = 64
    real_kv_hash_mode: RealKvHashMode = RealKvHashMode.BIT
    input_check_mode: CanaryInputCheckMode = CanaryInputCheckMode.OFF
    perturb_req_to_token_prob: float = 0.0
    perturb_real_kv_prob: float = 0.0
    stats_print_every_n_steps: int = 0
    allreduce_violation_signal: bool = True

    @classmethod
    def from_env(cls, server_args: "ServerArgs") -> "CanaryConfig":
        return cls(
            mode=_parse_mode(os.getenv("SGLANG_KV_CANARY_MODE", "off")),
            ring_capacity=int(os.getenv("SGLANG_KV_CANARY_RING_CAPACITY", "1024")),
            sweep_every_n_steps=int(
                os.getenv("SGLANG_KV_CANARY_SWEEP_EVERY_N_STEPS", "64")
            ),
            real_kv_hash_mode=_parse_real_kv_hash_mode(
                os.getenv("SGLANG_KV_CANARY_REAL_KV_HASH_MODE", "BIT")
            ),
            input_check_mode=_parse_input_check_mode(
                os.getenv("SGLANG_KV_CANARY_INPUT_CHECK_MODE", "OFF")
            ),
            perturb_req_to_token_prob=float(
                os.getenv("SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN", "0.0")
            ),
            perturb_real_kv_prob=float(
                os.getenv("SGLANG_KV_CANARY_PERTURB_REAL_KV", "0.0")
            ),
            stats_print_every_n_steps=int(
                os.getenv("SGLANG_KV_CANARY_STATS_PRINT_EVERY_N_STEPS", "0")
            ),
            allreduce_violation_signal=_parse_bool(
                os.getenv("SGLANG_KV_CANARY_ALLREDUCE_VIOLATION_SIGNAL", "1")
            ),
        )


def _parse_mode(value: str) -> Literal["off", "on", "raise"]:
    lowered = value.strip().lower()
    if lowered not in ("off", "on", "raise"):
        raise ValueError(
            f"kv-canary: SGLANG_KV_CANARY_MODE must be one of off/on/raise, got {value!r}"
        )
    return lowered  # type: ignore[return-value]


def _parse_real_kv_hash_mode(value: str) -> RealKvHashMode:
    upper = value.strip().upper()
    if upper not in RealKvHashMode.__members__:
        raise ValueError(
            f"kv-canary: SGLANG_KV_CANARY_REAL_KV_HASH_MODE must be one of "
            f"{list(RealKvHashMode.__members__)}, got {value!r}"
        )
    return RealKvHashMode[upper]


def _parse_input_check_mode(value: str) -> CanaryInputCheckMode:
    upper = value.strip().upper()
    if upper not in CanaryInputCheckMode.__members__:
        raise ValueError(
            f"kv-canary: SGLANG_KV_CANARY_INPUT_CHECK_MODE must be one of "
            f"{list(CanaryInputCheckMode.__members__)}, got {value!r}"
        )
    return CanaryInputCheckMode[upper]


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in ("1", "true", "yes", "y", "on"):
        return True
    if lowered in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"kv-canary: cannot parse boolean from {value!r}")
