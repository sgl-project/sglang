from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

from sglang.jit_kernel.kv_canary.consts import (
    RealKvHashMode,
)
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


class CanaryMode(str, Enum):
    OFF = "off"
    LOG = "log"
    RAISE = "raise"


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryConfig:
    """Top-level canary configuration. All knobs live here; nothing reads env vars deeper in the stack.

    Constructed once at server startup from CLI flags / env vars, then frozen and passed into
    install_canary(server_args, model_runner) once. Subsequent runtime never mutates it.

    Fields:
        mode: "off" | "log" | "raise". off = no canary installed; log = canary runs, violations are logged
            but do NOT raise (used for production observability + canary self-test perturb); raise =
            violations propagate to host as RuntimeError after the next D2H pump.
        ring_capacity: Violation ring capacity (rows in ViolationLog.violation_ring). Sized generously;
            overflow only drops detail beyond row N, the monotonic counter still grows.
        sweep_interval: 0 disables sweep entirely; positive N means every N-th forward step the runner
            additionally walks all radix-tree-held slots (overlap with per-forward HEAD/TAIL is harmless
            redundancy) and verifies them.
        real_kv_hash_mode: RealKvHashMode (OFF / PARTIAL / ALL). Uniform across head/tail/sweep launches;
            PARTIAL (first 16B, hard cap) is cheap enough for production defaults.
        input_check_mode: bool. True = launch_canary_write_kernel additionally compares
            forward_batch.input_ids[i] / positions[i] against caller-supplied expected_input_tokens[i] /
            expected_input_positions[i]; mismatch records a violation. Only useful when something else
            (e.g. token_oracle.oracle_manager.fill_expected_inputs) is feeding the expected_* placeholders
            per forward — canary itself knows no oracle.
        stats_print_every_n_steps: 0 disables periodic stats logging; positive N prints
            "canary protected N tokens, ran M sweep passes, K violations so far" every N forward steps.
    """

    mode: Literal["off", "log", "raise"]
    ring_capacity: int
    sweep_interval: int
    real_kv_hash_mode: RealKvHashMode
    input_check_mode: bool
    stats_print_every_n_steps: int

    @classmethod
    def from_env(cls, server_args: "ServerArgs") -> "CanaryConfig":
        mode_raw = server_args.kv_canary.strip().lower()
        if mode_raw not in ("off", "log", "raise"):
            raise ValueError(
                f"kv-canary: kv_canary must be one of off/log/raise, got {mode_raw!r}"
            )

        real_kv_raw = server_args.kv_canary_real_data.strip().upper()

        return cls(
            mode=mode_raw,  # type: ignore[arg-type]
            ring_capacity=envs.SGLANG_KV_CANARY_RING_CAPACITY.get(),
            sweep_interval=server_args.kv_canary_sweep_interval,
            real_kv_hash_mode=RealKvHashMode[real_kv_raw],
            input_check_mode=envs.SGLANG_KV_CANARY_INPUT_CHECK.get(),
            stats_print_every_n_steps=envs.SGLANG_KV_CANARY_STATS_PRINT_EVERY_N_STEPS.get(),
        )
