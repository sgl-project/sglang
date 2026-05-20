from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

from sglang.jit_kernel.kv_canary.verify import RealKvHashMode
from sglang.jit_kernel.kv_canary.write import CanaryPseudoMode as CanaryInputCheckMode
from sglang.srt.environ import envs
from sglang.srt.kv_canary.runner.jitter import JitterConfig

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


class CanaryMode(str, Enum):
    OFF = "off"
    ON = "on"
    RAISE = "raise"


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
        sweep_interval: 0 disables sweep entirely; positive N means every N-th forward step the runner
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
            Reads from SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_PROB env var if not set explicitly.
        perturb_real_kv_prob: Same as above but trampling real KV bytes (only meaningful when
            real_kv_hash_mode != OFF). Reads from SGLANG_KV_CANARY_REAL_PERTURB_BYTES_PROB if not set.
        perturb_real_kv_require_orphan: When True the real-KV perturb hook skips this step if no
            radix-cache orphan is currently available (no fallback to running-req slots). Used by
            sweep-only self-tests that must guarantee any fired violation has kernel_kind=SWEEP_*.
            Reads from SGLANG_KV_CANARY_REAL_PERTURB_BYTES_REQUIRE_ORPHAN if not set.
        stats_print_every_n_steps: 0 disables periodic stats logging; positive N prints
            "canary protected N tokens, ran M sweep passes, K violations so far" every N forward steps.
        allreduce_violation_signal: True = end-of-step pump performs cross-rank allreduce on the local
            is_errored byte so all ranks raise in lockstep; False = each rank raises independently (faster
            but produces partial-failure logs across TP groups). Default True.
        jitter_config: timing-jitter fuzzer settings. Default-constructed JitterConfig is disabled; when
            ``mode == "off"`` the runner never installs jitter even if this is enabled.
    """

    mode: Literal["off", "on", "raise"]
    ring_capacity: int = 1024
    sweep_interval: int = 64
    real_kv_hash_mode: RealKvHashMode = RealKvHashMode.BIT
    input_check_mode: CanaryInputCheckMode = CanaryInputCheckMode.OFF
    perturb_req_to_token_prob: float = 0.0
    perturb_real_kv_prob: float = 0.0
    perturb_real_kv_require_orphan: bool = False
    stats_print_every_n_steps: int = 0
    allreduce_violation_signal: bool = True
    jitter_config: JitterConfig = JitterConfig()

    @classmethod
    def from_env(cls, server_args: "ServerArgs") -> "CanaryConfig":
        mode_raw = (server_args.kv_canary or "").strip().lower()
        if mode_raw in ("", "off"):
            mode_raw = envs.SGLANG_KV_CANARY_MODE.get().strip().lower()
        if mode_raw not in ("off", "on", "raise"):
            raise ValueError(
                f"kv-canary: kv_canary must be one of off/on/raise, got {mode_raw!r}"
            )

        real_kv_cli = (server_args.kv_canary_real_data or "").strip().upper()
        if real_kv_cli:
            real_kv_raw = real_kv_cli
        else:
            real_kv_raw = envs.SGLANG_KV_CANARY_REAL_KV_HASH_MODE.get().strip().upper()
        if real_kv_raw not in RealKvHashMode.__members__:
            raise ValueError(
                f"kv-canary: kv_canary_real_data must be one of "
                f"{list(RealKvHashMode.__members__)}, got {real_kv_raw!r}"
            )

        input_check_mode = (
            CanaryInputCheckMode.ON
            if server_args.kv_canary_input_check
            else CanaryInputCheckMode.OFF
        )

        sweep_cli = int(server_args.kv_canary_sweep_interval or 0)
        if sweep_cli > 0:
            sweep_interval = sweep_cli
        else:
            sweep_interval = envs.SGLANG_KV_CANARY_SWEEP_EVERY_N_STEPS.get()

        jitter_config = JitterConfig(
            enabled=envs.SGLANG_KV_CANARY_JITTER_ENABLED.get(),
            per_slot_fire_prob=envs.SGLANG_KV_CANARY_JITTER_PER_SLOT_FIRE_PROB.get(),
            max_cycles=envs.SGLANG_KV_CANARY_JITTER_MAX_CYCLES.get(),
            seed=envs.SGLANG_KV_CANARY_JITTER_SEED.get(),
        )
        if jitter_config.enabled and mode_raw == "off":
            raise ValueError(
                "SGLANG_KV_CANARY_JITTER_ENABLED requires --kv-canary in {on, raise}"
            )

        return cls(
            mode=mode_raw,  # type: ignore[arg-type]
            ring_capacity=envs.SGLANG_KV_CANARY_RING_CAPACITY.get(),
            sweep_interval=sweep_interval,
            real_kv_hash_mode=RealKvHashMode[real_kv_raw],
            input_check_mode=input_check_mode,
            perturb_req_to_token_prob=envs.SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_PROB.get(),
            perturb_real_kv_prob=envs.SGLANG_KV_CANARY_REAL_PERTURB_BYTES_PROB.get(),
            perturb_real_kv_require_orphan=envs.SGLANG_KV_CANARY_REAL_PERTURB_BYTES_REQUIRE_ORPHAN.get(),
            stats_print_every_n_steps=envs.SGLANG_KV_CANARY_STATS_PRINT_EVERY_N_STEPS.get(),
            allreduce_violation_signal=envs.SGLANG_KV_CANARY_ALLREDUCE_VIOLATION_SIGNAL.get(),
            jitter_config=jitter_config,
        )
