from __future__ import annotations

import functools
from dataclasses import dataclass
from enum import Enum

from sglang.kernels.ops.kv_canary.consts import (
    RealKvHashMode,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_observability,
    get_parallel,
    get_serving,
    get_spec,
)


class CanaryMode(str, Enum):
    NONE = "none"
    LOG = "log"
    RAISE = "raise"


def _parse_canary_mode() -> CanaryMode:
    mode_raw = get_observability().kv_canary.strip().lower()
    if mode_raw not in ("none", "log", "raise"):
        raise ValueError(
            f"kv-canary: kv_canary must be one of none/log/raise, got {mode_raw!r}"
        )
    return CanaryMode(mode_raw)


@functools.cache
def kv_canary_mode() -> CanaryMode:
    return _parse_canary_mode()


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryConfig:
    """Top-level canary configuration. All knobs live here; nothing reads env vars deeper in the stack.

    Constructed once inside install_canary(model_runner=..., token_oracle_manager=...) via
    CanaryConfig.from_env(), then frozen and threaded through the canary stack.
    Subsequent runtime never mutates it.

    Fields:
        mode: CanaryMode value. none = no canary installed; log = canary runs, violations are logged
            but do NOT raise (used for production observability + canary self-test perturb); raise =
            violations propagate to host as RuntimeError after the next D2H pump.
        ring_capacity: Violation ring capacity (rows in ViolationLog.violation_ring). Sized generously;
            overflow only drops detail beyond row N, the monotonic counter still grows.
        sweep_interval: 0 disables sweep entirely; positive N means every N-th forward step the runner
            additionally walks all radix-tree-held slots (overlap with per-forward HEAD/TAIL is harmless
            redundancy) and verifies them.
        real_kv_hash_mode: RealKvHashMode (NONE / PARTIAL / ALL). Uniform across head/tail/sweep launches;
            PARTIAL (first 16B, hard cap) is cheap enough for production defaults.
        enable_write_input_assert: bool. True = launch_canary_write_kernel additionally compares
            forward_batch.input_ids[i] / positions[i] against caller-supplied expected_input_tokens[i] /
            expected_input_positions[i]; mismatch records a violation. Only useful when something else
            (e.g. token_oracle.oracle_manager.fill_expected_inputs) is feeding the expected_* placeholders
            per forward — canary itself knows no oracle.
        enable_verify_token_assert: bool. True = real-model token-id validator: build
            expected_tokens from each req's ``origin_input_ids + output_ids`` (snapshotted at
            ForwardBatch.init_new) and compare against the canary's stored tokens at verify time.
            Independent of ``enable_write_input_assert``.
        stats_print_every_n_steps: 0 disables periodic stats logging; positive N prints
            "canary protected N tokens, ran M sweep passes, K violations so far" every N forward steps.
    """

    mode: CanaryMode
    ring_capacity: int
    sweep_interval: int
    real_kv_hash_mode: RealKvHashMode
    enable_write_input_assert: bool
    enable_verify_token_assert: bool
    enable_info: bool = False
    stats_print_every_n_steps: int

    @classmethod
    def from_env(cls) -> CanaryConfig:
        real_kv_raw = get_observability().kv_canary_real_data.strip().upper()
        mode = _parse_canary_mode()
        enable_info = envs.SGLANG_KV_CANARY_ENABLE_INFO.get()
        if enable_info and mode is not CanaryMode.NONE:
            assert get_exec().dllm.dllm_algorithm is None, (
                "KvCanaryInfo does not support dLLM"
            )
            assert get_parallel().pp_size == 1, "KvCanaryInfo does not support PP"
            assert get_disagg().disaggregation_mode == "null", (
                "KvCanaryInfo does not support PD disaggregation"
            )
            assert not get_serving().allow_auto_truncate, (
                "KvCanaryInfo does not support allow_auto_truncate"
            )
            assert get_spec().speculative_algorithm in (
                None,
                "EAGLE",
                "EAGLE3",
                "STANDALONE",
            ), "KvCanaryInfo does not support this speculative algorithm"
        if (
            enable_info
            and mode is not CanaryMode.NONE
            and get_spec().speculative_algorithm is not None
        ):
            assert get_spec().speculative_eagle_topk == 1, (
                "KvCanaryInfo only supports speculative_eagle_topk=1"
            )
            assert (
                get_spec().speculative_num_draft_tokens
                == get_spec().speculative_num_steps + 1
            ), (
                "KvCanaryInfo requires speculative_num_draft_tokens=speculative_num_steps+1"
            )

        return cls(
            mode=mode,
            ring_capacity=envs.SGLANG_KV_CANARY_RING_CAPACITY.get(),
            sweep_interval=get_observability().kv_canary_sweep_interval,
            real_kv_hash_mode=RealKvHashMode[real_kv_raw],
            enable_write_input_assert=envs.SGLANG_KV_CANARY_ENABLE_WRITE_INPUT_ASSERT.get(),
            enable_verify_token_assert=envs.SGLANG_KV_CANARY_ENABLE_VERIFY_TOKEN_ASSERT.get(),
            enable_info=enable_info,
            stats_print_every_n_steps=envs.SGLANG_KV_CANARY_STATS_PRINT_EVERY_N_STEPS.get(),
        )
