"""Timing-jitter fuzzer runner — owns the static cycle buffer and per-step RNG.

Four fixed slots inside the monkey-patched model.forward (PRE_HEAD, POST_HEAD,
POST_ATTN, POST_TAIL). Each slot launches a single ``spin_wait_step`` reading
``cycles[slot]``; host fills new cycle values into the static buffer between
graph replays via ``randomize_for_next_step``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import torch

from sglang.jit_kernel.kv_canary.jitter import spin_wait_step
from sglang.srt.environ import envs


class JitterSlot(IntEnum):
    PRE_HEAD = 0
    POST_HEAD = 1
    POST_ATTN = 2
    POST_TAIL = 3


_NUM_SLOTS: int = len(JitterSlot)


@dataclass(frozen=True, slots=True, kw_only=True)
class JitterConfig:
    """Timing-jitter fuzzer configuration.

    Fields:
        enabled: master switch. When False the runner skips all device-side launches and never
            allocates the cycle buffer.
        per_slot_fire_prob: per-slot per-step probability of firing (cycles > 0). 0.0 -> always 0
            cycles; 1.0 -> every slot fires every step.
        max_cycles: upper bound on the per-slot cycle count. Cycles are sampled log-uniformly over
            [1, max_cycles] when the slot fires; 100_000 is roughly 50us at a 2 GHz SM clock.
        seed: host-side RNG seed. Same seed reruns produce the same cycles sequence.
    """

    enabled: bool = False
    per_slot_fire_prob: float = 0.5
    max_cycles: int = 100_000
    seed: int = 0

    def __post_init__(self) -> None:
        if not (0.0 <= self.per_slot_fire_prob <= 1.0):
            raise ValueError(
                f"kv-canary jitter: per_slot_fire_prob must be in [0, 1], "
                f"got {self.per_slot_fire_prob}"
            )
        if self.max_cycles < 1:
            raise ValueError(
                f"kv-canary jitter: max_cycles must be >= 1, got {self.max_cycles}"
            )

    @classmethod
    def from_env(cls) -> "JitterConfig":
        return cls(
            enabled=envs.SGLANG_KV_CANARY_JITTER_ENABLED.get(),
            per_slot_fire_prob=envs.SGLANG_KV_CANARY_JITTER_PER_SLOT_FIRE_PROB.get(),
            max_cycles=envs.SGLANG_KV_CANARY_JITTER_MAX_CYCLES.get(),
            seed=envs.SGLANG_KV_CANARY_JITTER_SEED.get(),
        )


class JitterRunner:
    """Owns the static [num_slots] int64 cycle tensor and the per-step RNG.

    Constructed only when ``JitterConfig.enabled`` is True; the disabled case is represented by
    ``CanaryRunner._jitter = None`` so the hot path never even checks a flag.
    """

    NUM_SLOTS: int = _NUM_SLOTS

    def __init__(self, *, config: JitterConfig, device: torch.device) -> None:
        if not config.enabled:
            raise ValueError(
                "kv-canary jitter: JitterRunner must only be constructed when enabled=True"
            )

        self._config = config
        self._device = device
        self._cycles: torch.Tensor = torch.zeros(
            _NUM_SLOTS, dtype=torch.int64, device=device
        )
        self._cycles_staging: torch.Tensor = torch.zeros(
            _NUM_SLOTS, dtype=torch.int64, device="cpu", pin_memory=False
        )
        self._slot_views: tuple[torch.Tensor, ...] = tuple(
            self._cycles[slot : slot + 1] for slot in range(_NUM_SLOTS)
        )
        self._generator: torch.Generator = torch.Generator(device="cpu")
        self._generator.manual_seed(int(config.seed))
        self._step_counter: int = 0
        self._log_max: float = math.log(float(max(config.max_cycles, 1)))

    @property
    def config(self) -> JitterConfig:
        return self._config

    @property
    def cycles(self) -> torch.Tensor:
        return self._cycles

    @property
    def step_counter(self) -> int:
        return self._step_counter

    def randomize_for_next_step(self) -> None:
        """Sample 4 cycle values and ``copy_()`` them into the static device tensor.

        Called from ``CanaryRunner.before_forward`` AFTER all canary host-side prep, so the static
        cycle tensor is up-to-date by the time the cuda-graph replay starts.
        """
        fire_prob = self._config.per_slot_fire_prob
        max_cycles = self._config.max_cycles

        if fire_prob <= 0.0:
            self._cycles_staging.zero_()
        else:
            fires = (
                torch.rand(_NUM_SLOTS, generator=self._generator, dtype=torch.float64)
                < fire_prob
            )
            log_samples = (
                torch.rand(_NUM_SLOTS, generator=self._generator, dtype=torch.float64)
                * self._log_max
            )
            cycles_float = torch.exp(log_samples)
            cycles_int = torch.clamp(
                cycles_float.to(torch.int64), min=1, max=int(max_cycles)
            )
            cycles_int[~fires] = 0
            self._cycles_staging.copy_(cycles_int)

        self._cycles.copy_(self._cycles_staging, non_blocking=False)
        self._step_counter += 1

    def launch_slot(self, *, slot: JitterSlot) -> None:
        """Launch ``spin_wait_step`` for one slot. Captured into the cuda graph when called from the
        monkey-patched ``model.forward``.
        """
        spin_wait_step(cycles=self._slot_views[int(slot)])
