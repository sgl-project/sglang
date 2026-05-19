"""CanaryConfig. Schema retained from the legacy module minus the ``seed`` field (anchor is hardcoded in
jit_kernel; see kernels.md §6.1) plus a ``pseudo_mode`` toggle for caller-driven write-time expectation checks
(kernels.md §2.6, §2.8).
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import torch

from sglang.jit_kernel.kv_cache_canary_verify import RealKvHashMode
from sglang.jit_kernel.kv_cache_canary_write import CanaryPseudoMode
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class CanaryMode(str, enum.Enum):
    OFF = "off"
    LOG = "log"
    RAISE = "raise"

    @classmethod
    def parse(cls, value: "str | CanaryMode | None") -> "CanaryMode":
        if value is None:
            return cls.OFF
        if isinstance(value, cls):
            return value
        return cls(value)


def parse_real_kv_hash_mode(value: "str | RealKvHashMode | None") -> RealKvHashMode:
    """Parse a server-arg value into the jit_kernel :class:`RealKvHashMode` IntEnum.

    Accepts the new BIT / ALL / OFF names, plus the legacy ``portion`` alias that the existing CLI flag still
    uses (mapped to BIT — both are "cheap fingerprint of a leading prefix").
    """
    if value is None:
        return RealKvHashMode.OFF
    if isinstance(value, RealKvHashMode):
        return value
    lowered = str(value).lower()
    if lowered in ("off", "0"):
        return RealKvHashMode.OFF
    if lowered in ("bit", "portion"):
        return RealKvHashMode.BIT
    if lowered in ("all", "full"):
        return RealKvHashMode.ALL
    raise ValueError(f"kv-canary: unknown RealKvHashMode value {value!r}")


PseudoOracleCallback = Callable[
    ["ForwardBatch"], Tuple[torch.Tensor, torch.Tensor]
]


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryConfig:
    """Configuration for one canary runner.

    Schema is retained from the legacy module with two exceptions:

    - ``seed`` is dropped — the chain anchor is now the hardcoded :data:`CANARY_CHAIN_ANCHOR` constant in
      ``jit_kernel.kv_cache_canary_verify`` (kernels.md §6.1).
    - ``pseudo_mode`` and ``pseudo_oracle`` are added — when ``pseudo_mode == ON`` the runner invokes the
      oracle once per forward to produce ``(pseudo_expected_tokens, pseudo_expected_positions)`` for
      ``canary_write_step`` (kernels.md §2.6).
    """

    mode: CanaryMode
    violation_ring_capacity: int = 1024
    health_print_every_n_forwards: int = 1024
    counter_zero_warmup_forwards: int = 64
    perturb_req_to_token_prob: float = 0.0
    perturb_req_to_token_seed: int = 0
    swa_window_size: Optional[int] = None
    real_kv_hash_mode: RealKvHashMode = RealKvHashMode.OFF
    real_kv_read_bytes: int = 16
    real_data_sweep_every_n_steps: int = 0
    real_perturb_bytes_prob: float = 0.0
    real_perturb_bytes_seed: int = 0
    pseudo_mode: CanaryPseudoMode = CanaryPseudoMode.OFF
    pseudo_oracle: Optional[PseudoOracleCallback] = None

    def __post_init__(self) -> None:
        if self.real_data_sweep_every_n_steps < 0:
            raise ValueError(
                "kv-canary: real_data_sweep_every_n_steps must be >= 0, "
                f"got {self.real_data_sweep_every_n_steps}"
            )
        if self.real_perturb_bytes_prob < 0.0:
            raise ValueError(
                "kv-canary: real_perturb_bytes_prob must be >= 0.0, "
                f"got {self.real_perturb_bytes_prob}"
            )
        if self.real_perturb_bytes_seed < 0:
            raise ValueError(
                "kv-canary: real_perturb_bytes_seed must be >= 0, "
                f"got {self.real_perturb_bytes_seed}"
            )
        if self.real_kv_read_bytes < 0:
            raise ValueError(
                "kv-canary: real_kv_read_bytes must be >= 0, "
                f"got {self.real_kv_read_bytes}"
            )
        if self.pseudo_mode is CanaryPseudoMode.ON and self.pseudo_oracle is None:
            raise ValueError(
                "kv-canary: pseudo_mode == ON requires a pseudo_oracle callback"
            )

    @classmethod
    def from_server_args(
        cls,
        mode: "str | CanaryMode | None",
        real_kv_hash_mode: "str | RealKvHashMode | None" = None,
        real_data_sweep_every_n_steps: int = 0,
    ) -> "CanaryConfig":
        parsed = CanaryMode.parse(mode)
        raw_prob = envs.SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_PROB.get()
        perturb_prob = max(0.0, min(1.0, raw_prob))
        if perturb_prob != raw_prob:
            logger.warning(
                "kv-canary: SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_PROB %f "
                "out of [0,1]; clamped to %f",
                raw_prob,
                perturb_prob,
            )
        raw_real_prob = envs.SGLANG_KV_CANARY_REAL_PERTURB_BYTES_PROB.get()
        real_perturb_prob = max(0.0, min(1.0, raw_real_prob))
        if real_perturb_prob != raw_real_prob:
            logger.warning(
                "kv-canary: SGLANG_KV_CANARY_REAL_PERTURB_BYTES_PROB %f "
                "out of [0,1]; clamped to %f",
                raw_real_prob,
                real_perturb_prob,
            )
        return cls(
            mode=parsed,
            perturb_req_to_token_prob=perturb_prob,
            perturb_req_to_token_seed=envs.SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_SEED.get(),
            real_kv_hash_mode=parse_real_kv_hash_mode(real_kv_hash_mode),
            real_data_sweep_every_n_steps=int(real_data_sweep_every_n_steps),
            real_perturb_bytes_prob=real_perturb_prob,
            real_perturb_bytes_seed=int(
                envs.SGLANG_KV_CANARY_REAL_PERTURB_BYTES_SEED.get()
            ),
        )

    @property
    def enabled(self) -> bool:
        return self.mode is not CanaryMode.OFF
