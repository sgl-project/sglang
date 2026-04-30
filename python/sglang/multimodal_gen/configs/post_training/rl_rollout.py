# SPDX-License-Identifier: Apache-2.0

"""CLI- and API-facing configuration for diffusion post-training / rollout paths."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any, Callable

from sglang.multimodal_gen.utils import StoreBoolean

_VALID_ROLLOUT_SDE_TYPES = ("sde", "cps", "ode")


@dataclass
class RLRolloutArgs:
    """Rollout (log-prob trajectory) options used by SamplingParams and APIs."""

    rollout: bool = False
    rollout_sde_type: str = "sde"
    rollout_noise_level: float = 0.7
    rollout_log_prob_no_const: bool = False
    rollout_debug_mode: bool = False

    def validate(self) -> None:
        noise = self.rollout_noise_level
        if isinstance(noise, bool) or not isinstance(noise, (int, float)):
            raise ValueError(f"rollout_noise_level must be a number, got {noise!r}")
        if not math.isfinite(float(noise)):
            raise ValueError(f"rollout_noise_level must be finite, got {noise!r}")
        if float(noise) < 0.0:
            raise ValueError(f"rollout_noise_level must be non-negative, got {noise!r}")

        if self.rollout_sde_type not in _VALID_ROLLOUT_SDE_TYPES:
            raise ValueError(
                f"rollout_sde_type must be one of {_VALID_ROLLOUT_SDE_TYPES}, "
                f"got {self.rollout_sde_type!r}"
            )

    @classmethod
    def validate_sampling_params(cls, params: Any) -> None:
        """Validate rollout fields on a duck-typed object (e.g. ``SamplingParams``).

        Mirrors how ``ServerArgs`` runs ``NunchakuSVDQuantArgs.validate()`` from
        ``_adjust_quant_config`` instead of inlining checks in a large validator.
        """
        cls(
            rollout=params.rollout,
            rollout_sde_type=params.rollout_sde_type,
            rollout_noise_level=params.rollout_noise_level,
            rollout_log_prob_no_const=params.rollout_log_prob_no_const,
            rollout_debug_mode=params.rollout_debug_mode,
        ).validate()

    @staticmethod
    def add_cli_args(
        parser: Any,
        add_argument: Callable[..., Any] | None = None,
    ) -> None:
        """Register rollout-related CLI flags on ``parser``.

        If ``add_argument`` is provided (e.g. SamplingParams' wrapper with
        ``default=argparse.SUPPRESS``), it is used; otherwise a local wrapper
        is applied.
        """

        if add_argument is None:

            def _add(*name_or_flags: Any, **kwargs: Any):
                kwargs.setdefault("default", argparse.SUPPRESS)
                return parser.add_argument(*name_or_flags, **kwargs)

            add_argument = _add

        add_argument(
            "--rollout",
            action="store_true",
            help="Enable rollout mode and return per-step log_prob trajectory",
        )
        add_argument(
            "--rollout-sde-type",
            type=str,
            choices=list(_VALID_ROLLOUT_SDE_TYPES),
            help="Rollout step objective type used in log-prob computation.",
        )
        add_argument(
            "--rollout-noise-level",
            type=float,
            help="Noise level used by rollout SDE/CPS step objective.",
        )
        add_argument(
            "--rollout-log-prob-no-const",
            action=StoreBoolean,
            help="If true, return rollout log-prob without constant terms.",
        )
        add_argument(
            "--rollout-debug-mode",
            action=StoreBoolean,
            help=(
                "If true, return rollout debug tensors "
                "(variance noise, mean, std, model output)."
            ),
        )

    @classmethod
    def from_dict(cls, kwargs: dict[str, Any]) -> RLRolloutArgs:
        return cls(
            rollout=bool(kwargs.get("rollout", cls.rollout)),
            rollout_sde_type=str(kwargs.get("rollout_sde_type", cls.rollout_sde_type)),
            rollout_noise_level=float(
                kwargs.get("rollout_noise_level", cls.rollout_noise_level)
            ),
            rollout_log_prob_no_const=bool(
                kwargs.get("rollout_log_prob_no_const", cls.rollout_log_prob_no_const)
            ),
            rollout_debug_mode=bool(
                kwargs.get("rollout_debug_mode", cls.rollout_debug_mode)
            ),
        )
