# SPDX-License-Identifier: Apache-2.0

"""CLI- and API-facing configuration for diffusion post-training / rollout paths."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from sglang.multimodal_gen.runtime.utils.argparse import StoreBoolean

_VALID_ROLLOUT_SDE_TYPES = ("sde", "cps", "ode")
# The settings of a stream's SDE kernel; an extra stream may override each of them.
_STREAM_SDE_KERNEL_FIELDS = (
    "rollout_sde_type",
    "rollout_noise_level",
    "rollout_log_prob_no_const",
)

# Name of the stream the denoise loop iterates.
PRIMARY_STREAM = "primary"
# Stream name of ``OutputBatch.audio``; a pipeline that rolls out audio uses it too.
AUDIO_STREAM = "audio"
# Extra stream names become output-store path segments, which are dot-separated.
_STREAM_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")


def stream_sde_kernel_settings(params: Any, stream: str) -> dict[str, Any]:
    """The SDE kernel settings ``stream`` is rolled out with: the top-level fields for the
    primary stream, and for an extra stream those fields with its override applied."""
    settings = {name: getattr(params, name) for name in _STREAM_SDE_KERNEL_FIELDS}
    if stream != PRIMARY_STREAM:
        settings.update(params.rollout_extra_stream_sde_args.get(stream, {}))
    return settings


@dataclass
class RLRolloutArgs:
    """Rollout (log-prob trajectory) options used by SamplingParams and APIs."""

    rollout: bool = False
    rollout_extra_streams: tuple[str, ...] = ()
    rollout_extra_stream_sde_args: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    rollout_sde_type: str = "sde"
    rollout_noise_level: float = 0.7
    rollout_log_prob_no_const: bool = False
    rollout_debug_mode: bool = False
    rollout_return_generated_output: bool = True

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

        self._validate_extra_streams()
        self._validate_extra_stream_sde_args()

        if not self.rollout_return_generated_output and not self.rollout:
            raise ValueError(
                "rollout_return_generated_output=False requires rollout=True; "
                "it is a /rollout/generate option"
            )

    def _validate_extra_streams(self) -> None:
        extra_streams = self.rollout_extra_streams
        if isinstance(extra_streams, str):
            raise ValueError(
                "rollout_extra_streams must be a list of stream names, "
                f"got {extra_streams!r}"
            )
        if extra_streams and not self.rollout:
            raise ValueError("rollout_extra_streams requires rollout=True")
        if len(set(extra_streams)) != len(extra_streams):
            raise ValueError(
                f"rollout_extra_streams has duplicates: {list(extra_streams)}"
            )
        for stream in extra_streams:
            if stream == PRIMARY_STREAM or not (
                isinstance(stream, str) and _STREAM_NAME_PATTERN.fullmatch(stream)
            ):
                raise ValueError(f"Invalid rollout extra stream name: {stream!r}")

    def _validate_extra_stream_sde_args(self) -> None:
        overrides = self.rollout_extra_stream_sde_args
        if not isinstance(overrides, dict):
            raise ValueError(
                "rollout_extra_stream_sde_args must map stream names to "
                f"SDE kernel overrides, got {overrides!r}"
            )
        for stream, override in overrides.items():
            if stream not in self.rollout_extra_streams:
                raise ValueError(
                    f"rollout_extra_stream_sde_args names {stream!r}, "
                    "which is not in rollout_extra_streams "
                    f"{list(self.rollout_extra_streams)}; the top-level fields are "
                    "the primary stream's"
                )
            if not isinstance(override, dict):
                raise ValueError(
                    f"The override of stream {stream!r} must be a dict, got {override!r}"
                )
            unknown = sorted(set(override) - set(_STREAM_SDE_KERNEL_FIELDS))
            if unknown:
                raise ValueError(
                    f"The override of stream {stream!r} sets {unknown}; an override may "
                    f"set only {list(_STREAM_SDE_KERNEL_FIELDS)}"
                )
            _validate_override_values(stream=stream, override=override)
        for stream in self.rollout_extra_streams:
            _validate_stream_sde_kernel(
                stream=stream,
                settings=stream_sde_kernel_settings(self, stream),
                overridden=set(overrides.get(stream, {})),
            )

    @classmethod
    def validate_sampling_params(cls, params: Any) -> None:
        """Validate rollout fields on a duck-typed object (e.g. ``SamplingParams``).

        Mirrors how ``ServerArgs`` runs ``NunchakuSVDQuantArgs.validate()`` from
        ``_adjust_quant_config`` instead of inlining checks in a large validator.
        """
        cls(
            rollout=params.rollout,
            rollout_extra_streams=params.rollout_extra_streams,
            rollout_extra_stream_sde_args=params.rollout_extra_stream_sde_args,
            rollout_sde_type=params.rollout_sde_type,
            rollout_noise_level=params.rollout_noise_level,
            rollout_log_prob_no_const=params.rollout_log_prob_no_const,
            rollout_debug_mode=params.rollout_debug_mode,
            rollout_return_generated_output=params.rollout_return_generated_output,
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
            "--rollout-extra-streams",
            type=str,
            nargs="+",
            help="Streams to roll out besides the primary one, e.g. 'audio' (LTX2Pipeline).",
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
        extra_streams = kwargs.get("rollout_extra_streams", cls.rollout_extra_streams)
        return cls(
            rollout=bool(kwargs.get("rollout", cls.rollout)),
            rollout_extra_streams=(
                extra_streams
                if isinstance(extra_streams, str)
                else tuple(extra_streams)
            ),
            rollout_extra_stream_sde_args=kwargs.get(
                "rollout_extra_stream_sde_args", {}
            ),
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


def _validate_override_values(*, stream: str, override: dict[str, Any]) -> None:
    # Unlike the top-level fields, override values reach here as sent: no coercion.
    if "rollout_sde_type" in override and (
        override["rollout_sde_type"] not in _VALID_ROLLOUT_SDE_TYPES
    ):
        raise ValueError(
            f"Stream {stream!r}: rollout_sde_type must be one of "
            f"{_VALID_ROLLOUT_SDE_TYPES}, got {override['rollout_sde_type']!r}"
        )
    if "rollout_noise_level" in override:
        noise = override["rollout_noise_level"]
        if (
            isinstance(noise, bool)
            or not isinstance(noise, (int, float))
            or not math.isfinite(noise)
            or noise < 0
        ):
            raise ValueError(
                f"Stream {stream!r}: rollout_noise_level must be a finite "
                f"non-negative number, got {noise!r}"
            )
    if "rollout_log_prob_no_const" in override and not isinstance(
        override["rollout_log_prob_no_const"], bool
    ):
        raise ValueError(
            f"Stream {stream!r}: rollout_log_prob_no_const must be a bool, "
            f"got {override['rollout_log_prob_no_const']!r}"
        )


def _validate_stream_sde_kernel(
    *, stream: str, settings: dict[str, Any], overridden: set[str]
) -> None:
    """Reject the SDE kernels FlowRolloutController would reject at its first step."""

    def described(name: str) -> str:
        source = "override" if name in overridden else "inherited from the top level"
        return f"{name}={settings[name]!r} ({source})"

    sde_type = settings["rollout_sde_type"]
    if settings["rollout_log_prob_no_const"]:
        return
    if sde_type == "ode":
        raise ValueError(
            f"Stream {stream!r}: {described('rollout_sde_type')} has no log-prob "
            f"with constants, so it needs rollout_log_prob_no_const=True, got "
            f"{described('rollout_log_prob_no_const')}"
        )
    if not float(settings["rollout_noise_level"]) > 0:
        raise ValueError(
            f"Stream {stream!r}: {described('rollout_sde_type')} with "
            f"{described('rollout_log_prob_no_const')} needs rollout_noise_level > 0, "
            f"got {described('rollout_noise_level')}"
        )
