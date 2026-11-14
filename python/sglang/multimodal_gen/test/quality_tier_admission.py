# SPDX-License-Identifier: Apache-2.0
"""Admission checks for a fast path that claims the ``lossless`` tier.

A ``lossless``-tier fast path keeps the reference math and the precision of
every operand and accumulator; only the place or the order of the rounding
moves. That claim is not provable from a single tolerance, so these helpers
implement the checks a fusion has to pass before its tier is believed, and
replace the per-case thresholds each fusion test used to invent:

* :func:`assert_error_no_worse_than_reference` -- the operator-level gate. Both
  the current implementation and the candidate are scored against an fp64
  evaluation of the same math; the candidate may not be more than
  ``max_error_ratio`` times worse on RMS or max error, and its error may not
  carry a systematic sign.
* :func:`assert_special_values_match` -- NaN, Inf, fully-masked rows and
  non-contiguous inputs keep their current behavior.
* :func:`discrete_flip_rate` -- when a quantizer, top-k or threshold reads the
  output, the quantity that matters is how often the discrete result changes,
  not the size of the continuous error.

Two gates are deliberately not automated here, because they are properties of
the change rather than of one tensor comparison:

* the static conditions (accumulation at fp32 or better, no operand or
  intermediate precision lowered, an approximate transcendental within 0.1 ulp
  of the output dtype, stability no worse than the reference: two-pass or
  Welford variance, max-subtracted softmax). A fast path that breaks one of
  them is approximate by construction, and no measurement can admit it.
* the end-to-end drift band -- the difference against the exact tier has to sit
  inside the spread the same reference path already shows across two legal
  environments (H100 against H200, TP1 against TP2, two cuBLAS versions). That
  needs a server run, so it belongs to the model's consistency suite.

Feed the operator-level gate activations captured from the model, not only
random tensors: DiT activations carry outlier channels that decide whether a
reassociated reduction stays within the reference's error.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

Impl = Callable[..., torch.Tensor]


@dataclass(frozen=True)
class ErrorScore:
    """One implementation's error against the fp64 evaluation of its math."""

    rms: float
    max_abs: float
    mean: float
    """Signed mean error; a value far from 0 means the rounding is biased."""

    def __str__(self) -> str:
        return f"rms={self.rms:.3e} max={self.max_abs:.3e} mean={self.mean:+.3e}"


def score_against(reference_fp64: torch.Tensor, actual: torch.Tensor) -> ErrorScore:
    """Score ``actual`` against an fp64 evaluation of the same math."""
    if reference_fp64.shape != actual.shape:
        raise ValueError(
            f"shape mismatch: reference {tuple(reference_fp64.shape)} vs "
            f"actual {tuple(actual.shape)}"
        )
    error = actual.to(torch.float64) - reference_fp64
    return ErrorScore(
        rms=float(error.pow(2).mean().sqrt()),
        max_abs=float(error.abs().max()),
        mean=float(error.mean()),
    )


def assert_error_no_worse_than_reference(
    *,
    reference_fp64: torch.Tensor,
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    max_error_ratio: float = 1.5,
    max_bias_in_rms: float = 0.25,
    label: str = "candidate",
) -> tuple[ErrorScore, ErrorScore]:
    """Admit ``candidate`` to the lossless tier on operator-level error.

    ``baseline`` is what the model runs today and ``reference_fp64`` the same
    math evaluated in fp64. The candidate passes when it is no more than
    ``max_error_ratio`` times the baseline's RMS and max error -- the baseline
    sets the scale, because the reference path is itself only one sample of
    the rounding -- and when its mean error stays within ``max_bias_in_rms``
    of its own RMS, which catches a rounding that always leans one way and
    would accumulate over layers and steps.
    """
    base = score_against(reference_fp64, baseline)
    cand = score_against(reference_fp64, candidate)

    for metric, base_value, cand_value in (
        ("RMS", base.rms, cand.rms),
        ("max", base.max_abs, cand.max_abs),
    ):
        # An exactly-zero baseline error means the reference math is
        # representable in the output dtype; hold the candidate to the same.
        budget = base_value * max_error_ratio
        if cand_value > budget:
            raise AssertionError(
                f"{label} is not lossless-tier: {metric} error {cand_value:.3e} "
                f"exceeds {max_error_ratio}x the reference path's "
                f"{base_value:.3e}. reference={base} candidate={cand}"
            )
    if cand.rms > 0 and abs(cand.mean) > max_bias_in_rms * cand.rms:
        raise AssertionError(
            f"{label} has a biased rounding: mean error {cand.mean:+.3e} is "
            f"more than {max_bias_in_rms} of its RMS {cand.rms:.3e}, so the "
            f"error accumulates instead of cancelling. candidate={cand}"
        )
    return base, cand


def assert_special_values_match(
    *,
    baseline: Impl,
    candidate: Impl,
    cases: dict[str, Sequence[torch.Tensor]],
    label: str = "candidate",
) -> None:
    """Both paths agree on NaN, Inf, empty and non-contiguous inputs.

    Agreement is structural: the same NaN positions, the same infinities with
    the same signs, finite where the reference is finite, or the same
    exception type. A fast path that swallows a NaN (``fmaxf``), returns zeros
    where the reference returns NaN, or lets a NaN leak into rows it should
    not reach is a defect rather than a tier.

    A kernel that ignores a stride or aliases its inputs reads the wrong
    elements, which is a magnitude error rather than a structural one: pass a
    non-contiguous input to :func:`assert_error_no_worse_than_reference` as
    well, where it shows up as an error far outside the budget.

    It deliberately does not compare the magnitude of the finite values. A
    lossless-tier path differs from the reference there by construction, and
    judging that difference is :func:`assert_error_no_worse_than_reference`'s
    job, on real activations rather than on these edge-case inputs. An earlier
    version compared them with ``torch.allclose`` defaults, which rejected
    every bf16 fast path this gate exists to admit.
    """
    for name, inputs in cases.items():
        try:
            expected = baseline(*inputs)
        except Exception as exc:  # noqa: BLE001 - the reference's own behavior
            expected_error: type[BaseException] | None = type(exc)
            expected = None
        else:
            expected_error = None

        try:
            actual = candidate(*inputs)
        except Exception as exc:  # noqa: BLE001 - compared against the reference
            if expected_error is None:
                raise AssertionError(
                    f"{label} raised {type(exc).__name__} on {name!r} where "
                    f"the reference path returned a tensor"
                ) from exc
            if type(exc) is not expected_error:
                raise AssertionError(
                    f"{label} raised {type(exc).__name__} on {name!r}, the "
                    f"reference path raised {expected_error.__name__}"
                ) from exc
            continue

        if expected_error is not None:
            raise AssertionError(
                f"{label} returned a tensor on {name!r} where the reference "
                f"path raised {expected_error.__name__}"
            )
        if actual.shape != expected.shape:
            raise AssertionError(
                f"{label} returns {tuple(actual.shape)} on {name!r} where the "
                f"reference path returns {tuple(expected.shape)}"
            )
        if not torch.equal(actual.isnan(), expected.isnan()):
            raise AssertionError(f"{label} moves NaNs on {name!r}")
        # Signed, so an inf that flips direction is caught too.
        for sign, what in ((1, "+Inf"), (-1, "-Inf")):
            if not torch.equal(
                actual == sign * torch.inf, expected == sign * torch.inf
            ):
                raise AssertionError(f"{label} moves {what}s on {name!r}")


def discrete_flip_rate(baseline: torch.Tensor, candidate: torch.Tensor) -> float:
    """Fraction of entries whose discrete downstream result differs.

    Pass the discrete values -- quantized codes, top-k indices, threshold
    decisions -- not the continuous tensors they came from. One ulp upstream of
    a quantizer or a ``skip-softmax`` threshold is no longer a rounding
    difference, so a lossless-tier claim in front of one is judged on this rate.
    """
    if baseline.shape != candidate.shape:
        raise ValueError(
            f"shape mismatch: {tuple(baseline.shape)} vs {tuple(candidate.shape)}"
        )
    if baseline.numel() == 0:
        return 0.0
    return float((baseline != candidate).to(torch.float64).mean())
