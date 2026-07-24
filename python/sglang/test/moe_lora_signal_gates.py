"""Signal-relative correctness gates for MoE LoRA validation.

Implements the binding tolerance policy of the BF16 MoE-LoRA execution plan
(§21.1): every numerical comparison is made on the base-subtracted LoRA delta
and gated relative to the recorded reference-signal magnitude ``S``.  Absolute
default tolerances are forbidden; degenerate signals must use the bitwise
classes instead.

Three check classes exist:

1. Signal-gated: ``max|err| <= S / 10`` and ``rel_l2(err) <= gate(dtype)``.
2. Bitwise-zero: zero-LoRA parity, replay determinism, fixed-order repeats.
3. Poison hygiene: NaN-poisoned inactive domains must stay untouched and never
   leak into outputs.

Every signal-gated check returns a :class:`SignalCheckRecord` so results can be
persisted and re-adjudicated post hoc.
"""

from __future__ import annotations

import msgspec
import torch

# Half-ULP relative rounding scale of BF16 (8 mantissa bits).
BF16_RELATIVE_QUANTUM = 2.0**-9
# A case is certifiable only when the LoRA signal is at least this many BF16
# quanta of the base output; below it the delta drowns in storage rounding.
MIN_SIGNAL_TO_NOISE = 32.0
# Default fraction of S allowed as max-abs error: a fully dropped or misrouted
# delta errs at ~S and can never pass.
DEFAULT_SIGNAL_FRACTION = 0.1

_REL_L2_GATES = {
    "bfloat16": 1e-2,
    "float32": 1e-5,
}


class DegenerateSignalError(ValueError):
    """The reference signal cannot support a signal-relative gate."""


class SignalGates(msgspec.Struct, frozen=True, kw_only=True):
    """Resolved gates for one comparison boundary."""

    signal_max_abs: float
    signal_l2: float
    max_abs_gate: float
    rel_l2_gate: float
    destination_dtype: str
    noise_floor: float
    calibration_note: str = ""


class SignalCheckRecord(msgspec.Struct, frozen=True, kw_only=True):
    """One executed signal-gated comparison, persistable as JSON."""

    label: str
    gates: SignalGates
    observed_max_abs: float
    observed_rel_l2: float
    passed: bool


def bf16_noise_floor(base_reference: torch.Tensor) -> float:
    """BF16 storage quantum of the surrounding base output."""
    if base_reference.numel() == 0:
        return 0.0
    return float(base_reference.abs().max()) * BF16_RELATIVE_QUANTUM


def resolve_signal_gates(
    delta_reference: torch.Tensor,
    *,
    destination_dtype: torch.dtype,
    base_reference: torch.Tensor | None = None,
    signal_fraction: float = DEFAULT_SIGNAL_FRACTION,
    calibrated_max_abs: float | None = None,
    calibration_note: str = "",
) -> SignalGates:
    """Derive gates from the FP32 reference delta.

    Raises :class:`DegenerateSignalError` when the signal is zero (use a
    bitwise check instead) or, when ``base_reference`` is supplied, when the
    signal sits below ``MIN_SIGNAL_TO_NOISE`` BF16 quanta of the base output
    (the case itself is invalid and must be re-scaled, not tolerated).

    ``calibrated_max_abs`` (for example three times a known-good decomposed
    arm's observed error) may only tighten the default ``S * signal_fraction``
    gate, never loosen it.
    """
    reference = delta_reference.detach().to(torch.float64)
    signal_max_abs = float(reference.abs().max()) if reference.numel() else 0.0
    signal_l2 = float(torch.linalg.vector_norm(reference)) if reference.numel() else 0.0
    if signal_max_abs == 0.0:
        raise DegenerateSignalError(
            "reference delta is exactly zero; use require_bitwise_equal for "
            "zero-LoRA parity instead of a signal gate"
        )

    noise_floor = 0.0
    if base_reference is not None:
        noise_floor = bf16_noise_floor(base_reference)
        if signal_max_abs < MIN_SIGNAL_TO_NOISE * noise_floor:
            raise DegenerateSignalError(
                f"signal S={signal_max_abs:.3e} is below "
                f"{MIN_SIGNAL_TO_NOISE:g} BF16 quanta of the base output "
                f"({noise_floor:.3e}); re-scale the case inputs"
            )

    max_abs_gate = signal_max_abs * signal_fraction
    if calibrated_max_abs is not None and calibrated_max_abs < max_abs_gate:
        max_abs_gate = calibrated_max_abs

    dtype_name = str(destination_dtype).removeprefix("torch.")
    if dtype_name not in _REL_L2_GATES:
        raise ValueError(f"no relative-L2 gate defined for dtype {dtype_name}")

    return SignalGates(
        signal_max_abs=signal_max_abs,
        signal_l2=signal_l2,
        max_abs_gate=max_abs_gate,
        rel_l2_gate=_REL_L2_GATES[dtype_name],
        destination_dtype=dtype_name,
        noise_floor=noise_floor,
        calibration_note=calibration_note,
    )


def check_delta(
    observed_delta: torch.Tensor,
    delta_reference: torch.Tensor,
    gates: SignalGates,
    *,
    label: str = "",
) -> SignalCheckRecord:
    """Compare an observed delta against the reference under resolved gates."""
    if observed_delta.shape != delta_reference.shape:
        raise ValueError(
            f"shape mismatch: observed {tuple(observed_delta.shape)} vs "
            f"reference {tuple(delta_reference.shape)}"
        )
    error = observed_delta.detach().to(torch.float64) - delta_reference.detach().to(
        torch.float64
    )
    observed_max_abs = float(error.abs().max()) if error.numel() else 0.0
    observed_rel_l2 = (
        float(torch.linalg.vector_norm(error)) / gates.signal_l2
        if gates.signal_l2 > 0.0
        else 0.0
    )
    passed = (
        observed_max_abs <= gates.max_abs_gate
        and observed_rel_l2 <= gates.rel_l2_gate
    )
    return SignalCheckRecord(
        label=label,
        gates=gates,
        observed_max_abs=observed_max_abs,
        observed_rel_l2=observed_rel_l2,
        passed=passed,
    )


def require_signal_close(
    observed_output: torch.Tensor,
    reference_output: torch.Tensor,
    *,
    base_reference: torch.Tensor,
    destination_dtype: torch.dtype,
    label: str = "",
    signal_fraction: float = DEFAULT_SIGNAL_FRACTION,
    calibrated_max_abs: float | None = None,
) -> SignalCheckRecord:
    """Full-output comparison via the base-subtracted delta; raises on failure.

    ``observed_output`` and ``reference_output`` are complete outputs at the
    same boundary; ``base_reference`` is the matched base-only output that both
    are reduced by, so the gate sees only the LoRA contribution.
    """
    gates = resolve_signal_gates(
        reference_output.detach().to(torch.float64)
        - base_reference.detach().to(torch.float64),
        destination_dtype=destination_dtype,
        base_reference=base_reference,
        signal_fraction=signal_fraction,
        calibrated_max_abs=calibrated_max_abs,
    )
    record = check_delta(
        observed_output.detach().to(torch.float64)
        - base_reference.detach().to(torch.float64),
        reference_output.detach().to(torch.float64)
        - base_reference.detach().to(torch.float64),
        gates,
        label=label,
    )
    if not record.passed:
        raise AssertionError(
            f"signal gate failed [{label}]: max|err|={record.observed_max_abs:.3e} "
            f"(gate {gates.max_abs_gate:.3e}), rel_l2={record.observed_rel_l2:.3e} "
            f"(gate {gates.rel_l2_gate:.1e}), S={gates.signal_max_abs:.3e}"
        )
    return record


def require_delta_close(
    observed: torch.Tensor,
    reference: torch.Tensor,
    *,
    destination_dtype: torch.dtype,
    label: str = "",
    signal_fraction: float = DEFAULT_SIGNAL_FRACTION,
) -> SignalCheckRecord:
    """Direct signal-gated comparison of two delta-domain tensors.

    Use when the compared quantities are already LoRA-only (kernel outputs,
    materialized deltas) or when the operation's own output is the signal
    (single-op tests); full-output comparisons subtract a matched base via
    :func:`require_signal_close` instead.

    ``destination_dtype`` selects the relative-L2 gate and must reflect the
    least precise stage shared between observed and reference: a BF16-computed
    pipeline compares at the ``bfloat16`` gate even when it writes an FP32
    destination.
    """
    gates = resolve_signal_gates(
        reference.detach().to(torch.float64),
        destination_dtype=destination_dtype,
        signal_fraction=signal_fraction,
    )
    record = check_delta(
        observed.detach().to(torch.float64),
        reference.detach().to(torch.float64),
        gates,
        label=label,
    )
    if not record.passed:
        raise AssertionError(
            f"signal gate failed [{label}]: max|err|={record.observed_max_abs:.3e} "
            f"(gate {gates.max_abs_gate:.3e}), rel_l2={record.observed_rel_l2:.3e} "
            f"(gate {gates.rel_l2_gate:.1e}), S={gates.signal_max_abs:.3e}"
        )
    return record


def require_bitwise_equal(
    observed: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str = "",
) -> None:
    """Bitwise-zero class: parity/determinism checks allow no difference."""
    if observed.shape != expected.shape or observed.dtype != expected.dtype:
        raise AssertionError(
            f"bitwise check failed [{label}]: shape/dtype mismatch "
            f"{tuple(observed.shape)}/{observed.dtype} vs "
            f"{tuple(expected.shape)}/{expected.dtype}"
        )
    if not torch.equal(observed, expected):
        difference = (observed != expected).sum().item()
        raise AssertionError(
            f"bitwise check failed [{label}]: {difference} differing elements"
        )


def nan_poison_(tensor: torch.Tensor, mask: torch.Tensor | None = None) -> None:
    """Poison a tensor (or masked region) so any read of it is detectable."""
    if not tensor.dtype.is_floating_point:
        raise TypeError("NaN poison requires a floating-point tensor")
    if mask is None:
        tensor.fill_(float("nan"))
    else:
        tensor.masked_fill_(mask, float("nan"))


def require_finite(tensor: torch.Tensor, *, label: str = "") -> None:
    """Assert no poison leaked into an output."""
    bad = (~torch.isfinite(tensor)).sum().item()
    if bad:
        raise AssertionError(
            f"poison hygiene failed [{label}]: {bad} non-finite elements"
        )


__all__ = [
    "BF16_RELATIVE_QUANTUM",
    "DEFAULT_SIGNAL_FRACTION",
    "MIN_SIGNAL_TO_NOISE",
    "DegenerateSignalError",
    "SignalCheckRecord",
    "SignalGates",
    "bf16_noise_floor",
    "check_delta",
    "nan_poison_",
    "require_bitwise_equal",
    "require_finite",
    "require_signal_close",
    "resolve_signal_gates",
]
