# SPDX-License-Identifier: Apache-2.0
"""Pure-stdlib calibration helpers for the BLASST skip-softmax backend.

Extracted so every threshold-resolution branch is CPU unit-testable without
torch / sglang server-args / modelopt installed. The companion file
``modelopt_skip_softmax_attn.py`` re-exports these for the impl and for tests.

Schema reference: ``modelopt/torch/sparsity/attention_sparsity/plugins/vllm.py``
(see ``_resolve_skip_softmax_calibration`` and
``_target_sparse_ratio_for_phase``).
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Mapping

# Default per-phase target sparsity (matches modelopt vllm plugin default).
DEFAULT_TARGET_SPARSE_RATIO: dict[str, float] = {"prefill": 0.5, "decode": 0.5}

# Internal canonical entry shape:
#   {"phases": {"prefill": (a, b), "decode": (a, b)},
#    "target_sparse_ratio": {"prefill": float, "decode": float}}
CalibEntry = dict[str, Any]


def normalize_target_sparse_ratio(value: Any) -> dict[str, float]:
    """Normalise ``target_sparse_ratio`` into a {phase: float} dict.

    Accepts:
        - scalar -> same value for both phases
        - {"prefill": float, "decode": float}
        - anything else -> defaults to 0.5/0.5 (modelopt vllm convention)
    """
    # bool is a subclass of int -- exclude it explicitly to avoid
    # ``target_sparse_ratio = True`` silently being treated as 1.0.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        r = float(value)
        return {"prefill": r, "decode": r}
    if isinstance(value, Mapping):
        return {
            "prefill": float(
                value.get("prefill", DEFAULT_TARGET_SPARSE_RATIO["prefill"])
            ),
            "decode": float(value.get("decode", DEFAULT_TARGET_SPARSE_RATIO["decode"])),
        }
    return dict(DEFAULT_TARGET_SPARSE_RATIO)


def entry_from_flat_ab(a: Any, b: Any) -> CalibEntry:
    """Lift legacy {"a", "b"} pair into the canonical per-phase entry."""
    a_f, b_f = float(a), float(b)
    return {
        "phases": {"prefill": (a_f, b_f), "decode": (a_f, b_f)},
        "target_sparse_ratio": dict(DEFAULT_TARGET_SPARSE_RATIO),
    }


def entry_from_modelopt_canonical(raw: Mapping[str, Any]) -> CalibEntry | None:
    """Lift a single component dict in the modelopt schema into the canonical
    entry. Returns ``None`` if the dict has no usable phase params.

    Example input::

        {
            "threshold_scale_factor": {
                "prefill": {"a": 12.34, "b": 5.67},
                "decode":  {"a": 12.34, "b": 5.67},
            },
            "target_sparse_ratio": {"prefill": 0.5, "decode": 0.5},
        }
    """
    tsf = raw.get("threshold_scale_factor")
    if not isinstance(tsf, Mapping):
        return None
    phases: dict[str, tuple[float, float]] = {}
    for phase in ("prefill", "decode"):
        p = tsf.get(phase)
        if isinstance(p, Mapping) and "a" in p and "b" in p:
            try:
                phases[phase] = (float(p["a"]), float(p["b"]))
            except (TypeError, ValueError):
                continue
    if not phases:
        return None
    # Fill missing phase with the other one's params (diffusion only does prefill).
    if "prefill" not in phases and "decode" in phases:
        phases["prefill"] = phases["decode"]
    if "decode" not in phases and "prefill" in phases:
        phases["decode"] = phases["prefill"]
    return {
        "phases": phases,
        "target_sparse_ratio": normalize_target_sparse_ratio(
            raw.get("target_sparse_ratio")
        ),
    }


def normalize_calibration(raw: Any) -> dict[str, CalibEntry]:
    """Parse a raw JSON-loaded structure into ``{component: CalibEntry}``.

    Returns ``{}`` if nothing usable is found (caller falls back to fixed
    threshold). Never raises on malformed input - we want to log and
    degrade gracefully so a bad calibration file doesn't crash inference.
    """
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        warnings.warn(
            f"[skip_softmax] calibration must be a JSON object, got {type(raw).__name__}; "
            "ignoring.",
            stacklevel=2,
        )
        return {}

    # Case A1: flat {"a", "b"} - one model.
    if "a" in raw and "b" in raw:
        try:
            return {"_default": entry_from_flat_ab(raw["a"], raw["b"])}
        except (TypeError, ValueError) as e:
            warnings.warn(f"[skip_softmax] bad flat (a, b): {e!r}", stacklevel=2)
            return {}

    out: dict[str, CalibEntry] = {}
    for component, payload in raw.items():
        if not isinstance(payload, Mapping):
            warnings.warn(
                f"[skip_softmax] calibration entry {component!r} is not a dict; skipping.",
                stacklevel=2,
            )
            continue
        # Case B: modelopt canonical (per-component).
        entry = entry_from_modelopt_canonical(payload)
        if entry is not None:
            out[component] = entry
            continue
        # Case A2: flat per-component {"a", "b"}.
        if "a" in payload and "b" in payload:
            try:
                out[component] = entry_from_flat_ab(payload["a"], payload["b"])
            except (TypeError, ValueError) as e:
                warnings.warn(
                    f"[skip_softmax] calibration entry {component!r}: bad (a, b) -> {e!r}",
                    stacklevel=2,
                )
            continue
        warnings.warn(
            f"[skip_softmax] calibration entry {component!r}: no recognised "
            "schema (need either {a,b} or {threshold_scale_factor:{prefill:{a,b}}}).",
            stacklevel=2,
        )
    return out


def pick_calibration_entry(
    calib: Mapping[str, CalibEntry],
    component_key: str,
) -> CalibEntry | None:
    """Look up the component's entry, falling back to ``"_default"``."""
    if component_key in calib:
        return calib[component_key]
    return calib.get("_default")


def compute_threshold(
    *,
    a: float,
    b: float,
    target_sparsity: float,
    seq_len_k: int,
) -> float | None:
    """Apply ``threshold = a * exp(b * target_sparsity) / seq_len_k``.

    Returns the threshold if it lies in the kernel's valid range ``(0, 1)``,
    else ``None`` (caller falls back to dense). Mirrors modelopt vllm plugin's
    ``_resolve_skip_softmax_calibration`` guard.
    """
    if seq_len_k <= 0 or a <= 0.0 or target_sparsity <= 0.0:
        return None
    try:
        scale_factor = a * math.exp(b * target_sparsity)
    except OverflowError:
        return None
    threshold = scale_factor / seq_len_k
    if not (0.0 < threshold < 1.0):
        warnings.warn(
            "[skip_softmax] calibrated threshold out of valid lambda range "
            f"(0, 1): phase=prefill seq_len_k={seq_len_k} a={a:.6g} b={b:.6g} "
            f"target_sparsity={target_sparsity:.3f} threshold={threshold:.6g}; "
            "falling back to dense for this launch.",
            stacklevel=2,
        )
        return None
    return threshold


def load_calibration_file(path: str) -> dict[str, CalibEntry]:
    """Read + parse a calibration JSON file from disk."""
    return normalize_calibration(json.loads(Path(path).read_text()))


def component_key_from_prefix(prefix: str) -> str:
    """Wan2.2 14B uses ``transformer`` + ``transformer_2``; take the leftmost
    component name in the impl's ``prefix`` (e.g.
    ``"transformer.blocks.0.attn1.impl"`` -> ``"transformer"``). Empty prefix
    maps to ``"_default"``.
    """
    head = prefix.split(".")[0] if prefix else ""
    return head or "_default"


def resolve_target_sparsity(
    *,
    override: float,
    calib_entry: CalibEntry | None,
    phase: str,
) -> float:
    """Pick effective target_sparsity given the override and calibration entry.

    Override > 0 wins (came from CLI / env). Else use the calibration's
    own ``target_sparse_ratio[phase]`` (set at calibration time). Else
    DEFAULT_TARGET_SPARSE_RATIO[phase].
    """
    if override > 0.0:
        return override
    if calib_entry is not None:
        return float(
            calib_entry["target_sparse_ratio"].get(
                phase, DEFAULT_TARGET_SPARSE_RATIO[phase]
            )
        )
    return DEFAULT_TARGET_SPARSE_RATIO[phase]


__all__ = [
    "DEFAULT_TARGET_SPARSE_RATIO",
    "CalibEntry",
    "normalize_target_sparse_ratio",
    "entry_from_flat_ab",
    "entry_from_modelopt_canonical",
    "normalize_calibration",
    "pick_calibration_entry",
    "compute_threshold",
    "load_calibration_file",
    "component_key_from_prefix",
    "resolve_target_sparsity",
]
