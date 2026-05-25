"""Phase / backend identifiers, the canonical default for
``cuda_graph_settings``, and the ``--cuda-graph-settings`` JSON CLI parser.

Module-level imports are pure stdlib — no torch / sglang.srt deps — so
``ServerArgs`` can import everything here without pulling in backend
classes. ``check_cuda_graph_backend`` lazy-imports ``get_global_server_args``
inside the function body to preserve that invariant.
"""

import argparse
import copy
import json
from typing import Any, Dict, Optional


class Phase:
    """The two phases of model forward."""

    DECODE = "decode"
    PREFILL = "prefill"
    ALL = (DECODE, PREFILL)


class Backend:
    """CUDA graph capture backends a phase can use."""

    FULL = "full"
    BREAKABLE = "breakable"
    TC_PIECEWISE = "tc_piecewise"
    DISABLED = "disabled"
    ALL = (FULL, BREAKABLE, TC_PIECEWISE, DISABLED)


ALLOWED_BACKENDS_PER_PHASE = {
    Phase.DECODE: (
        Backend.FULL,
        Backend.BREAKABLE,
        Backend.TC_PIECEWISE,
        Backend.DISABLED,
    ),
    # ``full`` is rejected for prefill — full CUDA graph capture only
    # fits fixed-shape and prefill is variable-shape. Use ``breakable``
    # or ``tc_piecewise`` for prefill.
    Phase.PREFILL: (Backend.BREAKABLE, Backend.TC_PIECEWISE, Backend.DISABLED),
}

# Per-phase settings schema. Keys other than ``backend`` are runner-level
# (read by any backend in that phase); ``tc_compiler`` is the lone
# backend-specific knob (only meaningful when backend == tc_piecewise).
ALLOWED_KEYS_PER_PHASE = {
    Phase.DECODE: ("backend", "max_bs", "bs", "max_num_tokens", "num_tokens"),
    Phase.PREFILL: (
        "backend",
        "max_bs",
        "bs",
        "max_num_tokens",
        "num_tokens",
        "tc_compiler",
    ),
}

DEFAULT_CUDA_GRAPH_SETTINGS: Dict[str, Dict[str, Any]] = {
    Phase.DECODE: {
        "backend": Backend.FULL,
        "max_bs": None,
        "bs": None,
        "max_num_tokens": None,
        "num_tokens": None,
    },
    Phase.PREFILL: {
        "backend": Backend.TC_PIECEWISE,
        "max_bs": None,
        "bs": None,
        "max_num_tokens": None,
        "num_tokens": None,
        # Only meaningful when ``backend == tc_piecewise``; ignored otherwise.
        "tc_compiler": "eager",
    },
}


def default_cuda_graph_settings() -> Dict[str, Dict[str, Any]]:
    """Fresh deep copy of the canonical defaults."""
    return copy.deepcopy(DEFAULT_CUDA_GRAPH_SETTINGS)


def check_cuda_graph_backend(phase: str, backend: str) -> bool:
    """True if ``cuda_graph_settings[phase]['backend'] == backend`` on the
    global server args. Returns False if the global server args have not
    been initialized yet (e.g. unit tests, early startup)."""
    from sglang.srt.server_args import get_global_server_args

    try:
        server_args = get_global_server_args()
    except ValueError:
        return False
    if server_args.cuda_graph_settings is None:
        return False
    phase_settings = server_args.cuda_graph_settings.get(phase)
    if phase_settings is None:
        return False
    return phase_settings.get("backend") == backend


def parse_cuda_graph_settings_arg(raw: str) -> Dict[str, Dict[str, Any]]:
    """argparse type for ``--cuda-graph-settings``: parse JSON dict of
    phase → settings dict. Each phase's settings dict is itself validated
    against ``ALLOWED_KEYS_PER_PHASE``."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"--cuda-graph-settings must be JSON: {e}")
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(
            f"--cuda-graph-settings must be a JSON object, got {type(parsed).__name__}"
        )

    result: Dict[str, Dict[str, Any]] = {}
    for phase, phase_settings in parsed.items():
        phase = str(phase)
        if phase not in Phase.ALL:
            raise argparse.ArgumentTypeError(
                f"--cuda-graph-settings: unknown phase '{phase}', expected one of {Phase.ALL}"
            )
        if not isinstance(phase_settings, dict):
            raise argparse.ArgumentTypeError(
                f"--cuda-graph-settings['{phase}'] must be a JSON object, got "
                f"{type(phase_settings).__name__}"
            )
        allowed = ALLOWED_KEYS_PER_PHASE[phase]
        result[phase] = {}
        for key, value in phase_settings.items():
            if key not in allowed:
                raise argparse.ArgumentTypeError(
                    f"--cuda-graph-settings['{phase}']: unknown key '{key}', expected one of {allowed}"
                )
            result[phase][key] = value
    return result


def explicit_keys_in(
    settings: Optional[Dict[str, Dict[str, Any]]],
) -> set:
    """Return the set of ``(phase, key)`` tuples present in ``settings``.
    Used by ``ServerArgs`` to track keys the user explicitly set so the
    auto-disable cascade can skip them (the old ``--enforce-piecewise-cuda-graph``
    contract, generalized to every setting)."""
    out: set = set()
    if not settings:
        return out
    for phase, phase_settings in settings.items():
        if not isinstance(phase_settings, dict):
            continue
        for key in phase_settings.keys():
            out.add((phase, key))
    return out
