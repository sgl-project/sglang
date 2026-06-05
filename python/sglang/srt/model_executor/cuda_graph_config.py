"""Phase / backend identifiers, the canonical default for
cuda_graph_config, and the --cuda-graph-config JSON CLI parser.

Module-level imports are pure stdlib — no torch / sglang.srt deps — so
ServerArgs can import everything here without pulling in backend
classes. check_cuda_graph_backend lazy-imports get_global_server_args
inside the function body to preserve that invariant.
"""

import argparse
import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
    # full is rejected for prefill — full CUDA graph capture only
    # fits fixed-shape and prefill is variable-shape. Use breakable
    # or tc_piecewise for prefill.
    Phase.PREFILL: (Backend.BREAKABLE, Backend.TC_PIECEWISE, Backend.DISABLED),
}

# Per-phase settings schema. Keys other than backend are runner-level
# (read by any backend in that phase); tc_compiler is the lone
# backend-specific knob (only meaningful when backend == tc_piecewise).
# For prefill, bs carries the captured shape size (token count for
# tc_piecewise, request count for breakable) — one shape knob per phase.
ALLOWED_KEYS_PER_PHASE = {
    Phase.DECODE: ("backend", "max_bs", "bs", "tc_compiler"),
    Phase.PREFILL: ("backend", "max_bs", "bs", "tc_compiler"),
}


@dataclass
class PhaseConfig:
    """Per-phase CUDA graph settings."""

    backend: str = Backend.DISABLED
    max_bs: Optional[int] = None
    bs: Optional[List[int]] = None
    # Only meaningful when backend == tc_piecewise; ignored otherwise.
    tc_compiler: str = "eager"


@dataclass
class CudaGraphConfig:
    """Top-level CUDA graph config: one PhaseConfig per phase."""

    decode: PhaseConfig = field(
        default_factory=lambda: PhaseConfig(backend=Backend.FULL)
    )
    prefill: PhaseConfig = field(
        default_factory=lambda: PhaseConfig(backend=Backend.TC_PIECEWISE)
    )

    def __getitem__(self, phase: str) -> PhaseConfig:
        """Phase-string lookup; kept for migration ergonomics."""
        if phase not in Phase.ALL:
            raise KeyError(phase)
        return getattr(self, phase)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        # Diff-only, not asdict: the parser locks every (phase, key) it sees,
        # so emitting defaults would lock fields the caller never set.
        baseline = default_cuda_graph_config()
        return {
            Phase.DECODE: _diff_phase(self.decode, baseline.decode),
            Phase.PREFILL: _diff_phase(self.prefill, baseline.prefill),
        }

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Dict[str, Any]]]) -> "CudaGraphConfig":
        """Build from a (partial) dict of overrides, defaults fill the rest.
        Unknown phases / keys are silently dropped — the JSON-input
        validator (parse_cuda_graph_config_arg) rejects them upstream."""
        cfg = cls()
        if not raw:
            return cfg
        for phase, phase_settings in raw.items():
            if phase not in Phase.ALL or not isinstance(phase_settings, dict):
                continue
            phase_cfg = getattr(cfg, phase)
            allowed = ALLOWED_KEYS_PER_PHASE[phase]
            for key, value in phase_settings.items():
                if key in allowed:
                    setattr(phase_cfg, key, value)
        return cfg


def default_cuda_graph_config() -> CudaGraphConfig:
    """Fresh CudaGraphConfig populated with canonical defaults."""
    return CudaGraphConfig()


def _diff_phase(actual: PhaseConfig, baseline: PhaseConfig) -> Dict[str, Any]:
    """Return only fields whose value differs from the per-phase default."""
    return {
        f.name: getattr(actual, f.name)
        for f in dataclasses.fields(actual)
        if getattr(actual, f.name) != getattr(baseline, f.name)
    }


def check_cuda_graph_backend(phase: str, backend: str) -> bool:
    """True if cuda_graph_config[phase].backend == backend on the
    global server args. Returns False if the global server args have not
    been initialized yet (e.g. unit tests, early startup)."""
    from sglang.srt.server_args import get_global_server_args

    try:
        server_args = get_global_server_args()
    except ValueError:
        return False
    cfg = server_args.cuda_graph_config
    if cfg is None or phase not in Phase.ALL:
        return False
    return getattr(cfg, phase).backend == backend


def cuda_graph_fully_disabled() -> bool:
    """True iff cuda_graph_config has Backend.DISABLED on every phase.

    Use at sites that ask the legacy server_args.disable_cuda_graph
    question ("no CG anywhere globally") — e.g., preallocating buffers
    that any captured graph would otherwise reuse, or one-shot init
    that's a no-op when CG is completely off.
    """
    return check_cuda_graph_backend(
        Phase.DECODE, Backend.DISABLED
    ) and check_cuda_graph_backend(Phase.PREFILL, Backend.DISABLED)


def parse_cuda_graph_config_arg(raw: str) -> Dict[str, Dict[str, Any]]:
    """argparse type for --cuda-graph-config: parse JSON dict of
    phase → settings dict. Each phase's settings dict is itself validated
    against ALLOWED_KEYS_PER_PHASE. Returns a plain dict — the
    precedence pipeline in ServerArgs converts to CudaGraphConfig
    after merging."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"--cuda-graph-config must be JSON: {e}")
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(
            f"--cuda-graph-config must be a JSON object, got {type(parsed).__name__}"
        )

    result: Dict[str, Dict[str, Any]] = {}
    for phase, phase_settings in parsed.items():
        phase = str(phase)
        if phase not in Phase.ALL:
            raise argparse.ArgumentTypeError(
                f"--cuda-graph-config: unknown phase '{phase}', expected one of {Phase.ALL}"
            )
        if not isinstance(phase_settings, dict):
            raise argparse.ArgumentTypeError(
                f"--cuda-graph-config['{phase}'] must be a JSON object, got "
                f"{type(phase_settings).__name__}"
            )
        allowed = ALLOWED_KEYS_PER_PHASE[phase]
        result[phase] = {}
        for key, value in phase_settings.items():
            if key not in allowed:
                raise argparse.ArgumentTypeError(
                    f"--cuda-graph-config['{phase}']: unknown key '{key}', expected one of {allowed}"
                )
            result[phase][key] = value
    return result


def explicit_keys_in(
    settings: Optional[Dict[str, Dict[str, Any]]],
) -> set:
    """Return the set of (phase, key) tuples present in settings
    (the raw dict form, as it arrives from CLI/SDK). Used by ServerArgs
    to track keys the user explicitly set so the auto-disable cascade can
    skip them."""
    out: set = set()
    if not settings:
        return out
    for phase, phase_settings in settings.items():
        if not isinstance(phase_settings, dict):
            continue
        for key in phase_settings.keys():
            out.add((phase, key))
    return out
