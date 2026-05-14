"""Phase / backend identifiers, the canonical default for
``cuda_graph_mode``, and the ``--cuda-graph-mode`` JSON CLI parser.

Module-level imports are pure stdlib — no torch / sglang.srt deps — so
``ServerArgs`` can import everything here without pulling in backend
classes. ``check_cuda_graph_backend`` lazy-imports ``get_global_server_args``
inside the function body to preserve that invariant.
"""

import argparse
import json
from typing import Dict


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

DEFAULT_CUDA_GRAPH_MODE = {
    Phase.DECODE: Backend.FULL,
    Phase.PREFILL: Backend.TC_PIECEWISE,
}


def check_cuda_graph_backend(phase: str, backend: str) -> bool:
    """True if ``cuda_graph_mode[phase] == backend`` on the global server args.

    Returns False if the global server args have not been initialized yet
    (e.g. unit tests, early startup).
    """
    from sglang.srt.server_args import get_global_server_args

    try:
        server_args = get_global_server_args()
    except ValueError:
        return False
    if server_args.cuda_graph_mode is None:
        return False
    return server_args.cuda_graph_mode[phase] == backend


def parse_cuda_graph_mode_arg(raw: str) -> Dict[str, str]:
    """argparse type for ``--cuda-graph-mode``: parse JSON dict of phase → backend."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"--cuda-graph-mode must be JSON: {e}")
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(
            f"--cuda-graph-mode must be a JSON object, got {type(parsed).__name__}"
        )
    return {str(k): str(v) for k, v in parsed.items()}
