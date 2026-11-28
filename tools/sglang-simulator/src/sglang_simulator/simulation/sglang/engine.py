"""SGLang engine entry point with simulator-aware worker processes."""

from sglang_simulator.simulation.sglang.hook_bootstrap import (
    install_simulator_hooks,
    run_simulator_detokenizer_process,
    run_simulator_scheduler_process,
)

install_simulator_hooks()

# Install hooks before importing Engine so its worker entry points are patched.
from sglang.srt.entrypoints.engine import Engine  # noqa: E402


class SGLangSimulationEngine(Engine):
    """Engine whose spawned workers install SGLang Simulator hooks explicitly."""

    run_scheduler_process_func = staticmethod(run_simulator_scheduler_process)
    run_detokenizer_process_func = staticmethod(run_simulator_detokenizer_process)
