"""Bind native controller spawning to the simulator's early child bootstrap."""

from importlib import import_module

from sglang_simulator.compat import SGLangCompatibilityError


def install_controller_process_entry() -> None:
    """Patch only the opt-in parent's Engine alias, retaining the native callee.

    Native Engine does not expose a controller entry-point argument like its
    scheduler/detokenizer arguments. Its module-level alias is the actual spawn
    target. The wrapper lives in hook_bootstrap so unpickling that target also
    prepares compatibility before native launch arguments are unpickled.
    """
    engine = import_module("sglang.srt.entrypoints.engine")
    bootstrap = import_module("sglang_simulator.simulation.sglang.hook_bootstrap")
    wrapper = bootstrap.run_simulator_data_parallel_controller_process
    current = getattr(engine, "run_data_parallel_controller_process", None)
    if current is wrapper:
        return
    controller = import_module("sglang.srt.managers.data_parallel_controller")
    native = getattr(controller, "run_data_parallel_controller_process", None)
    if not callable(native) or current is not native:
        raise SGLangCompatibilityError(
            "SGLang Simulator requires Engine's native data-parallel controller "
            "spawn alias; an absent or replaced entry point cannot be adapted"
        )
    engine.run_data_parallel_controller_process = wrapper
