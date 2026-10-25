"""Controller spawn aliases retain native execution behind the child bootstrap."""

from types import SimpleNamespace

import pytest
from sglang_simulator.compat import SGLangCompatibilityError
from sglang_simulator.simulation.sglang import process_entry


def bindings(monkeypatch, *, current=None, native=None):
    def run_native(*args, **kwargs):
        raise AssertionError("installing a spawn target must not start a controller")

    def wrapper(*args, **kwargs):
        raise AssertionError("installing a spawn target must not start a controller")

    native = run_native if native is None else native
    engine = SimpleNamespace(
        run_data_parallel_controller_process=(native if current is None else current),
        run_scheduler_process_func=object(),
        run_detokenizer_process_func=object(),
    )
    controller = SimpleNamespace(run_data_parallel_controller_process=native)
    bootstrap = SimpleNamespace(run_simulator_data_parallel_controller_process=wrapper)
    modules = {
        "sglang.srt.entrypoints.engine": engine,
        "sglang.srt.managers.data_parallel_controller": controller,
        "sglang_simulator.simulation.sglang.hook_bootstrap": bootstrap,
    }
    monkeypatch.setattr(process_entry, "import_module", modules.__getitem__)
    return engine, controller, wrapper


def test_controller_target_is_idempotent_and_preserves_native_callee(monkeypatch):
    engine, controller, wrapper = bindings(monkeypatch)
    native = controller.run_data_parallel_controller_process
    sched, detok = (
        engine.run_scheduler_process_func,
        engine.run_detokenizer_process_func,
    )
    process_entry.install_controller_process_entry()
    process_entry.install_controller_process_entry()
    assert engine.run_data_parallel_controller_process is wrapper
    assert controller.run_data_parallel_controller_process is native
    assert engine.run_scheduler_process_func is sched
    assert engine.run_detokenizer_process_func is detok


@pytest.mark.parametrize("current,native", [(object(), None), (None, object())])
def test_unknown_native_controller_entry_is_not_overwritten(
    monkeypatch, current, native
):
    engine, _, _ = bindings(monkeypatch, current=current, native=native)
    before = engine.run_data_parallel_controller_process
    with pytest.raises(SGLangCompatibilityError, match="controller spawn alias"):
        process_entry.install_controller_process_entry()
    assert engine.run_data_parallel_controller_process is before
