"""Worker lifecycle, failure propagation and instance isolation."""

import multiprocessing as mp
import os
import sys
import types
from dataclasses import dataclass

import pytest

from sglang.multimodal_gen.apps import comfyui_worker_patches as patches


@pytest.fixture(autouse=True)
def registered_test_profile(monkeypatch):
    monkeypatch.setitem(patches.PROFILES, "test_v1", "test_provider")


def test_registered_names_only():
    assert patches.canonical_profiles(None) == ()
    assert patches.canonical_profiles(["test_v1"] * 2) == ("test_v1",)
    for names in (["arbitrary.module:run"], "test_v1", [None]):
        with pytest.raises(ValueError, match="Unknown"):
            patches.canonical_profiles(names)


def test_composition_rejected(monkeypatch):
    monkeypatch.setitem(patches.PROFILES, "other", "other")
    with pytest.raises(ValueError, match="one worker patch"):
        patches.canonical_profiles(["test_v1", "other"])


def test_default_launch_and_restore_on_failure(monkeypatch):
    original = lambda spec: None
    module = types.ModuleType("sglang.multimodal_gen.runtime.launch_server")
    module.bootstrap_scheduler_process = original
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with patches.scheduler_profiles([]):
        assert module.bootstrap_scheduler_process is original
    with (
        pytest.raises(RuntimeError, match="launch failed"),
        patches.scheduler_profiles(["test_v1"]),
    ):
        assert module.bootstrap_scheduler_process is not original
        raise RuntimeError("launch failed")
    assert module.bootstrap_scheduler_process is original


@dataclass
class _Args:
    def materialize(self):
        return "args-ready"


@dataclass
class _Spec:
    server_args: object


def _spawn_probe(connection, fail):
    # A real spawn boundary with a tiny worker instead of allocating GPU weights.
    worker_module = types.ModuleType(
        "sglang.multimodal_gen.runtime.managers.gpu_worker"
    )

    class Worker:
        def __init__(self):
            self.pipeline = types.SimpleNamespace(loaded=True)

    worker_module.GPUWorker = Worker
    sys.modules[worker_module.__name__] = worker_module
    provider = types.ModuleType(patches.__name__ + ".test_provider")

    def install(pipeline):
        assert pipeline.loaded
        if fail:
            raise RuntimeError("installer failed")
        pipeline.patched = True

    provider.install = install
    sys.modules[provider.__name__] = provider
    patches.PROFILES["test"] = "test_provider"

    def stock_bootstrap(spec):
        assert spec.server_args.materialize() == "args-ready"
        return Worker().pipeline.patched

    try:
        result = patches._bootstrap(stock_bootstrap, "test", _Spec(_Args()))
    except RuntimeError as error:
        result = str(error)
    connection.send((os.getpid(), result))
    connection.close()


@pytest.mark.parametrize("fail", [False, True])
def test_real_spawn_installs_after_load_or_aborts(fail):
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_spawn_probe, args=(child, fail))
    process.start()
    child.close()
    try:
        assert parent.poll(60), "spawn did not report"
        pid, result = parent.recv()
        process.join(10)
        assert process.exitcode == 0 and pid != os.getpid()
        assert result == ("installer failed" if fail else True)
        assert "test" not in patches.PROFILES
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()
