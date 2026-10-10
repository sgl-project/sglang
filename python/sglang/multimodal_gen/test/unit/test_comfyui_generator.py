# SPDX-License-Identifier: Apache-2.0
"""Process-wide SGLD worker ownership for ComfyUI loaders."""

from types import SimpleNamespace

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core.generator import (
    SGLDiffusionGenerator,
)


def test_shared_is_process_singleton() -> None:
    SGLDiffusionGenerator.reset_shared()
    assert SGLDiffusionGenerator.shared() is SGLDiffusionGenerator.shared()
    SGLDiffusionGenerator.reset_shared()


def test_reuse_requires_live_worker() -> None:
    runtime = SGLDiffusionGenerator()
    options = {"model_path": "z.safetensors"}
    runtime.last_options = options
    runtime.generator = object()
    runtime._patcher = object()
    runtime.executor = object()
    runtime._is_live = lambda: False
    assert runtime._can_reuse(options) is False
    runtime._is_live = lambda: True
    assert runtime._can_reuse(options) is True
    assert runtime._can_reuse({"model_path": "h3.safetensors"}) is False


def test_ensure_rebuilds_when_another_model_owns_the_worker() -> None:
    runtime = SGLDiffusionGenerator()
    stale = object()
    fresh = object()
    loads = []
    sampler_runs = []
    executor = SimpleNamespace(
        generator=stale,
        _sgld_reload={
            "model_path": "z.safetensors",
            "model_options": {},
            "sgld_options": {},
        },
        _lora_input=None,
        begin_sampler_run=lambda: sampler_runs.append(True),
    )

    def fake_load(**kwargs):
        loads.append(kwargs)
        runtime.generator = fresh
        return "patcher"

    runtime.load_model = fake_load
    runtime.ensure_executor(executor)
    assert loads == [executor._sgld_reload]
    assert executor.generator is fresh
    assert sampler_runs == [True]


def test_ensure_is_noop_when_executor_still_owns_live_worker() -> None:
    runtime = SGLDiffusionGenerator()
    gen = object()
    runtime.generator = gen
    runtime._is_live = lambda: True
    executor = SimpleNamespace(generator=gen, _sgld_reload={"model_path": "z"})
    runtime.load_model = lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("should not reload")
    )
    runtime.ensure_executor(executor)
    assert executor.generator is gen


def test_kill_generator_only_touches_owned_workers() -> None:
    runtime = SGLDiffusionGenerator()
    owned = SimpleNamespace(alive=True, terminated=False, killed=False, pid=9)

    def terminate():
        owned.terminated = True
        owned.alive = False

    owned.is_alive = lambda: owned.alive
    owned.terminate = terminate
    owned.join = lambda timeout=None: None
    owned.kill = lambda: setattr(owned, "killed", True)
    runtime.generator = SimpleNamespace(local_scheduler_process=[owned])
    runtime.kill_generator()
    assert owned.terminated is True
    assert owned.killed is False


def _load_with_blocked_import(relpath, module_name, blocked):
    """Import a private copy of a plugin module while ``blocked`` fails to import."""
    import importlib.util
    import sys
    from pathlib import Path
    from unittest import mock

    import sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion as plugin

    path = Path(plugin.__file__).parent / relpath
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {blocked: None}):  # None makes import raise
        spec.loader.exec_module(module)
    return module


def test_executor_reports_real_runtime_import_error(capsys) -> None:
    import pytest
    import torch

    blocked = "sglang.multimodal_gen.runtime.entrypoints.utils"
    base = _load_with_blocked_import("executors/base.py", "_sgld_base_blocked", blocked)
    printed = capsys.readouterr().out
    assert blocked in printed
    assert "is not installed" not in printed

    class _Executor(base.SGLDiffusionExecutor):
        def __init__(self):
            torch.nn.Module.__init__(self)

    with pytest.raises(RuntimeError, match="failed to import") as err:
        _Executor()._execute_packed(None, None, None)
    assert isinstance(err.value.__cause__, ImportError)
    assert blocked in str(err.value.__cause__)


def test_generator_reports_real_runtime_import_error(caplog) -> None:
    import pytest

    blocked = "sglang.multimodal_gen"
    module = _load_with_blocked_import(
        "core/generator.py",
        "sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core._generator_blocked",
        blocked,
    )
    assert any(blocked in record.getMessage() for record in caplog.records)
    with pytest.raises(RuntimeError, match="failed to import") as err:
        module.SGLDiffusionGenerator().init_generator("flux", "FluxPipeline", {})
    assert isinstance(err.value.__cause__, ImportError)
