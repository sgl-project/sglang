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
    executor = SimpleNamespace(
        generator=stale,
        _sgld_reload={
            "model_path": "z.safetensors",
            "model_options": {},
            "sgld_options": {},
        },
        _lora_input=None,
    )

    def fake_load(**kwargs):
        loads.append(kwargs)
        runtime.generator = fresh
        return "patcher"

    runtime.load_model = fake_load
    runtime.ensure_executor(executor)
    assert loads == [executor._sgld_reload]
    assert executor.generator is fresh


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


def test_worker_start_pins_cfg_parallel_off(monkeypatch) -> None:
    """ComfyUI owns CFG, so the worker must not auto-enable CFG parallel."""
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core import generator

    seen = {}

    def from_pretrained(**kwargs):
        seen.clear()
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(
        generator, "DiffGenerator", SimpleNamespace(from_pretrained=from_pretrained)
    )
    runtime = SGLDiffusionGenerator()
    runtime.init_generator("m.safetensors", "P", {"num_gpus": 2})
    assert seen["cfg_parallel_degree"] == 1
    runtime.generator = None
    runtime.init_generator(
        "m.safetensors", "P", {"num_gpus": 2, "enable_cfg_parallel": True}
    )
    assert "cfg_parallel_degree" not in seen


def test_should_suppress_logs_accepts_batched_timestep() -> None:
    """ComfyUI batches cond/uncond rows into one call with a [B] timestep."""
    import torch

    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.base import (
        SGLDiffusionExecutor,
    )

    assert SGLDiffusionExecutor.should_suppress_logs(torch.tensor([0.5, 0.5]))
    assert not SGLDiffusionExecutor.should_suppress_logs(torch.tensor([2.0, 2.0]))


def test_extra_server_args_merge_into_sgld_options() -> None:
    """Lets two workers run side by side, e.g. on different master ports."""
    import sys
    import types
    from unittest import mock

    import pytest

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.folder_names_and_paths = {}
    comfy_api_input = types.ModuleType("comfy_api.input")
    comfy_api_input.VideoInput = type("VideoInput", (), {})
    comfy_api = types.ModuleType("comfy_api")
    comfy_api.input = comfy_api_input
    stubs = {
        "folder_paths": folder_paths,
        "comfy_api": comfy_api,
        "comfy_api.input": comfy_api_input,
    }
    with mock.patch.dict(sys.modules, stubs):
        from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion import nodes
    # Do not leak the stub-bound module: later `from ... import nodes` would
    # find it on the package even though patch.dict dropped it from sys.modules.
    package = sys.modules.get("sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion")
    if package is not None and getattr(package, "nodes", None) is nodes:
        delattr(package, "nodes")

    (options,) = nodes.SGLDOptions().create_options(
        extra_server_args='{"master_port": 30105, "scheduler_port": 5655}'
    )
    assert options["master_port"] == 30105 and options["scheduler_port"] == 5655
    with pytest.raises(ValueError, match="JSON object"):
        nodes.SGLDOptions().create_options(extra_server_args="[1]")
