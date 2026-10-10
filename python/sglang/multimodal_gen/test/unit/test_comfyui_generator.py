# SPDX-License-Identifier: Apache-2.0
"""Process-wide SGLD worker ownership for ComfyUI loaders."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

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


def _runtime_that_must_reject() -> SGLDiffusionGenerator:
    """A loader whose model build fails loudly, so a check that should reject
    first is caught if it lets the options through."""
    runtime = SGLDiffusionGenerator()
    runtime.get_comfyui_model = Mock(
        side_effect=AssertionError("must reject before building the model")
    )
    return runtime


def test_non_default_weight_dtype_is_rejected_before_worker_load() -> None:
    """weight_dtype only reached the ComfyUI architecture companion, so fp8
    produced output bit-identical to the default load."""
    with pytest.raises(ValueError, match="weight_dtype must be 'default'"):
        _runtime_that_must_reject().load_model(
            model_path="h3.safetensors",
            model_options={"dtype": torch.float8_e4m3fn},
            sgld_options={},
        )


@pytest.mark.parametrize(
    "options,error",
    [
        ({"attention_backend": "sage_attn_3"}, "would run torch_sdpa instead"),
        ({"component_attention_backends": "transformer=sol_attn"}, "not installed"),
        ({"attention_backend": "not_a_backend"}, "not an SGLang attention backend"),
        ({"attention_backend": "sage_attn"}, None),
    ],
)
def test_attention_backends_are_checked_before_worker_load(
    monkeypatch, options, error
) -> None:
    """SGLang serves a missing sage_attn / sage_attn_3 kernel as FA / SDPA, so an
    explicit choice ran another backend; a missing sparse kernel only failed
    after the full model load."""
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core import preflight
    from sglang.multimodal_gen.runtime.platforms.interface import (
        AttentionBackendEnum as Backend,
    )

    def resolved(backend):
        if backend is Backend.SOL_ATTN:
            raise ImportError("Sol-Attn backend is not installed")
        return {Backend.SAGE_ATTN_3: Backend.TORCH_SDPA}.get(backend, backend)

    monkeypatch.setattr(preflight, "_resolved_backend", resolved)
    with pytest.raises(
        ValueError if error else AssertionError, match=error or "must reject before"
    ):
        _runtime_that_must_reject().load_model(
            model_path="h3.safetensors", sgld_options=options
        )


@pytest.mark.parametrize(
    "options,error",
    [
        ({"num_gpus": 2, "tp_size": 1, "sp_degree": 1}, "must equal"),
        ({"num_gpus": 2, "tp_size": 2, "sp_degree": 2}, "must equal"),
        ({"num_gpus": 2, "dp_size": 2}, "dp_size > 1"),
        ({"num_gpus": 2, "tp_size": 2, "sp_degree": 1}, None),
        ({"num_gpus": 2, "tp_size": 1, "sp_degree": None}, None),
    ],
)
def test_parallel_layout_is_checked_before_worker_load(options, error) -> None:
    """num_gpus=2 with tp=sp=1 left rank 1 without a process group, hanging
    worker startup forever; dp_size=2 sent sampler steps to a replica without
    the run's cached conditioning."""
    with pytest.raises(
        ValueError if error else AssertionError, match=error or "must reject before"
    ):
        _runtime_that_must_reject().load_model(
            model_path="h3.safetensors", sgld_options=options
        )


def test_worker_exit_during_load_names_where_the_reason_is(monkeypatch) -> None:
    """A worker that raised while loading surfaced as an empty EOFError."""
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core import generator

    def exit_during_load(**kwargs):
        raise EOFError

    monkeypatch.setattr(generator.DiffGenerator, "from_pretrained", exit_during_load)
    with pytest.raises(RuntimeError, match="traceback is in the ComfyUI console"):
        SGLDiffusionGenerator().init_generator("h3.safetensors", "MiniMaxH3Pipeline")


def test_sampler_selects_lora_from_each_patcher_and_clears_previous_adapter():
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.base import (
        SGLDiffusionExecutor,
    )

    events = []
    payload = {"lora_path": "four-step.safetensors", "strength": 1.0}
    runtime = SimpleNamespace(
        _ensure_runtime=None,
        _lora_input=payload.copy(),
        generator=SimpleNamespace(unmerge_lora_weights=lambda: events.append("clear")),
        begin_sampler_run=lambda: events.append("begin"),
        end_sampler_run=lambda: events.append("end"),
    )

    def set_lora(**desired):
        events.append(("load", desired))
        runtime._lora_input = desired.copy()

    runtime.set_lora = set_lora
    base = SimpleNamespace(model_patcher=SimpleNamespace(model_options={}))
    adapted = SimpleNamespace(
        model_patcher=SimpleNamespace(model_options={"sgld_lora_input": payload})
    )
    sample = lambda *args, **kwargs: "sampled"
    assert (
        SGLDiffusionExecutor.sampler_sample_wrapper(runtime, sample, base) == "sampled"
    )
    assert runtime._lora_input is None
    assert events == ["clear", "begin", "end"]
    events.clear()
    SGLDiffusionExecutor.sampler_sample_wrapper(runtime, sample, adapted)
    SGLDiffusionExecutor.sampler_sample_wrapper(runtime, sample, adapted)
    assert events.count(("load", payload)) == 1
    assert "clear" not in events


def test_cached_base_model_resets_request_accelerations_after_spectrum_run():
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.base import (
        SGLDiffusionExecutor,
    )

    runtime = SimpleNamespace(
        _ensure_runtime=None,
        _lora_input=None,
        begin_sampler_run=lambda: None,
        end_sampler_run=lambda: None,
    )
    accelerated = SimpleNamespace(
        model_patcher=SimpleNamespace(
            model_options={
                "sgld_request_flags": {
                    "enable_cache_dit": True,
                    "cache_dit_params": {"residual_diff_threshold": 0.12},
                    "request_options": {"enable_spectrum": True},
                }
            }
        )
    )
    base = SimpleNamespace(model_patcher=SimpleNamespace(model_options={}))
    SGLDiffusionExecutor.sampler_sample_wrapper(
        runtime, lambda *args: None, accelerated
    )
    assert runtime.request_options == {"enable_spectrum": True}
    assert runtime.enable_cache_dit is True
    SGLDiffusionExecutor.sampler_sample_wrapper(runtime, lambda *args: None, base)
    assert runtime.request_options == {}
    assert runtime.enable_cache_dit is None
    assert runtime.cache_dit_params is None


def test_failed_lora_request_does_not_claim_adapter_is_active():
    import pytest

    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.base import (
        SGLDiffusionExecutor,
    )

    def reject(**kwargs):
        raise RuntimeError("Dynamic LoRA supports only one adapter")

    runtime = SimpleNamespace(
        _lora_input=None, generator=SimpleNamespace(set_lora=reject)
    )
    with pytest.raises(RuntimeError, match="one adapter"):
        SGLDiffusionExecutor.set_lora(
            runtime,
            lora_nickname=["a", "b"],
            lora_path=["a.safetensors", "b.safetensors"],
            strength=[1, 0.25],
            target=["all", "all"],
        )
    assert runtime._lora_input is None


def test_spawned_workers_do_not_reexecute_launcher_main() -> None:
    """Workers must not re-run ComfyUI's main.py; its imports break under spawn."""
    import multiprocessing.spawn
    import sys

    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core.generator import (
        _spawn_without_launcher_main,
    )

    main_dict = vars(sys.modules["__main__"])
    before = {k: main_dict.get(k, "<absent>") for k in ("__file__", "__spec__")}
    main_dict["__file__"] = "/comfy/main.py"
    try:
        with _spawn_without_launcher_main():
            data = multiprocessing.spawn.get_preparation_data("worker")
            assert "init_main_from_path" not in data
            assert "init_main_from_name" not in data
        assert main_dict["__file__"] == "/comfy/main.py"
    finally:
        for key, value in before.items():
            if value == "<absent>":
                main_dict.pop(key, None)
            else:
                main_dict[key] = value


def test_fasth3_single_file_as_base_h3_is_rejected_before_worker_load(
    tmp_path,
) -> None:
    """A FastH3 single file loaded as minimax_h3 used to start the worker and
    die on a raw state-dict mapping error for to_gate_compress."""
    import pytest
    import torch
    from safetensors.torch import save_file

    path = tmp_path / "fasth3.safetensors"
    save_file({"blocks.0.attn.to_gate_compress.weight": torch.zeros(1)}, path)
    runtime = SGLDiffusionGenerator()
    runtime.get_comfyui_model = lambda *a: (SimpleNamespace(), None, "minimax_h3")
    runtime.init_generator = lambda *a: (_ for _ in ()).throw(
        AssertionError("worker must not start")
    )
    with pytest.raises(ValueError, match="model_type fast_h3"):
        runtime.load_model(model_path=str(path), sgld_options={})


def test_fasth3_runtime_dir_is_checked_before_worker_load(tmp_path) -> None:
    """The 4-step preview and a raw (unmaterialized) FastH3 V2 download were
    only rejected by the worker after a full model load."""
    import json

    import pytest

    release = {
        "schema_version": 1,
        "partition": "fl2va",
        "tasks": ["t2va"],
        "task_aliases": {},
        "sigma_shift_scales": {"video": 10.0, "audio": 3.0},
    }
    (tmp_path / "transformer").mkdir()
    index = tmp_path / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
    runtime = SGLDiffusionGenerator()
    runtime.get_comfyui_model = lambda *a: (SimpleNamespace(), None, "minimax_h3")
    runtime.init_generator = lambda *a: (_ for _ in ()).throw(
        AssertionError("worker must not start")
    )
    options = {"model_type": "fast_h3", "runtime_model_path": str(tmp_path)}
    for dmd, weight_map, message in (
        (None, {}, "no trained DMD rungs"),
        ([999, 500], {"blocks.0.x": "a.safetensors"}, "raw FastH3 download"),
    ):
        meta = dict(release, **({"dmd_denoising_steps": dmd} if dmd else {}))
        (tmp_path / "model_index.json").write_text(json.dumps({"_minimax_h3": meta}))
        index.write_text(json.dumps({"weight_map": weight_map}))
        with pytest.raises(ValueError, match=message):
            runtime.load_model(model_path="h3.safetensors", sgld_options=options)


def test_h3_per_request_attention_options_are_checked_before_worker_load(
    monkeypatch,
) -> None:
    """H3 has no per-request switchable attention, so an override only failed
    at the first sampling step; skip_softmax_params needs the FA backend."""
    import pytest

    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core import preflight

    monkeypatch.setattr(preflight, "_resolved_backend", lambda backend: backend)
    runtime = SGLDiffusionGenerator()
    runtime.get_comfyui_model = lambda *a: (SimpleNamespace(), None, "minimax_h3")
    runtime.init_generator = lambda *a: (_ for _ in ()).throw(
        AssertionError("worker must not start")
    )
    skip = {"skip_softmax_params": {"threshold_scale_factor": 1.0}}
    for options, message in (
        ({"request_options": {"attention_backend_override": "fa"}}, "per request"),
        (
            {"attention_backend": "torch_sdpa", "request_options": skip},
            "attention_backend=fa",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            runtime.load_model(model_path="h3.ckpt", sgld_options=options)
    with pytest.raises(AssertionError, match="worker must not start"):
        runtime.load_model(
            model_path="h3.ckpt",
            sgld_options={"attention_backend": "fa", "request_options": skip},
        )
