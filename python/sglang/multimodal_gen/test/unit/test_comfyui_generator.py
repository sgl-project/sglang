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


def test_multi_gpu_single_file_starts_without_cfg_parallel(
    tmp_path, monkeypatch
) -> None:
    """Auto CFG parallel read a model_index.json, so a Flux single file on two GPUs
    failed ServerArgs validation; ComfyUI already splits cond/uncond requests."""
    import torch
    from safetensors.torch import save_file

    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core import generator
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    path = tmp_path / "flux.safetensors"
    save_file({"double_blocks.0.img_attn.qkv.weight": torch.zeros(1)}, path)
    seen = {}
    monkeypatch.setattr(
        generator.DiffGenerator,
        "from_pretrained",
        lambda **kwargs: seen.update(kwargs) or object(),
    )
    SGLDiffusionGenerator().init_generator(str(path), "FluxPipeline", {"num_gpus": 2})
    assert seen["cfg_parallel_degree"] == 1
    if torch.cuda.device_count() >= 2:
        server_args = ServerArgs.from_kwargs(**seen)
        assert server_args.enable_cfg_parallel is False
        assert server_args.sp_degree == 2
    seen.clear()
    SGLDiffusionGenerator().init_generator(
        str(path), "FluxPipeline", {"num_gpus": 2, "enable_cfg_parallel": True}
    )
    assert "cfg_parallel_degree" not in seen


def _status_runtime(alive=True):
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.worker_status import (
        StepLatency,
    )

    runtime = SGLDiffusionGenerator()
    runtime.model_path = "z.safetensors"
    runtime._patcher = SimpleNamespace(model_type="zimage")
    runtime.generator = SimpleNamespace(
        server_args=SimpleNamespace(
            pipeline_class_name="ZImagePipeline",
            component_quantizations={},
            **{
                k: 1
                for k in (
                    "num_gpus tp_size sp_degree ulysses_degree ring_degree "
                    "dp_size enable_cfg_parallel attention_backend "
                    "dit_cpu_offload dit_layerwise_offload"
                ).split()
            },
        )
    )
    runtime.executor = SimpleNamespace(step_latency=StepLatency())
    runtime._worker_processes = lambda: [SimpleNamespace(pid=7, is_alive=lambda: alive)]
    runtime._is_live = lambda: alive
    runtime.last_options = {
        "model_path": "z",
        "sgld_options": {"hf_token": "abc", "tp_size": 2},
    }
    return runtime


def test_step_latency_groups_cfg_calls_and_is_bounded() -> None:
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.worker_status import (
        StepLatency,
    )

    tracker = StepLatency(window=3)
    tracker.record(9.0, 1.0)
    tracker.record(9.0, 1.0)
    for i in range(2, 8):
        tracker.record(1.0, float(i))
        tracker.record(1.0, float(i))
    stats = tracker.stats()
    assert stats["first"] == 18.0
    assert stats["mean"] == stats["p50"] == stats["p95"] == stats["last"] == 2.0
    assert stats["calls_per_step"] == 2.0
    assert stats["count"] == 7
    before = tracker.stats()
    assert tracker.stats() == before


def test_step_latency_new_run_splits_equal_timesteps() -> None:
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.worker_status import (
        StepLatency,
    )

    tracker = StepLatency()
    tracker.record(5.0, 0.5)
    tracker.begin_run()
    tracker.record(1.0, 0.5)
    tracker.begin_run()
    tracker.record(3.0, 0.5)
    stats = tracker.stats()
    assert stats["count"] == 3
    assert stats["mean"] == 2.0
    assert stats["calls_per_step"] == 1.0


def test_worker_status_node_show_without_worker(monkeypatch) -> None:
    import importlib
    import sys
    import types

    stub = types.ModuleType("folder_paths")
    stub.folder_names_and_paths = {}
    stub.get_filename_list = lambda name: []
    stub.get_full_path = lambda folder, name: f"/{folder}/{name}"
    monkeypatch.setitem(sys.modules, "folder_paths", stub)
    nodes_name = "sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.nodes"
    # Restore any earlier import afterwards; drop the one bound to this stub.
    monkeypatch.delitem(sys.modules, nodes_name, raising=False)
    try:
        nodes = importlib.import_module(nodes_name)
    except ImportError as exc:
        pytest.skip(f"nodes.py needs ComfyUI: {exc!r}")
    finally:
        sys.modules.pop(nodes_name, None)
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.base import (
        SGLDiffusionExecutor,
    )

    SGLDiffusionGenerator.reset_shared()
    executor = SGLDiffusionExecutor.__new__(SGLDiffusionExecutor)
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.worker_status import (
        StepLatency,
    )

    executor.step_latency = StepLatency()
    model = SimpleNamespace(model=SimpleNamespace(diffusion_model=executor))
    out = nodes.SGLDWorkerStatus().show(model, 0.0)
    SGLDiffusionGenerator.reset_shared()
    assert out["result"] == out["ui"]["text"]
    assert "not started" in out["result"][0]


def test_status_without_worker_does_not_crash() -> None:
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.worker_status import (
        collect_status,
        format_status,
    )

    text = format_status(collect_status(SGLDiffusionGenerator(), gpu_query=list))
    assert "not started" in text
    assert "unavailable" in text


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_query_gpus_reports_the_real_device() -> None:
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.worker_status import (
        query_gpus,
    )

    gpus = query_gpus()
    assert len(gpus) == torch.cuda.device_count()
    assert gpus[0]["total_mb"] > 0
    assert 0 <= gpus[0]["used_mb"] <= gpus[0]["total_mb"]


def test_status_reports_dead_worker_and_broken_gpu_query() -> None:
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.worker_status import (
        collect_status,
        format_status,
    )

    def boom():
        raise OSError("no nvml")

    text = format_status(collect_status(_status_runtime(alive=False), gpu_query=boom))
    assert "NOT live" in text
    assert "dead=[7]" in text
    assert "VRAM unavailable" in text


def test_speedup_only_with_baseline_and_secrets_stripped() -> None:
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.worker_status import (
        collect_status,
        format_status,
    )

    runtime = _status_runtime()
    for i, value in enumerate((5.0, 0.5, 0.5)):
        runtime.executor.step_latency.record(value, float(i))
    status = collect_status(runtime, gpu_query=list)
    assert "speedup" not in format_status(status)
    assert "speedup vs native: 2.00x" in format_status(status, 1.0)
    text = format_status(status)
    assert "abc" not in text
    assert "'tp_size': 2" in text
