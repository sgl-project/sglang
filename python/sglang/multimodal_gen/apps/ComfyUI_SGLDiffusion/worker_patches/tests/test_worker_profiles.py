"""Worker lifecycle, failure propagation and H3 compatibility constraints."""

import multiprocessing as mp
import os
import subprocess
import sys
import types
from dataclasses import dataclass

import pytest

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion import worker_patches as patches


def test_registered_names_only():
    assert patches.canonical_profiles(None) == ()
    assert patches.canonical_profiles(["h3_comfyui_v1"] * 2) == ("h3_comfyui_v1",)
    for names in (["arbitrary.module:run"], "h3_comfyui_v1", [None]):
        with pytest.raises(ValueError, match="Unknown"):
            patches.canonical_profiles(names)


def test_composition_rejected(monkeypatch):
    monkeypatch.setitem(patches.PROFILES, "other", "other")
    with pytest.raises(ValueError, match="one worker patch"):
        patches.canonical_profiles(["h3_comfyui_v1", "other"])


def test_wrong_family_rejected():
    with pytest.raises(ValueError, match="model_type"):
        patches.validate_options(["h3_comfyui_v1"], "flux", {})


@pytest.mark.parametrize(
    "option",
    [
        {"num_gpus": 2},
        {"enable_torch_compile": True},
        {"enable_cuda_graph": True},
        {"quantization": "fp8"},
        {"cache_mode": "cache_dit"},
        {"attention_backend": "sage_attn"},
        {"component_attention_backends": {"transformer": "fa"}},
        {"lora_path": "unreviewed.safetensors"},
    ],
)
def test_unsupported_contract_rejected(option):
    with pytest.raises(ValueError):
        patches.validate_options(
            ["h3_comfyui_v1"],
            "minimax_h3",
            {"attention_backend": "torch_sdpa"} | option,
        )


def test_default_launch_and_restore_on_failure(monkeypatch):
    original = lambda spec: None
    module = types.ModuleType("sglang.multimodal_gen.runtime.launch_server")
    module.bootstrap_scheduler_process = original
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with patches.scheduler_profiles([]):
        assert module.bootstrap_scheduler_process is original
    with (
        pytest.raises(RuntimeError, match="launch failed"),
        patches.scheduler_profiles(["h3_comfyui_v1"]),
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


def test_bound_helper_does_not_change_class_or_other_instances():
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.worker_patches.h3 import (
        bind_with_globals,
    )

    class Example:
        def run(self):
            return os.getpid()

    patched, original = Example(), Example()
    function = Example.run
    bind_with_globals(
        patched, "run", {"os": types.SimpleNamespace(getpid=lambda: -123)}
    )
    assert patched.run() == -123
    assert original.run() == os.getpid()
    assert Example.run is function


def test_h3_dynamic_lora_rejected():
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.worker_patches.h3 import (
        _reject_lora,
    )

    with pytest.raises(ValueError, match="LoRA"):
        _reject_lora("adapter")


@pytest.fixture
def options_module():
    # Load the schema without importing ComfyUI's node/runtime dependencies.
    import importlib.util
    from pathlib import Path

    path = Path(patches.__file__).parent.parent / "options.py"
    spec = importlib.util.spec_from_file_location("sgld_options_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_alignment_default_preserves_existing_options(options_module):
    node = options_module.SGLDOptions()
    options = node.create_options()[0]
    assert options == node.create_options(alignment_mode="default")[0]
    assert "worker_patches" not in options
    assert "attention_backend" not in options


def test_comfyui_mode_auto_attention_is_valid_for_h3(options_module):
    options = options_module.SGLDOptions().create_options(alignment_mode="comfyui")[0]
    assert options["worker_patches"] == ["h3_comfyui_v1"]
    assert options["attention_backend"] == "torch_sdpa"
    patches.validate_options(options["worker_patches"], "minimax_h3", options)


def test_comfyui_mode_preserves_and_rejects_explicit_attention(options_module):
    options = options_module.SGLDOptions().create_options(
        alignment_mode="comfyui", attention_backend="flash_attention"
    )[0]
    assert options["attention_backend"] == "fa"
    with pytest.raises(ValueError, match="torch_sdpa"):
        patches.validate_options(options["worker_patches"], "minimax_h3", options)


def test_unknown_alignment_mode_rejected(options_module):
    with pytest.raises(ValueError, match="alignment_mode"):
        options_module.SGLDOptions().create_options(alignment_mode="typo")


def test_legacy_widget_positions_preserved(options_module):
    legacy = (
        "model_type num_gpus parallel_mode tp_size ulysses_degree ring_degree "
        "attention_backend attention_backend_config quantization int8_gemm "
        "cache_mode cache_dit_tuning cache_warmup_steps cache_threshold "
        "cache_max_cached_steps cache_fn_blocks execution_mode regional_compile "
        "memory_mode resident_layers prefetch_layers"
    ).split()
    assert list(options_module._FIELDS) == legacy + ["alignment_mode"]


def test_worker_import_does_not_load_comfyui():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
class NoComfyUI:
    def find_spec(self, fullname, *args):
        if fullname.endswith("ComfyUI_SGLDiffusion.nodes") or fullname in ("folder_paths", "comfy"):
            raise AssertionError(f"Worker imported {fullname}")
sys.meta_path.insert(0, NoComfyUI())
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.worker_patches import h3
""",
        ],
        check=True,
        timeout=30,
    )


@pytest.mark.parametrize("first", ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"])
def test_lazy_node_registration(monkeypatch, first):
    import importlib.util
    from pathlib import Path

    # ComfyUI loads plugins by file path under its own module name.
    name = "sgld_custom_node_test"
    path = Path(patches.__file__).parent.parent / "__init__.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    assert "NODE_CLASS_MAPPINGS" not in vars(module)
    assert module.WEB_DIRECTORY == "./web"
    nodes = types.ModuleType(name + ".nodes")
    nodes.NODE_CLASS_MAPPINGS = {"SGLDOptions": object()}
    nodes.NODE_DISPLAY_NAME_MAPPINGS = {"SGLDOptions": "SGLDiffusion Options"}
    monkeypatch.setitem(sys.modules, nodes.__name__, nodes)
    assert getattr(module, first) is getattr(nodes, first)
    assert module.NODE_CLASS_MAPPINGS is nodes.NODE_CLASS_MAPPINGS
    assert module.NODE_DISPLAY_NAME_MAPPINGS is nodes.NODE_DISPLAY_NAME_MAPPINGS
    with pytest.raises(AttributeError):
        module.unknown_attribute


def test_reviewed_h3_source_matches_checkout():
    import hashlib
    from pathlib import Path

    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.worker_patches.h3 import (
        SOURCE_SHA256,
    )

    source = Path(patches.__file__).parents[3] / "runtime/models/dits/minimax_h3.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA256
