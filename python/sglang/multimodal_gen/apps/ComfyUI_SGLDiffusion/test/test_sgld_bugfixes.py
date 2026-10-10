"""Regression tests for a dozen ComfyUI_SGLDiffusion bugs found in audit.

Each bug gets a test that fails against the pre-fix code and passes after.
These stub ComfyUI/ torch-adjacent modules the same way test_h3_request.py
does, so they run without a ComfyUI install or a GPU.
"""

import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

import pytest
import torch

PLUGIN_DIR = Path(__file__).resolve().parents[1]
PKG = "sgld_bugfix_under_test"


def _install_comfy_stubs() -> None:
    folder_paths = types.ModuleType("folder_paths")
    _tmp = tempfile.mkdtemp(prefix="sgld_test_temp_")
    folder_paths.get_temp_directory = lambda: _tmp
    folder_paths._diffusion_models_dirs = []
    folder_paths.folder_names_and_paths = {
        "diffusion_models": ([], {".safetensors", ".sft"}),
        "loras": ([], {".safetensors"}),
    }

    def get_filename_list(name):
        names = set()
        for folder in folder_paths.folder_names_and_paths.get(name, ([], set()))[0]:
            if os.path.isdir(folder):
                names.update(os.listdir(folder))
        return sorted(names)

    def get_folder_paths(name):
        return list(folder_paths.folder_names_and_paths.get(name, ([], set()))[0])

    def get_full_path(name, filename):
        for folder in get_folder_paths(name):
            candidate = os.path.join(folder, filename)
            if os.path.isfile(candidate):
                return candidate
        return None

    folder_paths.get_filename_list = get_filename_list
    folder_paths.get_folder_paths = get_folder_paths
    folder_paths.get_full_path = get_full_path
    sys.modules["folder_paths"] = folder_paths

    comfy_api = types.ModuleType("comfy_api")
    comfy_api_input = types.ModuleType("comfy_api.input")

    class VideoInput:
        pass

    comfy_api_input.VideoInput = VideoInput
    comfy_api.input = comfy_api_input
    sys.modules["comfy_api"] = comfy_api
    sys.modules["comfy_api.input"] = comfy_api_input
    # No comfy_api.input_impl: exercises the SGLDVideoInput fallback path.
    sys.modules.pop("comfy_api.input_impl", None)

    comfy = types.ModuleType("comfy")
    comfy.model_detection = types.ModuleType("comfy.model_detection")
    comfy.model_management = types.ModuleType("comfy.model_management")

    comfy_utils = types.ModuleType("comfy.utils")
    for name in (
        "calculate_parameters",
        "load_torch_file",
        "state_dict_prefix_replace",
        "unet_to_diffusers",
    ):
        setattr(comfy_utils, name, lambda *a, **k: None)
    comfy.utils = comfy_utils

    comfy_model_patcher = types.ModuleType("comfy.model_patcher")

    class ModelPatcher:
        def __init__(self, model, load_device, offload_device, size=0, *a, **k):
            self.model = model
            self.load_device = load_device
            self.offload_device = offload_device
            self.size = size
            self.patches = {}
            self.object_patches = {}
            self.model_options = {}
            self.backup = {}
            self.object_patches_backup = {}
            self.wrappers = {}
            self.callbacks = {}
            self.patches_uuid = None
            self.weight_inplace_update = False

        def add_wrapper_with_key(self, *a, **k):
            pass

    comfy_model_patcher.ModelPatcher = ModelPatcher
    comfy.model_patcher = comfy_model_patcher

    comfy_patcher_extension = types.ModuleType("comfy.patcher_extension")

    class WrappersMP:
        SAMPLER_SAMPLE = "SAMPLER_SAMPLE"

    comfy_patcher_extension.WrappersMP = WrappersMP
    comfy.patcher_extension = comfy_patcher_extension

    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_detection"] = comfy.model_detection
    sys.modules["comfy.model_management"] = comfy.model_management
    sys.modules["comfy.utils"] = comfy_utils
    sys.modules["comfy.model_patcher"] = comfy_model_patcher
    sys.modules["comfy.patcher_extension"] = comfy_patcher_extension
    return folder_paths


def _load(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(
        module_name, PLUGIN_DIR / relative_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_plugin():
    folder_paths = _install_comfy_stubs()

    package = types.ModuleType(PKG)
    package.__path__ = [str(PLUGIN_DIR)]
    sys.modules[PKG] = package

    core = types.ModuleType(f"{PKG}.core")
    core.__path__ = [str(PLUGIN_DIR / "core")]
    sys.modules[f"{PKG}.core"] = core

    server_api = _load(f"{PKG}.core.server_api", "core/server_api.py")
    core.SGLDiffusionServerAPI = server_api.SGLDiffusionServerAPI
    model_patcher = _load(f"{PKG}.core.model_patcher", "core/model_patcher.py")
    core.SGLDModelPatcher = model_patcher.SGLDModelPatcher
    generator = _load(f"{PKG}.core.generator", "core/generator.py")
    core.SGLDiffusionGenerator = generator.SGLDiffusionGenerator

    _load(f"{PKG}.utils", "utils.py")
    nodes = _load(f"{PKG}.nodes", "nodes.py")
    return server_api, model_patcher, generator, nodes, sys.modules[f"{PKG}.utils"], folder_paths


SERVER_API, MODEL_PATCHER, GENERATOR, NODES, UTILS, FOLDER_PATHS = _load_plugin()
SGLDiffusionServerAPI = SERVER_API.SGLDiffusionServerAPI
SGLDModelPatcher = MODEL_PATCHER.SGLDModelPatcher


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload

    def iter_content(self, chunk_size=None):
        yield b"fake-mp4-bytes"


# --- Bug 8: a missing sglang runtime made nodes disappear with no logging --


def test_init_logs_when_node_registration_fails(caplog):
    import logging

    pkg_name = f"{PKG}_init_fail"
    package = types.ModuleType(pkg_name)
    package.__path__ = [str(PLUGIN_DIR)]
    sys.modules[pkg_name] = package

    # Make the inner `from .nodes import ...` fail so __init__.py's except
    # branch runs.
    broken_nodes = types.ModuleType(f"{pkg_name}.nodes")

    def _raise(*a, **k):
        raise ImportError("simulated missing runtime")

    broken_nodes.__getattr__ = _raise
    sys.modules[f"{pkg_name}.nodes"] = broken_nodes
    # Force the `from .nodes import NODE_CLASS_MAPPINGS` to raise by not
    # defining the attributes at all (AttributeError via __getattr__ above
    # doesn't trigger ImportError machinery, so delete the module and let
    # the real loader fail via a stub raising at import time instead).
    del sys.modules[f"{pkg_name}.nodes"]

    init_path = PLUGIN_DIR / "__init__.py"
    spec = importlib.util.spec_from_file_location(pkg_name, init_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = module

    with mock.patch.dict(
        sys.modules,
        {f"{pkg_name}.nodes": None},
    ):
        with caplog.at_level(logging.ERROR):
            spec.loader.exec_module(module)

    assert module.NODE_CLASS_MAPPINGS == {}
    assert any("failed to register nodes" in r.message for r in caplog.records)


def test_minimax_h3_module_importable_without_sglang_runtime():
    """The missing-runtime guard belongs in minimax_h3.py itself (mirroring
    base.py); previously its unguarded top-level import crashed the whole
    executors package, and generator.py imports it unconditionally.
    """
    pkg_name = f"{PKG}_no_runtime"
    package = types.ModuleType(pkg_name)
    package.__path__ = [str(PLUGIN_DIR)]
    sys.modules[pkg_name] = package
    executors_pkg = types.ModuleType(f"{pkg_name}.executors")
    executors_pkg.__path__ = [str(PLUGIN_DIR / "executors")]
    sys.modules[f"{pkg_name}.executors"] = executors_pkg

    _load(f"{pkg_name}.executors.adapter", "executors/adapter.py")
    _load(f"{pkg_name}.executors.base", "executors/base.py")

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("sglang.multimodal_gen"):
            raise ImportError("sglang[diffusion] not installed")
        return real_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=fake_import):
        module = _load(f"{pkg_name}.executors.minimax_h3", "executors/minimax_h3.py")

    assert module._H3_RUNTIME_IMPORT_ERROR is not None
    with pytest.raises(RuntimeError, match="failed to import"):
        module.MiniMaxH3Executor(generator=None, model_path="x", model=None, config=None)


# --- Bug 1: any .gguf was treated as MiniMax-H3 for architecture detect ----


def test_gguf_detect_infers_architecture_from_model_type_hint():
    model_type = GENERATOR._infer_gguf_model_type("whatever.gguf", "flux")
    assert model_type == "flux"


def test_gguf_detect_infers_architecture_from_filename_when_no_hint():
    assert GENERATOR._infer_gguf_model_type("qwen_image-Q4.gguf", None) == "qwen_image"
    assert GENERATOR._infer_gguf_model_type("flux1-dev-Q8.gguf", None) == "flux"


def test_gguf_detect_does_not_default_to_h3_for_unknown_architecture():
    with pytest.raises(ValueError, match="Cannot tell which architecture"):
        GENERATOR._h3_detect_companion("/models/mystery_model.gguf")


def test_gguf_detect_looks_for_matching_companion_not_h3():
    with tempfile.TemporaryDirectory() as tmp:
        # Only a flux companion exists; a flux GGUF should accept it, but an
        # unrelated qwen GGUF in the same folder must not silently grab it.
        open(os.path.join(tmp, "flux1-dev.safetensors"), "wb").close()
        gguf_path = os.path.join(tmp, "flux1-schnell-Q4.gguf")
        found = GENERATOR._h3_detect_companion(gguf_path, model_type_hint="flux")
        assert found == os.path.join(tmp, "flux1-dev.safetensors")

        qwen_gguf = os.path.join(tmp, "qwen_image-Q4.gguf")
        with pytest.raises(ValueError, match="qwen_image"):
            GENERATOR._h3_detect_companion(qwen_gguf)


# --- Bug 2: .gguf was added to ComfyUI's global diffusion_models list ------


def test_gguf_support_does_not_leak_into_global_folder_registry():
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, "model.gguf"), "wb").close()
        FOLDER_PATHS.folder_names_and_paths["diffusion_models"][0].append(tmp)
        try:
            names = NODES._list_unet_names_including_gguf()
            assert "model.gguf" in names
            # The stock extension set used by every other node (e.g.
            # ComfyUI's own "Load Diffusion Model") must stay untouched.
            assert ".gguf" not in FOLDER_PATHS.folder_names_and_paths[
                "diffusion_models"
            ][1]
        finally:
            FOLDER_PATHS.folder_names_and_paths["diffusion_models"][0].remove(tmp)


