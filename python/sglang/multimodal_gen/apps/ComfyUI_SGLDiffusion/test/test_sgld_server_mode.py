"""Regression tests for ComfyUI_SGLDiffusion server-mode bugs found in audit.

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
PKG = "sgld_server_mode_under_test"


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


# --- Bug 9: hardcoded 300s timeout on image requests -----------------------


def test_image_timeout_is_configurable():
    client = SGLDiffusionServerAPI(base_url="http://127.0.0.1:1234", image_timeout=900)
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None, **kwargs):
        captured["timeout"] = timeout
        return _Response({"data": [{"b64_json": "x"}]})

    with mock.patch(f"{PKG}.core.server_api.requests.post", side_effect=fake_post):
        client.generate_image(prompt="a cat")

    assert captured["timeout"] == 900
