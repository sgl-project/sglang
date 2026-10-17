"""Load the real ComfyUI nodes and server client without a ComfyUI install.

The plugin only runs inside ComfyUI, so its runtime modules are stubbed for the
duration of the import and removed from ``sys.modules`` afterwards. Tests drive
the real node through the real ``SGLDiffusionServerAPI`` and mock only HTTP.
"""

import functools
import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

# Import real dependencies before sys.modules is patched, or the rollback drops
# them and a later import loads a second copy (torch cannot be loaded twice).
import numpy  # noqa: F401
import PIL.Image  # noqa: F401
import requests  # noqa: F401
import torch  # noqa: F401

PLUGIN_DIR = Path(__file__).resolve().parents[1]
PKG = "sgld_comfy_harness"

TEMP_DIR = tempfile.mkdtemp(prefix="sgld_comfy_test_")

# A completed job's file_path must exist locally, or the client fetches the
# result over HTTP as it would from a remote server.
OUTPUT_VIDEO = os.path.join(TEMP_DIR, "out.mp4")
open(OUTPUT_VIDEO, "wb").close()


def _stubs() -> dict:
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.folder_names_and_paths = {}
    folder_paths.get_temp_directory = lambda: TEMP_DIR
    folder_paths.get_filename_list = lambda folder: []
    folder_paths.get_full_path = lambda folder, name: f"/{folder}/{name}"

    comfy = types.ModuleType("comfy")
    comfy_utils = types.ModuleType("comfy.utils")
    comfy_utils.ProgressBar = lambda total: types.SimpleNamespace(
        update_absolute=lambda value, total=None, preview=None: None
    )
    comfy_mm = types.ModuleType("comfy.model_management")
    comfy_mm.InterruptProcessingException = type(
        "InterruptProcessingException", (Exception,), {}
    )
    comfy_mm.throw_exception_if_processing_interrupted = lambda: None
    comfy.utils, comfy.model_management = comfy_utils, comfy_mm

    comfy_api = types.ModuleType("comfy_api")
    comfy_api_input = types.ModuleType("comfy_api.input")
    comfy_api_input.VideoInput = type("VideoInput", (), {})
    comfy_api.input = comfy_api_input

    return {
        "folder_paths": folder_paths,
        "comfy": comfy,
        "comfy.utils": comfy_utils,
        "comfy.model_management": comfy_mm,
        "comfy_api": comfy_api,
        "comfy_api.input": comfy_api_input,
    }


def _load(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(
        module_name, PLUGIN_DIR / relative_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@functools.cache
def load_plugin():
    """Return ``(server_api, utils, nodes)`` modules from the plugin sources.

    Loaded once per session so every test file shares the module copies that
    ``mock.patch("pkg.nodes.x")`` resolves by name.

    ``core.generator`` is replaced by a placeholder: it reaches for ComfyUI's
    model machinery, which the server-mode nodes never call.
    """
    with mock.patch.dict(sys.modules, _stubs()):
        package = types.ModuleType(PKG)
        package.__path__ = [str(PLUGIN_DIR)]
        sys.modules[PKG] = package

        server_api = _load(f"{PKG}.core.server_api", "core/server_api.py")
        core = types.ModuleType(f"{PKG}.core")
        core.__path__ = [str(PLUGIN_DIR / "core")]
        core.SGLDiffusionServerAPI = server_api.SGLDiffusionServerAPI
        core.SGLDiffusionGenerator = object
        sys.modules[f"{PKG}.core"] = core

        utils = _load(f"{PKG}.utils", "utils.py")
        nodes = _load(f"{PKG}.nodes", "nodes.py")
        plugin_modules = {
            name: mod for name, mod in sys.modules.items() if name.startswith(PKG)
        }
    # patch.dict rolled back sys.modules; keep the plugin so mock.patch("pkg.x") resolves
    sys.modules.update(plugin_modules)
    package.nodes, package.utils, package.core = nodes, utils, core
    core.server_api = server_api
    return server_api, utils, nodes


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload
