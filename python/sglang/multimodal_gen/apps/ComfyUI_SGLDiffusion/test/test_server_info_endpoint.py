"""Tests for SGLDiffusionServerAPI.get_model_info.

The server only registers /models under the /v1 prefix (see
runtime/entrypoints/openai/common_api.py), and its response wraps the model
card in an OpenAI-style {"object": "list", "data": [...]} envelope. These
tests mock only HTTP, so a regression on either side of that contract fails
here without a running server or a GPU.
"""

import importlib.util
import sys
import types
from pathlib import Path
from unittest import mock

PLUGIN_DIR = Path(__file__).resolve().parents[1]
PKG = "sgld_comfy_under_test_server_info"


def _load(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(
        module_name, PLUGIN_DIR / relative_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_server_api():
    package = types.ModuleType(PKG)
    package.__path__ = [str(PLUGIN_DIR)]
    sys.modules[PKG] = package

    core = types.ModuleType(f"{PKG}.core")
    core.__path__ = [str(PLUGIN_DIR / "core")]
    sys.modules[f"{PKG}.core"] = core

    return _load(f"{PKG}.core.server_api", "core/server_api.py")


SERVER_API = _load_server_api()
SGLDiffusionServerAPI = SERVER_API.SGLDiffusionServerAPI

MODEL_CARD = {
    "id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "object": "model",
    "created": 1700000000,
    "owned_by": "sglang",
    "num_gpus": 4,
    "task_type": "T2V",
    "pipeline_name": "WanPipeline",
    "dit_precision": "bf16",
    "vae_precision": "fp32",
}


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _fake_get_models(url, headers=None, timeout=None):
    return _Response({"object": "list", "data": [MODEL_CARD]})


def test_get_model_info_requests_the_v1_models_route():
    """The server only mounts /models under /v1; hitting the bare host 404s."""
    client = SGLDiffusionServerAPI(base_url="http://localhost:30010/v1")
    requested_urls = []

    def fake_get(url, headers=None, timeout=None):
        requested_urls.append(url)
        return _fake_get_models(url, headers, timeout)

    with mock.patch(f"{PKG}.core.server_api.requests.get", side_effect=fake_get):
        client.get_model_info()

    assert requested_urls == ["http://localhost:30010/v1/models"]


def test_get_model_info_unwraps_the_openai_list_envelope():
    """Callers expect the flat model card, not the {object, data} envelope."""
    client = SGLDiffusionServerAPI(base_url="http://localhost:30010/v1")

    with mock.patch(
        f"{PKG}.core.server_api.requests.get", side_effect=_fake_get_models
    ):
        model_info = client.get_model_info()

    assert model_info["task_type"] == "T2V"
    assert model_info["pipeline_name"] == "WanPipeline"
    assert model_info["num_gpus"] == 4
    assert "data" not in model_info
    assert model_info["object"] == "model"
