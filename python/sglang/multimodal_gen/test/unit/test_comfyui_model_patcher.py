# SPDX-License-Identifier: Apache-2.0
"""SGLDModelPatcher: clone tracking and freeing the worker's GPU memory."""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

model_management = pytest.importorskip("comfy.model_management")

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core.generator import (
    SGLDiffusionGenerator,
    _native_model_size_bytes,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core.model_patcher import (
    SGLDModelPatcher,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.flux import (
    FluxExecutor,
)


def test_expired_clone_returns_loaded_model_to_live_parent():
    model = torch.nn.Linear(1, 1)
    device = torch.device("cpu")
    parent = SGLDModelPatcher(model, device, device, model_type="minimax_h3")
    tracked = model_management.LoadedModel(parent.clone().clone())
    tracked.real_model = weakref.ref(model)

    gc.collect()

    assert not tracked.is_dead()
    assert tracked.model is parent
    assert tracked.real_model() is model


def _patcher(worker=None, size=100, token=None, model_type="qwen_image"):
    cpu = torch.device("cpu")
    return SGLDModelPatcher(
        torch.nn.Module(),
        cpu,
        cpu,
        size=size,
        model_type=model_type,
        worker=worker,
        worker_token=token,
    )


def _live_generator(monkeypatch, output=None):
    """A generator whose worker answers every memory RPC with ``output``."""
    gen = SGLDiffusionGenerator()
    sent = []

    def send(reqs, timeout_ms=None):
        sent.append(type(reqs[0]).__name__)
        return SimpleNamespace(error=None, output=output or {"success": True})

    gen.generator = SimpleNamespace(_send_to_scheduler_and_wait_for_response=send)
    gen._patcher = object()
    monkeypatch.setattr(gen, "_is_live", lambda: True)
    return gen, sent


def test_comfyui_unload_sleeps_the_worker_and_the_next_use_wakes_it(monkeypatch):
    gen, sent = _live_generator(monkeypatch)
    patcher = _patcher(gen, token=gen.generator)
    assert patcher.loaded_size() == 100
    assert patcher.partially_unload(torch.device("cpu"), 123) == 0
    patcher.clone().detach()  # clones share the worker handle
    patcher.detach()  # already asleep: no second request
    assert sent == ["ReleaseMemoryOccupationReqInput"]
    assert patcher.loaded_size() == 0

    # set_lora goes through ensure_executor, which wakes the worker first
    gen.generator.set_lora = lambda **kw: sent.append("set_lora")
    executor = object.__new__(FluxExecutor)
    executor.generator = gen.generator
    executor._ensure_runtime = gen.ensure_executor
    executor.set_lora(lora_nickname="a", lora_path="/x.safetensors")
    assert sent[1:] == ["ResumeMemoryOccupationReqInput", "set_lora"]
    assert not gen.worker_asleep


def test_stale_dead_or_off_gpu_worker_is_not_slept(monkeypatch):
    gen, sent = _live_generator(monkeypatch)
    stale = _patcher(gen, token=object())  # patcher of a replaced worker
    assert stale.loaded_size() == 0
    stale.detach()
    gen.sleep_worker(object())
    monkeypatch.setattr(gen, "_is_live", lambda: False)
    gen.sleep_worker(gen.generator)
    assert sent == []

    monkeypatch.setattr(gen, "_is_live", lambda: True)
    gen.shares_comfy_gpu = False
    off_gpu = _patcher(gen, token=gen.generator)
    # ComfyUI frees model_size() - loaded_size() on its own GPU before load
    assert off_gpu.model_size() == off_gpu.clone().model_size() == 0
    assert off_gpu.loaded_size() == 0
    off_gpu.detach()
    gen.release_worker_cache()
    assert sent == [] and not gen.worker_asleep


def test_released_cache_measures_what_the_worker_keeps_resident(monkeypatch):
    gen, sent = _live_generator(monkeypatch, {"success": True, "resident_bytes": 40})
    patcher = _patcher(gen, token=gen.generator)
    gen.release_worker_cache()
    assert sent == ["ReleaseAllocatorCacheReqInput"]
    assert patcher.loaded_size() == 40
    gen.worker_asleep = True
    gen.release_worker_cache()
    assert len(sent) == 1


def test_model_size_falls_back_to_the_table_only_when_unknown():
    assert _patcher(size=13_000_000_000).model_size() == 13_000_000_000
    assert _patcher(size=0, model_type="flux").model_size() == 27 * 1024**3


def test_worker_gpu_topology(monkeypatch):
    share = SGLDiffusionGenerator._worker_shares_comfy_gpu
    monkeypatch.setattr(
        model_management, "get_torch_device", lambda: torch.device("cuda", 1)
    )
    assert share({"base_gpu_id": 1})
    assert share({"base_gpu_id": 0, "num_gpus": 2})
    assert not share({"base_gpu_id": 2})
    assert not share({})
    monkeypatch.setattr(
        model_management, "get_torch_device", lambda: torch.device("cpu")
    )
    assert not share({})


def test_native_model_size_counts_only_dit_weight_files(tmp_path):
    weights = tmp_path / "transformer"
    weights.mkdir()
    (weights / "config.json").write_bytes(b"{}")
    (weights / "model-00001.safetensors").write_bytes(b"weights")
    (weights / "model-00002.safetensors").write_bytes(b"more weights")
    (tmp_path / "vae").mkdir()
    (tmp_path / "vae" / "model.safetensors").write_bytes(b"not the dit")
    assert _native_model_size_bytes(str(tmp_path)) == 19
