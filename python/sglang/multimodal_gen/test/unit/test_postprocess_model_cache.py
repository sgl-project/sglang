# SPDX-License-Identifier: Apache-2.0
"""Model caches keyed by client-chosen paths must stay bounded."""

from collections import OrderedDict
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.postprocess import (
    realesrgan_upscaler,
    rife_interpolator,
)
from sglang.multimodal_gen.runtime.postprocess.bounded_cache import (
    cache_get,
    cache_put,
)

_CPU = SimpleNamespace(get_local_torch_device=lambda: torch.device("cpu"))


def test_cache_evicts_least_recently_used():
    """A get must refresh recency, so eviction drops the stale entry, not the oldest insert."""
    cache = OrderedDict()
    cache_put(cache, "a", 1, max_size=2)
    cache_put(cache, "b", 2, max_size=2)
    assert cache_get(cache, "a") == 1
    cache_put(cache, "c", 3, max_size=2)
    assert list(cache) == ["a", "c"]


def test_upscaler_uses_bounded_cache(monkeypatch):
    monkeypatch.setattr(realesrgan_upscaler, "_MODEL_CACHE", OrderedDict())
    monkeypatch.setattr(realesrgan_upscaler, "_MAX_CACHED_MODELS", 1)
    monkeypatch.setattr(realesrgan_upscaler, "_resolve_model_path", lambda p: p)
    monkeypatch.setattr(realesrgan_upscaler.torch, "load", lambda *a, **k: {})
    monkeypatch.setattr(
        realesrgan_upscaler,
        "_build_net_from_state_dict",
        lambda sd: torch.nn.Identity(),
    )
    monkeypatch.setattr(realesrgan_upscaler, "current_platform", _CPU)
    for path in ["a.pth", "b.pth"]:
        realesrgan_upscaler.ImageUpscaler(model_path=path)._ensure_model_loaded()
    assert list(realesrgan_upscaler._MODEL_CACHE) == ["b.pth"]


def test_rife_uses_bounded_cache(monkeypatch):
    import sglang.multimodal_gen.runtime.utils.hf_diffusers_utils as hf

    class FakeModel:
        def __init__(self):
            self.flownet = torch.nn.Identity()

        def load_model(self, *a, **k):
            pass

        def eval(self):
            pass

    monkeypatch.setattr(rife_interpolator, "_MODEL_CACHE", OrderedDict())
    monkeypatch.setattr(rife_interpolator, "_MAX_CACHED_MODELS", 1)
    monkeypatch.setattr(hf, "maybe_download_model", lambda p: p)
    monkeypatch.setattr(rife_interpolator, "Model", FakeModel)
    monkeypatch.setattr(rife_interpolator, "current_platform", _CPU)
    for path in ["a", "b"]:
        rife_interpolator.FrameInterpolator(model_path=path)._ensure_model_loaded()
    assert list(rife_interpolator._MODEL_CACHE) == ["b"]
