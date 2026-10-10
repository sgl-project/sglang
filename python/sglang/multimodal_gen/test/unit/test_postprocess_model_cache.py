# SPDX-License-Identifier: Apache-2.0
"""Model caches keyed by client-chosen paths must stay bounded."""

from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.postprocess import (
    realesrgan_upscaler,
    rife_interpolator,
)

EXTRA = 10


def test_upscaler_model_cache_is_bounded(monkeypatch):
    realesrgan_upscaler._MODEL_CACHE.clear()
    monkeypatch.setattr(realesrgan_upscaler, "_resolve_model_path", lambda p: p)
    monkeypatch.setattr(realesrgan_upscaler.torch, "load", lambda *a, **k: {})
    monkeypatch.setattr(
        realesrgan_upscaler,
        "_build_net_from_state_dict",
        lambda sd: torch.nn.Identity(),
    )
    monkeypatch.setattr(
        realesrgan_upscaler,
        "current_platform",
        SimpleNamespace(get_local_torch_device=lambda: torch.device("cpu")),
    )
    limit = realesrgan_upscaler._MAX_CACHED_MODELS
    for i in range(limit + EXTRA):
        realesrgan_upscaler.ImageUpscaler(model_path=f"m{i}.pth")._ensure_model_loaded()
    assert len(realesrgan_upscaler._MODEL_CACHE) == limit
    realesrgan_upscaler._MODEL_CACHE.clear()


def test_rife_model_cache_is_bounded(monkeypatch):
    rife_interpolator._MODEL_CACHE.clear()
    import sglang.multimodal_gen.runtime.utils.hf_diffusers_utils as hf

    class FakeModel:
        def __init__(self):
            self.flownet = torch.nn.Identity()

        def load_model(self, *a, **k):
            pass

        def eval(self):
            pass

    monkeypatch.setattr(hf, "maybe_download_model", lambda p: p)
    monkeypatch.setattr(rife_interpolator, "Model", FakeModel)
    monkeypatch.setattr(
        rife_interpolator,
        "current_platform",
        SimpleNamespace(get_local_torch_device=lambda: torch.device("cpu")),
    )
    limit = rife_interpolator._MAX_CACHED_MODELS
    for i in range(limit + EXTRA):
        rife_interpolator.FrameInterpolator(model_path=f"m{i}")._ensure_model_loaded()
    assert len(rife_interpolator._MODEL_CACHE) == limit
    rife_interpolator._MODEL_CACHE.clear()
