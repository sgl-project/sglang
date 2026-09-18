"""Unit tests for diffusion model-overlay cache paths."""

from sglang.multimodal_gen.runtime.utils.model_overlay import (
    get_diffusion_cache_root,
)


def test_uses_xdg_cache_home_by_default(monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/sglang-xdg-cache")
    monkeypatch.delenv("SGLANG_DIFFUSION_CACHE_ROOT", raising=False)

    assert get_diffusion_cache_root() == "/tmp/sglang-xdg-cache/sgl_diffusion"


def test_explicit_diffusion_cache_root_takes_precedence(monkeypatch):
    monkeypatch.setenv("SGLANG_DIFFUSION_CACHE_ROOT", "/tmp/sglang-diffusion-cache")
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/sglang-xdg-cache")

    assert get_diffusion_cache_root() == "/tmp/sglang-diffusion-cache"
