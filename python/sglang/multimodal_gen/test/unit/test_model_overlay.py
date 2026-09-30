"""Unit tests for diffusion model-overlay cache paths and materialized trees."""

import os

from sglang.multimodal_gen.runtime.utils import model_overlay
from sglang.multimodal_gen.runtime.utils.model_overlay import (
    _copytree_link_or_copy,
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


def _hf_snapshot(tmp_path):
    """A snapshot laid out like the HF cache: files are symlinks into blobs/."""
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    config = blobs / "config-blob"
    config.write_text('{"source": true}')
    weights = blobs / "weights-blob"
    weights.write_bytes(b"w" * 4096)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").symlink_to(config)
    (snapshot / "model.safetensors").symlink_to(weights)
    return snapshot, config, weights


def test_rewriting_a_materialized_config_leaves_the_cache_blob_alone(
    tmp_path, monkeypatch
):
    # The LTX-2.3 overlay rewrites text_encoder/config.json after linking the
    # tree; through a link that rewrote the shared cache's blob.
    monkeypatch.setattr(model_overlay, "_OVERLAY_LINK_MIN_BYTES", 1024)
    snapshot, config, _ = _hf_snapshot(tmp_path)
    out = tmp_path / "materialized"

    _copytree_link_or_copy(str(snapshot), str(out))
    with open(out / "config.json", "w") as f:
        f.write('{"source": false}')

    assert config.read_text() == '{"source": true}'
    assert not os.path.islink(out / "config.json")


def test_large_files_are_still_shared_with_the_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(model_overlay, "_OVERLAY_LINK_MIN_BYTES", 1024)
    snapshot, _, weights = _hf_snapshot(tmp_path)
    out = tmp_path / "materialized"

    _copytree_link_or_copy(str(snapshot), str(out))

    assert os.path.samefile(out / "model.safetensors", weights)
