"""Unit tests for diffusion model-overlay cache paths and materialized trees."""

import json
import os
from pathlib import Path

import pytest
from filelock import FileLock

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


def test_bundled_only_overlay_fails_loudly_when_not_installed():
    # A bundled-only entry has no HF repo to fall back to.
    spec = {"bundled_overlay_subdir": "not_installed"}
    with pytest.raises(ValueError, match="missing from this SGLang installation"):
        model_overlay.download_overlay_metadata(
            "org/model", spec, snapshot_download_fn=None
        )


def test_large_files_are_still_shared_with_the_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(model_overlay, "_OVERLAY_LINK_MIN_BYTES", 1024)
    snapshot, _, weights = _hf_snapshot(tmp_path)
    out = tmp_path / "materialized"

    _copytree_link_or_copy(str(snapshot), str(out))

    assert os.path.samefile(out / "model.safetensors", weights)


@pytest.mark.parametrize("repair_cache", [False, True])
def test_overlay_uses_pinned_source_weights(tmp_path, monkeypatch, repair_cache):
    overlay = tmp_path / "overlay"
    (overlay / "_overlay").mkdir(parents=True)
    (overlay / "_overlay" / "overlay_manifest.json").write_text(
        json.dumps(
            {
                "required_source_files": ["model.safetensors"],
                "file_mappings": [
                    {"src": "model.safetensors", "dst": "transformer/model.safetensors"}
                ],
            }
        )
    )
    (overlay / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "TestPipeline",
                "_diffusers_version": "0.35.0",
                "transformer": ["diffusers", "Transformer"],
            }
        )
    )
    for revision in ("main", "original", "newer"):
        source = tmp_path / revision
        source.mkdir()
        if revision != "original" or not repair_cache:
            (source / "model.safetensors").write_text(revision)

    spec = {"overlay_repo_id": "test/overlay", "source_revision": "original"}
    monkeypatch.setattr(
        model_overlay, "resolve_model_overlay_target", lambda _: ("test/model", spec)
    )
    monkeypatch.setattr(
        model_overlay, "download_overlay_metadata", lambda *a, **kw: str(overlay)
    )
    monkeypatch.setattr(
        model_overlay, "get_diffusion_cache_root", lambda: str(tmp_path / "cache")
    )
    monkeypatch.setattr(
        model_overlay, "get_lock", lambda _: FileLock(tmp_path / "overlay.lock")
    )

    def cached_snapshot(*args, revision=None, **kwargs):
        return str(tmp_path / (revision or "main"))

    def download_snapshot(*args, revision=None, **kwargs):
        source = Path(cached_snapshot(revision=revision))
        (source / "model.safetensors").write_text(revision or "main")
        return str(source)

    def resolve():
        return model_overlay.maybe_resolve_overlay_model_path(
            "test/model",
            local_dir=None,
            download=True,
            allow_patterns=None,
            snapshot_download_fn=download_snapshot,
            hf_hub_download_fn=None,
            verify_diffusers_model_complete_fn=lambda path: os.path.isfile(
                os.path.join(path, "model_index.json")
            ),
            base_model_download_fn=cached_snapshot,
        )

    original = Path(resolve())
    assert (original / "transformer/model.safetensors").read_text() == "original"
    # A new source revision must not reuse weights materialized from the old one.
    spec["source_revision"] = "newer"
    newer = Path(resolve())
    assert (newer / "transformer/model.safetensors").read_text() == "newer"
    assert original != newer


def test_registry_source_revision_survives_upstream_removing_required_file(
    tmp_path, monkeypatch
):
    """An overlay whose source repo later deleted a required file still resolves
    on a host with an empty cache, because both source downloads use the pin."""
    pinned = "4187f9a53c6eff3a76c51e79bd27f70d10f7591b"
    monkeypatch.setenv("SGLANG_DIFFUSION_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setattr(
        model_overlay,
        "_load_model_overlay_registry",
        lambda: {
            "org/source": {
                "overlay_repo_id": "org/source-overlay",
                "overlay_revision": "overlay-sha",
                "source_revision": pinned,
            }
        },
    )
    overlay_dir = tmp_path / "overlay"
    (overlay_dir / "_overlay").mkdir(parents=True)
    (overlay_dir / "_overlay" / "overlay_manifest.json").write_text(
        '{"source_model_id": "org/source",'
        ' "required_source_files": ["release.safetensors"]}'
    )

    def source_snapshot(revision):
        # Upstream main no longer ships the file; the pinned commit does.
        snapshot = tmp_path / f"source-{revision}"
        snapshot.mkdir(exist_ok=True)
        if revision == pinned:
            (snapshot / "release.safetensors").write_bytes(b"w")
        return str(snapshot)

    def snapshot_download_fn(*, repo_id, revision=None, **_):
        if repo_id == "org/source-overlay":
            return str(overlay_dir)
        return source_snapshot(revision)

    def base_model_download_fn(model_id, *, revision=None, **_):
        return source_snapshot(revision)

    resolved = model_overlay.maybe_resolve_overlay_model_path(
        "org/source",
        local_dir=None,
        download=True,
        allow_patterns=None,
        snapshot_download_fn=snapshot_download_fn,
        hf_hub_download_fn=lambda **_: None,
        verify_diffusers_model_complete_fn=lambda _: True,
        base_model_download_fn=base_model_download_fn,
    )

    assert os.path.isdir(resolved)
