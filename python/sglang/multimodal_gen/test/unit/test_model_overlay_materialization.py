import json
from pathlib import Path

import pytest

from sglang.multimodal_gen.runtime.utils import hf_diffusers_utils as hf_utils


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _build_metadata_overlay(
    overlay_repo: Path, *, source_model_id: str, custom_materializer: bool = False
) -> None:
    _write_json(
        overlay_repo / "model_index.json",
        {
            "_class_name": "WanPipeline",
            "_diffusers_version": "0.35.0",
            "transformer": ["diffusers", "WanTransformer3DModel"],
            "vae": ["diffusers", "AutoencoderKLWan"],
        },
    )
    _write_json(
        overlay_repo / "transformer" / "config.json",
        {"_class_name": "WanTransformer3DModel"},
    )
    _write_json(
        overlay_repo / "vae" / "config.json",
        {"_class_name": "AutoencoderKLWan"},
    )
    manifest = {
        "source_model_id": source_model_id,
        "materializer_version": "overlay-test-v1",
    }
    if custom_materializer:
        manifest["custom_materializer"] = "_overlay/materialize.py"
        _write_text(
            overlay_repo / "_overlay" / "materialize.py",
            "\n".join(
                [
                    "from pathlib import Path",
                    "import shutil",
                    "",
                    "def materialize(*, overlay_dir, source_dir, output_dir, manifest):",
                    "    out = Path(output_dir)",
                    "    shutil.copytree(Path(source_dir) / 'transformer', out / 'transformer', dirs_exist_ok=True)",
                    "    shutil.copytree(Path(source_dir) / 'vae', out / 'vae', dirs_exist_ok=True)",
                ]
            ),
        )
    else:
        manifest["file_mappings"] = [
            {"type": "tree", "src": "transformer", "dst_dir": "transformer"},
            {"type": "tree", "src": "vae", "dst_dir": "vae"},
        ]
    _write_json(overlay_repo / "_overlay" / "overlay_manifest.json", manifest)


def _build_source_repo(source_repo: Path) -> None:
    _write_text(
        source_repo / "transformer" / "diffusion_pytorch_model.safetensors",
        "transformer-weights",
    )
    _write_text(
        source_repo / "vae" / "diffusion_pytorch_model.safetensors",
        "vae-weights",
    )


@pytest.mark.parametrize("use_local_source_path", [False, True])
def test_overlay_supports_model_id_and_local_source_path(
    monkeypatch, tmp_path, use_local_source_path
):
    source_model_id = "Wan-AI/Wan2.2-S2V-14B"
    overlay_repo_id = "your-org/Wan2.2-S2V-14B-overlay"
    cache_root = tmp_path / "cache-root"
    source_repo = tmp_path / "Wan2.2-S2V-14B"
    overlay_repo = tmp_path / "overlay-repo"
    _build_source_repo(source_repo)
    _build_metadata_overlay(overlay_repo, source_model_id=source_model_id)

    monkeypatch.setenv(
        "SGLANG_DIFFUSION_MODEL_OVERLAY_REGISTRY",
        json.dumps({source_model_id: {"overlay_repo_id": overlay_repo_id}}),
    )
    monkeypatch.setenv("SGLANG_DIFFUSION_CACHE_ROOT", str(cache_root))
    hf_utils.clear_model_overlay_registry_cache()

    snapshot_calls: list[dict] = []

    def fake_snapshot_download(repo_id, **kwargs):
        snapshot_calls.append({"repo_id": repo_id, **kwargs})
        if repo_id == overlay_repo_id:
            return str(overlay_repo)
        if repo_id == source_model_id:
            return str(source_repo)
        raise AssertionError(f"unexpected repo_id: {repo_id}")

    monkeypatch.setattr(hf_utils, "snapshot_download", fake_snapshot_download)

    source_input = str(source_repo) if use_local_source_path else source_model_id
    model_index = hf_utils.maybe_download_model_index(source_input)
    assert model_index["_class_name"] == "WanPipeline"

    materialized_dir = hf_utils.maybe_download_model(
        source_input, force_diffusers_model=True
    )
    materialized_path = Path(materialized_dir)
    assert (materialized_path / "model_index.json").exists()
    assert (
        materialized_path / "transformer" / "diffusion_pytorch_model.safetensors"
    ).read_text(encoding="utf-8") == "transformer-weights"
    assert (
        materialized_path / "vae" / "diffusion_pytorch_model.safetensors"
    ).read_text(encoding="utf-8") == "vae-weights"

    overlay_calls = [c for c in snapshot_calls if c["repo_id"] == overlay_repo_id]
    assert overlay_calls
    if use_local_source_path:
        assert all(c["repo_id"] != source_model_id for c in snapshot_calls)
    else:
        assert any(c["repo_id"] == source_model_id for c in snapshot_calls)


def test_direct_diffusers_overlay_repo(monkeypatch, tmp_path):
    source_model_id = "Wan-AI/Wan2.2-TI2V-5B"
    overlay_repo_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    overlay_repo = tmp_path / "overlay-diffusers"

    _write_json(
        overlay_repo / "model_index.json",
        {
            "_class_name": "WanPipeline",
            "_diffusers_version": "0.35.0",
            "transformer": ["diffusers", "WanTransformer3DModel"],
            "vae": ["diffusers", "AutoencoderKLWan"],
        },
    )
    _write_json(
        overlay_repo / "transformer" / "config.json",
        {"_class_name": "WanTransformer3DModel"},
    )
    _write_json(
        overlay_repo / "vae" / "config.json",
        {"_class_name": "AutoencoderKLWan"},
    )
    _write_text(
        overlay_repo / "transformer" / "diffusion_pytorch_model.safetensors",
        "transformer-weights",
    )
    _write_text(
        overlay_repo / "vae" / "diffusion_pytorch_model.safetensors",
        "vae-weights",
    )

    monkeypatch.setenv(
        "SGLANG_DIFFUSION_MODEL_OVERLAY_REGISTRY",
        json.dumps({source_model_id: {"overlay_repo_id": overlay_repo_id}}),
    )
    hf_utils.clear_model_overlay_registry_cache()

    snapshot_calls: list[dict] = []

    def fake_snapshot_download(repo_id, **kwargs):
        snapshot_calls.append({"repo_id": repo_id, **kwargs})
        if repo_id == overlay_repo_id:
            metadata_dir = tmp_path / "overlay-metadata"
            if kwargs.get("allow_patterns") is not None:
                _write_json(
                    metadata_dir / "model_index.json",
                    json.loads(
                        (overlay_repo / "model_index.json").read_text(encoding="utf-8")
                    ),
                )
                return str(metadata_dir)
            return str(overlay_repo)
        raise AssertionError(f"unexpected repo_id: {repo_id}")

    monkeypatch.setattr(hf_utils, "snapshot_download", fake_snapshot_download)

    materialized_dir = hf_utils.maybe_download_model(
        source_model_id, force_diffusers_model=True
    )
    assert materialized_dir == str(overlay_repo)
    assert [c["repo_id"] for c in snapshot_calls] == [overlay_repo_id, overlay_repo_id]


def test_overlay_repo_can_be_used_directly_as_model_path(monkeypatch, tmp_path):
    source_model_id = "Wan-AI/Wan2.2-S2V-14B"
    cache_root = tmp_path / "cache-root"
    source_repo = tmp_path / "source-repo"
    overlay_repo = tmp_path / "overlay-repo"
    _build_source_repo(source_repo)
    _build_metadata_overlay(
        overlay_repo, source_model_id=source_model_id, custom_materializer=True
    )

    monkeypatch.setenv("SGLANG_DIFFUSION_CACHE_ROOT", str(cache_root))
    hf_utils.clear_model_overlay_registry_cache()

    def fake_hf_hub_download(repo_id, filename, **kwargs):
        assert repo_id == str(overlay_repo)
        assert filename == "_overlay/overlay_manifest.json"
        return str(overlay_repo / "_overlay" / "overlay_manifest.json")

    def fake_snapshot_download(repo_id, **kwargs):
        assert repo_id == source_model_id
        return str(source_repo)

    monkeypatch.setattr(hf_utils, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(hf_utils, "snapshot_download", fake_snapshot_download)

    model_index = hf_utils.maybe_download_model_index(str(overlay_repo))
    assert model_index["_class_name"] == "WanPipeline"

    materialized_dir = hf_utils.maybe_download_model(
        str(overlay_repo), force_diffusers_model=True
    )
    materialized_path = Path(materialized_dir)
    assert (materialized_path / "model_index.json").exists()
    assert (
        materialized_path / "transformer" / "diffusion_pytorch_model.safetensors"
    ).read_text(encoding="utf-8") == "transformer-weights"
