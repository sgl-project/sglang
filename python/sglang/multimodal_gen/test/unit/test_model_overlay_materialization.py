import json
from pathlib import Path

from sglang.multimodal_gen.runtime.utils import hf_diffusers_utils as hf_utils


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _touch_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_overlay_model_index_resolution_and_materialization(monkeypatch, tmp_path):
    source_model_id = "Wan-AI/Wan2.2-TI2V-5B"
    overlay_repo_id = "your-org/Wan2.2-TI2V-5B-overlay"

    cache_root = tmp_path / "cache-root"
    source_repo = tmp_path / "source-repo"
    overlay_repo = tmp_path / "overlay-repo"

    _touch_text(
        source_repo / "transformer" / "diffusion_pytorch_model.safetensors",
        "transformer-weights",
    )
    _touch_text(
        source_repo / "vae" / "diffusion_pytorch_model.safetensors",
        "vae-weights",
    )

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
    _write_json(
        overlay_repo / "_overlay" / "overlay_manifest.json",
        {
            "source_model_id": source_model_id,
            "materializer_version": "test-v1",
            "file_mappings": [
                {"type": "tree", "src": "transformer", "dst_dir": "transformer"},
                {"type": "tree", "src": "vae", "dst_dir": "vae"},
            ],
        },
    )

    monkeypatch.setenv(
        "SGLANG_DIFFUSION_MODEL_OVERLAY_REGISTRY",
        json.dumps(
            {
                source_model_id: {
                    "overlay_repo_id": overlay_repo_id,
                    "overlay_revision": "main",
                }
            }
        ),
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

    model_index = hf_utils.maybe_download_model_index(source_model_id)
    assert model_index["_class_name"] == "WanPipeline"
    assert snapshot_calls[0]["repo_id"] == overlay_repo_id
    assert snapshot_calls[0]["allow_patterns"] is not None
    assert all(
        "safetensors" not in pattern for pattern in snapshot_calls[0]["allow_patterns"]
    )

    materialized_dir = hf_utils.maybe_download_model(
        source_model_id, force_diffusers_model=True
    )
    materialized_path = Path(materialized_dir)
    assert (materialized_path / "model_index.json").exists()
    assert (materialized_path / "transformer" / "config.json").exists()
    assert (
        materialized_path / "transformer" / "diffusion_pytorch_model.safetensors"
    ).read_text(encoding="utf-8") == "transformer-weights"
    assert (
        materialized_path / "vae" / "diffusion_pytorch_model.safetensors"
    ).read_text(encoding="utf-8") == "vae-weights"
    assert (materialized_path / ".sglang_overlay_materialized.json").exists()

    second_dir = hf_utils.maybe_download_model(
        source_model_id, force_diffusers_model=True
    )
    assert second_dir == materialized_dir

    source_calls = [
        call for call in snapshot_calls if call["repo_id"] == source_model_id
    ]
    overlay_calls = [
        call for call in snapshot_calls if call["repo_id"] == overlay_repo_id
    ]
    assert source_calls, "source repo should be consulted for weights"
    assert overlay_calls, "overlay repo should be consulted for metadata"


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
    _touch_text(
        overlay_repo / "transformer" / "diffusion_pytorch_model.safetensors",
        "transformer-weights",
    )
    _touch_text(
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
            if kwargs.get("allow_patterns") is not None:
                metadata_dir = tmp_path / "overlay-metadata"
                _write_json(
                    metadata_dir / "model_index.json",
                    json.loads(
                        (overlay_repo / "model_index.json").read_text(encoding="utf-8")
                    ),
                )
                _write_json(
                    metadata_dir / "transformer" / "config.json",
                    {"_class_name": "WanTransformer3DModel"},
                )
                _write_json(
                    metadata_dir / "vae" / "config.json",
                    {"_class_name": "AutoencoderKLWan"},
                )
                return str(metadata_dir)
            return str(overlay_repo)
        raise AssertionError(f"unexpected repo_id: {repo_id}")

    monkeypatch.setattr(hf_utils, "snapshot_download", fake_snapshot_download)

    materialized_dir = hf_utils.maybe_download_model(
        source_model_id, force_diffusers_model=True
    )
    assert materialized_dir == str(overlay_repo)
    assert len(snapshot_calls) == 2
    assert snapshot_calls[0]["repo_id"] == overlay_repo_id
    assert snapshot_calls[0]["allow_patterns"] is not None
    assert snapshot_calls[1]["repo_id"] == overlay_repo_id
    assert snapshot_calls[1]["allow_patterns"] is None


def test_overlay_custom_materializer(monkeypatch, tmp_path):
    source_model_id = "Wan-AI/Wan2.2-S2V-14B"
    overlay_repo_id = "your-org/Wan2.2-S2V-14B-overlay"
    cache_root = tmp_path / "cache-root"
    source_repo = tmp_path / "source-repo"
    overlay_repo = tmp_path / "overlay-repo"

    _touch_text(
        source_repo / "official_repo" / "wan" / "__init__.py",
        "__all__ = []\n",
    )
    _touch_text(
        source_repo / "checkpoints" / "weights.bin",
        "fake-weights",
    )

    _write_json(
        overlay_repo / "model_index.json",
        {
            "_class_name": "WanSpeechToVideoPipeline",
            "_diffusers_version": "0.35.0",
            "transformer": ["diffusers", "WanS2VOfficialEngine"],
        },
    )
    _write_json(
        overlay_repo / "transformer" / "config.json",
        {
            "_class_name": "WanS2VOfficialEngine",
            "wan_code_root": "official_code",
            "wan_checkpoint_root": "checkpoints",
        },
    )
    _write_json(
        overlay_repo / "_overlay" / "overlay_manifest.json",
        {
            "source_model_id": source_model_id,
            "materializer_version": "test-s2v-v1",
            "custom_materializer": "_overlay/materialize.py",
        },
    )
    _touch_text(
        overlay_repo / "_overlay" / "materialize.py",
        "\n".join(
            [
                "import os",
                "from pathlib import Path",
                "import shutil",
                "",
                "def materialize(*, overlay_dir, source_dir, output_dir, manifest):",
                "    transformer_dir = Path(output_dir) / 'transformer'",
                "    transformer_dir.mkdir(parents=True, exist_ok=True)",
                "    shutil.copytree(Path(source_dir) / 'official_repo', transformer_dir / 'official_code', dirs_exist_ok=True)",
                "    shutil.copytree(Path(source_dir) / 'checkpoints', transformer_dir / 'checkpoints', dirs_exist_ok=True)",
            ]
        ),
    )

    monkeypatch.setenv(
        "SGLANG_DIFFUSION_MODEL_OVERLAY_REGISTRY",
        json.dumps({source_model_id: {"overlay_repo_id": overlay_repo_id}}),
    )
    monkeypatch.setenv("SGLANG_DIFFUSION_CACHE_ROOT", str(cache_root))
    hf_utils.clear_model_overlay_registry_cache()

    def fake_snapshot_download(repo_id, **kwargs):
        if repo_id == overlay_repo_id:
            return str(overlay_repo)
        if repo_id == source_model_id:
            return str(source_repo)
        raise AssertionError(f"unexpected repo_id: {repo_id}")

    monkeypatch.setattr(hf_utils, "snapshot_download", fake_snapshot_download)

    materialized_dir = hf_utils.maybe_download_model(
        source_model_id, force_diffusers_model=True
    )
    materialized_path = Path(materialized_dir)
    assert (materialized_path / "model_index.json").exists()
    assert (materialized_path / "transformer" / "config.json").exists()
    assert (
        materialized_path / "transformer" / "official_code" / "wan" / "__init__.py"
    ).exists()
    assert (
        materialized_path / "transformer" / "checkpoints" / "weights.bin"
    ).read_text(encoding="utf-8") == "fake-weights"


def test_overlay_repo_can_be_used_directly_as_model_path(monkeypatch, tmp_path):
    source_model_id = "Wan-AI/Wan2.2-S2V-14B"
    overlay_repo_id = "your-org/Wan2.2-S2V-14B-overlay"
    cache_root = tmp_path / "cache-root"
    source_repo = tmp_path / "source-repo"
    overlay_repo = tmp_path / "overlay-repo"

    _touch_text(source_repo / "checkpoints" / "weights.bin", "fake-weights")
    _touch_text(
        source_repo / "official_repo" / "wan" / "__init__.py",
        "__all__ = []\n",
    )
    _write_json(
        overlay_repo / "model_index.json",
        {
            "_class_name": "WanSpeechToVideoPipeline",
            "_diffusers_version": "0.35.0",
            "transformer": ["diffusers", "WanS2VTransformer3DModel"],
            "scheduler": ["diffusers", "WanS2VOfficialScheduler"],
        },
    )
    _write_json(
        overlay_repo / "transformer" / "config.json",
        {"_class_name": "WanS2VTransformer3DModel"},
    )
    _write_json(
        overlay_repo / "scheduler" / "scheduler_config.json",
        {"_class_name": "WanS2VOfficialScheduler"},
    )
    _write_json(
        overlay_repo / "_overlay" / "overlay_manifest.json",
        {
            "source_model_id": source_model_id,
            "materializer_version": "overlay-direct-v1",
            "source_allow_patterns": [
                "checkpoints/weights.bin",
                "official_repo/wan/__init__.py",
            ],
            "custom_materializer": "_overlay/materialize.py",
        },
    )
    _touch_text(
        overlay_repo / "_overlay" / "materialize.py",
        "\n".join(
            [
                "from pathlib import Path",
                "import shutil",
                "",
                "def materialize(*, overlay_dir, source_dir, output_dir, manifest):",
                "    transformer_dir = Path(output_dir) / 'transformer'",
                "    transformer_dir.mkdir(parents=True, exist_ok=True)",
                "    shutil.copytree(Path(source_dir) / 'official_repo', transformer_dir / 'official_code', dirs_exist_ok=True)",
                "    shutil.copytree(Path(source_dir) / 'checkpoints', transformer_dir / 'checkpoints', dirs_exist_ok=True)",
            ]
        ),
    )

    monkeypatch.setenv("SGLANG_DIFFUSION_CACHE_ROOT", str(cache_root))

    hub_download_calls: list[dict] = []
    snapshot_calls: list[dict] = []

    def fake_hf_hub_download(repo_id, filename, local_dir=None, **kwargs):
        hub_download_calls.append(
            {
                "repo_id": repo_id,
                "filename": filename,
                "local_dir": local_dir,
                **kwargs,
            }
        )
        assert repo_id == overlay_repo_id
        assert filename == "_overlay/overlay_manifest.json"
        return str(overlay_repo / "_overlay" / "overlay_manifest.json")

    def fake_snapshot_download(repo_id, **kwargs):
        snapshot_calls.append({"repo_id": repo_id, **kwargs})
        if repo_id == source_model_id:
            return str(source_repo)
        raise AssertionError(f"unexpected repo_id: {repo_id}")

    monkeypatch.setattr(hf_utils, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(hf_utils, "snapshot_download", fake_snapshot_download)

    model_index = hf_utils.maybe_download_model_index(overlay_repo_id)
    assert model_index["_class_name"] == "WanSpeechToVideoPipeline"

    materialized_dir = hf_utils.maybe_download_model(
        overlay_repo_id, force_diffusers_model=True
    )
    materialized_path = Path(materialized_dir)
    assert (materialized_path / "model_index.json").exists()
    assert (
        materialized_path / "transformer" / "checkpoints" / "weights.bin"
    ).read_text(encoding="utf-8") == "fake-weights"
    assert (
        materialized_path / "transformer" / "official_code" / "wan" / "__init__.py"
    ).exists()
    assert (
        hub_download_calls
    ), "overlay manifest should be discovered via hf_hub_download"
    assert snapshot_calls and snapshot_calls[0]["repo_id"] == source_model_id
    assert snapshot_calls[0]["allow_patterns"] == [
        "checkpoints/weights.bin",
        "official_repo/wan/__init__.py",
    ]
