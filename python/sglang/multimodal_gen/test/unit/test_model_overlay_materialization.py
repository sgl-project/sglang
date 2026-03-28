import importlib.util
import json
import logging
import sys
import types
from pathlib import Path

import pytest


def _load_overlay_module():
    module_name = "test_model_overlay_module"
    if module_name in sys.modules:
        return sys.modules[module_name]

    class _Lock:
        def acquire(self, poll_interval=2):
            class _Ctx:
                def __enter__(self_inner):
                    return None

                def __exit__(self_inner, exc_type, exc, tb):
                    return False

            return _Ctx()

    def _get_lock(_name):
        return _Lock()

    def _init_logger(name):
        return logging.getLogger(name)

    # Stub only the internal modules that model_overlay.py imports.
    pkg_names = [
        "sglang",
        "sglang.multimodal_gen",
        "sglang.multimodal_gen.runtime",
        "sglang.multimodal_gen.runtime.loader",
        "sglang.multimodal_gen.runtime.utils",
        "sglang.multimodal_gen.test",
        "sglang.multimodal_gen.test.unit",
    ]
    for pkg_name in pkg_names:
        module = sys.modules.setdefault(pkg_name, types.ModuleType(pkg_name))
        module.__path__ = []  # mark as package

    weight_utils = types.ModuleType(
        "sglang.multimodal_gen.runtime.loader.weight_utils"
    )
    weight_utils.get_lock = _get_lock
    logging_utils = types.ModuleType(
        "sglang.multimodal_gen.runtime.utils.logging_utils"
    )
    logging_utils.init_logger = _init_logger

    sys.modules.setdefault(
        "sglang.multimodal_gen.runtime.loader.weight_utils", weight_utils
    )
    sys.modules.setdefault(
        "sglang.multimodal_gen.runtime.utils.logging_utils", logging_utils
    )

    module_path = (
        Path(__file__).resolve().parents[2] / "runtime" / "utils" / "model_overlay.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


hf_utils = _load_overlay_module()


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
    base_download_calls: list[str] = []

    def fake_snapshot_download(repo_id, **kwargs):
        snapshot_calls.append({"repo_id": repo_id, **kwargs})
        if repo_id == overlay_repo_id:
            return str(overlay_repo)
        if repo_id == source_model_id:
            return str(source_repo)
        raise AssertionError(f"unexpected repo_id: {repo_id}")

    def fake_hf_hub_download(**_kwargs):
        raise AssertionError("hf_hub_download should not be used in this case")

    source_input = str(source_repo) if use_local_source_path else source_model_id

    if not use_local_source_path:
        model_index = hf_utils.maybe_load_overlay_model_index(
            source_input,
            snapshot_download_fn=fake_snapshot_download,
            hf_hub_download_fn=fake_hf_hub_download,
        )
        assert model_index["_class_name"] == "WanPipeline"

    materialized_dir = hf_utils.maybe_resolve_overlay_model_path(
        source_input,
        local_dir=None,
        download=True,
        allow_patterns=None,
        snapshot_download_fn=fake_snapshot_download,
        hf_hub_download_fn=fake_hf_hub_download,
        verify_diffusers_model_complete_fn=lambda path: (
            Path(path) / "model_index.json"
        ).exists(),
        base_model_download_fn=lambda model_name_or_path, **kwargs: (
            base_download_calls.append(model_name_or_path) or str(source_repo)
            if model_name_or_path == source_model_id
            else (_ for _ in ()).throw(
                AssertionError(f"unexpected model download: {model_name_or_path}")
            )
        ),
    )
    assert materialized_dir is not None
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
        assert base_download_calls == []
    else:
        assert base_download_calls == [source_model_id]


def test_full_diffusers_overlay_repo_passthrough(monkeypatch, tmp_path):
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

    def fake_snapshot_download(repo_id, **kwargs):
        assert repo_id == overlay_repo_id
        return str(overlay_repo)

    overlay_path = hf_utils.maybe_resolve_overlay_model_path(
        source_model_id,
        local_dir=None,
        download=True,
        allow_patterns=None,
        snapshot_download_fn=fake_snapshot_download,
        hf_hub_download_fn=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("hf_hub_download should not be used")
        ),
        verify_diffusers_model_complete_fn=lambda _path: False,
        base_model_download_fn=lambda model_name_or_path, **kwargs: (
            str(overlay_repo)
            if model_name_or_path == overlay_repo_id
            else (_ for _ in ()).throw(
                AssertionError(f"unexpected model download: {model_name_or_path}")
            )
        ),
    )
    assert overlay_path == str(overlay_repo)


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

    def fake_hf_hub_download(**_kwargs):
        raise AssertionError("hf_hub_download should not be used for local overlays")

    materialized_dir = hf_utils.maybe_resolve_overlay_model_path(
        str(overlay_repo),
        local_dir=None,
        download=True,
        allow_patterns=None,
        snapshot_download_fn=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("snapshot_download should not be used for local overlays")
        ),
        hf_hub_download_fn=fake_hf_hub_download,
        verify_diffusers_model_complete_fn=lambda path: (
            Path(path) / "model_index.json"
        ).exists(),
        base_model_download_fn=lambda model_name_or_path, **kwargs: (
            str(source_repo)
            if model_name_or_path == source_model_id
            else (_ for _ in ()).throw(
                AssertionError(f"unexpected model download: {model_name_or_path}")
            )
        ),
    )
    assert materialized_dir is not None
    materialized_path = Path(materialized_dir)
    assert (materialized_path / "model_index.json").exists()
    assert (
        materialized_path / "transformer" / "diffusion_pytorch_model.safetensors"
    ).read_text(encoding="utf-8") == "transformer-weights"
