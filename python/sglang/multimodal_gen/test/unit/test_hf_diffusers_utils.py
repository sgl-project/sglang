import json

import modelscope
import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from sglang.multimodal_gen.runtime.utils import hf_diffusers_utils
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    _check_index_files_for_missing_shards,
    _is_revisionless_snapshot_root,
    _verify_diffusers_model_complete,
    maybe_download_model,
)
from sglang.srt.environ import envs


def _write_model_index(root):
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "LongLive2Pipeline",
                "_diffusers_version": "0.34.0",
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                "text_encoder": ["transformers", "T5EncoderModel"],
                "tokenizer": ["transformers", "T5TokenizerFast"],
                "transformer": ["diffusers", "LongLive2Transformer3DModel"],
                "transformer_2": [None, None],
                "vae": ["diffusers", "AutoencoderKLWan"],
            }
        )
    )


def test_diffusers_cache_validation_rejects_declared_component_without_weights(
    tmp_path,
):
    _write_model_index(tmp_path)
    for subdir in ("scheduler", "text_encoder", "tokenizer", "transformer", "vae"):
        (tmp_path / subdir).mkdir()
    (tmp_path / "text_encoder" / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "vae" / "diffusion_pytorch_model.bin").write_bytes(b"weights")

    assert not _verify_diffusers_model_complete(str(tmp_path))

    is_valid, missing_files, checked_subdirs = _check_index_files_for_missing_shards(
        str(tmp_path)
    )
    assert not is_valid
    assert "transformer/<weights>" in missing_files
    assert "transformer" in checked_subdirs


def test_diffusers_cache_validation_checks_declared_component_shards(tmp_path):
    _write_model_index(tmp_path)
    for subdir in ("scheduler", "text_encoder", "tokenizer", "transformer", "vae"):
        (tmp_path / subdir).mkdir()
        (tmp_path / subdir / "model.safetensors").write_bytes(b"weights")

    index_path = (
        tmp_path / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
    )
    index_path.write_text(
        json.dumps(
            {
                "weight_map": {
                    "block.0.weight": "model.safetensors",
                    "block.1.weight": "missing.safetensors",
                }
            }
        )
    )

    assert _verify_diffusers_model_complete(str(tmp_path))

    is_valid, missing_files, checked_subdirs = _check_index_files_for_missing_shards(
        str(tmp_path)
    )
    assert not is_valid
    assert "transformer/missing.safetensors" in missing_files
    assert "transformer" in checked_subdirs


def _populate_components(root, components, *, weights=True):
    for name in components:
        (root / name).mkdir(parents=True, exist_ok=True)
        if weights:
            (root / name / "model.safetensors").write_bytes(b"weights")


@pytest.fixture
def recording_snapshot_download(monkeypatch):
    """Stub ``snapshot_download`` to return ``path``, recording each call as an
    offline ``"probe"`` or a real ``"download"``."""
    calls = []

    def factory(path):
        def fake_snapshot_download(**kwargs):
            calls.append("probe" if kwargs.get("local_files_only") else "download")
            return str(path)

        monkeypatch.setattr(
            hf_diffusers_utils, "snapshot_download", fake_snapshot_download
        )
        return calls

    return factory


@pytest.mark.parametrize(
    "relative_path, expected",
    [
        ("models--org--repo/snapshots", True),
        ("models--org--repo/snapshots/", True),
        ("datasets--org--repo/snapshots", True),
        # A healthy entry resolves to snapshots/<commit_sha>.
        ("models--org--repo/snapshots/" + "a" * 40, False),
        # A local_dir may legitimately be named "snapshots".
        ("mymodels/snapshots", False),
        ("mymodels/snapshots/inner", False),
        ("models--org--repo", False),
    ],
)
def test_revisionless_snapshot_root_detection(tmp_path, relative_path, expected):
    assert _is_revisionless_snapshot_root(str(tmp_path / relative_path)) is expected


def test_corrupt_ref_resolving_to_snapshots_parent_is_a_cache_miss(
    recording_snapshot_download, tmp_path
):
    """The snapshots/ parent holds only revision subdirs, so it is not a hit."""
    snapshots_parent = tmp_path / "models--org--repo" / "snapshots"
    (snapshots_parent / ("a" * 40)).mkdir(parents=True)
    calls = recording_snapshot_download(snapshots_parent)

    with pytest.raises(ValueError, match="not found in local cache"):
        maybe_download_model("org/repo", download=False)

    assert calls == ["probe"]


def test_local_dir_named_snapshots_is_not_treated_as_corrupt(
    recording_snapshot_download, tmp_path
):
    """The guard requires a models--*/ folder above, so this stays a valid hit."""
    local_dir = tmp_path / "mymodels" / "snapshots"
    local_dir.mkdir(parents=True)
    _write_model_index(local_dir)
    _populate_components(local_dir, ("text_encoder", "transformer", "vae"))
    calls = recording_snapshot_download(local_dir)

    result = maybe_download_model("org/repo", local_dir=str(local_dir), download=False)

    assert result == str(local_dir)
    assert calls == ["probe"]


def test_local_path_is_returned_as_given(tmp_path):
    """A local path is not a repo id, so it is returned without any download."""
    _write_model_index(tmp_path)
    _populate_components(tmp_path, ("text_encoder", "transformer", "vae"))

    assert maybe_download_model(str(tmp_path), download=False) == str(tmp_path)


def test_metadata_only_local_path_is_still_returned_as_given(tmp_path):
    """Falling through would download the filesystem path as a repo id and fail."""
    _write_model_index(tmp_path)

    assert maybe_download_model(str(tmp_path), download=False) == str(tmp_path)


def test_metadata_only_cached_snapshot_is_not_a_usable_hit(
    recording_snapshot_download, tmp_path
):
    """The probe stub resolves as an offline hit but has no component weights."""
    _write_model_index(tmp_path)
    calls = recording_snapshot_download(tmp_path)

    with pytest.raises(ValueError, match="only contains pipeline metadata"):
        maybe_download_model("org/repo", download=False)

    assert calls == ["probe"]


def test_metadata_only_cached_snapshot_falls_through_to_one_download(
    recording_snapshot_download, tmp_path
):
    """Exactly one download -- no duplicate fetch, and no force_download retry."""
    _write_model_index(tmp_path)
    calls = recording_snapshot_download(tmp_path)

    assert maybe_download_model("org/repo") == str(tmp_path)
    assert calls == ["probe", "download"]


def test_complete_cached_snapshot_is_served_without_download(
    recording_snapshot_download, tmp_path
):
    _write_model_index(tmp_path)
    _populate_components(tmp_path, ("text_encoder", "transformer", "vae"))
    calls = recording_snapshot_download(tmp_path)

    assert maybe_download_model("org/repo") == str(tmp_path)
    assert calls == ["probe"]


def test_partially_populated_cached_snapshot_is_served_without_download(
    recording_snapshot_download, tmp_path
):
    """An allow_patterns fetch leaves components absent, so only a total absence of
    weights counts as the stub."""
    _write_model_index(tmp_path)
    _populate_components(tmp_path, ("vae",))
    calls = recording_snapshot_download(tmp_path)

    assert maybe_download_model("org/repo") == str(tmp_path)
    assert calls == ["probe"]


def test_cached_component_repo_without_model_index_is_served_without_download(
    recording_snapshot_download, tmp_path
):
    """Declares nothing, so the stub check must not match on 0 missing == 0."""
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    calls = recording_snapshot_download(tmp_path)

    assert maybe_download_model("org/repo") == str(tmp_path)
    assert calls == ["probe"]


def test_metadata_only_cached_lora_snapshot_is_a_usable_hit(
    recording_snapshot_download, tmp_path
):
    """LoRA repos declare no components, so they can never be the stub."""
    _write_model_index(tmp_path)
    calls = recording_snapshot_download(tmp_path)

    assert maybe_download_model("org/repo", is_lora=True) == str(tmp_path)
    assert calls == ["probe"]


def test_force_diffusers_model_stub_keeps_its_existing_path(
    recording_snapshot_download, tmp_path
):
    """Already rejected by _verify_diffusers_model_complete, so its path is
    unchanged: download, then the force_download retry."""
    _write_model_index(tmp_path)
    calls = recording_snapshot_download(tmp_path)

    with pytest.raises(ValueError, match="still incomplete after forced re-download"):
        maybe_download_model("org/repo", force_diffusers_model=True)

    assert calls == ["probe", "download", "download"]


def test_modelscope_file_download_preserves_local_dir(monkeypatch, tmp_path):
    calls = []

    def model_file_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / kwargs["file_path"])

    monkeypatch.setattr(modelscope, "model_file_download", model_file_download)

    with envs.SGLANG_USE_MODELSCOPE.override(True):
        result = hf_diffusers_utils.hf_hub_download(
            "MiniMax/MiniMax-H3",
            "FL2VA/model_index.json",
            local_dir=tmp_path,
            revision="master",
        )

    assert result == str(tmp_path / "FL2VA/model_index.json")
    assert calls == [
        {
            "model_id": "MiniMax/MiniMax-H3",
            "file_path": "FL2VA/model_index.json",
            "local_dir": str(tmp_path),
            "revision": "master",
        }
    ]


def test_modelscope_snapshot_download_selects_h3_partition(monkeypatch, tmp_path):
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(modelscope, "snapshot_download", snapshot_download)

    with envs.SGLANG_USE_MODELSCOPE.override(True):
        result = hf_diffusers_utils.snapshot_download(
            "MiniMax/MiniMax-H3",
            local_dir=tmp_path,
            allow_patterns=["Ref2VA/**"],
            force_download=True,
        )

    assert result == str(tmp_path)
    assert calls == [
        {
            "model_id": "MiniMax/MiniMax-H3",
            "local_dir": str(tmp_path),
            "ignore_patterns": None,
            "allow_patterns": ["Ref2VA/**"],
            "local_files_only": False,
            "max_workers": 8,
        }
    ]


def test_modelscope_empty_selected_partition_is_a_cache_miss(monkeypatch, tmp_path):
    monkeypatch.setattr(modelscope, "snapshot_download", lambda **_: str(tmp_path))

    with (
        envs.SGLANG_USE_MODELSCOPE.override(True),
        pytest.raises(LocalEntryNotFoundError, match="Ref2VA"),
    ):
        hf_diffusers_utils.snapshot_download(
            "MiniMax/MiniMax-H3",
            allow_patterns=["Ref2VA/**"],
            local_files_only=True,
        )


def test_modelscope_selected_partition_cache_hit_requires_a_file(monkeypatch, tmp_path):
    model_index = tmp_path / "FL2VA" / "model_index.json"
    model_index.parent.mkdir()
    model_index.write_text("{}")
    monkeypatch.setattr(modelscope, "snapshot_download", lambda **_: str(tmp_path))

    with envs.SGLANG_USE_MODELSCOPE.override(True):
        result = hf_diffusers_utils.snapshot_download(
            "MiniMax/MiniMax-H3",
            allow_patterns=["FL2VA/**"],
            local_files_only=True,
        )

    assert result == str(tmp_path)
