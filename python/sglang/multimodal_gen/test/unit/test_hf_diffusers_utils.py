import json

import modelscope
import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from sglang.multimodal_gen.runtime.utils import hf_diffusers_utils
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    _check_index_files_for_missing_shards,
    _verify_diffusers_model_complete,
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
