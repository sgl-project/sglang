import json

from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    _check_index_files_for_missing_shards,
    _verify_diffusers_model_complete,
)


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
