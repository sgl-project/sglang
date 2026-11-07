import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.srt.utils.model_file_verifier import IntegrityError, generate_checksums

_BASE = "sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base"


def _make_repo(root: Path) -> None:
    (root / "transformer").mkdir(parents=True)
    (root / "vae").mkdir()
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": "FakePipeline", "transformer": ["d", "T"]})
    )
    (root / "transformer" / "weights.bin").write_text("transformer weights")
    (root / "vae" / "weights.bin").write_text("vae weights")


def _pipeline(model_path: str, **server_args) -> SimpleNamespace:
    return SimpleNamespace(
        model_path=model_path,
        default_model_subfolder=None,
        server_args=SimpleNamespace(
            model_subfolder=None, revision=None, model_checksum=None, **server_args
        ),
    )


def test_checksum_mismatch_fails_before_any_component_is_loaded(tmp_path):
    repo = tmp_path / "repo"
    _make_repo(repo)
    manifest = tmp_path / "checksums.json"
    generate_checksums(source=str(repo), output_path=str(manifest), recursive=True)
    pipeline = _pipeline(str(repo))
    pipeline.server_args.model_checksum = str(manifest)

    with patch(f"{_BASE}.maybe_download_model", return_value=str(repo)):
        with patch(
            f"{_BASE}.verify_model_config_and_directory", return_value={"ok": True}
        ):
            assert ComposedPipelineBase._load_config(pipeline) == {"ok": True}

    (repo / "vae" / "weights.bin").write_text("tampered")
    with patch(f"{_BASE}.maybe_download_model", return_value=str(repo)):
        with patch(f"{_BASE}.verify_model_config_and_directory") as load_config:
            with pytest.raises(IntegrityError, match="vae/weights.bin"):
                ComposedPipelineBase._load_config(pipeline)
    load_config.assert_not_called()


@pytest.mark.parametrize("subfolder", ["transformer", None])
def test_checksum_is_rejected_where_it_cannot_cover_the_load(subfolder):
    pipeline = _pipeline("org/repo")
    pipeline.server_args.model_checksum = "checksums.json"
    pipeline.server_args.model_subfolder = subfolder
    pipeline.default_model_subfolder = None if subfolder else "transformer"

    with patch(f"{_BASE}.maybe_download_model") as download:
        with pytest.raises(ValueError, match="--model-checksum does not support"):
            ComposedPipelineBase._load_config(pipeline)
    download.assert_not_called()
