from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.multimodal_gen.runtime.weights.source import (
    parse_weight_source,
    resolve_weight,
    resolve_weight_inventory,
)


def test_parse_weight_source_accepts_repo_subfolder_and_exact_url():
    subfolder = parse_weight_source("owner/repo/text_encoder", revision="v1")
    repo_file = parse_weight_source("owner/repo/adapter.safetensors")
    exact_file = parse_weight_source(
        "https://huggingface.co/owner/repo/resolve/main/weights/model.safetensors"
    )

    assert subfolder.repo_id == "owner/repo"
    assert subfolder.subfolder == "text_encoder"
    assert subfolder.revision == "v1"
    assert repo_file.filename == "adapter.safetensors"
    assert repo_file.subfolder is None
    assert exact_file.repo_id == "owner/repo"
    assert exact_file.revision == "main"
    assert exact_file.filename == "weights/model.safetensors"


def test_parse_weight_source_rejects_conflicting_url_revision():
    with pytest.raises(ValueError, match="conflicts with revision"):
        parse_weight_source(
            "https://huggingface.co/owner/repo/tree/main/transformer",
            revision="v2",
        )


def test_resolve_local_inventory_lists_files_without_loading_tensors(tmp_path):
    component = tmp_path / "component"
    component.mkdir()
    (component / "config.json").write_text("{}")
    (component / "model.safetensors").write_bytes(b"header-only-fixture")

    inventory = resolve_weight_inventory(parse_weight_source(str(component)))

    assert inventory.resolved_revision is None
    assert list(inventory.files) == [
        "config.json",
        "model.safetensors",
    ]


def test_resolve_remote_inventory_pins_revision_and_filters_subfolder():
    source = parse_weight_source("owner/repo/text_encoder", revision="main")
    model_info = SimpleNamespace(
        sha="immutable-sha",
        siblings=[
            SimpleNamespace(rfilename="text_encoder/config.json"),
            SimpleNamespace(rfilename="text_encoder/model.safetensors"),
            SimpleNamespace(rfilename="vae/config.json"),
        ],
    )

    with patch(
        "sglang.multimodal_gen.runtime.weights.source.HfApi.model_info",
        return_value=model_info,
    ):
        inventory = resolve_weight_inventory(source)

    assert inventory.resolved_revision == "immutable-sha"
    assert inventory.files == (
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
    )


def test_weight_source_rejects_ambiguous_files(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"a")
    (tmp_path / "b.safetensors").write_bytes(b"b")

    with pytest.raises(ValueError, match="multiple independent weight files"):
        resolve_weight(str(tmp_path))
