import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.loader.artifact_resolver import (
    ArtifactFile,
    ArtifactInventory,
    ArtifactRequest,
    parse_artifact_source,
    resolve_artifact,
    resolve_artifact_inventory,
)


def test_parse_artifact_source_accepts_repo_subfolder_and_exact_url():
    subfolder = parse_artifact_source("owner/repo/text_encoder", revision="v1")
    repo_file = parse_artifact_source("owner/repo/adapter.safetensors")
    exact_file = parse_artifact_source(
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


def test_parse_artifact_source_rejects_conflicting_url_revision():
    with pytest.raises(ValueError, match="conflicts with --revision"):
        parse_artifact_source(
            "https://huggingface.co/owner/repo/tree/main/transformer",
            revision="v2",
        )


def test_resolve_local_inventory_does_not_load_tensor_payloads(tmp_path):
    component = tmp_path / "component"
    component.mkdir()
    (component / "config.json").write_text("{}")
    (component / "model.safetensors").write_bytes(b"header-only-fixture")

    inventory = resolve_artifact_inventory(parse_artifact_source(str(component)))

    assert inventory.resolved_revision is None
    assert [item.path for item in inventory.files] == [
        "config.json",
        "model.safetensors",
    ]


def test_resolve_remote_inventory_pins_revision_and_filters_subfolder():
    source = parse_artifact_source("owner/repo/text_encoder", revision="main")
    model_info = SimpleNamespace(
        sha="immutable-sha",
        siblings=[
            SimpleNamespace(rfilename="text_encoder/config.json", size=12, blob_id="a"),
            SimpleNamespace(
                rfilename="text_encoder/model.safetensors", size=42, blob_id="b"
            ),
            SimpleNamespace(rfilename="vae/config.json", size=10, blob_id="c"),
        ],
    )

    with patch(
        "sglang.multimodal_gen.runtime.loader.artifact_resolver.HfApi.model_info",
        return_value=model_info,
    ):
        inventory = resolve_artifact_inventory(source)

    assert inventory.resolved_revision == "immutable-sha"
    assert inventory.files == (
        ArtifactFile(path="text_encoder/config.json", size=12, blob_id="a"),
        ArtifactFile(path="text_encoder/model.safetensors", size=42, blob_id="b"),
    )


def test_materialized_inventory_shape_is_serializable():
    source = parse_artifact_source("owner/repo")
    inventory = ArtifactInventory(
        source=source,
        resolved_revision="sha",
        files=(ArtifactFile(path="config.json", size=1),),
    )

    assert os.fspath(inventory.source.original) == "owner/repo"


def test_weights_only_artifact_rejects_ambiguous_files(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"a")
    (tmp_path / "b.safetensors").write_bytes(b"b")

    with pytest.raises(ValueError, match="multiple independent weight files"):
        resolve_artifact(
            ArtifactRequest(
                name="transformer",
                role="component_weights",
                component="transformer",
                source=str(tmp_path),
            )
        )


def test_lora_artifact_reports_tensor_and_quant_metadata(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"quantization_config": {"quant_method": "bitsandbytes"}}'
    )
    save_file(
        {
            "blocks.0.to_q.lora_A.weight": torch.zeros(4, 8),
            "blocks.0.to_q.lora_B.weight": torch.zeros(8, 4),
        },
        tmp_path / "adapter.safetensors",
        metadata={"sampler_steps": "4"},
    )

    artifact = resolve_artifact(
        ArtifactRequest(
            name="startup_lora",
            role="lora",
            source=str(tmp_path),
        )
    )

    assert artifact.selected_files == ("adapter.safetensors",)
    assert artifact.quantization_method == "bitsandbytes"
    assert artifact.quantization_source == "quantization_config"
    assert artifact.tensor_summary is not None
    assert artifact.tensor_summary.tensor_count == 2
    assert artifact.tensor_summary.lora_ranks == (4,)
    assert artifact.tensor_summary.metadata["sampler_steps"] == "4"
