from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.checkpoint_inspection.resolver import (
    CheckpointFile,
    CheckpointRequest,
    parse_checkpoint_source,
    resolve_checkpoint,
    resolve_checkpoint_inventory,
)


def test_parse_checkpoint_source_accepts_repo_subfolder_and_exact_url():
    subfolder = parse_checkpoint_source("owner/repo/text_encoder", revision="v1")
    repo_file = parse_checkpoint_source("owner/repo/adapter.safetensors")
    exact_file = parse_checkpoint_source(
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


def test_parse_checkpoint_source_rejects_conflicting_url_revision():
    with pytest.raises(ValueError, match="conflicts with --revision"):
        parse_checkpoint_source(
            "https://huggingface.co/owner/repo/tree/main/transformer",
            revision="v2",
        )


def test_resolve_local_inventory_does_not_load_tensor_payloads(tmp_path):
    component = tmp_path / "component"
    component.mkdir()
    (component / "config.json").write_text("{}")
    (component / "model.safetensors").write_bytes(b"header-only-fixture")

    inventory = resolve_checkpoint_inventory(parse_checkpoint_source(str(component)))

    assert inventory.resolved_revision is None
    assert [item.path for item in inventory.files] == [
        "config.json",
        "model.safetensors",
    ]


def test_resolve_remote_inventory_pins_revision_and_filters_subfolder():
    source = parse_checkpoint_source("owner/repo/text_encoder", revision="main")
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
        "sglang.multimodal_gen.checkpoint_inspection.resolver.HfApi.model_info",
        return_value=model_info,
    ):
        inventory = resolve_checkpoint_inventory(source)

    assert inventory.resolved_revision == "immutable-sha"
    assert inventory.files == (
        CheckpointFile(path="text_encoder/config.json", size=12, blob_id="a"),
        CheckpointFile(path="text_encoder/model.safetensors", size=42, blob_id="b"),
    )


def test_weights_only_checkpoint_rejects_ambiguous_files(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"a")
    (tmp_path / "b.safetensors").write_bytes(b"b")

    with pytest.raises(ValueError, match="multiple independent weight files"):
        resolve_checkpoint(
            CheckpointRequest(
                name="transformer",
                role="component_weights",
                component="transformer",
                source=str(tmp_path),
            )
        )


def test_lora_checkpoint_reports_tensor_and_quant_metadata(tmp_path):
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

    checkpoint = resolve_checkpoint(
        CheckpointRequest(
            name="startup_lora",
            role="lora",
            source=str(tmp_path),
        )
    )

    assert checkpoint.selected_files == ("adapter.safetensors",)
    assert checkpoint.quantization_method == "bitsandbytes"
    assert checkpoint.quantization_source == "quantization_config"
    assert checkpoint.tensor_summary is not None
    assert checkpoint.tensor_summary.tensor_count == 2
    assert checkpoint.tensor_summary.lora_ranks == (4,)
    assert checkpoint.tensor_summary.metadata["sampler_steps"] == "4"
