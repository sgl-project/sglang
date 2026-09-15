from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.multimodal_gen.runtime.weights.source import (
    NoSafetensorsWeightsError,
    is_explicit_weight_file_reference,
    materialize_weight,
    materialize_weight_set,
    materialize_weight_set_config,
    parse_weight_source,
    resolve_safetensors_weight_set,
    resolve_weight,
    resolve_weight_inventory,
)


def test_explicit_weight_file_reference_does_not_claim_directories(tmp_path):
    component = tmp_path / "component.safetensors"
    component.mkdir()

    assert not is_explicit_weight_file_reference(str(component))
    assert is_explicit_weight_file_reference("owner/repo/model.safetensors")
    assert not is_explicit_weight_file_reference(
        "owner/repo/model.safetensors.index.json"
    )
    assert is_explicit_weight_file_reference(
        "https://huggingface.co/owner/repo/resolve/main/model.gguf?download=true"
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


def test_safetensors_index_selects_only_declared_shards(tmp_path):
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"one")
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"two")
    (tmp_path / "alternate.safetensors").write_bytes(b"other variant")
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"a":"model-00001-of-00002.safetensors",'
        '"b":"model-00002-of-00002.safetensors"}}'
    )

    resolved = resolve_safetensors_weight_set(str(tmp_path))

    assert resolved.index_file == "model.safetensors.index.json"
    assert resolved.selected_files == (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    )
    assert materialize_weight_set(resolved) == tuple(
        str(tmp_path / filename) for filename in resolved.selected_files
    )


def test_exact_local_safetensors_index_resolves_adjacent_shards(tmp_path):
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"one")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text('{"weight_map":{"a":"model-00001-of-00001.safetensors"}}')

    resolved = resolve_safetensors_weight_set(str(index))

    assert resolved.index_file == index.name
    assert materialize_weight_set(resolved) == (str(shard),)


def test_safetensors_weight_set_rejects_unindexed_variants(tmp_path):
    (tmp_path / "base.safetensors").write_bytes(b"base")
    (tmp_path / "distilled.safetensors").write_bytes(b"distilled")

    with pytest.raises(ValueError, match="without an index"):
        resolve_safetensors_weight_set(str(tmp_path))


def test_safetensors_weight_set_prefers_canonical_precision_family(tmp_path):
    canonical = tmp_path / "model.safetensors"
    canonical.write_bytes(b"canonical")
    (tmp_path / "model.fp16.safetensors").write_bytes(b"fp16")

    resolved = resolve_safetensors_weight_set(str(tmp_path))

    assert resolved.selected_files == (canonical.name,)


def test_explicit_precision_variant_overrides_canonical_fallback(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"canonical")
    variant = tmp_path / "model.fp16.safetensors"
    variant.write_bytes(b"fp16")

    resolved = resolve_safetensors_weight_set(str(tmp_path), weight_name=variant.name)

    assert resolved.selected_files == (variant.name,)


def test_safetensors_weight_set_prefers_canonical_precision_index(tmp_path):
    canonical = tmp_path / "model.safetensors"
    variant = tmp_path / "model.fp16.safetensors"
    canonical.write_bytes(b"canonical")
    variant.write_bytes(b"fp16")
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"weight":"model.safetensors"}}'
    )
    (tmp_path / "model.fp16.safetensors.index.json").write_text(
        '{"weight_map":{"weight":"model.fp16.safetensors"}}'
    )

    resolved = resolve_safetensors_weight_set(str(tmp_path))

    assert resolved.index_file == "model.safetensors.index.json"
    assert resolved.selected_files == (canonical.name,)


def test_safetensors_index_rejects_non_weight_shard(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"a":"config.json"}}'
    )

    with pytest.raises(ValueError, match="non-safetensors shard"):
        resolve_safetensors_weight_set(str(tmp_path))


def test_safetensors_index_missing_shard_is_not_no_weights_fallback(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"a":"missing.safetensors"}}'
    )

    with pytest.raises(FileNotFoundError, match="missing shard") as error:
        resolve_safetensors_weight_set(str(tmp_path))

    assert not isinstance(error.value, NoSafetensorsWeightsError)


def test_source_without_safetensors_has_distinct_error(tmp_path):
    (tmp_path / "pytorch_model.bin").write_bytes(b"bin")

    with pytest.raises(NoSafetensorsWeightsError):
        resolve_safetensors_weight_set(str(tmp_path))


def test_exact_non_safetensors_file_has_distinct_error(tmp_path):
    weights = tmp_path / "pytorch_model.bin"
    weights.write_bytes(b"bin")

    with pytest.raises(NoSafetensorsWeightsError):
        resolve_safetensors_weight_set(str(weights))


def test_remote_safetensors_shards_use_one_pinned_revision(tmp_path):
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        '{"weight_map":{"a":"model-00001-of-00002.safetensors",'
        '"b":"model-00002-of-00002.safetensors"}}'
    )
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    model_info = SimpleNamespace(
        sha="immutable-sha",
        siblings=[
            SimpleNamespace(rfilename=f"transformer/{filename}")
            for filename in (
                "model.safetensors.index.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "config.json",
                "quant_model_description_w8a8.json",
            )
        ],
    )

    def download(*, filename, **_kwargs):
        if filename.endswith("index.json"):
            return str(index_path)
        if filename.endswith("config.json"):
            return str(config_path)
        if "quant_model_description" in filename:
            return str(tmp_path / "quant_model_description_w8a8.json")
        return f"/{filename}"

    with (
        patch(
            "sglang.multimodal_gen.runtime.weights.source.HfApi.model_info",
            return_value=model_info,
        ),
        patch(
            "sglang.multimodal_gen.runtime.weights.source.hf_hub_download",
            side_effect=download,
        ) as hub_download,
    ):
        resolved = resolve_safetensors_weight_set("owner/repo/transformer")
        materialize_weight_set(resolved)
        assert materialize_weight_set_config(resolved) == str(config_path)

    assert all(
        call.kwargs["revision"] == "immutable-sha"
        for call in hub_download.call_args_list
    )
    assert "transformer/quant_model_description_w8a8.json" in {
        call.kwargs["filename"] for call in hub_download.call_args_list
    }


def test_materialize_local_weight_returns_selected_file(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"fixture")

    assert materialize_weight(resolve_weight(str(checkpoint))) == str(checkpoint)
