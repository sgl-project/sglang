# SPDX-License-Identifier: Apache-2.0

import argparse
import fnmatch
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

import pytest

from sglang.srt.arg_groups.model_path_hook import handle_model_source_paths
from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.model_hub import parse_model_hub_reference, resolve_model_hub_paths
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture
def hubs(monkeypatch, tmp_path):
    for source in ("hf", "ms"):
        folder = tmp_path / source
        folder.mkdir()
        (folder / "model.safetensors").touch()
    hf, ms = (Mock(return_value=str(tmp_path / source)) for source in ("hf", "ms"))
    monkeypatch.setattr("huggingface_hub.snapshot_download", hf)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setitem(
        sys.modules,
        "modelscope.hub.snapshot_download",
        SimpleNamespace(snapshot_download=ms),
    )
    return {"hf": hf, "modelscope": ms}


@pytest.mark.parametrize("source", ["hf", "modelscope"])
def test_validation_and_legacy_paths(hubs, source):
    assert parse_model_hub_reference(f"{source}://Org/Model-1.5") == (
        source,
        "Org/Model-1.5",
    )
    for repo in (
        "",
        "owner",
        "owner/repo/file",
        "owner/repo@v1",
        "owner/repo?x=1",
        "owner/repo#v1",
        "../repo",
        "owner/repo.git",
        "owner/.repo",
        "owner/repo-",
    ):
        with pytest.raises(ValueError, match="Expected"):
            resolve_model_hub_paths(
                ServerArgs(
                    model_path="hf://Org/valid", tokenizer_path=f"{source}://{repo}"
                )
            )
    for path in (
        "owner/repo",
        "/models/repo",
        "./model",
        "s3://bucket/model",
        "gs://bucket/model",
        "az://bucket/model",
        "redis://host/model",
        "instance://host/model",
        "https://huggingface.co/Org/Model",
        "owner/repo:Q4_K_M",
    ):
        assert resolve_model_hub_paths(ServerArgs(model_path=path)) == {}
    for download in hubs.values():
        download.assert_not_called()


@pytest.mark.parametrize("global_source", ["true", "false"])
@pytest.mark.parametrize("source", ["hf", "modelscope"])
@pytest.mark.parametrize(
    "field", ["model_path", "tokenizer_path", "speculative_draft_model_path"]
)
def test_fields_override_global_source(hubs, monkeypatch, global_source, source, field):
    monkeypatch.setenv("SGLANG_USE_MODELSCOPE", global_source)
    args = ServerArgs(**{"model_path": "Org/target", field: f"{source}://Org/repo"})
    resolved = resolve_model_hub_paths(args)
    assert resolved[field] == hubs[source].return_value
    if field == "model_path":
        assert resolved["tokenizer_path"] == resolved["model_path"]
        assert resolved["served_model_name"] == "Org/repo"
    hubs[source].assert_called_once_with(
        "Org/repo",
        revision=None,
        cache_dir=None,
        local_files_only=True,
        allow_patterns=ANY,
    )
    assert (hubs[source].call_args.kwargs["allow_patterns"] is None) == (
        field != "tokenizer_path"
    )
    hubs["hf" if source == "modelscope" else "modelscope"].assert_not_called()
    assert getattr(args, field) == f"{source}://Org/repo"
    assert os.environ["SGLANG_USE_MODELSCOPE"] == global_source


@pytest.mark.parametrize(
    "source,draft_source", [("hf", "modelscope"), ("modelscope", "hf")]
)
@pytest.mark.parametrize("cli", [False, True])
@pytest.mark.parametrize("global_source", ["true", "false"])
def test_mixed_sources_through_server_args(
    hubs, monkeypatch, source, draft_source, cli, global_source
):
    monkeypatch.setenv("SGLANG_USE_MODELSCOPE", global_source)
    options = dict(
        model_path=f"{source}://Org/target",
        tokenizer_path=f"{draft_source}://Org/tokenizer",
        speculative_draft_model_path=f"{draft_source}://Org/draft",
        revision="v1",
        speculative_draft_model_revision="v2",
        download_dir="/custom",
    )
    if cli:
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        argv = [
            item
            for key, value in options.items()
            for item in ("--" + key.replace("_", "-"), value)
        ]
        args = ServerArgs.from_cli_args(parser.parse_args(argv))
    else:
        args = ServerArgs(**options)
    with patch("sglang.srt.arg_groups.model_path_hook.resolve_hf_gguf_model_path"):
        handle_model_source_paths(args)
    view = resolving_view(args)
    assert view.model_path == hubs[source].return_value
    assert (
        view.tokenizer_path
        == view.speculative_draft_model_path
        == hubs[draft_source].return_value
    )
    assert view.served_model_name == "Org/target"
    hubs[source].assert_called_once_with(
        "Org/target",
        revision="v1",
        cache_dir="/custom",
        local_files_only=True,
        allow_patterns=None,
    )
    assert [call.kwargs["revision"] for call in hubs[draft_source].call_args_list] == [
        "v2",
        "v1",
    ]
    assert args.model_path == options["model_path"]


def test_resolution_pipeline_preserves_implicit_dspark_draft(hubs, monkeypatch):
    from transformers import LlamaConfig

    snapshot = hubs["hf"].return_value
    LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=2,
        vocab_size=128,
        max_position_embeddings=2048,
        dspark_block_size=2,
        dspark_markov_rank=1,
        dspark_noise_token_id=1,
        dspark_target_layer_ids=[0],
    ).save_pretrained(snapshot)
    monkeypatch.setenv("SGLANG_USE_MODELSCOPE", "true")
    args = ServerArgs(
        model_path="hf://Org/repo",
        device="cuda",
        speculative_algorithm="DSPARK",
        revision="v1",
    )
    args.resolve_once()
    cfg = resolving_view(args)
    assert (
        cfg.model_path
        == cfg.tokenizer_path
        == cfg.speculative_draft_model_path
        == snapshot
    )
    assert cfg.speculative_draft_model_revision == "v1"
    assert cfg.served_model_name == "Org/repo"
    hubs["hf"].assert_called_once()
    hubs["modelscope"].assert_not_called()


@pytest.mark.parametrize(
    "revision,draft_format,load_format,calls",
    [
        ("v1", None, "auto", 1),
        ("v1", "pt", "auto", 1),
        ("v2", None, "auto", 2),
        ("v1", "dummy", "auto", 2),
        ("v1", None, "DUMMY", 1),
    ],
)
def test_shared_snapshots(hubs, revision, draft_format, load_format, calls):
    result = resolve_model_hub_paths(
        ServerArgs(
            model_path="hf://Org/repo",
            tokenizer_path="hf://Org/repo",
            speculative_draft_model_path="hf://Org/repo",
            served_model_name="custom",
            revision="v1",
            speculative_draft_model_revision=revision,
            speculative_draft_load_format=draft_format,
            load_format=load_format,
        )
    )
    assert result["tokenizer_path"] == result["model_path"]
    assert hubs["hf"].call_count == calls
    assert "served_model_name" not in result
    if load_format == "DUMMY":
        patterns = hubs["hf"].call_args.kwargs["allow_patterns"]
        assert "*.json" in patterns and "*.safetensors" not in patterns


@pytest.mark.parametrize("source", ["hf", "modelscope"])
def test_failure_preserves_exception_without_fallback(hubs, caplog, source):
    error = OSError("Hub unavailable")
    hubs[source].side_effect = error
    with pytest.raises(OSError) as caught:
        resolve_model_hub_paths(ServerArgs(model_path=f"{source}://Org/repo"))
    assert caught.value is error
    assert f"{source}://Org/repo" in caplog.text
    hubs["hf" if source == "modelscope" else "modelscope"].assert_not_called()


@pytest.mark.parametrize("source", ["hf", "modelscope"])
def test_full_snapshot_preserves_bundled_weights_in_one_fetch(hubs, source, tmp_path):
    folder = tmp_path / "snapshot"
    folder.mkdir()
    files = [
        "config.json",
        "pytorch_model.bin",
        "draft/model.safetensors",
        "original/model.safetensors",
    ]

    def download(repo, *, allow_patterns, **kwargs):
        for name in files:
            if allow_patterns is None or any(
                fnmatch.fnmatch(name, p) for p in allow_patterns
            ):
                path = folder / name
                path.parent.mkdir(exist_ok=True, parents=True)
                path.write_text("{}")
        return str(folder)

    hubs[source].side_effect = download
    result = resolve_model_hub_paths(
        ServerArgs(model_path=f"{source}://Org/repo", revision="main")
    )
    assert result["model_path"] == str(folder)
    assert all((folder / name).is_file() for name in files)
    # A single SDK resolution cannot mix metadata from one branch tip with
    # weights downloaded after the branch has advanced to another commit.
    assert hubs[source].call_count == 1
    assert hubs[source].call_args.kwargs["allow_patterns"] is None


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize(
    "filename,config",
    [
        ("config.json", {"auto_map": {"AutoConfig": "Org/code--config.Custom"}}),
        (
            "tokenizer_config.json",
            {"auto_map": {"AutoTokenizer": ["Org/code--tok.Custom", None]}},
        ),
        ("tokenizer_config.json", {"auto_map": ["Org/code--tok.Custom", None]}),
        (
            "config.json",
            {"text_config": {"auto_map": {"AutoConfig": "Org/code--config.Custom"}}},
        ),
        ("dspark/config.json", {"auto_map": {"AutoConfig": "Org/code--config.Custom"}}),
    ],
)
def test_cross_repo_auto_map_rejected_without_code_fetch(hubs, trust, filename, config):
    path = Path(hubs["modelscope"].return_value) / filename
    path.parent.mkdir(exist_ok=True)
    original = json.dumps(config)
    path.write_text(original)
    with pytest.raises(ValueError, match="auto_map"):
        resolve_model_hub_paths(
            ServerArgs(model_path="modelscope://Org/repo", trust_remote_code=trust)
        )
    assert hubs["modelscope"].call_count == 1
    hubs["hf"].assert_not_called()
    assert path.read_text() == original


def test_local_auto_map_and_unrelated_json_are_left_unchanged(hubs):
    snapshot = Path(hubs["hf"].return_value)
    configs = {
        "config.json": json.dumps({"auto_map": {"AutoConfig": "configuration.Local"}}),
        "tokenizer_config.json": json.dumps({"auto_map": ["tokenization.Local", None]}),
        "other_config.json": "not valid JSON",
        "list_config.json": "[null, 1]",
        "scalar_config.json": "null",
    }
    for name, content in configs.items():
        (snapshot / name).write_text(content)
    result = resolve_model_hub_paths(
        ServerArgs(model_path="hf://Org/repo", trust_remote_code=True)
    )
    assert result["model_path"] == str(snapshot)
    assert all(
        (snapshot / name).read_text() == content for name, content in configs.items()
    )


def test_unsupported_loader_fails_before_download(hubs):
    with pytest.raises(ValueError, match="local snapshot"):
        resolve_model_hub_paths(
            ServerArgs(model_path="hf://Org/repo", load_format="gguf")
        )
    hubs["hf"].assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
