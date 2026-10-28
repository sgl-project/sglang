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
    assert resolve_model_hub_paths(args)[field] == hubs[source].return_value
    assert hubs[source].call_count == (1 if field == "tokenizer_path" else 2)
    hubs[source].assert_any_call(
        "Org/repo",
        revision=None,
        cache_dir=None,
        local_files_only=True,
        ignore_patterns=["original/**"],
        allow_patterns=ANY,
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
    hubs[source].assert_any_call(
        "Org/target",
        revision="v1",
        cache_dir="/custom",
        local_files_only=True,
        ignore_patterns=["original/**"],
        allow_patterns=["*.safetensors"],
    )
    assert [call.kwargs["revision"] for call in hubs[draft_source].call_args_list] == [
        "v2",
        "v2",
        "v1",
    ]
    assert args.model_path == options["model_path"]


@pytest.mark.parametrize(
    "tokenizer,draft,name,revision,calls",
    [
        (None, None, None, "v1", 2),
        ("hf://Org/repo", "hf://Org/repo", "custom", "v1", 2),
        (None, "hf://Org/repo", None, "v2", 4),
    ],
)
def test_default_tokenizer_and_shared_snapshot(
    hubs, tokenizer, draft, name, revision, calls
):
    result = resolve_model_hub_paths(
        ServerArgs(
            model_path="hf://Org/repo",
            tokenizer_path=tokenizer,
            speculative_draft_model_path=draft,
            served_model_name=name,
            revision="v1",
            speculative_draft_model_revision=revision,
        )
    )
    assert result["tokenizer_path"] == result["model_path"]
    assert hubs["hf"].call_count == calls
    assert ("speculative_draft_model_path" in result) == (draft is not None)
    assert result.get("served_model_name") == (None if name else "Org/repo")


def test_dummy_loading_skips_weights(hubs):
    resolve_model_hub_paths(
        ServerArgs(
            model_path="hf://Org/repo",
            speculative_draft_model_path="modelscope://Org/draft",
            load_format="dummy",
        )
    )
    for download in hubs.values():
        assert download.call_count == 1
        assert "*.safetensors" not in download.call_args.kwargs["allow_patterns"]


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
@pytest.mark.parametrize(
    "load_format,files,expected",
    [
        ("auto", ["model.safetensors", "pytorch_model.bin"], ["model.safetensors"]),
        ("auto", ["pytorch_model.bin"], ["pytorch_model.bin"]),
        ("pt", ["model.safetensors", "model.pt"], ["model.pt"]),
        ("dummy", ["model.safetensors"], []),
    ],
)
def test_download_only_selected_weights(
    hubs, source, load_format, files, expected, tmp_path
):
    folder = tmp_path / "filtered"
    folder.mkdir()
    files += [
        "config.json",
        "original/model.pth",
        "original/model.safetensors",
        "model.h5",
        "model.msgpack",
        "model.onnx",
        "Q4.gguf",
    ]

    def download(repo, *, allow_patterns, ignore_patterns, **kwargs):
        for name in files:
            if any(fnmatch.fnmatch(name, p) for p in allow_patterns) and not any(
                fnmatch.fnmatch(name, p) for p in ignore_patterns
            ):
                path = folder / name
                path.parent.mkdir(exist_ok=True, parents=True)
                path.write_text("{}")
        return str(folder)

    hubs[source].side_effect = download
    result = resolve_model_hub_paths(
        ServerArgs(model_path=f"{source}://Org/repo", load_format=load_format)
    )
    assert result["model_path"] == str(folder)
    assert sorted(p.name for p in folder.iterdir()) == sorted(
        ["config.json", *expected]
    )


@pytest.mark.parametrize("source", ["hf", "modelscope"])
def test_shard_index_avoids_duplicate_weights(hubs, source):
    folder = Path(hubs[source].return_value)
    (folder / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "model-00001.safetensors"}})
    )
    resolve_model_hub_paths(ServerArgs(model_path=f"{source}://Org/repo"))
    assert hubs[source].call_args.kwargs["allow_patterns"] == [
        "model-00001.safetensors",
        "**/*.safetensors",
    ]


def test_distinct_load_formats_do_not_share_resolution(hubs):
    resolve_model_hub_paths(
        ServerArgs(
            model_path="hf://Org/repo",
            speculative_draft_model_path="hf://Org/repo",
            speculative_draft_load_format="pt",
        )
    )
    assert hubs["hf"].call_args.kwargs["allow_patterns"] == ["*.pt"]


@pytest.mark.parametrize("source", ["hf", "modelscope"])
@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("global_source", [False, True])
def test_cross_repo_code_stays_on_selected_hub(
    hubs, monkeypatch, source, trust, tmp_path, global_source
):
    monkeypatch.setenv("SGLANG_USE_MODELSCOPE", str(global_source).lower())
    snapshot = Path(hubs[source].return_value)
    config = {
        "model_type": "llama",
        "auto_map": {"AutoConfig": "Org/code--configuration.CustomConfig"},
    }
    (snapshot / "config.json").write_text(json.dumps(config))
    (snapshot / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "auto_map": {
                    "AutoTokenizer": ["Org/code--configuration.CustomTokenizer", None]
                }
            }
        )
    )
    code = tmp_path / "code"
    code.mkdir()
    (code / "configuration.py").write_text(
        "from transformers import PretrainedConfig\n"
        "class CustomConfig(PretrainedConfig):\n"
        "    model_type = 'llama'\n"
        f"    hub_source = {source!r}\n"
    )
    hubs[source].side_effect = lambda repo, **kw: str(
        code if repo == "Org/code" else snapshot
    )
    result = resolve_model_hub_paths(
        ServerArgs(model_path=f"{source}://Org/repo", trust_remote_code=trust)
    )
    if global_source:
        from modelscope import AutoConfig
    else:
        from transformers import AutoConfig

    loaded = AutoConfig.from_pretrained(
        result["model_path"], trust_remote_code=trust, local_files_only=True
    )
    assert getattr(loaded, "hub_source", None) == (source if trust else None)
    code_calls = [c for c in hubs[source].call_args_list if c.args == ("Org/code",)]
    assert len(code_calls) == int(trust)
    if trust:
        assert code_calls[0].kwargs["allow_patterns"] == ["*.py"]
        localized = json.loads(
            (Path(result["tokenizer_path"]) / "tokenizer_config.json").read_text()
        )
        assert "--" not in localized["auto_map"]["AutoTokenizer"][0]
    assert json.loads((snapshot / "config.json").read_text()) == config
    hubs["modelscope" if source == "hf" else "hf"].assert_not_called()


def test_unsupported_loader_fails_before_download(hubs):
    with pytest.raises(ValueError, match="local snapshot"):
        resolve_model_hub_paths(
            ServerArgs(model_path="hf://Org/repo", load_format="gguf")
        )
    hubs["hf"].assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
