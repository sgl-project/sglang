# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys
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
def hubs(monkeypatch):
    hf, ms = Mock(return_value="/cache/hf"), Mock(return_value="/cache/ms")
    monkeypatch.setattr("huggingface_hub.snapshot_download", hf)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)
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
    kwargs = dict(revision=None, cache_dir=None, local_files_only=True)
    if field == "tokenizer_path":
        kwargs["ignore_patterns"] = ANY
    hubs[source].assert_called_once_with("Org/repo", **kwargs)
    hubs["hf" if source == "modelscope" else "modelscope"].assert_not_called()
    assert getattr(args, field) == f"{source}://Org/repo"
    assert os.environ["SGLANG_USE_MODELSCOPE"] == global_source


@pytest.mark.parametrize(
    "source,draft_source", [("hf", "modelscope"), ("modelscope", "hf")]
)
@pytest.mark.parametrize("cli", [False, True])
def test_mixed_sources_through_server_args(hubs, source, draft_source, cli):
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
        "Org/target", revision="v1", cache_dir="/custom", local_files_only=True
    )
    assert [call.kwargs["revision"] for call in hubs[draft_source].call_args_list] == [
        "v2",
        "v1",
    ]
    assert args.model_path == options["model_path"]


@pytest.mark.parametrize(
    "tokenizer,draft,name,revision,calls",
    [
        (None, None, None, "v1", 1),
        ("hf://Org/repo", "hf://Org/repo", "custom", "v1", 1),
        (None, "hf://Org/repo", None, "v2", 2),
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
        assert "*.safetensors" in download.call_args.kwargs["ignore_patterns"]


@pytest.mark.parametrize("source", ["hf", "modelscope"])
def test_failure_preserves_exception_without_fallback(hubs, caplog, source):
    error = OSError("Hub unavailable")
    hubs[source].side_effect = error
    with pytest.raises(OSError) as caught:
        resolve_model_hub_paths(ServerArgs(model_path=f"{source}://Org/repo"))
    assert caught.value is error
    assert f"{source}://Org/repo" in caplog.text
    hubs["hf" if source == "modelscope" else "modelscope"].assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
