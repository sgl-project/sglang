import json
from types import SimpleNamespace
from unittest.mock import patch

from sglang.multimodal_gen.checkpoint_inspection.resolver import (
    CheckpointFile,
    CheckpointInventory,
    CheckpointRequest,
    CheckpointSource,
    ResolvedCheckpoint,
    TensorSummary,
)
from sglang.multimodal_gen.runtime.entrypoints.cli.checkpoint_inspection import (
    checkpoint_requests_from_launch_args,
    run_checkpoint_inspection,
)
from sglang.multimodal_gen.runtime.entrypoints.cli.generate import generate_cmd
from sglang.multimodal_gen.runtime.entrypoints.cli.serve import execute_serve_cmd
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _launch_args(**overrides):
    values = {
        "model_path": "owner/base",
        "revision": "v1",
        "component_paths": {"text_encoder": "owner/encoder"},
        "transformer_weights_path": "/weights/dit.safetensors",
        "lora_path": "owner/lora",
        "lora_weight_name": "turbo.safetensors",
        "inspect_checkpoints": True,
        "checkpoint_report_format": "text",
    }
    values.update(overrides)
    return values


def _resolved_checkpoint(weight_file_count: int = 1) -> ResolvedCheckpoint:
    request = CheckpointRequest(name="model", role="pipeline", source="owner/base")
    source = CheckpointSource(
        original="owner/base", kind="huggingface", repo_id="owner/base"
    )
    weight_files = tuple(
        f"model-{index:05d}.safetensors" for index in range(weight_file_count)
    )
    return ResolvedCheckpoint(
        request=request,
        inventory=CheckpointInventory(
            source=source,
            resolved_revision="sha",
            files=(CheckpointFile(path="config.json", size=10),)
            + tuple(CheckpointFile(path=path, size=20) for path in weight_files),
        ),
        selected_files=weight_files,
        container_format="safetensors",
        quantization_method=None,
        quantization_source=None,
        tensor_summary=TensorSummary(
            tensor_count=1,
            dtypes=("BF16",),
            key_samples=("blocks.0.weight",),
            metadata={"private_build_path": "/internal/build"},
            lora_ranks=(),
        ),
    )


def test_inspection_collects_all_configured_checkpoint_roles():
    requests = checkpoint_requests_from_launch_args(_launch_args())

    assert [(request.name, request.role) for request in requests] == [
        ("model", "pipeline"),
        ("component:text_encoder", "component"),
        ("component_weights:transformer", "component_weights"),
        ("startup_lora", "lora"),
    ]
    assert requests[-1].weight_name == "turbo.safetensors"


def test_json_report_is_bounded_and_does_not_expose_raw_metadata(capsys):
    launch_args = _launch_args(
        checkpoint_report_format="json",
        component_paths={},
        transformer_weights_path=None,
        lora_path=None,
    )
    with patch(
        "sglang.multimodal_gen.runtime.entrypoints.cli.checkpoint_inspection."
        "resolve_checkpoint",
        return_value=_resolved_checkpoint(weight_file_count=40),
    ):
        assert run_checkpoint_inspection(launch_args)

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["schema_version"] == 1
    checkpoint = report["checkpoints"][0]
    assert checkpoint["resolved_revision"] == "sha"
    assert checkpoint["weight_file_count"] == 40
    assert len(checkpoint["weight_files"]) == 32
    assert checkpoint["weight_files_truncated"] is True
    assert checkpoint["tensors"]["key_samples"] == ["blocks.0.weight"]
    assert "metadata" not in checkpoint["tensors"]
    assert "/internal/build" not in captured.out


def test_generate_and_serve_inspect_before_server_args_construction():
    launch_args = _launch_args()
    with (
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.cli.generate."
            "ServerArgs.collect_cli_args",
            return_value=launch_args,
        ),
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.cli.generate."
            "run_checkpoint_inspection",
            return_value=True,
        ),
        patch.object(ServerArgs, "from_dict") as generate_server_args,
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.cli.generate."
            "DiffGenerator.from_pretrained"
        ) as generate,
    ):
        generate_cmd(SimpleNamespace(), [])
        generate_server_args.assert_not_called()
        generate.assert_not_called()

    with (
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.cli.serve."
            "ServerArgs.collect_cli_args",
            return_value=launch_args,
        ),
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.cli.serve."
            "run_checkpoint_inspection",
            return_value=True,
        ),
        patch.object(ServerArgs, "from_dict") as serve_server_args,
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.cli.serve.dispatch_launch"
        ) as launch,
    ):
        execute_serve_cmd(SimpleNamespace(), [])
        serve_server_args.assert_not_called()
        launch.assert_not_called()
