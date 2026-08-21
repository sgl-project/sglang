import json
from types import SimpleNamespace
from unittest.mock import patch

from sglang.multimodal_gen.checkpoints.preflight import (
    checkpoint_requests_from_server_args,
    run_checkpoint_preflight,
)
from sglang.multimodal_gen.checkpoints.resolver import (
    CheckpointFile,
    CheckpointInventory,
    CheckpointRequest,
    CheckpointSource,
    ResolvedCheckpoint,
)
from sglang.multimodal_gen.runtime.entrypoints.cli.generate import generate_cmd
from sglang.multimodal_gen.runtime.entrypoints.cli.serve import execute_serve_cmd
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _server_args(**overrides):
    values = {
        "model_path": "owner/base",
        "revision": "v1",
        "component_paths": {"text_encoder": "owner/encoder"},
        "transformer_weights_path": "/weights/dit.safetensors",
        "lora_path": "owner/lora",
        "lora_weight_name": "turbo.safetensors",
        "checkpoint_preflight": "metadata",
        "checkpoint_report_format": "text",
        "webui": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _resolved_checkpoint() -> ResolvedCheckpoint:
    request = CheckpointRequest(name="model", role="pipeline", source="owner/base")
    source = CheckpointSource(
        original="owner/base", kind="huggingface", repo_id="owner/base"
    )
    return ResolvedCheckpoint(
        request=request,
        inventory=CheckpointInventory(
            source=source,
            resolved_revision="sha",
            files=(CheckpointFile(path="model_index.json", size=10),),
        ),
        selected_files=(),
        container_format=None,
        quantization_method=None,
        quantization_source=None,
        tensor_summary=None,
    )


def test_preflight_collects_all_configured_checkpoint_roles():
    requests = checkpoint_requests_from_server_args(_server_args())

    assert [(request.name, request.role) for request in requests] == [
        ("model", "pipeline"),
        ("component:text_encoder", "component"),
        ("component_weights:transformer", "component_weights"),
        ("startup_lora", "lora"),
    ]
    assert requests[-1].weight_name == "turbo.safetensors"


def test_json_report_keeps_server_args_logs_off_stdout(capsys):
    with (
        patch.object(ServerArgs, "_adjust_parameters"),
        patch.object(ServerArgs, "_validate_parameters"),
    ):
        server_args = ServerArgs(
            model_path="owner/base",
            checkpoint_preflight="metadata",
            checkpoint_report_format="json",
        )
    with patch(
        "sglang.multimodal_gen.checkpoints.preflight." "resolve_server_checkpoints",
        return_value=(_resolved_checkpoint(),),
    ):
        assert run_checkpoint_preflight(server_args)

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["schema_version"] == 1
    assert len(report["checkpoints"]) == 1
    assert "server_args" in captured.err


def test_generate_and_serve_stop_before_launch_after_preflight():
    server_args = _server_args()
    with (
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.cli.generate."
            "ServerArgs.from_cli_args",
            return_value=server_args,
        ),
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.cli.generate."
            "run_checkpoint_preflight",
            return_value=True,
        ),
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.cli.generate."
            "DiffGenerator.from_pretrained"
        ) as generate,
    ):
        generate_cmd(SimpleNamespace(), [])
        generate.assert_not_called()

    with (
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.cli.serve."
            "ServerArgs.from_cli_args",
            return_value=server_args,
        ),
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.cli.serve."
            "run_checkpoint_preflight",
            return_value=True,
        ),
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.cli.serve.dispatch_launch"
        ) as launch,
    ):
        execute_serve_cmd(SimpleNamespace(), [])
        launch.assert_not_called()
