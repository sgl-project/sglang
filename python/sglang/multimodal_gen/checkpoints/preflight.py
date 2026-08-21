"""Integrated checkpoint preflight for diffusion generate and serve."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

from sglang.multimodal_gen.checkpoints.resolver import (
    CheckpointRequest,
    ResolvedCheckpoint,
    materialize_resolved_checkpoint,
    resolve_checkpoint,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.server_args import ServerArgs


def checkpoint_requests_from_server_args(
    server_args: ServerArgs,
) -> tuple[CheckpointRequest, ...]:
    requests = [
        CheckpointRequest(
            name="model",
            role="pipeline",
            source=server_args.model_path,
            revision=server_args.revision,
        )
    ]
    requests.extend(
        CheckpointRequest(
            name=f"component:{component}",
            role="component",
            component=component,
            source=source,
        )
        for component, source in sorted(server_args.component_paths.items())
    )
    if server_args.transformer_weights_path is not None:
        requests.append(
            CheckpointRequest(
                name="component_weights:transformer",
                role="component_weights",
                component="transformer",
                source=server_args.transformer_weights_path,
            )
        )
    if server_args.lora_path is not None:
        requests.append(
            CheckpointRequest(
                name="startup_lora",
                role="lora",
                source=server_args.lora_path,
                weight_name=server_args.lora_weight_name,
            )
        )
    return tuple(requests)


def resolve_server_checkpoints(
    server_args: ServerArgs,
) -> tuple[ResolvedCheckpoint, ...]:
    return tuple(
        resolve_checkpoint(request)
        for request in checkpoint_requests_from_server_args(server_args)
    )


def _source_label(checkpoint: ResolvedCheckpoint) -> str:
    source = checkpoint.inventory.source
    if source.kind == "local":
        assert source.local_path is not None
        return source.local_path
    assert source.repo_id is not None
    revision = checkpoint.inventory.resolved_revision or source.revision or "main"
    suffix = source.filename or source.subfolder
    label = f"{source.repo_id}@{revision}"
    return f"{label}/{suffix}" if suffix is not None else label


def _format_text_report(mode: str, checkpoints: tuple[ResolvedCheckpoint, ...]) -> str:
    lines = [f"Checkpoint preflight: {mode}"]
    for checkpoint in checkpoints:
        request = checkpoint.request
        summary = checkpoint.tensor_summary
        lines.append(f"- {request.name} [{request.role}]: {_source_label(checkpoint)}")
        displayed_files = checkpoint.selected_files[:8]
        selected = ", ".join(displayed_files) or "no weight files"
        if len(checkpoint.selected_files) > len(displayed_files):
            selected += f", ... ({len(checkpoint.selected_files)} files)"
        lines.append(f"  selected: {selected}")
        if checkpoint.quantization_method is not None:
            lines.append(
                "  quantization: "
                f"{checkpoint.quantization_method} ({checkpoint.quantization_source})"
            )
        if summary is not None:
            details = f"{summary.tensor_count} tensors; dtypes={list(summary.dtypes)}"
            if summary.lora_ranks:
                details += f"; lora_ranks={list(summary.lora_ranks)}"
            lines.append(f"  metadata: {details}")
    return "\n".join(lines)


def _format_json_report(mode: str, checkpoints: tuple[ResolvedCheckpoint, ...]) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "mode": mode,
            "checkpoints": [asdict(checkpoint) for checkpoint in checkpoints],
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def run_checkpoint_preflight(server_args: ServerArgs) -> bool:
    """Run an explicitly requested preflight and return whether execution stops."""
    mode = server_args.checkpoint_preflight
    if mode is None:
        return False
    checkpoints = resolve_server_checkpoints(server_args)
    if mode == "full":
        for checkpoint in checkpoints:
            materialize_resolved_checkpoint(checkpoint)
    if server_args.checkpoint_report_format == "json":
        print(_format_json_report(mode, checkpoints))
    else:
        print(_format_text_report(mode, checkpoints))
    return True
