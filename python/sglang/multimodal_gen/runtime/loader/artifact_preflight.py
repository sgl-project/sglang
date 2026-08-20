"""Integrated artifact preflight for diffusion generate and serve."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

from sglang.multimodal_gen.runtime.loader.artifact_resolver import (
    ArtifactRequest,
    ResolvedArtifact,
    materialize_resolved_artifact,
    resolve_artifact,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.server_args import ServerArgs


def artifact_requests_from_server_args(
    server_args: ServerArgs,
) -> tuple[ArtifactRequest, ...]:
    requests = [
        ArtifactRequest(
            name="model",
            role="pipeline",
            source=server_args.model_path,
            revision=server_args.revision,
        )
    ]
    requests.extend(
        ArtifactRequest(
            name=f"component:{component}",
            role="component",
            component=component,
            source=source,
        )
        for component, source in sorted(server_args.component_paths.items())
    )
    if server_args.transformer_weights_path is not None:
        requests.append(
            ArtifactRequest(
                name="component_weights:transformer",
                role="component_weights",
                component="transformer",
                source=server_args.transformer_weights_path,
            )
        )
    if server_args.lora_path is not None:
        requests.append(
            ArtifactRequest(
                name="startup_lora",
                role="lora",
                source=server_args.lora_path,
                weight_name=server_args.lora_weight_name,
            )
        )
    return tuple(requests)


def resolve_server_artifacts(server_args: ServerArgs) -> tuple[ResolvedArtifact, ...]:
    return tuple(
        resolve_artifact(request)
        for request in artifact_requests_from_server_args(server_args)
    )


def _source_label(artifact: ResolvedArtifact) -> str:
    source = artifact.inventory.source
    if source.kind == "local":
        assert source.local_path is not None
        return source.local_path
    assert source.repo_id is not None
    revision = artifact.inventory.resolved_revision or source.revision or "main"
    suffix = source.filename or source.subfolder
    label = f"{source.repo_id}@{revision}"
    return f"{label}/{suffix}" if suffix is not None else label


def _format_text_report(mode: str, artifacts: tuple[ResolvedArtifact, ...]) -> str:
    lines = [f"Artifact preflight: {mode}"]
    for artifact in artifacts:
        request = artifact.request
        summary = artifact.tensor_summary
        lines.append(f"- {request.name} [{request.role}]: {_source_label(artifact)}")
        displayed_files = artifact.selected_files[:8]
        selected = ", ".join(displayed_files) or "no weight files"
        if len(artifact.selected_files) > len(displayed_files):
            selected += f", ... ({len(artifact.selected_files)} files)"
        lines.append(f"  selected: {selected}")
        if artifact.quantization_method is not None:
            lines.append(
                "  quantization: "
                f"{artifact.quantization_method} ({artifact.quantization_source})"
            )
        if summary is not None:
            details = f"{summary.tensor_count} tensors; dtypes={list(summary.dtypes)}"
            if summary.lora_ranks:
                details += f"; lora_ranks={list(summary.lora_ranks)}"
            lines.append(f"  metadata: {details}")
    return "\n".join(lines)


def _format_json_report(mode: str, artifacts: tuple[ResolvedArtifact, ...]) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "mode": mode,
            "artifacts": [asdict(artifact) for artifact in artifacts],
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def run_artifact_preflight(server_args: ServerArgs) -> bool:
    """Run an explicitly requested preflight and return whether execution stops."""
    mode = server_args.artifact_preflight
    if mode is None:
        return False
    artifacts = resolve_server_artifacts(server_args)
    if mode == "full":
        for artifact in artifacts:
            materialize_resolved_artifact(artifact)
    if server_args.artifact_report_format == "json":
        print(_format_json_report(mode, artifacts))
    else:
        print(_format_text_report(mode, artifacts))
    return True
