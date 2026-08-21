"""Checkpoint inspection shared by the generate and serve CLIs."""

from __future__ import annotations

import argparse
import json
from typing import Any, Mapping

from sglang.multimodal_gen.checkpoints.resolver import (
    CheckpointRequest,
    ResolvedCheckpoint,
    resolve_checkpoint,
)

_REPORT_FILE_LIMIT = 32


def add_checkpoint_inspection_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--inspect-checkpoints",
        action="store_true",
        default=False,
        help=(
            "Inspect the configured model, component overrides, transformer "
            "weights override, and startup LoRA, print a report, then exit "
            "before model construction. Tensor payloads are not downloaded."
        ),
    )
    parser.add_argument(
        "--checkpoint-report-format",
        choices=("text", "json"),
        default="text",
        help="Output format used by --inspect-checkpoints.",
    )
    return parser


def checkpoint_requests_from_launch_args(
    launch_args: Mapping[str, Any],
) -> tuple[CheckpointRequest, ...]:
    model_path = launch_args.get("model_path")
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("--inspect-checkpoints requires --model-path")

    requests = [
        CheckpointRequest(
            name="model",
            role="pipeline",
            source=model_path,
            revision=launch_args.get("revision"),
        )
    ]
    component_paths = launch_args.get("component_paths") or {}
    if not isinstance(component_paths, Mapping):
        raise TypeError("component_paths must be a mapping")
    requests.extend(
        CheckpointRequest(
            name=f"component:{component}",
            role="component",
            component=component,
            source=source,
        )
        for component, source in sorted(component_paths.items())
    )
    transformer_weights_path = launch_args.get("transformer_weights_path")
    if transformer_weights_path is not None:
        requests.append(
            CheckpointRequest(
                name="component_weights:transformer",
                role="component_weights",
                component="transformer",
                source=transformer_weights_path,
            )
        )
    lora_path = launch_args.get("lora_path")
    if lora_path is not None:
        requests.append(
            CheckpointRequest(
                name="startup_lora",
                role="lora",
                source=lora_path,
                weight_name=launch_args.get("lora_weight_name"),
            )
        )
    return tuple(requests)


def _format_text_report(checkpoints: tuple[ResolvedCheckpoint, ...]) -> str:
    lines = ["Checkpoint inspection"]
    for checkpoint in checkpoints:
        request = checkpoint.request
        summary = checkpoint.tensor_summary
        source = checkpoint.inventory.source
        revision = checkpoint.inventory.resolved_revision or source.revision
        label = source.original if revision is None else f"{source.original}@{revision}"
        lines.append(f"- {request.name} [{request.role}]: {label}")
        displayed_files = checkpoint.selected_files[:8]
        selected = ", ".join(displayed_files) or "no weight files"
        if len(checkpoint.selected_files) > len(displayed_files):
            selected += f", ... ({len(checkpoint.selected_files)} files)"
        lines.append(f"  weights: {selected}")
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


def _checkpoint_report(checkpoint: ResolvedCheckpoint) -> dict[str, Any]:
    request = checkpoint.request
    source = checkpoint.inventory.source
    displayed_files = checkpoint.selected_files[:_REPORT_FILE_LIMIT]
    report: dict[str, Any] = {
        "name": request.name,
        "role": request.role,
        "component": request.component,
        "source": source.original,
        "source_kind": source.kind,
        "resolved_revision": checkpoint.inventory.resolved_revision,
        "container_format": checkpoint.container_format,
        "weight_file_count": len(checkpoint.selected_files),
        "weight_files": list(displayed_files),
    }
    if len(displayed_files) != len(checkpoint.selected_files):
        report["weight_files_truncated"] = True
    if checkpoint.quantization_method is not None:
        report["quantization"] = {
            "method": checkpoint.quantization_method,
            "source": checkpoint.quantization_source,
        }
    if checkpoint.tensor_summary is not None:
        summary = checkpoint.tensor_summary
        report["tensors"] = {
            "count": summary.tensor_count,
            "dtypes": list(summary.dtypes),
            "lora_ranks": list(summary.lora_ranks),
            "key_samples": list(summary.key_samples),
        }
    return report


def run_checkpoint_inspection(launch_args: Mapping[str, Any]) -> bool:
    """Run an explicitly requested inspection and return whether execution stops."""
    if not launch_args.get("inspect_checkpoints", False):
        return False
    checkpoints = tuple(
        resolve_checkpoint(request)
        for request in checkpoint_requests_from_launch_args(launch_args)
    )
    if launch_args.get("checkpoint_report_format", "text") == "json":
        report = {
            "schema_version": 1,
            "checkpoints": [_checkpoint_report(item) for item in checkpoints],
        }
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(_format_text_report(checkpoints))
    return True
