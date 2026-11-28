"""JSONL request manifests for reproducible offline diffusion benchmarks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_TOP_LEVEL_SAMPLING_FIELDS = {
    "fps",
    "guidance_scale",
    "height",
    "image_paths",
    "negative_prompt",
    "num_frames",
    "num_inference_steps",
    "num_outputs_per_prompt",
    "seed",
    "width",
}
_RESERVED_SAMPLING_FIELDS = {
    "image_path",
    "output_file_name",
    "output_path",
    "prompt",
    "request_id",
    "return_file_paths_only",
    "save_output",
}
_ALLOWED_FIELDS = {
    "prompt",
    "request_id",
    "sampling_params",
    *_TOP_LEVEL_SAMPLING_FIELDS,
}


@dataclass(frozen=True)
class ManifestRequest:
    """One fully resolved request from a benchmark manifest."""

    request_id: str
    prompt: str
    sampling_params: dict[str, Any]


@dataclass(frozen=True)
class LoadedRequestManifest:
    """Parsed requests plus the digest of the exact input manifest."""

    path: str
    sha256: str
    requests: list[ManifestRequest]


def file_sha256(path: str | Path) -> str:
    """Return the SHA256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_url(value: str) -> bool:
    return urlparse(value).scheme in {"http", "https"}


def _resolve_image_paths(value: Any, base_dir: Path, line_number: int) -> Any:
    if isinstance(value, str):
        image_paths = [value]
        return_scalar = True
    elif (
        isinstance(value, list)
        and value
        and all(isinstance(item, str) and item for item in value)
    ):
        image_paths = value
        return_scalar = False
    else:
        raise ValueError(
            f"Manifest line {line_number}: image_paths must be a non-empty "
            "string or list of strings"
        )

    resolved = [
        item if _is_url(item) else str((base_dir / item).resolve())
        for item in image_paths
    ]
    return resolved[0] if return_scalar else resolved


def load_request_manifest(path: str | Path) -> LoadedRequestManifest:
    """Load and validate a JSONL request manifest.

    Relative condition-image paths are resolved against the manifest directory.
    A manifest is authoritative: every non-empty line becomes exactly one request.
    """
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError(f"Request manifest does not exist: {manifest_path}")

    requests: list[ManifestRequest] = []
    request_ids: set[str] = set()
    with manifest_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Manifest line {line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"Manifest line {line_number}: each line must be a JSON object"
                )

            unknown_fields = set(record) - _ALLOWED_FIELDS
            if unknown_fields:
                raise ValueError(
                    f"Manifest line {line_number}: unsupported field(s): "
                    f"{', '.join(sorted(unknown_fields))}"
                )

            prompt = record.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"Manifest line {line_number}: prompt must be a non-empty string"
                )

            request_id = record.get("request_id", f"request-{line_number:05d}")
            if not isinstance(request_id, str) or not request_id:
                raise ValueError(
                    f"Manifest line {line_number}: request_id must be a non-empty string"
                )
            if request_id in request_ids:
                raise ValueError(
                    f"Manifest line {line_number}: duplicate request_id {request_id!r}"
                )
            request_ids.add(request_id)

            sampling_params = record.get("sampling_params", {})
            if not isinstance(sampling_params, dict):
                raise ValueError(
                    f"Manifest line {line_number}: sampling_params must be an object"
                )
            sampling_params = dict(sampling_params)
            reserved_fields = set(sampling_params) & _RESERVED_SAMPLING_FIELDS
            if reserved_fields:
                raise ValueError(
                    f"Manifest line {line_number}: sampling_params cannot set reserved "
                    f"field(s): {', '.join(sorted(reserved_fields))}"
                )

            for field in _TOP_LEVEL_SAMPLING_FIELDS - {"image_paths"}:
                if field in record:
                    sampling_params[field] = record[field]
            if "image_paths" in record:
                sampling_params["image_path"] = _resolve_image_paths(
                    record["image_paths"], manifest_path.parent, line_number
                )

            requests.append(
                ManifestRequest(
                    request_id=request_id,
                    prompt=prompt,
                    sampling_params=sampling_params,
                )
            )

    if not requests:
        raise ValueError(f"Request manifest contains no requests: {manifest_path}")

    return LoadedRequestManifest(
        path=str(manifest_path),
        sha256=file_sha256(manifest_path),
        requests=requests,
    )
