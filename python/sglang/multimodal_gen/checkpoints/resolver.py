"""Resolve diffusion checkpoints without constructing model components."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import unquote, urlparse

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import validate_repo_id
from safetensors import safe_open

from sglang.srt.model_loader.checkpoint_quantization import (
    resolve_checkpoint_quant_spec,
)

CheckpointSourceKind = Literal["local", "huggingface"]
CheckpointRole = Literal["pipeline", "component", "component_weights", "lora"]

_WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".bin", ".pt", ".pth", ".ckpt")


@dataclass(frozen=True)
class CheckpointSource:
    original: str
    kind: CheckpointSourceKind
    local_path: str | None = None
    repo_id: str | None = None
    revision: str | None = None
    subfolder: str | None = None
    filename: str | None = None


@dataclass(frozen=True)
class CheckpointFile:
    path: str
    size: int | None = None
    blob_id: str | None = None


@dataclass(frozen=True)
class CheckpointInventory:
    source: CheckpointSource
    resolved_revision: str | None
    files: tuple[CheckpointFile, ...]


@dataclass(frozen=True)
class CheckpointRequest:
    name: str
    role: CheckpointRole
    source: str
    component: str | None = None
    revision: str | None = None
    weight_name: str | None = None


@dataclass(frozen=True)
class TensorSummary:
    tensor_count: int
    dtypes: tuple[str, ...]
    key_samples: tuple[str, ...]
    metadata: dict[str, str]
    lora_ranks: tuple[int, ...]


@dataclass(frozen=True)
class ResolvedCheckpoint:
    request: CheckpointRequest
    inventory: CheckpointInventory
    selected_files: tuple[str, ...]
    container_format: str | None
    quantization_method: str | None
    quantization_source: str | None
    tensor_summary: TensorSummary | None


def _validate_relative_hub_path(path: str, field_name: str) -> str:
    normalized = str(PurePosixPath(path))
    pure_path = PurePosixPath(normalized)
    if not path or pure_path.is_absolute() or ".." in pure_path.parts:
        raise ValueError(f"Invalid Hugging Face {field_name}: {path!r}")
    return normalized


def _merge_revision(url_revision: str | None, revision: str | None) -> str | None:
    if url_revision is not None and revision is not None and url_revision != revision:
        raise ValueError(
            f"Checkpoint URL pins revision {url_revision!r}, which conflicts with "
            f"--revision {revision!r}"
        )
    return url_revision or revision


def _parse_huggingface_url(source: str, revision: str | None) -> CheckpointSource:
    parsed = urlparse(source)
    if parsed.netloc.lower() not in ("huggingface.co", "www.huggingface.co"):
        raise ValueError(
            "Only huggingface.co checkpoint URLs are supported; use a local path "
            "or an owner/repo reference for other sources"
        )

    raw_parts = [part for part in parsed.path.split("/") if part]
    if raw_parts and raw_parts[0] in ("datasets", "spaces"):
        raise ValueError(
            "Diffusion checkpoints must come from a Hugging Face model repo"
        )
    if len(raw_parts) < 2:
        raise ValueError(f"Hugging Face checkpoint URL has no model repo: {source!r}")

    repo_id = "/".join(unquote(part) for part in raw_parts[:2])
    validate_repo_id(repo_id)
    action = raw_parts[2] if len(raw_parts) > 2 else None
    if action is None:
        return CheckpointSource(
            original=source,
            kind="huggingface",
            repo_id=repo_id,
            revision=revision,
        )
    if action not in ("tree", "blob", "resolve") or len(raw_parts) < 4:
        raise ValueError(f"Unsupported Hugging Face checkpoint URL: {source!r}")

    url_revision = unquote(raw_parts[3])
    selected_revision = _merge_revision(url_revision, revision)
    tail = "/".join(unquote(part) for part in raw_parts[4:])
    if action == "tree":
        subfolder = _validate_relative_hub_path(tail, "subfolder") if tail else None
        return CheckpointSource(
            original=source,
            kind="huggingface",
            repo_id=repo_id,
            revision=selected_revision,
            subfolder=subfolder,
        )
    if not tail:
        raise ValueError(f"Hugging Face file URL has no filename: {source!r}")
    return CheckpointSource(
        original=source,
        kind="huggingface",
        repo_id=repo_id,
        revision=selected_revision,
        filename=_validate_relative_hub_path(tail, "filename"),
    )


def parse_checkpoint_source(
    source: str,
    *,
    revision: str | None = None,
) -> CheckpointSource:
    """Parse local paths, Hub repo IDs, subfolders, and exact Hub URLs."""
    expanded = os.path.expanduser(source)
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        return _parse_huggingface_url(source, revision)

    looks_local = (
        os.path.exists(expanded)
        or os.path.isabs(expanded)
        or source.startswith(("./", "../", "~"))
    )
    if looks_local:
        return CheckpointSource(
            original=source,
            kind="local",
            local_path=os.path.abspath(expanded),
        )

    parts = source.split("/")
    if len(parts) < 2 or not all(parts[:2]):
        raise ValueError(
            f"Checkpoint source {source!r} is neither a local path nor an "
            "owner/repo Hugging Face reference"
        )
    repo_id = "/".join(parts[:2])
    validate_repo_id(repo_id)
    tail = "/".join(parts[2:]) or None
    filename = (
        _validate_relative_hub_path(tail, "filename")
        if tail is not None and tail.lower().endswith(_WEIGHT_SUFFIXES)
        else None
    )
    subfolder = tail if filename is None else None
    if subfolder is not None:
        subfolder = _validate_relative_hub_path(subfolder, "subfolder")
    return CheckpointSource(
        original=source,
        kind="huggingface",
        repo_id=repo_id,
        revision=revision,
        subfolder=subfolder,
        filename=filename,
    )


def _filter_inventory_files(
    files: tuple[CheckpointFile, ...], source: CheckpointSource
) -> tuple[CheckpointFile, ...]:
    if source.filename is not None:
        selected = tuple(item for item in files if item.path == source.filename)
        if not selected:
            raise FileNotFoundError(
                f"Checkpoint file {source.filename!r} was not found in {source.repo_id}"
            )
        return selected
    if source.subfolder is None:
        return files
    prefix = source.subfolder.rstrip("/") + "/"
    selected = tuple(item for item in files if item.path.startswith(prefix))
    if not selected:
        raise FileNotFoundError(
            f"Checkpoint subfolder {source.subfolder!r} was not found in {source.repo_id}"
        )
    return selected


def resolve_checkpoint_inventory(source: CheckpointSource) -> CheckpointInventory:
    """List checkpoint files and pin a remote source to an immutable revision."""
    if source.kind == "local":
        assert source.local_path is not None
        local_path = Path(source.local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {local_path}")
        if local_path.is_file():
            files = (
                CheckpointFile(path=local_path.name, size=local_path.stat().st_size),
            )
        else:
            files = tuple(
                CheckpointFile(
                    path=path.relative_to(local_path).as_posix(),
                    size=path.stat().st_size,
                )
                for path in sorted(local_path.rglob("*"))
                if path.is_file()
            )
        return CheckpointInventory(
            source=source,
            resolved_revision=None,
            files=files,
        )

    assert source.repo_id is not None
    model_info = HfApi().model_info(
        source.repo_id,
        revision=source.revision,
        files_metadata=True,
    )
    files = tuple(
        CheckpointFile(
            path=sibling.rfilename,
            size=sibling.size,
            blob_id=sibling.blob_id,
        )
        for sibling in model_info.siblings
    )
    return CheckpointInventory(
        source=source,
        resolved_revision=model_info.sha,
        files=_filter_inventory_files(files, source),
    )


def materialize_checkpoint(inventory: CheckpointInventory) -> str:
    """Download the pinned checkpoint selected by an inventory."""
    source = inventory.source
    if source.kind == "local":
        assert source.local_path is not None
        return source.local_path

    assert source.repo_id is not None
    revision = inventory.resolved_revision or source.revision
    if source.filename is not None:
        return hf_hub_download(
            repo_id=source.repo_id,
            filename=source.filename,
            revision=revision,
        )
    allow_patterns = None
    if source.subfolder is not None:
        allow_patterns = [f"{source.subfolder}/**", f"{source.subfolder}/*"]
    local_repo = snapshot_download(
        repo_id=source.repo_id,
        revision=revision,
        allow_patterns=allow_patterns,
    )
    if source.subfolder is None:
        return local_repo
    return os.path.join(local_repo, source.subfolder)


def _local_inventory_file_path(
    inventory: CheckpointInventory, inventory_path: str
) -> str:
    source = inventory.source
    assert source.local_path is not None
    if os.path.isfile(source.local_path):
        return source.local_path
    return os.path.join(source.local_path, inventory_path)


def _materialize_inventory_file(
    inventory: CheckpointInventory, inventory_path: str
) -> str:
    source = inventory.source
    if source.kind == "local":
        return _local_inventory_file_path(inventory, inventory_path)
    assert source.repo_id is not None
    return hf_hub_download(
        repo_id=source.repo_id,
        filename=inventory_path,
        revision=inventory.resolved_revision or source.revision,
    )


def _read_inventory_json(inventory: CheckpointInventory, inventory_path: str) -> dict:
    local_path = _materialize_inventory_file(inventory, inventory_path)
    with open(local_path, encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint metadata must be an object: {inventory_path}")
    return value


def _select_named_file(
    candidates: tuple[str, ...], weight_name: str
) -> tuple[str, ...]:
    exact = tuple(path for path in candidates if path == weight_name)
    if exact:
        return exact
    basename_matches = tuple(
        path for path in candidates if PurePosixPath(path).name == weight_name
    )
    if len(basename_matches) == 1:
        return basename_matches
    if not basename_matches:
        raise FileNotFoundError(
            f"Requested checkpoint weight {weight_name!r} was not found"
        )
    raise ValueError(
        f"Checkpoint weight name {weight_name!r} matches multiple files: "
        f"{list(basename_matches)}"
    )


def select_checkpoint_weight_files(
    request: CheckpointRequest, inventory: CheckpointInventory
) -> tuple[str, ...]:
    """Select weights deterministically; never guess among independent files."""
    candidates = tuple(
        item.path
        for item in inventory.files
        if item.path.lower().endswith(_WEIGHT_SUFFIXES)
    )
    if inventory.source.filename is not None:
        return tuple(item.path for item in inventory.files)
    if request.weight_name is not None:
        return _select_named_file(candidates, request.weight_name)
    if request.role in ("pipeline", "component"):
        return candidates

    index_files = tuple(
        item.path
        for item in inventory.files
        if item.path.endswith(".safetensors.index.json")
    )
    if len(index_files) == 1:
        weight_map = _read_inventory_json(inventory, index_files[0]).get(
            "weight_map", {}
        )
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Checkpoint index has no weight_map: {index_files[0]}")
        index_dir = str(PurePosixPath(index_files[0]).parent)
        prefix = "" if index_dir == "." else f"{index_dir}/"
        selected = tuple(
            sorted({f"{prefix}{str(filename)}" for filename in weight_map.values()})
        )
        missing = [path for path in selected if path not in candidates]
        if missing:
            raise FileNotFoundError(
                f"Checkpoint index {index_files[0]!r} references missing files: {missing}"
            )
        return selected
    if len(index_files) > 1:
        raise ValueError(
            f"Checkpoint contains multiple weight indexes: {list(index_files)}"
        )
    if len(candidates) == 1:
        return candidates
    if not candidates:
        raise FileNotFoundError("Checkpoint contains no recognized weight files")
    raise ValueError(
        "Checkpoint contains multiple independent weight files; select one with "
        f"an exact file URL or weight name. Candidates: {list(candidates)}"
    )


def _lora_rank(key: str, shape: list[int]) -> int | None:
    if len(shape) < 2:
        return None
    if ".lora_A" in key or "lora_down" in key:
        return int(shape[-2])
    if ".lora_B" in key or "lora_up" in key:
        return int(shape[-1])
    return None


def _summarize_local_safetensors(file_path: str) -> TensorSummary:
    with safe_open(file_path, framework="pt", device="cpu") as weights:
        keys = list(weights.keys())
        metadata = weights.metadata() or {}
        dtypes: set[str] = set()
        lora_ranks: set[int] = set()
        for key in keys:
            tensor_slice = weights.get_slice(key)
            dtypes.add(str(tensor_slice.get_dtype()))
            rank = _lora_rank(key, tensor_slice.get_shape())
            if rank is not None:
                lora_ranks.add(rank)
    return TensorSummary(
        tensor_count=len(keys),
        dtypes=tuple(sorted(dtypes)),
        key_samples=tuple(keys[:20]),
        metadata=dict(metadata),
        lora_ranks=tuple(sorted(lora_ranks)),
    )


def _summarize_remote_safetensors(
    inventory: CheckpointInventory, inventory_path: str
) -> TensorSummary:
    source = inventory.source
    assert source.repo_id is not None
    metadata = HfApi().parse_safetensors_file_metadata(
        source.repo_id,
        inventory_path,
        revision=inventory.resolved_revision or source.revision,
    )
    keys = list(metadata.tensors)
    lora_ranks = {
        rank
        for key, tensor in metadata.tensors.items()
        if (rank := _lora_rank(key, tensor.shape)) is not None
    }
    return TensorSummary(
        tensor_count=len(keys),
        dtypes=tuple(sorted(metadata.parameter_count)),
        key_samples=tuple(keys[:20]),
        metadata=dict(metadata.metadata or {}),
        lora_ranks=tuple(sorted(lora_ranks)),
    )


def _merge_tensor_summaries(summaries: list[TensorSummary]) -> TensorSummary | None:
    if not summaries:
        return None
    metadata: dict[str, str] = {}
    for summary in summaries:
        metadata.update(summary.metadata)
    return TensorSummary(
        tensor_count=sum(summary.tensor_count for summary in summaries),
        dtypes=tuple(
            sorted({dtype for summary in summaries for dtype in summary.dtypes})
        ),
        key_samples=tuple(key for summary in summaries for key in summary.key_samples)[
            :20
        ],
        metadata=metadata,
        lora_ranks=tuple(
            sorted({rank for summary in summaries for rank in summary.lora_ranks})
        ),
    )


def _resolve_quantization_metadata(
    inventory: CheckpointInventory,
) -> tuple[str | None, str | None]:
    config_files = tuple(
        item.path
        for item in inventory.files
        if PurePosixPath(item.path).name == "config.json"
    )
    if len(config_files) != 1:
        return None, None
    quant_spec = resolve_checkpoint_quant_spec(
        _read_inventory_json(inventory, config_files[0])
    )
    if quant_spec is None:
        return None, None
    return quant_spec.declared_method, quant_spec.source


def resolve_checkpoint(request: CheckpointRequest) -> ResolvedCheckpoint:
    """Resolve one checkpoint request using the same metadata path as preflight."""
    source = parse_checkpoint_source(request.source, revision=request.revision)
    inventory = resolve_checkpoint_inventory(source)
    selected_files = select_checkpoint_weight_files(request, inventory)
    summaries = []
    summary_files = () if request.role == "pipeline" else selected_files
    for selected_file in summary_files:
        if not selected_file.endswith(".safetensors"):
            continue
        if source.kind == "local":
            summaries.append(
                _summarize_local_safetensors(
                    _local_inventory_file_path(inventory, selected_file)
                )
            )
        else:
            summaries.append(_summarize_remote_safetensors(inventory, selected_file))
    quantization_method, quantization_source = _resolve_quantization_metadata(inventory)
    formats = {PurePosixPath(path).suffix.removeprefix(".") for path in selected_files}
    if len(formats) == 1:
        container_format = next(iter(formats))
    elif formats:
        container_format = "mixed"
    else:
        container_format = None
    return ResolvedCheckpoint(
        request=request,
        inventory=inventory,
        selected_files=selected_files,
        container_format=container_format,
        quantization_method=quantization_method,
        quantization_source=quantization_source,
        tensor_summary=_merge_tensor_summaries(summaries),
    )


def materialize_resolved_checkpoint(checkpoint: ResolvedCheckpoint) -> tuple[str, ...]:
    """Download exactly the files selected by a resolved weights checkpoint."""
    if checkpoint.request.role in ("pipeline", "component"):
        return (materialize_checkpoint(checkpoint.inventory),)
    return tuple(
        _materialize_inventory_file(checkpoint.inventory, selected_file)
        for selected_file in checkpoint.selected_files
    )
