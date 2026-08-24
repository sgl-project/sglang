"""Resolve weight sources for runtime loaders."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import unquote, urlparse

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import validate_repo_id

WeightSourceKind = Literal["local", "huggingface"]

_WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".bin", ".pt", ".pth", ".ckpt")


@dataclass(frozen=True)
class WeightSource:
    original: str
    kind: WeightSourceKind
    local_path: str | None = None
    repo_id: str | None = None
    revision: str | None = None
    subfolder: str | None = None
    filename: str | None = None


@dataclass(frozen=True)
class WeightInventory:
    source: WeightSource
    resolved_revision: str | None
    files: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedWeight:
    inventory: WeightInventory
    selected_file: str


def is_explicit_weight_file_reference(source: str) -> bool:
    """Whether a component override names one weight file, not a component root."""
    expanded = os.path.expanduser(source)
    if os.path.isdir(expanded):
        return False
    return urlparse(source).path.lower().endswith(_WEIGHT_SUFFIXES)


def _validate_relative_hub_path(path: str, field_name: str) -> str:
    normalized = str(PurePosixPath(path))
    pure_path = PurePosixPath(normalized)
    if not path or pure_path.is_absolute() or ".." in pure_path.parts:
        raise ValueError(f"Invalid Hugging Face {field_name}: {path!r}")
    return normalized


def _merge_revision(url_revision: str | None, revision: str | None) -> str | None:
    if url_revision is not None and revision is not None and url_revision != revision:
        raise ValueError(
            f"Weight URL pins revision {url_revision!r}, which conflicts with "
            f"revision {revision!r}"
        )
    return url_revision or revision


def _parse_huggingface_url(source: str, revision: str | None) -> WeightSource:
    parsed = urlparse(source)
    if parsed.netloc.lower() not in ("huggingface.co", "www.huggingface.co"):
        raise ValueError(
            "Only huggingface.co weight URLs are supported; use a local path "
            "or an owner/repo reference for other sources"
        )

    raw_parts = [part for part in parsed.path.split("/") if part]
    if raw_parts and raw_parts[0] in ("datasets", "spaces"):
        raise ValueError("Diffusion weights must come from a Hugging Face model repo")
    if len(raw_parts) < 2:
        raise ValueError(f"Hugging Face weight URL has no model repo: {source!r}")

    repo_id = "/".join(unquote(part) for part in raw_parts[:2])
    validate_repo_id(repo_id)
    action = raw_parts[2] if len(raw_parts) > 2 else None
    if action is None:
        return WeightSource(
            original=source,
            kind="huggingface",
            repo_id=repo_id,
            revision=revision,
        )
    if action not in ("tree", "blob", "resolve") or len(raw_parts) < 4:
        raise ValueError(f"Unsupported Hugging Face weight URL: {source!r}")

    url_revision = unquote(raw_parts[3])
    selected_revision = _merge_revision(url_revision, revision)
    tail = "/".join(unquote(part) for part in raw_parts[4:])
    if action == "tree":
        subfolder = _validate_relative_hub_path(tail, "subfolder") if tail else None
        return WeightSource(
            original=source,
            kind="huggingface",
            repo_id=repo_id,
            revision=selected_revision,
            subfolder=subfolder,
        )
    if not tail:
        raise ValueError(f"Hugging Face weight URL has no filename: {source!r}")
    return WeightSource(
        original=source,
        kind="huggingface",
        repo_id=repo_id,
        revision=selected_revision,
        filename=_validate_relative_hub_path(tail, "filename"),
    )


def parse_weight_source(
    source: str,
    *,
    revision: str | None = None,
) -> WeightSource:
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
        return WeightSource(
            original=source,
            kind="local",
            local_path=os.path.abspath(expanded),
        )

    parts = source.split("/")
    if len(parts) < 2 or not all(parts[:2]):
        raise ValueError(
            f"Weight source {source!r} is neither a local path nor an "
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
    return WeightSource(
        original=source,
        kind="huggingface",
        repo_id=repo_id,
        revision=revision,
        subfolder=subfolder,
        filename=filename,
    )


def _filter_inventory_files(
    files: tuple[str, ...], source: WeightSource
) -> tuple[str, ...]:
    if source.filename is not None:
        selected = tuple(path for path in files if path == source.filename)
        if not selected:
            raise FileNotFoundError(
                f"Weight file {source.filename!r} was not found in {source.repo_id}"
            )
        return selected
    if source.subfolder is None:
        return files
    prefix = source.subfolder.rstrip("/") + "/"
    selected = tuple(path for path in files if path.startswith(prefix))
    if not selected:
        raise FileNotFoundError(
            f"Weight subfolder {source.subfolder!r} was not found in {source.repo_id}"
        )
    return selected


def resolve_weight_inventory(source: WeightSource) -> WeightInventory:
    """List source files and pin a remote source to an immutable revision."""
    if source.kind == "local":
        assert source.local_path is not None
        local_path = Path(source.local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Weight path does not exist: {local_path}")
        if local_path.is_file():
            files = (local_path.name,)
        else:
            files = tuple(
                path.relative_to(local_path).as_posix()
                for path in sorted(local_path.rglob("*"))
                if path.is_file()
            )
        return WeightInventory(
            source=source,
            resolved_revision=None,
            files=files,
        )

    assert source.repo_id is not None
    model_info = HfApi().model_info(
        source.repo_id,
        revision=source.revision,
    )
    files = tuple(sibling.rfilename for sibling in model_info.siblings)
    return WeightInventory(
        source=source,
        resolved_revision=model_info.sha,
        files=_filter_inventory_files(files, source),
    )


def _select_named_file(candidates: tuple[str, ...], weight_name: str) -> str:
    exact = tuple(path for path in candidates if path == weight_name)
    if exact:
        return exact[0]
    basename_matches = tuple(
        path for path in candidates if PurePosixPath(path).name == weight_name
    )
    if len(basename_matches) == 1:
        return basename_matches[0]
    if not basename_matches:
        raise FileNotFoundError(f"Requested weight {weight_name!r} was not found")
    raise ValueError(
        f"Weight name {weight_name!r} matches multiple files: "
        f"{list(basename_matches)}"
    )


def select_weight_file(
    inventory: WeightInventory, weight_name: str | None = None
) -> str:
    """Select weights deterministically; never guess among independent files."""
    candidates = tuple(
        path for path in inventory.files if path.lower().endswith(_WEIGHT_SUFFIXES)
    )
    if inventory.source.filename is not None:
        return inventory.files[0]
    if weight_name is not None:
        return _select_named_file(candidates, weight_name)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError("Source contains no recognized weight files")
    raise ValueError(
        "Source contains multiple independent weight files; select one with "
        f"an exact file URL or weight name. Candidates: {list(candidates)}"
    )


def resolve_weight(
    source: str,
    *,
    revision: str | None = None,
    weight_name: str | None = None,
) -> ResolvedWeight:
    """Resolve one weight file without downloading its tensor payload."""
    parsed_source = parse_weight_source(source, revision=revision)
    inventory = resolve_weight_inventory(parsed_source)
    selected_file = select_weight_file(inventory, weight_name)
    return ResolvedWeight(
        inventory=inventory,
        selected_file=selected_file,
    )


def materialize_weight(resolved: ResolvedWeight) -> str:
    """Return the selected local file, downloading one pinned Hub file if needed."""
    source = resolved.inventory.source
    if source.kind == "local":
        assert source.local_path is not None
        if os.path.isfile(source.local_path):
            return source.local_path
        return os.path.join(source.local_path, resolved.selected_file)

    assert source.repo_id is not None
    return hf_hub_download(
        repo_id=source.repo_id,
        filename=resolved.selected_file,
        revision=resolved.inventory.resolved_revision,
    )
