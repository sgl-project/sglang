"""Resolve diffusion artifact sources without constructing model components."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import unquote, urlparse

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import validate_repo_id

ArtifactSourceKind = Literal["local", "huggingface"]


@dataclass(frozen=True)
class ArtifactSource:
    original: str
    kind: ArtifactSourceKind
    local_path: str | None = None
    repo_id: str | None = None
    revision: str | None = None
    subfolder: str | None = None
    filename: str | None = None


@dataclass(frozen=True)
class ArtifactFile:
    path: str
    size: int | None = None
    blob_id: str | None = None


@dataclass(frozen=True)
class ArtifactInventory:
    source: ArtifactSource
    resolved_revision: str | None
    files: tuple[ArtifactFile, ...]


def _validate_relative_hub_path(path: str, field_name: str) -> str:
    normalized = str(PurePosixPath(path))
    pure_path = PurePosixPath(normalized)
    if not path or pure_path.is_absolute() or ".." in pure_path.parts:
        raise ValueError(f"Invalid Hugging Face {field_name}: {path!r}")
    return normalized


def _merge_revision(url_revision: str | None, revision: str | None) -> str | None:
    if url_revision is not None and revision is not None and url_revision != revision:
        raise ValueError(
            f"Artifact URL pins revision {url_revision!r}, which conflicts with "
            f"--revision {revision!r}"
        )
    return url_revision or revision


def _parse_huggingface_url(source: str, revision: str | None) -> ArtifactSource:
    parsed = urlparse(source)
    if parsed.netloc.lower() not in ("huggingface.co", "www.huggingface.co"):
        raise ValueError(
            "Only huggingface.co artifact URLs are supported; use a local path "
            "or an owner/repo reference for other sources"
        )

    raw_parts = [part for part in parsed.path.split("/") if part]
    if raw_parts and raw_parts[0] in ("datasets", "spaces"):
        raise ValueError("Diffusion artifacts must come from a Hugging Face model repo")
    if len(raw_parts) < 2:
        raise ValueError(f"Hugging Face artifact URL has no model repo: {source!r}")

    repo_id = "/".join(unquote(part) for part in raw_parts[:2])
    validate_repo_id(repo_id)
    action = raw_parts[2] if len(raw_parts) > 2 else None
    if action is None:
        return ArtifactSource(
            original=source,
            kind="huggingface",
            repo_id=repo_id,
            revision=revision,
        )
    if action not in ("tree", "blob", "resolve") or len(raw_parts) < 4:
        raise ValueError(f"Unsupported Hugging Face artifact URL: {source!r}")

    url_revision = unquote(raw_parts[3])
    selected_revision = _merge_revision(url_revision, revision)
    tail = "/".join(unquote(part) for part in raw_parts[4:])
    if action == "tree":
        subfolder = _validate_relative_hub_path(tail, "subfolder") if tail else None
        return ArtifactSource(
            original=source,
            kind="huggingface",
            repo_id=repo_id,
            revision=selected_revision,
            subfolder=subfolder,
        )
    if not tail:
        raise ValueError(f"Hugging Face file URL has no filename: {source!r}")
    return ArtifactSource(
        original=source,
        kind="huggingface",
        repo_id=repo_id,
        revision=selected_revision,
        filename=_validate_relative_hub_path(tail, "filename"),
    )


def parse_artifact_source(
    source: str,
    *,
    revision: str | None = None,
) -> ArtifactSource:
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
        return ArtifactSource(
            original=source,
            kind="local",
            local_path=os.path.abspath(expanded),
        )

    parts = source.split("/")
    if len(parts) < 2 or not all(parts[:2]):
        raise ValueError(
            f"Artifact source {source!r} is neither a local path nor an "
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
    return ArtifactSource(
        original=source,
        kind="huggingface",
        repo_id=repo_id,
        revision=revision,
        subfolder=subfolder,
        filename=filename,
    )


def _filter_inventory_files(
    files: tuple[ArtifactFile, ...], source: ArtifactSource
) -> tuple[ArtifactFile, ...]:
    if source.filename is not None:
        selected = tuple(item for item in files if item.path == source.filename)
        if not selected:
            raise FileNotFoundError(
                f"Artifact file {source.filename!r} was not found in {source.repo_id}"
            )
        return selected
    if source.subfolder is None:
        return files
    prefix = source.subfolder.rstrip("/") + "/"
    selected = tuple(item for item in files if item.path.startswith(prefix))
    if not selected:
        raise FileNotFoundError(
            f"Artifact subfolder {source.subfolder!r} was not found in {source.repo_id}"
        )
    return selected


def resolve_artifact_inventory(source: ArtifactSource) -> ArtifactInventory:
    """List artifact files and pin a remote source to an immutable revision."""
    if source.kind == "local":
        assert source.local_path is not None
        local_path = Path(source.local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Artifact path does not exist: {local_path}")
        if local_path.is_file():
            files = (
                ArtifactFile(path=local_path.name, size=local_path.stat().st_size),
            )
        else:
            files = tuple(
                ArtifactFile(
                    path=path.relative_to(local_path).as_posix(),
                    size=path.stat().st_size,
                )
                for path in sorted(local_path.rglob("*"))
                if path.is_file()
            )
        return ArtifactInventory(
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
        ArtifactFile(
            path=sibling.rfilename,
            size=sibling.size,
            blob_id=sibling.blob_id,
        )
        for sibling in model_info.siblings
    )
    return ArtifactInventory(
        source=source,
        resolved_revision=model_info.sha,
        files=_filter_inventory_files(files, source),
    )


def materialize_artifact(inventory: ArtifactInventory) -> str:
    """Download the pinned artifact selected by an inventory."""
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
