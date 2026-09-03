"""Resolve weight sources for runtime loaders."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import unquote, urlparse

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import validate_repo_id

WeightSourceKind = Literal["local", "huggingface"]

_WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".bin", ".pt", ".pth", ".ckpt")
_SAFETENSORS_INDEX_SUFFIX = ".safetensors.index.json"
_WEIGHT_REFERENCE_SUFFIXES = _WEIGHT_SUFFIXES + (_SAFETENSORS_INDEX_SUFFIX,)


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


@dataclass(frozen=True)
class ResolvedWeightSet:
    inventory: WeightInventory
    selected_files: tuple[str, ...]
    index_file: str | None = None


class NoSafetensorsWeightsError(FileNotFoundError):
    """The source has no safetensors payload to resolve as a weight set."""


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


def _local_index_inventory(index_path: Path) -> tuple[str, ...]:
    """List only an exact local index and the shards it declares."""
    with index_path.open(encoding="utf-8") as index_stream:
        index = json.load(index_stream)
    weight_map = index.get("weight_map") if isinstance(index, Mapping) else None
    shard_names = weight_map.values() if isinstance(weight_map, Mapping) else ()
    files = {index_path.name}
    for shard_name in shard_names:
        if not isinstance(shard_name, str):
            continue
        shard_name = _validate_relative_hub_path(shard_name, "index shard")
        shard_path = index_path.parent / shard_name
        if shard_path.is_file():
            files.add(shard_path.relative_to(index_path.parent).as_posix())
    return tuple(sorted(files))


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
        if tail is not None and tail.lower().endswith(_WEIGHT_REFERENCE_SUFFIXES)
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
        if source.filename not in files:
            raise FileNotFoundError(
                f"Weight file {source.filename!r} was not found in {source.repo_id}"
            )
        return (source.filename,)
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
        f"Weight name {weight_name!r} matches multiple files: {list(basename_matches)}"
    )


def select_weight_file(
    inventory: WeightInventory, weight_name: str | None = None
) -> str:
    """Select weights deterministically; never guess among independent files."""
    candidates = tuple(
        path for path in inventory.files if path.lower().endswith(_WEIGHT_SUFFIXES)
    )
    if inventory.source.filename is not None:
        return _select_named_file(candidates, inventory.source.filename)
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


def _materialize_inventory_file(inventory: WeightInventory, filename: str) -> str:
    source = inventory.source
    if source.kind == "local":
        assert source.local_path is not None
        if os.path.isfile(source.local_path):
            local_path = Path(source.local_path)
            if filename == local_path.name:
                return source.local_path
            return str(local_path.parent / filename)
        return os.path.join(source.local_path, filename)

    assert source.repo_id is not None
    return hf_hub_download(
        repo_id=source.repo_id,
        filename=filename,
        revision=inventory.resolved_revision,
    )


def _resolve_index_shard(
    inventory: WeightInventory, index_file: str, shard_name: str
) -> str:
    shard_name = _validate_relative_hub_path(shard_name, "index shard")
    if not shard_name.lower().endswith(".safetensors"):
        raise ValueError(
            f"Safetensors index {index_file!r} references a non-safetensors "
            f"shard: {shard_name!r}"
        )
    index_parent = PurePosixPath(index_file).parent
    candidates = (shard_name, (index_parent / shard_name).as_posix())
    matches = tuple(
        dict.fromkeys(path for path in candidates if path in inventory.files)
    )
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Safetensors index {index_file!r} references missing shard {shard_name!r}"
        )
    raise ValueError(
        f"Safetensors index shard {shard_name!r} is ambiguous: {list(matches)}"
    )


def _read_safetensors_index(
    inventory: WeightInventory, index_file: str
) -> tuple[str, ...]:
    with open(
        _materialize_inventory_file(inventory, index_file), encoding="utf-8"
    ) as index_stream:
        index = json.load(index_stream)
    weight_map = index.get("weight_map") if isinstance(index, Mapping) else None
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError(
            f"Safetensors index {index_file!r} must contain a non-empty weight_map"
        )
    shard_names = tuple(weight_map.values())
    if not all(isinstance(name, str) for name in shard_names):
        raise ValueError(
            f"Safetensors index {index_file!r} contains a non-string shard name"
        )
    return tuple(
        sorted(
            {
                _resolve_index_shard(inventory, index_file, shard_name)
                for shard_name in shard_names
            }
        )
    )


def resolve_safetensors_weight_set(
    source: str,
    *,
    revision: str | None = None,
    weight_name: str | None = None,
    select_unindexed_weight: Callable[[tuple[str, ...]], str | None] | None = None,
) -> ResolvedWeightSet:
    """Resolve one safetensors checkpoint, using its index as shard authority."""
    parsed_source = parse_weight_source(source, revision=revision)
    if (
        parsed_source.kind == "local"
        and parsed_source.local_path is not None
        and os.path.isfile(parsed_source.local_path)
        and parsed_source.local_path.lower().endswith(_SAFETENSORS_INDEX_SUFFIX)
    ):
        inventory = WeightInventory(
            parsed_source,
            None,
            _local_index_inventory(Path(parsed_source.local_path)),
        )
    elif parsed_source.kind == "huggingface" and parsed_source.filename is not None:
        parent = PurePosixPath(parsed_source.filename).parent
        scoped_inventory = resolve_weight_inventory(
            replace(
                parsed_source,
                filename=None,
                subfolder=None if parent == PurePosixPath(".") else parent.as_posix(),
            )
        )
        inventory = WeightInventory(
            parsed_source,
            scoped_inventory.resolved_revision,
            scoped_inventory.files,
        )
    else:
        inventory = resolve_weight_inventory(parsed_source)
    weights = tuple(
        path for path in inventory.files if path.lower().endswith(".safetensors")
    )
    indexes = tuple(
        path
        for path in inventory.files
        if path.lower().endswith(_SAFETENSORS_INDEX_SUFFIX)
    )
    selected_name = inventory.source.filename or weight_name
    if (
        selected_name is None
        and inventory.source.kind == "local"
        and inventory.source.local_path is not None
        and os.path.isfile(inventory.source.local_path)
    ):
        selected_name = Path(inventory.source.local_path).name
    if selected_name is not None:
        if not selected_name.lower().endswith(
            (".safetensors", _SAFETENSORS_INDEX_SUFFIX)
        ):
            raise NoSafetensorsWeightsError(
                f"Selected file is not safetensors: {selected_name!r}"
            )
        selected = _select_named_file(weights + indexes, selected_name)
        if selected in weights:
            return ResolvedWeightSet(inventory, (selected,))
        return ResolvedWeightSet(
            inventory,
            _read_safetensors_index(inventory, selected),
            index_file=selected,
        )
    if len(indexes) == 1:
        return ResolvedWeightSet(
            inventory,
            _read_safetensors_index(inventory, indexes[0]),
            index_file=indexes[0],
        )
    if len(indexes) > 1:
        raise ValueError(
            "Source contains multiple safetensors indexes; select one with an "
            f"exact index name. Candidates: {list(indexes)}"
        )
    if len(weights) == 1:
        return ResolvedWeightSet(inventory, weights)
    if not weights:
        raise NoSafetensorsWeightsError("Source contains no safetensors weights")
    if select_unindexed_weight is not None:
        selected = select_unindexed_weight(weights)
        if selected is not None:
            if selected not in weights:
                raise ValueError(
                    "Unindexed weight selector returned a file outside the source: "
                    f"{selected!r}"
                )
            return ResolvedWeightSet(inventory, (selected,))
    raise ValueError(
        "Source contains multiple safetensors files without an index; they may "
        "be independent variants. Select one exact file or provide a standard "
        f"safetensors index. Candidates: {list(weights)}"
    )


def materialize_weight_set(resolved: ResolvedWeightSet) -> tuple[str, ...]:
    """Materialize every selected file from one pinned checkpoint revision."""
    return tuple(
        _materialize_inventory_file(resolved.inventory, filename)
        for filename in resolved.selected_files
    )


def materialize_weight_set_config(resolved: ResolvedWeightSet) -> str | None:
    """Materialize adjacent runtime configuration from the pinned revision."""
    source = resolved.inventory.source
    if source.kind == "local" and source.local_path is not None:
        local_path = Path(source.local_path)
        if local_path.is_file():
            config_path = local_path.with_name("config.json")
            return str(config_path) if config_path.is_file() else None

    anchor = resolved.index_file or resolved.selected_files[0]
    config_file = PurePosixPath(anchor).with_name("config.json").as_posix()
    if config_file not in resolved.inventory.files:
        return None
    config_parent = PurePosixPath(config_file).parent
    metadata_files = tuple(
        path
        for path in resolved.inventory.files
        if PurePosixPath(path).parent == config_parent
        and PurePosixPath(path).name.startswith("quant_model_description")
        and path.lower().endswith(".json")
    )
    config_path = _materialize_inventory_file(resolved.inventory, config_file)
    for metadata_file in metadata_files:
        _materialize_inventory_file(resolved.inventory, metadata_file)
    return config_path


def materialize_weight(resolved: ResolvedWeight) -> str:
    """Return the selected local file, downloading one pinned Hub file if needed."""
    return _materialize_inventory_file(resolved.inventory, resolved.selected_file)
