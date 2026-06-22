#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Synchronize pinned third-party native source checkouts for OmniDreams."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "thirdparty_sources.json"
DEFAULT_DEST_ROOT = ROOT / "3rdparty"
STAMP_NAME = ".flashdreams_source.json"
SCHEMA_VERSION = 1


class ThirdPartySyncError(RuntimeError):
    """Raised when a managed third-party checkout cannot be synchronized."""


@dataclass(frozen=True)
class PatchSpec:
    path: Path
    strip: int = 1


@dataclass(frozen=True)
class OverlaySpec:
    source: Path
    destination: str


@dataclass(frozen=True)
class SourceSpec:
    name: str
    repo: str
    commit: str
    directory: str
    delete_paths: tuple[str, ...] = ()
    patches: tuple[PatchSpec, ...] = ()
    overlays: tuple[OverlaySpec, ...] = ()

    @property
    def destination_name(self) -> str:
        return self.directory


@dataclass(frozen=True)
class SyncResult:
    source: SourceSpec
    path: Path
    commit: str
    changed: bool


def _run(
    args: Sequence[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
) -> str:
    proc = subprocess.run(
        list(args),
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if check and proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip()
        where = f" in {cwd}" if cwd is not None else ""
        raise ThirdPartySyncError(f"{' '.join(args)} failed{where}: {message}")
    return proc.stdout.strip()


def _run_git(cwd: Path, args: Sequence[str], *, check: bool = True) -> str:
    return _run(["git", *args], cwd=cwd, check=check)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_tree(path: Path, *, exclude_top_level: set[str] | None = None) -> str:
    digest = hashlib.sha256()
    excluded = exclude_top_level or set()
    if not path.exists():
        return digest.hexdigest()
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = file_path.relative_to(path).as_posix()
        if rel.split("/", 1)[0] in excluded:
            continue
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_hash_file(file_path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_relative_path(path: str, *, field: str) -> str:
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or ".." in parsed.parts or not parsed.parts:
        raise ThirdPartySyncError(f"Invalid {field} path: {path!r}")
    return parsed.as_posix()


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def _as_mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThirdPartySyncError(f"{context} must be an object")
    return cast(Mapping[str, Any], value)


def _source_from_mapping(
    manifest_path: Path,
    value: Mapping[str, Any],
) -> SourceSpec:
    try:
        name = str(value["name"])
        repo = str(value["repo"])
        commit = str(value["commit"])
    except KeyError as exc:
        raise ThirdPartySyncError(
            f"Missing required source field: {exc.args[0]}"
        ) from exc

    directory = str(value.get("directory", name))
    _validate_relative_path(directory, field=f"{name}.directory")

    delete_paths = tuple(
        _validate_relative_path(str(item), field=f"{name}.delete_paths")
        for item in value.get("delete_paths", ())
    )

    patches: list[PatchSpec] = []
    for item in value.get("patches", ()):
        patch = _as_mapping(item, context=f"{name}.patch")
        if "path" not in patch:
            raise ThirdPartySyncError(f"{name}.patch is missing path")
        patches.append(
            PatchSpec(
                path=_resolve_manifest_path(manifest_path, str(patch["path"])),
                strip=int(patch.get("strip", 1)),
            )
        )

    overlays: list[OverlaySpec] = []
    for item in value.get("overlays", ()):
        overlay = _as_mapping(item, context=f"{name}.overlay")
        if "source" not in overlay or "destination" not in overlay:
            raise ThirdPartySyncError(f"{name}.overlay requires source and destination")
        overlays.append(
            OverlaySpec(
                source=_resolve_manifest_path(manifest_path, str(overlay["source"])),
                destination=_validate_relative_path(
                    str(overlay["destination"]),
                    field=f"{name}.overlay.destination",
                ),
            )
        )

    return SourceSpec(
        name=name,
        repo=repo,
        commit=commit,
        directory=directory,
        delete_paths=delete_paths,
        patches=tuple(patches),
        overlays=tuple(overlays),
    )


def load_manifest(path: Path = DEFAULT_MANIFEST) -> tuple[SourceSpec, ...]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError as exc:
        raise ThirdPartySyncError(f"Manifest does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ThirdPartySyncError(f"Invalid manifest JSON at {path}: {exc}") from exc
    manifest = _as_mapping(data, context=str(path))

    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ThirdPartySyncError(
            f"{path} schema_version must be {SCHEMA_VERSION}, "
            f"got {manifest.get('schema_version')!r}"
        )

    sources = manifest.get("sources", ())
    if not isinstance(sources, list):
        raise ThirdPartySyncError(f"{path} sources must be a list")

    parsed = tuple(
        _source_from_mapping(path, _as_mapping(item, context="source"))
        for item in sources
    )
    names = [source.name for source in parsed]
    if len(set(names)) != len(names):
        raise ThirdPartySyncError(f"{path} contains duplicate source names")
    return parsed


def _source_hash(source: SourceSpec) -> str:
    data = {
        "name": source.name,
        "repo": source.repo,
        "commit": source.commit,
        "directory": source.directory,
        "delete_paths": list(source.delete_paths),
        "patches": [
            {
                "path": str(patch.path),
                "strip": patch.strip,
                "sha256": _hash_file(patch.path),
            }
            for patch in source.patches
        ],
        "overlays": [
            {
                "source": str(overlay.source),
                "destination": overlay.destination,
                "sha256": _hash_tree(overlay.source),
            }
            for overlay in source.overlays
        ],
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


def _source_tree_hash(path: Path) -> str:
    return _hash_tree(path, exclude_top_level={".git", STAMP_NAME})


def _stamp_metadata(source: SourceSpec) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "name": source.name,
        "repo": source.repo,
        "commit": source.commit,
        "directory": source.directory,
        "source_sha256": _source_hash(source),
    }


def _stamp_path(path: Path) -> Path:
    return path / STAMP_NAME


def _read_stamp(path: Path) -> Mapping[str, object] | None:
    try:
        with _stamp_path(path).open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        raise ThirdPartySyncError(f"Invalid stamp file at {_stamp_path(path)}: {exc}")
    return _as_mapping(data, context=str(_stamp_path(path)))


def _write_stamp(path: Path, source: SourceSpec) -> None:
    stamp = _stamp_metadata(source)
    stamp["tree_sha256"] = _source_tree_hash(path)
    with _stamp_path(path).open("w", encoding="utf-8") as fh:
        json.dump(stamp, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _is_git_checkout(path: Path) -> bool:
    return (path / ".git").exists()


def _clone_source(source: SourceSpec, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            source.repo,
            str(path),
        ]
    )


def _sync_git_checkout(source: SourceSpec, path: Path, *, force: bool) -> bool:
    if path.exists():
        if not _is_git_checkout(path):
            if force:
                shutil.rmtree(path)
                _clone_source(source, path)
                return True
            raise ThirdPartySyncError(
                f"{path} exists but is not a Git checkout; remove it or pass --force"
            )
        if _read_stamp(path) is None and not force:
            raise ThirdPartySyncError(
                f"{path} is not marked as managed by {STAMP_NAME}; pass --force to replace it"
            )
        origin_url = _run_git(path, ["remote", "get-url", "origin"])
        if origin_url != source.repo and not force:
            raise ThirdPartySyncError(
                f"{path} origin is {origin_url!r}, expected {source.repo!r}; pass --force to replace it"
            )
        if force and origin_url != source.repo:
            shutil.rmtree(path)
            _clone_source(source, path)
            return True
        _run_git(path, ["fetch", "--quiet", "--filter=blob:none", "--tags", "origin"])
        return False

    _clone_source(source, path)
    return True


def _checkout_commit(source: SourceSpec, path: Path) -> None:
    _run_git(path, ["checkout", "--quiet", "--force", source.commit])
    head = _run_git(path, ["rev-parse", "HEAD"])
    if head != source.commit:
        raise ThirdPartySyncError(
            f"{path} checked out {head}, expected {source.commit}"
        )
    _run_git(path, ["reset", "--hard", "--quiet", source.commit])
    _run_git(path, ["clean", "-ffdx", "--quiet"])


def _delete_paths(source: SourceSpec, path: Path) -> None:
    for rel in source.delete_paths:
        target = path / rel
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def _apply_patches(source: SourceSpec, path: Path) -> None:
    for patch in source.patches:
        if not patch.path.is_file():
            raise ThirdPartySyncError(
                f"{source.name} patch does not exist: {patch.path}"
            )
        for args in (
            ["apply", f"-p{patch.strip}", "--check", str(patch.path)],
            ["apply", f"-p{patch.strip}", str(patch.path)],
        ):
            try:
                _run_git(path, args)
            except ThirdPartySyncError as exc:
                raise ThirdPartySyncError(
                    f"Failed to apply {patch.path} to {path}"
                ) from exc


def _copy_overlays(source: SourceSpec, path: Path) -> None:
    for overlay in source.overlays:
        if not overlay.source.exists():
            raise ThirdPartySyncError(
                f"{source.name} overlay source does not exist: {overlay.source}"
            )
        destination = path / overlay.destination
        if overlay.source.is_dir():
            shutil.copytree(overlay.source, destination, dirs_exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(overlay.source, destination)


def sync_source(
    source: SourceSpec, dest_root: Path, *, force: bool = False
) -> SyncResult:
    path = dest_root / source.destination_name
    created_or_replaced = _sync_git_checkout(source, path, force=force)
    _checkout_commit(source, path)
    _delete_paths(source, path)
    _apply_patches(source, path)
    _copy_overlays(source, path)
    _write_stamp(path, source)
    return SyncResult(
        source=source,
        path=path,
        commit=source.commit,
        changed=created_or_replaced,
    )


def sync_sources(
    sources: Sequence[SourceSpec],
    dest_root: Path = DEFAULT_DEST_ROOT,
    *,
    selected: set[str] | None = None,
    force: bool = False,
) -> tuple[SyncResult, ...]:
    results: list[SyncResult] = []
    for source in sources:
        if selected is not None and source.name not in selected:
            continue
        results.append(sync_source(source, dest_root, force=force))
    return tuple(results)


def verify_source(source: SourceSpec, dest_root: Path) -> SyncResult:
    path = dest_root / source.destination_name
    if not path.exists():
        raise ThirdPartySyncError(f"{source.name} is missing: {path}")
    if not _is_git_checkout(path):
        raise ThirdPartySyncError(f"{source.name} is not a Git checkout: {path}")
    head = _run_git(path, ["rev-parse", "HEAD"])
    if head != source.commit:
        raise ThirdPartySyncError(
            f"{source.name} is at {head}, expected {source.commit}"
        )
    stamp = _read_stamp(path)
    if stamp is None:
        raise ThirdPartySyncError(
            f"{source.name} is missing {STAMP_NAME}; run sync again"
        )
    expected_stamp = _stamp_metadata(source)
    for key, value in expected_stamp.items():
        if stamp.get(key) != value:
            raise ThirdPartySyncError(
                f"{source.name} stamp does not match manifest; run sync again"
            )
    recorded_tree_hash = stamp.get("tree_sha256")
    if not isinstance(recorded_tree_hash, str):
        raise ThirdPartySyncError(
            f"{source.name} stamp does not match manifest; run sync again"
        )
    actual_tree_hash = _source_tree_hash(path)
    if actual_tree_hash != recorded_tree_hash:
        raise ThirdPartySyncError(
            f"{source.name} source tree does not match its stamp; run sync again"
        )
    return SyncResult(source=source, path=path, commit=head, changed=False)


def verify_sources(
    sources: Sequence[SourceSpec],
    dest_root: Path = DEFAULT_DEST_ROOT,
    *,
    selected: set[str] | None = None,
) -> tuple[SyncResult, ...]:
    results: list[SyncResult] = []
    for source in sources:
        if selected is not None and source.name not in selected:
            continue
        results.append(verify_source(source, dest_root))
    return tuple(results)


def _selected_sources(
    names: Sequence[str], sources: Sequence[SourceSpec]
) -> set[str] | None:
    if not names:
        return None
    available = {source.name for source in sources}
    selected = set(names)
    unknown = selected - available
    if unknown:
        raise ThirdPartySyncError(f"Unknown source(s): {', '.join(sorted(unknown))}")
    return selected


def _print_results(results: Sequence[SyncResult]) -> None:
    for result in results:
        print(f"{result.source.name}: {result.commit} -> {result.path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=("list", "sync", "verify"),
        default="sync",
        help="operation to perform (default: sync)",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dest-root", type=Path, default=DEFAULT_DEST_ROOT)
    parser.add_argument(
        "--source", action="append", default=[], help="source name to operate on"
    )
    parser.add_argument(
        "--force", action="store_true", help="replace conflicting managed checkouts"
    )
    args = parser.parse_args(argv)

    try:
        sources = load_manifest(args.manifest)
        selected = _selected_sources(args.source, sources)
        if args.command == "list":
            _print_results(
                tuple(
                    SyncResult(
                        source=s,
                        path=args.dest_root / s.destination_name,
                        commit=s.commit,
                        changed=False,
                    )
                    for s in sources
                    if selected is None or s.name in selected
                )
            )
            return 0
        if args.command == "verify":
            _print_results(verify_sources(sources, args.dest_root, selected=selected))
            return 0
        _print_results(
            sync_sources(sources, args.dest_root, selected=selected, force=args.force)
        )
        return 0
    except ThirdPartySyncError as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
