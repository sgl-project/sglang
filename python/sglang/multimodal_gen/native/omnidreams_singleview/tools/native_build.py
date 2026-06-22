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

"""Build preparation helpers for the OmniDreams single-view native path."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
THIRDPARTY_DIR = ROOT / "3rdparty"
CUTLASS_DIR = THIRDPARTY_DIR / "cutlass"
SAGE_ATTENTION_DIR = THIRDPARTY_DIR / "SageAttention"
SYNC_THIRDPARTY_PATH = ROOT / "tools" / "sync_thirdparty.py"
STAMP_NAME = ".flashdreams_source.json"

_DEFAULT_BUILD_ROOT_ENV = "OMNIDREAMS_SINGLEVIEW_NATIVE_BUILD_ROOT"
_sync_thirdparty_module: ModuleType | None = None


class NativeBuildError(RuntimeError):
    """Raised when native build preparation cannot safely continue."""


@dataclass(frozen=True)
class SourceInfo:
    """Validated script-managed native source checkout state."""

    name: str
    path: Path
    repo: str
    commit: str
    source_sha256: str
    tree_sha256: str
    stamp_path: Path

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "path": str(self.path),
            "repo": self.repo,
            "commit": self.commit,
            "source_sha256": self.source_sha256,
            "tree_sha256": self.tree_sha256,
            "stamp_path": str(self.stamp_path),
        }


def _sync_tool() -> ModuleType:
    global _sync_thirdparty_module
    if _sync_thirdparty_module is not None:
        return _sync_thirdparty_module

    spec = importlib.util.spec_from_file_location(
        "omnidreams_singleview_sync_thirdparty",
        SYNC_THIRDPARTY_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot import third-party sync helpers from {SYNC_THIRDPARTY_PATH}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _sync_thirdparty_module = module
    return module


def _stamp_path(path: Path) -> Path:
    return path / STAMP_NAME


def _read_stamp(path: Path) -> Mapping[str, Any]:
    stamp_path = _stamp_path(path)
    try:
        with stamp_path.open("r", encoding="utf-8") as fh:
            stamp = json.load(fh)
    except FileNotFoundError as exc:
        raise NativeBuildError(
            f"{path} is missing {STAMP_NAME}; run {SYNC_THIRDPARTY_PATH} sync"
        ) from exc
    except json.JSONDecodeError as exc:
        raise NativeBuildError(
            f"Invalid native source stamp at {stamp_path}: {exc}"
        ) from exc
    if not isinstance(stamp, Mapping):
        raise NativeBuildError(
            f"Invalid native source stamp at {stamp_path}: not an object"
        )
    return stamp


def _stamp_text(stamp: Mapping[str, Any], key: str, path: Path) -> str:
    value = stamp.get(key)
    if not isinstance(value, str):
        raise NativeBuildError(f"{_stamp_path(path)} is missing string field {key!r}")
    return value


def _source_info(source: Any, path: Path) -> SourceInfo:
    stamp = _read_stamp(path)
    return SourceInfo(
        name=source.name,
        path=path,
        repo=source.repo,
        commit=source.commit,
        source_sha256=_stamp_text(stamp, "source_sha256", path),
        tree_sha256=_stamp_text(stamp, "tree_sha256", path),
        stamp_path=_stamp_path(path),
    )


def _sources() -> tuple[Any, ...]:
    return tuple(_sync_tool().load_manifest())


def sync_thirdparty(*, force: bool = False) -> dict[str, SourceInfo]:
    """Synchronize native source checkouts and return their pinned provenance."""

    try:
        tool = _sync_tool()
        sources = _sources()
        results = tool.sync_sources(sources, force=force)
        return {
            result.source.name: _source_info(result.source, result.path)
            for result in results
        }
    except NativeBuildError:
        raise
    except Exception as exc:
        raise NativeBuildError(str(exc)) from exc


def validate_thirdparty() -> dict[str, SourceInfo]:
    """Validate native source checkouts and return their pinned provenance."""

    try:
        tool = _sync_tool()
        sources = _sources()
        results = tool.verify_sources(sources)
        return {
            result.source.name: _source_info(result.source, result.path)
            for result in results
        }
    except NativeBuildError:
        raise
    except Exception as exc:
        raise NativeBuildError(str(exc)) from exc


def _default_build_root() -> Path:
    override = os.environ.get(_DEFAULT_BUILD_ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return ROOT / "build"


def resolve_build_root(build_root: Path | str | None = None) -> Path:
    """Return the native build root, honoring the environment override."""

    return (
        Path(build_root).resolve() if build_root is not None else _default_build_root()
    )


def torch_extension_build_dir(
    extension_name: str,
    build_root: Path | str | None = None,
) -> Path:
    """Return the colocated PyTorch extension build directory."""

    return resolve_build_root(build_root) / "torch_extensions" / extension_name


def native_provenance(build_root: Path | str | None = None) -> dict[str, object]:
    """Return source provenance without compiling the native extension."""

    thirdparty = validate_thirdparty()
    return {
        "root": str(ROOT),
        "build_root": str(resolve_build_root(build_root)),
        "thirdparty": {name: info.as_dict() for name, info in thirdparty.items()},
        "cutlass_include": str(CUTLASS_DIR / "include"),
    }
