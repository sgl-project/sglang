# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""On-disk layout of a CUDA graph artifact (design section 11).

::

    <root>/<fingerprint digest>/
      manifest.json                 ArtifactManifest; written last by rank 0
                                    after a CPU-group barrier
      images/<sha256>.img           kernel container bytes, content-addressed,
                                    shared by every rank
      rank_<world rank>/
        rank.json                   RankManifest
        <runner>.msgpack            RunnerBundle (ShapeArtifacts + comm blobs)
        diagnostics.json            CoverageReport
      .tmp-<pid>-rank_<world rank>/ staging; renamed atomically onto rank_<r>

Bundles are msgpack (bulk binary node parameters); the three index files are
pretty-printed JSON so an operator can read them. Every write is atomic with
respect to readers: a rank's files are staged in a sibling ``.tmp-<pid>-...``
directory and the whole directory is renamed onto ``rank_<r>`` in one
``os.replace``, so a concurrent loader sees either no ``rank_<r>`` (the old one
is removed just before the rename) or a complete one, never a half-written one. Images and the top-level
manifest are single files and get the same tmp + ``os.replace`` treatment.

Constructing a store has no side effects: directories appear only inside the
``write_*`` / ``put_*`` methods. Nothing in this module touches the GPU.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import msgspec

from sglang.srt.model_executor.graph_serialization.format import (
    FORMAT_VERSION,
    ArtifactManifest,
    RankManifest,
    RunnerBundle,
)

logger = logging.getLogger(__name__)

MANIFEST_FILE = "manifest.json"
RANK_MANIFEST_FILE = "rank.json"
DIAGNOSTICS_FILE = "diagnostics.json"
IMAGES_DIR = "images"
IMAGE_SUFFIX = ".img"
BUNDLE_SUFFIX = ".msgpack"
STAGING_PREFIX = ".tmp-"

# Deterministic key order so two ranks writing equal records produce equal
# bytes (the fingerprint digest uses the same setting).
_JSON = msgspec.json.Encoder(order="deterministic")
_MSGPACK = msgspec.msgpack.Encoder()


class _FormatVersionProbe(msgspec.Struct, frozen=True):
    """Reads only ``format_version`` so a manifest written by a newer build is
    rejected by version before its (possibly unknown) fields are decoded."""

    format_version: int = FORMAT_VERSION


def _json_pretty(obj: Any) -> bytes:
    return msgspec.json.format(_JSON.encode(obj), indent=2)


def _write_bytes(path: Path, data: bytes) -> None:
    with open(path, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _check_component(kind: str, name: str) -> None:
    """A runner name, digest or image key is used verbatim as one path
    component; refuse anything that could escape the artifact directory."""
    if not name or name != Path(name).name or name.startswith("."):
        raise ValueError(f"{kind} {name!r} must be a bare file-name component")


def _check_format_version(what: str, data: bytes) -> None:
    probe = msgspec.json.decode(data, type=_FormatVersionProbe)
    if probe.format_version != FORMAT_VERSION:
        raise ValueError(
            f"{what} was written with format_version {probe.format_version}; "
            f"this build reads format_version {FORMAT_VERSION}"
        )


class GraphArtifactStore:
    """File layout and atomic writes for one artifact directory.

    ``root`` is ``--cuda-graph-cache-dir``; ``digest`` is the fingerprint
    digest that names the artifact (``fingerprint_digest``), or a placeholder
    such as ``"pending"`` before the post-capture fingerprint exists (design
    section 12, save step 5).
    """

    def __init__(self, root: str | os.PathLike, digest: str) -> None:
        digest = str(digest)
        _check_component("artifact digest", digest)
        self._root = Path(root)
        self._digest = digest

    # -- layout -------------------------------------------------------------

    @property
    def root(self) -> Path:
        return self._root

    @property
    def digest(self) -> str:
        return self._digest

    @property
    def artifact_dir(self) -> Path:
        return self._root / self._digest

    @property
    def images_dir(self) -> Path:
        return self.artifact_dir / IMAGES_DIR

    @property
    def manifest_path(self) -> Path:
        return self.artifact_dir / MANIFEST_FILE

    def rank_dir(self, rank: int) -> Path:
        return self.artifact_dir / f"rank_{int(rank)}"

    def _staging_dir(self, rank: int) -> Path:
        return self.artifact_dir / f"{STAGING_PREFIX}{os.getpid()}-rank_{int(rank)}"

    def _image_path(self, sha256: str) -> Path:
        return self.images_dir / f"{sha256}{IMAGE_SUFFIX}"

    # -- existence ----------------------------------------------------------

    def has_rank(self, rank: int) -> bool:
        """Pure existence check; a rank directory is complete iff it has a
        ``rank.json`` (the staged directory is renamed as a whole)."""
        return (self.rank_dir(rank) / RANK_MANIFEST_FILE).is_file()

    def has_manifest(self) -> bool:
        return self.manifest_path.is_file()

    # -- per-rank bundles ---------------------------------------------------

    def write_rank(
        self,
        rank: int,
        *,
        manifest: RankManifest,
        bundles: Mapping[str, RunnerBundle],
        diagnostics: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        """Write ``rank.json``, one ``<runner>.msgpack`` per bundle and an
        optional ``diagnostics.json``, atomically replacing any earlier
        ``rank_<r>``.

        ``manifest.runners`` is filled from the bundle names when the caller
        left it empty, so ``rank.json`` always indexes its own files; when the
        caller set it, it must name exactly the bundles written.
        """
        runners = tuple(sorted(bundles))
        for name in runners:
            _check_component("runner name", name)
        if manifest.runners:
            if set(manifest.runners) != set(runners):
                raise ValueError(
                    f"RankManifest.runners {sorted(manifest.runners)} does not "
                    f"match the bundles being written {list(runners)}"
                )
        else:
            manifest = msgspec.structs.replace(manifest, runners=runners)

        staging = self._staging_dir(rank)
        _remove_path(staging)
        staging.mkdir(parents=True)
        try:
            _write_bytes(staging / RANK_MANIFEST_FILE, _json_pretty(manifest))
            for name in runners:
                _write_bytes(
                    staging / f"{name}{BUNDLE_SUFFIX}",
                    _MSGPACK.encode(bundles[name]),
                )
            if diagnostics is not None:
                _write_bytes(
                    staging / DIAGNOSTICS_FILE, _json_pretty(dict(diagnostics))
                )
            target = self.rank_dir(rank)
            # os.replace cannot rename onto a non-empty directory; the window
            # between the removal and the rename shows readers *no* rank, never
            # a partial one.
            _remove_path(target)
            os.replace(staging, target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        logger.debug("wrote graph artifact rank %d to %s", rank, target)
        return target

    def read_rank(self, rank: int) -> tuple[RankManifest, dict[str, RunnerBundle]]:
        """Read a complete rank directory; ``ValueError`` on a format-version
        mismatch, ``FileNotFoundError`` when the rank or a listed bundle is
        absent."""
        rank_dir = self.rank_dir(rank)
        manifest_path = rank_dir / RANK_MANIFEST_FILE
        if not manifest_path.is_file():
            raise FileNotFoundError(f"no graph artifact for rank {rank} at {rank_dir}")
        raw = manifest_path.read_bytes()
        _check_format_version(f"rank manifest {manifest_path}", raw)
        manifest = msgspec.json.decode(raw, type=RankManifest)

        bundles: dict[str, RunnerBundle] = {}
        for path in sorted(rank_dir.glob(f"*{BUNDLE_SUFFIX}")):
            name = path.name[: -len(BUNDLE_SUFFIX)]
            bundles[name] = msgspec.msgpack.decode(path.read_bytes(), type=RunnerBundle)
        missing = sorted(set(manifest.runners) - set(bundles))
        if missing:
            raise FileNotFoundError(
                f"rank {rank} manifest lists runners {missing} with no bundle "
                f"file under {rank_dir}"
            )
        return manifest, bundles

    # -- kernel images ------------------------------------------------------

    def put_image(self, sha256: str, data: bytes) -> Path:
        """Store a kernel container under its content hash (atomic). The key
        must be the sha256 of ``data``; a mislabelled image would resolve the
        wrong kernels at load (design section 6.5)."""
        _check_component("image key", sha256)
        actual = hashlib.sha256(data).hexdigest()
        if actual != sha256:
            raise ValueError(
                f"kernel image key {sha256} is not the sha256 of its bytes ({actual})"
            )
        self.images_dir.mkdir(parents=True, exist_ok=True)
        target = self._image_path(sha256)
        tmp = self.images_dir / f"{STAGING_PREFIX}{os.getpid()}-{sha256}{IMAGE_SUFFIX}"
        try:
            _write_bytes(tmp, data)
            os.replace(tmp, target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return target

    def get_image(self, sha256: str) -> bytes:
        """Read a kernel container and verify its content hash; ``ValueError``
        on a mismatch, ``FileNotFoundError`` when absent."""
        _check_component("image key", sha256)
        path = self._image_path(sha256)
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != sha256:
            raise ValueError(
                f"kernel image {path} is corrupt: sha256 {actual} != {sha256}"
            )
        return data

    # -- top-level manifest -------------------------------------------------

    def write_manifest(self, manifest: ArtifactManifest) -> Path:
        """Write ``manifest.json`` atomically. By design (section 11) rank 0
        calls this last, after every rank directory exists and a barrier."""
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.artifact_dir / f"{STAGING_PREFIX}{os.getpid()}-{MANIFEST_FILE}"
        try:
            _write_bytes(tmp, _json_pretty(manifest))
            os.replace(tmp, self.manifest_path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return self.manifest_path

    def read_manifest(self) -> ArtifactManifest:
        path = self.manifest_path
        if not path.is_file():
            raise FileNotFoundError(f"no graph artifact manifest at {path}")
        raw = path.read_bytes()
        _check_format_version(f"artifact manifest {path}", raw)
        return msgspec.json.decode(raw, type=ArtifactManifest)
