"""File-backed storage for multimodal preprocess cache values shared across processes."""

from __future__ import annotations

import errno
import os
import uuid
from pathlib import Path
from typing import Any, Optional

import torch

from sglang.srt.multimodal.cache.identity import CONTENT_HASH_PREFIX


class DiskPreprocessCache:
    """A map from cache keys to values stored as files under one directory.

    Every process that sees the directory shares it, e.g. an RL job's rollout
    engines and trainers. A value is written to a temporary file and renamed into
    place, so a reader sees all of it or none. Nothing is deleted: the directory
    lives as long as the job that created it. Values are unpickled on read, since
    processor outputs hold non-tensor objects such as video metadata, so only
    trusted processes may write the directory.
    """

    def __init__(self, directory: str):
        self.directory = Path(directory)

    def get(self, key: str) -> Optional[Any]:
        try:
            return torch.load(self._path(key), map_location="cpu", weights_only=False)
        except OSError as error:
            # ESTALE: on NFS, the file was removed while this read was open.
            if error.errno in (errno.ENOENT, errno.ESTALE):
                return None
            raise

    def __setitem__(self, key: str, value: Any) -> None:
        path = self._path(key)
        # A key fixes its value, so a file another request already wrote is not rewritten.
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        # Not tempfile, which creates files 0600: other users must read them.
        temp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        with open(temp_path, "xb") as file:
            torch.save(value, file)
        os.replace(temp_path, path)

    def _path(self, key: str) -> Path:
        digest = key.removeprefix(CONTENT_HASH_PREFIX)
        # Two-character subdirectories keep any one directory small.
        return self.directory / digest[:2] / f"{digest}.pt"
