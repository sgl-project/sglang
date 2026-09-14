"""File-backed homes for the host copies a checkpoint mapping cannot provide.

A weight that is fused (q/k/v into one projection), sharded or otherwise
transformed at load has no checkpoint bytes to stay mapped on, so the loader
materializes it. Anonymous memory is the wrong home for that copy on a host
that keeps everything else mapped: it is never reclaimable, and on a shared
CPU/GPU pool it is memory the page cache and the device both lose. A shared
file mapping under the cache directory holds the same bytes as page cache
instead -- reclaimable, readable with O_DIRECT, and, once written, reusable
by the next start of the same checkpoint.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Callable, Iterable

import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Below this a copy is not worth a file: the bookkeeping costs more than the
# bytes it would return to the pool.
MIN_SPILL_BYTES = 64 << 20
# Free space to leave on the spill filesystem after a write.
SPILL_DISK_RESERVE_BYTES = 2 << 30

FusedTensorFactory = Callable[
    [str, torch.Size, torch.dtype], "tuple[torch.Tensor, bool] | None"
]


def checkpoint_fingerprint(weight_dirs: Iterable[str]) -> str:
    """Identity of a checkpoint on disk: each shard's path, size and mtime."""
    digest = hashlib.sha1()
    for weight_dir in sorted(str(d) for d in weight_dirs):
        root = Path(weight_dir)
        files = [root] if root.is_file() else sorted(root.glob("*.safetensors"))
        for path in files:
            try:
                stat = path.stat()
            except OSError:
                continue
            digest.update(
                f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}\n".encode()
            )
    return digest.hexdigest()[:20]


class HostSpill:
    """Hands out file-backed tensors keyed by (parameter, dtype, shape).

    A tensor comes back ``(tensor, filled)``: ``filled`` says an earlier run
    wrote and sealed the same key, so the caller can skip producing it. A
    caller that produces the bytes must ``seal`` the key afterwards; an
    unsealed file is treated as garbage and rewritten.
    """

    def __init__(self, directory: str | os.PathLike[str], fingerprint: str):
        self.directory = Path(directory) / fingerprint
        self._disabled_reason: str | None = None
        self.bytes_written = 0
        self.bytes_reused = 0
        self.count_written = 0
        self.count_reused = 0
        self._open: dict[str, str] = {}

    @classmethod
    def for_checkpoint(cls, weight_dirs: Iterable[str]) -> HostSpill | None:
        if envs.SGLANG_DIFFUSION_DISABLE_HOST_SPILL:
            return None
        directory = os.path.expanduser(envs.SGLANG_DIFFUSION_HOST_SPILL_DIR)
        return cls(directory, checkpoint_fingerprint(weight_dirs))

    def _path(self, key: str) -> Path:
        return self.directory / (hashlib.sha1(key.encode()).hexdigest() + ".bin")

    def _disable(self, reason: str) -> None:
        if self._disabled_reason is None:
            self._disabled_reason = reason
            logger.warning(
                "Host spill disabled for this load: %s; transformed weights "
                "fall back to anonymous memory.",
                reason,
            )

    def tensor(
        self, name: str, shape: torch.Size, dtype: torch.dtype
    ) -> tuple[torch.Tensor, bool] | None:
        """A file-backed tensor for ``name``, or None to use anonymous memory."""
        if self._disabled_reason is not None:
            return None
        numel = 1
        for dim in shape:
            numel *= int(dim)
        nbytes = numel * torch.empty((), dtype=dtype).element_size()
        if nbytes < MIN_SPILL_BYTES:
            return None
        key = f"{name}|{dtype}|{tuple(int(d) for d in shape)}"
        path = self._path(key)
        sealed = path.with_suffix(".ok")
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            filled = sealed.exists() and path.exists() and path.stat().st_size == nbytes
            if not filled:
                sealed.unlink(missing_ok=True)
                free = shutil.disk_usage(self.directory).free
                if free < nbytes + SPILL_DISK_RESERVE_BYTES:
                    self._disable(
                        f"{free / 2**30:.1f} GiB free under {self.directory}, "
                        f"{nbytes / 2**30:.1f} GiB needed"
                    )
                    return None
            storage = torch.from_file(str(path), shared=True, size=numel, dtype=dtype)
        except (OSError, RuntimeError) as exc:
            self._disable(f"{type(exc).__name__}: {exc}")
            return None
        tensor = storage.view(tuple(int(d) for d in shape))
        if filled:
            self.bytes_reused += nbytes
            self.count_reused += 1
        else:
            self._open[key] = str(sealed)
            self.bytes_written += nbytes
            self.count_written += 1
        return tensor, filled

    def seal(self, name: str, shape: torch.Size, dtype: torch.dtype) -> None:
        """Mark a key as completely written so the next start can reuse it."""
        key = f"{name}|{dtype}|{tuple(int(d) for d in shape)}"
        sealed = self._open.pop(key, None)
        if sealed is None:
            return
        try:
            with open(sealed, "w") as handle:
                handle.write("ok\n")
        except OSError as exc:
            logger.debug("could not seal %s: %s", sealed, exc)

    def log_summary(self, component: str) -> None:
        if self.count_written == 0 and self.count_reused == 0:
            return
        logger.info(
            "%s: %d transformed weights (%.2f GiB) live in file mappings under %s "
            "(%d written, %d reused from an earlier start).",
            component,
            self.count_written + self.count_reused,
            (self.bytes_written + self.bytes_reused) / 2**30,
            self.directory,
            self.count_written,
            self.count_reused,
        )
