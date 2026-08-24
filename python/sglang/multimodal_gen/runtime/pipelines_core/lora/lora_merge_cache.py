"""File-backed store for LoRA-merged weights.

Merging an adapter writes the base weight in place. Under layerwise offload
the base weight is a view into the checkpoint mapping, so the write is a
copy-on-write: every merged byte turns into anonymous host memory the kernel
cannot reclaim. MiniMax-H3's DiT alone is 61.7 GB — a real 32 GB host dies on
it, and on any host the pin budget collapses to zero before the offload
managers ever see the weights.

Written once to a per-layer cache file and mapped back, the same merged bytes
become page cache: droppable, refaultable, and invisible to the anonymous
accounting. The offload managers then classify them as mapped weights on
their own — no coordination needed. Rehoming happens layer by layer inside
the merge loop, so the anonymous high-water mark stays one layer wide, and a
later start with the same (base, adapters, strengths) adopts the store
without paying the merge at all.
"""

import hashlib
import json
import os
import shutil

import torch
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_MANIFEST = "manifest.json"
_DISK_HEADROOM = 1.15


def lora_merge_cache_key(
    base_paths: list[str],
    adapters: list[tuple[str, float, float | None]],
) -> str:
    """Key of one merged-weights combination.

    `base_paths` are the component checkpoint paths (HF snapshot paths carry
    the revision hash); `adapters` are ordered (lora_path, strength, alpha)
    triples — order matters, merges compose in order.
    """
    parts = [os.path.realpath(p) for p in sorted(base_paths)]
    for path, strength, alpha in adapters:
        real = os.path.realpath(path)
        try:
            size = os.path.getsize(real)
        except OSError:
            size = -1
        parts.append(f"{real}|{size}|{strength}|{alpha}")
    return hashlib.sha1("||".join(parts).encode()).hexdigest()[:16]


class LoraMergeCache:
    """Streams merged weights into a cache directory, one file per layer."""

    def __init__(self, key: str, expected_bytes: int) -> None:
        self.root = os.path.join(
            envs.SGLANG_DIFFUSION_CACHE_ROOT, "lora_merge_cache", key
        )
        self.manifest_path = os.path.join(self.root, _MANIFEST)
        self.expected_bytes = expected_bytes
        self._entries: dict[str, dict] = {}
        self._writable: bool | None = None

    # -- adoption (fast path) -------------------------------------------------

    def is_complete(self) -> bool:
        """A complete store from an earlier run of the same combination."""
        try:
            with open(self.manifest_path) as handle:
                manifest = json.load(handle)
        except (OSError, ValueError):
            return False
        entries = manifest.get("layers")
        if not isinstance(entries, dict) or not entries:
            return False
        for meta in entries.values():
            if not os.path.exists(os.path.join(self.root, meta.get("file", ""))):
                return False
        self._entries = entries
        return True

    def get(
        self, name: str, shape: torch.Size, dtype: torch.dtype
    ) -> torch.Tensor | None:
        """The cached merged tensor for `name`, mapped from its file.

        Purely a lookup: the caller decides what to do with the tensor. A
        missing or mismatched entry returns None — mismatch also drops the
        remaining entries, because one wrong file means the whole combination
        key no longer describes this module.
        """
        meta = self._entries.get(name)
        if meta is None:
            return None
        mapped = safetensors_load_file(os.path.join(self.root, meta["file"]))
        tensor = mapped.get("weight")
        if (
            tensor is None
            or tuple(tensor.shape) != tuple(shape)
            or tensor.dtype != dtype
        ):
            logger.warning(
                "LoRA merge cache entry for %s does not match the module; "
                "ignoring the cache",
                name,
            )
            self._entries = {}
            return None
        return tensor

    # -- capture (first run) --------------------------------------------------

    def _ensure_writable(self) -> bool:
        if self._writable is not None:
            return self._writable
        try:
            os.makedirs(self.root, exist_ok=True)
            usage = shutil.disk_usage(self.root)
            if usage.free < self.expected_bytes * _DISK_HEADROOM:
                logger.warning(
                    "LoRA merge cache needs %.1f GiB free under %s but only "
                    "%.1f GiB is available; merged weights stay in anonymous "
                    "host memory",
                    self.expected_bytes * _DISK_HEADROOM / 1024**3,
                    self.root,
                    usage.free / 1024**3,
                )
                self._writable = False
            else:
                self._writable = True
        except OSError as exc:
            logger.warning("LoRA merge cache unavailable (%s)", exc)
            self._writable = False
        return self._writable

    def put(self, name: str, merged: torch.Tensor) -> torch.Tensor | None:
        """Write one merged tensor to its cache file and return the mapping.

        The returned tensor is a view into the file — page cache the kernel
        can drop — and the only thing the cache hands back; what to install it
        into is the caller's business. None means the bytes could not be
        cached (disk shortage, write failure) and the caller should keep its
        own copy.
        """
        if not self._ensure_writable():
            return None
        fname = hashlib.sha1(name.encode()).hexdigest()[:16] + ".safetensors"
        path = os.path.join(self.root, fname)
        try:
            tmp = f"{path}.tmp.{os.getpid()}"
            safetensors_save_file({"weight": merged.contiguous()}, tmp)
            os.replace(tmp, path)
            mapped = safetensors_load_file(path)["weight"]
        except Exception as exc:
            logger.warning(
                "Could not cache merged weight %s (%s); it stays in "
                "anonymous host memory",
                name,
                exc,
            )
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
            return None
        self._entries[name] = {
            "file": fname,
            "shape": list(merged.shape),
            "dtype": str(merged.dtype),
        }
        return mapped

    def finalize(self, extra: dict | None = None) -> None:
        """Write the manifest; only a complete store is ever adopted."""
        if not self._entries or not self._ensure_writable():
            return
        manifest = {"layers": self._entries}
        if extra:
            manifest.update(extra)
        tmp = f"{self.manifest_path}.tmp.{os.getpid()}"
        try:
            with open(tmp, "w") as handle:
                json.dump(manifest, handle)
            os.replace(tmp, self.manifest_path)
        except OSError as exc:
            logger.warning("LoRA merge cache manifest not written (%s)", exc)
            return
        logger.info(
            "Merged weights cached to %s (%d layers); anonymous host memory "
            "no longer holds them",
            self.root,
            len(self._entries),
        )
