# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

import abc
import argparse
import ctypes
import importlib
import logging
import os
import threading
import uuid
from typing import Any, Dict, List

import torch

from sglang.srt.environ import envs
from sglang.srt.utils.common import human_readable_int

logger = logging.getLogger(__name__)


class EmbeddingStore(abc.ABC):
    """Abstract base class for multimodal embedding storage backends.

    Stores pre-computed vision/audio embeddings by content hash so they
    can be shared across nodes without re-running the encoder.
    """

    @abc.abstractmethod
    def batch_get(
        self, hashes: List[str], ptrs: List[int], sizes: List[int]
    ) -> List[bool]:
        raise NotImplementedError

    @abc.abstractmethod
    def batch_put(
        self, hashes: List[str], ptrs: List[int], sizes: List[int]
    ) -> List[bool]:
        raise NotImplementedError

    @abc.abstractmethod
    def batch_get_into_multi_buffers(
        self,
        hashes: List[str],
        ptrs: List[List[int]],
        sizes: List[List[int]],
    ) -> List[bool]:
        raise NotImplementedError

    @abc.abstractmethod
    def batch_put_from_multi_buffers(
        self,
        hashes: List[str],
        ptrs: List[List[int]],
        sizes: List[List[int]],
    ) -> List[bool]:
        raise NotImplementedError

    @abc.abstractmethod
    def batch_is_exist(self, hashes: List[str]) -> List[bool]:
        raise NotImplementedError

    def register_buffer(self, tensor: torch.Tensor) -> None:
        pass

    def get_key(self, mm_hash: str) -> str:
        return f"emb_{mm_hash}"


class FileEmbeddingStore(EmbeddingStore):
    """File-based embedding store for local or shared-filesystem caching."""

    def __init__(self, storage_dir: str = None):
        self.storage_dir = storage_dir or envs.SGLANG_MM_EMBEDDING_CACHE_DIR.get()
        os.makedirs(self.storage_dir, exist_ok=True)
        self._size_lock = threading.Lock()
        if envs.SGLANG_MM_EMBEDDING_CACHE_CLEAR_ON_START.get():
            self._clear_cache_files()
        self.max_size_bytes = self._parse_size_to_bytes(
            envs.SGLANG_MM_EMBEDDING_CACHE_MAX_SIZE.get(),
            "SGLANG_MM_EMBEDDING_CACHE_MAX_SIZE",
        )
        self._used_bytes = 0
        logger.info(f"File Embedding Store initialized at {self.storage_dir}")

    def _path_for_key(self, key: str) -> str:
        return os.path.join(self.storage_dir, f"{key}.bin")

    @staticmethod
    def _parse_size_to_bytes(value: str, name: str) -> int:
        if value is None:
            return 0
        value = str(value).strip()
        if not value or value == "0":
            return 0
        try:
            return max(0, human_readable_int(value))
        except (argparse.ArgumentTypeError, ValueError) as e:
            logger.warning(f"Invalid {name}={value!r}; disabling limit: {e}")
            return 0

    def _clear_cache_files(self) -> None:
        removed = 0
        with os.scandir(self.storage_dir) as entries:
            for entry in entries:
                is_cache_file = entry.name.startswith("emb_") and (
                    entry.name.endswith(".bin") or ".bin.tmp" in entry.name
                )
                if not is_cache_file or not entry.is_file():
                    continue
                try:
                    os.remove(entry.path)
                    removed += 1
                except OSError:
                    logger.exception(
                        f"Failed to remove stale embedding cache file {entry.path}"
                    )
        if removed:
            logger.info(
                f"Cleared {removed} embedding cache files from {self.storage_dir}"
            )

    def _put_buffers(self, h: str, ptrs: List[int], sizes: List[int]) -> bool:
        path = self._path_for_key(self.get_key(h))
        if os.path.isfile(path):
            return True
        if os.path.exists(path):
            logger.warning(
                f"Embedding cache path exists but is not a regular file: {path}"
            )
            return False

        total_size = sum(sizes)
        reserved = False
        if self.max_size_bytes > 0:
            with self._size_lock:
                if self._used_bytes + total_size > self.max_size_bytes:
                    logger.warning(
                        f"Embedding cache max size reached "
                        f"({self._used_bytes}+{total_size}>{self.max_size_bytes}); "
                        f"skipping file cache write for {h}."
                    )
                    return False
                self._used_bytes += total_size
                reserved = True

        tmp_path = f"{path}.tmp.{uuid.uuid4().hex}"
        try:
            with open(tmp_path, "wb") as f:
                for ptr, size in zip(ptrs, sizes):
                    buf = (ctypes.c_char * size).from_address(ptr)
                    f.write(buf)
            os.replace(tmp_path, path)
            return True
        except Exception:
            if reserved:
                with self._size_lock:
                    self._used_bytes -= total_size
            logger.exception(f"Failed to write embedding for hash {h}")
            return False
        finally:
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass
            except OSError:
                logger.exception(f"Failed to remove temp embedding file {tmp_path}")

    def _get_buffers(self, h: str, ptrs: List[int], sizes: List[int]) -> bool:
        path = self._path_for_key(self.get_key(h))
        try:
            expected_size = sum(sizes)
            if not os.path.isfile(path) or os.path.getsize(path) != expected_size:
                return False
            with open(path, "rb", buffering=0) as f:
                for ptr, size in zip(ptrs, sizes):
                    buf = (ctypes.c_char * size).from_address(ptr)
                    if f.readinto(buf) != size:
                        return False
            return True
        except FileNotFoundError:
            return False
        except OSError:
            logger.exception(f"Failed to read embedding for hash {h}")
            return False

    def batch_get(
        self, hashes: List[str], ptrs: List[int], sizes: List[int]
    ) -> List[bool]:
        return [
            self._get_buffers(h, [ptr], [size])
            for h, ptr, size in zip(hashes, ptrs, sizes)
        ]

    def batch_put(
        self, hashes: List[str], ptrs: List[int], sizes: List[int]
    ) -> List[bool]:
        return [
            self._put_buffers(h, [ptr], [size])
            for h, ptr, size in zip(hashes, ptrs, sizes)
        ]

    def batch_get_into_multi_buffers(
        self,
        hashes: List[str],
        ptrs: List[List[int]],
        sizes: List[List[int]],
    ) -> List[bool]:
        return [
            self._get_buffers(h, ptr_list, size_list)
            for h, ptr_list, size_list in zip(hashes, ptrs, sizes)
        ]

    def batch_put_from_multi_buffers(
        self,
        hashes: List[str],
        ptrs: List[List[int]],
        sizes: List[List[int]],
    ) -> List[bool]:
        return [
            self._put_buffers(h, ptr_list, size_list)
            for h, ptr_list, size_list in zip(hashes, ptrs, sizes)
        ]

    def batch_is_exist(self, hashes: List[str]) -> List[bool]:
        return [os.path.isfile(self._path_for_key(self.get_key(h))) for h in hashes]


class EmbeddingStoreFactory:
    """Factory for creating embedding store backend instances."""

    _registry: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _load_backend_class(
        module_path: str, class_name: str, backend_name: str
    ) -> type:
        try:
            module = importlib.import_module(module_path)
            backend_class = getattr(module, class_name)
            if not issubclass(backend_class, EmbeddingStore):
                raise TypeError(
                    f"Backend class {class_name} must inherit from EmbeddingStore"
                )
            return backend_class
        except ImportError as e:
            raise ImportError(
                f"Failed to import embedding store backend '{backend_name}' "
                f"from '{module_path}': {e}"
            ) from e
        except AttributeError as e:
            raise AttributeError(
                f"Class '{class_name}' not found in module '{module_path}': {e}"
            ) from e

    @classmethod
    def register_backend(cls, name: str, module_path: str, class_name: str) -> None:
        if name in cls._registry:
            logger.warning(
                f"Embedding store backend '{name}' is already registered, overwriting"
            )

        def loader() -> type:
            return cls._load_backend_class(module_path, class_name, name)

        cls._registry[name] = {
            "loader": loader,
            "module_path": module_path,
            "class_name": class_name,
        }

    @classmethod
    def create_backend(cls, backend_name: str, **kwargs) -> EmbeddingStore:
        if backend_name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown embedding store backend '{backend_name}'. "
                f"Registered backends: {available}."
            )

        entry = cls._registry[backend_name]
        backend_class = entry["loader"]()
        logger.info(
            f"Creating embedding store backend '{backend_name}' "
            f"({entry['module_path']}.{entry['class_name']})"
        )
        return backend_class(**kwargs)


EmbeddingStoreFactory.register_backend(
    "mooncake",
    "sglang.srt.mem_cache.storage.mooncake_store.mooncake_embedding_store",
    "MooncakeEmbeddingStore",
)

EmbeddingStoreFactory.register_backend(
    "file",
    "sglang.srt.mem_cache.embedding_store",
    "FileEmbeddingStore",
)
