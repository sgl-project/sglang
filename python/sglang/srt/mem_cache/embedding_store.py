# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

import abc
import ctypes
import importlib
import logging
import os
from typing import Any, Dict, List

import torch

from sglang.srt import environ as envs

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
        logger.info(f"File Embedding Store initialized at {self.storage_dir}")

    def _path_for_key(self, key: str) -> str:
        return os.path.join(self.storage_dir, f"{key}.bin")

    def batch_get(
        self, hashes: List[str], ptrs: List[int], sizes: List[int]
    ) -> List[bool]:
        results = []
        for h, ptr, size in zip(hashes, ptrs, sizes):
            path = self._path_for_key(self.get_key(h))
            try:
                with open(path, "rb") as f:
                    data = f.read(size)
                if len(data) != size:
                    results.append(False)
                    continue
                ctypes.memmove(ptr, data, size)
                results.append(True)
            except FileNotFoundError:
                results.append(False)
        return results

    def batch_put(
        self, hashes: List[str], ptrs: List[int], sizes: List[int]
    ) -> List[bool]:
        results = []
        for h, ptr, size in zip(hashes, ptrs, sizes):
            path = self._path_for_key(self.get_key(h))
            if os.path.exists(path):
                results.append(True)
                continue
            try:
                buf = (ctypes.c_char * size).from_address(ptr)
                tmp_path = path + ".tmp"
                with open(tmp_path, "wb") as f:
                    f.write(buf)
                os.replace(tmp_path, path)
                results.append(True)
            except Exception:
                logger.exception(f"Failed to write embedding for hash {h}")
                results.append(False)
        return results

    def batch_get_into_multi_buffers(
        self,
        hashes: List[str],
        ptrs: List[List[int]],
        sizes: List[List[int]],
    ) -> List[bool]:
        results = []
        for h, ptr_list, size_list in zip(hashes, ptrs, sizes):
            path = self._path_for_key(self.get_key(h))
            try:
                with open(path, "rb") as f:
                    data = f.read()
                offset = 0
                success = True
                for ptr, size in zip(ptr_list, size_list):
                    chunk = data[offset : offset + size]
                    if len(chunk) != size:
                        success = False
                        break
                    ctypes.memmove(ptr, chunk, size)
                    offset += size
                results.append(success)
            except FileNotFoundError:
                results.append(False)
        return results

    def batch_put_from_multi_buffers(
        self,
        hashes: List[str],
        ptrs: List[List[int]],
        sizes: List[List[int]],
    ) -> List[bool]:
        results = []
        for h, ptr_list, size_list in zip(hashes, ptrs, sizes):
            path = self._path_for_key(self.get_key(h))
            if os.path.exists(path):
                results.append(True)
                continue
            try:
                tmp_path = path + ".tmp"
                with open(tmp_path, "wb") as f:
                    for ptr, size in zip(ptr_list, size_list):
                        buf = (ctypes.c_char * size).from_address(ptr)
                        f.write(buf)
                os.replace(tmp_path, path)
                results.append(True)
            except Exception:
                logger.exception(f"Failed to write embedding for hash {h}")
                results.append(False)
        return results

    def batch_is_exist(self, hashes: List[str]) -> List[bool]:
        return [
            os.path.exists(self._path_for_key(self.get_key(h))) for h in hashes
        ]


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
