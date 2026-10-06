# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

import abc
import importlib
import logging
from typing import Any, Dict, List

import torch

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
