# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Abstract interface for Engram embedding storage.

This module defines the storage-only contract that all Engram backends must
implement.  It is deliberately decoupled from the model-computation side
(hash mapping, gating, projections) so that different storage technologies
(e.g. local CPU DRAM) can be plugged in without touching model code.

Key concepts
------------
* **vocab_table** – A 2-D ``(total_vocab_size, embedding_dim)`` tensor that
  holds the full embedding table for one Engram layer.
* **layer_id** – Each Engram-enabled transformer layer owns an independent
  embedding table identified by ``layer_id``.
* **put_sharded / get_many** – The two core operations: bulk-load an
  embedding table and batch-lookup embeddings by flat indices.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class EngramStoreConfig:
    """Backend-agnostic configuration carried through HiCache lifecycle."""

    embedding_dim: int = 64
    vocab_sizes: List[int] = field(default_factory=list)
    layer_ids: List[int] = field(default_factory=list)
    dtype: torch.dtype = torch.float16
    store_backend: str = "local"
    extra: Optional[dict] = None


class EngramStore(ABC):
    """Abstract base class for Engram embedding storage backends.

    Every concrete backend must implement at least ``put_sharded``,
    ``get_one``, ``get_many``, and ``close``.
    """

    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        layer_id: int,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.embedding_dim = int(embedding_dim)
        self.vocab_size = int(vocab_size)
        self.layer_id = int(layer_id)
        self.dtype = dtype

    @abstractmethod
    def put_sharded(self, vocab_table: torch.Tensor) -> None:
        """Bulk-load the full embedding table for this layer.

        Parameters
        ----------
        vocab_table : torch.Tensor
            Shape ``(vocab_size, embedding_dim)``, dtype may differ from
            ``self.dtype`` – implementations should cast internally.
        """

    @abstractmethod
    def get_one(self, index: int, layer_id: int, device: torch.device) -> torch.Tensor:
        """Retrieve a single embedding vector.

        Returns a 1-D tensor of shape ``(embedding_dim,)`` on *device*.
        Out-of-range indices should return a zero vector.
        """

    @abstractmethod
    def get_many(
        self, indices: torch.Tensor, layer_id: int, device: torch.device
    ) -> torch.Tensor:
        """Batch-retrieve embeddings.

        Parameters
        ----------
        indices : torch.Tensor
            Arbitrary-shaped integer tensor of flat vocabulary indices.
        layer_id : int
            The transformer layer requesting the lookup.
        device : torch.device
            Target device for the returned tensor.

        Returns
        -------
        torch.Tensor
            Shape ``(*indices.shape, embedding_dim)`` on *device*.
        """

    @abstractmethod
    def close(self) -> None:
        """Release resources held by this store."""
