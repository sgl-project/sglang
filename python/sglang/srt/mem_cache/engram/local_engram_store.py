# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Local (CPU/GPU memory) Engram embedding store.

This is the simplest backend: it keeps the embedding table in a standard
``nn.Embedding`` on a configurable device (defaults to CPU).  Suitable for
single-node development, testing, and small-scale deployments.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from sglang.srt.mem_cache.engram.engram_store import EngramStore


class LocalEngramStore(EngramStore):

    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        layer_id: int,
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(embedding_dim, vocab_size, layer_id, dtype)
        self.storage_dtype = torch.float16
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            dtype=self.storage_dtype,
        )
        self._device = torch.device("cpu")
        if device is not None:
            self._ensure_device(device)

    def _ensure_device(self, device: torch.device) -> None:
        target = torch.device(device)
        if target != self._device:
            self.embedding = self.embedding.to(device=target)
            self._device = target

    def put_sharded(self, vocab_table: torch.Tensor) -> None:
        if not isinstance(vocab_table, torch.Tensor):
            raise TypeError("vocab_table must be a torch.Tensor")
        if vocab_table.ndim != 2:
            raise ValueError("vocab_table must be 2D")
        if (
            vocab_table.shape[0] != self.vocab_size
            or vocab_table.shape[1] != self.embedding_dim
        ):
            raise ValueError(
                f"vocab_table shape {tuple(vocab_table.shape)} must match "
                f"({self.vocab_size}, {self.embedding_dim})"
            )

        data = vocab_table.detach()
        if data.dtype != self.storage_dtype:
            data = data.to(dtype=self.storage_dtype)
        if data.device != self._device:
            data = data.to(self._device)

        with torch.no_grad():
            self.embedding.weight.copy_(data)

    def get_one(self, index: int, layer_id: int, device: torch.device) -> torch.Tensor:
        if index < 0 or index >= self.vocab_size:
            return torch.zeros((self.embedding_dim,), dtype=self.dtype, device=device)

        target_device = device or self._device
        self._ensure_device(target_device)

        idx = torch.tensor([int(index)], dtype=torch.long, device=self._device)
        out = self.embedding(idx).squeeze(0)
        if self.dtype != self.storage_dtype:
            out = out.to(dtype=self.dtype)
        if device is not None and out.device != device:
            out = out.to(device=device)
        return out

    def get_many(
        self, indices: torch.Tensor, layer_id: int, device: torch.device
    ) -> torch.Tensor:
        target_device = device or indices.device
        self._ensure_device(target_device)

        if indices.numel() == 0:
            return torch.empty(
                (*indices.shape, self.embedding_dim),
                device=target_device,
                dtype=self.dtype,
            )

        idx = indices.to(device=self._device, dtype=torch.long)
        out = self.embedding(idx)
        if self.dtype != self.storage_dtype:
            out = out.to(dtype=self.dtype)
        if device is not None and out.device != device:
            out = out.to(device=device)
        return out.view(*indices.shape, self.embedding_dim)

    def close(self) -> None:
        pass
