# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The DeepSpec Authors
# See LICENSE in this directory for the original license.

"""Inference-only S1/S3 selector used by the SGLang UBTree draft model."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.speculative.ubtree_components.config import parse_ubtree_config

TokenFactorRole = Literal["predecessor", "successor"]


class _TokenFactorMLP(nn.Module):
    def __init__(self, embedding_size: int, hidden_size: int, rank: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(embedding_size, hidden_size, bias=False)
        self.output_projection = nn.Linear(hidden_size, rank, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.output_projection(F.silu(self.input_projection(values)))


class TargetEmbeddingTokenFactorizer(nn.Module):
    def __init__(
        self, embedding_size: int, hidden_size: int, rank: int, vocab_size: int
    ) -> None:
        super().__init__()
        self.embedding_size = int(embedding_size)
        self.rank = int(rank)
        self.vocab_size = int(vocab_size)
        self.predecessor_mlp = _TokenFactorMLP(embedding_size, hidden_size, rank)
        self.successor_mlp = _TokenFactorMLP(embedding_size, hidden_size, rank)
        self.register_buffer("_predecessor_cache", torch.empty(0), persistent=False)
        self.register_buffer("_successor_cache", torch.empty(0), persistent=False)
        self._cache_signature = None

    def _apply(self, fn, recurse=True):
        if self._predecessor_cache.is_meta or self._successor_cache.is_meta:
            self._predecessor_cache = torch.empty(0)
            self._successor_cache = torch.empty(0)
            self._cache_signature = None
        return super()._apply(fn, recurse)

    def factors(self, tokens: torch.Tensor, *, role: TokenFactorRole) -> torch.Tensor:
        cache = (
            self._predecessor_cache if role == "predecessor" else self._successor_cache
        )
        if cache.numel() == 0:
            raise RuntimeError("UBTree selector cache has not been materialized")
        return F.embedding(tokens, cache)

    def _parameter_signature(self):
        return tuple((int(p.data_ptr()), int(p._version)) for p in self.parameters())

    def _store_factor_cache(
        self,
        predecessor: list[torch.Tensor],
        successor: list[torch.Tensor],
        *,
        signature,
    ) -> None:
        self._predecessor_cache = torch.cat(predecessor).contiguous()
        self._successor_cache = torch.cat(successor).contiguous()
        self._cache_signature = signature

    @torch.inference_mode()
    def materialize_cache(
        self, embedding_weight: torch.Tensor, *, chunk_size: int = 4096
    ) -> None:
        expected = (self.vocab_size, self.embedding_size)
        if tuple(embedding_weight.shape) != expected:
            raise ValueError(
                f"UBTree target embedding must have shape {expected}, got "
                f"{tuple(embedding_weight.shape)}"
            )
        signature = (
            "weight",
            int(embedding_weight.data_ptr()),
            int(embedding_weight._version),
            self._parameter_signature(),
        )
        if signature == self._cache_signature:
            return
        predecessor, successor = [], []
        pred_dtype = self.predecessor_mlp.input_projection.weight.dtype
        succ_dtype = self.successor_mlp.input_projection.weight.dtype
        for start in range(0, self.vocab_size, int(chunk_size)):
            rows = embedding_weight[start : start + int(chunk_size)]
            predecessor.append(self.predecessor_mlp(rows.to(dtype=pred_dtype)))
            successor.append(self.successor_mlp(rows.to(dtype=succ_dtype)))
        self._store_factor_cache(predecessor, successor, signature=signature)

    @torch.inference_mode()
    def materialize_cache_from_embedding(
        self,
        embedding_module: nn.Module,
        *,
        device: torch.device,
        chunk_size: int = 4096,
    ) -> None:
        """Materialize factors through a possibly TP-sharded embedding.

        SGLang's ``VocabParallelEmbedding`` returns full embedding rows after
        its TP all-reduce, even though ``embedding_module.weight`` contains
        only the local vocabulary shard.  Calling the module in identical
        chunks on every TP rank therefore builds the same full selector cache
        without gathering the much larger H-dimensional embedding table.
        """

        weight = getattr(embedding_module, "weight", None)
        if weight is None:
            raise TypeError("UBTree target embedding module must expose weight")
        signature = (
            "module",
            id(embedding_module),
            int(weight.data_ptr()),
            int(weight._version),
            self._parameter_signature(),
        )
        if signature == self._cache_signature:
            return
        predecessor, successor = [], []
        pred_dtype = self.predecessor_mlp.input_projection.weight.dtype
        succ_dtype = self.successor_mlp.input_projection.weight.dtype
        for start in range(0, self.vocab_size, int(chunk_size)):
            stop = min(self.vocab_size, start + int(chunk_size))
            tokens = torch.arange(start, stop, dtype=torch.long, device=device)
            rows = embedding_module(tokens)
            expected = (stop - start, self.embedding_size)
            if tuple(rows.shape) != expected:
                raise ValueError(
                    "UBTree target embedding lookup must return shape "
                    f"{expected}, got {tuple(rows.shape)}"
                )
            predecessor.append(self.predecessor_mlp(rows.to(dtype=pred_dtype)))
            successor.append(self.successor_mlp(rows.to(dtype=succ_dtype)))
        self._store_factor_cache(predecessor, successor, signature=signature)

    @torch.inference_mode()
    def materialize_cache_from_vocab_parallel_embedding(
        self,
        embedding_module: nn.Module,
        *,
        tp_group=None,
        chunk_size: int = 4096,
    ) -> None:
        """Factorize local embedding shards, then gather the smaller caches."""

        weight = getattr(embedding_module, "weight", None)
        shard = getattr(embedding_module, "shard_indices", None)
        if weight is None or shard is None:
            raise TypeError(
                "UBTree TP cache construction requires a vocabulary-parallel "
                "embedding with weight and shard_indices"
            )
        if not torch.is_floating_point(weight):
            raise ValueError("UBTree does not support quantized target embeddings")
        if tp_group is None:
            from sglang.srt.distributed import get_tp_group

            tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)
        local_padded_count = int(shard.num_org_elements_padded)
        signature = (
            "vocab_parallel",
            int(weight.data_ptr()),
            int(weight._version),
            tp_size,
            local_padded_count,
            self._parameter_signature(),
        )
        if signature == self._cache_signature:
            return

        predecessor, successor = [], []
        pred_dtype = self.predecessor_mlp.input_projection.weight.dtype
        succ_dtype = self.successor_mlp.input_projection.weight.dtype
        for start in range(0, local_padded_count, int(chunk_size)):
            rows = weight[start : start + int(chunk_size)]
            predecessor.append(self.predecessor_mlp(rows.to(dtype=pred_dtype)))
            successor.append(self.successor_mlp(rows.to(dtype=succ_dtype)))
        local_predecessor = torch.cat(predecessor).contiguous()
        local_successor = torch.cat(successor).contiguous()

        if tp_size == 1:
            full_predecessor = local_predecessor[: self.vocab_size]
            full_successor = local_successor[: self.vocab_size]
        else:
            gathered_shape = (tp_size * local_padded_count, self.rank)
            full_predecessor = torch.empty(
                gathered_shape,
                dtype=local_predecessor.dtype,
                device=local_predecessor.device,
            )
            full_successor = torch.empty(
                gathered_shape,
                dtype=local_successor.dtype,
                device=local_successor.device,
            )
            tp_group.all_gather_into_tensor(full_predecessor, local_predecessor)
            tp_group.all_gather_into_tensor(full_successor, local_successor)
            full_predecessor = full_predecessor[: self.vocab_size]
            full_successor = full_successor[: self.vocab_size]
        self._store_factor_cache(
            [full_predecessor],
            [full_successor],
            signature=signature,
        )


class DepthScoreConditioner(nn.Module):
    def __init__(self, depth_count: int) -> None:
        super().__init__()
        self.unary_log_scale = nn.Parameter(torch.zeros(depth_count))
        self.correction_log_scale = nn.Parameter(torch.zeros(depth_count))

    def scales(self, *, dtype: torch.dtype, device: torch.device):
        return (
            self.unary_log_scale.exp().to(dtype=dtype, device=device),
            self.correction_log_scale.exp().to(dtype=dtype, device=device),
        )


class UBTreeSelector(nn.Module):
    """S1/S3 selector for target-embedding MLP and DFlash2 lookup checkpoints."""

    def __init__(self, config) -> None:
        super().__init__()
        cfg = parse_ubtree_config(config)
        vocab_size = self.vocab_size = int(config.vocab_size)
        hidden_size = int(config.hidden_size)
        self.rank = cfg.rank
        self.candidate_k = cfg.candidate_k
        self.token_factorizer = cfg.token_factorizer
        self.hidden_projection = nn.Linear(hidden_size, cfg.rank, bias=False)
        if self.token_factorizer == "lookup":
            self.predecessor_codebook = nn.Embedding(vocab_size, cfg.rank)
            self.successor_codebook = nn.Embedding(vocab_size, cfg.rank)
            self.tied_token_factorizer = None
        else:
            self.predecessor_codebook = self.successor_codebook = None
            self.tied_token_factorizer = TargetEmbeddingTokenFactorizer(
                cfg.target_embedding_dim,
                cfg.factorizer_hidden_width,
                cfg.rank,
                vocab_size,
            )
        self.depth_conditioner = DepthScoreConditioner(cfg.depth_count)

    def materialize_cache(self, target_embedding_weight: torch.Tensor) -> None:
        if self.tied_token_factorizer is not None:
            self.tied_token_factorizer.materialize_cache(target_embedding_weight)

    def bind_target_embedding(self, embedding_module: nn.Module, *, device) -> None:
        """Build the serving cache and declare its online-update constraint."""
        if self.tied_token_factorizer is None:
            return
        weight = getattr(embedding_module, "weight", None)
        if weight is None:
            raise RuntimeError("UBTREE requires target input embeddings with weights")
        from sglang.srt.speculative.dflash_utils import is_dense_head_weight

        if not is_dense_head_weight(weight):
            raise ValueError("UBTREE requires dense FP16/BF16/FP32 target embeddings")
        if (
            int(getattr(embedding_module, "tp_size", 1)) == 1
            and int(weight.shape[0]) >= self.vocab_size
        ):
            self.materialize_cache(weight[: self.vocab_size])
        elif hasattr(embedding_module, "shard_indices"):
            self.materialize_cache_from_vocab_parallel_embedding(embedding_module)
        else:
            self.materialize_cache_from_embedding(embedding_module, device=device)

        # Both target embeddings and draft factorizer weights feed these caches.
        # The updater checks nested modules before modifying either model.
        reason = (
            "UBTREE target-embedding-MLP selector caches do not support online "
            "weight updates. Restart the server with the new target/draft weights."
        )
        self._derived_weight_cache_error = reason
        embedding_module._derived_weight_cache_error = reason

    def materialize_cache_from_embedding(
        self,
        embedding_module: nn.Module,
        *,
        device: torch.device,
        chunk_size: int = 4096,
    ) -> None:
        if self.tied_token_factorizer is not None:
            self.tied_token_factorizer.materialize_cache_from_embedding(
                embedding_module,
                device=device,
                chunk_size=chunk_size,
            )

    def materialize_cache_from_vocab_parallel_embedding(
        self,
        embedding_module: nn.Module,
        *,
        tp_group=None,
        chunk_size: int = 4096,
    ) -> None:
        if self.tied_token_factorizer is not None:
            self.tied_token_factorizer.materialize_cache_from_vocab_parallel_embedding(
                embedding_module,
                tp_group=tp_group,
                chunk_size=chunk_size,
            )

    def _factors(self, tokens: torch.Tensor, *, role: TokenFactorRole) -> torch.Tensor:
        if self.tied_token_factorizer is not None:
            return self.tied_token_factorizer.factors(tokens, role=role)
        table = (
            self.predecessor_codebook
            if role == "predecessor"
            else self.successor_codebook
        )
        assert table is not None
        return table(tokens)

    def build_edge_log_probs(
        self,
        hidden: torch.Tensor,
        anchor_tokens: torch.Tensor,
        candidate_ids: torch.Tensor,
        *,
        unary_candidate_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``[batch, depth, K, K]`` normalized transition scores."""

        if hidden.ndim != 3 or candidate_ids.ndim != 3:
            raise ValueError("UBTree hidden/candidate tensors must be rank 3")
        batch, depth, candidate_k = candidate_ids.shape
        if hidden.shape[:2] != (batch, depth) or candidate_k != self.candidate_k:
            raise ValueError("UBTree selector shape does not match its config")
        predecessors = torch.cat(
            (
                anchor_tokens.reshape(batch, 1, 1).expand(batch, 1, candidate_k),
                candidate_ids[:, :-1],
            ),
            dim=1,
        )
        predecessor_factors = self._factors(predecessors, role="predecessor")
        successor_factors = self._factors(candidate_ids, role="successor")
        projected = self.hidden_projection(hidden)
        correction = torch.einsum(
            "btpr,btkr->btpk",
            predecessor_factors * projected[:, :, None],
            successor_factors,
        )
        alpha, lam = self.depth_conditioner.scales(
            dtype=correction.dtype, device=correction.device
        )
        logits = (
            alpha[:depth][None, :, None, None]
            * unary_candidate_logits[:, :, None, :].to(correction)
            + lam[:depth][None, :, None, None] * correction
        )
        return torch.log_softmax(logits.float(), dim=-1)


__all__ = ["UBTreeSelector"]
