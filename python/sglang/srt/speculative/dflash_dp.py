from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator
from torch import nn


def embedding_tp_group(embedding: VocabParallelEmbedding) -> GroupCoordinator:
    return get_attention_tp_group() if embedding.use_attn_tp_group else get_tp_group()


@torch.no_grad()
def gather_draft_embedding(
    embedding: nn.Module, draft_tp_group: GroupCoordinator
) -> torch.Tensor | None:
    """Cache only tables whose collectives would cross independent drafts.

    Run in target scope, after installing any trained mask row. MiMo's existing
    attention-local embedding and replicated tables keep their ordinary lookup.
    """
    if not isinstance(embedding, VocabParallelEmbedding) or embedding.tp_size == 1:
        return None
    group = embedding_tp_group(embedding)
    if group is draft_tp_group:
        return None
    if group.world_size != embedding.tp_size:
        raise ValueError(
            "DFlash embedding TP group does not match its vocabulary shards"
        )

    if isinstance(embedding.quant_method, UnquantizedEmbeddingMethod):
        weight = embedding.weight
    else:
        ids = torch.arange(
            embedding.num_embeddings_per_partition,
            device=next(embedding.parameters()).device,
        )
        weight = embedding.quant_method.embedding(embedding, ids)
    output_dtype = getattr(embedding, "output_dtype", None)
    if output_dtype is not None:
        weight = weight.to(output_dtype)
    shards = group.all_gather(weight, dim=0).view(embedding.tp_size, *weight.shape)
    full_weight = weight.new_empty((embedding.num_embeddings, weight.shape[1]))
    for rank, shard in enumerate(shards):
        indices = embedding._get_indices(
            embedding.num_embeddings_padded,
            embedding.org_vocab_size_padded,
            embedding.num_embeddings,
            embedding.org_vocab_size,
            rank,
            embedding.tp_size,
        )
        full_weight[indices.org_vocab_start_index : indices.org_vocab_end_index].copy_(
            shard[: indices.num_org_elements]
        )
        offset = indices.num_org_elements_padded
        full_weight[
            indices.added_vocab_start_index : indices.added_vocab_end_index
        ].copy_(shard[offset : offset + indices.num_added_elements])
    return full_weight
