from __future__ import annotations

import torch
from torch import nn

from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding


@torch.no_grad()
def gather_draft_embedding(embedding: nn.Module) -> torch.Tensor | None:
    # Other modules may scale their output or own a checkpoint embedding.
    if not isinstance(embedding, VocabParallelEmbedding):
        return None
    if isinstance(embedding.quant_method, UnquantizedEmbeddingMethod):
        weight = embedding.weight
    else:
        # Packed weights are not embedding rows. Materialize the table through
        # the same quantization method as an ordinary embedding lookup.
        ids = torch.arange(
            embedding.num_embeddings_per_partition,
            device=next(embedding.parameters()).device,
        )
        weight = embedding.quant_method.embedding(embedding, ids)
    if embedding.output_dtype is not None:
        weight = weight.to(embedding.output_dtype)
    if embedding.tp_size == 1 and embedding.num_embeddings == embedding.org_vocab_size:
        return weight[: embedding.num_embeddings]

    shards = (
        embedding.tp_group.all_gather(weight, dim=0).view(
            embedding.tp_size, *weight.shape
        )
        if embedding.tp_size > 1
        else weight.unsqueeze(0)
    )
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
