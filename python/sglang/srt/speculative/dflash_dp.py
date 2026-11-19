from __future__ import annotations

import torch
from torch import nn

from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.utils import is_npu


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

    if embedding.tp_size > 1:
        if is_npu():
            # One-shot init gather. The device-side cross-node all_gather trips
            # CANN's aicpu RPC service ("stream is not registered with any
            # allocator"), so route it over the gloo group on the host instead.
            shard = weight.detach().to("cpu")
            gathered = [torch.empty_like(shard) for _ in range(embedding.tp_size)]
            torch.distributed.all_gather(
                gathered, shard, group=embedding.tp_group.cpu_group
            )
            shards = torch.stack(gathered).to(weight.device)
        else:
            shards = embedding.tp_group.all_gather(weight, dim=0).view(
                embedding.tp_size, *weight.shape
            )
    else:
        shards = weight.unsqueeze(0)
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
