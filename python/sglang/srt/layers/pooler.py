# adapted from
# https://github.com/vllm-project/vllm/blob/82a1b1a82b1fbb454c82a9ef95730b929c9b270c/vllm/model_executor/layers/pooler.py

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from sglang.srt.layers.activation import get_cross_encoder_activation_function

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class PoolingType(IntEnum):
    LAST = 0
    CLS = 1


@dataclass
class EmbeddingPoolerOutput:
    """Output of pooler or score_and_pool.

    Attributes:
        embeddings: Pooled embeddings or classification logits.  May be a list
            of tensors when per-request matryoshka dim truncation produces
            different shapes, or when MIS yields a variable number of scores
            per request.
        pooled_hidden_states: Raw transformer hidden states *before* the
            task-specific head, present only when
            ``forward_batch.return_pooled_hidden_states`` is True.  Tensor
            (standard path) or list of tensors (MIS path, one per delimiter).
    """

    # Pooler can return list[tensor] instead of tensor if the dimension of each tensor in the batch is different
    # due to different per-request matryoshka dim truncation
    embeddings: torch.Tensor | list[torch.Tensor]
    pooled_hidden_states: Optional[torch.Tensor | list[torch.Tensor]] = None


def pool_hidden_states(
    pooling_type: PoolingType,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
) -> torch.Tensor:
    """Pool hidden_states by PoolingType (LAST/CLS).

    Raw pooling only — no normalize, no dim truncation.
    Returns shape (batch_size, hidden_size).
    """
    if pooling_type == PoolingType.LAST:
        last_token_indices = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
        return hidden_states[last_token_indices]
    elif pooling_type == PoolingType.CLS:
        prompt_lens = forward_batch.extend_seq_lens
        first_token_flat_indices = torch.zeros_like(prompt_lens)
        first_token_flat_indices[1:] += torch.cumsum(prompt_lens, dim=0)[:-1]
        return hidden_states[first_token_flat_indices]
    else:
        raise ValueError(f"Unsupported pooling type: {pooling_type}")


def pool_at_delimiter_positions(
    data: torch.Tensor,
    forward_batch: ForwardBatch,
    device: torch.device,
) -> List[torch.Tensor]:
    """Pool a tensor at the position before each MIS delimiter for every request.

    Uses pre-computed delimiter indices from ForwardBatch (CPU tensors),
    moves to GPU with non_blocking=True to avoid CUDA syncs.

    Args:
        data: 2-D tensor [total_tokens, dim] — hidden states or logits.
        forward_batch: Forward batch with extend_seq_lens_cpu and
                       multi_item_delimiter_indices populated.
        device: Device for the index tensor.

    Returns:
        One tensor per request, shaped [num_delimiters, dim].
    """
    all_index_tensors: List[torch.Tensor] = []
    delim_counts: List[int] = []
    offset = 0
    for req_idx, req_seq_len in enumerate(forward_batch.extend_seq_lens_cpu):
        indices_tensor = forward_batch.multi_item_delimiter_indices[req_idx]
        n = len(indices_tensor)
        if n > 0:
            # Note: if the first delimiter is at position 0 (empty query),
            # indices - 1 wraps to -1. This is harmless — the first delimiter
            # entry is always discarded by _process_multi_item_scoring_results.
            all_index_tensors.append(indices_tensor + (offset - 1))
        delim_counts.append(n)
        offset += req_seq_len

    if all_index_tensors:
        index_tensor = torch.cat(all_index_tensors).to(device, non_blocking=True)
    else:
        index_tensor = torch.tensor([], dtype=torch.long, device=device)
    return list(data[index_tensor].split(delim_counts))


def score_and_pool(
    score_head: nn.Module,
    pooler: "Pooler",
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    input_ids: torch.Tensor,
) -> EmbeddingPoolerOutput:
    """Apply a classification/score head with MIS and pooled-hidden-states support.

    MIS path (pre-computed delimiter indices on forward_batch): extract hidden
    states at positions just before each delimiter, apply the score head, then
    split per-request.

    Standard path: pool hidden states, then apply the score head.

    When ``forward_batch.return_pooled_hidden_states`` is True, the raw pooled
    hidden states (before the score head) are included in the output.
    """
    if (
        forward_batch.multi_item_delimiter_indices is not None
        and forward_batch.is_prefill_only
    ):
        # Pool hidden states at pre-delimiter positions, score only those —
        # avoids wasting compute on tokens that never contribute to the output.
        # pool_at_delimiter_positions returns one tensor per request; we concat
        # to call score_head once, then split back per request.
        per_request_phs = pool_at_delimiter_positions(
            hidden_states, forward_batch, input_ids.device
        )
        phs_flat = torch.cat(per_request_phs, dim=0)
        scores_flat = score_head(phs_flat)
        delim_counts = [t.shape[0] for t in per_request_phs]
        per_request_scores = list(scores_flat.split(delim_counts))
        return EmbeddingPoolerOutput(
            embeddings=per_request_scores,
            pooled_hidden_states=(
                per_request_phs if forward_batch.return_pooled_hidden_states else None
            ),
        )

    # Standard classification path: pool hidden states, then score.
    pooled_hs = pool_hidden_states(pooler.pooling_type, hidden_states, forward_batch)
    scores = score_head(pooled_hs)
    return EmbeddingPoolerOutput(
        embeddings=scores,
        pooled_hidden_states=(
            pooled_hs if forward_batch.return_pooled_hidden_states else None
        ),
    )


class Pooler(nn.Module):
    """A layer that pools specific information from hidden states.
    This layer does the following:
    1. Extracts specific tokens or aggregates data based on pooling method.
    2. Normalizes output if specified.
    3. Returns structured results as `PoolerOutput`.
    Attributes:
        pooling_type: The type of pooling to use (LAST, AVERAGE, MAX).
        normalize: Whether to normalize the pooled data.
    """

    def __init__(self, pooling_type: PoolingType, normalize: bool):
        super().__init__()
        self.pooling_type = pooling_type
        self.normalize = normalize

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> EmbeddingPoolerOutput:
        pooled_data = pool_hidden_states(
            self.pooling_type, hidden_states, forward_batch
        )

        if forward_batch.dimensions is not None:
            all_same_dimensions = len(set(forward_batch.dimensions)) == 1
            if all_same_dimensions:
                pooled_data = pooled_data[..., : forward_batch.dimensions[0]]
            else:
                pooled_data = [
                    tensor[..., :dim]
                    for tensor, dim in zip(pooled_data, forward_batch.dimensions)
                ]

        if self.normalize:
            if isinstance(pooled_data, list):
                pooled_data = [
                    nn.functional.normalize(tensor, p=2, dim=-1)
                    for tensor in pooled_data
                ]
            else:
                pooled_data = nn.functional.normalize(pooled_data, p=2, dim=-1)

        return EmbeddingPoolerOutput(embeddings=pooled_data)


class CrossEncodingPooler(nn.Module):
    """A layer that pools specific information from hidden states.

    This layer does the following:
    1. Extracts specific tokens or aggregates data based on pooling method.
    2. Normalizes output if specified.
    3. Returns structured results as `EmbeddingPoolerOutput`.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        classifier: nn.Module,
        pooler: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.classifier = classifier
        self.pooler = pooler
        self.default_activation_function = get_cross_encoder_activation_function(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> EmbeddingPoolerOutput:
        """Pools sentence pair scores from the hidden_states."""

        prompt_lens = forward_batch.extend_seq_lens

        offset = 0
        pooled_data_lst = []
        for prompt_len in prompt_lens:
            pooled_data_i = hidden_states[offset : offset + prompt_len]

            if self.pooler is not None:
                final_shape_tensor = self.pooler(pooled_data_i, forward_batch)
            else:
                final_shape_tensor = self.classifier(pooled_data_i)

            pooled_data_lst.append(final_shape_tensor)
            offset += prompt_len

        pooled_output = torch.stack(pooled_data_lst)

        if self.pooler is not None:
            # apply classifier once on the full batch if possible
            pooled_output = self.classifier(pooled_output)

        scores = self.default_activation_function(pooled_output).squeeze(-1)

        return EmbeddingPoolerOutput(embeddings=scores)
