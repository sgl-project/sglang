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
from sglang.srt.server_args import get_global_server_args

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


def score_and_pool(
    score_head: nn.Module,
    pooler: "Pooler",
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    input_ids: torch.Tensor,
) -> EmbeddingPoolerOutput:
    """Apply a classification/score head with MIS and pooled-hidden-states support.

    MIS path (when ``multi_item_scoring_delimiter`` is set and found in ``input_ids``):
    extract hidden states at positions just before each delimiter, apply the score head,
    then split per-request.

    Standard path: apply the score head to all hidden states, then pool.

    When ``forward_batch.return_pooled_hidden_states`` is True, the raw pooled
    hidden states (before the score head) are included in the output.
    """
    delimiter_token = get_global_server_args().multi_item_scoring_delimiter
    if delimiter_token is not None and forward_batch.is_prefill_only:
        delim_positions = (input_ids == delimiter_token).nonzero(as_tuple=True)[0]
        # A delimiter at flat index 0 has no preceding hidden state to pool
        delim_positions = delim_positions[delim_positions > 0]

        if delim_positions.numel() > 0:
            # Score only the tokens that precede a delimiter
            pre_delim_hidden = hidden_states[delim_positions - 1]
            scores = score_head(pre_delim_hidden)

            # Split per-request so the scheduler gets one tensor per request.
            # Use CPU sequence lengths to avoid per-iteration GPU<->CPU sync
            # from `.item()` calls on device tensors.
            seq_lens = forward_batch.extend_seq_lens_cpu
            start = 0
            per_request_scores: List[torch.Tensor] = []
            per_request_phs: Optional[List[torch.Tensor]] = (
                [] if forward_batch.return_pooled_hidden_states else None
            )
            for seq_len in seq_lens:
                end = start + seq_len
                mask = (delim_positions >= start) & (delim_positions < end)
                per_request_scores.append(scores[mask])
                if per_request_phs is not None:
                    per_request_phs.append(pre_delim_hidden[mask])
                start = end

            return EmbeddingPoolerOutput(
                embeddings=per_request_scores,
                pooled_hidden_states=per_request_phs,
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
