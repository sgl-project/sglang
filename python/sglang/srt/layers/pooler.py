# adapted from
# https://github.com/vllm-project/vllm/blob/82a1b1a82b1fbb454c82a9ef95730b929c9b270c/vllm/model_executor/layers/pooler.py

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from sglang.srt.layers.activation import get_cross_encoder_activation_function
from sglang.srt.model_executor.model_runner import ForwardBatch


class PoolingType(IntEnum):
    LAST = 0
    CLS = 1
    MEAN = 2


@dataclass
class EmbeddingPoolerOutput:
    embeddings: torch.Tensor


class Pooler(nn.Module):
    """A layer that pools specific information from hidden states.
    This layer does the following:
    1. Extracts specific tokens or aggregates data based on pooling method.
    2. Normalizes output if specified.
    3. Returns structured results as `PoolerOutput`.
    Attributes:
        pooling_type: The type of pooling to use (LAST, CLS, MEAN).
        normalize: Whether to normalize the pooled data.
    """

    def __init__(self, pooling_type: PoolingType, normalize: bool):
        super().__init__()
        self.pooling_type = pooling_type
        self.normalize = normalize

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> EmbeddingPoolerOutput:
        if self.pooling_type == PoolingType.LAST:
            last_token_indices = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            pooled_data = hidden_states[last_token_indices]
        elif self.pooling_type == PoolingType.CLS:
            prompt_lens = forward_batch.extend_seq_lens
            first_token_flat_indices = torch.zeros_like(prompt_lens)
            first_token_flat_indices[1:] += torch.cumsum(prompt_lens, dim=0)[:-1]
            pooled_data = hidden_states[first_token_flat_indices]
        elif self.pooling_type == PoolingType.MEAN:
            seq_lens = forward_batch.extend_seq_lens
            num_sequences = seq_lens.shape[0]
            hidden_size = hidden_states.size(-1)

            device = hidden_states.device
            dtype = hidden_states.dtype

            # Build segment ids for each token in the flattened hidden_states
            segment_ids = torch.arange(num_sequences, device=device).repeat_interleave(
                seq_lens
            )

            # Sum hidden states per segment
            sums = torch.zeros((num_sequences, hidden_size), device=device, dtype=dtype)
            sums.scatter_add_(
                0,
                segment_ids.unsqueeze(1).expand(-1, hidden_size),
                hidden_states,
            )

            # Divide by sequence lengths to get means
            pooled_data = sums / seq_lens.to(dtype=dtype).unsqueeze(1)
        else:
            raise ValueError(f"Invalid pooling type: {self.pooling_type}")

        if self.normalize:
            pooled_data = nn.functional.normalize(pooled_data, p=2, dim=1)

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
