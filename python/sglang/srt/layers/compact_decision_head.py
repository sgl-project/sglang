"""Compact last-token readout for PPLX decision checkpoints."""

from typing import TYPE_CHECKING

import torch
from torch import nn

from sglang.srt.layers.pooler import (
    EmbeddingPoolerOutput,
    PoolingType,
    pool_hidden_states,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class CompactDecisionPooler(nn.Module):
    """Pool before the readout, without constructing a vocabulary-sized output.

    Rows retain decision_config.json's code order. Candidate selection and
    calibration belong to the serving layer, after this unnormalized readout.
    """

    def __init__(self, hidden_size: int, num_codes: int):
        super().__init__()
        self.readout = nn.Linear(hidden_size, num_codes, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: "ForwardBatch"
    ) -> EmbeddingPoolerOutput:
        pooled = pool_hidden_states(PoolingType.LAST, hidden_states, forward_batch)
        return EmbeddingPoolerOutput(
            embeddings=self.readout(pooled).float(),
            pooled_hidden_states=(
                pooled if forward_batch.return_pooled_hidden_states else None
            ),
        )
