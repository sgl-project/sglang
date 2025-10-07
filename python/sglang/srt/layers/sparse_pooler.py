import os
from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from sglang.srt.model_executor.model_runner import ForwardBatch


@dataclass
class SparseEmbeddingOutput:
    embeddings: torch.Tensor  # [batch_size, vocab_size]


class SparsePooler(nn.Module):
    """A layer that pools hidden states into sparse vocabulary-space embeddings.

    This layer does the following:
    1. Applies a linear transformation + ReLU to get token-level weights
    2. Maps these weights to vocabulary positions using token IDs
    3. Aggregates weights for repeated tokens using max pooling
    4. Returns sparse embeddings in vocabulary space

    Attributes:
        config: Model configuration containing vocab_size and hidden_size
        sparse_linear: Linear layer for computing token weights
        vocab_size: Size of vocabulary for output embeddings
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()

        # Validate required attributes
        if not hasattr(config, "vocab_size"):
            raise AttributeError(
                f"Config {type(config)} missing required 'vocab_size' attribute"
            )
        if not hasattr(config, "hidden_size"):
            raise AttributeError(
                f"Config {type(config)} missing required 'hidden_size' attribute"
            )

        self.vocab_size = config.vocab_size
        self.sparse_linear = nn.Linear(config.hidden_size, 1)
        self._weights_loaded = False

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> SparseEmbeddingOutput:
        """
        Forward pass for sparse pooling.

        Args:
            hidden_states: Packed sequence hidden states [total_tokens, hidden_size]
            forward_batch: Batch information with sequence lengths and input_ids

        Returns:
            SparseEmbeddingOutput with embeddings of shape [batch_size, vocab_size]
        """
        if not self._weights_loaded:
            raise ValueError(
                "Sparse pooling weights not loaded. Call load_weights() first"
            )

        # Apply sparse linear + ReLU to get token weights
        token_weights = F.relu(self.sparse_linear(hidden_states)).squeeze(
            -1
        )  # [total_tokens]

        # Number of items in batch
        batch_len = len(forward_batch.extend_seq_lens)

        # Create batch indices for packed sequences
        batch_indices = torch.repeat_interleave(
            torch.arange(batch_len, device=hidden_states.device),
            forward_batch.extend_seq_lens,
        )

        # Create a tensor of (batch_idx, token_id) pairs
        token_indices = torch.stack([batch_indices, forward_batch.input_ids], dim=0)

        # Find unique pairs and their inverse mapping
        unique_indices, inverse_indices = torch.unique(
            token_indices, dim=1, return_inverse=True
        )

        # Create a tensor for the unique values and apply scatter_reduce
        unique_values = torch.zeros(
            unique_indices.shape[1],
            dtype=token_weights.dtype,
            device=token_weights.device,
        )
        unique_values.scatter_reduce_(
            0, inverse_indices, token_weights, reduce="amax", include_self=False
        )

        # Create the final sparse tensor
        sparse_embeddings = torch.sparse_coo_tensor(
            unique_indices,
            unique_values,
            (batch_len, self.vocab_size),
        )

        return SparseEmbeddingOutput(embeddings=sparse_embeddings)

    def load_weights(self, state_dict: dict):
        """Load weights from state dict (called by the model)."""
        self.sparse_linear.load_state_dict(state_dict)
        self._weights_loaded = True
