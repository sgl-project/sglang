# SPDX-License-Identifier: Apache-2.0
"""OmniDreams text-embedding transform (Cosmos-Reason1-7B ``full_concat``).

OmniDreams conditions cross-attention on a 100352-dim text embedding built from
the Cosmos-Reason1-7B (Qwen2.5-VL, 28 transformer layers x 3584 hidden) hidden
states. The transform is a faithful port of FlashDreams
``flashdreams/infra/encoder/text/cosmos_reason1.py``:

1. Run the LM with ``output_hidden_states=True`` -> tuple of ``num_layers + 1``
   tensors (index 0 is the embedding layer).
2. Drop index 0 (embedding layer); per-token mean/std-normalize each of the
   remaining 28 layers independently.
3. Concatenate along the feature dim -> ``28 * 3584 = 100352``.

This module holds the **pure tensor transform** (CPU-testable, no HF model). The
GPU-only encoder wrapper that actually runs Cosmos-Reason1-7B lives in the
pipeline's text-encoding stage and calls :func:`full_concat_embeddings` on the
LM's ``output_hidden_states``.
"""

from collections.abc import Sequence

import torch
from torch import Tensor

# Cosmos-Reason1-7B: 28 transformer layers x 3584 hidden = 100352 concat dim.
COSMOS_REASON1_NUM_LAYERS = 28
COSMOS_REASON1_HIDDEN = 3584
FULL_CONCAT_DIM = COSMOS_REASON1_NUM_LAYERS * COSMOS_REASON1_HIDDEN  # 100352

_NORM_EPS = 1e-8


def mean_normalize(tensor: Tensor) -> Tensor:
    """Per-token mean/std normalization over the last (feature) dim.

    Matches FlashDreams ``_mean_normalize``: ``std`` uses torch's default
    (unbiased / Bessel-corrected) estimator.
    """
    return (tensor - tensor.mean(dim=-1, keepdim=True)) / (
        tensor.std(dim=-1, keepdim=True) + _NORM_EPS
    )


def full_concat_embeddings(
    hidden_states: Sequence[Tensor],
    *,
    skip_embedding_layer: bool = True,
) -> Tensor:
    """Build the OmniDreams ``full_concat`` text embedding.

    Args:
        hidden_states: LM ``output_hidden_states`` -- a sequence of
            ``num_layers + 1`` tensors each ``[B, L, H]`` (index 0 is the
            embedding layer). For Cosmos-Reason1-7B this is 29 tensors of
            ``[B, L, 3584]``.
        skip_embedding_layer: drop ``hidden_states[0]`` (the embedding layer)
            before normalizing, as FlashDreams does.

    Returns:
        ``[B, L, num_transformer_layers * H]`` -- ``[B, L, 100352]`` for
        Cosmos-Reason1-7B.
    """
    start = 1 if skip_embedding_layer else 0
    layers = [
        mean_normalize(hidden_states[i]) for i in range(start, len(hidden_states))
    ]
    if not layers:
        raise ValueError(
            "full_concat_embeddings got no transformer layers to concatenate; "
            f"received {len(hidden_states)} hidden_states with "
            f"skip_embedding_layer={skip_embedding_layer}"
        )
    return torch.cat(layers, dim=-1)
