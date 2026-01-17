# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Type definitions for logit lens output."""

import dataclasses
from typing import Dict, List, Optional

import torch


@dataclasses.dataclass
class LogitLensLayerResult:
    """Results from probing a single layer."""

    layer_id: int
    # Top-k token IDs at this layer [batch, top_k]
    top_token_ids: torch.Tensor
    # Top-k probabilities (after softmax) [batch, top_k]
    top_probs: torch.Tensor
    # Entropy of the distribution at this layer [batch]
    entropy: torch.Tensor
    # KL divergence from final layer distribution [batch] (only computed if final_logits provided)
    kl_from_final: Optional[torch.Tensor] = None


@dataclasses.dataclass
class LogitLensOutput:
    """Complete logit lens output for a batch."""

    # Per-layer results, keyed by layer_id
    layer_results: Dict[int, LogitLensLayerResult]
    # Which layer IDs were probed (in order)
    probed_layers: List[int]
    # Total number of layers in the model
    total_layers: int
    # Final layer's top-k for reference
    final_top_token_ids: Optional[torch.Tensor] = None
    final_top_probs: Optional[torch.Tensor] = None

    def to_dict(self, tokenizer=None) -> dict:
        """Convert to JSON-serializable dict, optionally decoding tokens."""
        result = {
            "probed_layers": self.probed_layers,
            "total_layers": self.total_layers,
            "layers": {},
        }

        for layer_id, layer_result in self.layer_results.items():
            layer_data = {
                "layer_id": layer_id,
                "top_token_ids": layer_result.top_token_ids.tolist(),
                "top_probs": layer_result.top_probs.tolist(),
                "entropy": layer_result.entropy.tolist(),
            }
            if layer_result.kl_from_final is not None:
                layer_data["kl_from_final"] = layer_result.kl_from_final.tolist()

            # Decode tokens if tokenizer provided
            if tokenizer is not None:
                batch_tokens = []
                for batch_idx in range(layer_result.top_token_ids.shape[0]):
                    tokens = []
                    for tok_id in layer_result.top_token_ids[batch_idx].tolist():
                        try:
                            tokens.append(tokenizer.decode([tok_id]))
                        except Exception:
                            tokens.append(f"<{tok_id}>")
                    batch_tokens.append(tokens)
                layer_data["top_tokens"] = batch_tokens

            result["layers"][str(layer_id)] = layer_data

        # Add final layer reference
        if self.final_top_token_ids is not None:
            result["final"] = {
                "top_token_ids": self.final_top_token_ids.tolist(),
                "top_probs": (
                    self.final_top_probs.tolist()
                    if self.final_top_probs is not None
                    else None
                ),
            }
            if tokenizer is not None:
                batch_tokens = []
                for batch_idx in range(self.final_top_token_ids.shape[0]):
                    tokens = []
                    for tok_id in self.final_top_token_ids[batch_idx].tolist():
                        try:
                            tokens.append(tokenizer.decode([tok_id]))
                        except Exception:
                            tokens.append(f"<{tok_id}>")
                    batch_tokens.append(tokens)
                result["final"]["top_tokens"] = batch_tokens

        return result
