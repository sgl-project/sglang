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
"""
Logit Lens Extractor - Captures intermediate layer outputs and projects to vocabulary space.

The logit lens technique (Nostalgebraist, 2020) allows us to see what tokens the model
would predict at each layer by projecting intermediate hidden states through the
unembedding matrix (lm_head).

This helps visualize:
- When the model "commits" to a prediction
- How token predictions evolve through layers
- Which layers are most influential for specific predictions
"""

import logging
import threading
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.logit_lens.types import LogitLensLayerResult, LogitLensOutput

logger = logging.getLogger(__name__)


class LogitLensExtractor:
    """
    Extracts intermediate layer outputs and projects them to vocabulary space.

    Usage:
        extractor = LogitLensExtractor(model, num_layers=32)
        extractor.register_hooks([0, 8, 16, 24, 31])

        # During forward pass, hooks capture layer outputs
        output = model.forward(...)

        # Get logit lens results
        lens_output = extractor.compute_lens(lm_head, top_k=5)
        extractor.clear()  # Clear captured states
    """

    def __init__(
        self,
        model: nn.Module,
        num_layers: int,
        layer_pattern: str = "model.layers.{}.mlp",
        norm_module: Optional[nn.Module] = None,
    ):
        """
        Initialize the logit lens extractor.

        Args:
            model: The transformer model
            num_layers: Total number of layers in the model
            layer_pattern: Pattern for layer module names, with {} for layer index
            norm_module: Optional final layer norm to apply before projection
        """
        self.model = model
        self.num_layers = num_layers
        self.layer_pattern = layer_pattern
        self.norm_module = norm_module

        # Thread-local storage for captured hidden states
        self._local = threading.local()
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._registered_layers: List[int] = []

    @property
    def _captured_states(self) -> Dict[int, torch.Tensor]:
        """Thread-local captured hidden states."""
        if not hasattr(self._local, "captured_states"):
            self._local.captured_states = {}
        return self._local.captured_states

    def _get_module_for_layer(self, layer_id: int) -> Optional[nn.Module]:
        """Get the module for a specific layer."""
        # Try different patterns that work for various architectures
        patterns = [
            self.layer_pattern.format(layer_id),  # User-specified pattern
            f"model.layers.{layer_id}",  # Llama/Qwen style
            f"transformer.h.{layer_id}",  # GPT-2 style
            f"encoder.layer.{layer_id}",  # BERT style
        ]

        for pattern in patterns:
            parts = pattern.split(".")
            module = self.model
            try:
                for part in parts:
                    module = getattr(module, part)
                return module
            except AttributeError:
                continue

        return None

    def _create_hook(self, layer_id: int) -> Callable:
        """Create a forward hook for a specific layer."""

        def hook(module, input, output):
            # Handle different output formats
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Store only the last token's hidden state for each sequence
            # This reduces memory usage while still enabling logit lens analysis
            if hidden_states.dim() == 3:
                # [batch, seq_len, hidden] -> store last token [batch, hidden]
                self._captured_states[layer_id] = hidden_states[:, -1, :].detach()
            else:
                self._captured_states[layer_id] = hidden_states.detach()

        return hook

    def register_hooks(self, layer_ids: Optional[List[int]] = None) -> List[int]:
        """
        Register forward hooks on specified layers.

        Args:
            layer_ids: List of layer indices to probe. If None, auto-selects ~4 evenly spaced layers.

        Returns:
            List of layer IDs that were successfully registered.
        """
        # Clear any existing hooks
        self.remove_hooks()

        # Auto-select layers if not specified
        if layer_ids is None:
            # Select ~4 evenly spaced layers: early, mid-early, mid-late, late
            layer_ids = self._auto_select_layers(4)

        registered = []
        for layer_id in layer_ids:
            if layer_id < 0 or layer_id >= self.num_layers:
                logger.warning(
                    f"Layer {layer_id} out of range [0, {self.num_layers}), skipping"
                )
                continue

            module = self._get_module_for_layer(layer_id)
            if module is None:
                logger.warning(f"Could not find module for layer {layer_id}")
                continue

            hook = self._create_hook(layer_id)
            handle = module.register_forward_hook(hook)
            self._hooks.append(handle)
            registered.append(layer_id)

        self._registered_layers = registered
        logger.debug(f"Registered logit lens hooks on layers: {registered}")
        return registered

    def _auto_select_layers(self, count: int = 4) -> List[int]:
        """Auto-select evenly spaced layers for probing."""
        if self.num_layers <= count:
            return list(range(self.num_layers))

        # Include first, some middle layers, and last layer
        indices = []
        for i in range(count):
            idx = int(i * (self.num_layers - 1) / (count - 1))
            indices.append(idx)

        return sorted(set(indices))

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._registered_layers.clear()

    def clear(self):
        """Clear captured hidden states."""
        self._captured_states.clear()

    def compute_lens(
        self,
        lm_head: nn.Module,
        top_k: int = 5,
        final_logits: Optional[torch.Tensor] = None,
        sample_indices: Optional[torch.Tensor] = None,
    ) -> Optional[LogitLensOutput]:
        """
        Project captured hidden states to vocabulary space.

        Args:
            lm_head: The language model head (unembedding matrix)
            top_k: Number of top tokens to return per layer
            final_logits: Optional final layer logits for KL divergence computation
            sample_indices: Optional indices to select specific positions

        Returns:
            LogitLensOutput with per-layer predictions, or None if no states captured
        """
        if not self._captured_states:
            return None

        layer_results = {}
        probed_layers = sorted(self._captured_states.keys())

        # Get final layer distribution if provided
        final_probs = None
        final_top_ids = None
        final_top_probs = None
        if final_logits is not None:
            final_probs = F.softmax(final_logits.float(), dim=-1)
            final_top_probs, final_top_ids = torch.topk(final_probs, k=top_k, dim=-1)

        for layer_id in probed_layers:
            hidden_states = self._captured_states[layer_id]

            # Apply sample indices if provided
            if sample_indices is not None and hidden_states.dim() >= 2:
                hidden_states = hidden_states[sample_indices]

            # Apply final layer norm if available
            if self.norm_module is not None:
                hidden_states = self.norm_module(hidden_states)

            # Project to vocabulary space
            logits = self._project_to_vocab(hidden_states, lm_head)

            # Compute probabilities
            probs = F.softmax(logits.float(), dim=-1)

            # Get top-k predictions
            top_probs, top_ids = torch.topk(
                probs, k=min(top_k, probs.shape[-1]), dim=-1
            )

            # Compute entropy
            log_probs = F.log_softmax(logits.float(), dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1)

            # Compute KL divergence from final layer
            kl_div = None
            if final_probs is not None:
                # KL(final || layer) - how surprised would the final layer be by this layer's prediction
                kl_div = F.kl_div(
                    log_probs, final_probs, reduction="none", log_target=False
                ).sum(dim=-1)

            layer_results[layer_id] = LogitLensLayerResult(
                layer_id=layer_id,
                top_token_ids=top_ids,
                top_probs=top_probs,
                entropy=entropy,
                kl_from_final=kl_div,
            )

        return LogitLensOutput(
            layer_results=layer_results,
            probed_layers=probed_layers,
            total_layers=self.num_layers,
            final_top_token_ids=final_top_ids,
            final_top_probs=final_top_probs,
        )

    def _project_to_vocab(
        self, hidden_states: torch.Tensor, lm_head: nn.Module
    ) -> torch.Tensor:
        """Project hidden states to vocabulary logits."""
        # Handle different lm_head types
        if hasattr(lm_head, "set_lora") and hasattr(lm_head, "apply_lora"):
            # LoRA-wrapped module
            return lm_head(hidden_states)
        elif hasattr(lm_head, "weight"):
            # Standard linear layer
            return torch.matmul(
                hidden_states.to(lm_head.weight.dtype), lm_head.weight.T
            )
        elif hasattr(lm_head, "quant_method"):
            # Quantized layer
            return lm_head.quant_method.apply(lm_head, hidden_states, None)
        else:
            # Fallback: assume it's a callable
            return lm_head(hidden_states)

    def get_layer_names(self) -> Dict[int, str]:
        """Get the module names for each registered layer."""
        names = {}
        for layer_id in self._registered_layers:
            module = self._get_module_for_layer(layer_id)
            if module is not None:
                # Find the name in the model
                for name, mod in self.model.named_modules():
                    if mod is module:
                        names[layer_id] = name
                        break
        return names


def create_logit_lens_extractor(
    model: nn.Module,
    config,
    norm_module: Optional[nn.Module] = None,
) -> LogitLensExtractor:
    """
    Factory function to create a LogitLensExtractor with appropriate settings.

    Args:
        model: The transformer model
        config: Model configuration (should have num_hidden_layers)
        norm_module: Optional final layer norm

    Returns:
        Configured LogitLensExtractor
    """
    num_layers = getattr(config, "num_hidden_layers", 32)

    # Determine layer pattern based on model architecture
    layer_pattern = "model.layers.{}"  # Default for Llama/Qwen

    # Check for other architectures
    if hasattr(model, "transformer"):
        if hasattr(model.transformer, "h"):
            layer_pattern = "transformer.h.{}"
    elif hasattr(model, "encoder"):
        layer_pattern = "encoder.layer.{}"

    return LogitLensExtractor(
        model=model,
        num_layers=num_layers,
        layer_pattern=layer_pattern,
        norm_module=norm_module,
    )
