"""Thin wrapper adapting ColaTextVAEModel to the framework's VAE convention.

Cola-DLM's Text VAE operates on token sequences (not images/video). It encodes
token IDs to continuous latents and decodes latents back to vocabulary logits.
This wrapper provides a unified interface while delegating to the original model.
"""

from __future__ import annotations

from torch import nn


class ColaTextVAEWrapper(nn.Module):
    """Wrapper adapting ColaTextVAEModel to the framework's VAE convention."""

    def __init__(self, config=None):
        super().__init__()
        self._model: nn.Module | None = None
        self.config = config

    def load_model(self, model: nn.Module) -> None:
        """Inject the pre-loaded ColaTextVAEModel instance."""
        self._model = model

    def encode(self, input_ids_list, **kwargs):
        """Encode token IDs to continuous latents.

        Args:
            input_ids_list: List of token ID tensors.

        Returns:
            Encoded latents with .latents_list attribute.
        """
        return self._model.encode(input_ids_list, **kwargs)

    def set_kv_cache(self, flag: bool) -> None:
        """Enable or disable KV cache on the underlying VAE model."""
        self._model.set_kv_cache(flag)

    def decode(self, z, txt_shape, txt_q_shape, update_kv=False, **kwargs):
        """Decode continuous latents to vocabulary logits.

        Args:
            z: Latent tensor.
            txt_shape: Full sequence shape tensor (K length).
            txt_q_shape: Query shape tensor (Q length).
            update_kv: Whether to update KV cache.

        Returns:
            Decoded logits tensor.
        """
        return self._model.decode(
            z=z,
            txt_shape=txt_shape,
            txt_q_shape=txt_q_shape,
            update_kv=update_kv,
        )


EntryClass = ColaTextVAEWrapper
