"""Thin wrapper adapting ColaDiTModel to the framework's BaseDiT interface.

Cola-DLM's DiT uses a non-standard NA (no-padding) form with txt_shape/txt_q_shape
tensors and operates on continuous text latents rather than image latents. This wrapper
satisfies the BaseDiT contract while delegating to the original ColaDiTModel.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from sglang.multimodal_gen.configs.models.dits.base import DiTConfig
from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT


class ColaDiTWrapper(BaseDiT):
    """Wrapper adapting ColaDiTModel to the framework's BaseDiT interface."""

    _fsdp_shard_conditions: list = []
    _compile_conditions: list = []
    param_names_mapping: dict = {r"^(.*)$": r"\1"}
    reverse_param_names_mapping: dict = {}

    def __init__(
        self, config: DiTConfig, hf_config: dict[str, Any] | None = None, **kwargs
    ):
        super().__init__(config, hf_config=hf_config or {}, **kwargs)
        arch = config.arch_config
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.num_channels_latents
        self._model: nn.Module | None = None

    def load_model(self, model: nn.Module) -> None:
        """Inject the pre-loaded ColaDiTModel instance."""
        self._model = model

    def set_kv_cache(self, flag: bool) -> None:
        """Enable or disable KV cache on the underlying DiT model."""
        for block in self._model.blocks:
            block.set_kv_cache(flag)

    def forward(
        self,
        hidden_states: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor] | None = None,
        timestep: torch.LongTensor | None = None,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        guidance=None,
        txt: torch.Tensor | None = None,
        **kwargs,
    ) -> Any:
        """Framework-standard forward. Delegates to ColaDiTModel.forward().

        Maps framework args to Cola-DLM's interface:
        - hidden_states or txt → txt (latent sequence)
        - timestep → timestep
        - kwargs["txt_shape"], kwargs["txt_q_shape"] → Cola-DLM shape tensors
        - kwargs["update_kv"], kwargs["use_kv_cache"] → KV cache control
        """
        txt = txt if txt is not None else hidden_states
        txt_shape = kwargs.get("txt_shape")
        txt_q_shape = kwargs.get("txt_q_shape")
        update_kv = kwargs.get("update_kv", False)
        use_kv_cache = kwargs.get("use_kv_cache", False)

        return self._model(
            txt=txt,
            txt_shape=txt_shape,
            txt_q_shape=txt_q_shape,
            timestep=timestep,
            update_kv=update_kv,
            use_kv_cache=use_kv_cache,
        )


EntryClass = ColaDiTWrapper
