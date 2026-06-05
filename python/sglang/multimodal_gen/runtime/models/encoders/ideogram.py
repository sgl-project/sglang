# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Tuple

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.configs.models.encoders.ideogram import (
    Ideogram4TextEncoderConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.weight_only_fp8 import (
    swap_linears_to_weight_only_fp8,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import default_weight_loader
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.encoders.base import TextEncoder
from sglang.multimodal_gen.runtime.models.encoders.qwen3vl import Qwen3VLTextModel


class IdeogramQwen3VLTextEncoder(TextEncoder):
    """Language-only Qwen3-VL text encoder stored inside Ideogram checkpoints."""

    _activation_layers = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)

    def __init__(self, config: Ideogram4TextEncoderConfig) -> None:
        super().__init__(config)
        arch_config = config.arch_config
        text_config = getattr(arch_config, "text_config")
        if isinstance(text_config, dict):
            text_config = Qwen3VLTextConfig(**text_config)
        self.language_model = Qwen3VLTextModel(text_config)
        if getattr(arch_config, "ideogram_fp8_weight_only", False):
            swap_linears_to_weight_only_fp8(self.language_model)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> BaseEncoderOutput:
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        return BaseEncoderOutput(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def encode_ideogram_features(
        self,
        token_ids: torch.Tensor,
        text_position_ids: torch.Tensor,
        indicator: torch.Tensor,
        llm_token_indicator: int,
    ) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        hidden_size = self.language_model.config.hidden_size
        out_dim = hidden_size * len(self._activation_layers)
        features = torch.zeros(
            batch_size,
            seq_len,
            out_dim,
            dtype=torch.float32,
            device=token_ids.device,
        )
        for batch_idx in range(batch_size):
            text_mask = indicator[batch_idx] == llm_token_indicator
            cur_token_ids = token_ids[batch_idx, text_mask].unsqueeze(0)
            if cur_token_ids.numel() == 0:
                continue
            pos_2d = text_position_ids[batch_idx, text_mask, 0].unsqueeze(0)
            position_ids = pos_2d[None, ...].expand(4, 1, -1)
            attention_mask = torch.ones_like(cur_token_ids)
            with set_forward_context(current_timestep=0, attn_metadata=None):
                outputs = self.forward(
                    input_ids=cur_token_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            assert outputs.hidden_states is not None
            selected = [outputs.hidden_states[i] for i in self._activation_layers]
            stacked = torch.stack(selected, dim=0).permute(1, 2, 3, 0)
            features[batch_idx, text_mask] = stacked.reshape(
                1, cur_token_ids.shape[1], -1
            )[0].to(torch.float32)
        return features

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loaded_params: set[str] = set()
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if name.startswith("visual."):
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            param = params_dict.get(name)
            if param is None:
                raise KeyError(
                    f"Unexpected weight name while loading Ideogram text encoder: {name}"
                )
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight.to(param.dtype))
            loaded_params.add(name)
        return loaded_params


EntryClass = IdeogramQwen3VLTextEncoder
