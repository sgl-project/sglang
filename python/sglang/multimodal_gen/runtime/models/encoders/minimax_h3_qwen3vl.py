# SPDX-License-Identifier: Apache-2.0
"""Native, TP-foldable Qwen3-VL layer-50 encoder for MiniMax H3."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.models.encoders.base import BaseEncoderOutput
from sglang.multimodal_gen.configs.models.encoders.minimax_h3_qwen3vl import (
    MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER,
    MiniMaxH3Qwen3VLConfig,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.weight_utils import default_weight_loader
from sglang.multimodal_gen.runtime.models.encoders.base import TextEncoder
from sglang.multimodal_gen.runtime.models.encoders.qwen3vl import Qwen3VLModel

MINIMAX_H3_QWEN3VL_HIDDEN_DIM = 5120
_LAYER_WEIGHT_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")


def _is_unconsumed_checkpoint_weight(name: str) -> bool:
    """Weights intentionally absent from the layer-50 feature extractor."""

    if name == "lm_head.weight" or name.startswith("model.language_model.norm."):
        return True
    match = _LAYER_WEIGHT_RE.match(name)
    return bool(match and int(match.group(1)) >= MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER)


class MiniMaxH3Qwen3VLEncoder(TextEncoder):
    """Qwen3-VL-32B multimodal backbone ending at hidden_states[50].

    The component loader builds and loads this module under the encoder-folding
    TP group. A TP=1/SP=8 DiT deployment therefore shards the encoder over all
    eight otherwise-idle ranks during encoding.
    """

    supports_dp_encode = True

    @staticmethod
    def should_materialize_checkpoint_weight(name: str) -> bool:
        return (
            "rotary_emb.inv_freq" not in name
            and not _is_unconsumed_checkpoint_weight(name)
        )

    def __init__(self, config: MiniMaxH3Qwen3VLConfig) -> None:
        super().__init__(config)
        arch = config.arch_config
        selected_layer = MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER
        if int(arch.text_config.num_hidden_layers) != selected_layer:
            raise ValueError(
                "MiniMax H3 Qwen3-VL config must be trimmed to "
                f"{selected_layer} language layers before construction"
            )
        self.model = Qwen3VLModel(arch, use_tensor_parallel=True)
        # H3 consumes the unnormalized output immediately after layer 49.
        self.model.language_model.norm = nn.Identity()
        self.image_token_id = int(arch.image_token_id)
        self.video_token_id = int(arch.video_token_id)
        self.selected_lm_layer = selected_layer
        self.hidden_dim = MINIMAX_H3_QWEN3VL_HIDDEN_DIM

    @property
    def device(self) -> torch.device:
        """Device this encoder's forward runs on.

        Deliberately not `next(self.parameters()).device`. `--text-encoder-cpu-offload`
        loads this component under an FSDP CPU offload policy, which keeps the sharded
        parameters on CPU and all-gathers them to the accelerator for the forward. The
        parameter device then names the storage side, not the compute side, so inputs
        built from it stay on CPU while the forward runs on the accelerator.
        """
        return get_local_torch_device()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs: Any,
    ) -> BaseEncoderOutput:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
            **kwargs,
        )
        return BaseEncoderOutput(last_hidden_state=outputs.last_hidden_state)

    @torch.no_grad()
    def encode_ids(
        self,
        input_ids: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.dim() != 1:
            raise ValueError(f"input_ids must be 1-D, got {list(input_ids.shape)}")
        if (pixel_values is None) != (image_grid_thw is None):
            raise ValueError("pixel_values and image_grid_thw must be given together")
        if (pixel_values_videos is None) != (video_grid_thw is None):
            raise ValueError(
                "pixel_values_videos and video_grid_thw must be given together"
            )

        host_ids = input_ids.to(device="cpu", dtype=torch.long)[None]
        host_image_grid_thw = (
            image_grid_thw.to(device="cpu", dtype=torch.long)
            if image_grid_thw is not None
            else None
        )
        host_video_grid_thw = (
            video_grid_thw.to(device="cpu", dtype=torch.long)
            if video_grid_thw is not None
            else None
        )
        position_ids = None
        if host_image_grid_thw is not None or host_video_grid_thw is not None:
            position_ids, _ = self.model.get_rope_index(
                host_ids,
                host_image_grid_thw,
                host_video_grid_thw,
                attention_mask=torch.ones_like(host_ids),
            )
        ids = host_ids.to(self.device)
        call_kwargs: dict[str, Any] = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict": True,
            "use_cache": False,
        }
        if position_ids is not None:
            call_kwargs["position_ids"] = position_ids.to(self.device)
        if pixel_values is not None:
            call_kwargs["pixel_values"] = pixel_values.to(self.device, torch.bfloat16)
            call_kwargs["image_grid_thw"] = host_image_grid_thw
        if pixel_values_videos is not None:
            call_kwargs["pixel_values_videos"] = pixel_values_videos.to(
                self.device, torch.bfloat16
            )
            call_kwargs["video_grid_thw"] = host_video_grid_thw

        hidden = self.model(**call_kwargs).last_hidden_state[0].to(torch.bfloat16)
        expected_shape = [int(ids.shape[1]), self.hidden_dim]
        if list(hidden.shape) != expected_shape:
            raise ValueError(
                f"unexpected hidden shape {list(hidden.shape)}, "
                f"expected {expected_shape}"
            )
        return hidden

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        params = dict(self.named_parameters(remove_duplicate=False))
        loaded: set[str] = set()
        for name, loaded_weight in weights:
            if not self.should_materialize_checkpoint_weight(name):
                continue
            param = params.get(name)
            if param is None:
                raise KeyError(
                    f"Unexpected MiniMax H3 Qwen3-VL checkpoint weight: {name}"
                )
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            try:
                weight_loader(param, loaded_weight.to(param.dtype))
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load MiniMax H3 Qwen3-VL weight "
                    f"{name!r}: checkpoint={tuple(loaded_weight.shape)}, "
                    f"parameter={tuple(param.shape)}"
                ) from exc
            loaded.add(name)
        return loaded


EntryClass = MiniMaxH3Qwen3VLEncoder

__all__ = ["MiniMaxH3Qwen3VLEncoder"]
