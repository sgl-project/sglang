# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.configs.neo_chat import NEOChatConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from sglang.srt.utils import add_prefix


@dataclass(frozen=True, slots=True)
class NEOVLMInputInfo:
    """U1 VLM index metadata derived from token ids and image grids."""

    thw_indexes: torch.Tensor
    image_context_token_count: int
    image_token_count: int


def build_abs_positions_from_grid_hw(
    grid_hw: torch.Tensor,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return U1 row-major image patch x/y positions.

    This is the SGLang-owned equivalent of U1's official helper. It is kept
    model-local because it is a U1 spatial-index rule, not a generic UG rule.
    """

    if not torch.is_tensor(grid_hw):
        grid_hw = torch.tensor(grid_hw, dtype=torch.long, device=device)
    if device is None:
        device = grid_hw.device
    grid_hw = grid_hw.to(device=device, dtype=torch.long)
    if grid_hw.ndim != 2 or grid_hw.shape[-1] != 2:
        raise ValueError(f"grid_hw must have shape (B, 2), got {tuple(grid_hw.shape)}")

    height = grid_hw[:, 0]
    width = grid_hw[:, 1]
    patch_counts = height * width
    total_patches = int(patch_counts.sum().item())
    if total_patches == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    patch_to_sample = torch.repeat_interleave(
        torch.arange(grid_hw.shape[0], device=device),
        patch_counts,
    )
    starts = torch.cumsum(
        torch.cat(
            [torch.zeros(1, dtype=torch.long, device=device), patch_counts[:-1]],
            dim=0,
        ),
        dim=0,
    )
    patch_id_within_image = (
        torch.arange(total_patches, dtype=torch.long, device=device)
        - starts[patch_to_sample]
    )
    width_per_patch = width[patch_to_sample]
    abs_x = patch_id_within_image % width_per_patch
    abs_y = patch_id_within_image // width_per_patch
    return abs_x, abs_y


def build_u1_vlm_thw_indexes(
    input_ids: torch.Tensor | list[int] | tuple[int, ...],
    *,
    grid_hw: torch.Tensor | list[list[int]] | tuple[tuple[int, int], ...] | None = None,
    img_start_token_id: int = 151670,
    img_context_token_id: int = 151669,
    downsample_ratio: float = 0.5,
) -> torch.Tensor:
    """Build U1's T/H/W indexes for one interleaved VLM sequence."""

    if not torch.is_tensor(input_ids):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_ids = input_ids.to(dtype=torch.long)
    if input_ids.ndim != 1:
        raise ValueError(f"input_ids must be 1-D, got {tuple(input_ids.shape)}")

    img_start_shift = torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=input_ids.device),
            (input_ids == img_start_token_id).long(),
        ],
        dim=0,
    )[:-1]
    not_img_token = (input_ids != img_context_token_id).long()
    t_indexes = ((img_start_shift + not_img_token).cumsum(0) - 1).clamp_min(0)
    h_indexes = torch.zeros_like(t_indexes)
    w_indexes = torch.zeros_like(t_indexes)

    if grid_hw is not None and (input_ids == img_context_token_id).any():
        merge_size = _merge_size_from_downsample_ratio(downsample_ratio)
        if not torch.is_tensor(grid_hw):
            grid_hw = torch.tensor(
                grid_hw,
                dtype=torch.long,
                device=input_ids.device,
            )
        grid_hw = grid_hw.to(device=input_ids.device, dtype=torch.long)
        merged_grid_hw = grid_hw // merge_size
        abs_pos_w, abs_pos_h = build_abs_positions_from_grid_hw(
            merged_grid_hw,
            device=input_ids.device,
        )
        selected = input_ids == img_context_token_id
        selected_count = int(selected.long().sum().item())
        if selected_count != abs_pos_h.numel():
            raise ValueError(
                "U1 image context token count does not match grid_hw: "
                f"{selected_count} != {abs_pos_h.numel()}"
            )
        h_indexes[selected] = abs_pos_h.to(dtype=t_indexes.dtype)
        w_indexes[selected] = abs_pos_w.to(dtype=t_indexes.dtype)

    return torch.stack([t_indexes, h_indexes, w_indexes], dim=0)


def build_u1_vlm_input_info(
    input_ids: torch.Tensor | list[int] | tuple[int, ...],
    *,
    grid_hw: torch.Tensor | list[list[int]] | tuple[tuple[int, int], ...] | None = None,
    img_start_token_id: int = 151670,
    img_context_token_id: int = 151669,
    downsample_ratio: float = 0.5,
) -> NEOVLMInputInfo:
    if not torch.is_tensor(input_ids):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    thw_indexes = build_u1_vlm_thw_indexes(
        input_ids,
        grid_hw=grid_hw,
        img_start_token_id=img_start_token_id,
        img_context_token_id=img_context_token_id,
        downsample_ratio=downsample_ratio,
    )
    image_context_token_count = int((input_ids == img_context_token_id).long().sum())
    image_token_count = image_context_token_count + int(
        (input_ids == img_start_token_id).long().sum()
    )
    return NEOVLMInputInfo(
        thw_indexes=thw_indexes,
        image_context_token_count=image_context_token_count,
        image_token_count=image_token_count,
    )


def iter_u1_language_model_weights(
    weights: Iterable[Tuple[str, torch.Tensor]],
) -> Iterable[Tuple[str, torch.Tensor]]:
    """Route only U1 language-model weights into SRT's Qwen3 loader."""

    for name, loaded_weight in weights:
        mapped_name = map_u1_language_model_weight_name(name)
        if mapped_name is not None:
            yield mapped_name, loaded_weight


def map_u1_language_model_weight_name(name: str) -> str | None:
    if name.startswith("language_model."):
        return name[len("language_model.") :]
    if name.startswith("model.") or name.startswith("lm_head."):
        return name
    return None


class NEOChatModel(nn.Module):
    """Native SenseNova U1 model shell.

    The first native slice is deliberately narrow: SRT owns the Qwen3 U path
    and U1's VLM index semantics; vision/pixel-flow modules are added behind
    this model-local boundary in later steps.
    """

    def __init__(
        self,
        config: NEOChatConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.patch_size = config.vision_config.patch_size
        self.template = config.template
        self.downsample_ratio = config.downsample_ratio
        self.img_context_token_id = config.img_context_token_id
        self.img_start_token_id = config.img_start_token_id
        self.img_end_token_id = config.img_end_token_id
        self.language_model = Qwen3ForCausalLM(
            config=config.llm_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        self.model = self.language_model.model

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            get_embedding=get_embedding,
            pp_proxy_tensors=pp_proxy_tensors,
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.lm_head

    def get_embed_and_head(self):
        return self.language_model.get_embed_and_head()

    def set_embed_and_head(self, embed, head):
        return self.language_model.set_embed_and_head(embed, head)

    def get_u1_vlm_input_info(
        self,
        input_ids: torch.Tensor | list[int] | tuple[int, ...],
        *,
        grid_hw: torch.Tensor | list[list[int]] | tuple[tuple[int, int], ...] | None,
    ) -> NEOVLMInputInfo:
        return build_u1_vlm_input_info(
            input_ids,
            grid_hw=grid_hw,
            img_start_token_id=self.img_start_token_id,
            img_context_token_id=self.img_context_token_id,
            downsample_ratio=self.downsample_ratio,
        )

    def get_thw_indexes(
        self,
        input_ids: torch.Tensor,
        grid_hw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return build_u1_vlm_thw_indexes(
            input_ids,
            grid_hw=grid_hw,
            img_start_token_id=self.img_start_token_id,
            img_context_token_id=self.img_context_token_id,
            downsample_ratio=self.downsample_ratio,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        self.language_model.load_weights(iter_u1_language_model_weights(weights))

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.language_model.load_kv_cache_scales(quantization_param_path)


def _merge_size_from_downsample_ratio(downsample_ratio: float) -> int:
    if downsample_ratio <= 0:
        raise ValueError(f"downsample_ratio must be > 0, got {downsample_ratio}")
    merge_size = int(1 / downsample_ratio)
    if merge_size <= 0 or abs((1 / merge_size) - downsample_ratio) > 1e-6:
        raise ValueError(
            "U1 downsample_ratio must be the reciprocal of an integer, "
            f"got {downsample_ratio}"
        )
    return merge_size


EntryClass = NEOChatModel
