# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.configs.neo_chat import NEOChatConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from sglang.srt.utils import add_prefix


@dataclass(frozen=True, slots=True)
class NEOVLMInputInfo:
    """U1 VLM index metadata derived from token ids and image grids."""

    thw_indexes: torch.Tensor
    image_context_token_count: int
    image_token_count: int


def precompute_rope_freqs_sincos(
    dim: int,
    max_position: int,
    base: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, device=device).float() / dim)
    )
    positions = torch.arange(max_position, device=device).type_as(inv_freq)
    freqs = torch.outer(positions, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


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


def apply_rotary_emb_1d(
    x: torch.Tensor,
    cos_cached: torch.Tensor,
    sin_cached: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    cos = cos_cached[positions]
    sin = sin_cached[positions]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rotated = torch.empty_like(x)
    rotated[..., 0::2] = x1 * cos - x2 * sin
    rotated[..., 1::2] = x1 * sin + x2 * cos
    return rotated


def apply_2d_rotary_pos_emb(
    x: torch.Tensor,
    cos_cached_x: torch.Tensor,
    sin_cached_x: torch.Tensor,
    cos_cached_y: torch.Tensor,
    sin_cached_y: torch.Tensor,
    abs_positions_x: torch.Tensor,
    abs_positions_y: torch.Tensor,
) -> torch.Tensor:
    dim_half = x.shape[-1] // 2
    rotated_x = apply_rotary_emb_1d(
        x[..., :dim_half],
        cos_cached_x,
        sin_cached_x,
        abs_positions_x,
    )
    rotated_y = apply_rotary_emb_1d(
        x[..., dim_half:],
        cos_cached_y,
        sin_cached_y,
        abs_positions_y,
    )
    return torch.cat((rotated_x, rotated_y), dim=-1)


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
    if "mot_gen" in name:
        return None
    if name.startswith("language_model."):
        return name[len("language_model.") :]
    if name.startswith("model.") or name.startswith("lm_head."):
        return name
    return None


class NEOVisionEmbeddings(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = int(config.hidden_size)
        self.llm_embed_dim = int(config.llm_hidden_size)
        self.downsample_factor = _merge_size_from_downsample_ratio(
            float(config.downsample_ratio)
        )
        self.patch_size = int(config.patch_size)

        self.patch_embedding = nn.Conv2d(
            in_channels=int(config.num_channels),
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.dense_embedding = nn.Conv2d(
            in_channels=self.embed_dim,
            out_channels=self.llm_embed_dim,
            kernel_size=self.downsample_factor,
            stride=self.downsample_factor,
        )
        self.gelu = nn.GELU()
        self.rope_dim_part = self.embed_dim // 2
        cos_x, sin_x = precompute_rope_freqs_sincos(
            self.rope_dim_part,
            int(config.max_position_embeddings_vision),
            base=float(config.rope_theta_vision),
        )
        cos_y, sin_y = precompute_rope_freqs_sincos(
            self.rope_dim_part,
            int(config.max_position_embeddings_vision),
            base=float(config.rope_theta_vision),
        )
        self.register_buffer("cos_cached_x", cos_x, persistent=False)
        self.register_buffer("sin_cached_x", sin_x, persistent=False)
        self.register_buffer("cos_cached_y", cos_y, persistent=False)
        self.register_buffer("sin_cached_y", sin_y, persistent=False)

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embedding.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embedding.weight.device

    def forward(self, pixel_values: torch.Tensor, grid_hw: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 2:
            raise ValueError(
                "U1 native vision expects flattened patch pixels with shape "
                f"(num_patches, 3 * patch_size^2), got {tuple(pixel_values.shape)}"
            )
        grid_hw = grid_hw.to(device=pixel_values.device, dtype=torch.long)
        pixel_values = pixel_values.view(-1, 3, self.patch_size, self.patch_size)
        patch_embeds = self.gelu(self.patch_embedding(pixel_values)).view(
            -1,
            self.embed_dim,
        )
        patch_embeds = self._apply_2d_rotary_pos_emb(patch_embeds, grid_hw)
        expected_patches = int((grid_hw[:, 0] * grid_hw[:, 1]).sum().item())
        if expected_patches != patch_embeds.shape[0]:
            raise ValueError(
                "U1 grid_hw patch count does not match pixel_values: "
                f"{expected_patches} != {patch_embeds.shape[0]}"
            )

        outputs = []
        cursor = 0
        for h, w in grid_hw.tolist():
            num_patches = int(h) * int(w)
            patches = patch_embeds[cursor : cursor + num_patches]
            patches = patches.view(int(h), int(w), -1).unsqueeze(0)
            patches = self.dense_embedding(patches.permute(0, 3, 1, 2))
            patches = patches.permute(0, 2, 3, 1)
            patches = patches.reshape(-1, patches.shape[-1])
            outputs.append(patches)
            cursor += num_patches

        embeddings = torch.cat(outputs, dim=0) if outputs else patch_embeds[:0]
        expected_embeddings = expected_patches // (self.downsample_factor**2)
        if embeddings.shape[0] != expected_embeddings:
            raise ValueError(
                "U1 dense image embedding count mismatch: "
                f"{embeddings.shape[0]} != {expected_embeddings}"
            )
        return embeddings

    def _apply_2d_rotary_pos_emb(
        self,
        patch_embeds: torch.Tensor,
        grid_hw: torch.Tensor,
    ) -> torch.Tensor:
        abs_pos_x, abs_pos_y = build_abs_positions_from_grid_hw(
            grid_hw,
            device=patch_embeds.device,
        )
        return apply_2d_rotary_pos_emb(
            patch_embeds.to(torch.float32),
            self.cos_cached_x.to(patch_embeds.device),
            self.sin_cached_x.to(patch_embeds.device),
            self.cos_cached_y.to(patch_embeds.device),
            self.sin_cached_y.to(patch_embeds.device),
            abs_pos_x,
            abs_pos_y,
        ).to(self.patch_embedding.weight.dtype)


class NEOVisionModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embeddings = NEOVisionEmbeddings(config)

    @property
    def dtype(self) -> torch.dtype:
        return self.embeddings.dtype

    @property
    def device(self) -> torch.device:
        return self.embeddings.device

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        *,
        pixel_embeds: torch.Tensor | None = None,
        grid_hw: torch.Tensor | None = None,
        **kwargs,
    ):
        del kwargs
        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        else:
            if pixel_values is None or grid_hw is None:
                raise ValueError("U1 vision forward requires pixel_values and grid_hw")
            hidden_states = self.embeddings(pixel_values, grid_hw=grid_hw)
        return type(
            "NEOVisionOutput",
            (),
            {
                "last_hidden_state": hidden_states,
                "pooler_output": None,
                "hidden_states": None,
                "attentions": None,
            },
        )()


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
        self.vision_model = NEOVisionModel(config.vision_config)
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
        del input_embeds
        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={Modality.IMAGE: self.get_image_feature},
            positions=positions,
            get_embedding=get_embedding,
            pp_proxy_tensors=pp_proxy_tensors,
        )

    def pad_input_ids(self, input_ids: list[int], mm_inputs: MultimodalInputs):
        if getattr(mm_inputs, "im_token_id", None) is None:
            mm_inputs.im_token_id = self.img_context_token_id
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        if not items:
            hidden_size = int(self.config.llm_config.hidden_size)
            return torch.empty(0, hidden_size, device=self.vision_model.device)
        if all(item.precomputed_embeddings is not None for item in items):
            return torch.cat([item.precomputed_embeddings for item in items], dim=0)

        pixel_values = torch.cat([item.feature for item in items], dim=0).to(
            device=self.vision_model.device,
            dtype=self.vision_model.dtype,
        )
        grid_hw = torch.cat([_u1_item_grid_hw(item) for item in items], dim=0).to(
            device=self.vision_model.device,
            dtype=torch.long,
        )
        return self.vision_model(
            pixel_values=pixel_values,
            grid_hw=grid_hw,
        ).last_hidden_state

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
        params_dict = dict(self.named_parameters())

        def llm_weights():
            for name, loaded_weight in weights:
                if name.startswith("vision_model."):
                    if name in params_dict:
                        param = params_dict[name]
                        weight_loader = getattr(
                            param,
                            "weight_loader",
                            default_weight_loader,
                        )
                        weight_loader(param, loaded_weight)
                    continue
                mapped_name = map_u1_language_model_weight_name(name)
                if mapped_name is not None:
                    yield mapped_name, loaded_weight

        self.language_model.load_weights(llm_weights())

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


def _u1_item_grid_hw(item: MultimodalDataItem) -> torch.Tensor:
    grid_hw = getattr(item, "image_grid_hws", None)
    if grid_hw is None:
        grid_hw = getattr(item, "grid_hw", None)
    if grid_hw is None:
        grid_hw = getattr(item, "image_grid_hw", None)
    if grid_hw is None:
        raise ValueError("U1 image item requires image_grid_hws/grid_hw metadata")
    if not torch.is_tensor(grid_hw):
        grid_hw = torch.tensor(grid_hw, dtype=torch.long)
    if grid_hw.ndim == 1:
        grid_hw = grid_hw.view(1, 2)
    if grid_hw.ndim != 2 or grid_hw.shape[-1] != 2:
        raise ValueError(f"U1 image grid_hw must have shape (B, 2), got {grid_hw.shape}")
    return grid_hw


EntryClass = NEOChatModel
