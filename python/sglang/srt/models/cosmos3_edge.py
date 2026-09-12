# Copyright 2023-2025 SGLang Team
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
"""Inference-only Cosmos3-Edge VLM.

Cosmos3-Edge stores a dense UND text tower in ``transformer/`` and a SigLIP2
vision tower plus projector in ``vision_encoder/``. The text tower matches the
Arcee causal-LM structure used by SGLang, while the vision path uses the native
SigLIP2 implementation and an Edge-specific spatial-merge projector.
"""

from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from sglang.srt.configs.cosmos3 import (
    Cosmos3EdgeConfig,
    Cosmos3EdgeProjectorConfig,
    Cosmos3EdgeTextConfig,
)
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    embed_mm_inputs,
)
from sglang.srt.managers.schedule_batch import (
    MultimodalDataItem,
    MultimodalInputFormat,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.arcee import ArceeForCausalLM
from sglang.srt.models.siglip2 import Siglip2Model
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.utils import add_prefix


class Cosmos3EdgeVisionProjector(nn.Module):
    def __init__(
        self,
        config: Cosmos3EdgeProjectorConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        self.use_postshuffle_norm = config.use_postshuffle_norm
        self.context_dim = config.input_hidden_size
        self.hidden_size = self.context_dim * (self.spatial_merge_size**2)

        norm_dim = self.hidden_size if self.use_postshuffle_norm else self.context_dim
        self.norm = nn.LayerNorm(norm_dim, eps=1e-6)
        self.linear_fc1 = ColumnParallelLinear(
            self.hidden_size,
            config.merger_intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("linear_fc1", prefix),
        )
        self.act_fn = nn.GELU()
        self.linear_fc2 = RowParallelLinear(
            config.merger_intermediate_size,
            config.out_hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("linear_fc2", prefix),
        )

    def _spatial_merge(
        self, vision_features: torch.Tensor, spatial_shapes: torch.Tensor
    ) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        hidden_size = vision_features.shape[-1]
        lengths = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).tolist()
        tiles = torch.split(vision_features, lengths, dim=0)

        merged_parts = []
        for tile, (height, width) in zip(tiles, spatial_shapes.tolist()):
            height = int(height)
            width = int(width)
            if height == 0 or width == 0:
                continue
            if height % merge_size != 0 or width % merge_size != 0:
                raise ValueError(
                    "Cosmos3-Edge vision grid must be divisible by "
                    f"spatial_merge_size={merge_size}, got {(height, width)}."
                )
            tile = tile.view(height, width, hidden_size)
            tile = tile.view(
                height // merge_size,
                merge_size,
                width // merge_size,
                merge_size,
                hidden_size,
            )
            tile = tile.permute(0, 2, 1, 3, 4).reshape(
                (height // merge_size) * (width // merge_size),
                merge_size * merge_size * hidden_size,
            )
            merged_parts.append(tile)

        if not merged_parts:
            return vision_features.new_empty((0, merge_size * merge_size * hidden_size))
        return torch.cat(merged_parts, dim=0)

    def forward(
        self, vision_features: torch.Tensor, spatial_shapes: torch.Tensor
    ) -> torch.Tensor:
        if self.use_postshuffle_norm:
            vision_features = self._spatial_merge(vision_features, spatial_shapes)
            vision_features = self.norm(vision_features)
        else:
            vision_features = self.norm(vision_features)
            vision_features = self._spatial_merge(vision_features, spatial_shapes)

        hidden_states, _ = self.linear_fc1(vision_features)
        hidden_states = self.act_fn(hidden_states)
        hidden_states, _ = self.linear_fc2(hidden_states)
        return hidden_states


class Cosmos3EdgeForConditionalGeneration(ArceeForCausalLM):
    # Multimodal serving needs both text and vision subfolders. In
    # --language-model-only mode __init__ narrows this instance attribute to the
    # transformer shards so the vision files are not downloaded or loaded.
    allow_patterns_overrides = ["[tv]*er/*.safetensors"]

    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_substr={
            # Drop ModelOpt calibration buffers and generation-side tensors.
            "_quantizer.": None,
            "_moe_gen": None,
            "k_norm_und_for_gen": None,
            ".add_q_proj.": None,
            ".add_k_proj.": None,
            ".add_v_proj.": None,
            ".to_add_out.": None,
            ".norm_added_q.": None,
            ".norm_added_k.": None,
            # Text attention projection names -> SGLang/Arcee names.
            ".to_q.": ".q_proj.",
            ".to_k.": ".k_proj.",
            ".to_v.": ".v_proj.",
            ".to_out.": ".o_proj.",
        },
        orig_to_new_prefix={
            "embed_tokens.": "model.embed_tokens.",
            "layers.": "model.layers.",
            "norm.": "model.norm.",
            # Diffusion-only top-level modules.
            "proj_in.": None,
            "proj_out.": None,
            "time_embedder.": None,
            "action_": None,
            "audio_": None,
            # Vision/projector weights are routed before applying this mapper.
            "model.visual.": None,
            "model.projector.": None,
        },
    )

    def __init__(
        self,
        config: Cosmos3EdgeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        self.root_config = config
        self.language_model_only = bool(getattr(config, "language_model_only", False))
        self.allow_patterns_overrides = (
            ["transformer/*.safetensors"]
            if self.language_model_only
            else ["[tv]*er/*.safetensors"]
        )

        text_config = getattr(config, "text_config", config)
        if isinstance(text_config, dict):
            text_config = Cosmos3EdgeTextConfig(**text_config)
        super().__init__(
            config=text_config,
            quant_config=quant_config,
            prefix=prefix,
        )

        self.image_token_id = getattr(config, "image_token_id", None)
        self.video_token_id = getattr(config, "video_token_id", None)

        if not self.language_model_only:
            self.visual = Siglip2Model(
                config=config.vision_config,
                quant_config=quant_config,
                prefix=add_prefix("visual", prefix),
            )
            self.projector = Cosmos3EdgeVisionProjector(
                config=config.projector_config,
                quant_config=quant_config,
                prefix=add_prefix("projector", prefix),
            )

    def pad_input_ids(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    @staticmethod
    def _as_tensor(value) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        return torch.as_tensor(value)

    @classmethod
    def _get_item_value(cls, item: MultimodalDataItem, *names: str):
        for name in names:
            try:
                value = getattr(item, name)
            except AttributeError:
                value = None
            if value is not None:
                return value
        return None

    @classmethod
    def _get_spatial_shapes(cls, item: MultimodalDataItem) -> torch.Tensor:
        spatial_shapes = cls._get_item_value(
            item, "spatial_shapes", "image_grid_hws", "grid_hws"
        )
        if spatial_shapes is not None:
            spatial_shapes = cls._as_tensor(spatial_shapes).to(dtype=torch.long)
            if spatial_shapes.ndim == 1:
                spatial_shapes = spatial_shapes.view(1, -1)
            if spatial_shapes.shape[-1] == 3:
                rows = []
                for t, height, width in spatial_shapes.view(-1, 3).tolist():
                    rows.extend([[height, width]] * int(t))
                return torch.tensor(rows, dtype=torch.long)
            if spatial_shapes.shape[-1] != 2:
                raise ValueError(
                    "Cosmos3-Edge spatial_shapes must have shape (..., 2) or (..., 3), "
                    f"got {tuple(spatial_shapes.shape)}."
                )
            return spatial_shapes.view(-1, 2)

        grid = cls._get_item_value(item, "image_grid_thw", "video_grid_thw")
        grid = cls._as_tensor(grid).to(dtype=torch.long) if grid is not None else None
        if grid is None:
            raise ValueError(
                "Cosmos3-Edge vision item is missing spatial_shapes or *_grid_thw."
            )
        if grid.ndim == 1:
            grid = grid.view(1, -1)
        if grid.shape[-1] != 3:
            raise ValueError(
                "Cosmos3-Edge grid metadata must have shape (..., 3), "
                f"got {tuple(grid.shape)}."
            )
        rows = []
        for t, height, width in grid.view(-1, 3).tolist():
            rows.extend([[height, width]] * int(t))
        return torch.tensor(rows, dtype=torch.long)

    def _pack_visual_items(
        self, items: List[MultimodalDataItem]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        packed_features = []
        all_spatial_shapes = []

        for item in items:
            pixel_values = self._as_tensor(item.feature)
            if pixel_values is None:
                raise ValueError("Cosmos3-Edge vision item is missing pixel values.")

            spatial_shapes = self._get_spatial_shapes(item).cpu()
            lengths = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).tolist()
            expected_tokens = int(sum(lengths))

            if pixel_values.ndim == 2:
                if pixel_values.shape[0] != expected_tokens:
                    raise ValueError(
                        "Cosmos3-Edge packed pixel count does not match "
                        f"spatial_shapes: {pixel_values.shape[0]} vs "
                        f"{expected_tokens}."
                    )
                packed_features.append(pixel_values)
            elif pixel_values.ndim == 3:
                if pixel_values.shape[0] != len(lengths):
                    raise ValueError(
                        "Cosmos3-Edge padded pixel batch does not match "
                        f"spatial_shapes: {pixel_values.shape[0]} vs "
                        f"{len(lengths)}."
                    )
                attention_mask = self._as_tensor(
                    self._get_item_value(item, "pixel_attention_mask", "attention_mask")
                )
                for idx, length in enumerate(lengths):
                    if attention_mask is None:
                        packed_features.append(pixel_values[idx, : int(length)])
                    else:
                        mask = attention_mask[idx].reshape(-1).bool()
                        packed_features.append(pixel_values[idx][mask])
            else:
                raise ValueError(
                    "Cosmos3-Edge pixel_values must be packed 2D or padded 3D, "
                    f"got {tuple(pixel_values.shape)}."
                )

            all_spatial_shapes.append(spatial_shapes)

        spatial_shapes_cpu = torch.cat(all_spatial_shapes, dim=0)
        pixel_values_packed = torch.cat(packed_features, dim=0).to(
            device=self.visual.device,
            dtype=self.visual.dtype,
        )
        lengths = (spatial_shapes_cpu[:, 0] * spatial_shapes_cpu[:, 1]).to(
            dtype=torch.int32, device=pixel_values_packed.device
        )
        cu_seqlens = torch.zeros(
            lengths.numel() + 1, dtype=torch.int32, device=pixel_values_packed.device
        )
        cu_seqlens[1:] = torch.cumsum(lengths, dim=0)
        max_seqlen = lengths.max()
        return pixel_values_packed, spatial_shapes_cpu, cu_seqlens, max_seqlen

    def _get_visual_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        if self.language_model_only:
            raise RuntimeError("Cosmos3-Edge was loaded with --language-model-only.")
        if not items:
            return torch.empty(0, device=self.visual.device, dtype=self.visual.dtype)

        if any(
            item.format == MultimodalInputFormat.PRECOMPUTED_EMBEDDING for item in items
        ):
            if not all(
                item.format == MultimodalInputFormat.PRECOMPUTED_EMBEDDING
                for item in items
            ):
                raise ValueError(
                    "Cosmos3-Edge cannot mix raw features and precomputed "
                    "embeddings within the same modality."
                )
            embeddings = [self._as_tensor(item.feature) for item in items]
            if any(embedding is None for embedding in embeddings):
                raise ValueError(
                    "Cosmos3-Edge precomputed embedding items must contain feature."
                )
            result = torch.cat(embeddings, dim=0)
            return result.reshape(-1, result.shape[-1])

        pixel_values_packed, spatial_shapes, cu_seqlens, max_seqlen = (
            self._pack_visual_items(items)
        )
        vision_outputs = self.visual(
            pixel_values_packed=pixel_values_packed,
            spatial_shapes=spatial_shapes,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        if vision_outputs.dim() == 3:
            vision_outputs = vision_outputs[0]
        return self.projector(vision_outputs, spatial_shapes)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        return self._get_visual_feature(items)

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        return self._get_visual_feature(items)

    def _embed_multimodal_inputs(
        self, input_ids: torch.Tensor, forward_batch: ForwardBatch
    ) -> Optional[torch.Tensor]:
        if self.pp_group.is_first_rank:
            if (
                not forward_batch.forward_mode.is_decode()
                and not forward_batch.forward_mode.is_target_verify()
                and forward_batch.contains_mm_inputs()
            ):
                mm_inputs_list = [
                    mm_input
                    for mm_input in forward_batch.mm_inputs
                    if mm_input is not None
                ]
                extend_prefix_lens = [
                    prefix_len
                    for i, prefix_len in enumerate(forward_batch.extend_prefix_lens_cpu)
                    if forward_batch.mm_inputs[i] is not None
                ]
                extend_seq_lens = [
                    seq_len
                    for i, seq_len in enumerate(forward_batch.extend_seq_lens_cpu)
                    if forward_batch.mm_inputs[i] is not None
                ]
                input_embeds, _ = embed_mm_inputs(
                    mm_inputs_list=mm_inputs_list,
                    extend_prefix_lens=extend_prefix_lens,
                    extend_seq_lens=extend_seq_lens,
                    input_ids=input_ids,
                    input_embedding=self.get_input_embeddings(),
                    multimodal_model=self,
                )

                for mm_input in mm_inputs_list:
                    if mm_input and hasattr(mm_input, "mm_items"):
                        for item in mm_input.mm_items:
                            feature = getattr(item, "feature", None)
                            if isinstance(feature, torch.Tensor) and feature.is_cuda:
                                item.feature = feature.to("cpu", non_blocking=True)
                forward_batch.mm_inputs = None
                forward_batch.mm_input_embeds = input_embeds
            else:
                input_embeds = self.get_input_embeddings()(input_ids)

            if forward_batch.input_embeds is not None:
                forward_batch.input_embeds.copy_(input_embeds)
                input_embeds = forward_batch.input_embeds
            return input_embeds
        return None

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> LogitsProcessorOutput:
        if forward_batch.mrope_positions is not None:
            positions = forward_batch.mrope_positions

        needs_mm_embedding = (
            not forward_batch.forward_mode.is_decode()
            and not forward_batch.forward_mode.is_target_verify()
            and forward_batch.contains_mm_inputs()
        )
        if (
            input_embeds is not None
            or self.language_model_only
            or not needs_mm_embedding
        ):
            return super().forward(
                input_ids=input_ids,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=input_embeds,
                get_embedding=get_embedding,
                pp_proxy_tensors=pp_proxy_tensors,
            )

        input_embeds = self._embed_multimodal_inputs(input_ids, forward_batch)
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            if not get_embedding:
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
            return self.pooler(hidden_states, forward_batch)
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        text_weights = []
        visual_weights = []
        projector_weights = []

        for name, loaded_weight in weights:
            if name.startswith("model.visual."):
                if not self.language_model_only:
                    new_name = name.replace("model.visual.", "vision_model.", 1)
                    visual_weights.append((new_name, loaded_weight))
            elif name.startswith("model.projector."):
                if not self.language_model_only:
                    new_name = name.replace("model.projector.", "projector.", 1)
                    projector_weights.append((new_name, loaded_weight))
            else:
                text_weights.append((name, loaded_weight))

        super().load_weights(self.hf_to_sglang_mapper.apply(text_weights))

        if self.language_model_only:
            return

        self.visual.load_weights(visual_weights)

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in projector_weights:
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


EntryClass = Cosmos3EdgeForConditionalGeneration
