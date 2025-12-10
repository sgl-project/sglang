# Reference: ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from collections.abc import Iterable
from typing import List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from transformers.activations import GELUActivation
from transformers.utils import torch_int

from sglang.srt.layers.activation import get_act_fn
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.ernie4 import Ernie4_5_ForCausalLM
from sglang.srt.utils import add_prefix


class Projector(nn.Module):

    def __init__(
        self,
        text_config,
        vision_config,
        prefix: str = "",
    ):
        super().__init__()
        self.text_config = text_config
        self.vision_config = vision_config
        self.merge_kernel_size = (2, 2)

        self.hidden_size = (
            self.vision_config.hidden_size
            * self.merge_kernel_size[0]
            * self.merge_kernel_size[1]
        )

        self.pre_norm = torch.nn.LayerNorm(self.vision_config.hidden_size, eps=1e-05)
        self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.act = GELUActivation()
        self.linear_2 = nn.Linear(
            self.hidden_size, self.text_config.hidden_size, bias=True
        )

    def forward(
        self,
        image_features: torch.Tensor,
        image_grid_thw: List[Tuple[int, int, int]],
    ) -> torch.Tensor:
        m1, m2 = self.merge_kernel_size
        if isinstance(image_features, (list, tuple)):
            processed_features = list()
            for image_feature, image_grid in zip(image_features, image_grid_thw):
                image_feature = self.pre_norm(image_feature)
                t, h, w = image_grid

                image_feature = rearrange(
                    image_feature,
                    "(t h p1 w p2) d -> (t h w) (p1 p2 d)",
                    t=t,
                    h=h // m1,
                    p1=m1,
                    w=w // m2,
                    p2=m2,
                )
                hidden_states = self.linear_1(image_feature)
                hidden_states = self.act(hidden_states)
                hidden_states = self.linear_2(hidden_states)
                processed_features.append(hidden_states)

            return processed_features

        dims = image_features.shape[:-1]
        dim = image_features.shape[-1]
        image_features = image_features.view(np.prod(dims), dim)
        hidden_states = self.pre_norm(image_features).view(-1, self.hidden_size)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)

        return hidden_states.view(*dims, -1)


class SiglipVisionEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.cache_position_embedding = dict()
        self.cache_position_count = dict()
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.packing_position_embedding = nn.Embedding(32768, self.embed_dim)

        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def interpolate_pos_encoding(
        self,
        embeddings: torch.Tensor,
        height: int,
        width: int,
        is_after_patchify: bool = False,
    ) -> torch.Tensor:

        num_positions = self.position_embedding.weight.shape[0]

        patch_pos_embed = self.position_embedding.weight.unsqueeze(0)

        dim = embeddings.shape[-1]

        if is_after_patchify:
            new_height = height
            new_width = width
        else:
            new_height = height // self.patch_size
            new_width = width // self.patch_size

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(
            1, sqrt_num_positions, sqrt_num_positions, dim
        )
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def fetch_position_embedding_lfu_cache(self, embeddings, h, w, max_cache: int = 20):
        grid = (h, w)
        if grid in self.cache_position_embedding:
            self.cache_position_count[grid] += 1
            return self.cache_position_embedding[grid]

        if len(self.cache_position_embedding) >= max_cache:
            min_hit_grid = min(
                self.cache_position_count,
                key=self.cache_position_count.get,
            )
            self.cache_position_count.pop(min_hit_grid)
            self.cache_position_embedding.pop(min_hit_grid)

        position_embedding = self.interpolate_pos_encoding(embeddings, h, w, True)
        self.cache_position_count[grid] = 1
        self.cache_position_embedding[grid] = position_embedding
        return position_embedding

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        position_ids: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[
            List[
                Union[
                    Tuple[int, int, int],
                    List[Tuple[int, int, int]],
                ]
            ]
        ] = None,
        interpolate_pos_encoding=False,
    ) -> torch.Tensor:
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(0)
        if pixel_values.dim() == 5:
            if position_ids is None:
                raise ValueError(
                    "position_ids cannot be None when pixel_values.dim() is 5."
                )
            (
                batch_size,
                squence_len,
                channel,
                height,
                width,
            ) = pixel_values.shape
            target_dtype = self.patch_embedding.weight.dtype
            pixel_values = rearrange(pixel_values, "b l c h w -> (b l) c h w")
            patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
            embeddings = patch_embeds.flatten(-2).squeeze(-1)

            if interpolate_pos_encoding and image_grid_thw is not None:
                start = 0
                tmp_embeddings = list()
                for image_grid in image_grid_thw:
                    t, h, w = image_grid
                    end = start + t * h * w
                    image_embeddings = embeddings[start:end, :]
                    position_embedding = (
                        self.interpolate_pos_encoding(image_embeddings, h, w, True)
                        .squeeze(0)
                        .repeat(t, 1)
                    )
                    image_embeddings = image_embeddings + position_embedding
                    tmp_embeddings.append(image_embeddings)
                    start = end
                embeddings = torch.concat(tmp_embeddings, dim=0).unsqueeze(0)
            else:
                embeddings = embeddings + self.packing_position_embedding(position_ids)
            return embeddings
        else:
            raise ValueError(
                "Unsupported pixel_values dimension:"
                f" {pixel_values.dim()}. Expected 4 or 5."
            )


class SigLIPRotaryEmbedding(nn.Module):

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.rope_init()

    def rope_init(self):
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(
            seqlen,
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype,
        )
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class SiglipMLP(nn.Module):

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)
        if quant_config and quant_config.get_name() in ["bitsandbytes", "torchao"]:
            quantizable = True
        else:
            quantizable = (
                config.hidden_size % 64 == 0 and config.intermediate_size % 64 == 0
            )
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            quant_config=quant_config if quantizable else None,
            prefix=add_prefix("fc1", prefix),
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            quant_config=quant_config if quantizable else None,
            prefix=add_prefix("fc2", prefix),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Module):

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.self_attn = VisionAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            projection_size=self.embed_dim,
            use_qkv_parallel=True,
            qkv_bias=True,
            flatten_batch=True,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(
            config, quant_config=quant_config, prefix=add_prefix("mlp", prefix)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[List[torch.Tensor]] = None,
        rope_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor]:

        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)

        hidden_states = self.self_attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=rope_emb,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states


class SiglipEncoder(nn.Module):

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.layers = nn.ModuleList(
            [
                SiglipEncoderLayer(
                    config,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_idx}", prefix),
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.rotary_pos_emb = SigLIPRotaryEmbedding(head_dim // 2)

    @staticmethod
    def flatten_list(image_grid_thw):
        tmp_image_grid_thw = list()
        for image_grid in image_grid_thw:
            if isinstance(image_grid, list):
                tmp_image_grid_thw.extend(image_grid)
            else:
                tmp_image_grid_thw.append(image_grid)
        return tmp_image_grid_thw

    def forward(
        self,
        inputs_embeds,
        cu_seqlens: Optional[List[torch.Tensor]] = None,
        image_grid_thw: Optional[
            List[
                Union[
                    Tuple[int, int, int],
                    List[Tuple[int, int, int]],
                ]
            ]
        ] = None,
        height_position_ids: Optional[torch.Tensor] = None,
        width_position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = inputs_embeds.device
        hidden_states = inputs_embeds
        flatten_image_grid_thw = self.flatten_list(image_grid_thw)

        if width_position_ids is None or height_position_ids is None:
            split_hids = list()
            split_wids = list()
            for t, h, w in flatten_image_grid_thw:
                image_pids = torch.arange(t * h * w, device=device) % (h * w)
                sample_hids = image_pids // w
                sample_wids = image_pids % w
                split_hids.append(sample_hids)
                split_wids.append(sample_wids)
            width_position_ids = torch.concat(split_wids, dim=0)
            height_position_ids = torch.concat(split_hids, dim=0)

        pids = torch.stack(
            [height_position_ids, width_position_ids],
            dim=-1,
        )
        max_grid_size = pids.max() + 1
        rope_emb_max_grid = self.rotary_pos_emb(max_grid_size)
        rope_emb = rope_emb_max_grid[pids].flatten(1)
        rope_emb = rope_emb.repeat(1, 2)
        rope_emb = (rope_emb.cos(), rope_emb.sin())

        attn_cu_seqlens = cu_seqlens
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                cu_seqlens=attn_cu_seqlens,
                rope_emb=rope_emb,
            )
        return hidden_states


class SiglipVisionTransformer(nn.Module):

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(
            config,
            quant_config=quant_config,
            prefix=add_prefix("encoder", prefix),
        )
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        pixel_values,
        interpolate_pos_encoding: Optional[bool] = False,
        position_ids: Optional[torch.Tensor] = None,
        height_position_ids: Optional[torch.Tensor] = None,
        width_position_ids: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[List[torch.Tensor]] = None,
        image_grid_thw: Optional[
            List[
                Union[
                    Tuple[int, int, int],
                    List[Tuple[int, int, int]],
                ]
            ]
        ] = None,
    ) -> list[torch.Tensor]:

        hidden_states = self.embeddings(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
        )

        last_hidden_state = self.encoder(
            inputs_embeds=hidden_states,
            cu_seqlens=cu_seqlens,
            image_grid_thw=image_grid_thw,
            height_position_ids=height_position_ids,
            width_position_ids=width_position_ids,
        )

        last_hidden_state = self.post_layernorm(last_hidden_state)

        sample_hidden_state = list()
        if cu_seqlens is None:
            raise ValueError(
                "cu_seqlens cannot be None for "
                "SiglipVisionTransformer output processing."
            )
        for i in range(cu_seqlens.shape[0] - 1):
            start = cu_seqlens[i]
            end = cu_seqlens[i + 1]
            tensor = last_hidden_state[:, start:end, :].squeeze(0)
            sample_hidden_state.append(tensor)

        return sample_hidden_state


class SiglipVisionModel(nn.Module):
    config_class = "PaddleOCRVisionConfig"
    main_input_name = "pixel_values"

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        self.vision_model = SiglipVisionTransformer(
            config,
            quant_config=quant_config,
            prefix=add_prefix("vision_model", prefix),
        )
        self.quant_config = quant_config

    @property
    def dtype(self) -> torch.dtype:
        return self.vision_model.embeddings.patch_embedding.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.vision_model.embeddings.patch_embedding.weight.device

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values,
        interpolate_pos_encoding: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[
            List[
                Union[
                    Tuple[int, int, int],
                    List[Tuple[int, int, int]],
                ]
            ]
        ] = None,
        cu_seqlens: Optional[List[torch.Tensor]] = None,
    ) -> list[torch.Tensor]:

        return self.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
            cu_seqlens=cu_seqlens,
        )


class PaddleOCRVLForConditionalGeneration(Ernie4_5_ForCausalLM):

    def __init__(self, *, config, quant_config=None, prefix: str = ""):
        super().__init__(config=config, prefix=prefix)
        config = self.config

        self.mlp_AR = Projector(
            config, config.vision_config, prefix=add_prefix("mlp_AR", prefix)
        )
        self.visual = SiglipVisionModel(
            config=config.vision_config, prefix=add_prefix("visual", prefix)
        )
        if not hasattr(self.model, "get_input_embeddings"):
            import types

            self.model.get_input_embeddings = types.MethodType(
                get_input_embeddings, self.model
            )
        self.is_mrope_enabled = "mrope_section" in self.config.rope_scaling

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def encode_image(self, pixel_values, image_grid_thw):
        pixel_values = pixel_values.type(self.visual.dtype)
        siglip_position_ids = list()
        image_grid_hws = list()
        cu_seqlens = [0]

        for idx, grid_thw in enumerate(image_grid_thw):
            thw_tuple = tuple(grid_thw.detach().cpu().numpy().tolist())
            numel = np.prod(thw_tuple)
            image_grid_hws.append(thw_tuple)
            image_position_ids = torch.arange(numel) % np.prod(thw_tuple[1:])
            siglip_position_ids.append(image_position_ids)
            cu_seqlens.append(cu_seqlens[-1] + numel)

        siglip_position_ids = torch.concat(siglip_position_ids, dim=0).to(
            pixel_values.device
        )
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32).to(pixel_values.device)
        vision_outputs = self.visual(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_hws,
            position_ids=siglip_position_ids,
            interpolate_pos_encoding=True,
            cu_seqlens=cu_seqlens,
        )
        image_embeds = self.mlp_AR(vision_outputs, image_grid_thw)

        # image_embeds = torch.stack(image_embeds, dim=0)
        image_embeds = torch.cat(image_embeds, dim=0)

        return image_embeds

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)
        image_embeds = self.encode_image(pixel_values, image_grid_thw)

        return image_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
    ):
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions
        if not (
            forward_batch.forward_mode.is_decode()
            or not forward_batch.contains_image_inputs()
        ):
            if self.is_mrope_enabled:
                assert positions.ndim == 2 and positions.size(0) == 3, (
                    "multimodal section rotary embedding requires "
                    f"(3, seq_len) positions, but got {positions.size()}"
                )

        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
        )

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, weight_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "head.attention" in name or "head.layernorm" in name:
                continue
            if "head.mlp" in name or "head.probe" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if "vision_model" in name and "out_proj" in name:
                    # adapt to VisionAttention
                    name = name.replace(".self_attn.out_proj", ".self_attn.proj")
                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    raise KeyError(f"Parameter '{name}' not found in model.")


# monkey patch
def get_input_embeddings(self) -> nn.Embedding:
    return self.embed_tokens


EntryClass = [PaddleOCRVLForConditionalGeneration]
