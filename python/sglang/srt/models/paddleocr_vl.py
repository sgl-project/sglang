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

"""PaddleOCR-VL: a NaViT-style SigLIP vision encoder on an ERNIE-4.5 backbone.

The vision tower runs on a *packed* layout: every image of a (possibly
cross-request) batch is concatenated into one ``[total_patches, dim]`` tensor and
the per-image boundaries live on the host as ``grid_thws`` plus ``cu_seqlens``.
Keeping the boundaries host-side is what lets the whole ViT forward run without a
single device-to-host synchronization, and it lets the shape-independent
projections run once for the batch instead of once per image.
"""

import itertools
from collections.abc import Iterable
from typing import List, Optional, Set, Tuple

import torch
import torch.nn as nn
from transformers.activations import GELUActivation
from transformers.utils import torch_int

from sglang.srt.layers.activation import get_act_fn
from sglang.srt.layers.attention.vision import (
    VisionAttention,
    VisionAttentionMetadata,
    prepare_vision_attention_metadata,
)
from sglang.srt.layers.conv import Conv2dLayer
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
from sglang.srt.utils import add_prefix, is_npu

_is_npu = is_npu()

# Patch counts (t, h, w) of one image, always materialized on the host.
GridTHW = Tuple[int, int, int]


def build_packed_2d_position_ids(
    grid_thws: List[GridTHW], device: torch.device
) -> Tuple[torch.Tensor, int]:
    """Row/column patch indices of a packed batch, plus the rope table size.

    Returns ``([total_patches, 2], max_grid_size)``. ``max_grid_size`` is derived
    from the host grids rather than from a device-side ``max()``, so building the
    rope table never synchronizes.
    """
    split_hids = list()
    split_wids = list()
    for t, h, w in grid_thws:
        frame_ids = torch.arange(h * w, device=device)
        hids = frame_ids // w
        wids = frame_ids - hids * w
        if t > 1:
            hids = hids.repeat(t)
            wids = wids.repeat(t)
        split_hids.append(hids)
        split_wids.append(wids)

    if len(grid_thws) == 1:
        height_position_ids, width_position_ids = split_hids[0], split_wids[0]
    else:
        height_position_ids = torch.cat(split_hids, dim=0)
        width_position_ids = torch.cat(split_wids, dim=0)

    pids = torch.stack([height_position_ids, width_position_ids], dim=-1)
    max_grid_size = max(max(h, w) for _, h, w in grid_thws)
    return pids, max_grid_size


def merge_patch_neighbourhoods(
    hidden_states: torch.Tensor,
    grid_thws: List[GridTHW],
    merge_kernel_size: Tuple[int, int],
) -> torch.Tensor:
    """Group each image's ``m1 x m2`` patch neighbourhoods into single tokens.

    Takes the packed ``[total_patches, dim]`` batch and returns
    ``[total_patches / (m1 * m2), m1 * m2 * dim]``. Only this step depends on an
    image's ``h``/``w``, which is why it is separated from the projections that
    follow: those are row-wise and run once for the whole batch.
    """
    m1, m2 = merge_kernel_size
    dim = hidden_states.shape[-1]
    merged = hidden_states.new_empty(hidden_states.shape[0] // (m1 * m2), m1 * m2 * dim)

    in_offset = out_offset = 0
    for t, h, w in grid_thws:
        num_patches = t * h * w
        num_merged = num_patches // (m1 * m2)
        # Row-major patches (t, h, w) regroup as (t, h/m1, m1, w/m2, m2, d); the
        # merged token concatenates the m1*m2 neighbours along `d`.
        merged[out_offset : out_offset + num_merged].view(
            t, h // m1, w // m2, m1, m2, dim
        ).copy_(
            hidden_states[in_offset : in_offset + num_patches]
            .view(t, h // m1, m1, w // m2, m2, dim)
            .permute(0, 1, 3, 2, 4, 5)
        )
        in_offset += num_patches
        out_offset += num_merged
    return merged


class Projector(nn.Module):
    """Merge 2x2 patch neighbourhoods, then project into the language space."""

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
        grid_thws: List[GridTHW],
    ) -> torch.Tensor:
        """Project packed ViT features ``[total_patches, dim]`` for the batch.

        Only the 2x2 merge depends on an image's ``h``/``w``; the norm and both
        projections are row-wise, so they run once over the packed batch. Each
        image contributes a single strided copy into the merged buffer, so an
        N-image batch costs N copies plus 3 kernels rather than 4N kernels.
        """
        hidden_states = self.pre_norm(image_features)
        merged = merge_patch_neighbourhoods(
            hidden_states, grid_thws, self.merge_kernel_size
        )
        hidden_states = self.linear_1(merged)
        hidden_states = self.act(hidden_states)
        return self.linear_2(hidden_states)


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        # kernel_size == stride and padding == 0, so this convolution is exactly
        # an unfold plus a matmul. Taking that path avoids a cuDNN convolution
        # launch over a [total_patches, 3, p, p] input on every ViT forward.
        self.patch_embedding = Conv2dLayer(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
            disable_linear=False,
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

    def interpolate_pos_encoding(self, height: int, width: int) -> torch.Tensor:
        """Resample the square learned position grid onto a ``height x width`` grid."""
        num_positions = self.position_embedding.weight.shape[0]
        patch_pos_embed = self.position_embedding.weight.unsqueeze(0)

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(
            1, sqrt_num_positions, sqrt_num_positions, self.embed_dim
        )
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(
            1, -1, self.embed_dim
        )
        # Materialize contiguously. The permute leaves the channel dim strided,
        # and this tensor is cached and broadcast-added to the packed activations
        # on every forward, so a strided read would be paid over and over.
        return patch_pos_embed.contiguous()

    def fetch_position_embedding_lfu_cache(
        self, h: int, w: int, max_cache: int = 20
    ) -> torch.Tensor:
        """Return the interpolated position grid for ``(h, w)``, LFU-cached.

        The interpolation depends only on the grid, so document batches that
        repeat a resolution reuse the tensor instead of re-running the bilinear
        resample once per image per forward. The cache holds at most `max_cache`
        grids of `h * w * hidden_size` each (~12 MiB at the checkpoint's default
        1280-token page budget), evicting the least frequently used.
        """
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

        position_embedding = self.interpolate_pos_encoding(h, w)
        self.cache_position_count[grid] = 1
        self.cache_position_embedding[grid] = position_embedding
        return position_embedding

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        grid_thws: List[GridTHW],
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pixel_values.dim() == 5:
            # [batch, patches, c, ph, pw] -> [batch * patches, c, ph, pw]
            pixel_values = pixel_values.flatten(0, 1)
        if pixel_values.dim() != 4:
            raise ValueError(
                "Unsupported pixel_values dimension:"
                f" {pixel_values.dim()}. Expected 4 or 5."
            )

        patch_embeds = self.patch_embedding(
            pixel_values.to(dtype=self.patch_embedding.weight.dtype)
        )
        # Each patch convolves to a 1x1 map, so this is a reshape to [patches, dim].
        embeddings = patch_embeds.flatten(-2).squeeze(-1)

        if position_ids is None:
            # Interpolated per-image position grids, added in place so the packed
            # activation is never copied into a second buffer.
            offset = 0
            for t, h, w in grid_thws:
                num_patches = t * h * w
                embeddings[offset : offset + num_patches].view(
                    t, h * w, self.embed_dim
                ).add_(self.fetch_position_embedding_lfu_cache(h, w))
                offset += num_patches
        else:
            embeddings += self.packing_position_embedding(position_ids)

        return embeddings.unsqueeze(0)


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
        if quant_config and quant_config.get_name() == "bitsandbytes":
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
        cu_seqlens: torch.Tensor,
        rope_emb: Tuple[torch.Tensor, torch.Tensor],
        forward_metadata: VisionAttentionMetadata,
    ) -> torch.Tensor:

        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)

        hidden_states = self.self_attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=rope_emb,
            forward_metadata=forward_metadata,
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

    def _build_rope_emb(
        self, grid_thws: List[GridTHW], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the packed 2D rope cos/sin table for the batch."""
        pids, max_grid_size = build_packed_2d_position_ids(grid_thws, device)
        rope_emb = self.rotary_pos_emb(max_grid_size)[pids].flatten(1)
        rope_emb = rope_emb.repeat(1, 2)
        return rope_emb.cos(), rope_emb.sin()

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        cu_seqlens: torch.Tensor,
        grid_thws: List[GridTHW],
        max_seqlen: int,
    ) -> torch.Tensor:
        rope_emb = self._build_rope_emb(grid_thws, inputs_embeds.device)

        # cu_seqlens must be on cpu because of npu_flash_attention_unpad operator restriction
        if _is_npu:
            cu_seqlens = cu_seqlens.to("cpu")
        # `max_seqlen` comes from the host grids, so the metadata is built once
        # for every layer without reading a device tensor back.
        forward_metadata = prepare_vision_attention_metadata(
            cu_seqlens, device=inputs_embeds.device, max_seqlen=max_seqlen
        )

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rope_emb=rope_emb,
                forward_metadata=forward_metadata,
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
        pixel_values: torch.Tensor,
        grid_thws: List[GridTHW],
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(
            pixel_values,
            grid_thws=grid_thws,
            position_ids=position_ids,
        )

        hidden_states = self.encoder(
            inputs_embeds=hidden_states,
            cu_seqlens=cu_seqlens,
            grid_thws=grid_thws,
            max_seqlen=max_seqlen,
        )

        # Stay packed: the projector slices per image on the host, so splitting
        # here would index `cu_seqlens` on the device and stall once per image.
        return self.post_layernorm(hidden_states).squeeze(0)


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
        pixel_values: torch.Tensor,
        grid_thws: List[GridTHW],
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.vision_model(
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_ids=position_ids,
        )


class PaddleOCRVLForConditionalGeneration(Ernie4_5_ForCausalLM):
    def __init__(self, *, config, quant_config=None, prefix: str = ""):
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        config = self.config

        self.mlp_AR = Projector(
            config, config.vision_config, prefix=add_prefix("mlp_AR", prefix)
        )
        # NOTE: only BitsAndBytes 4-bit quantization is exercised for the SigLIP
        # tower; other methods fall back to bf16 through SiglipMLP's own gate.
        self.visual = SiglipVisionModel(
            config=config.vision_config,
            quant_config=quant_config,
            prefix=add_prefix("visual", prefix),
        )
        self.is_mrope_enabled = "mrope_section" in (self.config.rope_scaling or {})

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def encode_image(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> torch.Tensor:
        # One host transfer for the whole batch. Every consumer of the grids
        # (rope table, patch merge, cu_seqlens) needs them on the host, so
        # reading them per image would cost one synchronization per image.
        grid_thws: List[GridTHW] = [(t, h, w) for t, h, w in image_grid_thw.tolist()]
        seq_lens = [t * h * w for t, h, w in grid_thws]
        cu_seqlens = torch.tensor(
            [0, *itertools.accumulate(seq_lens)],
            dtype=torch.int32,
            device=pixel_values.device,
        )

        vision_outputs = self.visual(
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            cu_seqlens=cu_seqlens,
            max_seqlen=max(seq_lens),
        )
        return self.mlp_AR(vision_outputs, grid_thws)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        if len(items) == 1:
            # torch.cat allocates even for a single input; a document batch is
            # usually one image, and its pixel buffer is the largest tensor here.
            pixel_values = items[0].feature
            image_grid_thw = items[0].image_grid_thw
        else:
            pixel_values = torch.cat([item.feature for item in items], dim=0)
            image_grid_thw = torch.cat([item.image_grid_thw for item in items], dim=0)
        return self.encode_image(pixel_values, image_grid_thw)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
    ):
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions
            if (
                not forward_batch.forward_mode.is_decode()
                and forward_batch.contains_image_inputs()
            ):
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


EntryClass = [PaddleOCRVLForConditionalGeneration]
