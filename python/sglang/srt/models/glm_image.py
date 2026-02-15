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

"""Inference-only GlmImage model compatible with HuggingFace weights."""

import copy
import logging
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from sglang.srt.distributed.parallel_state import get_pp_group
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.glm4 import Glm4Model
from sglang.srt.models.utils import compute_cu_seqlens_from_grid_numpy
from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, is_npu

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Vision encoder components
# --------------------------------------------------------------------------- #


class GlmImageVisionMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()
        self.tp_size = 1 if use_data_parallel else get_attention_tp_size()
        self.tp_rank = 0 if use_data_parallel else get_attention_tp_rank()
        self.fc1 = ColumnParallelLinear(
            in_features,
            hidden_features,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("fc1", prefix),
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
        )
        self.fc2 = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("fc2", prefix),
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            use_dp_attention_reduce=is_dp_attention_enabled(),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc1(x)
        x = self.act(x)
        x, _ = self.fc2(x)
        return x


class GlmImageVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel_size = [self.patch_size, self.patch_size]
        self.proj = nn.Conv2d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(
            -1, self.embed_dim
        )
        return hidden_states


class GlmImageVisionEmbeddings(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.interpolated_method = "bilinear"

    def forward(
        self,
        embeddings: torch.Tensor,
        lengths,
        image_shapes: torch.Tensor,
        h_coords: torch.Tensor,
        w_coords: torch.Tensor,
    ) -> torch.Tensor:
        pos_embed_weight = self.position_embedding.weight
        hidden_size = pos_embed_weight.shape[1]
        device = pos_embed_weight.device

        if isinstance(lengths, list):
            lengths = torch.tensor(lengths, device=device, dtype=torch.long)

        orig_size_sq = pos_embed_weight.shape[0]
        orig_size = int(orig_size_sq**0.5)
        pos_embed_2d = (
            pos_embed_weight.view(orig_size, orig_size, hidden_size)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
        )

        target_h = torch.cat(
            [image_shapes[i, 1].repeat(lengths[i]) for i in range(len(lengths))]
        ).to(device=device, dtype=torch.float32)
        target_w = torch.cat(
            [image_shapes[i, 2].repeat(lengths[i]) for i in range(len(lengths))]
        ).to(device=device, dtype=torch.float32)

        h_coords = h_coords.to(device=device, dtype=torch.float32)
        w_coords = w_coords.to(device=device, dtype=torch.float32)
        norm_w = ((w_coords + 0.5) / target_w) * 2 - 1
        norm_h = ((h_coords + 0.5) / target_h) * 2 - 1

        grid = torch.stack((norm_w, norm_h), dim=-1).unsqueeze(0).unsqueeze(2)

        interpolated_embed_fp32 = F.grid_sample(
            pos_embed_2d,
            grid,
            mode=self.interpolated_method,
            align_corners=False,
            padding_mode="border",
        )

        adapted_pos_embed_fp32 = (
            interpolated_embed_fp32.squeeze(0).squeeze(-1).permute(1, 0)
        )
        adapted_pos_embed = adapted_pos_embed_fp32.to(pos_embed_weight.dtype).to(
            embeddings.device
        )

        embeddings = embeddings + adapted_pos_embed
        return embeddings


class GlmImageVisionBlock(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.attn = VisionAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            projection_size=config.hidden_size,
            use_qkv_parallel=True,
            proj_bias=config.attention_bias,
            qkv_bias=config.attention_bias,
            flatten_batch=True,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            use_data_parallel=use_data_parallel,
            use_dp_attention_reduce=is_dp_attention_enabled(),
        )
        self.mlp = GlmImageVisionMLP(
            in_features=config.hidden_size,
            hidden_features=config.intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
            use_data_parallel=use_data_parallel,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        # x shape: (S, B, H) where B=1
        hidden_states = self.norm1(x)
        hidden_states = rearrange(hidden_states, "s b ... -> b s ...")
        attn = self.attn(hidden_states, cu_seqlens=cu_seqlens)
        attn = rearrange(attn, "b s ... -> s b ...")
        x = x + attn

        hidden_states = self.norm2(x)
        mlp = self.mlp(hidden_states)
        x = x + mlp
        return x


class GlmImageVisionModel(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.spatial_merge_size = getattr(config, "spatial_merge_size", 1)
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        # No patch merger in GlmImage, output dim = hidden_size
        self.out_hidden_size = config.hidden_size

        self.embeddings = GlmImageVisionEmbeddings(config)
        self.patch_embed = GlmImageVisionPatchEmbed(config)

        self.blocks = nn.ModuleList(
            [
                GlmImageVisionBlock(
                    config,
                    quant_config=quant_config,
                    prefix=add_prefix(f"blocks.{i}", prefix),
                    use_data_parallel=use_data_parallel,
                )
                for i in range(config.depth)
            ]
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def rot_pos_emb(self, grid_thw):
        """Compute position coordinate IDs for position embedding interpolation."""
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        return pos_ids

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        hidden_states = self.patch_embed(pixel_values)

        if isinstance(grid_thw, list):
            grid_thw_list = grid_thw
            grid_thw = torch.tensor(grid_thw, dtype=torch.int32)
        else:
            grid_thw_list = grid_thw.tolist()

        image_type_ids = self.rot_pos_emb(grid_thw_list)

        # Compute cu_seqlens using numpy for efficiency
        grid_thw_cpu = grid_thw if grid_thw.device.type == "cpu" else grid_thw.cpu()
        cu_seqlens = compute_cu_seqlens_from_grid_numpy(grid_thw_cpu)
        if not is_npu():
            cu_seqlens = cu_seqlens.to(self.device, non_blocking=True)
        else:
            cu_seqlens = cu_seqlens.to("cpu")

        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

        hidden_states = self.embeddings(
            hidden_states,
            seqlens,
            grid_thw,
            image_type_ids[:, 0].to(hidden_states.device),
            image_type_ids[:, 1].to(hidden_states.device),
        )

        # (S, H) -> (S, 1, H) for block processing
        hidden_states = hidden_states.unsqueeze(1)

        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens)

        # (S, 1, H) -> (S, H)
        return hidden_states.squeeze(1)


# --------------------------------------------------------------------------- #
# VQ-VAE
# --------------------------------------------------------------------------- #


class GlmImageVQVAE(nn.Module):
    """VQ-VAE module for encoding vision features into discrete tokens.

    Follows the HF transformers GlmImageVQVAE architecture:
    quant_conv (Conv2d) -> L2 normalize -> nearest codebook lookup -> indices
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.num_embeddings = config.num_embeddings
        self.embedding_dim = config.embed_dim
        self.latent_channels = config.latent_channels

        # Codebook (quantize.embedding in HF)
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # Convolutions
        self.quant_conv = nn.Conv2d(self.latent_channels, self.embedding_dim, 1)
        self.post_quant_conv = nn.Conv2d(self.embedding_dim, self.latent_channels, 1)

        self.eval()  # frozen

    def encode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Encode spatial features to discrete codebook indices.

        Args:
            hidden_states: [B, latent_channels, H, W] spatial feature maps
        Returns:
            indices: [B*H*W] discrete codebook indices
        """
        conv_hidden = self.quant_conv(hidden_states)
        # Permute to [B, H, W, embed_dim] then flatten for distance computation
        z = conv_hidden.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, self.embedding_dim)

        # L2 normalize
        z_flat = F.normalize(z_flat, p=2, dim=-1)
        codebook = F.normalize(self.embedding.weight, p=2, dim=-1)

        # Compute distances: (z - e)^2 = z^2 + e^2 - 2*z*e
        distances = (
            torch.sum(z_flat**2, dim=1, keepdim=True)
            + torch.sum(codebook**2, dim=1)
            - 2 * torch.matmul(z_flat, codebook.t())
        )
        indices = torch.argmin(distances, dim=1)
        return indices


# --------------------------------------------------------------------------- #
# Main model
# --------------------------------------------------------------------------- #


class GlmImageForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.vision_config = config.vision_config
        self.vq_config = config.vq_config
        self.text_config = config.text_config
        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder

        # Bridge rope_parameters -> rope_scaling so Glm4Model can pick it up
        if hasattr(self.text_config, "rope_parameters") and not getattr(
            self.text_config, "rope_scaling", None
        ):
            self.text_config.rope_scaling = self.text_config.rope_parameters

        # Vision encoder
        self.visual = GlmImageVisionModel(
            self.vision_config,
            quant_config=quant_config,
            prefix=add_prefix("visual", prefix),
            use_data_parallel=self.use_data_parallel,
        )

        # VQ-VAE (small frozen module, no TP needed)
        self.vqvae = GlmImageVQVAE(self.vq_config)

        # Language model (reuse Glm4Model)
        self.model = Glm4Model(
            self.text_config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        # GLM-Image uses split-half (neox-style) rotary embedding, unlike GLM4
        # which uses interleaved rotation.  Glm4Attention defaults to
        # is_neox_style=False, so we override it here after construction.
        for layer in self.model.layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "rotary_emb"):
                layer.self_attn.rotary_emb.is_neox_style = True

        # LogitsProcessor with vision_vocab_size
        vision_vocab_size = getattr(self.text_config, "vision_vocab_size", None)
        if vision_vocab_size is not None:
            logits_config = copy.copy(self.text_config)
            logits_config.vocab_size = vision_vocab_size
        else:
            logits_config = self.text_config

        # lm_head: maps hidden_size -> vision_vocab_size
        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                logits_config.vocab_size,
                self.text_config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.is_mrope_enabled = (
            hasattr(self.text_config, "rope_scaling")
            and self.text_config.rope_scaling is not None
            and "mrope_section" in self.text_config.rope_scaling
        )

        self.logits_processor = LogitsProcessor(logits_config)

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Run vision encoder -> VQ-VAE encode -> embed_tokens on discrete indices."""
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()

        # Vision encoder forward (with optional DP sharding)
        if self.use_data_parallel:
            vision_hidden = run_dp_sharded_mrope_vision_model(
                self.visual,
                pixel_values,
                image_grid_thw.tolist(),
                rope_type="rope_3d",
            )
        else:
            vision_hidden = self.visual(pixel_values, grid_thw=image_grid_thw)

        # Split by image, reshape to spatial, run VQ-VAE encode, then embed
        hidden_size = vision_hidden.shape[-1]
        split_sizes = (image_grid_thw.prod(dim=-1)).tolist()
        hidden_list = torch.split(vision_hidden, split_sizes, dim=0)

        embed_tokens = self.model.get_input_embeddings()
        all_embeds = []
        for idx, hs in enumerate(hidden_list):
            grid_t, grid_h, grid_w = image_grid_thw[idx].tolist()
            grid_t, grid_h, grid_w = int(grid_t), int(grid_h), int(grid_w)
            # Reshape to spatial: [t, h, w, hidden] -> [t, hidden, h, w]
            hs = hs.view(grid_t, grid_h, grid_w, hidden_size)
            hs = hs.permute(0, 3, 1, 2).contiguous()
            # VQ-VAE encode: get discrete codebook indices
            indices = self.vqvae.encode(hs)
            # Embed via LLM embedding table
            embeds = embed_tokens(indices)
            all_embeds.append(embeds)

        return torch.cat(all_embeds, dim=0)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def _get_decode_mrope_positions(self, forward_batch: ForwardBatch) -> torch.Tensor:
        """Look up pre-computed 2D spatial MRoPE positions during decode.

        Instead of using the default delta-based positions (which are sequential
        and identical across all 3 dims), this looks up the pre-computed 2D
        spatial positions stored in each request's MultimodalInputs.mrope_positions.
        This gives each generated image token the correct (temporal, height, width)
        coordinates based on its position in the target image grid.
        """
        batch_size = forward_batch.batch_size
        positions_list = []
        seq_lens = forward_batch.seq_lens_cpu

        for i in range(batch_size):
            mm_input = forward_batch.mm_inputs[i] if forward_batch.mm_inputs else None
            seq_len = int(seq_lens[i])

            if mm_input is not None and mm_input.mrope_positions is not None:
                stored = mm_input.mrope_positions
                idx = seq_len - 1
                if idx < stored.shape[1]:
                    # Look up the pre-computed 2D spatial position
                    pos = stored[:, idx : idx + 1]
                else:
                    # Beyond stored positions: fall back to sequential from last
                    last_max = stored[:, -1:].max().item()
                    overflow = idx - stored.shape[1] + 1
                    pos = torch.full((3, 1), last_max + overflow, dtype=torch.int64)
            else:
                pos = torch.full((3, 1), seq_len - 1, dtype=torch.int64)
            positions_list.append(pos)

        return torch.cat(positions_list, dim=1).to(
            device=forward_batch.input_ids.device, dtype=torch.int64
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        if self.is_mrope_enabled:
            if forward_batch.forward_mode.is_decode():
                # Use pre-computed 2D spatial positions for image generation
                positions = self._get_decode_mrope_positions(forward_batch)
            else:
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
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                forward_batch,
            )
        else:
            return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".up_proj", 1),
            (".gate_up_proj", ".gate_proj", 0),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # Weight name mapping from HF checkpoint
            if "language_model" in name:
                name = name.replace("model.language_model.", "model.")
            if "model.visual." in name:
                name = name.replace("model.visual.", "visual.")
            if "model.vqmodel." in name:
                name = name.replace("model.vqmodel.", "vqvae.")
            if "vqvae.quantize.embedding" in name:
                name = name.replace("vqvae.quantize.embedding", "vqvae.embedding")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Vision uses fused QKV, skip stacked mapping
                if "visual" in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if "visual" in name:
                    # Map fused attn.qkv -> attn.qkv_proj for QKVParallelLinear
                    name = name.replace("attn.qkv.", "attn.qkv_proj.")

                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = [GlmImageForConditionalGeneration]
