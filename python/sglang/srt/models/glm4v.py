import logging
from functools import lru_cache, partial
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.glm4v.configuration_glm4v import Glm4vConfig, Glm4vVisionConfig

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.attention import vision_utils
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.managers.schedule_batch import MultimodalDataItem
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.glm4 import Glm4Model
from sglang.srt.models.qwen2_5_vl import (
    Qwen2_5_VisionBlock,
    Qwen2_5_VLForConditionalGeneration,
)
from sglang.srt.utils import add_prefix
from sglang.srt.utils.hf_transformers_utils import get_processor

logger = logging.getLogger(__name__)

cached_get_processor = lru_cache(get_processor)


class Glm4vRMSNorm(RMSNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_2d = x.contiguous().reshape(-1, original_shape[-1])
        x_2d = super().forward(x_2d)
        x = x_2d.reshape(original_shape)
        return x


class Glm4vVisionMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=in_features,
            output_sizes=[hidden_features] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Glm4vVisionBlock(Qwen2_5_VisionBlock):
    def __init__(
        self,
        config: Glm4vVisionConfig,
        norm_layer: Optional[nn.Module] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            dim=config.hidden_size,
            intermediate_dim=config.out_hidden_size,
            num_heads=config.num_heads,
            hidden_act=config.hidden_act,
            norm_layer=norm_layer,
            quant_config=quant_config,
            prefix=prefix,
            num_dummy_heads=config.num_dummy_heads,
            rms_norm_eps=config.rms_norm_eps,
        )

        self.mlp = Glm4vVisionMLP(
            config.hidden_size,
            config.out_hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )


class Glm4vVisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1536,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size
        self.in_channels = in_channels

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        x = self.proj(x).view(-1, self.hidden_size)
        return x


class Glm4vPatchMerger(nn.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = d_model
        self.proj = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("proj", prefix),
            gather_output=True,
        )
        self.post_projection_norm = nn.LayerNorm(self.hidden_size)
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[context_dim] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            context_dim,
            self.hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.extra_activation_func = nn.GELU()

    def forward(self, x: torch.Tensor):
        x, _ = self.proj(x)
        x = self.extra_activation_func(self.post_projection_norm(x))
        gate_up, _ = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        x = F.silu(gate) * up
        x, _ = self.down_proj(x)
        return x


class Glm4vVisionEmbeddings(nn.Module):
    def __init__(self, config: Glm4vVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(
        self, embeddings, lengths, image_shapes, h_coords, w_coords
    ) -> torch.Tensor:
        pos_embed_weight = self.position_embedding.weight
        hidden_size = pos_embed_weight.shape[1]
        total_seq = h_coords.shape[0]
        device = pos_embed_weight.device

        # Move coordinates to correct device
        h_coords, w_coords = h_coords.to(device), w_coords.to(device)

        # Handle empty sequence case
        if total_seq == 0:
            adapted_pos_embed = torch.empty(
                0, hidden_size, device=device, dtype=pos_embed_weight.dtype
            )
        else:
            # Convert inputs to tensors if needed
            if isinstance(lengths, list):
                lengths = torch.tensor(lengths, device=device, dtype=torch.long)
            if not isinstance(image_shapes, torch.Tensor):
                image_shapes = torch.tensor(
                    image_shapes, device=device, dtype=torch.long
                )

            # Prepare 2D position embedding
            orig_size_sq = pos_embed_weight.shape[0]
            orig_size = int(orig_size_sq**0.5)
            pos_embed_2d = (
                pos_embed_weight.view(orig_size, orig_size, hidden_size)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device=device, dtype=torch.float32)
            )

            # Calculate target dimensions for each patch
            target_h = torch.cat(
                [image_shapes[i, 1].repeat(lengths[i]) for i in range(len(lengths))]
            ).to(device=device, dtype=torch.float32)
            target_w = torch.cat(
                [image_shapes[i, 2].repeat(lengths[i]) for i in range(len(lengths))]
            ).to(device=device, dtype=torch.float32)

            # Normalize coordinates to [-1, 1] range for grid_sample
            h_coords = h_coords.to(device=device, dtype=torch.float32)
            w_coords = w_coords.to(device=device, dtype=torch.float32)
            norm_w = ((w_coords + 0.5) / target_w) * 2 - 1
            norm_h = ((h_coords + 0.5) / target_h) * 2 - 1

            # Create sampling grid
            grid = torch.stack((norm_w, norm_h), dim=-1).unsqueeze(0).unsqueeze(2)

            # Perform bicubic interpolation
            interpolated_embed_fp32 = F.grid_sample(
                pos_embed_2d,
                grid,
                mode="bicubic",
                align_corners=False,
                padding_mode="border",
            )

            # Reshape and convert back to original dtype
            adapted_pos_embed_fp32 = (
                interpolated_embed_fp32.squeeze(0).squeeze(-1).permute(1, 0)
            )
            adapted_pos_embed = adapted_pos_embed_fp32.to(pos_embed_weight.dtype).to(
                embeddings.device
            )

        # Add adapted position encoding to embeddings
        embeddings = embeddings + adapted_pos_embed
        return embeddings


class Glm4vVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._freqs_cached = None

    def update_freqs_cache(self, seqlen: int) -> None:
        if seqlen > self._seq_len_cached:
            seqlen *= 2
            self._seq_len_cached = seqlen
            self.inv_freq = 1.0 / (
                self.theta
                ** (
                    torch.arange(
                        0,
                        self.dim,
                        2,
                        dtype=torch.float,
                        device=self.inv_freq.device,
                    )
                    / self.dim
                )
            )
            seq = torch.arange(
                seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
            )
            freqs = torch.outer(seq, self.inv_freq)
            self._freqs_cached = freqs

    def forward(self, seqlen: int) -> torch.Tensor:
        self.update_freqs_cache(seqlen)
        return self._freqs_cached[:seqlen]


class Glm4vVisionModel(nn.Module):
    def __init__(
        self,
        vision_config: Glm4vVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        patch_size = vision_config.patch_size
        temporal_patch_size = vision_config.temporal_patch_size
        in_channels = vision_config.in_channels
        depth = vision_config.depth
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads

        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.out_hidden_size = vision_config.out_hidden_size

        self.patch_embed = Glm4vVisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            hidden_size=self.hidden_size,
        )

        norm_layer = partial(Glm4vRMSNorm, eps=norm_eps)
        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = Glm4vVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [
                Glm4vVisionBlock(
                    config=vision_config,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=add_prefix(f"blocks.{layer_idx}", prefix),
                )
                for layer_idx in range(depth)
            ]
        )

        self.merger = Glm4vPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=vision_config.intermediate_size,
            quant_config=quant_config,
            bias=False,
            prefix=add_prefix("merger", prefix),
        )

        self.embeddings = Glm4vVisionEmbeddings(vision_config)

        self.post_conv_layernorm = Glm4vRMSNorm(
            vision_config.hidden_size, eps=vision_config.rms_norm_eps
        )
        self.downsample = nn.Conv2d(
            in_channels=vision_config.hidden_size,
            out_channels=vision_config.out_hidden_size,
            kernel_size=vision_config.spatial_merge_size,
            stride=vision_config.spatial_merge_size,
        )
        self.post_layernorm = Glm4vRMSNorm(
            vision_config.hidden_size, eps=vision_config.rms_norm_eps
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            hpos_ids = (
                hpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            wpos_ids = (
                wpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb, pos_ids

    def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        # patchify
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)
        x = self.post_conv_layernorm(x)

        # compute position embedding
        rotary_pos_emb, image_type_ids = self.rot_pos_emb(grid_thw)
        # compute cu_seqlens
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])

        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        x = self.embeddings(
            x, seqlens, grid_thw, image_type_ids[:, 0], image_type_ids[:, 1]
        )

        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        rotary_pos_emb_tuple = (emb.cos(), emb.sin())

        # x.shape: (s, b, d) where b=1 for vision processing
        # transformers
        x = x.unsqueeze(1)
        for blk in self.blocks:
            x = blk(x, cu_seqlens=cu_seqlens, position_embeddings=rotary_pos_emb_tuple)

        # adapter
        x = self.post_layernorm(x)
        x = x.view(-1, self.spatial_merge_size, self.spatial_merge_size, x.shape[-1])
        x = x.permute(0, 3, 1, 2)
        x = self.downsample(x).view(-1, self.out_hidden_size)
        x = self.merger(x)

        return x


class Glm4vForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    def __init__(
        self,
        config: Glm4vConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)

        self.config = config
        vision_utils.update_vit_attn_dummy_heads_config(self.config)
        self.model = Glm4Model(
            config,
            quant_config,
            prefix=add_prefix("model", prefix),
        )
        self.visual = Glm4vVisionModel(
            config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-5),
            quant_config=quant_config,
            prefix=add_prefix("visual", prefix),
        )

        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        self.is_mrope_enabled = "mrope_section" in self.config.rope_scaling

        # For EAGLE3 support
        self.capture_aux_hidden_states = False

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        pixel_values = torch.cat(
            [item.feature.squeeze(0) for item in items], dim=0
        ).type(self.visual.dtype)
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)
        # For multi-image, pixel_values is [num_of_images, L, C] shape
        # assert pixel_values.dim() == 2, pixel_values.dim()
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (
            image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2
        ).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return torch.cat(image_embeds)

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        pixel_values_videos = torch.cat(
            [item.feature.squeeze(0) for item in items], dim=0
        ).type(self.visual.dtype)
        video_grid_thw = torch.concat([item.video_grid_thw for item in items], dim=0)
        # For multi-video, pixel_values_videos is [num_of_videos, L, C] shape
        # assert pixel_values_videos.dim() == 2, pixel_values_videos.dim()
        assert video_grid_thw.dim() == 2, video_grid_thw.dim()

        # reshape video_grid_thw -> [b, 3] -> [1, h, w] * frames
        temp_frames_hw = []
        for t, h, w in video_grid_thw:
            repeated_row = (
                torch.tensor([1, h.item(), w.item()]).unsqueeze(0).repeat(t, 1)
            )
            temp_frames_hw.append(repeated_row)
        flattened_video_grid_thw = torch.cat(temp_frames_hw, dim=0)
        video_embeds = self.visual(
            pixel_values_videos, grid_thw=flattened_video_grid_thw
        )
        split_sizes = (
            video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2
        ).tolist()
        video_embeds = torch.split(video_embeds, split_sizes)
        return torch.cat(video_embeds)

    def _update_hf_config(self):
        """update hf config to ensure vision attention num_attention_heads is divisible by tp_size"""
        tp_size = get_attention_tp_size()
        num_heads = self.config.vision_config.num_heads
        head_dim = self.config.vision_config.hidden_size // num_heads
        num_dummy_heads = 0

        if num_heads % tp_size != 0:
            num_dummy_heads = (
                (num_heads + tp_size - 1) // tp_size
            ) * tp_size - num_heads

        setattr(self.config.vision_config, "head_dim", head_dim)
        setattr(self.config.vision_config, "num_dummy_heads", num_dummy_heads)

    def _pad_vit_attn_dummy_heads(self, name: str, loaded_weight: torch.Tensor):
        """pad attn qkv weights for dummy heads"""
        num_dummy_heads = self.config.vision_config.num_dummy_heads
        if num_dummy_heads == 0:
            return loaded_weight
        head_dim = self.config.vision_config.head_dim

        if "attn.qkv_proj" in name:
            wq, wk, wv = loaded_weight.chunk(3, dim=0)
            if name.endswith(".weight"):
                dummy_shape = [num_dummy_heads, head_dim, wq.shape[-1]]
            elif name.endswith(".bias"):
                dummy_shape = [num_dummy_heads, head_dim]
            else:
                raise RuntimeError(f"Unsupported weight with name={name}")
            pad_func = lambda x: torch.cat(
                [x.unflatten(0, (-1, head_dim)), x.new_zeros(dummy_shape)], dim=0
            ).flatten(0, 1)
            wq, wk, wv = pad_func(wq), pad_func(wk), pad_func(wv)
            loaded_weight = torch.cat([wq, wk, wv], dim=0)
        elif "attn.proj.weight" in name:
            padded_weight = loaded_weight.new_zeros(
                loaded_weight.shape[0], head_dim * num_dummy_heads
            )
            loaded_weight = torch.cat([loaded_weight, padded_weight], dim=-1)
        elif "attn.q_norm.weight" in name or "attn.k_norm.weight" in name:
            padded_weight = loaded_weight.new_zeros(head_dim * num_dummy_heads)
            loaded_weight = torch.cat([loaded_weight, padded_weight], dim=0)
        return loaded_weight

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
            if "language_model." in name:
                name = name.replace("language_model.", "")
            if "model.visual." in name:
                name = name.replace("model.visual.", "visual.")

            if "rotary_emb.inv_freq" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if "visual" in name:
                    # adapt to VisionAttention
                    name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")

                try:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                except KeyError:
                    print(params_dict.keys())
                    raise

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if "visual" in name:
                    loaded_weight = vision_utils.pad_vit_attn_dummy_heads(
                        self.config, name, loaded_weight
                    )
                weight_loader(param, loaded_weight)


EntryClass = [Glm4vForConditionalGeneration]
