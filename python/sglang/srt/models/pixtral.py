# Copyright 2024 SGLang Team
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

"""
Using mistral-community/pixtral-12b as reference.
"""

from dataclasses import dataclass, fields
from typing import Iterable, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PixtralVisionConfig, PretrainedConfig
from transformers.models.pixtral.modeling_pixtral import PixtralRotaryEmbedding
from transformers.models.pixtral.modeling_pixtral import (
    generate_block_attention_mask as _get_pixtral_attention_mask,
)
from transformers.models.pixtral.modeling_pixtral import position_ids_in_meshgrid

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.mistral_large_3 import MistralLarge3ForCausalLM

USE_XFORMERS_OPS = False
PATCH_MERGE = "patch_merge"


# Vision encoder
@dataclass
class VisionEncoderArgs:
    hidden_size: int
    num_channels: int
    image_size: int
    patch_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    rope_theta: float  # for rope-2D
    image_token_id: int
    adapter_bias: bool = True
    spatial_merge_size: int = 1
    add_pre_mm_projector_layer_norm: bool = False
    mm_projector_id: str = ""


class PixtralForConditionalGeneration(nn.Module):
    merge_by_field_config = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return None

        raise ValueError("Only image modality is supported")

    def __init__(self, *, config, prefix: str = "", **kwargs):
        super().__init__()
        self.config = config
        dataclass_fields = {field.name for field in fields(VisionEncoderArgs)}
        vision_args = {
            key: value
            for key, value in self.config.vision_config.to_dict().items()
            if key in dataclass_fields
        }

        self.vision_args = VisionEncoderArgs(**vision_args)

        self.language_model = MistralLarge3ForCausalLM(
            config=self.config.text_config,
            quant_config=kwargs.get("quant_config"),
        )

        self.vision_encoder = VisionTransformer(self.vision_args)

        if self.vision_args.add_pre_mm_projector_layer_norm:
            self.pre_mm_projector_norm = RMSNorm(self.vision_args.hidden_size, eps=1e-5)

        if self.vision_args.mm_projector_id == PATCH_MERGE:
            self.patch_merger = PatchMerger(
                vision_encoder_dim=self.vision_args.hidden_size,
                spatial_merge_size=self.vision_args.spatial_merge_size,
                use_mlp_bias=False,
            )

        self.vision_language_adapter = VisionLanguageAdapter(
            self.vision_args, dim=self.config.text_config.hidden_size
        )

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        def is_vision_encoder_weights(weight: tuple[str, torch.Tensor]):
            return weight[0].startswith("vision_encoder")

        def is_vision_lang_adapter_weights(weight: tuple[str, torch.Tensor]):
            return weight[0].startswith("vision_language_adapter")

        def is_patch_merger(weight: tuple[str, torch.Tensor]):
            return weight[0].startswith("patch_merger")

        def is_pre_mm_projector_norm(weight: tuple[str, torch.Tensor]):
            return weight[0].startswith("pre_mm_projector_norm")

        # Get references to parameters for direct loading
        vision_encoder_dict = dict(self.vision_encoder.named_parameters())
        patch_merger_dict = (
            dict(self.patch_merger.named_parameters())
            if self.vision_args.mm_projector_id == PATCH_MERGE
            else dict()
        )
        pre_mm_projector_norm_dict = (
            dict(self.pre_mm_projector_norm.named_parameters())
            if self.vision_args.add_pre_mm_projector_layer_norm
            else dict()
        )
        vision_lang_adapter_dict = dict(self.vision_language_adapter.named_parameters())

        def llm_weights_generator():
            # Single pass over weights
            for name, w in weights:
                if is_vision_encoder_weights((name, w)):
                    # Load vision encoder weights directly
                    trimmed_name = ".".join(name.split(".")[1:])
                    # NOTE: The current nvfp4 model has extra weights that we need to ignore, called
                    # vision_encoder.transformer.layers.*.attention.{k,v}_fake_quantizer.qscale_act
                    # TODO: Remove this if condition once the model is fixed
                    if "fake_quantizer.qscale_act" in trimmed_name:
                        continue
                    param = vision_encoder_dict[trimmed_name]
                    with torch.no_grad():
                        default_weight_loader(param, w)
                elif is_patch_merger((name, w)):
                    # Load vision patch merger weights directly
                    trimmed_name = ".".join(name.split(".")[1:])
                    param = patch_merger_dict[trimmed_name]
                    with torch.no_grad():
                        default_weight_loader(param, w)
                elif is_pre_mm_projector_norm((name, w)):
                    # Load vision pre_mm_projector_norm weights directly
                    trimmed_name = ".".join(name.split(".")[1:])
                    param = pre_mm_projector_norm_dict[trimmed_name]
                    with torch.no_grad():
                        default_weight_loader(param, w)
                elif is_vision_lang_adapter_weights((name, w)):
                    # Load vision-language adapter weights directly
                    trimmed_name = ".".join(name.split(".")[1:])
                    param = vision_lang_adapter_dict[trimmed_name]
                    with torch.no_grad():
                        default_weight_loader(param, w)
                else:
                    # LLM weights: yield them to be loaded
                    # by language_model.load_weights
                    yield (name, w)

        # Now we call the language model load with the generator
        self.language_model.load_weights(llm_weights_generator())

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        images = [item.feature for item in items]
        # Process images through vision encoder
        image_features = self.vision_encoder(images)
        if self.vision_args.add_pre_mm_projector_layer_norm:
            image_features = image_features.view(-1, image_features.shape[-1])
            image_features = self.pre_mm_projector_norm(image_features)
        if self.vision_args.mm_projector_id == PATCH_MERGE:
            patch_size = self.vision_args.patch_size
            img_patch_dims = [
                (img.shape[-2] // patch_size, img.shape[-1] // patch_size)
                for img in images
                for _ in range(img.shape[0])
            ]
            image_features = self.patch_merger(
                image_features, image_sizes=img_patch_dims
            )
        image_embeds = self.vision_language_adapter(image_features)
        return image_embeds

    def forward(self, input_ids, positions, forward_batch):
        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            multimodal_model=self,
            positions=positions,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def get_embed_and_head(self):
        return self.language_model.get_embed_and_head()


class PatchMerger(nn.Module):
    """
    Learned merging of spatial_merge_size ** 2 patches
    """

    def __init__(
        self,
        vision_encoder_dim: int,
        spatial_merge_size: int,
        use_mlp_bias: bool = False,
    ) -> None:
        super().__init__()

        mlp_input_dim = vision_encoder_dim * (spatial_merge_size**2)

        self.spatial_merge_size = spatial_merge_size
        self.mlp_input_dim = mlp_input_dim

        self.merging_layer = nn.Linear(
            mlp_input_dim,
            vision_encoder_dim,
            bias=use_mlp_bias,
        )

    def forward(
        self, x: torch.Tensor, image_sizes: list[tuple[int, int]]
    ) -> torch.Tensor:
        # image_sizes specified in tokens
        assert sum([h * w for h, w in image_sizes]) == x.shape[-2]

        # x is (N, vision_encoder_dim)
        x = self.permute(x, image_sizes)

        # x is (N / spatial_merge_size ** 2,
        #       vision_encoder_dim * spatial_merge_size ** 2)
        x = self.merging_layer(x)

        # x is (N / spatial_merge_size ** 2, vision_encoder_dim)
        return x

    def permute(
        self,
        x: torch.Tensor,
        image_sizes: list[tuple[int, int]],
    ) -> torch.Tensor:
        """
        Args:
            x: (N, D) where N is flattened and concatenated patch tokens
                for all images
            image_sizes: list of tuple of (height, width) in tokens for
                each image
        Returns:
            image_features: reorders patch tokens so each grid of
                (spatial_merge_size, spatial_merge_size) is contiguous.
                now (N / spatial_merge_size ** 2, D * spatial_merge_size ** 2)
        """

        sub_grids = get_sub_grids(
            x=x, image_sizes=image_sizes, spatial_merge_size=self.spatial_merge_size
        )  # list of [d x sub_grid_size x sub_grid_size x n_patches]
        permuted_tensor: list[torch.Tensor] = []
        for grid in sub_grids:
            n_patches = grid.shape[-1]
            permuted_tensor.append(
                grid.view(-1, n_patches).t()
            )  # n_patches x d * sub_grid_size * sub_grid_size
        return torch.cat(
            permuted_tensor, dim=0
        )  # (N / spatial_merge_size ** 2, d * spatial_merge_size ** 2)


def get_sub_grids(
    x: torch.Tensor,
    image_sizes: list[tuple[int, int]],
    spatial_merge_size: int,
) -> list[torch.Tensor]:
    # image_sizes specified in tokens
    tokens_per_image = [h * w for h, w in image_sizes]
    d = x.shape[-1]
    all_img_sub_grids: list[torch.Tensor] = []
    sub_grid_size = spatial_merge_size

    for image_index, image_tokens in enumerate(x.split(tokens_per_image)):
        # Reshape image_tokens into a 2D grid
        h, w = image_sizes[image_index]
        image_grid = image_tokens.view(h, w, d).permute(2, 0, 1)[
            None, :, :, :
        ]  # 1 x d x h x w
        sub_grids = torch.nn.functional.unfold(
            image_grid, kernel_size=sub_grid_size, stride=sub_grid_size
        )
        sub_grids = sub_grids.view(
            1, d, sub_grid_size, sub_grid_size, -1
        )  # 1 x d x sub_grid_size x sub_grid_size x n_patches

        all_img_sub_grids.append(sub_grids[0])

    return all_img_sub_grids


class VisionTransformer(nn.Module):
    def __init__(self, args: VisionEncoderArgs):
        super().__init__()
        self.args = args
        self.patch_conv = nn.Conv2d(
            in_channels=args.num_channels,
            out_channels=args.hidden_size,
            kernel_size=args.patch_size,
            stride=args.patch_size,
            bias=False,
        )
        self.ln_pre = RMSNorm(args.hidden_size, eps=1e-5)
        self.transformer = Transformer(args)

        head_dim = self.args.hidden_size // self.args.num_attention_heads
        assert head_dim % 2 == 0, "ROPE requires even head_dim"
        self._freqs_cis: torch.Tensor | None = None

    @property
    def max_patches_per_side(self) -> int:
        return self.args.image_size // self.args.patch_size

    @property
    def device(self) -> torch.types.Device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def freqs_cis(self) -> torch.Tensor:
        if self._freqs_cis is None:
            self._freqs_cis = precompute_freqs_cis_2d(
                dim=self.args.hidden_size // self.args.num_attention_heads,
                height=self.max_patches_per_side,
                width=self.max_patches_per_side,
                theta=self.args.rope_theta,
            )

        if self._freqs_cis.device != self.device:
            self._freqs_cis = self._freqs_cis.to(device=self.device)

        return self._freqs_cis

    def forward(
        self,
        images: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            images: list of N_img images of variable sizes,
                each of shape (B, C, H, W)
        Returns:
            image_features: tensor of token features for
                all tokens of all images of shape (N_toks, D)
        """
        patch_embeds_list = [self.patch_conv(img.to(self.dtype)) for img in images]

        patch_embeds = [p.flatten(2).permute(0, 2, 1) for p in patch_embeds_list]

        patch_embeds = torch.cat(patch_embeds, dim=1)
        patch_embeds_shape = patch_embeds.shape
        patch_embeds = patch_embeds.view(-1, patch_embeds_shape[-1])
        patch_embeds = self.ln_pre(patch_embeds)
        patch_embeds = patch_embeds.view(patch_embeds_shape)

        # positional embeddings
        positions = position_meshgrid(patch_embeds_list).to(self.device)
        freqs_cis = self.freqs_cis[positions[:, 0], positions[:, 1]]

        # pass through Transformer with a block diagonal mask delimiting images
        if USE_XFORMERS_OPS:
            from xformers import ops as xops

            mask = xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(
                [p.shape[-2] * p.shape[-1] for p in patch_embeds_list],
            )
        else:
            from transformers.models.pixtral.modeling_pixtral import (
                generate_block_attention_mask,
            )

            mask = generate_block_attention_mask(
                [p.shape[-2] * p.shape[-1] for p in patch_embeds_list], patch_embeds
            )
        return self.transformer(patch_embeds, mask=mask, freqs_cis=freqs_cis)


def position_meshgrid(
    patch_embeds_list: list[torch.Tensor],
) -> torch.Tensor:
    positions = torch.cat(
        [
            torch.stack(
                torch.meshgrid(
                    torch.arange(p.shape[-2]),
                    torch.arange(p.shape[-1]),
                    indexing="ij",
                ),
                dim=-1,
            ).reshape(-1, 2)
            for p in patch_embeds_list
        ]
    )
    return positions


class PixtralHFMLP(nn.Module):
    """MLP for PixtralHFVisionModel using SGLang components."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()

        assert config.intermediate_size is not None

        # Use MergedColumnParallelLinear for gate_up_proj to handle combined weights
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[config.intermediate_size, config.intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )

        self.down_proj = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )

        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up_output, _ = self.gate_up_proj(x)

        # Apply SiLU activation and multiply
        gate_up = self.act_fn(gate_up_output)

        # Project back to hidden size
        out, _ = self.down_proj(gate_up)
        return out


class VisionLanguageAdapter(nn.Module):
    def __init__(self, args: VisionEncoderArgs, dim: int):
        super().__init__()
        assert isinstance(args, VisionEncoderArgs)
        self.w_in = nn.Linear(
            args.hidden_size,
            dim,
            bias=args.adapter_bias,
        )
        self.gelu = nn.GELU()
        self.w_out = nn.Linear(dim, dim, bias=args.adapter_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(self.gelu(self.w_in(x)))


class PixtralHFTransformerBlock(nn.Module):
    """Transformer block for PixtralHFVisionModel using SGLang components."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.hidden_size, eps=1e-5)

        # Use SGLang's VisionAttention instead of vLLM's PixtralHFAttention
        self.attention = VisionAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            projection_size=config.hidden_size,
            use_qkv_parallel=True,
            quant_config=quant_config,
            dropout=0.0,
            use_context_forward=False,
            flatten_batch=False,
            qkv_bias=False,
            proj_bias=False,
            prefix=f"{prefix}.attention",
        )

        self.feed_forward = PixtralHFMLP(
            config, quant_config=quant_config, prefix=f"{prefix}.feed_forward"
        )

        self.ffn_norm = RMSNorm(config.hidden_size, eps=1e-5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        # Ensure hidden_states has the batch dimension [batch, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Apply attention norm - normalize along the last dimension
        attn_normalized = self.attention_norm(hidden_states.view(-1, hidden_dim)).view(
            batch_size, seq_len, hidden_dim
        )

        # Pass through attention layer
        attention_output = self.attention(
            attn_normalized,
            attention_mask=attention_mask,
            cu_seqlens=None,
            position_embeddings=position_embeddings,
        )

        # Apply first residual connection
        hidden_states = hidden_states + attention_output

        # Apply feed-forward norm - normalize along the last dimension
        ffn_normalized = self.ffn_norm(hidden_states.view(-1, hidden_dim)).view(
            batch_size, seq_len, hidden_dim
        )

        # Pass through feed-forward layer
        # First reshape to 2D for the feed-forward network, then reshape back
        ffn_output = self.feed_forward(ffn_normalized)

        # Apply second residual connection
        output = hidden_states + ffn_output

        return output


def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    freqs_cis: complex - (seq_len, head_dim / 2)
    x: complex - (bsz, seq_len, head_dim / 2)
    """
    ndim = x.ndim
    assert ndim > 1
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (
        freqs_cis.shape,
        (x.shape[1], x.shape[-1]),
    )
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def precompute_freqs_cis_2d(
    dim: int,
    height: int,
    width: int,
    theta: float,
) -> torch.Tensor:
    """
    freqs_cis: 2D complex tensor of shape (height, width, dim // 2)
        to be indexed by (height, width) position tuples
    """
    # (dim / 2) frequency bases
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    h = torch.arange(height, device=freqs.device)
    w = torch.arange(width, device=freqs.device)

    freqs_h = torch.outer(h, freqs[::2]).float()
    freqs_w = torch.outer(w, freqs[1::2]).float()
    freqs_2d = torch.cat(
        [
            freqs_h[:, None, :].repeat(1, width, 1),
            freqs_w[None, :, :].repeat(height, 1, 1),
        ],
        dim=-1,
    )
    return torch.polar(torch.ones_like(freqs_2d), freqs_2d)


def apply_rotary_emb_vit(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    assert freqs_cis.dtype == torch.complex64
    freqs_cis = _reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class FeedForward(nn.Module):
    def __init__(self, args: VisionEncoderArgs):
        super().__init__()
        assert args.intermediate_size is not None
        self.w1 = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.w2 = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.w3 = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    def __init__(self, args: VisionEncoderArgs):
        super().__init__()
        self.args = args
        assert not args.hidden_size % args.num_attention_heads
        self.n_heads = args.num_attention_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.wq = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.wk = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.wv = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.wo = nn.Linear(args.hidden_size, args.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        batch, patches, _ = x.shape

        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = q.reshape(batch, patches, self.n_heads, self.head_dim)
        k = k.reshape(batch, patches, self.n_heads, self.head_dim)
        v = v.reshape(batch, patches, self.n_heads, self.head_dim)

        q, k = apply_rotary_emb_vit(q, k, freqs_cis=freqs_cis)

        if USE_XFORMERS_OPS:
            from xformers import ops as xops

            out = xops.memory_efficient_attention(q, k, v, attn_bias=mask)
        else:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            out = out.transpose(1, 2)

        out = out.reshape(batch, patches, self.n_heads * self.head_dim)
        return self.wo(out)


class TransformerBlock(nn.Module):
    def __init__(self, args: VisionEncoderArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.hidden_size, eps=1e-5)
        self.ffn_norm = RMSNorm(args.hidden_size, eps=1e-5)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        attention_norm_x = self.attention_norm(x.view(-1, x.shape[-1]))
        attention_norm_x = attention_norm_x.view(x.shape)
        r = self.attention.forward(attention_norm_x, mask=mask, freqs_cis=freqs_cis)
        h = x + r
        ffn_norm_h = self.ffn_norm(h.view(-1, h.shape[-1]))
        ffn_norm_h = ffn_norm_h.view(h.shape)
        r = self.feed_forward.forward(ffn_norm_h)
        out = h + r
        return out


class Transformer(nn.Module):
    def __init__(self, args: VisionEncoderArgs):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for _ in range(args.num_hidden_layers):
            self.layers.append(TransformerBlock(args))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        freqs_cis: torch.Tensor | None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask, freqs_cis=freqs_cis)
        return x


class PixtralHFTransformer(nn.Module):
    """Transformer for PixtralHFVisionModel using SGLang components."""

    def __init__(
        self,
        config: PixtralVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        num_hidden_layers = config.num_hidden_layers
        if num_hidden_layers_override is not None:
            num_hidden_layers = num_hidden_layers_override

        self.layers = nn.ModuleList(
            [
                PixtralHFTransformerBlock(
                    config=config,
                    layer_id=layer_idx,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(num_hidden_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        return_all_hidden_states: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Forward pass through transformer layers.

        Args:
            x: Input tensor
            attention_mask: Optional attention mask
            position_embeddings: Optional position embeddings for rotary attention
            return_all_hidden_states: Whether to return all hidden states

        Returns:
            Either the final hidden state, or a list of all hidden states if
            return_all_hidden_states is True
        """
        # For HF model compatibility, always start with the input
        hidden_states = x
        all_hidden_states = [hidden_states] if return_all_hidden_states else None

        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, attention_mask, position_embeddings)
            if return_all_hidden_states:
                all_hidden_states.append(hidden_states)

        if return_all_hidden_states:
            return all_hidden_states
        return hidden_states


def resolve_visual_encoder_outputs(
    outputs: Union[torch.Tensor, List[torch.Tensor]],
    feature_sample_layers: Optional[List[int]],
    post_norm: Optional[nn.Module],
    num_hidden_layers: int,
) -> torch.Tensor:
    """Resolve outputs from visual encoder based on feature_sample_layers."""
    if feature_sample_layers is None:
        # Just use the last layer's output
        if isinstance(outputs, list):
            outputs = outputs[-1]
        if post_norm is not None:
            outputs = post_norm(outputs)
        return outputs

    # Handle the case where we want to use specific layers
    if not isinstance(outputs, list):
        raise ValueError(
            "Expected outputs to be a list when feature_sample_layers is provided"
        )

    # Validate layer indices
    for layer_idx in feature_sample_layers:
        if layer_idx < 0 or layer_idx > num_hidden_layers:
            raise ValueError(
                f"Feature sample layer index {layer_idx} is out of range "
                f"[0, {num_hidden_layers}]"
            )

    # Collect outputs from specified layers
    selected_outputs = [outputs[layer_idx] for layer_idx in feature_sample_layers]

    # Combine the outputs
    combined_outputs = torch.cat(selected_outputs, dim=-1)

    if post_norm is not None:
        combined_outputs = post_norm(combined_outputs)

    return combined_outputs


class PixtralHFVisionModel(nn.Module):
    """Hugging Face Pixtral Vision Model implemented using SGLang components."""

    DEFAULT_IMAGE_TOKEN_ID = 10

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return self.input_padder.pad_input_tokens(input_ids, mm_inputs)

    def __init__(
        self,
        config: PixtralVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config

        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_conv = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )

        self.ln_pre = RMSNorm(config.hidden_size, eps=1e-5)

        self.transformer = PixtralHFTransformer(
            config,
            quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            prefix=f"{prefix}.transformer",
        )

        # Check that num_hidden_layers is valid
        num_hidden_layers = config.num_hidden_layers
        if len(self.transformer.layers) > config.num_hidden_layers:
            raise ValueError(
                f"The original encoder only has {num_hidden_layers} "
                f"layers, but you requested {len(self.transformer.layers)} "
                "layers."
            )

        # Initialize patch position embedding
        self.patch_positional_embedding = PixtralRotaryEmbedding(config)
        self.input_padder = MultiModalityDataPaddingPatternMultimodalTokens()

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_sizes: list[tuple[int, int]],
        output_hidden_states: bool = False,
        feature_sample_layers: Optional[list[int]] = None,
    ) -> Union[torch.Tensor, tuple]:
        """
        Args:
            pixel_values: [batch_size, C, H, W], padded if multiple images
            image_sizes: list of (H, W) for each image in the batch
            output_hidden_states: Whether to return all hidden states.
            feature_sample_layers: Layer indices whose features should be
                concatenated and used as the visual encoder output. If none
                are provided, the last layer is used.

        Returns:
            A tuple containing:
              - hidden_states: Final model outputs (or selected layers if feature_sample_layers given)
              - hidden_states tuple (optional): All hidden states if output_hidden_states=True
        """
        # batch patch images
        embeds_orig = self.patch_conv(
            pixel_values.to(device=self.device, dtype=self.dtype)
        )
        # crop the embeddings
        embeds_2d = [
            embed[..., : h // self.patch_size, : w // self.patch_size]
            for embed, (h, w) in zip(embeds_orig, image_sizes)
        ]

        # flatten to sequence
        embeds_1d = torch.cat([p.flatten(1).T for p in embeds_2d], dim=0)
        embeds_featurized = self.ln_pre(embeds_1d).unsqueeze(0)

        # positional embeddings
        position_ids = position_ids_in_meshgrid(
            embeds_2d,
            max_width=self.image_size // self.patch_size,
        ).to(self.device)

        # The original PixtralRotaryEmbedding expects 2D input but returns a tuple of tensors (cos, sin)
        # These tensors are used by apply_rotary_pos_emb in the transformer blocks
        position_embedding = self.patch_positional_embedding(
            embeds_featurized, position_ids
        )
        attention_mask = _get_pixtral_attention_mask(
            [p.shape[-2] * p.shape[-1] for p in embeds_2d], embeds_featurized
        )

        return_all_hidden_states = (
            output_hidden_states or feature_sample_layers is not None
        )

        transformer_outputs = self.transformer(
            embeds_featurized,  # add batch dimension
            attention_mask,
            position_embedding,
            return_all_hidden_states=return_all_hidden_states,
        )

        # Store all hidden states if requested
        all_hidden_states = None
        if isinstance(transformer_outputs, list):
            all_hidden_states = transformer_outputs
            # Use the last layer by default if feature_sample_layers is not specified
            if feature_sample_layers is None:
                out = transformer_outputs[-1]
            else:
                # Resolve outputs based on feature sample layers
                out = resolve_visual_encoder_outputs(
                    transformer_outputs,
                    feature_sample_layers,
                    None,
                    self.config.num_hidden_layers,
                )
        else:
            out = transformer_outputs

        # Format return to be compatible with HuggingFace vision models
        if output_hidden_states:
            return type(
                "VisualOutput",
                (),
                {
                    "last_hidden_state": out,
                    "hidden_states": all_hidden_states,
                },
            )
        else:
            return out

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        """Load weights from a HuggingFace checkpoint with proper parameter mapping."""
        params_dict = dict(self.named_parameters())

        # for (param, weight, shard_id): load weight into param as param's shard_id part
        stacked_params_mapping = [
            (".attention.qkv_proj", ".attention.q_proj", "q"),
            (".attention.qkv_proj", ".attention.k_proj", "k"),
            (".attention.qkv_proj", ".attention.v_proj", "v"),
            (".feed_forward.gate_up_proj", ".feed_forward.gate_proj", 0),
            (".feed_forward.gate_up_proj", ".feed_forward.up_proj", 1),
        ]

        # Process each weight
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name in name:
                    # Replace the weight name part with the combined parameter name
                    transformed_name = name.replace(weight_name, param_name)
                    if transformed_name in params_dict:
                        param = params_dict[transformed_name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight, shard_id)
                        break
            else:
                if ".attention.o_proj" in name:
                    alt_name = name.replace(".attention.o_proj", ".attention.proj")
                    if alt_name in params_dict:
                        name = alt_name
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)


class PixtralVisionModel(PixtralHFVisionModel):
    pass


# Register the model classes for external access
EntryClass = [PixtralForConditionalGeneration, PixtralVisionModel]
