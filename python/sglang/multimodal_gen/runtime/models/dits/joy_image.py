# SPDX-License-Identifier: Apache-2.0

import math
from functools import lru_cache
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from sglang.multimodal_gen.configs.models.dits.joy_image import JoyImageDiTConfig
from sglang.multimodal_gen.runtime.distributed import (
    get_sp_group,
    get_sp_world_size,
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.layers.layernorm import (
    LayerNormScaleShift,
    RMSNorm,
    apply_qk_norm_with_optional_rope,
)
from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
from sglang.multimodal_gen.runtime.layers.mlp import MLP
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.layers.rotary_embedding import NDRotaryEmbedding
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.managers.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.models.dits.wanvideo import WanTimeTextImageEmbedding
from sglang.multimodal_gen.runtime.models.utils import set_weight_attrs
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
_MODULATION_FACTOR = 6


def fused_add_gate(
    residual: torch.Tensor, x: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    """Fused residual addition with gate.

    Computes: residual + x * gate.unsqueeze(1)

    This fuses the gate multiplication and residual addition to reduce
    intermediate tensor allocations and memory bandwidth.

    Args:
        residual (torch.Tensor): The residual tensor to add to. Shape: (B, L, D)
        x (torch.Tensor): The input tensor to be gated. Shape: (B, L, D)
        gate (torch.Tensor): The gate tensor. Shape: (B, D)

    Returns:
        torch.Tensor: residual + x * gate.unsqueeze(1)
    """
    return torch.addcmul(residual, x, gate.unsqueeze(1))


class ModulateWan(nn.Module):
    """Modulation layer for WanX."""

    def __init__(self, hidden_size: int, factor: int, dtype=None, device=None):
        super().__init__()
        self.factor = factor
        self.modulate_table = nn.Parameter(
            torch.zeros(1, factor, hidden_size, dtype=dtype, device=device)
            / hidden_size**0.5,
            requires_grad=False,
        )
        set_weight_attrs(
            self.modulate_table,
            {
                "input_dim": 1,
                "output_dim": 2,
            },
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 3:
            x = x.unsqueeze(1)
        return [
            o.squeeze(1) for o in (self.modulate_table + x).chunk(self.factor, dim=1)
        ]


class MMDoubleStreamBlock(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_pytorch_tanh",
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.heads_num = heads_num
        self.hidden_size = hidden_size
        self.head_dim = self.hidden_size // self.heads_num
        self.mlp_hidden_dim = int(self.hidden_size * mlp_width_ratio)

        self.img_mod = ModulateWan(self.hidden_size, factor=_MODULATION_FACTOR)
        self.fused_modulate_img_norm1 = LayerNormScaleShift(
            self.hidden_size,
            eps=1e-6,
            elementwise_affine=False,
        )

        self.img_attn_qkv = ReplicatedLinear(
            self.hidden_size,
            hidden_size * 3,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.img_attn_qkv",
        )
        self.img_attn_q_norm = RMSNorm(
            self.head_dim,
            eps=1e-6,
        )
        self.img_attn_k_norm = RMSNorm(
            self.head_dim,
            eps=1e-6,
        )
        self.img_attn_proj = ReplicatedLinear(
            self.hidden_size,
            hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.img_attn_proj",
        )

        self.fused_modulate_img_norm2 = LayerNormScaleShift(
            self.hidden_size,
            eps=1e-6,
            elementwise_affine=False,
        )
        self.img_mlp = MLP(
            input_dim=self.hidden_size,
            mlp_hidden_dim=self.mlp_hidden_dim,
            act_type=mlp_act_type,
            quant_config=quant_config,
            prefix=f"{prefix}.img_mlp",
        )

        # Text modulation and attention
        self.txt_mod = ModulateWan(self.hidden_size, factor=_MODULATION_FACTOR)
        self.fused_modulate_txt_norm1 = LayerNormScaleShift(
            self.hidden_size,
            eps=1e-6,
            elementwise_affine=False,
        )
        self.txt_attn_qkv = ReplicatedLinear(
            self.hidden_size,
            self.hidden_size * 3,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.txt_attn_qkv",
        )
        self.txt_attn_q_norm = RMSNorm(
            self.head_dim,
            eps=1e-6,
        )
        self.txt_attn_k_norm = RMSNorm(
            self.head_dim,
            eps=1e-6,
        )
        self.txt_attn_proj = ReplicatedLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.txt_attn_proj",
        )

        self.fused_modulate_txt_norm2 = LayerNormScaleShift(
            self.hidden_size,
            eps=1e-6,
            elementwise_affine=False,
        )
        self.txt_mlp = MLP(
            input_dim=self.hidden_size,
            mlp_hidden_dim=self.mlp_hidden_dim,
            act_type=mlp_act_type,
            quant_config=quant_config,
            prefix=f"{prefix}.txt_mlp",
        )
        self.attn = USPAttention(
            num_heads=self.heads_num,
            head_size=self.head_dim,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            softmax_scale=None,
        )

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        vis_freqs_cis: Optional[torch.Tensor] = None,
        txt_freqs_cis: Optional[torch.Tensor] = None,
        num_replicated_suffix: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through multimodal double stream block."""
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(vec)
        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(vec)

        # Image attention
        img_modulated = self.fused_modulate_img_norm1(
            img, shift=img_mod1_shift, scale=img_mod1_scale
        )
        img_qkv, _ = self.img_attn_qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )

        if vis_freqs_cis is None:
            raise ValueError(
                "vis_freqs_cis is required for fused QK-Norm + RoPE kernel"
            )
        if not (isinstance(vis_freqs_cis, torch.Tensor) and vis_freqs_cis.dim() == 2):
            raise ValueError("vis_freqs_cis must be a 2D cos_sin_cache tensor")
        if img_q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"Fused QK-Norm + RoPE kernel only supports float16/bfloat16, but got {img_q.dtype}"
            )
        img_q = img_q.contiguous()
        img_k = img_k.contiguous()
        img_q, img_k = apply_qk_norm_with_optional_rope(
            q=img_q,
            k=img_k,
            q_norm=self.img_attn_q_norm,
            k_norm=self.img_attn_k_norm,
            head_dim=img_q.shape[-1],
            cos_sin_cache=vis_freqs_cis,
            is_neox=False,
            allow_inplace=True,
        )
        img_q, img_k = img_q.to(img_v), img_k.to(img_v)

        # Text attention
        txt_modulated = self.fused_modulate_txt_norm1(
            txt, shift=txt_mod1_shift, scale=txt_mod1_scale
        )
        txt_qkv, _ = self.txt_attn_qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )

        if txt_freqs_cis is not None and not (
            isinstance(txt_freqs_cis, torch.Tensor) and txt_freqs_cis.dim() == 2
        ):
            raise ValueError("txt_freqs_cis must be a 2D cos_sin_cache tensor")
        txt_q = txt_q.contiguous()
        txt_k = txt_k.contiguous()
        txt_q, txt_k = apply_qk_norm_with_optional_rope(
            q=txt_q,
            k=txt_k,
            q_norm=self.txt_attn_q_norm,
            k_norm=self.txt_attn_k_norm,
            head_dim=txt_q.shape[-1],
            cos_sin_cache=txt_freqs_cis,
            is_neox=False,
            allow_inplace=True,
        )
        txt_q, txt_k = txt_q.to(txt_v), txt_k.to(txt_v)

        # Attention
        joint_query = torch.cat([img_q, txt_q], dim=1)
        joint_key = torch.cat([img_k, txt_k], dim=1)
        joint_value = torch.cat([img_v, txt_v], dim=1)
        attn = self.attn(
            joint_query,
            joint_key,
            joint_value,
            num_replicated_suffix=num_replicated_suffix,
        )
        attn = attn.flatten(2, 3)
        img_attn, txt_attn = (
            attn[:, : img.shape[1]],
            attn[:, img.shape[1] :],
        )

        img = fused_add_gate(img, self.img_attn_proj(img_attn)[0], img_mod1_gate)
        img = fused_add_gate(
            img,
            self.img_mlp(
                self.fused_modulate_img_norm2(
                    img, shift=img_mod2_shift, scale=img_mod2_scale
                )
            ),
            img_mod2_gate,
        )

        # Text blocks
        txt = fused_add_gate(txt, self.txt_attn_proj(txt_attn)[0], txt_mod1_gate)
        txt = fused_add_gate(
            txt,
            self.txt_mlp(
                self.fused_modulate_txt_norm2(
                    txt, shift=txt_mod2_shift, scale=txt_mod2_scale
                )
            ),
            txt_mod2_gate,
        )

        return img, txt


class JoyTransformer3DModel(CachableDiT, OffloadableDiTMixin):
    """
    JoyImage Transformer 3D Model for image generation.

    """

    _supports_gradient_checkpointing = True
    _fsdp_shard_conditions = JoyImageDiTConfig()._fsdp_shard_conditions
    _compile_conditions = JoyImageDiTConfig()._compile_conditions
    _supported_attention_backends = JoyImageDiTConfig()._supported_attention_backends
    param_names_mapping = JoyImageDiTConfig().param_names_mapping
    reverse_param_names_mapping = JoyImageDiTConfig().reverse_param_names_mapping
    lora_param_names_mapping = JoyImageDiTConfig().lora_param_names_mapping

    def __init__(
        self,
        config: JoyImageDiTConfig,
        hf_config: dict[str, Any],
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__(
            config=config,
            hf_config=hf_config,
        )
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels or config.in_channels
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.rope_dim_list = config.rope_dim_list
        self.mm_double_blocks_depth = config.mm_double_blocks_depth
        self.rope_theta = config.rope_theta
        self.quant_config = quant_config
        self.num_channels_latents = self.out_channels

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size {self.hidden_size} must be divisible by num_attention_heads {self.num_attention_heads}"
            )

        # Image projection (patch embedding)
        self.img_in = nn.Conv3d(
            self.in_channels,
            self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # Condition embedding
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=self.hidden_size,
            time_freq_dim=config.freq_dim,
            text_embed_dim=config.text_states_dim,
        )

        # Double blocks (DiT layers)
        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.num_attention_heads,
                    mlp_width_ratio=config.mlp_width_ratio,
                    supported_attention_backends=self._supported_attention_backends,
                    quant_config=quant_config,
                    prefix=f"{config.prefix}.double_blocks.{i}",
                )
                for i in range(self.mm_double_blocks_depth)
            ]
        )
        # Layerwise offload expects ModuleList names here.
        self.layer_names = ["double_blocks"]

        # Output norm & projection
        self.norm_out = nn.LayerNorm(
            self.hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.proj_out = ReplicatedLinear(
            self.hidden_size,
            self.out_channels * math.prod(self.patch_size),
            quant_config=quant_config,
            prefix=f"proj_out",
        )
        self.__post_init__()

        self.sp_size = get_sp_world_size()
        self.rotary_emb = NDRotaryEmbedding(
            rope_dim_list=config.rope_dim_list,
            rope_theta=config.rope_theta,
            dtype=torch.float32,
        )

    @lru_cache(maxsize=1)
    def _compute_rope_for_local_shard(
        self,
        local_len: int,
        rank: int,
        vae_image_sizes: tuple[tuple[int, int, int], ...],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_start = rank * local_len
        token_indices = torch.arange(
            token_start,
            token_start + local_len,
            device=device,
            dtype=torch.long,
        )
        positions = torch.zeros(local_len, 3, device=device, dtype=torch.long)

        cumsum = 0
        current_t_offset = 0
        for t, h, w in vae_image_sizes:
            item_size = t * h * w
            mask = (token_indices >= cumsum) & (token_indices < cumsum + item_size)
            if mask.any():
                local_idx = token_indices[mask] - cumsum
                frame_stride = h * w
                positions[mask, 0] = local_idx // frame_stride + current_t_offset
                positions[mask, 1] = (local_idx % frame_stride) // w
                positions[mask, 2] = local_idx % w
            cumsum += item_size
            current_t_offset += t

        return self.rotary_emb.forward_uncached(positions)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_mask: torch.Tensor | list[torch.Tensor] | None = None,
        vis_freqs_cis: torch.Tensor | None = None,
        txt_freqs_cis: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through JoyImage Transformer."""
        forward_batch = get_forward_context().forward_batch
        sequence_shard_enabled = (
            forward_batch is not None
            and getattr(forward_batch, "enable_sequence_shard", False)
            and self.sp_size > 1
        )

        batch_size = hidden_states.shape[0]

        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]

        if isinstance(encoder_hidden_states_mask, list):
            encoder_hidden_states_mask = encoder_hidden_states_mask[0]

        cond_batch = int(encoder_hidden_states.shape[0])
        if cond_batch != int(batch_size):
            if cond_batch <= 0 or int(batch_size) % cond_batch != 0:
                raise ValueError(
                    "JoyImage conditioning batch mismatch: "
                    f"hidden_states batch={batch_size}, "
                    f"encoder_hidden_states batch={cond_batch}."
                )
            repeat_factor = int(batch_size) // cond_batch
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(
                repeat_factor, dim=0
            )
            if encoder_hidden_states_mask is not None:
                encoder_hidden_states_mask = (
                    encoder_hidden_states_mask.repeat_interleave(repeat_factor, dim=0)
                )

        # Prepare img
        x = rearrange(hidden_states, "b n c p1 p2 p3 -> (b n) c p1 p2 p3")
        x = self.img_in(x)
        img = rearrange(x, "(b n) d 1 1 1 -> b n d", b=batch_size)

        seq_len_orig = img.shape[1]
        seq_shard_pad = 0
        if sequence_shard_enabled:
            if seq_len_orig % self.sp_size != 0:
                seq_shard_pad = self.sp_size - (seq_len_orig % self.sp_size)
                pad = torch.zeros(
                    (batch_size, seq_shard_pad, img.shape[2]),
                    dtype=img.dtype,
                    device=img.device,
                )
                img = torch.cat([img, pad], dim=1)
            sp_rank = get_sp_group().rank_in_group
            local_seq_len = img.shape[1] // self.sp_size
            img = img.view(batch_size, self.sp_size, local_seq_len, img.shape[2])[
                :, sp_rank, :, :
            ].contiguous()

        # Compute rope in model for all SP modes
        if forward_batch is not None and forward_batch.vae_image_sizes is not None:
            vae_image_sizes = tuple(tuple(s) for s in forward_batch.vae_image_sizes)
            local_len = img.shape[1]
            rank = get_sp_group().rank_in_group if self.sp_size > 1 else 0
            freqs_cos, freqs_sin = self._compute_rope_for_local_shard(
                local_len,
                rank,
                vae_image_sizes,
                img.device,
            )
            vis_freqs_cis = torch.cat(
                [
                    freqs_cos.to(dtype=torch.float32).contiguous(),
                    freqs_sin.to(dtype=torch.float32).contiguous(),
                ],
                dim=-1,
            )

        _, vec, txt, _ = self.condition_embedder(timestep, encoder_hidden_states)
        if vec.shape[-1] > self.hidden_size:
            vec = vec.unflatten(1, (_MODULATION_FACTOR, -1))

        txt_suffix_len = txt.shape[1] if sequence_shard_enabled else 0

        # Pass through DiT blocks
        for block in self.double_blocks:
            img, txt = block(
                img,
                txt,
                vec,
                vis_freqs_cis,
                txt_freqs_cis,
                num_replicated_suffix=txt_suffix_len,
            )

        if sequence_shard_enabled:
            img = img.contiguous()
            img = sequence_model_parallel_all_gather(img, dim=1)
            if seq_shard_pad > 0:
                img = img[:, :seq_len_orig, :]

        img, _ = self.proj_out(self.norm_out(img))

        # Restore patch layout expected by downstream latent decoding.
        img = rearrange(
            img,
            "b n (pt ph pw c) -> b n c pt ph pw",
            pt=self.patch_size[0],
            ph=self.patch_size[1],
            pw=self.patch_size[2],
            c=self.out_channels,
        )

        return img


class JoyImageEditTransformer3DModel(JoyTransformer3DModel):
    """Backward-compatible alias for JoyImageEdit model configs."""

    pass


EntryClass = [JoyTransformer3DModel, JoyImageEditTransformer3DModel]
