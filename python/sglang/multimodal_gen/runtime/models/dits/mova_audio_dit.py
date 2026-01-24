# Copied and adapted from: mossVG/mova/diffusion/models/wan_audio_dit.py
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This module reuses common functions from mova_video_dit.py to reduce code duplication.
# Audio-specific functions (precompute_freqs_cis_1d, legacy_precompute_freqs_cis_1d) are kept here.

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange
from torch.distributed.tensor import DTensor

from sglang.multimodal_gen.configs.models.dits.mova_audio import MovaAudioConfig
from sglang.multimodal_gen.runtime.layers.mlp import MLP
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin

# Reuse common functions and classes from mova_video_dit
from .mova_video_dit import DiTBlock, precompute_freqs_cis, sinusoidal_embedding_1d


# Audio-specific positional encoding functions
def legacy_precompute_freqs_cis_1d(
    dim: int,
    end: int = 16384,
    theta: float = 10000.0,
    base_tps=4.0,
    target_tps=44100 / 2048,
):
    s = float(base_tps) / float(target_tps)
    # 1d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta, s)
    # 剩下的维度不施加位置编码
    no_freqs_cis = precompute_freqs_cis(dim // 3, end, theta, s)
    no_freqs_cis = torch.ones_like(no_freqs_cis)
    return f_freqs_cis, no_freqs_cis, no_freqs_cis


def precompute_freqs_cis_1d(dim: int, end: int = 16384, theta: float = 10000.0):
    f_freqs_cis = precompute_freqs_cis(dim, end, theta)
    return f_freqs_cis.chunk(3, dim=-1)


class Head(nn.Module):
    def __init__(
        self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        # print(f"{t_mod.shape = }")
        if len(t_mod.shape) == 3:
            shift, scale = (
                self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device)
                + t_mod.unsqueeze(2)
            ).chunk(2, dim=2)
            x = self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2))
        else:
            # NOTE: 这里 t_mod 原本是 [B, C], 当 B = 1 时可以通过广播机制正确处理，但 B > 1 后就会和 [1, 2, C] 匹配不上
            shift, scale = (
                self.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
                + t_mod.unsqueeze(1)
            ).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + scale) + shift)
        return x


class Conv1dLocalIsland(nn.Conv1d):
    """
    继承 Conv1d，只改 forward：
    - 参数继续保留为 DTensor（优化器一致性不变）
    - 前向把 x/weight/bias 统一聚合为 Replicate，再 to_local 本地卷积
    - 输出再 distribute 成 DTensor（默认 Replicate，可自定义 placements）
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if isinstance(x, DTensor):
            x_local = x.to_local()
            w_local = self.weight.to_local()
            b_local = self.bias.to_local()

            return self._conv_forward(x_local, w_local, b_local)
        else:
            return super().forward(x)


class WanAudioModel(CachableDiT, OffloadableDiTMixin, ModelMixin, ConfigMixin):
    _repeated_blocks = ("DiTBlock",)
    _fsdp_shard_conditions = MovaAudioConfig()._fsdp_shard_conditions
    _compile_conditions = MovaAudioConfig()._compile_conditions
    _supported_attention_backends = MovaAudioConfig()._supported_attention_backends
    param_names_mapping = MovaAudioConfig().param_names_mapping
    reverse_param_names_mapping = MovaAudioConfig().reverse_param_names_mapping
    lora_param_names_mapping = MovaAudioConfig().lora_param_names_mapping

    def __init__(self, config: MovaAudioConfig, hf_config: dict[str, Any]) -> None:
        super().__init__()

        # Extract parameters from config
        dim = config.dim
        in_dim = config.in_dim
        ffn_dim = config.ffn_dim
        out_dim = config.out_dim
        text_dim = config.text_dim
        freq_dim = config.freq_dim
        eps = config.eps
        patch_size = config.patch_size
        num_heads = config.num_heads
        num_layers = config.num_layers
        has_image_input = config.has_image_input
        has_image_pos_emb = config.has_image_pos_emb
        has_ref_conv = config.has_ref_conv
        add_control_adapter = config.add_control_adapter
        in_dim_control_adapter = config.in_dim_control_adapter
        seperated_timestep = config.seperated_timestep
        require_vae_embedding = config.require_vae_embedding
        require_clip_embedding = config.require_clip_embedding
        fuse_vae_embedding_in_latents = config.fuse_vae_embedding_in_latents
        vae_type = config.vae_type

        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.vae_type = vae_type
        # self.patch_embedding = nn.Conv3d(
        #     in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.patch_embedding = Conv1dLocalIsland(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = MLP(
            text_dim, dim, output_dim=dim, act_type="gelu_pytorch_tanh"
        )
        self.time_embedding = MLP(freq_dim, dim, output_dim=dim, act_type="silu")
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
                for _ in range(num_layers)
            ]
        )
        self.head = Head(dim, out_dim, patch_size, eps)
        self.num_heads = num_heads
        self.freqs = None

        if has_image_input:
            self.img_emb = MLP(
                1280, dim, output_dim=dim, act_type="gelu_pytorch_tanh"
            )  # clip_feature_dim = 1280
            self.img_pos_emb = (
                nn.Parameter(torch.zeros((1, 514, 1280))) if has_image_pos_emb else None
            )
        else:
            self.img_pos_emb = None
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        self.hidden_size = dim
        self.num_attention_heads = num_heads
        self.num_channels_latents = out_dim
        self.layer_names = ["blocks"]
        self.cnt = 0
        self.teacache_thresh = 0
        self.coefficients = []
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.previous_resiual = None
        self.previous_e0_even = None
        self.previous_e0_odd = None
        self.previous_residual_even = None
        self.previous_residual_odd = None
        self.is_even = False
        self.should_calc_even = True
        self.should_calc_odd = True
        self.accumulated_rel_l1_distance_even = 0
        self.accumulated_rel_l1_distance_odd = 0
        self.__post_init__()
        if add_control_adapter:
            from .wan_video_camera_controller import SimpleAdapter

            self.control_adapter = SimpleAdapter(
                in_dim_control_adapter,
                dim,
                kernel_size=patch_size[1:],
                stride=patch_size[1:],
            )
        else:
            self.control_adapter = None

    def _init_freqs(self):
        if self.freqs is not None:
            return
        head_dim = self.dim // self.num_heads
        if self.vae_type == "oobleck":
            # NOTE(dhyu): 这个位置编码算法没什么道理，4.0 也是错的，因为 Wan2.2 是 6.0
            self.freqs = legacy_precompute_freqs_cis_1d(
                head_dim, base_tps=4.0, target_tps=44100 / 2048
            )
        elif self.vae_type == "dac":
            self.freqs = precompute_freqs_cis_1d(head_dim)
        else:
            raise ValueError(f"Invalid VAE type: {self.vae_type}")

    def patchify(
        self,
        x: torch.Tensor,
        control_camera_latents_input: Optional[torch.Tensor] = None,
    ):
        x = self.patch_embedding(x)
        if (
            self.control_adapter is not None
            and control_camera_latents_input is not None
        ):
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f -> b f c").contiguous()
        return x, grid_size  # x, grid_size: (f)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, "b f (p c) -> b c (f p)", f=grid_size[0], p=self.patch_size[0]
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **kwargs,
    ):
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_feature_input = clip_feature
            if self.img_pos_emb is not None:
                clip_feature_input = clip_feature_input + self.img_pos_emb.to(
                    dtype=clip_feature.dtype, device=clip_feature.device
                )
            clip_embedding = self.img_emb(clip_feature_input)
            context = torch.cat([clip_embedding, context], dim=1)

        x, (f,) = self.patchify(x)

        freqs = (
            torch.cat(
                [
                    self.freqs[0][:f].view(f, -1).expand(f, -1),
                    self.freqs[1][:f].view(f, -1).expand(f, -1),
                    self.freqs[2][:f].view(f, -1).expand(f, -1),
                ],
                dim=-1,
            )
            .reshape(f, 1, -1)
            .to(x.device)
        )

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x,
                            context,
                            t_mod,
                            freqs,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        context,
                        t_mod,
                        freqs,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod, freqs)

        x = self.head(x, t)
        x = self.unpatchify(x, (f,))
        return x


EntryClass = WanAudioModel
