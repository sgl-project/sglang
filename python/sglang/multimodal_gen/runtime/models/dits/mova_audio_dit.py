# Copied and adapted from: mossVG/mova/diffusion/models/wan_audio_dit.py
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This module reuses common functions from mova_video_dit.py to reduce code duplication.
# Audio-specific functions (precompute_freqs_cis_1d, legacy_precompute_freqs_cis_1d) are kept here.

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.distributed.tensor import DTensor

from sglang.multimodal_gen.configs.models.dits.mova_audio import MOVAAudioConfig
from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
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
    # No positional encoding is applied to the remaining dimensions
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
        self.head = ReplicatedLinear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (
                self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device)
                + t_mod.unsqueeze(2)
            ).chunk(2, dim=2)
            x, _ = self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2))
        else:
            # NOTE: t_mod was originally [B, C]. This works correctly with broadcasting when B=1, but it won't match [1, 2, C] when B > 1.
            shift, scale = (
                self.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
                + t_mod.unsqueeze(1)
            ).chunk(2, dim=1)
            x, _ = self.head(self.norm(x) * (1 + scale) + shift)
        return x


class Conv1dLocalIsland(nn.Conv1d):
    """Inherits from Conv1d and overrides forward.

    - Parameters remain as DTensors (optimizer consistency is maintained).
    - In the forward pass, x, weight, and bias are aggregated as Replicate,
      and then local convolution is performed via to_local.
    - The output is then redistributed as a DTensor (default is Replicate,
      placements can be customized).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        if isinstance(input, DTensor):
            x_local = input.to_local()  # type: ignore[attr-defined]
            w_local = self.weight.to_local()  # type: ignore[attr-defined]
            b_local = (
                self.bias.to_local() if self.bias is not None else None  # type: ignore[attr-defined]
            )

            return self._conv_forward(x_local, w_local, b_local)
        else:
            return super().forward(input)


class WanAudioModel(CachableDiT, OffloadableDiTMixin):
    _fsdp_shard_conditions = MOVAAudioConfig()._fsdp_shard_conditions
    _compile_conditions = MOVAAudioConfig()._compile_conditions
    _supported_attention_backends = MOVAAudioConfig()._supported_attention_backends
    param_names_mapping = MOVAAudioConfig().param_names_mapping
    reverse_param_names_mapping = MOVAAudioConfig().reverse_param_names_mapping
    lora_param_names_mapping = MOVAAudioConfig().lora_param_names_mapping

    def __init__(self, config: MOVAAudioConfig, hf_config: dict[str, Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)

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
        # Preserve state_dict keys (time_projection.1.weight/bias).
        self.time_projection = nn.Sequential(nn.SiLU(), ReplicatedLinear(dim, dim * 6))
        self.blocks = nn.ModuleList(
            [DiTBlock(dim, num_heads, ffn_dim, eps) for _ in range(num_layers)]
        )
        self.head = Head(dim, out_dim, patch_size, eps)
        self.num_heads = num_heads
        self.freqs = None
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
        if self.vae_type == "dac":
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
            if isinstance(x, list):
                x = [u + v for u, v in zip(x, y_camera)]
                x = x[0].unsqueeze(0)
            else:
                if isinstance(y_camera, list):
                    x = x + y_camera[0]
                else:
                    x = x + y_camera
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f -> b f c").contiguous()
        return x, grid_size  # x, grid_size: (f)

    def unpatchify(self, x: torch.Tensor, grid_size: tuple[int]):
        return rearrange(
            x, "b f (p c) -> b c (f p)", f=grid_size[0], p=self.patch_size[0]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        guidance=None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        # MOVA audio uses x/context naming historically.
        x = hidden_states
        context = (
            encoder_hidden_states[0]
            if isinstance(encoder_hidden_states, list)
            else encoder_hidden_states
        )

        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_proj, _ = self.time_projection(t)
        t_mod = t_proj.unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

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
