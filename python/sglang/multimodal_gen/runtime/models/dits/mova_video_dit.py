# Copied and adapted from: mossVG/mova/diffusion/models/wan_video_dit.py
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This module shares common functions (sinusoidal_embedding_1d, precompute_freqs_cis, etc.)
# with wanvideo.py. These functions are kept here for MOVA-specific model architecture,
# but could be refactored to a common module in the future.

import math
from typing import Any, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.distributed.tensor import DTensor

from sglang.multimodal_gen.configs.models.dits.mova_video import MOVAVideoConfig
from sglang.multimodal_gen.runtime.distributed import get_tp_world_size
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention, USPAttention

# Reuse SGLang's optimized RMSNorm instead of torch.nn.RMSNorm or custom SlowRMSNorm
from sglang.multimodal_gen.runtime.layers.layernorm import (
    RMSNorm,
    tensor_parallel_rms_norm,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.mlp import MLP
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# @torch.compile(fullgraph=True)
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(
                dim // 2
            ),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(
    dim: int, end: int = 1024, theta: float = 10000.0, s: float = 1.0
):
    # 1d rope precompute
    # Note: s parameter is used for audio-specific scaling (e.g., tps adjustment)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    pos = torch.arange(end, dtype=torch.float64, device=freqs.device) * s
    freqs = torch.outer(pos, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(
        x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    )
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


def rope_apply_head_dim(x, freqs, head_dim):
    x = rearrange(x, "b s (n d) -> b s n d", d=head_dim)
    x_out = torch.view_as_complex(
        x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    )
    # print(f"{x_out.shape = }, {freqs.shape = }")
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class SelfAttention(nn.Module):
    """
    Self-Attention module for MOVA DiT with Sequence Parallelism support.

    SP is handled at the pipeline level (latents are pre-sharded before DiT forward).
    USPAttention internally handles the all-to-all communication for distributed attention.
    Input x should already be the local shard [B, S_local, D] when SP is enabled.
    """

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.tp_size = get_tp_world_size()
        if self.num_heads % self.tp_size != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by tp_size ({self.tp_size})."
            )
        self.num_heads_per_rank = self.num_heads // self.tp_size

        # TP strategy: shard Q/K/V over heads (column-parallel), then row-parallel output.
        self.q = ColumnParallelLinear(dim, dim, bias=True, gather_output=False)
        self.k = ColumnParallelLinear(dim, dim, bias=True, gather_output=False)
        self.v = ColumnParallelLinear(dim, dim, bias=True, gather_output=False)
        self.o = RowParallelLinear(dim, dim, bias=True, input_is_parallel=True)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = USPAttention(
            # Local heads per TP rank.
            num_heads=self.num_heads_per_rank,
            head_size=self.head_dim,
            causal=False,
            softmax_scale=None,
        )

    def forward(self, x, freqs):
        """
        Forward pass for self-attention.

        Args:
            x: Input tensor [B, S_local, D] - already sharded by SP when SP > 1
            freqs: RoPE frequencies [S_local, 1, head_dim] - should match x's sequence length

        Returns:
            Output tensor [B, S_local, D]
        """
        if isinstance(freqs, DTensor):
            freqs = freqs.to_local()

        # Compute Q, K, V on local sequence
        q, _ = self.q(x)
        k, _ = self.k(x)
        v, _ = self.v(x)

        # RMSNorm over sharded hidden dimension.
        if self.tp_size > 1:
            q = tensor_parallel_rms_norm(q, self.norm_q)
            k = tensor_parallel_rms_norm(k, self.norm_k)
        else:
            q = self.norm_q(q)
            k = self.norm_k(k)

        # Apply RoPE
        q = rope_apply_head_dim(q, freqs, self.head_dim)
        k = rope_apply_head_dim(k, freqs, self.head_dim)

        # USPAttention expects [B, S_local, H, D] format
        q = rearrange(q, "b s (n d) -> b s n d", n=self.num_heads_per_rank)
        k = rearrange(k, "b s (n d) -> b s n d", n=self.num_heads_per_rank)
        v = rearrange(v, "b s (n d) -> b s n d", n=self.num_heads_per_rank)

        # USPAttention handles SP communication internally
        out = self.attn(q, k, v)
        out = rearrange(out, "b s n d -> b s (n d)")

        out, _ = self.o(out)
        return out


class CrossAttention(nn.Module):
    """
    Cross-Attention module for MOVA DiT.

    Cross-attention does NOT require SP communication because:
    - Query comes from the main sequence (already sharded by SP)
    - Key/Value come from context (text embeddings, which are replicated across all ranks)

    Uses LocalAttention instead of USPAttention for efficiency.
    """

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.tp_size = get_tp_world_size()
        if self.num_heads % self.tp_size != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by tp_size ({self.tp_size})."
            )
        self.num_heads_per_rank = self.num_heads // self.tp_size

        self.q = ColumnParallelLinear(dim, dim, bias=True, gather_output=False)
        self.k = ColumnParallelLinear(dim, dim, bias=True, gather_output=False)
        self.v = ColumnParallelLinear(dim, dim, bias=True, gather_output=False)
        self.o = RowParallelLinear(dim, dim, bias=True, input_is_parallel=True)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        # Use LocalAttention for cross-attention (no SP communication needed)
        self.attn = LocalAttention(
            num_heads=self.num_heads_per_rank,
            head_size=self.head_dim,
            causal=False,
            softmax_scale=None,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Forward pass for cross-attention.

        Args:
            x: Query tensor [B, S_local, D] - the main sequence (sharded by SP)
            y: Context tensor [B, S_ctx, D] - text/image embeddings (replicated)

        Returns:
            Output tensor [B, S_local, D]
        """
        ctx = y

        q, _ = self.q(x)
        k, _ = self.k(ctx)
        v, _ = self.v(ctx)

        if self.tp_size > 1:
            q = tensor_parallel_rms_norm(q, self.norm_q)
            k = tensor_parallel_rms_norm(k, self.norm_k)
        else:
            q = self.norm_q(q)
            k = self.norm_k(k)

        q = rearrange(q, "b s (n d) -> b s n d", n=self.num_heads_per_rank)
        k = rearrange(k, "b s (n d) -> b s n d", n=self.num_heads_per_rank)
        v = rearrange(v, "b s (n d) -> b s n d", n=self.num_heads_per_rank)
        x = self.attn(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)")
        x, _ = self.o(x)
        return x


class GateModule(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = MLP(dim, ffn_dim, output_dim=dim, act_type="gelu_pytorch_tanh")
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class Head(nn.Module):
    def __init__(
        self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        # Output dim is small for MOVA; replicate to avoid TP shape coupling.
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
            shift, scale = (
                self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
            ).chunk(2, dim=1)
            x, _ = self.head(self.norm(x) * (1 + scale) + shift)
        return x


class Conv3dLocalIsland(nn.Conv3d):
    """
    Inherits from Conv3d and overrides the forward method.

    Key behaviors:
    - Parameters are kept as DTensor to maintain optimizer consistency.
    - The forward pass aggregates input, weight, and bias into a Replicate state,
      then performs the convolution locally using to_local().
    - The output is then redistributed as a DTensor (defaults to Replicate,
      but placements can be customized).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        if isinstance(input, DTensor):
            # NOTE: DTensor typing stubs are incomplete; at runtime DTensor has
            # to_local() and parameters may also be DTensor.
            x_local = input.to_local()  # type: ignore[attr-defined]
            w_local = self.weight.to_local()  # type: ignore[attr-defined]
            b_local = (
                self.bias.to_local() if self.bias is not None else None  # type: ignore[attr-defined]
            )

            return self._conv_forward(x_local, w_local, b_local)
        else:
            return super().forward(input)


class WanModel(CachableDiT, OffloadableDiTMixin):
    _fsdp_shard_conditions = MOVAVideoConfig()._fsdp_shard_conditions
    _compile_conditions = MOVAVideoConfig()._compile_conditions
    _supported_attention_backends = MOVAVideoConfig()._supported_attention_backends
    param_names_mapping = MOVAVideoConfig().param_names_mapping
    reverse_param_names_mapping = MOVAVideoConfig().reverse_param_names_mapping
    lora_param_names_mapping = MOVAVideoConfig().lora_param_names_mapping

    def __init__(self, config: MOVAVideoConfig, hf_config: dict[str, Any]) -> None:
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

        self.dim = dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents

        self.patch_embedding = Conv3dLocalIsland(
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
        self.freqs = precompute_freqs_cis_3d(head_dim)

    def patchify(
        self, x: torch.Tensor, control_camera_latents_input: torch.Tensor | None = None
    ):
        # NOTE(dhyu): avoid slow_conv
        x = x.contiguous(memory_format=torch.channels_last_3d)
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
                # Some adapters may return a list even when x is a Tensor.
                if isinstance(y_camera, list):
                    x = x + y_camera[0]
                else:
                    x = x + y_camera
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: tuple[int, int, int]):
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor],
        y: torch.Tensor,
        guidance=None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        # MOVA code historically uses x/context/y/clip_feature naming.
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

        x, (f, h, w) = self.patchify(x)

        freqs = (
            torch.cat(
                [
                    self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
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
        x = self.unpatchify(x, (f, h, w))
        return x


EntryClass = WanModel
