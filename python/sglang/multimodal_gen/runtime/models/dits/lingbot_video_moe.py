# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project.
# Adapted from LingBot-Video (https://github.com/Robbyant/lingbot-video).

"""SGLang-native LingBot-Video MoE 30B DiT (text-to-video, Phase-1 single-GPU MVP).

Ports the upstream ``LingBotVideoTransformer3DModel`` math **verbatim** (parity-critical):
fp32-sensitive boundaries (``LINGBOT_VIDEO_FP32_MODULES``), complex64 RoPE
(``torch.polar``/``view_as_complex``), the patchify permute ``(0,2,4,6,3,5,7,1)``
and unpatchify permute ``(0,7,1,4,2,5,3,6)``, and the DeepSeek-V3-style MoE FFN.

The MoE layer (``LingBotVideoSparseMoeBlock`` etc.) is imported from
``runtime.layers.moe`` and is NOT redefined here.

MVP scope: single-GPU, B=1, structured-JSON captions. The packed-B>1,
context-parallel (``cp_*``), ``packed_indices``/``flash_attn_varlen`` and
``parallel_config`` paths are dropped. Attention uses
``F.scaled_dot_product_attention`` (B=1, no padding -> ``attention_mask=None``).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from sglang.multimodal_gen.configs.models.dits.lingbot_video_moe import (
    LingBotVideoMoEConfig,
)
from sglang.multimodal_gen.runtime.layers.moe import (
    LingBotVideoMLP,
    LingBotVideoSparseMoeBlock,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from torch import nn

# ---------------------------------------------------------------------------
# fp32 boundaries (parity-critical: router / norms / scale_shift_table / time
# embedder must stay fp32; cast to the bulk compute dtype only at Linear
# boundaries). Ported verbatim from the upstream module.
# ---------------------------------------------------------------------------

LINGBOT_VIDEO_FP32_MODULES = (
    "time_embedder",
    "time_modulation",
    "scale_shift_table",
    "norm",
    "norm1",
    "norm2",
    "norm_q",
    "norm_k",
    "norm_post_attn",
    "norm_post_ffn",
    "norm_out",
    "norm_out_modulation",
    "router",
)


def should_keep_in_fp32(name: str) -> bool:
    return any(
        module_name in name.split(".") for module_name in LINGBOT_VIDEO_FP32_MODULES
    )


class LingBotVideoRMSNorm(nn.Module):
    """RMSNorm with fp32 accumulation."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply complex RoPE to ``(B, S, H, D)`` attention tensors."""
    with torch.amp.autocast("cuda", enabled=False):
        x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        out = torch.view_as_real(x_c * freqs_cis.unsqueeze(2)).flatten(3)
        return out.type_as(x)


class LingBotVideoRotaryEmbedding(nn.Module):
    """Complex64 RoPE table indexed by position ids."""

    def __init__(
        self, axes_dims: tuple[int, ...], axes_lens: tuple[int, ...], theta: float
    ):
        super().__init__()
        self.axes_dims = tuple(axes_dims)
        self.axes_lens = list(axes_lens)
        self.theta = theta
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim: tuple[int, ...], end: tuple[int, ...], theta: float):
        freqs_cis = []
        for d, e in zip(dim, end):
            freqs = 1.0 / (
                theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d)
            )
            timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
            freqs = torch.outer(timestep, freqs).float()
            freqs_cis.append(
                torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)
            )
        return freqs_cis

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        # position_ids: (S, 3) int -> (S, head_dim/2) complex64
        device = position_ids.device
        max_vals = position_ids.max(dim=0).values.tolist()
        needs_rebuild = self.freqs_cis is None or any(
            m >= l for m, l in zip(max_vals, self.axes_lens)
        )
        if needs_rebuild:
            for i in range(len(self.axes_lens)):
                if max_vals[i] >= self.axes_lens[i]:
                    self.axes_lens[i] = int(max_vals[i] * 1.5) + 1
            self.freqs_cis = self.precompute_freqs_cis(
                self.axes_dims, tuple(self.axes_lens), theta=self.theta
            )
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]
        elif self.freqs_cis[0].device != device:
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]

        return torch.cat(
            [self.freqs_cis[i][position_ids[:, i]] for i in range(len(self.axes_dims))],
            dim=-1,
        )


def make_joint_position_ids(
    text_len: int, grid_t: int, grid_h: int, grid_w: int, device: torch.device
) -> torch.Tensor:
    """3D positions in [video; text] order. Text t-axis is 1..text_len; video t-axis starts at text_len+1.

    Matches patchify_and_embed: cap start (1,0,0); vision start (cap_len+1,0,0);
    freqs ordered with x first and cap second (same order as cat_interleave).
    """
    tt = torch.arange(grid_t, device=device, dtype=torch.int32) + (text_len + 1)
    hh = torch.arange(grid_h, device=device, dtype=torch.int32)
    ww = torch.arange(grid_w, device=device, dtype=torch.int32)
    grid = torch.stack(torch.meshgrid(tt, hh, ww, indexing="ij"), dim=-1).flatten(0, 2)
    text_t = torch.arange(text_len, device=device, dtype=torch.int32) + 1
    text_pos = torch.stack(
        [text_t, torch.zeros_like(text_t), torch.zeros_like(text_t)], dim=-1
    )
    return torch.cat([grid, text_pos], dim=0)  # (Nx + L, 3)


class LingBotVideoTextEmbedder(nn.Module):
    """Matches CondProjection: RMSNorm(text_dim, eps=1e-6 fixed) -> Linear-SiLU-Linear."""

    def __init__(self, text_dim: int, hidden_size: int):
        super().__init__()
        self.norm = LingBotVideoRMSNorm(text_dim, eps=1e-6)
        self.linear_1 = nn.Linear(text_dim, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.linear_2(F.silu(self.linear_1(x)))


class LingBotVideoAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, norm_eps, qkv_bias, out_bias):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.norm_q = LingBotVideoRMSNorm(self.head_dim, norm_eps)
        self.norm_k = LingBotVideoRMSNorm(self.head_dim, norm_eps)
        self.to_out = nn.Linear(hidden_size, hidden_size, bias=out_bias)

    def forward(
        self,
        x: torch.Tensor,
        rotary_emb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, _, _ = x.shape
        q = self.to_q(x).unflatten(2, (self.num_heads, self.head_dim))
        k = self.to_k(x).unflatten(2, (self.num_heads, self.head_dim))
        v = self.to_v(x).unflatten(2, (self.num_heads, self.head_dim))
        q = apply_rotary_emb(self.norm_q(q), rotary_emb)
        k = apply_rotary_emb(self.norm_k(k), rotary_emb)
        # SDPA expects (B, H, S, D); the upstream tensors are (B, S, H, D).
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attention_mask,
        )
        out = out.transpose(1, 2)  # (B, S, H, D)
        return self.to_out(out.flatten(2, 3).type_as(x))


class LingBotVideoBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        intermediate_size,
        norm_eps,
        qkv_bias,
        out_bias,
        num_experts,
        num_experts_per_tok,
        moe_intermediate_size,
        decoder_sparse_step,
        mlp_only_layers,
        n_shared_experts,
        score_func,
        norm_topk_prob,
        n_group,
        topk_group,
        routed_scaling_factor,
        layer_idx: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        h = hidden_size
        self.scale_shift_table = nn.Parameter(torch.zeros(1, 6 * h))
        self.norm1 = LingBotVideoRMSNorm(h, norm_eps)
        self.attn = LingBotVideoAttention(
            h, num_attention_heads, norm_eps, qkv_bias, out_bias
        )
        self.norm_post_attn = LingBotVideoRMSNorm(h, norm_eps)
        self.norm2 = LingBotVideoRMSNorm(h, norm_eps)
        # Sparsity decision matches MoEBlock: mlp_only_layers + decoder_sparse_step + num_experts
        if layer_idx not in mlp_only_layers and (
            num_experts > 0 and (layer_idx + 1) % decoder_sparse_step == 0
        ):
            self.ffn = LingBotVideoSparseMoeBlock(
                hidden_size=h,
                intermediate_size=moe_intermediate_size,
                num_experts=num_experts,
                top_k=num_experts_per_tok,
                score_func=score_func,
                norm_topk_prob=norm_topk_prob,
                n_group=n_group,
                topk_group=topk_group,
                routed_scaling_factor=routed_scaling_factor,
                n_shared_experts=n_shared_experts,
            )
        else:
            self.ffn = LingBotVideoMLP(h, intermediate_size)
        self.norm_post_ffn = LingBotVideoRMSNorm(h, norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        temb6: torch.Tensor,
        rotary_emb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expected_tokens = x.shape[0] * x.shape[1]
        if temb6.ndim != 2 or temb6.shape[0] != expected_tokens:
            raise ValueError(
                "LingBotVideoBlock expects token-level temb6 with shape "
                f"(B*S, 6D); got {tuple(temb6.shape)} for hidden states {tuple(x.shape)}."
            )
        # AdaLN mod: dense and MoE both keep scale_shift_table fp32 (master
        # moe/models.py:80 dropped the accidental `.to(dtype=c.dtype)` cast).
        mod = temb6.view(x.shape[0], x.shape[1], -1) + self.scale_shift_table.unsqueeze(
            0
        )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(
            6, dim=-1
        )
        gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
        scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

        # AdaLN modulation / norms run in fp32 (sensitive path); cast to the bulk
        # compute dtype only at the bf16 Linear boundary. This replaces the old
        # ambient autocast, which rounded Linear inputs to bf16 at the same point.
        bulk_dtype = self.attn.to_q.weight.dtype
        attn_in = (self.norm1(x) * scale_msa + shift_msa).to(bulk_dtype)
        attn_out = self.attn(
            attn_in,
            rotary_emb,
            attention_mask,
        )
        x = x + (gate_msa * self.norm_post_attn(attn_out)).to(x.dtype)

        ffn_in = (self.norm2(x) * scale_mlp + shift_mlp).to(bulk_dtype)
        ffn_out = self.ffn(ffn_in)
        ffn_normed = self.norm_post_ffn(ffn_out)
        x = x + (gate_mlp * ffn_normed).to(x.dtype)
        return x


class LingBotVideoTransformer3DModel(CachableDiT, LayerwiseOffloadableModuleMixin):
    _no_split_modules = ("LingBotVideoBlock",)
    _keep_in_fp32_modules = tuple(LINGBOT_VIDEO_FP32_MODULES)

    _fsdp_shard_conditions = LingBotVideoMoEConfig()._fsdp_shard_conditions
    _compile_conditions = LingBotVideoMoEConfig()._compile_conditions
    _supported_attention_backends = (
        LingBotVideoMoEConfig()._supported_attention_backends
    )
    param_names_mapping = LingBotVideoMoEConfig().param_names_mapping
    reverse_param_names_mapping = LingBotVideoMoEConfig().reverse_param_names_mapping
    lora_param_names_mapping = LingBotVideoMoEConfig().lora_param_names_mapping

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is None or dtype == torch.float32:
            return super().to(*args, **kwargs)

        dtype_is_floating = torch.is_floating_point(torch.empty((), dtype=dtype))
        if not dtype_is_floating:
            return super().to(*args, **kwargs)

        if device is not None:
            super().to(device=device, non_blocking=non_blocking)

        for name, param in self.named_parameters():
            if not torch.is_floating_point(param):
                continue
            target_dtype = torch.float32 if should_keep_in_fp32(name) else dtype
            param.data = param.data.to(dtype=target_dtype, non_blocking=non_blocking)
            if param.grad is not None:
                param.grad.data = param.grad.data.to(
                    dtype=target_dtype, non_blocking=non_blocking
                )

        for name, buffer in self.named_buffers():
            if not torch.is_floating_point(buffer):
                continue
            target_dtype = torch.float32 if should_keep_in_fp32(name) else dtype
            buffer.data = buffer.data.to(dtype=target_dtype, non_blocking=non_blocking)

        return self

    def __init__(
        self,
        config: LingBotVideoMoEConfig,
        hf_config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config)

        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        head_dim = hidden_size // num_attention_heads
        assert head_dim == sum(config.axes_dims), (
            f"head_dim {head_dim} != sum(axes_dims) {sum(config.axes_dims)}"
        )
        mlp_only_layers = tuple(config.mlp_only_layers)

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.num_channels_latents = config.out_channels
        self.patch_size = config.patch_size

        self.patch_embedder = nn.Linear(
            config.in_channels * math.prod(config.patch_size),
            hidden_size,
            bias=config.patch_embed_bias,
        )
        self.time_proj = Timesteps(
            config.freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedder = TimestepEmbedding(
            config.freq_dim,
            hidden_size,
            act_fn="silu",
            sample_proj_bias=config.timestep_mlp_bias,
        )
        self.time_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size)
        )
        self.text_embedder = LingBotVideoTextEmbedder(config.text_dim, hidden_size)
        self.rope = LingBotVideoRotaryEmbedding(
            config.axes_dims, config.axes_lens, config.rope_theta
        )
        self.blocks = nn.ModuleList(
            [
                LingBotVideoBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    norm_eps=config.norm_eps,
                    qkv_bias=config.qkv_bias,
                    out_bias=config.out_bias,
                    num_experts=config.num_experts,
                    num_experts_per_tok=config.num_experts_per_tok,
                    moe_intermediate_size=config.moe_intermediate_size,
                    decoder_sparse_step=config.decoder_sparse_step,
                    mlp_only_layers=mlp_only_layers,
                    n_shared_experts=config.n_shared_experts,
                    score_func=config.score_func,
                    norm_topk_prob=config.norm_topk_prob,
                    n_group=config.n_group,
                    topk_group=config.topk_group,
                    routed_scaling_factor=config.routed_scaling_factor,
                    layer_idx=i,
                )
                for i in range(config.depth)
            ]
        )
        self.norm_out = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=config.norm_eps
        )
        self.norm_out_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size)
        )
        self.proj_out = nn.Linear(
            hidden_size, math.prod(config.patch_size) * config.out_channels
        )

        self.__post_init__()
        self.layer_names = ["blocks"]

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, C, T, H, W)
        timestep: torch.Tensor,  # (B,) in [0, 1000] (= sigma*1000)
        encoder_hidden_states: torch.Tensor,  # (B, L, text_dim)
        encoder_attention_mask: torch.Tensor | None = None,  # (B, L) 1=valid
        **kwargs,
    ) -> torch.Tensor:
        B, C, T, H, W = hidden_states.shape
        pF, pH, pW = self.patch_size
        gt, gh, gw = T // pF, H // pH, W // pW
        n_video = gt * gh * gw
        L = encoder_hidden_states.shape[1]
        device = hidden_states.device
        if encoder_attention_mask is not None:
            text_lens = encoder_attention_mask.sum(dim=-1).long()
        else:
            text_lens = torch.full((B,), L, dtype=torch.long, device=device)
        text_lens_list = [int(v) for v in text_lens.detach().cpu().tolist()]

        # patchify: token order (f h w), feature order (pf ph pw c) -- matches patchify_and_embed
        patch_tokens = hidden_states.reshape(B, C, gt, pF, gh, pH, gw, pW)
        patch_tokens = patch_tokens.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(
            B,
            n_video,
            pF * pH * pW * C,
        )
        x = self.patch_embedder(patch_tokens)

        text = self.text_embedder(encoder_hidden_states)
        joint = torch.cat([x, text], dim=1)  # [video; text]
        joint_seq_len = joint.shape[1]

        # Per-sample RoPE: video t-axis start = real text length of this sample + 1
        rotary_parts = [
            self.rope(make_joint_position_ids(text_lens_list[i], gt, gh, gw, device))
            for i in range(B)
        ]
        rotary = torch.stack(rotary_parts, dim=0)  # (B, S, head_dim/2) complex64

        # Attention mask: only materialize when there is padding (B=1, no padding
        # -> None, matching the upstream default path).
        attention_mask = None
        has_padding = encoder_attention_mask is not None and bool((text_lens < L).any())
        if has_padding:
            key_mask = torch.cat(
                [
                    torch.ones(B, n_video, dtype=torch.bool, device=device),
                    encoder_attention_mask.bool(),
                ],
                dim=1,
            )
            attention_mask = key_mask[:, None, None, :]  # (B,1,1,S) -> SDPA broadcast

        timestep_for_embed = timestep.float()
        timestep_proj = self.time_proj(timestep_for_embed)
        t_emb = self.time_embedder(timestep_proj)  # (B, D)
        temb_input = t_emb.unsqueeze(1).expand(B, joint_seq_len, -1)  # (B, S, D)
        temb6 = self.time_modulation(temb_input.reshape(B * joint_seq_len, -1))
        temb6 = temb6.reshape(B, joint_seq_len, -1)  # (B, S, 6D)
        temb6 = temb6.reshape(temb6.shape[0] * temb6.shape[1], -1)

        for block in self.blocks:
            joint = block(
                joint,
                temb6,
                rotary,
                attention_mask,
            )

        final_mod = self.norm_out_modulation(
            temb_input.reshape(joint.shape[0] * joint.shape[1], -1)
        )
        shift, scale = final_mod.reshape(joint.shape[0], joint.shape[1], -1).chunk(
            2, dim=-1
        )
        final_hidden = self.norm_out(joint) * (1.0 + scale) + shift
        projected = self.proj_out(final_hidden.to(self.proj_out.weight.dtype))
        x = projected[:, :n_video]

        # unpatchify (matches the rearrange in postprocess)
        Cout = self.out_channels
        x = x.reshape(B, gt, gh, gw, pF, pH, pW, Cout)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, Cout, T, H, W)
        return x


EntryClass = [LingBotVideoTransformer3DModel]
