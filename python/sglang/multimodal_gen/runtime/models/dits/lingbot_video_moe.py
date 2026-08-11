# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Any, Iterable, Iterator, Optional

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from torch import nn

from sglang.multimodal_gen.configs.models.dits.lingbot_video_moe import (
    LingBotVideoMoEConfig,
)
from sglang.multimodal_gen.runtime.distributed import divide, get_tp_world_size
from sglang.multimodal_gen.runtime.layers.attention import (
    USPAttention,
    build_varlen_mask_meta,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.moe import (
    LingBotVideoMLP,
    LingBotVideoSparseMoeBlock,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
    NDRotaryEmbedding,
    _apply_rotary_emb,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.srt.utils import add_prefix

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


def is_lingbot_block(name: str, _module: object) -> bool:
    return "blocks" in name and name.split(".")[-1].isdigit()


def should_keep_in_fp32(name: str) -> bool:
    return any(
        module_name in name.split(".") for module_name in LINGBOT_VIDEO_FP32_MODULES
    )


class LingBotVideoRMSNorm(nn.Module):
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


def make_joint_position_ids(
    text_len: int, grid_t: int, grid_h: int, grid_w: int, device: torch.device
) -> torch.Tensor:
    tt = torch.arange(grid_t, device=device, dtype=torch.int32) + (text_len + 1)
    hh = torch.arange(grid_h, device=device, dtype=torch.int32)
    ww = torch.arange(grid_w, device=device, dtype=torch.int32)
    grid = torch.stack(torch.meshgrid(tt, hh, ww, indexing="ij"), dim=-1).flatten(0, 2)
    text_t = torch.arange(text_len, device=device, dtype=torch.int32) + 1
    text_pos = torch.stack(
        [text_t, torch.zeros_like(text_t), torch.zeros_like(text_t)], dim=-1
    )
    return torch.cat([grid, text_pos], dim=0)  # (Nx + L, 3)


def _joint_position_ids(
    text_lens: torch.Tensor,
    grid_t: int,
    grid_h: int,
    grid_w: int,
    text_len_padded: int,
    device: torch.device,
) -> torch.Tensor:
    # Joint video;text positions for rotary_emb; on-device, padding masked in attention.
    B = text_lens.shape[0]
    n_video = grid_t * grid_h * grid_w
    seq_len = n_video + text_len_padded
    text_lens_i = text_lens.to(torch.int32)
    tt = torch.arange(grid_t, device=device, dtype=torch.int32)
    hh = torch.arange(grid_h, device=device, dtype=torch.int32)
    ww = torch.arange(grid_w, device=device, dtype=torch.int32)
    video_t = (text_lens_i + 1)[:, None] + tt[None, :]
    t_g = video_t[:, :, None, None].expand(B, grid_t, grid_h, grid_w)
    h_g = hh[None, None, :, None].expand(B, grid_t, grid_h, grid_w)
    w_g = ww[None, None, None, :].expand(B, grid_t, grid_h, grid_w)
    video_pos = torch.stack([t_g, h_g, w_g], dim=-1).reshape(B, n_video, 3)
    text_t = (
        torch.arange(text_len_padded, device=device, dtype=torch.int32)[None, :] + 1
    )
    real = (
        torch.arange(text_len_padded, device=device, dtype=torch.int32)[None, :]
        < text_lens_i[:, None]
    )
    text_t = torch.where(real, text_t, torch.zeros_like(text_t))
    text_pos = torch.stack(
        [text_t, torch.zeros_like(text_t), torch.zeros_like(text_t)], dim=-1
    )
    return torch.cat([video_pos, text_pos], dim=1).reshape(B * seq_len, 3)


class LingBotVideoTextEmbedder(nn.Module):
    def __init__(self, text_dim: int, hidden_size: int):
        super().__init__()
        self.norm = LingBotVideoRMSNorm(text_dim, eps=1e-6)
        self.linear_1 = nn.Linear(text_dim, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.linear_2(F.silu(self.linear_1(x)))


class LingBotVideoAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        norm_eps: float,
        qkv_bias: bool,
        out_bias: bool,
        prefix: str = "",
        supported_attention_backends: Optional[set[AttentionBackendEnum]] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        tp_size = get_tp_world_size()
        self.local_num_heads = divide(num_heads, tp_size)

        self.to_q = ColumnParallelLinear(
            hidden_size,
            hidden_size,
            bias=qkv_bias,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("to_q", prefix),
        )
        self.to_k = ColumnParallelLinear(
            hidden_size,
            hidden_size,
            bias=qkv_bias,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("to_k", prefix),
        )
        self.to_v = ColumnParallelLinear(
            hidden_size,
            hidden_size,
            bias=qkv_bias,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("to_v", prefix),
        )
        self.norm_q = LingBotVideoRMSNorm(self.head_dim, norm_eps)
        self.norm_k = LingBotVideoRMSNorm(self.head_dim, norm_eps)
        self.to_out = RowParallelLinear(
            hidden_size,
            hidden_size,
            bias=out_bias,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=add_prefix("to_out", prefix),
        )
        self.attn = USPAttention(
            num_heads=self.local_num_heads,
            head_size=self.head_dim,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            skip_sequence_parallel=False,
            quant_config=quant_config,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        attn_mask_meta: Optional[dict] = None,
    ) -> torch.Tensor:
        cos, sin = freqs_cis
        q, _ = self.to_q(x)
        k, _ = self.to_k(x)
        v, _ = self.to_v(x)
        q = self.norm_q(q.unflatten(2, (self.local_num_heads, self.head_dim)))
        k = self.norm_k(k.unflatten(2, (self.local_num_heads, self.head_dim)))
        v = v.unflatten(2, (self.local_num_heads, self.head_dim))

        B, S, H, D = q.shape
        # RoPE over the flattened batch; one batched call, the key mask isolates samples.
        q = _apply_rotary_emb(
            q.reshape(1, B * S, H, D), cos, sin, is_neox_style=False
        ).reshape(B, S, H, D)
        k = _apply_rotary_emb(
            k.reshape(1, B * S, H, D), cos, sin, is_neox_style=False
        ).reshape(B, S, H, D)
        out = self.attn(
            q,
            k,
            v,
            attn_mask=attention_mask,
            attn_mask_meta=attn_mask_meta,
        )
        out = out.flatten(2)
        out, _ = self.to_out(out)
        return out


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
        prefix: str = "",
        supported_attention_backends: Optional[set[AttentionBackendEnum]] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        h = hidden_size
        self.scale_shift_table = nn.Parameter(torch.zeros(1, 6 * h))
        self.norm1 = LingBotVideoRMSNorm(h, norm_eps)
        self.attn = LingBotVideoAttention(
            h,
            num_attention_heads,
            norm_eps,
            qkv_bias,
            out_bias,
            prefix=add_prefix("attn", prefix),
            supported_attention_backends=supported_attention_backends,
            quant_config=quant_config,
        )
        self.norm_post_attn = LingBotVideoRMSNorm(h, norm_eps)
        self.norm2 = LingBotVideoRMSNorm(h, norm_eps)
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
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        attn_mask_meta: Optional[dict] = None,
    ) -> torch.Tensor:
        expected_tokens = x.shape[0] * x.shape[1]
        if temb6.ndim != 2 or temb6.shape[0] != expected_tokens:
            raise ValueError(
                "LingBotVideoBlock expects token-level temb6 with shape "
                f"(B*S, 6D); got {tuple(temb6.shape)} for hidden states {tuple(x.shape)}."
            )
        mod = temb6.view(x.shape[0], x.shape[1], -1) + self.scale_shift_table.unsqueeze(
            0
        )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(
            6, dim=-1
        )
        gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
        scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

        bulk_dtype = self.attn.to_q.weight.dtype
        attn_in = (self.norm1(x) * scale_msa + shift_msa).to(bulk_dtype)
        attn_out = self.attn(
            attn_in,
            freqs_cis,
            attention_mask=attention_mask,
            attn_mask_meta=attn_mask_meta,
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

    _fsdp_shard_conditions = [is_lingbot_block]
    _compile_conditions = [is_lingbot_block]
    param_names_mapping = LingBotVideoMoEConfig().param_names_mapping
    reverse_param_names_mapping = LingBotVideoMoEConfig().reverse_param_names_mapping
    lora_param_names_mapping = LingBotVideoMoEConfig().lora_param_names_mapping

    def to(self, *args, **kwargs):
        # _parse_to is private but the only exact parser for .to() overloads.
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

    def preprocess_loaded_state_dict(
        self, weight_iterator: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterator[tuple[str, torch.Tensor]]:
        # Pack experts.w1+w3 into experts.w13_weight, gate then up on dim 1; w2 passes through.
        seen: dict[str, list[torch.Tensor | None]] = {}
        for name, tensor in weight_iterator:
            suffix = next(
                (s for s in (".ffn.experts.w1", ".ffn.experts.w3") if name.endswith(s)),
                None,
            )
            if suffix is None:
                yield name, tensor
                continue
            prefix = name[: -len(suffix)]
            pair = seen.setdefault(prefix, [None, None])
            pair[0 if suffix.endswith(".w1") else 1] = tensor
            if pair[0] is not None and pair[1] is not None:
                yield f"{prefix}.ffn.experts.w13_weight", torch.cat(pair, dim=1)
                del seen[prefix]

    def __init__(
        self,
        config: LingBotVideoMoEConfig,
        hf_config: dict[str, Any],
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config)

        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        head_dim = hidden_size // num_attention_heads
        assert head_dim == sum(
            config.axes_dims
        ), f"head_dim {head_dim} != sum(axes_dims) {sum(config.axes_dims)}"
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
        self.rotary_emb = NDRotaryEmbedding(
            rope_dim_list=list(config.axes_dims),
            rope_theta=config.rope_theta,
            dtype=(
                torch.float64
                if current_platform.is_float64_supported()
                else torch.float32
            ),
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
                    prefix=f"blocks.{i}",
                    supported_attention_backends=self._supported_attention_backends,
                    quant_config=quant_config,
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
        encoder_attention_mask: Optional[torch.Tensor] = None,  # (B, L) 1=valid
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

        positions = _joint_position_ids(text_lens, gt, gh, gw, L, device)
        cos, sin = self.rotary_emb.forward_uncached(positions)
        freqs_cis = (cos.float(), sin.float())

        attention_mask = attn_mask_meta = None
        # B==1 text is trimmed to true length upstream, so no mask; B>1 may pad, build a key mask.
        if B > 1 and encoder_attention_mask is not None:
            attention_mask = torch.cat(
                [
                    torch.ones(B, n_video, dtype=torch.bool, device=device),
                    encoder_attention_mask.bool(),
                ],
                dim=1,
            )
            attn_mask_meta = build_varlen_mask_meta(attention_mask)

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
                freqs_cis,
                attention_mask,
                attn_mask_meta,
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

        Cout = self.out_channels
        x = x.reshape(B, gt, gh, gw, pF, pH, pW, Cout)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, Cout, T, H, W)
        return x


EntryClass = [LingBotVideoTransformer3DModel]
