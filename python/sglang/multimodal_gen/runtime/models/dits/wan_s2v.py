# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import AdaLayerNorm
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange
from PIL import Image

from sglang.multimodal_gen.configs.models.dits import WanS2VConfig
from sglang.multimodal_gen.configs.models.dits.wan_s2v import (
    WAN_S2V_SAMPLE_NEG_PROMPT,
)
from sglang.multimodal_gen.runtime.distributed import (
    divide,
    get_sp_group,
    get_sp_world_size,
    get_tp_world_size,
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
)
from sglang.multimodal_gen.runtime.layers.attention.layer import (
    USPAttention,
    UlyssesAttention,
)
from sglang.multimodal_gen.runtime.layers.elementwise import MulAdd
from sglang.multimodal_gen.runtime.layers.layernorm import (
    FP32LayerNorm,
    LayerNormScaleShift,
    RMSNorm,
    tensor_parallel_rms_norm,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.mlp import MLP
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def sinusoidal_embedding_1d(dim, position):
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


@torch.amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


@torch.amp.autocast("cuda", enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n = x.size(2)
    output = []
    for i, _ in enumerate(x):
        seq_len = x.size(1)
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = freqs[i, :seq_len]
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).to(dtype=x.dtype)


@torch.amp.autocast("cuda", enabled=False)
def rope_precompute(x, grid_sizes, freqs, start=None):
    b, s, n, c = x.size(0), x.size(1), x.size(2), x.size(3) // 2
    if isinstance(freqs, list):
        trainable_freqs = freqs[1]
        freqs = freqs[0]
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = torch.view_as_complex(
        x.detach().reshape(b, s, n, -1, 2).to(torch.float64)
    )
    seq_bucket = [0]
    if not isinstance(grid_sizes, list):
        grid_sizes = [grid_sizes]
    for g in grid_sizes:
        if not isinstance(g, list):
            g = [torch.zeros_like(g), g]
        batch_size = g[0].shape[0]
        for i in range(batch_size):
            if start is None:
                f_o, h_o, w_o = g[0][i]
            else:
                f_o, h_o, w_o = start[i]
            f, h, w = g[1][i]
            t_f, t_h, t_w = g[2][i]
            seq_f, seq_h, seq_w = f - f_o, h - h_o, w - w_o
            seq_len = int(seq_f * seq_h * seq_w)
            if seq_len > 0:
                if t_f > 0:
                    if f_o >= 0:
                        f_sam = np.linspace(
                            f_o.item(), (t_f + f_o).item() - 1, seq_f
                        ).astype(int).tolist()
                    else:
                        f_sam = np.linspace(
                            -f_o.item(), (-t_f - f_o).item() + 1, seq_f
                        ).astype(int).tolist()
                    h_sam = np.linspace(
                        h_o.item(), (t_h + h_o).item() - 1, seq_h
                    ).astype(int).tolist()
                    w_sam = np.linspace(
                        w_o.item(), (t_w + w_o).item() - 1, seq_w
                    ).astype(int).tolist()
                    freqs_0 = (
                        freqs[0][f_sam]
                        if f_o >= 0
                        else freqs[0][f_sam].conj()
                    )
                    freqs_0 = freqs_0.view(seq_f, 1, 1, -1)
                    freqs_i = torch.cat(
                        [
                            freqs_0.expand(seq_f, seq_h, seq_w, -1),
                            freqs[1][h_sam]
                            .view(1, seq_h, 1, -1)
                            .expand(seq_f, seq_h, seq_w, -1),
                            freqs[2][w_sam]
                            .view(1, 1, seq_w, -1)
                            .expand(seq_f, seq_h, seq_w, -1),
                        ],
                        dim=-1,
                    ).reshape(seq_len, 1, -1)
                else:
                    freqs_i = trainable_freqs.unsqueeze(1)
                output[i, seq_bucket[-1] : seq_bucket[-1] + seq_len] = freqs_i
        seq_bucket.append(seq_bucket[-1] + seq_len)
    return output


class WanSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        skip_sequence_parallel: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.to_q = ColumnParallelLinear(
            dim, dim, gather_output=False, prefix="to_q"
        )
        self.to_k = ColumnParallelLinear(
            dim, dim, gather_output=False, prefix="to_k"
        )
        self.to_v = ColumnParallelLinear(
            dim, dim, gather_output=False, prefix="to_v"
        )
        self.to_out = RowParallelLinear(
            dim, dim, input_is_parallel=True, reduce_results=True, prefix="to_out"
        )
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.tp_rmsnorm = qk_norm and get_tp_world_size() > 1
        self.local_num_heads = divide(num_heads, get_tp_world_size())
        self.attn = USPAttention(
            num_heads=self.local_num_heads,
            head_size=self.head_dim,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            skip_sequence_parallel=skip_sequence_parallel,
        )
        self.ulysses_attn = UlyssesAttention(
            num_heads=self.local_num_heads,
            head_size=self.head_dim,
            causal=False,
            supported_attention_backends=supported_attention_backends,
        )

    def forward(self, x, seq_lens, grid_sizes, freqs, num_replicated_suffix=0):
        del seq_lens
        b, s, n, d = *x.shape[:2], self.local_num_heads, self.head_dim
        q, _ = self.to_q(x)
        if self.tp_rmsnorm:
            q = tensor_parallel_rms_norm(q, self.norm_q)
        else:
            q = self.norm_q(q)
        q = q.view(b, s, n, d)
        k, _ = self.to_k(x)
        if self.tp_rmsnorm:
            k = tensor_parallel_rms_norm(k, self.norm_k)
        else:
            k = self.norm_k(k)
        k = k.view(b, s, n, d)
        v, _ = self.to_v(x)
        v = v.view(b, s, n, d)
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)
        if (
            num_replicated_suffix > 0
            and get_sp_world_size() > 1
            and get_ring_parallel_world_size() == 1
        ):
            q_shard, q_rep = (
                q[:, :-num_replicated_suffix],
                q[:, -num_replicated_suffix:],
            )
            k_shard, k_rep = (
                k[:, :-num_replicated_suffix],
                k[:, -num_replicated_suffix:],
            )
            v_shard, v_rep = (
                v[:, :-num_replicated_suffix],
                v[:, -num_replicated_suffix:],
            )
            x, x_rep = self.ulysses_attn(
                q_shard,
                k_shard,
                v_shard,
                replicated_q=q_rep,
                replicated_k=k_rep,
                replicated_v=v_rep,
            )
            x = torch.cat([x, x_rep], dim=1)
        else:
            x = self.attn(
                q,
                k,
                v,
                num_replicated_suffix=num_replicated_suffix,
            )
        x, _ = self.to_out(x.flatten(2))
        return x


class WanCrossAttention(WanSelfAttention):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
    ):
        super().__init__(
            dim,
            num_heads,
            window_size,
            qk_norm,
            eps,
            supported_attention_backends=supported_attention_backends,
            skip_sequence_parallel=True,
        )

    def forward(self, x, context, context_lens):
        del context_lens
        b, n, d = x.size(0), self.local_num_heads, self.head_dim
        q, _ = self.to_q(x)
        if self.tp_rmsnorm:
            q = tensor_parallel_rms_norm(q, self.norm_q)
        else:
            q = self.norm_q(q)
        q = q.view(b, -1, n, d)
        k, _ = self.to_k(context)
        if self.tp_rmsnorm:
            k = tensor_parallel_rms_norm(k, self.norm_k)
        else:
            k = self.norm_k(k)
        k = k.view(b, -1, n, d)
        v, _ = self.to_v(context)
        v = v.view(b, -1, n, d)
        x = self.attn(q, k, v)
        x, _ = self.to_out(x.flatten(2))
        return x


class WanAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
    ):
        super().__init__()
        self.norm1 = LayerNormScaleShift(
            dim, eps=eps, elementwise_affine=False, dtype=torch.float32
        )
        self.self_attn = WanSelfAttention(
            dim,
            num_heads,
            window_size,
            qk_norm,
            eps,
            supported_attention_backends=supported_attention_backends,
        )
        self.norm3 = (
            FP32LayerNorm(dim, eps=eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = WanCrossAttention(
            dim,
            num_heads,
            (-1, -1),
            qk_norm,
            eps,
            supported_attention_backends=supported_attention_backends,
        )
        self.norm2 = LayerNormScaleShift(
            dim, eps=eps, elementwise_affine=False, dtype=torch.float32
        )
        self.ffn = MLP(dim, ffn_dim, dim, act_type="gelu_pytorch_tanh")
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.self_attn_residual = MulAdd()
        self.mlp_residual = MulAdd()

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        num_replicated_suffix=0,
    ):
        e = (self.modulation.float().unsqueeze(0) + e.float()).chunk(6, dim=2)
        y = self.self_attn(
            self.norm1(x, e[0].squeeze(2), e[1].squeeze(2)),
            seq_lens,
            grid_sizes,
            freqs,
            num_replicated_suffix=num_replicated_suffix,
        )
        x = self.self_attn_residual(y.float(), e[2].squeeze(2), x.float()).to(
            dtype=x.dtype
        )
        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        y = self.ffn(self.norm2(x, e[3].squeeze(2), e[4].squeeze(2)))
        x = self.mlp_residual(y.float(), e[5].squeeze(2), x.float()).to(dtype=x.dtype)
        return x


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        out_dim = math.prod(patch_size) * out_dim
        self.norm = FP32LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        e = (self.modulation.float().unsqueeze(0) + e.float().unsqueeze(2)).chunk(
            2, dim=2
        )
        return self.head(
            self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)
        )


class CausalConv1d(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size=3,
        stride=1,
        dilation=1,
        pad_mode="replicate",
        **kwargs,
    ):
        super().__init__()
        self.time_causal_padding = (kernel_size - 1, 0)
        self.pad_mode = pad_mode
        self.conv = nn.Conv1d(
            chan_in,
            chan_out,
            kernel_size,
            stride=stride,
            dilation=dilation,
            **kwargs,
        )

    def forward(self, x):
        return self.conv(
            nn.functional.pad(x, self.time_causal_padding, mode=self.pad_mode)
        )


class MotionEncoderTC(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_heads,
        need_global=True,
        dtype=None,
        device=None,
    ):
        super().__init__()
        factory_kwargs = {"dtype": dtype, "device": device}
        self.num_heads = num_heads
        self.need_global = need_global
        self.conv1_local = CausalConv1d(
            in_dim, hidden_dim // 4 * num_heads, 3, stride=1
        )
        if need_global:
            self.conv1_global = CausalConv1d(in_dim, hidden_dim // 4, 3, stride=1)
        self.norm1 = FP32LayerNorm(
            hidden_dim // 4, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.act = nn.SiLU()
        self.conv2 = CausalConv1d(hidden_dim // 4, hidden_dim // 2, 3, stride=2)
        self.conv3 = CausalConv1d(hidden_dim // 2, hidden_dim, 3, stride=2)
        if need_global:
            self.final_linear = nn.Linear(hidden_dim, hidden_dim, **factory_kwargs)
        self.norm2 = FP32LayerNorm(
            hidden_dim // 2, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.norm3 = FP32LayerNorm(
            hidden_dim, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))

    def forward(self, x):
        x = rearrange(x, "b t c -> b c t")
        x_ori = x.clone()
        b, _, _ = x.shape
        x = self.conv1_local(x)
        x = rearrange(x, "b (n c) t -> (b n) t c", n=self.num_heads)
        x = self.act(self.norm1(x))
        x = rearrange(x, "b t c -> b c t")
        x = self.conv2(x)
        x = self.act(self.norm2(rearrange(x, "b c t -> b t c")))
        x = rearrange(x, "b t c -> b c t")
        x = self.conv3(x)
        x = self.act(self.norm3(rearrange(x, "b c t -> b t c")))
        x = rearrange(x, "(b n) t c -> b t n c", b=b)
        x_local = torch.cat([x, self.padding_tokens.repeat(b, x.shape[1], 1, 1)], dim=-2)
        if not self.need_global:
            return x_local
        x = self.conv1_global(x_ori)
        x = self.act(self.norm1(rearrange(x, "b c t -> b t c")))
        x = rearrange(x, "b t c -> b c t")
        x = self.conv2(x)
        x = self.act(self.norm2(rearrange(x, "b c t -> b t c")))
        x = rearrange(x, "b t c -> b c t")
        x = self.conv3(x)
        x = self.act(self.norm3(rearrange(x, "b c t -> b t c")))
        x = self.final_linear(x)
        x = rearrange(x, "(b n) t c -> b t n c", b=b)
        return x, x_local


class CausalAudioEncoder(nn.Module):
    def __init__(
        self,
        dim=5120,
        num_layers=25,
        out_dim=2048,
        video_rate=8,
        num_token=4,
        need_global=False,
    ):
        super().__init__()
        self.encoder = MotionEncoderTC(
            in_dim=dim,
            hidden_dim=out_dim,
            num_heads=num_token,
            need_global=need_global,
        )
        self.weights = nn.Parameter(torch.ones((1, num_layers, 1, 1)) * 0.01)
        self.act = nn.SiLU()

    def forward(self, features):
        weights = self.act(self.weights.float())
        weighted_feat = (
            (features.float() * weights) / weights.sum(dim=1, keepdims=True)
        ).sum(dim=1)
        weighted_feat = weighted_feat.permute(0, 2, 1)
        return self.encoder(weighted_feat.to(dtype=features.dtype))


class AudioCrossAttention(WanCrossAttention):
    pass


class AudioInjectorWAN(nn.Module):
    def __init__(
        self,
        all_modules,
        all_module_names,
        dim=2048,
        num_heads=32,
        inject_layer=(0, 27),
        enable_adain=False,
        adain_dim=2048,
        need_adain_ont=False,
    ):
        super().__init__()
        self.injected_block_id = {}
        audio_injector_id = 0
        for mod_name, mod in zip(all_module_names, all_modules):
            if isinstance(mod, WanAttentionBlock):
                for inject_id in inject_layer:
                    if f"transformer_blocks.{inject_id}" in mod_name:
                        self.injected_block_id[inject_id] = audio_injector_id
                        audio_injector_id += 1
        self.injector = nn.ModuleList(
            [
                AudioCrossAttention(dim=dim, num_heads=num_heads, qk_norm=True)
                for _ in range(audio_injector_id)
            ]
        )
        self.injector_pre_norm_feat = nn.ModuleList(
            [
                FP32LayerNorm(dim, elementwise_affine=False, eps=1e-6)
                for _ in range(audio_injector_id)
            ]
        )
        if enable_adain:
            self.injector_adain_layers = nn.ModuleList(
                [
                    AdaLayerNorm(
                        output_dim=dim * 2, embedding_dim=adain_dim, chunk_dim=1
                    )
                    for _ in range(audio_injector_id)
                ]
            )
            if need_adain_ont:
                self.injector_adain_output_layers = nn.ModuleList(
                    [nn.Linear(dim, dim) for _ in range(audio_injector_id)]
                )


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def torch_dfs(model: nn.Module, parent_name="root"):
    module_names, modules = [parent_name if parent_name else "root"], [model]
    for name, child in model.named_children():
        child_name = f"{parent_name}.{name}" if parent_name else name
        child_modules, child_names = torch_dfs(child, child_name)
        module_names += child_names
        modules += child_modules
    return modules, module_names


class FramePackMotioner(nn.Module):
    def __init__(
        self,
        inner_dim=1024,
        num_heads=16,
        zip_frame_buckets=(1, 2, 16),
        drop_mode="drop",
    ):
        super().__init__()
        self.proj = nn.Conv3d(
            16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        self.proj_2x = nn.Conv3d(
            16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4)
        )
        self.proj_4x = nn.Conv3d(
            16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8)
        )
        self.zip_frame_buckets = torch.tensor(zip_frame_buckets, dtype=torch.long)
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        d = inner_dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )
        self.drop_mode = drop_mode

    def forward(self, motion_latents, add_last_motion=2):
        mot, mot_remb = [], []
        for m in motion_latents:
            lat_height, lat_width = m.shape[2], m.shape[3]
            padd_lat = torch.zeros(
                16,
                self.zip_frame_buckets.sum(),
                lat_height,
                lat_width,
                device=m.device,
                dtype=m.dtype,
            )
            overlap_frame = min(padd_lat.shape[1], m.shape[1])
            if overlap_frame > 0:
                padd_lat[:, -overlap_frame:] = m[:, -overlap_frame:]
            if add_last_motion < 2 and self.drop_mode != "drop":
                zero_end_frame = self.zip_frame_buckets[
                    : self.zip_frame_buckets.__len__() - add_last_motion - 1
                ].sum()
                padd_lat[:, -zero_end_frame:] = 0
            padd_lat = padd_lat.unsqueeze(0)
            clean_latents_4x, clean_latents_2x, clean_latents_post = padd_lat[
                :, :, -self.zip_frame_buckets.sum() :, :, :
            ].split(list(self.zip_frame_buckets)[::-1], dim=2)
            clean_latents_post = self.proj(clean_latents_post).flatten(2).transpose(1, 2)
            clean_latents_2x = self.proj_2x(clean_latents_2x).flatten(2).transpose(1, 2)
            clean_latents_4x = self.proj_4x(clean_latents_4x).flatten(2).transpose(1, 2)
            if add_last_motion < 2 and self.drop_mode == "drop":
                clean_latents_post = (
                    clean_latents_post[:, :0]
                    if add_last_motion < 2
                    else clean_latents_post
                )
                clean_latents_2x = (
                    clean_latents_2x[:, :0]
                    if add_last_motion < 1
                    else clean_latents_2x
                )
            motion_lat = torch.cat(
                [clean_latents_post, clean_latents_2x, clean_latents_4x], dim=1
            )
            start_time_id = -(self.zip_frame_buckets[:1].sum())
            end_time_id = start_time_id + self.zip_frame_buckets[0]
            grid_sizes = (
                []
                if add_last_motion < 2 and self.drop_mode == "drop"
                else [[
                    torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
                    torch.tensor([end_time_id, lat_height // 2, lat_width // 2])
                    .unsqueeze(0)
                    .repeat(1, 1),
                    torch.tensor([self.zip_frame_buckets[0], lat_height // 2, lat_width // 2])
                    .unsqueeze(0)
                    .repeat(1, 1),
                ]]
            )
            start_time_id = -(self.zip_frame_buckets[:2].sum())
            end_time_id = start_time_id + self.zip_frame_buckets[1] // 2
            grid_sizes_2x = (
                []
                if add_last_motion < 1 and self.drop_mode == "drop"
                else [[
                    torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
                    torch.tensor([end_time_id, lat_height // 4, lat_width // 4])
                    .unsqueeze(0)
                    .repeat(1, 1),
                    torch.tensor([self.zip_frame_buckets[1], lat_height // 2, lat_width // 2])
                    .unsqueeze(0)
                    .repeat(1, 1),
                ]]
            )
            start_time_id = -(self.zip_frame_buckets[:3].sum())
            end_time_id = start_time_id + self.zip_frame_buckets[2] // 4
            grid_sizes_4x = [[
                torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
                torch.tensor([end_time_id, lat_height // 8, lat_width // 8])
                .unsqueeze(0)
                .repeat(1, 1),
                torch.tensor([self.zip_frame_buckets[2], lat_height // 2, lat_width // 2])
                .unsqueeze(0)
                .repeat(1, 1),
            ]]
            motion_rope_emb = rope_precompute(
                motion_lat.detach().view(
                    1,
                    motion_lat.shape[1],
                    self.num_heads,
                    self.inner_dim // self.num_heads,
                ),
                grid_sizes + grid_sizes_2x + grid_sizes_4x,
                self.freqs,
                start=None,
            )
            mot.append(motion_lat)
            mot_remb.append(motion_rope_emb)
        return mot, mot_remb


class HeadS2V(Head):
    def forward(self, x, e):
        e = (self.modulation.float() + e.float().unsqueeze(1)).chunk(2, dim=1)
        x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class WanS2VSelfAttention(WanSelfAttention):
    pass


class WanS2VAttentionBlock(WanAttentionBlock):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
    ):
        super().__init__(
            dim,
            ffn_dim,
            num_heads,
            window_size,
            qk_norm,
            cross_attn_norm,
            eps,
            supported_attention_backends=supported_attention_backends,
        )
        self.self_attn = WanS2VSelfAttention(
            dim,
            num_heads,
            window_size,
            qk_norm,
            eps,
            supported_attention_backends=supported_attention_backends,
        )

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        num_replicated_suffix=0,
    ):
        seg_idx = min(max(0, e[1].item()), x.size(1))
        seg_idx = [0, seg_idx, x.size(1)]
        e = e[0]
        e = (self.modulation.float().unsqueeze(2) + e.float()).chunk(6, dim=1)
        e = [element.squeeze(1) for element in e]
        norm_x = torch.cat(
            [
                self.norm1(
                    x[:, seg_idx[i] : seg_idx[i + 1]],
                    e[0][:, i : i + 1],
                    e[1][:, i : i + 1],
                )
                for i in range(2)
            ],
            dim=1,
        )
        y = self.self_attn(
            norm_x,
            seq_lens,
            grid_sizes,
            freqs,
            num_replicated_suffix=num_replicated_suffix,
        )
        y = torch.cat(
            [
                y[:, seg_idx[i] : seg_idx[i + 1]].float() * e[2][:, i : i + 1]
                for i in range(2)
            ],
            dim=1,
        )
        x = x + y.to(dtype=x.dtype)
        cross = self.cross_attn(self.norm3(x), context, context_lens)
        x = x + cross
        norm2_x = torch.cat(
            [
                self.norm2(
                    x[:, seg_idx[i] : seg_idx[i + 1]],
                    e[3][:, i : i + 1],
                    e[4][:, i : i + 1],
                )
                for i in range(2)
            ],
            dim=1,
        )
        y = self.ffn(norm2_x)
        y = torch.cat(
            [
                y[:, seg_idx[i] : seg_idx[i + 1]].float() * e[5][:, i : i + 1]
                for i in range(2)
            ],
            dim=1,
        )
        x = x + y.to(dtype=x.dtype)
        return x


class WanModelS2V(ModelMixin, ConfigMixin):
    ignore_for_config = [
        "args",
        "kwargs",
        "patch_size",
        "cross_attn_norm",
        "qk_norm",
        "text_dim",
        "window_size",
    ]
    _no_split_modules = ["WanS2VAttentionBlock"]

    @register_to_config
    def __init__(
        self,
        cond_dim=0,
        audio_dim=5120,
        num_audio_token=4,
        enable_adain=False,
        adain_mode="attn_norm",
        audio_inject_layers=(0, 4, 8, 12, 16, 20, 24, 27),
        zero_init=False,
        zero_timestep=False,
        enable_motioner=False,
        add_last_motion=True,
        enable_tsm=False,
        trainable_token_pos_emb=False,
        motion_token_num=1024,
        enable_framepack=True,
        framepack_drop_mode="padd",
        model_type="s2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        *args,
        **kwargs,
    ):
        super().__init__()
        assert model_type == "s2v"
        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self._supported_attention_backends = {
            AttentionBackendEnum.FA,
            AttentionBackendEnum.TORCH_SDPA,
        }
        self.blocks = nn.ModuleList(
            [
                WanS2VAttentionBlock(
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    supported_attention_backends=self._supported_attention_backends,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = HeadS2V(dim, out_dim, patch_size, eps)
        d = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )
        self.init_weights()
        self.sp_size = get_sp_world_size()
        self.use_context_parallel = self.sp_size > 1
        if cond_dim > 0:
            self.cond_encoder = nn.Conv3d(
                cond_dim,
                self.dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
        self.enbale_adain = enable_adain
        self.casual_audio_encoder = CausalAudioEncoder(
            dim=audio_dim,
            out_dim=self.dim,
            num_token=num_audio_token,
            need_global=enable_adain,
        )
        all_modules, all_module_names = torch_dfs(
            self.blocks, parent_name="root.transformer_blocks"
        )
        self.audio_injector = AudioInjectorWAN(
            all_modules,
            all_module_names,
            dim=self.dim,
            num_heads=self.num_heads,
            inject_layer=list(audio_inject_layers),
            enable_adain=enable_adain,
            adain_dim=self.dim,
            need_adain_ont=adain_mode != "attn_norm",
        )
        self.adain_mode = adain_mode
        self.trainable_cond_mask = nn.Embedding(3, self.dim)
        if zero_init:
            self.zero_init_weights()
        self.zero_timestep = zero_timestep
        self.enable_motioner = enable_motioner
        self.add_last_motion = add_last_motion
        self.enable_framepack = enable_framepack
        if enable_motioner:
            raise NotImplementedError(
                "Native Wan S2V motioner path is not implemented"
            )
        if enable_framepack:
            self.frame_packer = FramePackMotioner(
                inner_dim=self.dim,
                num_heads=self.num_heads,
                zip_frame_buckets=[1, 2, 16],
                drop_mode=framepack_drop_mode,
            )

    def zero_init_weights(self):
        with torch.no_grad():
            self.trainable_cond_mask = zero_module(self.trainable_cond_mask)
            if hasattr(self, "cond_encoder"):
                self.cond_encoder = zero_module(self.cond_encoder)
            for i in range(len(self.audio_injector.injector)):
                self.audio_injector.injector[i].to_out = zero_module(
                    self.audio_injector.injector[i].to_out
                )
                if self.enbale_adain:
                    self.audio_injector.injector_adain_layers[i].linear = zero_module(
                        self.audio_injector.injector_adain_layers[i].linear
                    )

    def process_motion_frame_pack(
        self, motion_latents, drop_motion_frames=False, add_last_motion=2
    ):
        flattern_mot, mot_remb = self.frame_packer(motion_latents, add_last_motion)
        if drop_motion_frames:
            return [m[:, :0] for m in flattern_mot], [m[:, :0] for m in mot_remb]
        return flattern_mot, mot_remb

    def inject_motion(
        self,
        x,
        seq_lens,
        rope_embs,
        mask_input,
        motion_latents,
        drop_motion_frames=False,
        add_last_motion=True,
    ):
        mot, mot_remb = self.process_motion_frame_pack(
            motion_latents,
            drop_motion_frames=drop_motion_frames,
            add_last_motion=add_last_motion,
        )
        if len(mot) > 0:
            x = [torch.cat([u, m], dim=1) for u, m in zip(x, mot)]
            seq_lens = seq_lens + torch.tensor(
                [r.size(1) for r in mot], dtype=torch.long
            )
            rope_embs = [torch.cat([u, m], dim=1) for u, m in zip(rope_embs, mot_remb)]
            mask_input = [
                torch.cat(
                    [
                        m,
                        2
                        * torch.ones(
                            [1, u.shape[1] - m.shape[1]],
                            device=m.device,
                            dtype=m.dtype,
                        ),
                    ],
                    dim=1,
                )
                for m, u in zip(mask_input, x)
            ]
        return x, seq_lens, rope_embs, mask_input

    def after_transformer_block(self, block_idx, hidden_states):
        if block_idx in self.audio_injector.injected_block_id:
            audio_attn_id = self.audio_injector.injected_block_id[block_idx]
            audio_emb = self.merged_audio_emb
            num_frames = audio_emb.shape[1]
            if self.use_context_parallel:
                video_hidden_states = sequence_model_parallel_all_gather(
                    hidden_states[:, : self.local_original_seq_len].contiguous(),
                    dim=1,
                )
                input_hidden_states = video_hidden_states[:, : self.original_seq_len]
            else:
                input_hidden_states = hidden_states[:, : self.original_seq_len]
            input_hidden_states = input_hidden_states.clone()
            input_hidden_states = rearrange(
                input_hidden_states, "b (t n) c -> (b t) n c", t=num_frames
            )
            if self.enbale_adain and self.adain_mode == "attn_norm":
                audio_emb_global = rearrange(
                    self.audio_emb_global, "b t n c -> (b t) n c"
                )
                attn_hidden_states = self.audio_injector.injector_adain_layers[
                    audio_attn_id
                ](input_hidden_states, temb=audio_emb_global[:, 0])
            else:
                attn_hidden_states = self.audio_injector.injector_pre_norm_feat[
                    audio_attn_id
                ](input_hidden_states)
            attn_audio_emb = rearrange(
                audio_emb, "b t n c -> (b t) n c", t=num_frames
            )
            residual_out = self.audio_injector.injector[audio_attn_id](
                x=attn_hidden_states,
                context=attn_audio_emb,
                context_lens=torch.ones(
                    attn_hidden_states.shape[0],
                    dtype=torch.long,
                    device=attn_hidden_states.device,
                )
                * attn_audio_emb.shape[1],
            )
            residual_out = rearrange(
                residual_out, "(b t) n c -> b (t n) c", t=num_frames
            )
            if self.use_context_parallel:
                video_hidden_states[:, : self.original_seq_len] = (
                    video_hidden_states[:, : self.original_seq_len] + residual_out
                )
                sp_rank = get_sp_group().rank_in_group
                local_video = video_hidden_states[
                    :,
                    sp_rank
                    * self.local_original_seq_len : (sp_rank + 1)
                    * self.local_original_seq_len,
                ]
                hidden_states[:, : self.local_original_seq_len] = local_video
            else:
                hidden_states[:, : self.original_seq_len] = (
                    hidden_states[:, : self.original_seq_len] + residual_out
                )
        return hidden_states

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        ref_latents,
        motion_latents,
        cond_states,
        audio_input=None,
        motion_frames=(17, 5),
        add_last_motion=2,
        drop_motion_frames=False,
        *extra_args,
        **extra_kwargs,
    ):
        forward_batch = get_forward_context().forward_batch
        sequence_shard_enabled = (
            forward_batch is not None
            and forward_batch.enable_sequence_shard
            and self.sp_size > 1
        )
        add_last_motion = self.add_last_motion * add_last_motion
        audio_input = torch.cat(
            [audio_input[..., 0:1].repeat(1, 1, 1, motion_frames[0]), audio_input],
            dim=-1,
        )
        audio_emb_res = self.casual_audio_encoder(audio_input)
        if self.enbale_adain:
            audio_emb_global, audio_emb = audio_emb_res
            self.audio_emb_global = audio_emb_global[:, motion_frames[1] :].clone()
        else:
            audio_emb = audio_emb_res
        self.merged_audio_emb = audio_emb[:, motion_frames[1] :, :]
        num_frames = self.merged_audio_emb.shape[1]
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        cond = [self.cond_encoder(c.unsqueeze(0)) for c in cond_states]
        x = [x_ + pose for x_, pose in zip(x, cond)]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        original_grid_sizes = deepcopy(grid_sizes)
        grid_sizes = [[torch.zeros_like(grid_sizes), grid_sizes, grid_sizes]]
        self.lat_motion_frames = motion_latents[0].shape[1]
        ref = [self.patch_embedding(r.unsqueeze(0)) for r in ref_latents]
        batch_size = len(ref)
        height, width = ref[0].shape[3], ref[0].shape[4]
        ref_grid_sizes = [[
            torch.tensor([30, 0, 0]).unsqueeze(0).repeat(batch_size, 1),
            torch.tensor([31, height, width]).unsqueeze(0).repeat(batch_size, 1),
            torch.tensor([1, height, width]).unsqueeze(0).repeat(batch_size, 1),
        ]]
        ref = [r.flatten(2).transpose(1, 2) for r in ref]
        self.original_seq_len = seq_lens[0]
        seq_lens = seq_lens + torch.tensor([r.size(1) for r in ref], dtype=torch.long)
        grid_sizes = grid_sizes + ref_grid_sizes
        x = [torch.cat([u, r], dim=1) for u, r in zip(x, ref)]
        mask_input = [
            torch.zeros([1, u.shape[1]], dtype=torch.long, device=x[0].device)
            for u in x
        ]
        for i in range(len(mask_input)):
            mask_input[i][:, self.original_seq_len :] = 1
        x = torch.cat(x)
        b, s, n, d = x.size(0), x.size(1), self.num_heads, self.dim // self.num_heads
        self.pre_compute_freqs = rope_precompute(
            x.detach().view(b, s, n, d), grid_sizes, self.freqs, start=None
        )
        x = [u.unsqueeze(0) for u in x]
        self.pre_compute_freqs = [u.unsqueeze(0) for u in self.pre_compute_freqs]
        x, seq_lens, self.pre_compute_freqs, mask_input = self.inject_motion(
            x,
            seq_lens,
            self.pre_compute_freqs,
            mask_input,
            motion_latents,
            drop_motion_frames=drop_motion_frames,
            add_last_motion=add_last_motion,
        )
        x = torch.cat(x, dim=0)
        self.pre_compute_freqs = torch.cat(self.pre_compute_freqs, dim=0)
        mask_input = torch.cat(mask_input, dim=0)
        video_seq_len = self.original_seq_len
        condition_suffix_len = x.shape[1] - video_seq_len
        self.local_original_seq_len = self.original_seq_len
        seq_len_full = x.shape[1]
        seq_shard_pad = 0
        if sequence_shard_enabled:
            if video_seq_len % self.sp_size != 0:
                seq_shard_pad = self.sp_size - (video_seq_len % self.sp_size)
                pad_hidden = torch.zeros(
                    (x.shape[0], seq_shard_pad, x.shape[2]),
                    dtype=x.dtype,
                    device=x.device,
                )
                pad_freqs = torch.zeros(
                    (
                        self.pre_compute_freqs.shape[0],
                        seq_shard_pad,
                        self.pre_compute_freqs.shape[2],
                        self.pre_compute_freqs.shape[3],
                    ),
                    dtype=self.pre_compute_freqs.dtype,
                    device=self.pre_compute_freqs.device,
                )
                pad_mask = torch.full(
                    (mask_input.shape[0], seq_shard_pad),
                    2,
                    dtype=mask_input.dtype,
                    device=mask_input.device,
                )
                x = torch.cat([x[:, :video_seq_len], pad_hidden, x[:, video_seq_len:]], dim=1)
                self.pre_compute_freqs = torch.cat(
                    [
                        self.pre_compute_freqs[:, :video_seq_len],
                        pad_freqs,
                        self.pre_compute_freqs[:, video_seq_len:],
                    ],
                    dim=1,
                )
                mask_input = torch.cat(
                    [mask_input[:, :video_seq_len], pad_mask, mask_input[:, video_seq_len:]],
                    dim=1,
                )
            padded_video_seq_len = video_seq_len + seq_shard_pad
            if padded_video_seq_len % self.sp_size != 0:
                raise ValueError(
                    f"Wan S2V video sequence length {padded_video_seq_len} must be divisible by sp_size {self.sp_size}"
                )
            sp_rank = get_sp_group().rank_in_group
            self.local_original_seq_len = padded_video_seq_len // self.sp_size
            video_slice = slice(
                sp_rank * self.local_original_seq_len,
                (sp_rank + 1) * self.local_original_seq_len,
            )
            x_video = x[:, video_slice]
            x = torch.cat([x_video, x[:, padded_video_seq_len:]], dim=1)
            freqs_video = self.pre_compute_freqs[:, video_slice]
            self.pre_compute_freqs = torch.cat(
                [freqs_video, self.pre_compute_freqs[:, padded_video_seq_len:]],
                dim=1,
            )
            mask_video = mask_input[:, video_slice]
            mask_input = torch.cat(
                [mask_video, mask_input[:, padded_video_seq_len:]], dim=1
            )
            seq_lens = torch.full_like(
                seq_lens, self.local_original_seq_len + condition_suffix_len
            )
        x = x + self.trainable_cond_mask(mask_input).to(x.dtype)
        if self.zero_timestep:
            t = torch.cat([t, torch.zeros([1], dtype=t.dtype, device=t.device)])
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        if self.zero_timestep:
            e = e[:-1]
            zero_e0 = e0[-1:]
            e0 = e0[:-1]
            e0 = torch.cat(
                [
                    e0.unsqueeze(2),
                    zero_e0.unsqueeze(2).repeat(e0.size(0), 1, 1, 1),
                ],
                dim=2,
            )
            e0 = [e0, self.local_original_seq_len]
        else:
            e0 = [e0.unsqueeze(2).repeat(1, 1, 2, 1), 0]
        context = self.text_embedding(
            torch.stack(
                [
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))]
                    )
                    for u in context
                ]
            )
        )
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.pre_compute_freqs,
            context=context,
            context_lens=None,
            num_replicated_suffix=condition_suffix_len if sequence_shard_enabled else 0,
        )
        for idx, block in enumerate(self.blocks):
            x = block(x, **kwargs)
            x = self.after_transformer_block(idx, x)
        if sequence_shard_enabled:
            x_video = sequence_model_parallel_all_gather(
                x[:, : self.local_original_seq_len].contiguous(), dim=1
            )
            x = torch.cat([x_video, x[:, self.local_original_seq_len :]], dim=1)
            if seq_shard_pad > 0:
                x = x[:, :seq_len_full]
        x = x[:, : self.original_seq_len]
        x = self.head(x, e)
        x = self.unpatchify(x, original_grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, self.out_dim)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            out.append(
                u.reshape(
                    self.out_dim,
                    *[i * j for i, j in zip(v, self.patch_size)],
                )
            )
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        nn.init.zeros_(self.head.head.weight)


class WanS2VTransformer3DModel(WanModelS2V, OffloadableDiTMixin):
    _aliases = ["WanS2VTransformer3DModel"]
    _fsdp_shard_conditions = WanS2VConfig()._fsdp_shard_conditions
    _compile_conditions = WanS2VConfig()._compile_conditions
    _supported_attention_backends = WanS2VConfig()._supported_attention_backends
    param_names_mapping = WanS2VConfig().param_names_mapping
    reverse_param_names_mapping = WanS2VConfig().reverse_param_names_mapping
    lora_param_names_mapping = WanS2VConfig().lora_param_names_mapping
    layer_names = ["blocks"]

    def __init__(
        self,
        config: WanS2VConfig,
        hf_config: dict[str, Any],
        quant_config=None,
    ) -> None:
        del quant_config
        arch = config.arch_config
        super().__init__(
            cond_dim=arch.cond_dim,
            audio_dim=arch.audio_dim,
            num_audio_token=arch.num_audio_token,
            enable_adain=arch.enable_adain,
            adain_mode=arch.adain_mode,
            audio_inject_layers=tuple(arch.audio_inject_layers),
            zero_init=arch.zero_init,
            zero_timestep=arch.zero_timestep,
            enable_motioner=arch.enable_motioner,
            add_last_motion=arch.add_last_motion,
            enable_tsm=arch.enable_tsm,
            trainable_token_pos_emb=arch.trainable_token_pos_emb,
            motion_token_num=arch.motion_token_num,
            enable_framepack=arch.enable_framepack,
            framepack_drop_mode=arch.framepack_drop_mode,
            model_type=arch.model_type,
            patch_size=arch.patch_size,
            text_len=arch.text_len,
            in_dim=arch.in_channels,
            dim=arch.hidden_size,
            ffn_dim=arch.ffn_dim,
            freq_dim=arch.freq_dim,
            text_dim=arch.text_dim,
            out_dim=arch.out_channels,
            num_heads=arch.num_attention_heads,
            num_layers=arch.num_layers,
            qk_norm=arch.qk_norm,
            cross_attn_norm=arch.cross_attn_norm,
            eps=arch.eps,
        )
        self.hf_config = hf_config
        self.param_dtype = next(self.parameters()).dtype
        self.num_train_timesteps = int(hf_config.get("num_train_timesteps", 1000))
        self.sample_neg_prompt = str(
            hf_config.get("sample_neg_prompt", config.sample_neg_prompt)
        )
        self.motion_frames = int(hf_config.get("motion_frames", 73))
        self.drop_first_motion = bool(hf_config.get("drop_first_motion", True))
        self.fps = int(hf_config.get("sample_fps", 16))
        self.audio_sample_m = 0
        self.supports_standard_denoising = True

    def post_load_weights(self) -> None:
        device = next(self.parameters()).device
        d = self.dim // self.num_heads
        if isinstance(self.freqs, torch.Tensor) and self.freqs.is_meta:
            self.freqs = torch.cat(
                [
                    rope_params(1024, d - 4 * (d // 6)),
                    rope_params(1024, 2 * (d // 6)),
                    rope_params(1024, 2 * (d // 6)),
                ],
                dim=1,
            ).to(device=device)
        frame_packer = getattr(self, "frame_packer", None)
        if frame_packer is not None:
            if (
                isinstance(frame_packer.zip_frame_buckets, torch.Tensor)
                and frame_packer.zip_frame_buckets.is_meta
            ):
                frame_packer.zip_frame_buckets = torch.tensor(
                    (1, 2, 16), dtype=torch.long, device=device
                )
            if isinstance(frame_packer.freqs, torch.Tensor) and frame_packer.freqs.is_meta:
                d = frame_packer.inner_dim // frame_packer.num_heads
                frame_packer.freqs = torch.cat(
                    [
                        rope_params(1024, d - 4 * (d // 6)),
                        rope_params(1024, 2 * (d // 6)),
                        rope_params(1024, 2 * (d // 6)),
                    ],
                    dim=1,
                ).to(device=device)
        return None

    def get_default_negative_prompt(self) -> str:
        return self.sample_neg_prompt

    def _normalize_infer_frames(self, num_frames: int) -> int:
        infer_frames = max(int(num_frames) - 1, 4)
        if infer_frames % 4 != 0:
            infer_frames = max((infer_frames // 4) * 4, 4)
        return infer_frames

    def get_size_less_than_area(
        self,
        height: int,
        width: int,
        *,
        target_area: int = 1024 * 704,
        divisor: int = 64,
    ) -> tuple[int, int]:
        if height * width <= target_area:
            max_upper_area = target_area
            min_scale = 0.1
            max_scale = 1.0
        else:
            max_upper_area = target_area
            d = divisor - 1
            b = d * (height + width)
            a = height * width
            c = d**2 - max_upper_area
            min_scale = (-b + (b**2 - 2 * a * c) ** 0.5) / (2 * a)
            max_scale = (max_upper_area / (height * width)) ** 0.5

        find_it = False
        for i in range(100):
            scale = max_scale - (max_scale - min_scale) * i / 100
            new_height, new_width = int(height * scale), int(width * scale)
            pad_height = (64 - new_height % 64) % 64
            pad_width = (64 - new_width % 64) % 64
            padded_height = new_height + pad_height
            padded_width = new_width + pad_width
            if padded_height * padded_width <= max_upper_area:
                find_it = True
                break

        if find_it:
            return padded_height, padded_width

        aspect_ratio = width / height
        target_width = int((target_area * aspect_ratio) ** 0.5 // divisor * divisor)
        target_height = int((target_area / aspect_ratio) ** 0.5 // divisor * divisor)
        if target_width >= width or target_height >= height:
            target_width = int(width // divisor * divisor)
            target_height = int(height // divisor * divisor)
        return target_height, target_width

    def get_generation_size(self, *, image_path: str) -> tuple[int, int]:
        ref_image = np.array(Image.open(image_path).convert("RGB"))
        height, width = ref_image.shape[:2]
        return self.get_size_less_than_area(
            int(height),
            int(width),
            target_area=int(self.hf_config.get("max_area", 720 * 1280)),
        )

    def prepare_standard_s2v_latents(
        self,
        *,
        latent_shape: tuple[int, ...],
        generator: torch.Generator | list[torch.Generator] | None,
    ) -> torch.Tensor:
        from diffusers.utils.torch_utils import randn_tensor

        return randn_tensor(
            latent_shape,
            generator=generator,
            device=self.device,
            dtype=self.param_dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor] | None = None,
        ref_latents: torch.Tensor | None = None,
        motion_latents: torch.Tensor | None = None,
        cond_states: torch.Tensor | None = None,
        audio_input: torch.Tensor | None = None,
        motion_frames: list[int] | tuple[int, int] | None = None,
        drop_motion_frames: bool = False,
        add_last_motion: bool | int | None = None,
        seq_len: int | None = None,
        guidance=None,
        **kwargs,
    ) -> torch.Tensor:
        del guidance
        if timestep is None and "t" in kwargs:
            timestep = kwargs.pop("t")
        if timestep is None:
            raise ValueError("Wan S2V forward requires timestep")
        if encoder_hidden_states is None:
            raise ValueError("Wan S2V forward requires encoder_hidden_states")
        if isinstance(encoder_hidden_states, list):
            if len(encoder_hidden_states) == 0:
                raise ValueError("encoder_hidden_states list cannot be empty")
            context = encoder_hidden_states
        elif isinstance(encoder_hidden_states, torch.Tensor):
            if encoder_hidden_states.ndim == 3:
                context = [
                    encoder_hidden_states[i]
                    for i in range(encoder_hidden_states.shape[0])
                ]
            else:
                context = [encoder_hidden_states]
        else:
            raise TypeError(
                "Wan S2V encoder_hidden_states must be a tensor or list of tensors"
            )
        if seq_len is None:
            seq_len = int(
                hidden_states.shape[2]
                * hidden_states.shape[3]
                * hidden_states.shape[4]
                // 4
            )
        if ref_latents is None or motion_latents is None or cond_states is None:
            raise ValueError(
                "Wan S2V forward requires ref_latents, motion_latents, and cond_states"
            )
        if motion_frames is None:
            motion_frames = [
                self.motion_frames,
                (self.motion_frames + 3) // 4,
            ]
        if add_last_motion is None:
            add_last_motion = 2
        output = super().forward(
            hidden_states,
            t=timestep,
            context=context,
            seq_len=seq_len,
            ref_latents=ref_latents,
            motion_latents=motion_latents,
            cond_states=cond_states,
            audio_input=audio_input,
            motion_frames=motion_frames,
            drop_motion_frames=drop_motion_frames,
            add_last_motion=add_last_motion,
            **kwargs,
        )
        if isinstance(output, (list, tuple)):
            if len(output) != 1:
                raise ValueError(
                    f"Wan S2V noise model returned unexpected output length: {len(output)}"
                )
            return output[0]
        return output


EntryClass = WanS2VTransformer3DModel
