# Copyright 2026 Qwen-Image Team and The HuggingFace Team
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch import nn

from sglang.kernels.ops.diffusion import (
    BitExactFusionGate,
    can_use_fused_complex_rope,
    can_use_fused_layernorm_modulate,
    can_use_fused_silu_mul,
    can_use_rmsnorm_preserve_reduction,
    fused_complex_rope,
    fused_layernorm_modulate,
    fused_silu_mul_bitexact,
    residual_gate_add,
    rmsnorm_preserve_reduction,
    tensors_equal,
)
from sglang.kernels.ops.diffusion.rope.qknorm_complex_rope_kv_triton import (
    can_use_qknorm_complex_rope_kv,
    qknorm_complex_rope_kv,
)
from sglang.kernels.ops.diffusion.rope.qknorm_complex_rope_triton import (
    can_use_qknorm_complex_rope,
    qknorm_complex_rope,
)
from sglang.multimodal_gen.runtime.distributed import (
    get_sp_world_size,
    get_tp_world_size,
)
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_parallel_rank,
)
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention, USPAttention
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.layers.layernorm import RMSNorm

logger = init_logger(__name__)
_ROPE_FUSION = BitExactFusionGate("Qwen-Image 2.1 complex RoPE")
_SILU_MUL_FUSION = BitExactFusionGate("Qwen-Image 2.1 SiLU-mul")
_QK_ROPE_FUSION = BitExactFusionGate("Qwen-Image 2.1 Q/K RMSNorm + complex RoPE")
_KV_ROPE_FUSION = BitExactFusionGate("Qwen-Image 2.1 K RMSNorm + RoPE + KV packing")
_QK_NORM_FUSION = BitExactFusionGate("Qwen-Image 2.1 Q/K RMSNorm")
_MODULATION_FUSION = BitExactFusionGate("Qwen-Image 2.1 LayerNorm modulation")


def build_layout(image_slots, image_shapes, axes_dims, device):
    """Expand each condition-image slot to its complete latent grid before denoising."""
    indices, image_indices, positions, segments = [], [], [], []
    cursor = position = image_index = 0
    for text_index, is_image in enumerate(image_slots):
        if not is_image:
            indices.append(text_index)
            positions.append((position, position, position))
            position += 1
            continue
        start = len(indices)
        if start > cursor:
            segments.append((cursor, start, False))
        _, height, width = image_shapes[image_index]
        for h in range(-(height - height // 2), height // 2):
            for w in range(-(width - width // 2), width // 2):
                indices.append(text_index)
                image_indices.append(len(indices) - 1)
                positions.append((position, h, w))
        segments.append((start, len(indices), True))
        cursor = len(indices)
        position += max(height, width)
        image_index += 1
    if image_index != len(image_shapes) - 1:
        raise ValueError("condition-image slots do not match image_shapes")
    if len(indices) > cursor:
        segments.append((cursor, len(indices), False))
    prefix_len = len(indices)
    _, height, width = image_shapes[-1]
    for h in range(-(height - height // 2), height // 2):
        for w in range(-(width - width // 2), width // 2):
            positions.append((position, h, w))
    pos = torch.tensor(positions, device=device, dtype=torch.float32)
    angles = torch.cat(
        [
            pos[:, axis : axis + 1]
            * (10000.0 ** (-torch.arange(0, dim, 2, device=device).float() / dim))
            for axis, dim in enumerate(axes_dims)
        ],
        dim=-1,
    )
    rope = torch.polar(torch.ones_like(angles), angles)
    return dict(
        encoder_seq_len=len(image_slots),
        text_indices=torch.tensor(indices, device=device, dtype=torch.long),
        image_indices=torch.tensor(image_indices, device=device, dtype=torch.long),
        prefix_rope=rope[:prefix_len],
        target_rope=rope[prefix_len:],
        segments=tuple(segments),
    )


def apply_rope(x, rope):
    fused = None
    if can_use_fused_complex_rope(x, rope) and _ROPE_FUSION.can_attempt_once():
        fused = fused_complex_rope(x, rope)
        if _ROPE_FUSION.verified:
            return fused
    z = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    out = torch.view_as_real(z * rope[None, :, None]).flatten(-2).to(x.dtype)
    if fused is not None:
        return _ROPE_FUSION.accept_or_fallback(fused, out, logger=logger)
    return out


def apply_qk_norm(x, norm):
    fused = None
    if (
        can_use_rmsnorm_preserve_reduction(x, norm.weight)
        and _QK_NORM_FUSION.can_attempt_once()
    ):
        fused = rmsnorm_preserve_reduction(x, norm.weight, norm.variance_epsilon)
        if _QK_NORM_FUSION.verified:
            return fused
    out = norm(x)
    if fused is not None:
        return _QK_NORM_FUSION.accept_or_fallback(fused, out, logger=logger)
    return out


def apply_qk_norm_rope(x, norm, rope):
    fused = None
    if (
        can_use_qknorm_complex_rope(x, norm.weight, rope)
        and _QK_ROPE_FUSION.can_attempt_once()
    ):
        fused = qknorm_complex_rope(x, norm.weight, rope, norm.variance_epsilon)
        if _QK_ROPE_FUSION.verified:
            return fused
    out = apply_rope(apply_qk_norm(x, norm), rope)
    if fused is not None:
        return _QK_ROPE_FUSION.accept_or_fallback(fused, out, logger=logger)
    return out


def apply_modulation(x, norm, scale):
    fused = None
    if (
        can_use_fused_layernorm_modulate(x, scale.squeeze(1), None)
        and _MODULATION_FUSION.can_attempt_once()
    ):
        fused = fused_layernorm_modulate(x, scale.squeeze(1), None, norm.eps)
        if _MODULATION_FUSION.verified:
            return fused
    out = norm(x) * (1 + scale)
    if fused is not None:
        return _MODULATION_FUSION.accept_or_fallback(fused, out, logger=logger)
    return out


class QwenImage21ZeroCenterRMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        scale = self.weight.float() + 1
        value = x.float()
        return (
            value
            * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
            * scale
        ).to(x.dtype)


class QwenImage21TextProjection(nn.Module):
    def __init__(self, context_dim, dim, eps):
        super().__init__()
        self.text_norm = QwenImage21ZeroCenterRMSNorm(context_dim, eps)
        self.in_layer = nn.Linear(context_dim, dim, bias=False)
        self.out_layer = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.out_layer(
            nn.functional.gelu(self.in_layer(self.text_norm(x)), approximate="tanh")
        )


class QwenImage21TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.timestep_embedder = nn.Module()
        self.timestep_embedder.linear_1 = nn.Linear(256, dim, bias=False)
        self.timestep_embedder.linear_2 = nn.Linear(dim, dim, bias=False)

    def forward(self, t, dtype):
        freq = torch.exp(
            -math.log(10000) * torch.arange(128, device=t.device).float() / 128
        )
        angles = t.float()[:, None] * 1000 * freq
        x = torch.cat([angles.cos(), angles.sin()], dim=-1).to(dtype)
        return self.timestep_embedder.linear_2(
            nn.functional.silu(self.timestep_embedder.linear_1(x))
        )


class QwenImage21FeedForward(nn.Module):
    def __init__(self, dim, ratio, quant_config, prefix):
        super().__init__()
        self.proj = ColumnParallelLinear(
            dim,
            dim * ratio,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
        )
        self.gate_layer = ColumnParallelLinear(
            dim,
            dim * ratio,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_layer",
        )
        self.out = RowParallelLinear(
            dim * ratio,
            dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.out",
        )

    def forward(self, x):
        gate, value = self.gate_layer(x)[0], self.proj(x)[0]
        fused = None
        if can_use_fused_silu_mul(gate, value) and _SILU_MUL_FUSION.can_attempt_once():
            fused = fused_silu_mul_bitexact(gate, value)
            if _SILU_MUL_FUSION.verified:
                return self.out(fused)[0]
        hidden = nn.functional.silu(gate) * value
        if fused is not None:
            hidden = _SILU_MUL_FUSION.accept_or_fallback(fused, hidden, logger=logger)
        return self.out(hidden)[0]


class QwenImage21Attention(nn.Module):
    def __init__(self, ac, quant_config, prefix):
        super().__init__()
        dim = ac.hidden_size
        self.heads = ac.num_attention_heads // get_tp_world_size()
        self.head_dim = ac.attention_head_dim
        self.to_q = ColumnParallelLinear(
            dim, dim, bias=False, quant_config=quant_config, prefix=f"{prefix}.to_q"
        )
        self.to_k = ColumnParallelLinear(
            dim, dim, bias=False, quant_config=quant_config, prefix=f"{prefix}.to_k"
        )
        self.to_v = ColumnParallelLinear(
            dim, dim, bias=False, quant_config=quant_config, prefix=f"{prefix}.to_v"
        )
        self.to_out = nn.ModuleList(
            [
                RowParallelLinear(
                    dim,
                    dim,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.to_out.0",
                )
            ]
        )
        self.norm_q = RMSNorm(
            self.head_dim, ac.eps, cast_x_before_out_mul=True, force_native=True
        )
        self.norm_k = RMSNorm(
            self.head_dim, ac.eps, cast_x_before_out_mul=True, force_native=True
        )
        backends = QwenImage21Transformer2DModel._supported_attention_backends
        self.local_attn = LocalAttention(
            self.heads, self.head_dim, supported_attention_backends=backends
        )
        self.target_attn = USPAttention(
            self.heads, self.head_dim, supported_attention_backends=backends
        )

    def project_qkv(self, x):
        q = self.to_q(x)[0].unflatten(-1, (self.heads, self.head_dim))
        k = self.to_k(x)[0].unflatten(-1, (self.heads, self.head_dim))
        v = self.to_v(x)[0].unflatten(-1, (self.heads, self.head_dim))
        return q, k, v

    def qkv(self, x, rope):
        q, k, v = self.project_qkv(x)
        return (
            apply_qk_norm_rope(q, self.norm_q, rope),
            apply_qk_norm_rope(k, self.norm_k, rope),
            v,
        )

    def attend_sample(self, q, k, v, rope, prefix, prefix_rope, segments, cache):
        if cache:
            kp, vp = cache["key"], cache["value"]
            prefix_output = None
        else:
            qp, kp, vp = self.qkv(prefix, prefix_rope)
            outputs = []
            # text runs are causal; image blocks see the entire preceding sequence and themselves
            for start, end, is_image in segments:
                mask = None
                if not is_image:
                    mask = (
                        torch.arange(end, device=q.device)[None, :]
                        <= torch.arange(start, end, device=q.device)[:, None]
                    )
                    mask = mask[None, None]
                outputs.append(
                    self.local_attn(
                        qp[:, start:end], kp[:, :end], vp[:, :end], attn_mask=mask
                    )
                )
            prefix_output = self.to_out[0](torch.cat(outputs, dim=1).flatten(2))[0]
            if cache is not None:
                cache.update(key=kp, value=vp)
        q = apply_qk_norm_rope(q, self.norm_q, rope)
        packed = None
        if (
            get_sp_world_size() == 1
            and can_use_qknorm_complex_rope_kv(k, self.norm_k.weight, rope, v, kp, vp)
            and _KV_ROPE_FUSION.can_attempt_once()
        ):
            packed = qknorm_complex_rope_kv(
                k, self.norm_k.weight, rope, v, kp, vp, self.norm_k.variance_epsilon
            )
            if not _KV_ROPE_FUSION.verified:
                reference = (
                    torch.cat([kp, apply_rope(apply_qk_norm(k, self.norm_k), rope)], 1),
                    torch.cat([vp, v], 1),
                )
                packed = _KV_ROPE_FUSION.accept_or_fallback(
                    packed,
                    reference,
                    equal=tensors_equal,
                    logger=logger,
                )
        if packed is not None:
            out = self.target_attn(q, *packed)
        else:
            k = apply_qk_norm_rope(k, self.norm_k, rope)
            out = self.target_attn.forward_with_replicated_kv_prefix(q, kp, vp, k, v)
        return out, prefix_output

    def forward(self, x, ropes, prefixes, layouts, caches):
        # batch target projections while retaining each sample's unpadded prefix
        q, k, v = self.project_qkv(x)
        outputs, prefix_outputs = [], []
        for sample, layout in enumerate(layouts):
            out, prefix_out = self.attend_sample(
                q[sample : sample + 1],
                k[sample : sample + 1],
                v[sample : sample + 1],
                ropes[sample],
                prefixes[sample],
                layout["prefix_rope"],
                layout["segments"],
                caches[sample],
            )
            outputs.append(out)
            prefix_outputs.append(prefix_out)
        return self.to_out[0](torch.cat(outputs).flatten(2))[0], prefix_outputs


class QwenImage21TransformerBlock(nn.Module):
    def __init__(self, ac, quant_config, prefix):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(
            ac.hidden_size, eps=ac.eps, elementwise_affine=False
        )
        self.img_norm2 = nn.LayerNorm(
            ac.hidden_size, eps=ac.eps, elementwise_affine=False
        )
        self.attn = QwenImage21Attention(ac, quant_config, f"{prefix}.attn")
        self.img_mlp = QwenImage21FeedForward(
            ac.hidden_size, ac.mlp_ratio, quant_config, f"{prefix}.img_mlp"
        )

    def forward(
        self,
        hidden_states,
        modulation,
        prefix_states,
        prefix_modulation,
        layouts,
        ropes,
        caches,
    ):
        scale1, gate1, scale2, gate2 = modulation
        prefixes = [
            apply_modulation(
                state["hidden_states"], self.img_norm1, prefix_modulation[0]
            )
            if not cache
            else None
            for state, cache in zip(prefix_states, caches, strict=True)
        ]
        attention, prefix_attentions = self.attn(
            apply_modulation(hidden_states, self.img_norm1, scale1),
            ropes,
            prefixes,
            layouts,
            caches,
        )
        hidden_states = residual_gate_add(hidden_states, attention, gate1)
        hidden_states = residual_gate_add(
            hidden_states,
            self.img_mlp(apply_modulation(hidden_states, self.img_norm2, scale2)),
            gate2,
        )
        for state, attention in zip(prefix_states, prefix_attentions, strict=True):
            if attention is not None:
                _, pg1, ps2, pg2 = prefix_modulation
                prefix = residual_gate_add(state["hidden_states"], attention, pg1)
                state["hidden_states"] = residual_gate_add(
                    prefix,
                    self.img_mlp(apply_modulation(prefix, self.img_norm2, ps2)),
                    pg2,
                )
        return hidden_states


class QwenImage21OutputNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

    def forward(self, x, temb):
        return apply_modulation(
            x, self.norm, self.linear(nn.functional.silu(temb))[:, None]
        )


class QwenImage21Transformer2DModel(CachableDiT, LayerwiseOffloadableModuleMixin):
    _supported_attention_backends = {
        AttentionBackendEnum.FA,
        AttentionBackendEnum.SAGE_ATTN,
        AttentionBackendEnum.SAGE_ATTN_3,
        AttentionBackendEnum.TORCH_SDPA,
    }
    _fsdp_shard_conditions = [
        lambda name, module: isinstance(module, QwenImage21TransformerBlock)
    ]
    _compile_conditions = _fsdp_shard_conditions
    layer_names = ["transformer_blocks"]
    param_names_mapping = {}

    def __init__(self, config, hf_config, quant_config=None, **kwargs):
        super().__init__(config, hf_config=hf_config, **kwargs)
        ac = self.config
        if ac.patch_size != 1 or not ac.causal_condition or not ac.causal_block:
            raise ValueError(
                "Qwen-Image 2.1 requires patch_size=1, causal_condition=True and causal_block=True"
            )
        self.hidden_size = ac.hidden_size
        self.num_attention_heads = ac.num_attention_heads
        self.num_channels_latents = ac.in_channels
        self.img_in = nn.Linear(ac.in_channels, ac.hidden_size, bias=False)
        self.txt_in = QwenImage21TextProjection(
            ac.context_in_dim, ac.hidden_size, ac.eps
        )
        self.time_text_embed = QwenImage21TimeEmbedding(ac.hidden_size)
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(ac.hidden_size, ac.hidden_size * 4, bias=False)
        )
        self.transformer_blocks = nn.ModuleList(
            [
                QwenImage21TransformerBlock(ac, quant_config, f"transformer_blocks.{i}")
                for i in range(ac.num_layers)
            ]
        )
        self.norm_out = QwenImage21OutputNorm(ac.hidden_size, ac.eps)
        self.proj_out = nn.Linear(ac.hidden_size, ac.out_channels, bias=False)

    def prepare_modulation(self, temb):
        # All blocks share these gates. Preserve the native tanh and its dtype,
        # but compute it once per timestep instead of once per block.
        scale1, gate1, scale2, gate2 = self.modulation(temb)[:, None].chunk(4, dim=-1)
        return scale1, gate1.tanh(), scale2, gate2.tanh()

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        layouts,
        condition_latents=None,
        prefix_caches=None,
        **kwargs,
    ):
        if isinstance(encoder_hidden_states, list):
            encoder_hidden_states = encoder_hidden_states[0]
        sp = get_sp_world_size()
        target_len = hidden_states.shape[1]
        if target_len % sp:
            raise ValueError(
                f"target token count {target_len} must be divisible by SP degree {sp}"
            )
        local_len = target_len // sp
        rank = get_sp_parallel_rank()
        start, end = rank * local_len, (rank + 1) * local_len
        images = self.img_in(hidden_states[:, start:end])
        temb = self.time_text_embed((timestep.to(images.dtype) / 1000), images.dtype)
        modulation = self.prepare_modulation(temb)
        prefix_modulation = None
        if prefix_caches is None or any(not cache[0] for cache in prefix_caches):
            zero_temb = self.time_text_embed(
                timestep.new_zeros(1).to(images.dtype), images.dtype
            )
            prefix_modulation = self.prepare_modulation(zero_temb)
        if prefix_caches is None:
            prefix_caches = [[None] * len(self.transformer_blocks) for _ in layouts]
        prefix_states, ropes = [], []
        for sample, layout in enumerate(layouts):
            prefix = None
            if not prefix_caches[sample][0]:
                prefix = self.txt_in(
                    encoder_hidden_states[
                        sample : sample + 1, : layout["encoder_seq_len"]
                    ]
                ).index_select(1, layout["text_indices"])
                if condition_latents is not None:
                    prefix[:, layout["image_indices"]] = self.img_in(
                        condition_latents[sample : sample + 1]
                    )
            prefix_states.append({"hidden_states": prefix})
            ropes.append(layout["target_rope"][start:end])
        # visit each block once so layerwise offload transfers weights once per batch
        for i, block in enumerate(self.transformer_blocks):
            images = block(
                images,
                modulation,
                prefix_states,
                prefix_modulation,
                layouts,
                ropes,
                [cache[i] for cache in prefix_caches],
            )
        output = self.proj_out(self.norm_out(images, temb))
        if sp > 1:
            output = sequence_model_parallel_all_gather(output, dim=1)
        return output


EntryClass = QwenImage21Transformer2DModel
