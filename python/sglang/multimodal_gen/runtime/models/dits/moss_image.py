# SPDX-License-Identifier: Apache-2.0

# Reference implementation license:
# BSD 3-Clause License
#
# (c) Meta Platforms, Inc. and affiliates.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,this list
# of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may
# be used to endorse or promote products derived from this software without specific
# prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
# SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

# Adapted from OpenMOSS-Team/MOSS-Image-Base, revision 355120bd6b44ff67b8df18f57e40b928fc886cd3.

import msgspec
import torch
from torch import nn

from sglang.kernels.ops.diffusion import (
    BitExactFusionGate,
    can_use_fused_situ_softcap_mul,
    can_use_rmsnorm_cast_before_weight,
    can_use_rmsnorm_cast_before_weight_modulate,
    can_use_rmsnorm_cast_before_weight_rope,
    fused_rope_rotate_half_bitexact,
    fused_situ_softcap_mul_bitexact,
    modulate_scale_shift,
    residual_gate_add,
    rmsnorm_cast_before_weight_bitexact,
    rmsnorm_cast_before_weight_modulate_bitexact,
    rmsnorm_cast_before_weight_rope_bitexact,
)
from sglang.multimodal_gen.configs.models.dits.moss_image import MOSSImageDiTConfig
from sglang.multimodal_gen.configs.models.fsdp import is_layer
from sglang.multimodal_gen.runtime.distributed import (
    get_sp_world_size,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.distributed.sp_shard_utils import (
    gather_seq,
    shard_like,
    shard_seq,
    tail_attn_meta,
)
from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.layers.attention.layer import LocalAttention
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
_SITU_FUSION = BitExactFusionGate("MOSSImage SiTU")
_ROPE_FUSION = BitExactFusionGate("MOSSImage RoPE")
_RMSNORM_FUSION = BitExactFusionGate("MOSSImage RMSNorm", per_signature=True)
_NORM_MOD_FUSION = BitExactFusionGate("MOSSImage RMSNorm + modulation")
_NORM_ROPE_FUSION = BitExactFusionGate("MOSSImage RMSNorm + RoPE")


def validate_skip_layers(skip_layers, num_layers):
    try:
        skip_layers = tuple(skip_layers)
    except TypeError as exc:
        raise ValueError("skip_layers must be a sequence of layer indices") from exc
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < num_layers
        for index in skip_layers
    ):
        raise ValueError(
            f"skip_layers must contain integer indices in [0, {num_layers})."
        )
    return skip_layers


class _ConditioningGroup(msgspec.Struct, frozen=True):
    indices: torch.Tensor | None
    text: torch.Tensor
    rope: tuple[torch.Tensor, torch.Tensor]


class MOSSImageConditioning(msgspec.Struct, frozen=True):
    """Refined text and RoPE owned by one request and one CFG branch."""

    spatial_shape: tuple[int, int]
    groups: tuple[_ConditioningGroup, ...]

    @property
    def batch_size(self):
        return sum(group.text.shape[0] for group in self.groups)


class _RMSNorm(nn.Module):
    """Normalize in fp32, then round before applying the learned scale."""

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        if not can_use_rmsnorm_cast_before_weight(
            device=x.device, dtype=x.dtype, contiguous=x.is_contiguous()
        ):
            return self.forward_native(x)
        return _RMSNORM_FUSION.run(
            lambda: rmsnorm_cast_before_weight_bitexact(x, self.weight),
            lambda: self.forward_native(x),
            sig=(x.device.index, x.dtype, x.shape[-1], x.numel() // x.shape[-1] >= 16),
            logger=logger,
            reraise=(ValueError,),
        )

    def forward_native(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight


def _situ_native(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (
        25.0 * torch.tanh(hidden / 25.0)
    )


def _situ(hidden, gate):
    if not can_use_fused_situ_softcap_mul(device=hidden.device, dtype=hidden.dtype):
        return _situ_native(hidden, gate)
    return _SITU_FUSION.run(
        lambda: fused_situ_softcap_mul_bitexact(hidden, gate),
        lambda: _situ_native(hidden, gate),
        logger=logger,
        reraise=(ValueError,),
    )


def _prepare_rope(rope):
    # Every layer and both Q/K projections share these expanded coefficients.
    sin, cos = rope.chunk(2, -1)
    return torch.cat((sin, sin), -1), torch.cat((cos, cos), -1)


def _rotate_native(x, rope):
    sin, cos = (part.unsqueeze(-2) for part in rope)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def _rotate(x, rope):
    sin, cos = (part.flatten(0, 1) for part in rope)
    if not (x.is_cuda and x.dtype is torch.bfloat16):
        return _rotate_native(x, rope)
    return _ROPE_FUSION.run(
        lambda: fused_rope_rotate_half_bitexact(x, cos, sin),
        lambda: _rotate_native(x, rope),
        logger=logger,
        reraise=(ValueError,),
    )


def _attention(q, k, v, rope, heads, attention, attn_mask_meta=None):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q, k = q.reshape(shape), k.reshape(shape)
    if rope is not None:
        q = _rotate(q, rope)
        k = _rotate(k, rope)
    v = v.reshape(shape)
    kwargs = {"attn_mask_meta": attn_mask_meta} if attn_mask_meta is not None else {}
    return attention(q.to(v.dtype), k.to(v.dtype), v, **kwargs).flatten(2)


def _norm_modulate(norm, x, scale, shift):
    if not can_use_rmsnorm_cast_before_weight_modulate(
        device=x.device,
        dtype=x.dtype,
        dim=x.shape[-1],
        rows=x.numel() // x.shape[-1],
    ):
        return modulate_scale_shift(norm(x), scale, shift)
    return _NORM_MOD_FUSION.run(
        lambda: rmsnorm_cast_before_weight_modulate_bitexact(
            x, norm.weight, scale, shift
        ),
        lambda: modulate_scale_shift(norm(x), scale, shift),
        logger=logger,
        reraise=(ValueError,),
    )


def _norm_rope(norm, x, rope, heads):
    x = x.unflatten(-1, (heads, -1))
    sin, cos = (part.flatten(0, 1) for part in rope)
    if not can_use_rmsnorm_cast_before_weight_rope(
        device=x.device,
        dtype=x.dtype,
        dim=x.shape[-1],
        rows=x.numel() // x.shape[-1],
    ):
        return _rotate(norm(x), rope).flatten(2)
    return _NORM_ROPE_FUSION.run(
        lambda: rmsnorm_cast_before_weight_rope_bitexact(x, norm.weight, cos, sin),
        lambda: _rotate(norm(x), rope),
        logger=logger,
        reraise=(ValueError,),
    ).flatten(2)


class _TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.attention = LocalAttention(
            num_heads=heads,
            head_size=dim // heads,
            supported_attention_backends={
                AttentionBackendEnum.FA,
                AttentionBackendEnum.TORCH_SDPA,
            },
            default_attention_backend=AttentionBackendEnum.FA,
            compute_dtype=torch.bfloat16,
        )
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(_attention(q, k, v, rope, self.heads, self.attention))
        y = self.norm2(x)
        return x + self.ffn_down(_situ(self.ffn_up(y), self.ffn_gate(y)))


class _TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([_TextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, x, rope):
        x = self.proj(x.to(self.proj.weight.dtype))
        for block in self.blocks:
            x = block(x, rope)
        return x


class _Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        attention_cls = (
            USPAttention
            if model_parallel_is_initialized() and get_sp_world_size() > 1
            else LocalAttention
        )
        self.attention = attention_cls(
            num_heads=heads,
            head_size=dim // heads,
            supported_attention_backends={
                AttentionBackendEnum.FA,
                AttentionBackendEnum.TORCH_SDPA,
            },
            default_attention_backend=AttentionBackendEnum.FA,
            compute_dtype=torch.bfloat16,
        )
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.proj = nn.ModuleDict(
            {name: nn.Linear(dim, dim, bias=False) for name in ("q", "k", "v", "o")}
        )
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x, rope, modulation, attn_mask_meta=None):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = _norm_modulate(self.norm1, x, scale_a, shift_a)
        dtype = self.proj["q"].weight.dtype
        y = y.to(dtype)
        q = _norm_rope(self.q_norm, self.proj["q"](y), rope, self.heads)
        k = _norm_rope(self.k_norm, self.proj["k"](y), rope, self.heads)
        x = residual_gate_add(
            x,
            self.proj["o"](
                _attention(
                    q,
                    k,
                    self.proj["v"](y),
                    None,
                    self.heads,
                    self.attention,
                    attn_mask_meta,
                )
            ),
            gate_a[:, None],
        )
        y = _norm_modulate(self.norm2, x, scale_f, shift_f)
        hidden, gate = self.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)
        return residual_gate_add(
            x, self.proj["ffn_down"](_situ(hidden, gate)), gate_f[:, None]
        )


class _RoPE(nn.Module):
    def __init__(self, head_dim, base):
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base
        self.register_buffer("bands", self._default_bands(torch.device("cpu")))

    def _default_bands(self, device):
        # Match training's CPU fp32 construction, including under meta init.
        exponent = (
            torch.arange(self.num_bands, device="cpu", dtype=torch.float32)
            / self.num_bands
        )
        return (1 / (self.base**exponent)).to(device)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # .float() cannot recover frequencies already rounded by .to(bf16).
        self.reset_bands()
        return self

    def reset_bands(self):
        bands = self._default_bands(self.bands.device)
        # The shared assign=True loader can materialize buffers as Parameters.
        self._parameters.pop("bands", None)
        self.register_buffer("bands", bands)

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        # Frequencies are derived constants; retain the key for checkpoint compatibility.
        self.reset_bands()

    def forward(self, height, width, text_length, device):
        # Diffusers/Accelerate can install tensors directly, bypassing load hooks.
        if self.bands.dtype != torch.float32:
            self.reset_bands()
        # Coordinates: (r, f, h, w); sequence order is image followed by text.
        h, w = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        image = (
            torch.stack(
                (torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1
            )
            .flatten(0, 1)
            .float()
        )
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device)
        phase = torch.cat((image, text), 0).unsqueeze(-1) * self.bands
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


class MOSSImageTransformer(BaseDiT, LayerwiseOffloadableModuleMixin):
    """MOSSImage DiT with checkpoint-compatible names and SGLang attention.

    Each sample is evaluated at its true text length, including text refinement;
    padded tokens cannot influence either text or joint attention.
    """

    _fsdp_shard_conditions = [is_layer]
    _fsdp_forward_methods = ("prepare_conditioning",)
    _compile_conditions = []
    param_names_mapping = {}
    reverse_param_names_mapping = {}
    _supported_attention_backends = {
        AttentionBackendEnum.FA,
        AttentionBackendEnum.TORCH_SDPA,
    }

    def __init__(
        self, config: MOSSImageDiTConfig, hf_config=None, quant_config=None, **kwargs
    ):
        super().__init__(config, hf_config or {}, **kwargs)
        if quant_config is not None:
            raise NotImplementedError("MOSSImage DiT quantization is not implemented")
        arch = self.config
        dim, ffn_dim, num_heads = arch.dim, arch.ffn_dim, arch.num_heads
        num_layers, text_dim = arch.num_layers, arch.text_dim
        freq_dim, latent_channels, rope_base = (
            arch.freq_dim,
            arch.latent_channels,
            arch.rope_base,
        )
        self.layer_names = ["layers"]
        self.hidden_size = dim
        self._sequence_parallel = (
            model_parallel_is_initialized() and get_sp_world_size() > 1
        )
        self.num_attention_heads = num_heads
        self.num_channels_latents = latent_channels
        if dim % num_heads or (dim // num_heads) % 8 or freq_dim % 2:
            raise ValueError(
                "Require dim divisible by heads, head_dim divisible by 8, and even freq_dim."
            )
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = _TextProjection(text_dim, dim, ffn_dim, num_heads)
        self.rope = _RoPE(dim // num_heads, rope_base)
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList(
            [_Block(dim, ffn_dim, num_heads) for _ in range(num_layers)]
        )
        self.final_norm_video = _RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

    def _unpad_text(self, encoder_hidden_states, encoder_attention_mask):
        if (
            encoder_hidden_states.ndim != 3
            or encoder_hidden_states.shape[-1] != self.config.text_dim
        ):
            raise ValueError("encoder_hidden_states must be [B, S, text_dim].")
        mask = encoder_attention_mask
        if mask is None:
            contexts = tuple(encoder_hidden_states.split(1))
        else:
            if mask.shape != encoder_hidden_states.shape[:2] or not torch.all(
                (mask == 0) | (mask == 1)
            ):
                raise ValueError("encoder_attention_mask must be a binary [B, S] mask.")
            contexts = tuple(
                encoder_hidden_states[index : index + 1, mask[index].bool()]
                for index in range(encoder_hidden_states.shape[0])
            )
        if any(context.shape[1] == 0 for context in contexts):
            raise ValueError("Each sample needs at least one valid text token.")
        return contexts

    def _prepare_sample_conditioning(self, context, height, width, dtype):
        rope = _prepare_rope(
            self.rope(height, width, context.shape[1], context.device).to(dtype)
        )
        rope = tuple(
            part.expand(context.shape[0], -1, -1).contiguous() for part in rope
        )
        # Small text GEMMs are particularly sensitive to changing batch shape.
        texts = [
            self.text_embedder(
                context[index : index + 1],
                tuple(part[index : index + 1, height * width :] for part in rope),
            )
            for index in range(context.shape[0])
        ]
        text = texts[0] if len(texts) == 1 else torch.cat(texts)
        return text, rope

    def prepare_conditioning(
        self,
        encoder_hidden_states,
        encoder_attention_mask,
        height,
        width,
        dtype,
        *,
        batch_samples=False,
    ):
        # Neither the text refinement blocks nor RoPE depend on the timestep.
        contexts = self._unpad_text(encoder_hidden_states, encoder_attention_mask)
        by_length = {}
        for index, context in enumerate(contexts):
            key = context.shape[1] if batch_samples else index
            by_length.setdefault(key, []).append(index)
        groups = []
        for indices in by_length.values():
            context = torch.cat([contexts[index] for index in indices])
            text, rope = self._prepare_sample_conditioning(
                context, height, width, dtype
            )
            groups.append(
                _ConditioningGroup(
                    indices=(
                        torch.as_tensor(indices, device=context.device)
                        if len(by_length) > 1
                        else None
                    ),
                    text=text,
                    rope=rope,
                )
            )
        return MOSSImageConditioning(
            spatial_shape=(height, width),
            groups=tuple(groups),
        )

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask=None,
        prepared_conditioning: MOSSImageConditioning | None = None,
        skip_layers=(),
    ):
        skip_layers = validate_skip_layers(skip_layers, self.config.num_layers)
        if (
            hidden_states.ndim != 4
            or hidden_states.shape[1] != self.config.latent_channels
        ):
            raise ValueError("hidden_states must be [B, latent_channels, H, W].")
        batch, _, height, width = hidden_states.shape
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch:
            raise ValueError("encoder_hidden_states must be [B, S, text_dim].")
        if prepared_conditioning is not None and (
            prepared_conditioning.spatial_shape != (height, width)
            or prepared_conditioning.batch_size != batch
        ):
            raise ValueError(
                "Prepared conditioning must match the latent batch and size."
            )
        times = torch.as_tensor(
            timestep, device=hidden_states.device, dtype=torch.float32
        ).reshape(-1)
        times = times.expand(batch)
        # Training computes the sinusoid in float64 and feeds float32 into its MLP.
        half = self.config.freq_dim // 2
        freq = 10000 ** (
            -torch.arange(half, device=times.device, dtype=torch.float64) / half
        )
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(
            embedding.to(self.time_embedding[0].weight.dtype)
        ).float()
        mod = self.shared_block_modulation(
            time.to(self.shared_block_modulation.weight.dtype)
        ).view(batch, 6, -1)
        head = self.video_head_modulation(
            time.to(self.video_head_modulation.weight.dtype)
        ).view(batch, 2, -1)
        if prepared_conditioning is None:
            device_type = hidden_states.device.type
            dtype = (
                torch.get_autocast_dtype(device_type)
                if torch.is_autocast_enabled(device_type)
                else self.video_embedder.weight.dtype
            )
            prepared_conditioning = self.prepare_conditioning(
                encoder_hidden_states,
                encoder_attention_mask,
                height,
                width,
                dtype,
            )
        outputs = (
            torch.empty_like(hidden_states, dtype=torch.float32)
            if len(prepared_conditioning.groups) > 1
            else None
        )
        for group in prepared_conditioning.groups:
            indices = group.indices
            if indices is None:
                group_video, group_mod, group_head = hidden_states, mod, head
            else:
                group_video, group_mod, group_head = (
                    tensor.index_select(0, indices)
                    for tensor in (hidden_states, mod, head)
                )
            group_video = self.video_embedder(
                group_video.flatten(2)
                .transpose(1, 2)
                .to(self.video_embedder.weight.dtype)
                .contiguous()
            )
            x = torch.cat((group_video, group.text), 1)
            rope = group.rope
            layer_kwargs = {}
            if self._sequence_parallel:
                x, shard = shard_seq(x)
                rope = tuple(shard_like(part, shard) for part in rope)
                layer_kwargs["attn_mask_meta"] = tail_attn_meta(
                    shard, x.shape[0], x.device
                )
            for layer_index, layer in enumerate(self.layers):
                if layer_index not in skip_layers:
                    x = layer(x, rope, group_mod, **layer_kwargs)
            if self._sequence_parallel:
                x = gather_seq(x, shard.orig_len)
            x = self.final_norm_video(
                x[:, : height * width].to(self.final_norm_video.weight.dtype)
            )
            x = x * (1 + group_head[:, 1, None]) + group_head[:, 0, None]
            x = self.final_linear_video(x.to(self.final_linear_video.weight.dtype))
            sample = (
                x.transpose(1, 2)
                .reshape(x.shape[0], -1, height, width)
                .to(dtype=torch.float32, memory_format=torch.contiguous_format)
            )
            if indices is None:
                return sample
            outputs.index_copy_(0, indices, sample)
        return outputs

    def post_load_weights(self):
        self.rope.reset_bands()


EntryClass = MOSSImageTransformer
