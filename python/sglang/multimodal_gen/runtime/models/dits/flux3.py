# Copyright 2026 Black Forest Labs. All rights reserved.
#
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
# SPDX-License-Identifier: Apache-2.0
"""FLUX 3 ``JointSingleSeq`` DiT.

Adapted from the FLUX Action reference implementation
(https://github.com/black-forest-labs/flux-action,
``flux_action/models/transformer.py``). The module tree matches the released
checkpoints, so tensors load without renaming.

Every content stream (``video``, ``video_cond``, an action modality, ...) and
the text context first run through their own ``depth`` mode blocks
(self-attention within the stream). The text context and all active streams are
then concatenated and processed by ``depth_single_blocks`` joint blocks with
shared weights and per-stream modulation. Each stream carries its own
timesteps, so conditioning streams sit at ``t = 0`` while targets are noised.

Besides the reference-compatible :meth:`Flux3Transformer.forward`, the model
exposes the pieces separately (:meth:`encode_context`, :meth:`encode_stream`,
:meth:`denoise`) so pipelines can cache everything that does not depend on the
noised streams: the text context per caption, and the conditioning streams per
request.
"""

from __future__ import annotations

import math
import os
from typing import Any

import msgspec
import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.kernels.ops.diffusion import (
    BitExactFusionGate,
    can_use_fused_inplace_qknorm_rope,
    can_use_fused_layernorm_modulate,
    fused_inplace_qknorm_rope,
    fused_layernorm_modulate_raw,
    fused_packed_silu_mul_bitexact,
    is_plain_layer_norm,
    residual_gate_add,
)
from sglang.multimodal_gen.configs.models.dits.flux3 import (
    Flux3ArchConfig,
    Flux3DiTConfig,
)
from sglang.multimodal_gen.runtime.distributed import (
    divide,
    get_sp_world_size,
    get_tp_rank,
    get_tp_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.multimodal_gen.runtime.distributed.sp_shard_utils import (
    SpShard,
    build_shard_plan,
    gather_seq,
    shard_like,
    tail_attn_meta,
)
from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.layers.linear import (
    LinearBase,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

Modulation = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
TEXT_STREAM = "txt"


def rope_cos_sin(
    ids: torch.Tensor, axes_dim: tuple[int, ...], theta: int
) -> torch.Tensor:
    """Position ids ``(B, L, n_axes)`` -> fp32 ``(B, L, head_dim)`` rows ``[cos | sin]``.

    Pair ``i`` (channels ``2i, 2i + 1``) is rotated by angle ``i``; the angles of
    the axes are concatenated in order (FLUX-style interleaved RoPE).
    """
    angles = []
    for axis, dim in enumerate(axes_dim):
        scale = torch.arange(0, dim, 2, dtype=torch.float64, device=ids.device) / dim
        omega = 1.0 / (theta**scale)
        angles.append(torch.einsum("...n,d->...nd", ids[..., axis], omega))
    angles = torch.cat(angles, dim=-1)
    return torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1).float()


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos_sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate adjacent channel pairs of ``q``/``k`` ``(B, L, H, D)`` in fp32."""
    cos, sin = cos_sin[:, :, None].chunk(2, dim=-1)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        pairs = x.float().reshape(*x.shape[:-1], -1, 2)
        even = cos * pairs[..., 0] + (-sin) * pairs[..., 1]
        odd = sin * pairs[..., 0] + cos * pairs[..., 1]
        return torch.stack((even, odd), dim=-1).reshape_as(x).to(x.dtype)

    return rotate(q), rotate(k)


# Fused fast paths. The first two are bit-exact against the eager chain and
# verified per signature; QK-norm + RoPE is fused at bf16 rounding level.
_LN_MODULATE = BitExactFusionGate("FLUX 3 fused LN+modulate", per_signature=True)
_SWIGLU = BitExactFusionGate("FLUX 3 fused SwiGLU", per_signature=True)


def _eager_fast_path_allowed(x: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dtype is torch.bfloat16
        and not torch.compiler.is_compiling()
        and not torch.cuda.is_current_stream_capturing()
    )


def _norm_modulate(
    norm: nn.LayerNorm, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """``(1 + scale) * LN(x) + shift`` with ``(B, 1, D)`` modulation."""
    scale_row, shift_row = scale[:, 0], shift[:, 0]
    if (
        _LN_MODULATE.disabled
        or not _eager_fast_path_allowed(x)
        or not is_plain_layer_norm(norm, x.shape[-1])
        or not can_use_fused_layernorm_modulate(x, scale_row, shift_row)
    ):
        return (1 + scale) * norm(x) + shift
    sig = (x.device, x.shape[0], x.shape[-1], norm.eps)
    try:
        out = fused_layernorm_modulate_raw(x, scale_row, shift_row, norm.eps)
    except Exception as exc:
        _LN_MODULATE.on_exception(exc, logger=logger)
        return (1 + scale) * norm(x) + shift
    if _LN_MODULATE.is_verified(sig):
        return out
    return _LN_MODULATE.accept_or_fallback(
        out,
        (1 + scale) * norm(x) + shift,
        sig=sig,
        logger=logger,
        mismatch_msg="FLUX 3 fused LN+modulate is not bit-exact here; using eager",
    )


def _swiglu(packed: torch.Tensor) -> torch.Tensor:
    """``silu(gate) * value`` of a packed ``[gate | value]`` projection."""
    gate, value = packed.chunk(2, dim=-1)
    if _SWIGLU.disabled or not _eager_fast_path_allowed(packed):
        return F.silu(gate) * value
    sig = (packed.device, packed.shape[-1], packed.stride(-2))
    try:
        out = fused_packed_silu_mul_bitexact(packed)
    except Exception as exc:
        _SWIGLU.on_exception(exc, logger=logger)
        return F.silu(gate) * value
    if _SWIGLU.is_verified(sig):
        return out
    return _SWIGLU.accept_or_fallback(
        out,
        F.silu(gate) * value,
        sig=sig,
        logger=logger,
        mismatch_msg="FLUX 3 fused SwiGLU is not bit-exact here; using eager",
    )


def _fused_qknorm_rope_enabled(q: torch.Tensor, head_dim: int) -> bool:
    return (
        _eager_fast_path_allowed(q)
        and os.getenv("SGLANG_ENABLE_FUSED_QKNORM_ROPE", "1").lower()
        not in ("0", "false", "off", "no")
        and can_use_fused_inplace_qknorm_rope(
            head_dim=head_dim,
            rope_dim=head_dim,
            is_neox=False,
            dtype=q.dtype,
            cache_dtype=torch.float32,
            round_norm_before_rope=False,
        )
    )


def timestep_embedding(t: torch.Tensor, dim: int = 256) -> torch.Tensor:
    """Sinusoidal embedding of ``t`` in ``[0, 1]`` (scaled by 1000), fp32."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / half
    )
    args = (1000.0 * t)[..., None].float() * freqs
    return torch.cat((torch.cos(args), torch.sin(args)), dim=-1)


class Flux3MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=False)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class Flux3RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        rrms = torch.rsqrt(torch.mean(x_float**2, dim=-1, keepdim=True) + 1e-6)
        return (x_float * rrms).to(dtype) * self.scale


class Flux3QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = Flux3RMSNorm(dim)
        self.key_norm = Flux3RMSNorm(dim)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.query_norm(q).to(v), self.key_norm(k).to(v)


class Flux3Modulation(nn.Module):
    """shift / scale / gate from the timestep vector (shared by all blocks of a phase)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.lin = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

    def forward(self, vec: torch.Tensor) -> Modulation:
        out = self.lin(F.silu(vec))
        if out.ndim == 2:
            out = out[:, None, :]
        shift, scale, gate = out.chunk(3, dim=-1)
        return shift, scale, gate


FP8_E4M3_MAX = 448.0


def quantize_fp8_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(M, K)`` -> E4M3 values and one fp32 scale per row (amax / 448)."""
    rows = x.float()
    scale = (rows.abs().amax(dim=1) / FP8_E4M3_MAX).clamp(min=1e-12)
    quantized = (rows / scale[:, None]).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    return quantized.to(torch.float8_e4m3fn), scale


class Flux3Fp8RowwiseLinear(nn.Module):
    """FP8 "rowwise" linear of the FLUX 3 FP8r checkpoints.

    E4M3 weights with one fp32 scale per output row; activations are quantized
    per token on the fly and multiplied with ``torch._scaled_mm`` (fp32 fast
    accumulation, bf16 output), as in the reference FP8r inference path.
    ``tuple_output`` mirrors the ``(output, bias)`` convention of SGLang's
    parallel linears so the module can replace either kind.
    """

    # _scaled_mm needs M padded to a multiple of 16.
    ROW_ALIGNMENT = 16

    def __init__(
        self, weight: torch.Tensor, weight_scale: torch.Tensor, tuple_output: bool
    ):
        super().__init__()
        if weight.dtype != torch.float8_e4m3fn or weight_scale.shape != (
            weight.shape[0],
        ):
            raise ValueError(
                "expected an E4M3 weight with one fp32 scale per output row"
            )
        self.out_features, self.in_features = weight.shape
        self.tuple_output = tuple_output
        # Parameters (not buffers) so that layerwise offload streams them.
        self.weight = nn.Parameter(weight.contiguous(), requires_grad=False)
        self.weight_scale = nn.Parameter(
            weight_scale.float().contiguous(), requires_grad=False
        )

    def forward(self, x: torch.Tensor):
        leading = x.shape[:-1]
        flat = x.reshape(-1, self.in_features).contiguous()
        rows = flat.shape[0]
        pad = -rows % self.ROW_ALIGNMENT
        if pad:
            flat = F.pad(flat, (0, 0, 0, pad))
        activation, activation_scale = quantize_fp8_rowwise(flat)
        out = torch._scaled_mm(
            activation,
            self.weight.T,
            activation_scale[:, None],
            self.weight_scale[None, :],
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )[:rows]
        out = out.reshape(*leading, self.out_features)
        return (out, None) if self.tuple_output else out


class Flux3LastLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        )

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        if vec.ndim == 2:
            vec = vec[:, None, :]
        x = self.norm_final(x)
        projection = self.adaLN_modulation[1]
        if isinstance(projection, Flux3Fp8RowwiseLinear):
            shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
            return self.linear(x * (scale + 1) + shift)
        # BF16: two half projections avoid materializing the (L, 2 * hidden) product.
        activated = self.adaLN_modulation[0](vec)
        shift_w, scale_w = projection.weight.chunk(2)
        x.mul_(F.linear(activated, scale_w).add_(1))
        x.add_(F.linear(activated, shift_w))
        return self.linear(x)


# Streams shorter than this keep their mode blocks replicated under sequence
# parallelism (text, action and state tokens); arbitrary, not tuned.
SP_MIN_STREAM_TOKENS = 256


class Flux3SequenceShard(msgspec.Struct, frozen=True):
    """This rank's slice of a sequence-parallel sequence (Ulysses or K/V gather)."""

    shard: SpShard
    # Tail-pad varlen meta for USPAttention (None when the split is even).
    attn_mask_meta: dict[str, Any] | None


def _sequence_shard(
    length: int, like: torch.Tensor, min_tokens: int = 0
) -> Flux3SequenceShard | None:
    """The SP split of a ``length``-token sequence, or None to keep it replicated."""
    if get_sp_world_size() == 1 or length < max(min_tokens, get_sp_world_size()):
        return None
    shard = build_shard_plan(length)
    return Flux3SequenceShard(
        shard=shard,
        attn_mask_meta=tail_attn_meta(shard, like.shape[0], like.device),
    )


def _shard_mod(mod: Modulation, shard: SpShard) -> Modulation:
    """Per-token modulation ``(B, L, D)`` follows the tokens; ``(B, 1, D)`` broadcasts."""
    return tuple(m if m.shape[1] == 1 else shard_like(m, shard, dim=1) for m in mod)


def _local_segments(
    lengths: list[int], mods: list[Modulation], shard: SpShard
) -> tuple[list[int], list[Modulation]]:
    """Segments of the joint sequence inside this rank's shard.

    The tail pad of the last rank becomes a segment with zero modulation, so
    its rows pass through unchanged (attention masks them out).
    """
    start = shard.sp_rank * shard.local_len
    end = start + shard.local_real_len
    local_lengths, local_mods = [], []
    pos = 0
    for length, mod in zip(lengths, mods):
        lo, hi = max(pos, start), min(pos + length, end)
        if lo < hi:
            local_lengths.append(hi - lo)
            local_mods.append(
                tuple(m if m.shape[1] == 1 else m[:, lo - pos : hi - pos] for m in mod)
            )
        pos += length
    if shard.local_pad:
        zero = mods[-1][0].new_zeros(mods[-1][0].shape[0], 1, mods[-1][0].shape[-1])
        local_lengths.append(shard.local_pad)
        local_mods.append((zero, zero, zero))
    return local_lengths, local_mods


class Flux3Block(nn.Module):
    """Parallel attention + SwiGLU MLP block with QK-RMSNorm and 4-axis RoPE.

    Used both as a per-stream mode block (one modulation for the whole input)
    and as a joint block (one modulation per segment of the joint sequence).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        quant_config: QuantizationConfig | None = None,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # Tensor parallel as in the FLUX 3 native DiT: each rank owns a slice of
        # the heads and of the MLP channels; one all-reduce per block.
        self.tp_size = get_tp_world_size()
        self.local_heads = divide(num_heads, self.tp_size)
        self.local_hidden = self.local_heads * self.head_dim
        self.local_mlp_hidden = divide(self.mlp_hidden_dim, self.tp_size)

        # q, k, v and the MLP input share one GEMM; the checkpoint's separate
        # tensors are concatenated at load time (see Flux3ArchConfig). Gate and
        # value are separate partitions so each rank keeps matching halves.
        self.qkv_mlp = MergedColumnParallelLinear(
            hidden_size,
            [hidden_size] * 3 + [self.mlp_hidden_dim] * 2,
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_mlp",
        )

        def row_linear(name: str, in_features: int) -> RowParallelLinear:
            return RowParallelLinear(
                in_features,
                hidden_size,
                bias=False,
                input_is_parallel=True,
                reduce_results=False,
                quant_config=quant_config,
                prefix=f"{prefix}.{name}",
            )

        self.attn_out = row_linear("attn_out", hidden_size)
        self.mlp_out = row_linear("mlp_out", self.mlp_hidden_dim)
        self.norm = Flux3QKNorm(self.head_dim)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = USPAttention(
            num_heads=self.local_heads,
            head_size=self.head_dim,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.attn",
        )

    def _mix(
        self,
        modulated: torch.Tensor,
        rope: torch.Tensor,
        sp: Flux3SequenceShard | None,
    ) -> torch.Tensor:
        batch, length, _ = modulated.shape
        heads = self.local_heads
        q, k, v, mlp = self.qkv_mlp(modulated)[0].split(
            (
                self.local_hidden,
                self.local_hidden,
                self.local_hidden,
                2 * self.local_mlp_hidden,
            ),
            dim=-1,
        )
        q = q.view(batch, length, heads, self.head_dim)
        k = k.view(batch, length, heads, self.head_dim)
        v = v.view(batch, length, heads, self.head_dim)
        if _fused_qknorm_rope_enabled(q, self.head_dim):
            # In place on the fused projection: k follows q's heads in each row.
            fused_inplace_qknorm_rope(
                q=q.view(-1, heads, self.head_dim),
                k=k.view(-1, heads, self.head_dim),
                q_weight=self.norm.query_norm.scale,
                k_weight=self.norm.key_norm.scale,
                cos_sin_cache=rope.reshape(-1, self.head_dim),
                positions=torch.arange(batch * length, device=q.device),
                is_neox=False,
                eps=1e-6,
                round_norm_before_rope=False,
            )
        else:
            q, k = self.norm(q, k, v)
            q, k = apply_rope(q, k, rope)
        if sp is None:
            # Replicated input: every rank already holds the whole stream.
            attended = self.attn(q, k, v, skip_sequence_parallel_override=True)
        else:
            # q/k/v are strided views of the fused projection; the SP exchanges
            # (all-to-all, K/V all-gather) need dense tensors.
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
            attended = self.attn(q, k, v, attn_mask_meta=sp.attn_mask_meta)
        attended = attended.reshape(batch, length, self.local_hidden)
        out = self.attn_out(attended)[0] + self.mlp_out(_swiglu(mlp))[0]
        if self.tp_size > 1:
            # One reduce of the summed partials, not one per branch (the order
            # the FLUX 3 reference TP uses).
            out = tensor_model_parallel_all_reduce(out)
        return out

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        mods: list[Modulation],
        lengths: list[int] | None = None,
        sp: Flux3SequenceShard | None = None,
    ) -> torch.Tensor:
        """Segment ``i`` of ``x`` (``lengths[i]`` tokens) is modulated by ``mods[i]``.

        A mode block has a single segment (``lengths=None``); a joint block has
        one per stream. With ``sp``, ``x`` is this rank's sequence shard.
        Blocks are always entered through ``__call__`` so that forward hooks
        (layerwise offload) see every block.
        """
        if lengths is None:
            shift, scale, gate = mods[0]
            modulated = _norm_modulate(self.pre_norm, x, shift=shift, scale=scale)
            return residual_gate_add(x, self._mix(modulated, rope, sp), gate)
        segments = torch.split(x, lengths, dim=1)
        modulated = torch.cat(
            [
                _norm_modulate(self.pre_norm, seg, shift=m[0], scale=m[1])
                for seg, m in zip(segments, mods)
            ],
            dim=1,
        )
        output = torch.split(self._mix(modulated, rope, sp), lengths, dim=1)
        return torch.cat(
            [
                residual_gate_add(seg, out.contiguous(), m[2])
                for seg, out, m in zip(segments, output, mods)
            ],
            dim=1,
        )


class Flux3SegmentState(msgspec.Struct, frozen=True):
    """A stream (or the text context) after its mode blocks, ready for the joint blocks."""

    name: str
    hidden: torch.Tensor  # (B, L, hidden)
    rope: torch.Tensor  # (B, L, head_dim) fp32 [cos | sin]
    vec: torch.Tensor  # (B, 1 | L, hidden): timestep vector of the stream
    joint_mod: Modulation  # modulation of the stream in the joint blocks

    @property
    def length(self) -> int:
        return self.hidden.shape[1]


class Flux3Transformer(BaseDiT, LayerwiseOffloadableModuleMixin):
    _fsdp_shard_conditions = [
        lambda name, module: isinstance(module, Flux3Block),
    ]
    _compile_conditions = _fsdp_shard_conditions
    _fsdp_forward_methods = ("encode_context", "encode_stream", "denoise")
    param_names_mapping = Flux3ArchConfig().param_names_mapping
    reverse_param_names_mapping = {}
    _supported_attention_backends = {
        AttentionBackendEnum.FA,
        AttentionBackendEnum.TORCH_SDPA,
        AttentionBackendEnum.SAGE_ATTN,
        AttentionBackendEnum.SAGE_ATTN_3,
        AttentionBackendEnum.AITER,
    }

    def __init__(
        self,
        config: Flux3DiTConfig,
        hf_config: dict[str, Any] | None = None,
        quant_config: QuantizationConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config or {}, **kwargs)
        arch: Flux3ArchConfig = config.arch_config
        self.arch = arch
        self.param_names_mapping = arch.param_names_mapping
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.num_channels_latents
        self.in_channels = dict(arch.in_channels)
        self.sequence = dict(arch.sequence)
        self.depth = arch.depth
        self.axes_dim = tuple(arch.axes_dim)
        self.theta = arch.theta
        hidden = arch.hidden_size

        self.emb_in = nn.ModuleDict(
            {m: nn.Linear(c, hidden, bias=False) for m, c in self.in_channels.items()}
        )
        self.txt_in = nn.Linear(arch.context_in_dim, hidden, bias=False)
        self.time_in = Flux3MLPEmbedder(256, hidden)
        self.vector_in = (
            Flux3MLPEmbedder(arch.vec_in_dim, hidden)
            if arch.vec_in_dim is not None
            else None
        )
        streams = sorted(self.in_channels)
        self.early_stream_modulations = nn.ModuleDict(
            {m: Flux3Modulation(hidden) for m in (*streams, TEXT_STREAM)}
        )
        self.single_stream_modulations = nn.ModuleDict(
            {m: Flux3Modulation(hidden) for m in (*streams, TEXT_STREAM)}
        )

        def block(prefix: str) -> Flux3Block:
            return Flux3Block(
                hidden,
                arch.num_attention_heads,
                arch.mlp_ratio,
                quant_config=quant_config,
                supported_attention_backends=self._supported_attention_backends,
                prefix=prefix,
            )

        self.content_mode_blocks = nn.ModuleDict(
            {
                m: nn.ModuleList(
                    block(f"content_mode_blocks.{m}.{i}") for i in range(arch.depth)
                )
                for m in streams
            }
        )
        self.txt_mode_blocks = nn.ModuleList(
            block(f"txt_mode_blocks.{i}") for i in range(arch.depth)
        )
        self.single_blocks = nn.ModuleList(
            block(f"single_blocks.{i}") for i in range(arch.depth_single_blocks)
        )
        self.final_layer = nn.ModuleDict(
            {m: Flux3LastLayer(hidden, c) for m, c in self.in_channels.items()}
        )
        self.layer_names = [
            "txt_mode_blocks",
            *(f"content_mode_blocks.{m}" for m in streams),
            "single_blocks",
        ]
        self.__post_init__()

    # ------------------------------------------------------------------ pieces
    @property
    def dtype(self) -> torch.dtype:
        return _compute_dtype(self.txt_in)

    def rope(self, ids: torch.Tensor) -> torch.Tensor:
        return rope_cos_sin(ids, self.axes_dim, self.theta)

    def _timestep_vector(
        self, timesteps: torch.Tensor, vector: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``timesteps`` ``(B,)`` or ``(B, L)`` in ``[0, 1]`` -> ``(B, 1 | L, hidden)``."""
        if timesteps.ndim == 1:
            timesteps = timesteps[:, None]
        weight_dtype = _compute_dtype(self.time_in.in_layer)
        with torch.autocast(device_type=timesteps.device.type, enabled=False):
            vec = self.time_in(timestep_embedding(timesteps).to(weight_dtype))
        vec = vec.to(self.dtype)
        if self.vector_in is not None:
            if vector is None:
                vector = torch.zeros(
                    timesteps.shape[0],
                    self.vector_in.in_layer.in_features,
                    device=timesteps.device,
                    dtype=self.dtype,
                )
            vec = vec + self.vector_in(vector.to(self.dtype))[:, None, :]
        return vec

    def encode_context(
        self,
        ctx: torch.Tensor,
        ctx_ids: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        vector: torch.Tensor | None = None,
    ) -> Flux3SegmentState:
        """Text context ``(B, L, context_in_dim)`` through ``txt_in`` and the text mode blocks."""
        if timesteps is None:
            timesteps = torch.zeros(ctx.shape[0], device=ctx.device)
        vec = self._timestep_vector(timesteps=timesteps, vector=vector)
        rope = self.rope(ctx_ids)
        early = self.early_stream_modulations[TEXT_STREAM](vec)
        hidden = self.txt_in(ctx.to(self.dtype))
        for block in self.txt_mode_blocks:
            hidden = block(hidden, rope, [early])
        return Flux3SegmentState(
            name=TEXT_STREAM,
            hidden=hidden,
            rope=rope,
            vec=vec,
            joint_mod=self.single_stream_modulations[TEXT_STREAM](vec),
        )

    def encode_stream(
        self,
        name: str,
        x: torch.Tensor,
        ids: torch.Tensor,
        timesteps: torch.Tensor,
        vector: torch.Tensor | None = None,
        rope: torch.Tensor | None = None,
    ) -> Flux3SegmentState:
        """Stream tokens ``(B, L, in_channels[name])`` through ``emb_in`` and the stream's mode blocks."""
        vec = self._timestep_vector(timesteps=timesteps, vector=vector)
        rope = self.rope(ids) if rope is None else rope
        early = self.early_stream_modulations[name](vec)
        hidden = self.emb_in[name](x.to(self.dtype))
        # Long streams run their mode blocks sequence-parallel; the state keeps
        # the full stream so cached conditioning is layout independent.
        sp = _sequence_shard(hidden.shape[1], hidden, min_tokens=SP_MIN_STREAM_TOKENS)
        if sp is None:
            for block in self.content_mode_blocks[name]:
                hidden = block(hidden, rope, [early])
        else:
            local = shard_like(hidden, sp.shard)
            local_rope = shard_like(rope, sp.shard)
            local_early = _shard_mod(early, sp.shard)
            for block in self.content_mode_blocks[name]:
                local = block(local, local_rope, [local_early], sp=sp)
            hidden = gather_seq(local, sp.shard.orig_len)
        return Flux3SegmentState(
            name=name,
            hidden=hidden,
            rope=rope,
            vec=vec,
            joint_mod=self.single_stream_modulations[name](vec),
        )

    def _joint(
        self, context: Flux3SegmentState, streams: list[Flux3SegmentState]
    ) -> list[torch.Tensor]:
        """Run the joint blocks over ``[context, *streams]``; returns the hidden state of each stream."""
        segments = [context, *streams]
        lengths = [s.length for s in segments]
        mods = [s.joint_mod for s in segments]
        rope = torch.cat([s.rope for s in segments], dim=1)
        x = torch.cat([s.hidden for s in segments], dim=1)
        sp = _sequence_shard(x.shape[1], x)
        if sp is None:
            for block in self.single_blocks:
                x = block(x, rope, mods, lengths)
        else:
            local = shard_like(x, sp.shard)
            local_rope = shard_like(rope, sp.shard)
            local_lengths, local_mods = _local_segments(lengths, mods, sp.shard)
            for block in self.single_blocks:
                local = block(local, local_rope, local_mods, local_lengths, sp=sp)
            x = gather_seq(local, sp.shard.orig_len)
        return list(torch.split(x, lengths, dim=1)[1:])

    def denoise(
        self,
        context: Flux3SegmentState,
        streams: list[Flux3SegmentState],
        targets: list[str],
    ) -> dict[str, torch.Tensor]:
        """Joint blocks + output heads of the ``targets`` streams (by name)."""
        hidden = self._joint(context, streams)
        by_name = {s.name: (h, s) for h, s in zip(hidden, streams)}
        return {
            name: self.final_layer[name](by_name[name][0], by_name[name][1].vec)
            for name in targets
        }

    # ------------------------------------------------------------------ reference API
    def forward(
        self,
        ctx: torch.Tensor,
        ctx_ids: torch.Tensor,
        vector: torch.Tensor | None = None,
        timesteps_ctx: torch.Tensor | None = None,
        **kwargs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict every stream in ``kwargs`` (``x_<s>``, ``x_<s>_ids``, ``x_<s>_timesteps``).

        Matches the reference ``JointSingleSeq.forward`` for dense batches:
        streams absent from ``kwargs`` are skipped entirely (in the reference
        they only feed their own discarded heads).
        """
        context = self.encode_context(
            ctx=ctx, ctx_ids=ctx_ids, timesteps=_uniform(timesteps_ctx), vector=vector
        )
        streams, names = [], []
        for key, stream in self.sequence.items():
            if key not in kwargs:
                continue
            streams.append(
                self.encode_stream(
                    name=stream,
                    x=kwargs[key],
                    ids=kwargs[f"{key}_ids"],
                    timesteps=_uniform(kwargs[f"{key}_timesteps"]),
                    vector=vector,
                )
            )
            names.append(key)
        outputs = self.denoise(
            context=context, streams=streams, targets=[s.name for s in streams]
        )
        return {key: outputs[s.name] for key, s in zip(names, streams)}


def _compute_dtype(linear: nn.Module) -> torch.dtype:
    """Activation dtype of a linear: its weight dtype, bf16 for FP8 weights."""
    if isinstance(linear, Flux3Fp8RowwiseLinear):
        return torch.bfloat16
    return linear.weight.dtype


def load_fp8r_checkpoint(
    model: Flux3Transformer, state_dict: dict[str, torch.Tensor]
) -> None:
    """Load a native FP8r checkpoint into ``model`` (built on the meta device).

    Every linear whose checkpoint weight is E4M3 (with ``.weight_scale``)
    becomes a :class:`Flux3Fp8RowwiseLinear`; BF16 tensors (the embodiment's
    action boundary layers and all norms) load as they are.
    """
    from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping

    mapping = get_param_names_mapping(model.param_names_mapping)
    weights: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}
    fused: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor | None]]] = {}
    fused_counts: dict[str, int] = {}
    for name, tensor in state_dict.items():
        is_scale = name.endswith(".weight_scale")
        weight_name = name.removesuffix("_scale") if is_scale else name
        target, index, count = mapping(weight_name)
        if index is None:
            (scales if is_scale else weights)[target] = tensor
            continue
        weight, scale = fused.setdefault(target, {}).get(index, (None, None))
        fused[target][index] = (weight, tensor) if is_scale else (tensor, scale)
        fused_counts[target] = count
    for target, parts in fused.items():
        if sorted(parts) != list(range(fused_counts[target])) or any(
            w is None for w, _ in parts.values()
        ):
            raise ValueError(f"{target}: checkpoint lacks fused parts {sorted(parts)}")
        ordered = [parts[i] for i in sorted(parts)]
        weights[target] = torch.cat([w for w, _ in ordered])
        part_scales = [sc for _, sc in ordered if sc is not None]
        if part_scales:
            if len(part_scales) != len(ordered):
                raise ValueError(f"{target}: mixed FP8 and BF16 parts cannot be fused")
            scales[target] = torch.cat(part_scales)
    for name, scale in scales.items():
        module_path = name.removesuffix(".weight")
        parent_path, _, child = module_path.rpartition(".")
        parent = model.get_submodule(parent_path)
        original = getattr(parent, child)
        weight = _tp_partition(original, weights.pop(name), is_scale=False)
        if weight.shape != original.weight.shape:
            raise ValueError(
                f"{name}: checkpoint shape {tuple(weight.shape)} does not match "
                f"{tuple(original.weight.shape)}"
            )
        setattr(
            parent,
            child,
            Flux3Fp8RowwiseLinear(
                weight,
                _tp_partition(original, scale, is_scale=True),
                tuple_output=isinstance(original, LinearBase),
            ),
        )
    for name, tensor in weights.items():
        owner = model.get_submodule(name.rpartition(".")[0])
        weights[name] = _tp_partition(owner, tensor, is_scale=False)
    missing, unexpected = model.load_state_dict(weights, strict=False, assign=True)
    fp8_params = {
        n for n, _ in model.named_parameters() if n.removesuffix("_scale") in scales
    }
    missing = [n for n in missing if n not in fp8_params]
    if missing or unexpected:
        raise ValueError(
            f"FP8r checkpoint mismatch: missing {missing}, unexpected {unexpected}"
        )


def _tp_partition(
    module: nn.Module, tensor: torch.Tensor, *, is_scale: bool
) -> torch.Tensor:
    """This rank's slice of a full checkpoint weight (or per-row scale) of ``module``."""
    tp_size = get_tp_world_size()
    if tp_size == 1:
        return tensor
    rank = get_tp_rank()
    if isinstance(module, MergedColumnParallelLinear):
        parts = tensor.split(module.output_sizes)
        return torch.cat([part.chunk(tp_size)[rank] for part in parts])
    if isinstance(module, RowParallelLinear) and not is_scale:
        # Row scales stay whole: they scale output rows, which are not split.
        return tensor.chunk(tp_size, dim=1)[rank].contiguous()
    return tensor


def _uniform(timesteps: torch.Tensor | None) -> torch.Tensor | None:
    """Collapse per-token timesteps ``(B, L)`` that are constant along ``L`` to ``(B,)``."""
    if timesteps is None or timesteps.ndim == 1:
        return timesteps
    if bool((timesteps == timesteps[:, :1]).all()):
        return timesteps[:, 0]
    return timesteps


EntryClass = Flux3Transformer
