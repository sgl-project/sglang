# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""Native sglang-diffusion implementation of the Kandinsky6 T2VA/IT2VA transformer.

Ported from FastVideo's ``fastvideo/models/dits/kandinsky6.py``, which mirrors
``kandinsky5.py``'s structure and extends it with a second, parallel "audio"
tower plus a fused video<->audio decoder block
(``Kandinsky6FusedTransformerDecoderBlock``), following the reference
``diffusers.models.transformers.transformer_kandinsky6`` module.

Video latents are represented as ``[B, T, H, W, C]`` (channel-last) tensors and
audio latents as ``[B, A, D]`` tensors -- ordinary batched tensors, not the
packed ragged sequence with ``cu_seqlens`` boundaries the diffusers reference
uses. RoPE position tensors therefore carry an explicit/broadcastable batch
dimension instead of being derived from ``cu_seqlens``.

Attention: the dense/SDPA path (``attention_engine != "nabla"``) is the only
verified path -- no published Kandinsky6 checkpoint has been observed using
NABLA block-sparse attention, and this codebase has no NABLA attention
backend (no ``AttentionBackendEnum.NABLA_ATTN`` member, no
``torch.nn.attention.flex_attention``-based block-mask kernel). Rather than
guess-port FastVideo's ``nablaT_v2``/``flex_attention`` fallback logic with
nothing to verify it against, ``attention_engine == "nabla"`` raises
``NotImplementedError`` at model construction time. The ``use_nabla``
construction-time flag and ``sparse_params`` forward-time plumbing are still
threaded through every layer so a real NABLA backend can be dropped in later
without an architectural rewrite.

TP/SP: ``Kandinsky6Attention``'s Q/K/V and output projections are native
TP-sharded (``ColumnParallelLinear``/``RowParallelLinear``, matching
``wanvideo.py``'s/``krea2.py``'s pattern), and attention runs through
``USPAttention`` for Ulysses/ring sequence parallelism, instead of a
replicated ``ReplicatedLinear`` + single-GPU ``LocalAttention``. The
feed-forward was already TP-sharded. ``Kandinsky6Attention`` is reused for
five different roles (video self-attn, audio self-attn, two text
cross-attns, two video<->audio cross-attns with asymmetric ``kv_dim``); see
its docstring for how TP and SP are wired per role. TP preserves the model's
head layout for every role (each rank shards whole attention heads, a
per-head QK-norm, and an all-reduced output projection -- no role splits a
single head's channels across ranks). SP shards the video stream while
keeping the shorter audio and text streams replicated. Audio-to-video
cross-attention gathers the projected video K/V and removes tail padding;
video-to-audio and text cross-attention already have complete local K/V.
"""

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.kernels.ops.diffusion import apply_matrix_rope, residual_gate_fp32
from sglang.multimodal_gen.configs.models.dits.kandinsky6 import (
    Kandinsky6ArchConfig,
    Kandinsky6VideoAudioConfig,
)
from sglang.multimodal_gen.configs.models.fsdp import is_module_list_entry_in
from sglang.multimodal_gen.runtime.distributed import divide, get_tp_world_size
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_ctx,
    get_ulysses_ctx,
)
from sglang.multimodal_gen.runtime.distributed.sp_shard_utils import (
    gather_seq,
    shard_like,
    shard_seq,
    tail_attn_meta,
)
from sglang.multimodal_gen.runtime.layers.activation import get_act_fn
from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.layers.layernorm import LayerNormScaleShift
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
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
from sglang.srt.utils import add_prefix

logger = init_logger(__name__)

FRACTAL_PIXEL_SIZE = 8
_ARCH_CONFIG_DEFAULTS = Kandinsky6VideoAudioConfig().arch_config

# The plain (non-multimodal) T2V/I2V-parity path uses ``text_transformer_blocks``;
# the real multimodal (T2VA/IT2VA) path builds one independent 4-layer text
# tower per modality, named ``video_text_transformer_blocks`` /
# ``audio_text_transformer_blocks``. ``visual_transformer_blocks`` (x60) is
# shared by both paths. All four need individual-block FSDP/compile sharding.
_KANDINSKY6_BLOCK_CONTAINERS = (
    "text_transformer_blocks",
    "video_text_transformer_blocks",
    "audio_text_transformer_blocks",
    "visual_transformer_blocks",
)


def _is_kandinsky6_transformer_block(name: str, module: object) -> bool:
    return is_module_list_entry_in(name, _KANDINSKY6_BLOCK_CONTAINERS)


def _build_rotary_freqs(dim: int, max_period: float) -> torch.Tensor:
    return torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=dim, dtype=torch.float32)
        / dim
    )


def local_patching(
    x: torch.Tensor,
    shape: tuple[int, int, int, int],
    group_size: tuple[int, int, int],
    dim: int = 0,
) -> torch.Tensor:
    """Regroups a ``[..., T, H, W, ...]``-shaped tensor into local ``group_size``
    blocks along ``dim``. Only exercised by the non-multimodal / NABLA
    fractal-ordering path (see ``fractal_flatten``); ported for completeness.
    """
    _batch_size, duration, height, width = shape
    g1, g2, g3 = group_size
    x = x.reshape(
        *x.shape[:dim],
        duration // g1,
        g1,
        height // g2,
        g2,
        width // g3,
        g3,
        *x.shape[dim + 3 :],
    )
    x = x.permute(
        *range(len(x.shape[:dim])),
        dim,
        dim + 2,
        dim + 4,
        dim + 1,
        dim + 3,
        dim + 5,
        *range(dim + 6, len(x.shape)),
    )
    x = x.flatten(dim, dim + 2).flatten(dim + 1, dim + 3)
    return x


def local_merge(
    x: torch.Tensor,
    shape: tuple[int, int, int, int],
    group_size: tuple[int, int, int],
    dim: int = 0,
) -> torch.Tensor:
    """Inverse of ``local_patching``."""
    _batch_size, duration, height, width = shape
    g1, g2, g3 = group_size
    x = x.reshape(
        *x.shape[:dim],
        duration // g1,
        height // g2,
        width // g3,
        g1,
        g2,
        g3,
        *x.shape[dim + 2 :],
    )
    x = x.permute(
        *range(len(x.shape[:dim])),
        dim,
        dim + 3,
        dim + 1,
        dim + 4,
        dim + 2,
        dim + 5,
        *range(dim + 6, len(x.shape)),
    )
    x = x.flatten(dim, dim + 1).flatten(dim + 1, dim + 2).flatten(dim + 2, dim + 3)
    return x


def fractal_flatten(
    x: torch.Tensor,
    rope: torch.Tensor,
    shape: tuple[int, int, int, int],
    block_mask: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if block_mask:
        pixel_size = FRACTAL_PIXEL_SIZE
        x = local_patching(x, shape, (1, pixel_size, pixel_size), dim=1)
        rope = local_patching(rope, shape, (1, pixel_size, pixel_size), dim=1)
        x = x.flatten(1, 2)
        rope = rope.flatten(1, 2)
    else:
        x = x.flatten(1, 3)
        rope = rope.flatten(1, 3)
    return x, rope


def fractal_unflatten(
    x: torch.Tensor, shape: tuple[int, int, int, int], block_mask: bool = False
) -> torch.Tensor:
    if block_mask:
        pixel_size = FRACTAL_PIXEL_SIZE
        x = x.reshape(x.shape[0], -1, pixel_size**2, *x.shape[2:])
        x = local_merge(x, shape, (1, pixel_size, pixel_size), dim=1)
    else:
        x = x.reshape(*shape, *x.shape[2:])
    return x


class Kandinsky6TimeEmbeddings(nn.Module):
    """Sinusoidal timestep embedding -> Linear -> SiLU -> Linear."""

    def __init__(
        self,
        model_dim: int,
        time_dim: int,
        max_period: float = 10000.0,
        prefix: str = "",
    ):
        super().__init__()
        assert model_dim % 2 == 0
        self.model_dim = model_dim
        self.max_period = max_period
        # Plain attribute, not a registered buffer -- materialized on meta
        # device at construction and re-derived in post_load_weights() if
        # weights were loaded under a meta-device init context.
        self.freqs = _build_rotary_freqs(self.model_dim // 2, self.max_period)
        self.in_layer = ReplicatedLinear(
            model_dim, time_dim, bias=True, prefix=add_prefix("in_layer", prefix)
        )
        self.activation = nn.SiLU()
        self.out_layer = ReplicatedLinear(
            time_dim, time_dim, bias=True, prefix=add_prefix("out_layer", prefix)
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        # `self.freqs` is always fp32 (a plain attribute, not a registered buffer --
        # see __init__), so this sinusoidal embedding is always fp32, but `in_layer`'s
        # weight is whatever dtype the checkpoint loaded it as (bf16 under the default
        # loading config). `torch.autocast(device_type="cuda", dtype=torch.float32)`
        # does not bridge that gap -- it doesn't cast an already-materialized bf16
        # weight to fp32, and it's a no-op entirely on CPU/MPS (`device_type="cuda"`) --
        # so the first Linear call raised "mat1 and mat2 must have the same dtype".
        # Cast the embedding to the Linear's own weight dtype explicitly instead,
        # matching the diffusers reference's
        # `embed.to(get_parameter_dtype(self.timestep_embedder))`.
        args = torch.outer(time, self.freqs.to(device=time.device))
        time_embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        time_embed = time_embed.to(self.in_layer.weight.dtype)
        time_embed, _ = self.in_layer(time_embed)
        time_embed = self.activation(time_embed)
        time_embed, _ = self.out_layer(time_embed)
        return time_embed


class Kandinsky6TextEmbeddings(nn.Module):
    """Linear + LayerNorm projection, reused for text tokens and audio latents."""

    def __init__(self, in_dim: int, model_dim: int, prefix: str = ""):
        super().__init__()
        self.in_layer = ReplicatedLinear(
            in_dim, model_dim, bias=True, prefix=add_prefix("in_layer", prefix)
        )
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.in_layer(x)
        return self.norm(x).type_as(x)


class Kandinsky6VisualEmbeddings(nn.Module):
    """Patchifies ``[B, T, H, W, C]`` video into non-overlapping ``patch_size`` patches."""

    def __init__(
        self,
        visual_dim: int,
        model_dim: int,
        patch_size: tuple[int, int, int],
        prefix: str = "",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_layer = ReplicatedLinear(
            math.prod(patch_size) * visual_dim,
            model_dim,
            prefix=add_prefix("in_layer", prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, duration, height, width, dim = x.shape
        x = (
            x.view(
                batch_size,
                duration // self.patch_size[0],
                self.patch_size[0],
                height // self.patch_size[1],
                self.patch_size[1],
                width // self.patch_size[2],
                self.patch_size[2],
                dim,
            )
            .permute(0, 1, 3, 5, 2, 4, 6, 7)
            .flatten(4, 7)
        )
        x, _ = self.in_layer(x)
        return x


class Kandinsky6RoPE1D(nn.Module):
    """1D rotary embedding for text and audio sequences."""

    def __init__(
        self,
        dim: int,
        max_pos: int = 2048,
        max_period: float = 10000.0,
        freqs_scaling: float = 1.0,
    ):
        super().__init__()
        self.max_period = max_period
        self.dim = dim
        self.max_pos = max_pos
        self.freqs_scaling = freqs_scaling
        freq = _build_rotary_freqs(dim // 2, max_period) * freqs_scaling
        pos = torch.arange(max_pos, dtype=freq.dtype)
        self.register_buffer("args", torch.outer(pos, freq), persistent=False)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        args = self.args[pos]
        cosine = torch.cos(args)
        sine = torch.sin(args)
        rope = torch.stack([cosine, -sine, sine, cosine], dim=-1)
        rope = rope.view(*rope.shape[:-1], 2, 2)
        return rope.unsqueeze(-4)


class Kandinsky6RoPE3D(nn.Module):
    """3D rotary embedding for video spatial-temporal (T, H, W) tokens."""

    def __init__(
        self,
        axes_dims: tuple[int, int, int],
        max_pos: tuple[int, int, int] = (128, 128, 128),
        max_period: float = 10000.0,
    ):
        super().__init__()
        self.axes_dims = axes_dims
        self.max_pos = max_pos
        self.max_period = max_period

        for i, (axes_dim, ax_max_pos) in enumerate(
            zip(axes_dims, max_pos, strict=True)
        ):
            freq = _build_rotary_freqs(axes_dim // 2, max_period)
            pos = torch.arange(ax_max_pos, dtype=freq.dtype)
            self.register_buffer(f"args_{i}", torch.outer(pos, freq), persistent=False)

    def forward(
        self,
        shape: tuple[int, int, int, int],
        pos: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | list[torch.Tensor],
        scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> torch.Tensor:
        batch_size, duration, height, width = shape
        args_t = self.args_0[pos[0]] / scale_factor[0]
        args_h = self.args_1[pos[1]] / scale_factor[1]
        args_w = self.args_2[pos[2]] / scale_factor[2]

        args = torch.cat(
            [
                args_t.view(1, duration, 1, 1, -1).expand(
                    batch_size, -1, height, width, -1
                ),
                args_h.view(1, 1, height, 1, -1).expand(
                    batch_size, duration, -1, width, -1
                ),
                args_w.view(1, 1, 1, width, -1).expand(
                    batch_size, duration, height, -1, -1
                ),
            ],
            dim=-1,
        )
        cosine = torch.cos(args)
        sine = torch.sin(args)
        rope = torch.stack([cosine, -sine, sine, cosine], dim=-1)
        rope = rope.view(*rope.shape[:-1], 2, 2)
        return rope.unsqueeze(-4)


class Kandinsky6Modulation(nn.Module):
    """AdaLN-style modulation generator: SiLU -> Linear (zero-initialized)."""

    def __init__(self, time_dim: int, model_dim: int, num_params: int):
        super().__init__()
        self.activation = nn.SiLU()
        self.out_layer = ReplicatedLinear(time_dim, num_params * model_dim, bias=True)
        self.out_layer.weight.data.zero_()
        self.out_layer.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Same bf16-weight-vs-fp32-input gap as Kandinsky6TimeEmbeddings.forward (see
        # its comment): cast explicitly to out_layer's own weight dtype instead of
        # relying on torch.autocast(dtype=torch.float32) to bridge it.
        x = x.to(self.out_layer.weight.dtype)
        x = self.activation(x)
        x, _ = self.out_layer(x)
        return x


def _apply_rotary(
    x: torch.Tensor, rope: torch.Tensor, dtype: torch.dtype | None = None
) -> torch.Tensor:
    dtype = x.dtype if dtype is None else dtype
    if (
        x.is_cuda
        and torch.version.hip is None
        and not torch.is_grad_enabled()
        and x.is_contiguous()
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and rope.dtype == torch.float32
    ):
        return apply_matrix_rope(x, rope, dtype)
    x_ = x.to(dtype).reshape(*x.shape[:-1], -1, 1, 2).to(torch.float32)
    x_out = (rope * x_).sum(dim=-1)
    return x_out.reshape(*x.shape).to(dtype)


class Kandinsky6QKNorm(nn.RMSNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.is_cuda
            and torch.version.hip is None
            and x.dtype == torch.float32
            and self.normalized_shape == (128,)
            and not torch.is_grad_enabled()
        ):
            # matching dtypes enables native fusion without changing parameter
            # storage; other widths retain their original reduction order
            return F.rms_norm(x, self.normalized_shape, self.weight.float(), self.eps)
        return super().forward(x)


class Kandinsky6Attention(nn.Module):
    """Self- or cross-attention. ``kv_dim`` lets K/V come from a
    differently-sized stream (the video<->audio cross-modal attentions).

    TP: ``to_query``/``to_key``/``to_value`` are column-parallel (sharded
    across attention heads, ``gather_output=False``) and ``out_layer`` is
    row-parallel (``input_is_parallel=True``, all-reduces back to the full
    width) -- the native parallel-projection pattern ``wanvideo.py``'s
    ``WanSelfAttention`` and ``krea2.py`` use. ``query_norm``/``key_norm``
    stay replicated per-head RMSNorm instances: they normalize
    *within* one head's ``head_dim`` slice (see ``forward``'s per-head
    reshape before the norm call), and TP shards whole heads across ranks
    (never splits a single head's channels), so this per-head norm is exactly
    equivalent to the single-GPU computation on every rank with no
    cross-rank reduction needed -- unlike models whose QK-norm runs over the
    full (pre-head-split) channel width, which need a dedicated tensor-
    parallel norm helper.

    SP: attention runs through ``USPAttention`` instead of a plain
    ``LocalAttention``, giving Ulysses/ring sequence parallelism for the
    roles that can use it. ``is_cross_attention``/``skip_sequence_parallel``
    select the right SP behaviour per role (see each call site's comment);
    ``skip_sequence_parallel`` defaults to ``is_cross_attention`` (the
    ``wanvideo.py`` convention: a true self-attention role participates in
    the Ulysses all-to-all, a cross-attention role's replicated K/V does
    not), and can be set independently for a role that is architecturally
    self-attention but never sequence-sharded (the text towers).

    ``use_nabla`` is accepted for API/shape parity with FastVideo's port but
    always raises ``NotImplementedError`` at construction: sglang-diffusion
    has no ``AttentionBackendEnum.NABLA_ATTN`` backend, and no published
    Kandinsky6 checkpoint has been verified against NABLA sparse attention.
    """

    def __init__(
        self,
        num_channels: int,
        head_dim: int,
        supported_attention_backends: set[AttentionBackendEnum] | None,
        prefix: str = "",
        kv_dim: int | None = None,
        use_nabla: bool = False,
        quant_config: QuantizationConfig | None = None,
        is_cross_attention: bool = False,
        skip_sequence_parallel: bool | None = None,
    ):
        super().__init__()
        if use_nabla:
            raise NotImplementedError(
                "Kandinsky6Attention: use_nabla=True (NABLA block-sparse attention) is not "
                "implemented in sglang-diffusion -- there is no AttentionBackendEnum.NABLA_ATTN "
                "backend, and the only verified Kandinsky6 checkpoint uses dense (SDPA) "
                "attention. Construct with use_nabla=False (attention_engine != 'nabla')."
            )
        assert num_channels % head_dim == 0
        self.num_heads = num_channels // head_dim
        kv_dim = kv_dim or num_channels
        tp_size = get_tp_world_size()
        self.local_num_heads = divide(self.num_heads, tp_size)

        self.to_query = ColumnParallelLinear(
            num_channels,
            num_channels,
            bias=True,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("to_query", prefix),
        )
        self.to_key = ColumnParallelLinear(
            kv_dim,
            num_channels,
            bias=True,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("to_key", prefix),
        )
        self.to_value = ColumnParallelLinear(
            kv_dim,
            num_channels,
            bias=True,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("to_value", prefix),
        )
        self.query_norm = Kandinsky6QKNorm(head_dim)
        self.key_norm = Kandinsky6QKNorm(head_dim)
        self.out_layer = RowParallelLinear(
            num_channels,
            num_channels,
            bias=True,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=add_prefix("out_layer", prefix),
        )
        self.attention = USPAttention(
            num_heads=self.local_num_heads,
            head_size=head_dim,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            is_cross_attention=is_cross_attention,
            skip_sequence_parallel=(
                is_cross_attention
                if skip_sequence_parallel is None
                else skip_sequence_parallel
            ),
            prefix=add_prefix("attention", prefix),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        sparse_params: dict[str, Any] | None = None,
        rotary_emb: torch.Tensor | None = None,
        rotary_emb_kv: torch.Tensor | None = None,
        attn_mask_meta: dict | None = None,
        context_seq_len: int | None = None,
        skip_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        if sparse_params is not None:
            # This module is only ever constructed with use_nabla=False (see
            # __init__), so a real forward pass should never reach here --
            # the top-level model already refuses attention_engine="nabla"
            # at construction time. Kept as a defensive backstop.
            raise NotImplementedError(
                "Kandinsky6Attention: received sparse_params but NABLA sparse attention is not "
                "implemented in this port."
            )
        query, _ = self.to_query(hidden_states)

        kv_source = (
            hidden_states if encoder_hidden_states is None else encoder_hidden_states
        )
        key, _ = self.to_key(kv_source)
        value, _ = self.to_value(kv_source)

        # Column-parallel to_query/to_key/to_value already shard the channel
        # width down to `local_num_heads * head_dim` on this rank (see
        # __init__), so the per-rank reshape below splits local heads, not
        # the model's full head count.
        shape, kv_shape = query.shape[:-1], key.shape[:-1]
        query = query.reshape(*shape, self.local_num_heads, -1)
        key = key.reshape(*kv_shape, self.local_num_heads, -1)
        value = value.reshape(*kv_shape, self.local_num_heads, -1)

        query_dtype, key_dtype = query.dtype, key.dtype
        query = self.query_norm(query.float())
        if rotary_emb is not None:
            query = _apply_rotary(query, rotary_emb, query_dtype)
        else:
            query = query.to(query_dtype)
        kv_rope = (
            rotary_emb_kv
            if rotary_emb_kv is not None
            else (rotary_emb if encoder_hidden_states is None else None)
        )
        key = self.key_norm(key.float())
        if kv_rope is not None:
            key = _apply_rotary(key, kv_rope, key_dtype)
        else:
            key = key.to(key_dtype)

        if context_seq_len is not None:
            key = gather_seq(key, context_seq_len)
            value = gather_seq(value, context_seq_len)

        try:
            hidden_states = self.attention(
                query,
                key,
                value,
                attn_mask_meta=attn_mask_meta,
                skip_sequence_parallel_override=skip_sequence_parallel,
            )
        except AssertionError as exc:
            # USPAttention requires a pipeline forward context. Standalone
            # parity tests call the model directly, so fall back to SDPA
            # (single-rank only: this bypass does not implement SP).
            if "Forward context is not set" not in str(exc):
                raise

            query_shape = query.shape[:-2]
            key_shape = key.shape[:-2]
            query = query.reshape(
                query_shape[0], -1, self.local_num_heads, query.shape[-1]
            ).transpose(1, 2)
            key = key.reshape(
                key_shape[0], -1, self.local_num_heads, key.shape[-1]
            ).transpose(1, 2)
            value = value.reshape(
                key_shape[0], -1, self.local_num_heads, value.shape[-1]
            ).transpose(1, 2)
            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                is_causal=False,
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(
                *query_shape, self.local_num_heads, -1
            )

        hidden_states = hidden_states.flatten(-2, -1)
        hidden_states, _ = self.out_layer(hidden_states)
        return hidden_states


class _Kandinsky6MLP(nn.Module):
    """TP-sharded 2-layer MLP with ``fc_in``/``fc_out`` naming, matching
    ``runtime.layers.mlp.MLP``'s module layout -- but (unlike that class)
    actually honoring ``bias=False``.

    ``runtime.layers.mlp.MLP`` hardcodes ``bias=True`` on both of its
    internal ``ColumnParallelLinear``/``RowParallelLinear`` layers regardless
    of the ``bias`` constructor argument it accepts, so it cannot be reused
    verbatim here: the Kandinsky6 feed-forward projections have no bias. This
    class keeps the native ``fc_in``/``fc_out`` submodule names; the architecture
    config maps the current Diffusers ``net.0.proj``/``net.2`` checkpoint names
    onto them while retaining the same TP sharding.
    """

    def __init__(
        self,
        dim: int,
        ff_dim: int,
        prefix: str = "",
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        self.fc_in = ColumnParallelLinear(
            dim,
            ff_dim,
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("fc_in", prefix),
        )
        self.act = get_act_fn("gelu")
        self.fc_out = RowParallelLinear(
            ff_dim,
            dim,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=add_prefix("fc_out", prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc_in(x)
        x = self.act(x)
        x, _ = self.fc_out(x)
        return x


class Kandinsky6FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        ff_dim: int,
        prefix: str = "",
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        self.mlp = _Kandinsky6MLP(
            dim, ff_dim, prefix=add_prefix("mlp", prefix), quant_config=quant_config
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def _norm_scale_shift(
    norm: LayerNormScaleShift, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """float32 LayerNorm + AdaLN scale/shift, cast back to ``x``'s dtype.

    ``LayerNormScaleShift`` (unlike FastVideo's) has no
    ``convert_modulation_dtype`` kwarg -- it simply runs its internal norm in
    float32 and returns in the dtype of whatever tensor it's called with.
    Flooring ``x`` to float32 before the call and casting back after
    reproduces FastVideo's "compute the whole norm+affine in float32"
    behavior exactly.
    """
    return norm(x.float(), shift=shift, scale=scale).type_as(x)


class Kandinsky6OutLayer(nn.Module):
    """Projects visual hidden states back to packed latent patches."""

    def __init__(
        self,
        model_dim: int,
        time_dim: int,
        visual_dim: int,
        patch_size: tuple[int, int, int],
    ):
        super().__init__()
        self.patch_size = patch_size
        self.modulation = Kandinsky6Modulation(time_dim, model_dim, 2)
        self.norm = nn.LayerNorm(model_dim, eps=1e-5, elementwise_affine=False)
        self.out_layer = ReplicatedLinear(
            model_dim, math.prod(patch_size) * visual_dim, bias=True
        )

    def forward(
        self, visual_embed: torch.Tensor, time_embed: torch.Tensor
    ) -> torch.Tensor:
        shift, scale = torch.chunk(
            self.modulation(time_embed).unsqueeze(dim=1), 2, dim=-1
        )
        visual_embed = (
            self.norm(visual_embed.float()) * (scale.float()[:, None, None] + 1.0)
            + shift.float()[:, None, None]
        ).type_as(visual_embed)

        x, _ = self.out_layer(visual_embed)

        batch_size, duration, height, width, _ = x.shape
        x = (
            x.view(
                batch_size,
                duration,
                height,
                width,
                -1,
                self.patch_size[0],
                self.patch_size[1],
                self.patch_size[2],
            )
            .permute(0, 1, 5, 2, 6, 3, 7, 4)
            .flatten(1, 2)
            .flatten(2, 3)
            .flatten(3, 4)
        )
        return x


class Kandinsky6OutLayerAudio(nn.Module):
    """Projects audio hidden states back to audio latent channels."""

    def __init__(self, model_dim: int, time_dim: int, audio_dim: int):
        super().__init__()
        self.modulation = Kandinsky6Modulation(time_dim, model_dim, 2)
        self.norm = nn.LayerNorm(model_dim, eps=1e-5, elementwise_affine=False)
        self.out_layer = ReplicatedLinear(model_dim, audio_dim, bias=True)

    def forward(
        self, audio_embed: torch.Tensor, time_embed: torch.Tensor
    ) -> torch.Tensor:
        shift, scale = torch.chunk(
            self.modulation(time_embed).unsqueeze(dim=1), 2, dim=-1
        )
        x = (
            self.norm(audio_embed.float()) * (scale.float() + 1.0) + shift.float()
        ).type_as(audio_embed)
        # Reference diffusers implementation normalizes again after the
        # scale/shift affine, before the output projection. Replicated
        # verbatim (not "simplified" away) for numeric parity with the
        # upstream checkpoint conversion -- see FastVideo's identical
        # comment in fastvideo/models/dits/kandinsky6.py.
        x = self.norm(x.float()).type_as(audio_embed)
        out, _ = self.out_layer(x)
        return out


class Kandinsky6TransformerEncoderBlock(nn.Module):
    """Text-only self-attention + feed-forward block (video/audio text towers)."""

    def __init__(
        self,
        model_dim: int,
        time_dim: int,
        ff_dim: int,
        head_dim: int,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        prefix: str = "",
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        self.text_modulation = Kandinsky6Modulation(time_dim, model_dim, 6)

        self.self_attention_norm = LayerNormScaleShift(
            model_dim, eps=1e-5, elementwise_affine=False, dtype=torch.float32
        )
        self.self_attention = Kandinsky6Attention(
            model_dim,
            head_dim,
            supported_attention_backends=supported_attention_backends,
            prefix=add_prefix("self_attention", prefix),
            quant_config=quant_config,
            # Architecturally self-attention (over this tower's own text
            # tokens), but the text tower's sequence is never sequence-
            # sharded by this port (only the long visual stream is) -- every
            # SP rank holds the whole, identical text sequence, so this must
            # skip the Ulysses all-to-all like a cross-attention would.
            skip_sequence_parallel=True,
        )

        self.feed_forward_norm = LayerNormScaleShift(
            model_dim, eps=1e-5, elementwise_affine=False, dtype=torch.float32
        )
        self.feed_forward = Kandinsky6FeedForward(
            model_dim,
            ff_dim,
            prefix=add_prefix("feed_forward", prefix),
            quant_config=quant_config,
        )

    def forward(
        self, x: torch.Tensor, time_embed: torch.Tensor, rope: torch.Tensor
    ) -> torch.Tensor:
        self_attn_params, ff_params = torch.chunk(
            self.text_modulation(time_embed).unsqueeze(dim=1), 2, dim=-1
        )
        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        out = _norm_scale_shift(self.self_attention_norm, x, shift, scale)
        out = self.self_attention(out, rotary_emb=rope)
        x = (x.float() + gate.float() * out.float()).type_as(x)

        ff_shift, ff_scale, ff_gate = torch.chunk(ff_params, 3, dim=-1)
        out = _norm_scale_shift(self.feed_forward_norm, x, ff_shift, ff_scale)
        out = self.feed_forward(out)
        x = (x.float() + ff_gate.float() * out.float()).type_as(x)

        return x


class Kandinsky6TransformerDecoderBlock(nn.Module):
    """Self-attention + text cross-attention + feed-forward block.

    Used standalone for plain (non-multimodal) T2V/I2V-parity checkpoints,
    and as the ``videoT``/``audioT`` sub-block inside
    ``Kandinsky6FusedTransformerDecoderBlock`` for T2VA/IT2VA (whose
    ``forward`` re-drives this block's individual sub-layers manually rather
    than calling this class's own ``forward``, to splice a video<->audio
    cross-attention step in between).
    """

    def __init__(
        self,
        model_dim: int,
        time_dim: int,
        ff_dim: int,
        head_dim: int,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        prefix: str = "",
        use_nabla: bool = False,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        self.visual_modulation = Kandinsky6Modulation(time_dim, model_dim, 9)

        self.self_attention_norm = LayerNormScaleShift(
            model_dim, eps=1e-5, elementwise_affine=False, dtype=torch.float32
        )
        self.self_attention = Kandinsky6Attention(
            model_dim,
            head_dim,
            supported_attention_backends=supported_attention_backends,
            prefix=add_prefix("self_attention", prefix),
            use_nabla=use_nabla,
            quant_config=quant_config,
            # Video self-attention over the (long) visual token sequence --
            # the one role this port's sequence parallelism actually shards.
        )

        self.cross_attention_norm = LayerNormScaleShift(
            model_dim, eps=1e-5, elementwise_affine=False, dtype=torch.float32
        )
        self.cross_attention = Kandinsky6Attention(
            model_dim,
            head_dim,
            supported_attention_backends=supported_attention_backends,
            prefix=add_prefix("cross_attention", prefix),
            quant_config=quant_config,
            # Video queries attend to replicated text K/V -- skips the
            # all-to-all (skip_sequence_parallel defaults to
            # is_cross_attention=True).
            is_cross_attention=True,
        )

        self.feed_forward_norm = LayerNormScaleShift(
            model_dim, eps=1e-5, elementwise_affine=False, dtype=torch.float32
        )
        self.feed_forward = Kandinsky6FeedForward(
            model_dim,
            ff_dim,
            prefix=add_prefix("feed_forward", prefix),
            quant_config=quant_config,
        )

    def forward(
        self,
        visual_embed: torch.Tensor,
        text_embed: torch.Tensor,
        time_embed: torch.Tensor,
        rope: torch.Tensor | None,
        sparse_params: dict[str, Any] | None,
        attn_mask_meta: dict | None = None,
    ) -> torch.Tensor:
        self_attn_params, cross_attn_params, ff_params = torch.chunk(
            self.visual_modulation(time_embed).unsqueeze(dim=1), 3, dim=-1
        )

        self_shift, self_scale, self_gate = torch.chunk(self_attn_params, 3, dim=-1)
        visual_out = _norm_scale_shift(
            self.self_attention_norm, visual_embed, self_shift, self_scale
        )
        visual_out = self.self_attention(
            visual_out,
            rotary_emb=rope,
            sparse_params=sparse_params,
            attn_mask_meta=attn_mask_meta,
        )
        visual_embed = (
            visual_embed.float() + self_gate.float() * visual_out.float()
        ).type_as(visual_embed)

        cross_shift, cross_scale, cross_gate = torch.chunk(cross_attn_params, 3, dim=-1)
        visual_out = _norm_scale_shift(
            self.cross_attention_norm, visual_embed, cross_shift, cross_scale
        )
        visual_out = self.cross_attention(visual_out, encoder_hidden_states=text_embed)
        visual_embed = (
            visual_embed.float() + cross_gate.float() * visual_out.float()
        ).type_as(visual_embed)

        ff_shift, ff_scale, ff_gate = torch.chunk(ff_params, 3, dim=-1)
        visual_out = _norm_scale_shift(
            self.feed_forward_norm, visual_embed, ff_shift, ff_scale
        )
        visual_out = self.feed_forward(visual_out)
        visual_embed = (
            visual_embed.float() + ff_gate.float() * visual_out.float()
        ).type_as(visual_embed)

        return visual_embed


def _apply_gate_sum(
    x: torch.Tensor, out: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    return residual_gate_fp32(x, out, gate)


def _apply_scale_shift(
    norm: nn.LayerNorm, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    return (norm(x.float()) * (scale.float() + 1.0) + shift.float()).type_as(x)


class Kandinsky6FusedTransformerDecoderBlock(nn.Module):
    """Joint video+audio decoder block: per-modality self/text-cross
    attention plus a dedicated bidirectional video<->audio cross-attention,
    all independently AdaLN-modulated. Ported from
    ``diffusers.models.transformers.transformer_kandinsky6.Kandinsky6FusedTransformerDecoderBlock``.
    """

    def __init__(
        self,
        model_dim: int,
        time_dim: int,
        ff_dim: int,
        head_dim: int,
        model_dim_a: int,
        time_dim_a: int,
        ff_dim_a: int,
        head_dim_a: int,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        prefix: str = "",
        use_nabla: bool = False,
        ca_rope: bool = False,
        cross_gates: bool = False,
        fix_modulation: bool = False,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        self.videoT = Kandinsky6TransformerDecoderBlock(
            model_dim,
            time_dim,
            ff_dim,
            head_dim,
            supported_attention_backends,
            prefix=add_prefix("videoT", prefix),
            use_nabla=use_nabla,
            quant_config=quant_config,
        )
        self.audioT = Kandinsky6TransformerDecoderBlock(
            model_dim_a,
            time_dim_a,
            ff_dim_a,
            head_dim_a,
            supported_attention_backends,
            prefix=add_prefix("audioT", prefix),
            use_nabla=False,
            quant_config=quant_config,
        )
        self.va_cross_attention = Kandinsky6Attention(
            model_dim,
            head_dim,
            supported_attention_backends=supported_attention_backends,
            prefix=add_prefix("va_cross_attention", prefix),
            kv_dim=model_dim_a,
            quant_config=quant_config,
            is_cross_attention=True,
        )
        self.av_cross_attention = Kandinsky6Attention(
            model_dim_a,
            head_dim_a,
            supported_attention_backends=supported_attention_backends,
            prefix=add_prefix("av_cross_attention", prefix),
            kv_dim=model_dim,
            quant_config=quant_config,
            is_cross_attention=True,
        )
        self.va_modulation = Kandinsky6Modulation(
            time_dim,
            model_dim if not cross_gates else model_dim * 2 + model_dim_a,
            1 if cross_gates else 3,
        )
        self.av_modulation = Kandinsky6Modulation(
            time_dim_a,
            model_dim_a if not cross_gates else model_dim_a * 2 + model_dim,
            1 if cross_gates else 3,
        )
        self.va_normalization = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.av_normalization = nn.LayerNorm(model_dim_a, elementwise_affine=False)
        self.ca_rope = ca_rope
        self.cross_gates = cross_gates
        self.fix_modulation = fix_modulation
        self.model_dim = model_dim
        self.model_dim_a = model_dim_a

    def forward(
        self,
        vis: torch.Tensor,
        aud: torch.Tensor,
        text_v: torch.Tensor,
        text_a: torch.Tensor,
        time_embed: tuple[torch.Tensor, torch.Tensor],
        vis_rope: torch.Tensor | None,
        aud_rope: torch.Tensor | None,
        sparse_params: dict[str, Any] | None,
        va_gate_scale: float = 1.0,
        av_gate_scale: float = 1.0,
        video_seq_len: int | None = None,
        video_attn_meta: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t_v, t_a = time_embed

        sa_p, ca_p, ff_p = torch.chunk(
            self.videoT.visual_modulation(t_v).unsqueeze(dim=1), 3, dim=-1
        )
        shift, scale, gate = torch.chunk(sa_p, 3, dim=-1)
        vis = _apply_gate_sum(
            vis,
            self.videoT.self_attention(
                _norm_scale_shift(self.videoT.self_attention_norm, vis, shift, scale),
                rotary_emb=vis_rope,
                sparse_params=sparse_params,
                attn_mask_meta=video_attn_meta,
            ),
            gate,
        )
        shift, scale, gate_v = torch.chunk(ca_p, 3, dim=-1)
        vis_pre_ca = _norm_scale_shift(
            self.videoT.cross_attention_norm, vis, shift, scale
        )
        vis_out_t = self.videoT.cross_attention(
            vis_pre_ca, encoder_hidden_states=text_v
        )

        sa_p, ca_p, ff_p_a = torch.chunk(
            self.audioT.visual_modulation(t_a).unsqueeze(dim=1), 3, dim=-1
        )
        shift, scale, gate = torch.chunk(sa_p, 3, dim=-1)
        aud = _apply_gate_sum(
            aud,
            self.audioT.self_attention(
                _norm_scale_shift(self.audioT.self_attention_norm, aud, shift, scale),
                rotary_emb=aud_rope,
                skip_sequence_parallel=True,
            ),
            gate,
        )
        shift, scale, gate_a = torch.chunk(ca_p, 3, dim=-1)
        aud_pre_ca = _norm_scale_shift(
            self.audioT.cross_attention_norm, aud, shift, scale
        )
        aud_out_t = self.audioT.cross_attention(
            aud_pre_ca, encoder_hidden_states=text_a
        )
        aud = _apply_gate_sum(aud, aud_out_t, gate_a)

        t_va_mod = t_a if not self.fix_modulation else t_v
        t_av_mod = t_v if not self.fix_modulation else t_a
        va_params = self.va_modulation(t_va_mod).unsqueeze(dim=1)
        av_params = self.av_modulation(t_av_mod).unsqueeze(dim=1)
        if self.cross_gates:
            va_shift, va_scale, va_gate = torch.split(
                va_params, [self.model_dim, self.model_dim, self.model_dim_a], dim=-1
            )
            av_shift, av_scale, av_gate = torch.split(
                av_params, [self.model_dim_a, self.model_dim_a, self.model_dim], dim=-1
            )
        else:
            va_shift, va_scale, va_gate = torch.chunk(va_params, 3, dim=-1)
            av_shift, av_scale, av_gate = torch.chunk(av_params, 3, dim=-1)

        vis = _apply_gate_sum(vis, vis_out_t, gate_v)
        vis_for_va = _apply_scale_shift(self.va_normalization, vis, va_scale, va_shift)
        aud_for_av = _apply_scale_shift(self.av_normalization, aud, av_scale, av_shift)
        rq_v = vis_rope if self.ca_rope else None
        rk_a = aud_rope if self.ca_rope else None
        vis_from_aud = self.va_cross_attention(
            vis_for_va,
            encoder_hidden_states=aud_pre_ca,
            rotary_emb=rq_v,
            rotary_emb_kv=rk_a,
        )
        aud_from_vis = self.av_cross_attention(
            aud_for_av,
            encoder_hidden_states=vis_pre_ca,
            rotary_emb=rk_a,
            rotary_emb_kv=rq_v,
            context_seq_len=video_seq_len,
        )
        vis = _apply_gate_sum(
            vis,
            vis_from_aud,
            (va_gate if not self.cross_gates else av_gate) * va_gate_scale,
        )
        aud = _apply_gate_sum(
            aud,
            aud_from_vis,
            (av_gate if not self.cross_gates else va_gate) * av_gate_scale,
        )

        shift, scale, gate = torch.chunk(ff_p, 3, dim=-1)
        vis = _apply_gate_sum(
            vis,
            self.videoT.feed_forward(
                _norm_scale_shift(self.videoT.feed_forward_norm, vis, shift, scale)
            ),
            gate,
        )
        shift, scale, gate = torch.chunk(ff_p_a, 3, dim=-1)
        aud = _apply_gate_sum(
            aud,
            self.audioT.feed_forward(
                _norm_scale_shift(self.audioT.feed_forward_norm, aud, shift, scale)
            ),
            gate,
        )
        return vis, aud


@dataclass
class Kandinsky6TransformerOutput:
    sample: torch.Tensor | tuple[torch.Tensor, torch.Tensor]


class Kandinsky6Transformer3DModel(BaseDiT, LayerwiseOffloadableModuleMixin):
    """Native sglang-diffusion implementation of the Kandinsky6 T2VA/IT2VA transformer."""

    _fsdp_shard_conditions = [_is_kandinsky6_transformer_block]
    _compile_conditions = [_is_kandinsky6_transformer_block]
    param_names_mapping = _ARCH_CONFIG_DEFAULTS.param_names_mapping
    reverse_param_names_mapping = _ARCH_CONFIG_DEFAULTS.reverse_param_names_mapping
    lora_param_names_mapping = _ARCH_CONFIG_DEFAULTS.lora_param_names_mapping
    # Restricted to the two backends verified as safe general-purpose dense
    # attention for this model (matches BaseDiT's plain-SDPA-capable
    # subset); the wider default set on BaseDiT also includes several sparse
    # backends (video-sparse, STA, MoBA, ...) that Kandinsky6's attention
    # sub-layers never pass sparsity metadata to, so leaving those in the
    # candidate set risks a silently-wrong backend selection.
    _supported_attention_backends = {
        AttentionBackendEnum.FA,
        AttentionBackendEnum.TORCH_SDPA,
    }

    @staticmethod
    def _validate_tp_config(
        *, arch: Kandinsky6ArchConfig, tp_size: int, num_heads: int, num_heads_a: int
    ) -> None:
        if tp_size <= 0:
            raise ValueError("Kandinsky6 TP size must be positive.")
        for name, value in (
            ("num_attention_heads (video)", num_heads),
            ("num_attention_heads_a (audio)", num_heads_a),
            ("ff_dim", arch.ff_dim),
            ("ff_dim_a", arch.ff_dim_a),
        ):
            if value % tp_size:
                raise ValueError(
                    f"Kandinsky6 {name}={value} must be divisible by TP size {tp_size}."
                )

    @staticmethod
    def _validate_sequence_parallel_config(
        *, tp_size: int, num_heads: int, ulysses_size: int, ring_size: int
    ) -> None:
        if ulysses_size <= 0:
            raise ValueError("Kandinsky6 Ulysses size must be positive.")
        if ring_size <= 0:
            raise ValueError("Kandinsky6 ring size must be positive.")
        if ulysses_size == 1 and ring_size == 1:
            return
        # Ring Attention (see USPAttention) rotates whole K/V shards between
        # ranks rather than splitting heads, so only Ulysses constrains head
        # divisibility -- matches MiniMaxH3DiTModel's identical reasoning.
        local_heads = num_heads // tp_size
        if local_heads % ulysses_size:
            raise ValueError(
                f"Kandinsky6 TP-local video attention heads {local_heads} must be "
                f"divisible by Ulysses size {ulysses_size} (total video heads="
                f"{num_heads}, TP={tp_size})."
            )

    def __init__(
        self,
        config: Kandinsky6VideoAudioConfig,
        hf_config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config)
        arch: Kandinsky6ArchConfig = self.config
        self.quant_config = quant_config

        if arch.attention_engine == "nabla":
            raise NotImplementedError(
                "Kandinsky6Transformer3DModel: attention_engine='nabla' (NABLA block-sparse "
                "attention) is not yet ported to sglang-diffusion. The only verified Kandinsky6 "
                "checkpoint uses attention_engine='sdpa'; set attention_engine to 'auto' or "
                "'sdpa' to use the dense attention path."
            )
        use_nabla = False  # unreachable as True -- see guard above.

        head_dim = sum(arch.axes_dims)
        head_dim_a = sum(arch.axes_dims_a)

        tp_size = get_tp_world_size()
        ulysses_size, _ = get_ulysses_ctx()
        ring_size, _ = get_ring_ctx()
        self._validate_tp_config(
            arch=arch,
            tp_size=tp_size,
            num_heads=arch.model_dim // head_dim,
            num_heads_a=arch.model_dim_a // head_dim_a,
        )
        self._validate_sequence_parallel_config(
            tp_size=tp_size,
            num_heads=arch.model_dim // head_dim,
            ulysses_size=ulysses_size,
            ring_size=ring_size,
        )
        self.in_visual_dim = arch.in_visual_dim
        self.in_audio_dim = arch.in_audio_dim
        self.model_dim = arch.model_dim
        self.patch_size = arch.patch_size
        self.visual_cond = arch.visual_cond
        self.is_multimodal = arch.is_multimodal
        self.attention_engine = arch.attention_engine
        self.visual_token_type_num_embeddings = arch.visual_token_type_num_embeddings

        visual_embed_dim = (
            (2 * arch.in_visual_dim + 1) if arch.visual_cond else arch.in_visual_dim
        )

        self.visual_embeddings = Kandinsky6VisualEmbeddings(
            visual_embed_dim,
            arch.model_dim,
            arch.patch_size,
            prefix=add_prefix("visual_embeddings", self.prefix),
        )
        self.visual_token_type_embeddings: nn.Embedding | None = None
        if self.visual_token_type_num_embeddings > 0:
            self.visual_token_type_embeddings = nn.Embedding(
                self.visual_token_type_num_embeddings, arch.model_dim
            )
        self.visual_rope_embeddings = Kandinsky6RoPE3D(arch.axes_dims)
        self.out_layer = Kandinsky6OutLayer(
            arch.model_dim, arch.time_dim, arch.out_visual_dim, arch.patch_size
        )

        if not self.is_multimodal:
            self.time_embeddings = Kandinsky6TimeEmbeddings(
                arch.model_dim, arch.time_dim
            )
            self.text_embeddings = Kandinsky6TextEmbeddings(
                arch.in_text_dim, arch.model_dim
            )
            self.pooled_text_embeddings = Kandinsky6TextEmbeddings(
                arch.in_text_dim2, arch.time_dim
            )
            self.text_rope_embeddings = Kandinsky6RoPE1D(head_dim)
            self.text_transformer_blocks = nn.ModuleList(
                [
                    Kandinsky6TransformerEncoderBlock(
                        arch.model_dim,
                        arch.time_dim,
                        arch.ff_dim,
                        head_dim,
                        self._supported_attention_backends,
                        prefix=add_prefix(f"text_transformer_blocks.{i}", self.prefix),
                        quant_config=quant_config,
                    )
                    for i in range(arch.num_text_blocks)
                ]
            )
            self.visual_transformer_blocks = nn.ModuleList(
                [
                    Kandinsky6TransformerDecoderBlock(
                        arch.model_dim,
                        arch.time_dim,
                        arch.ff_dim,
                        head_dim,
                        self._supported_attention_backends,
                        prefix=add_prefix(
                            f"visual_transformer_blocks.{i}", self.prefix
                        ),
                        use_nabla=use_nabla,
                        quant_config=quant_config,
                    )
                    for i in range(arch.num_visual_blocks)
                ]
            )
        else:
            self.audio_embeddings = Kandinsky6TextEmbeddings(
                arch.in_audio_dim, arch.model_dim_a
            )
            self.audio_rope_embeddings = Kandinsky6RoPE1D(
                head_dim_a, freqs_scaling=arch.audio_freqs_scaling
            )
            self.audio_out_layer = Kandinsky6OutLayerAudio(
                arch.model_dim_a,
                arch.time_dim_a,
                arch.out_audio_dim or arch.in_audio_dim,
            )

            for (
                tower_prefix,
                tower_model_dim,
                tower_time_dim,
                tower_head_dim,
                tower_ff_dim,
            ) in (
                ("video", arch.model_dim, arch.time_dim, head_dim, arch.ff_dim),
                ("audio", arch.model_dim_a, arch.time_dim_a, head_dim_a, arch.ff_dim_a),
            ):
                setattr(
                    self,
                    f"{tower_prefix}_time_embeddings",
                    Kandinsky6TimeEmbeddings(tower_model_dim, tower_time_dim),
                )
                setattr(
                    self,
                    f"{tower_prefix}_text_embeddings",
                    Kandinsky6TextEmbeddings(arch.in_text_dim, tower_model_dim),
                )
                setattr(
                    self,
                    f"{tower_prefix}_pooled_text_embeddings",
                    Kandinsky6TextEmbeddings(arch.in_text_dim2, tower_time_dim),
                )
                setattr(
                    self,
                    f"{tower_prefix}_text_rope_embeddings",
                    Kandinsky6RoPE1D(tower_head_dim),
                )
                setattr(
                    self,
                    f"{tower_prefix}_text_transformer_blocks",
                    nn.ModuleList(
                        [
                            Kandinsky6TransformerEncoderBlock(
                                tower_model_dim,
                                tower_time_dim,
                                tower_ff_dim,
                                tower_head_dim,
                                self._supported_attention_backends,
                                prefix=add_prefix(
                                    f"{tower_prefix}_text_transformer_blocks.{i}",
                                    self.prefix,
                                ),
                                quant_config=quant_config,
                            )
                            for i in range(arch.num_text_blocks)
                        ]
                    ),
                )

            self.visual_transformer_blocks = nn.ModuleList(
                [
                    Kandinsky6FusedTransformerDecoderBlock(
                        arch.model_dim,
                        arch.time_dim,
                        arch.ff_dim,
                        head_dim,
                        arch.model_dim_a,
                        arch.time_dim_a,
                        arch.ff_dim_a,
                        head_dim_a,
                        self._supported_attention_backends,
                        prefix=add_prefix(
                            f"visual_transformer_blocks.{i}", self.prefix
                        ),
                        use_nabla=use_nabla,
                        ca_rope=arch.ca_rope,
                        cross_gates=arch.cross_gates,
                        fix_modulation=arch.fix_modulation,
                        quant_config=quant_config,
                    )
                    for i in range(arch.num_visual_blocks)
                ]
            )

        self.gradient_checkpointing = False
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.num_channels_latents
        self.layer_names = (
            ["video_text_transformer_blocks", "audio_text_transformer_blocks"]
            if self.is_multimodal
            else ["text_transformer_blocks"]
        ) + ["visual_transformer_blocks"]
        self.__post_init__()

    def _time_embed(
        self, prefix: str | None, time: torch.Tensor, pooled: torch.Tensor
    ) -> torch.Tensor:
        pooled_embeddings = (
            self.pooled_text_embeddings
            if prefix is None
            else getattr(self, f"{prefix}_pooled_text_embeddings")
        )
        time_embeddings = (
            self.time_embeddings
            if prefix is None
            else getattr(self, f"{prefix}_time_embeddings")
        )
        return time_embeddings(time) + pooled_embeddings(pooled)

    def _encode_text(
        self,
        prefix: str | None,
        text_embed: torch.Tensor,
        pooled: torch.Tensor,
        time: torch.Tensor,
        text_rope_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_embeddings = (
            self.text_embeddings
            if prefix is None
            else getattr(self, f"{prefix}_text_embeddings")
        )
        rope_embeddings = (
            self.text_rope_embeddings
            if prefix is None
            else getattr(self, f"{prefix}_text_rope_embeddings")
        )
        blocks = (
            self.text_transformer_blocks
            if prefix is None
            else getattr(self, f"{prefix}_text_transformer_blocks")
        )
        te = text_embeddings(text_embed)
        tm = self._time_embed(prefix, time, pooled)
        # Video and audio each own a text RoPE table sized to their own
        # head_dim (they can differ), so the position indices are looked up
        # per-tower rather than sharing one precomputed rope tensor.
        text_rope = rope_embeddings(text_rope_pos).unsqueeze(dim=0)
        for block in blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                te = torch.utils.checkpoint.checkpoint(
                    block, te, tm, text_rope, use_reentrant=False
                )
            else:
                te = block(te, tm, text_rope)
        return te, tm

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        pooled_projections: torch.Tensor | None = None,
        hidden_states_audio: torch.Tensor | None = None,
        audio_timestep: torch.Tensor | None = None,
        visual_rope_pos: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor] | list[torch.Tensor] | None
        ) = None,
        audio_rope_pos: torch.Tensor | None = None,
        text_rope_pos: torch.Tensor | None = None,
        scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
        sparse_params: dict[str, Any] | None = None,
        visual_token_type_ids: torch.Tensor | None = None,
        va_gate_scale: float = 1.0,
        av_gate_scale: float = 1.0,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | Kandinsky6TransformerOutput:
        if pooled_projections is None:
            if encoder_hidden_states_image is None:
                raise ValueError("pooled_projections must be provided for Kandinsky6.")
            pooled_projections = encoder_hidden_states_image
        if visual_rope_pos is None or text_rope_pos is None:
            raise ValueError(
                "visual_rope_pos and text_rope_pos are required for Kandinsky6."
            )
        if (
            visual_token_type_ids is not None
            and self.visual_token_type_num_embeddings == 0
        ):
            raise ValueError(
                "visual_token_type_ids requires visual_token_type_num_embeddings > 0."
            )

        x_video = hidden_states
        x_audio = hidden_states_audio
        both = self.is_multimodal and x_video is not None and x_audio is not None

        if not both:
            # The only real case reaching here is a plain (is_multimodal=False)
            # T2V/I2V-parity checkpoint denoising video through
            # Kandinsky6TransformerDecoderBlock directly. A *multimodal*
            # partial call (audio-only, or video-only) is not implemented:
            # unlike the diffusers reference, whose fused block accepts
            # vis=None/aud=None with every stage guarded accordingly, this
            # port's Kandinsky6FusedTransformerDecoderBlock.forward assumes
            # both streams are always present and takes a different call
            # signature than the plain decoder block. The TI2VA pipeline
            # always denoises both modalities together, so that guarded
            # partial-modality path was intentionally not ported.
            if self.is_multimodal:
                raise NotImplementedError(
                    "Kandinsky6Transformer3DModel: partial-modality forward (exactly one of "
                    "hidden_states/hidden_states_audio set) is not implemented for a multimodal "
                    "(is_multimodal=True) checkpoint. Provide both hidden_states and "
                    "hidden_states_audio."
                )
            te, tm = self._encode_text(
                None, encoder_hidden_states, pooled_projections, timestep, text_rope_pos
            )
            visual_embed = self.visual_embeddings(x_video)
            if (
                self.visual_token_type_embeddings is not None
                and visual_token_type_ids is not None
            ):
                type_embed = self.visual_token_type_embeddings(visual_token_type_ids)
                visual_embed = visual_embed + type_embed[:, :, None, None, :]
            visual_shape = visual_embed.shape[:-1]
            visual_rope = self.visual_rope_embeddings(
                visual_shape, visual_rope_pos, scale_factor
            )
            to_fractal = (
                sparse_params["to_fractal"] if sparse_params is not None else False
            )
            visual_embed, visual_rope = fractal_flatten(
                visual_embed, visual_rope, visual_shape, block_mask=to_fractal
            )
            visual_embed, shard = shard_seq(visual_embed)
            visual_rope = shard_like(visual_rope, shard, pad_mode="repeat_last")
            attn_meta = tail_attn_meta(
                shard, visual_embed.shape[0], visual_embed.device
            )

            for block in self.visual_transformer_blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    visual_embed = torch.utils.checkpoint.checkpoint(
                        block,
                        visual_embed,
                        te,
                        tm,
                        visual_rope,
                        sparse_params,
                        attn_mask_meta=attn_meta,
                        use_reentrant=False,
                    )
                else:
                    visual_embed = block(
                        visual_embed, te, tm, visual_rope, sparse_params, attn_meta
                    )

            visual_embed = fractal_unflatten(
                gather_seq(visual_embed, shard.orig_len),
                visual_shape,
                block_mask=to_fractal,
            )
            result: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = self.out_layer(
                visual_embed, tm
            )
        else:
            audio_timestep = audio_timestep if audio_timestep is not None else timestep
            audio_rope_pos = (
                audio_rope_pos
                if audio_rope_pos is not None
                else torch.arange(x_audio.shape[1], device=x_audio.device)
            )

            video_te, video_tm = self._encode_text(
                "video",
                encoder_hidden_states,
                pooled_projections,
                timestep,
                text_rope_pos,
            )
            audio_te, audio_tm = self._encode_text(
                "audio",
                encoder_hidden_states,
                pooled_projections,
                audio_timestep,
                text_rope_pos,
            )

            visual_embed = self.visual_embeddings(x_video)
            if (
                self.visual_token_type_embeddings is not None
                and visual_token_type_ids is not None
            ):
                type_embed = self.visual_token_type_embeddings(visual_token_type_ids)
                visual_embed = visual_embed + type_embed[:, :, None, None, :]
            visual_shape = visual_embed.shape[:-1]
            visual_rope = self.visual_rope_embeddings(
                visual_shape, visual_rope_pos, scale_factor
            )
            # Multimodal fused blocks always run the plain (non-fractal) token
            # order -- NABLA sparse attention only applies to the video
            # self-attention sub-layer (never built in this port), so no
            # fractal reordering happens at this level.
            visual_embed = visual_embed.flatten(1, 3)
            visual_rope = visual_rope.flatten(1, 3)
            visual_embed, shard = shard_seq(visual_embed)
            visual_rope = shard_like(visual_rope, shard, pad_mode="repeat_last")
            attn_meta = tail_attn_meta(
                shard, visual_embed.shape[0], visual_embed.device
            )

            audio_embed = self.audio_embeddings(x_audio)
            audio_rope = self.audio_rope_embeddings(audio_rope_pos).unsqueeze(dim=0)

            for block in self.visual_transformer_blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    visual_embed, audio_embed = torch.utils.checkpoint.checkpoint(
                        block,
                        visual_embed,
                        audio_embed,
                        video_te,
                        audio_te,
                        (video_tm, audio_tm),
                        visual_rope,
                        audio_rope,
                        sparse_params,
                        va_gate_scale,
                        av_gate_scale,
                        video_seq_len=shard.orig_len,
                        video_attn_meta=attn_meta,
                        use_reentrant=False,
                    )
                else:
                    visual_embed, audio_embed = block(
                        visual_embed,
                        audio_embed,
                        video_te,
                        audio_te,
                        (video_tm, audio_tm),
                        visual_rope,
                        audio_rope,
                        sparse_params,
                        va_gate_scale,
                        av_gate_scale,
                        video_seq_len=shard.orig_len,
                        video_attn_meta=attn_meta,
                    )

            visual_embed = fractal_unflatten(
                gather_seq(visual_embed, shard.orig_len), visual_shape, block_mask=False
            )
            video_out = self.out_layer(visual_embed, video_tm)
            audio_out = self.audio_out_layer(audio_embed, audio_tm)
            result = (video_out, audio_out)

        if return_dict:
            return Kandinsky6TransformerOutput(sample=result)
        return result

    def post_load_weights(self) -> None:
        """Re-derive RoPE/time-embedding frequency buffers left on the meta
        device by a meta-device-init-then-materialize loading flow.

        These are plain (non-persistent-buffer or plain-attribute) tensors
        computed from static config, not real checkpoint weights, so the
        loader never populates them -- they must be rebuilt here after the
        rest of the model's real parameters have landed on their target
        device.
        """
        device = next(self.parameters()).device

        for i, (axes_dim, ax_max_pos) in enumerate(
            zip(
                self.visual_rope_embeddings.axes_dims,
                self.visual_rope_embeddings.max_pos,
                strict=True,
            )
        ):
            name = f"args_{i}"
            buf = self.visual_rope_embeddings._buffers.get(name)
            if isinstance(buf, torch.Tensor) and buf.is_meta:
                freq = _build_rotary_freqs(
                    axes_dim // 2, self.visual_rope_embeddings.max_period
                ).to(device=device)
                pos = torch.arange(ax_max_pos, dtype=freq.dtype, device=device)
                self.visual_rope_embeddings._buffers[name] = torch.outer(pos, freq)

        rope1d_modules: list[Kandinsky6RoPE1D] = []
        if self.is_multimodal:
            rope1d_modules.extend(
                [
                    self.video_text_rope_embeddings,
                    self.audio_text_rope_embeddings,
                    self.audio_rope_embeddings,
                ]
            )
            time_embeds = [self.video_time_embeddings, self.audio_time_embeddings]
        else:
            rope1d_modules.append(self.text_rope_embeddings)
            time_embeds = [self.time_embeddings]

        for rope1d in rope1d_modules:
            if isinstance(rope1d.args, torch.Tensor) and rope1d.args.is_meta:
                freq = (
                    _build_rotary_freqs(rope1d.dim // 2, rope1d.max_period).to(
                        device=device
                    )
                    * rope1d.freqs_scaling
                )
                pos = torch.arange(rope1d.max_pos, dtype=freq.dtype, device=device)
                rope1d._buffers["args"] = torch.outer(pos, freq)

        for time_embed in time_embeds:
            if isinstance(time_embed.freqs, torch.Tensor) and time_embed.freqs.is_meta:
                time_embed.freqs = _build_rotary_freqs(
                    time_embed.model_dim // 2, time_embed.max_period
                ).to(device=device)


EntryClass = Kandinsky6Transformer3DModel
