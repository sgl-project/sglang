# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

from sglang.multimodal_gen.configs.models.dits import WanVideoConfig
from sglang.multimodal_gen.runtime.distributed import (
    divide,
    get_sp_world_size,
    get_tp_world_size,
)
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.layers.elementwise import MulAdd
from sglang.multimodal_gen.runtime.layers.layernorm import (
    FP32LayerNorm,
    LayerNormScaleShift,
    RMSNorm,
    ScaleResidualLayerNormScaleShift,
    tensor_parallel_rms_norm,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.mlp import MLP
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
    _apply_rotary_emb,
    get_rotary_pos_embed,
)
from sglang.multimodal_gen.runtime.layers.visual_embedding import PatchEmbed
from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
from sglang.multimodal_gen.runtime.models.dits.wanvideo import (
    WanT2VCrossAttention,
    WanTimeTextImageEmbedding,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs"
)


@dataclass(slots=True)
class _ForwardShapeInfo:
    batch_size: int
    num_frames: int
    post_patch_num_frames: int
    post_patch_height: int
    post_patch_width: int
    p_t: int
    p_h: int
    p_w: int


class BaseCausalWanSelfAttention(nn.Module):
    """Shared self-attention body; subclasses own cache behavior hooks."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm=True,
        eps=1e-6,
        parallel_attention=False,
        head_dim: int | None = None,
    ) -> None:
        if head_dim is None:
            assert dim % num_heads == 0
            head_dim = dim // num_heads
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.parallel_attention = parallel_attention
        self.max_attention_size = (
            32760 if local_attn_size == -1 else local_attn_size * 1560
        )

        # Scaled dot product attention.
        self.attn = LocalAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
            supported_attention_backends=(
                AttentionBackendEnum.FA,
                AttentionBackendEnum.AITER,
                AttentionBackendEnum.TORCH_SDPA,
            ),
        )

    def _should_use_flex_attention(self, block_mask: BlockMask, kv_cache: Any) -> bool:
        raise NotImplementedError

    def _prepare_flex_cache(
        self,
        roped_key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Any,
    ) -> None:
        return

    def _incremental_attention(
        self,
        roped_query: torch.Tensor,
        roped_key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Any,
        current_start: int,
        cache_start: int,
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _run_flex_attention(
        roped_query: torch.Tensor,
        roped_key: torch.Tensor,
        value: torch.Tensor,
        block_mask: BlockMask,
    ) -> torch.Tensor:
        padded_length = (
            math.ceil(roped_query.shape[1] / 128) * 128 - roped_query.shape[1]
        )
        padded_roped_query = torch.cat(
            [
                roped_query,
                torch.zeros(
                    [
                        roped_query.shape[0],
                        padded_length,
                        roped_query.shape[2],
                        roped_query.shape[3],
                    ],
                    device=roped_query.device,
                    dtype=value.dtype,
                ),
            ],
            dim=1,
        )
        padded_roped_key = torch.cat(
            [
                roped_key,
                torch.zeros(
                    [
                        roped_key.shape[0],
                        padded_length,
                        roped_key.shape[2],
                        roped_key.shape[3],
                    ],
                    device=roped_key.device,
                    dtype=value.dtype,
                ),
            ],
            dim=1,
        )
        padded_v = torch.cat(
            [
                value,
                torch.zeros(
                    [value.shape[0], padded_length, value.shape[2], value.shape[3]],
                    device=value.device,
                    dtype=value.dtype,
                ),
            ],
            dim=1,
        )
        x = flex_attention(
            query=padded_roped_query.transpose(2, 1),
            key=padded_roped_key.transpose(2, 1),
            value=padded_v.transpose(2, 1),
            block_mask=block_mask,
        )
        return x[:, :, :-padded_length].transpose(2, 1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        block_mask: BlockMask,
        kv_cache: Any = None,
        current_start: int = 0,
        cache_start: int | None = None,
    ):
        if cache_start is None:
            cache_start = current_start

        cos, sin = freqs_cis
        roped_query = _apply_rotary_emb(q, cos, sin, is_neox_style=False).type_as(v)
        roped_key = _apply_rotary_emb(k, cos, sin, is_neox_style=False).type_as(v)

        if self._should_use_flex_attention(block_mask, kv_cache):
            self._prepare_flex_cache(roped_key, v, kv_cache)
            return self._run_flex_attention(roped_query, roped_key, v, block_mask)

        return self._incremental_attention(
            roped_query, roped_key, v, kv_cache, current_start, cache_start
        )


class BaseCausalWanTransformerBlock(nn.Module):
    """Shared Causal Wan block body; subclasses inject self-attention implementation."""

    self_attn_cls: type[nn.Module] | None = None
    cross_attn_cls: type[nn.Module] = WanT2VCrossAttention

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        prefix: str = "",
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()

        if self.self_attn_cls is None:
            raise ValueError("self_attn_cls must be set on subclass")

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.tp_size = get_tp_world_size()
        self.local_num_heads = divide(num_heads, self.tp_size)
        dim_head = dim // num_heads

        self.to_q = ColumnParallelLinear(
            dim,
            dim,
            bias=True,
            gather_output=False,
            quant_config=quant_config,
        )
        self.to_k = ColumnParallelLinear(
            dim,
            dim,
            bias=True,
            gather_output=False,
            quant_config=quant_config,
        )
        self.to_v = ColumnParallelLinear(
            dim,
            dim,
            bias=True,
            gather_output=False,
            quant_config=quant_config,
        )

        self.to_out = RowParallelLinear(
            dim,
            dim,
            bias=True,
            input_is_parallel=True,
            reduce_results=True,
            quant_config=quant_config,
        )
        self.attn1 = self.self_attn_cls(
            dim,
            self.local_num_heads,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            qk_norm=qk_norm,
            eps=eps,
            head_dim=dim_head,
        )
        self.hidden_dim = dim
        self.num_attention_heads = num_heads
        self.local_attn_size = local_attn_size
        if qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, eps=eps)
            self.norm_k = RMSNorm(dim_head, eps=eps)
        elif qk_norm == "rms_norm_across_heads":
            self.norm_q = RMSNorm(dim, eps=eps)
            self.norm_k = RMSNorm(dim, eps=eps)
        else:
            raise ValueError(f"QK norm type not supported: {qk_norm}")
        self.tp_rmsnorm = qk_norm == "rms_norm_across_heads" and self.tp_size > 1
        assert cross_attn_norm is True
        self.self_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim, eps=eps, elementwise_affine=True, dtype=torch.float32
        )

        # 2. Cross-attention
        if supported_attention_backends is None:
            cross_attn_backends = None
        else:
            cross_attn_backends = {
                b for b in supported_attention_backends if not b.is_sparse
            }
        self.attn2 = self.cross_attn_cls(
            dim,
            num_heads,
            qk_norm=qk_norm,
            eps=eps,
            supported_attention_backends=cross_attn_backends,
            quant_config=quant_config,
        )
        self.cross_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim, eps=eps, elementwise_affine=False, dtype=torch.float32
        )

        # 3. Feed-forward
        self.ffn = MLP(
            dim, ffn_dim, act_type="gelu_pytorch_tanh", quant_config=quant_config
        )
        self.mlp_residual = MulAdd()

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        block_mask: BlockMask,
        kv_cache: Any = None,
        crossattn_cache: Any = None,
        current_start: int = 0,
        cache_start: int | None = None,
    ) -> torch.Tensor:
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        num_frames = temb.shape[1]
        frame_seqlen = hidden_states.shape[1] // num_frames
        bs, _, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype

        e = self.scale_shift_table + temb.float()
        assert e.shape == (bs, num_frames, 6, self.hidden_dim)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = e.chunk(
            6, dim=2
        )

        # 1. Self-attention
        norm_hidden_states = (
            (
                self.norm1(hidden_states.float()).unflatten(
                    dim=1, sizes=(num_frames, frame_seqlen)
                )
                * (1 + scale_msa)
                + shift_msa
            )
            .flatten(1, 2)
            .to(orig_dtype)
        )
        query, _ = self.to_q(norm_hidden_states)
        key, _ = self.to_k(norm_hidden_states)
        value, _ = self.to_v(norm_hidden_states)

        if self.norm_q is not None:
            if self.tp_rmsnorm:
                query = tensor_parallel_rms_norm(query, self.norm_q)
            else:
                query = self.norm_q(query)
        if self.norm_k is not None:
            if self.tp_rmsnorm:
                key = tensor_parallel_rms_norm(key, self.norm_k)
            else:
                key = self.norm_k(key)

        query = query.squeeze(1).unflatten(2, (self.local_num_heads, -1))
        key = key.squeeze(1).unflatten(2, (self.local_num_heads, -1))
        value = value.squeeze(1).unflatten(2, (self.local_num_heads, -1))

        attn_output = self.attn1(
            query,
            key,
            value,
            freqs_cis,
            block_mask,
            kv_cache,
            current_start,
            cache_start,
        )
        attn_output = attn_output.flatten(2)
        attn_output, _ = self.to_out(attn_output)
        attn_output = attn_output.squeeze(1)

        null_shift = null_scale = torch.zeros(
            (1,), device=hidden_states.device, dtype=hidden_states.dtype
        )
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states, attn_output, gate_msa, null_shift, null_scale
        )
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype
        ), hidden_states.to(orig_dtype)

        # 2. Cross-attention
        attn_output = self.attn2(
            norm_hidden_states,
            context=encoder_hidden_states,
            context_lens=None,
            crossattn_cache=crossattn_cache,
        )
        norm_hidden_states, hidden_states = self.cross_attn_residual_norm(
            hidden_states, attn_output, 1, c_shift_msa, c_scale_msa
        )
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype
        ), hidden_states.to(orig_dtype)

        # 3. Feed-forward
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = self.mlp_residual(ff_output, c_gate_msa, hidden_states)
        hidden_states = hidden_states.to(orig_dtype)

        return hidden_states


class BaseCausalWanTransformer3DModel(BaseDiT, OffloadableDiTMixin):
    _fsdp_shard_conditions = WanVideoConfig()._fsdp_shard_conditions
    _compile_conditions = WanVideoConfig()._compile_conditions
    _supported_attention_backends = WanVideoConfig()._supported_attention_backends
    param_names_mapping = WanVideoConfig().param_names_mapping
    reverse_param_names_mapping = WanVideoConfig().reverse_param_names_mapping
    lora_param_names_mapping = WanVideoConfig().lora_param_names_mapping

    block_cls: type[BaseCausalWanTransformerBlock] | None = None

    def __init__(
        self,
        config: WanVideoConfig,
        hf_config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config)

        if self.block_cls is None:
            raise ValueError("block_cls must be set on subclass")

        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_dim = config.attention_head_dim
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.num_channels_latents = config.num_channels_latents
        self.patch_size = config.patch_size
        self.text_len = config.text_len
        self.local_attn_size = config.local_attn_size

        # 1. Patch & position embedding
        self.patch_embedding = PatchEmbed(
            in_chans=config.in_channels,
            embed_dim=inner_dim,
            patch_size=config.patch_size,
            flatten=False,
        )

        # 2. Condition embeddings
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=config.freq_dim,
            text_embed_dim=config.text_dim,
            image_embed_dim=config.image_dim,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                self.block_cls(
                    inner_dim,
                    config.ffn_dim,
                    config.num_attention_heads,
                    config.local_attn_size,
                    config.sink_size,
                    config.qk_norm,
                    config.cross_attn_norm,
                    config.eps,
                    config.added_kv_proj_dim,
                    self._supported_attention_backends,
                    prefix=f"{config.prefix}.blocks.{i}",
                    quant_config=quant_config,
                )
                for i in range(config.num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = LayerNormScaleShift(
            inner_dim,
            eps=config.eps,
            elementwise_affine=False,
            dtype=torch.float32,
        )
        self.proj_out = ColumnParallelLinear(
            inner_dim,
            config.out_channels * math.prod(config.patch_size),
            bias=True,
            gather_output=True,
            quant_config=quant_config,
        )
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

        self.gradient_checkpointing = False

        # Causal-specific
        self.block_mask = None
        self.num_frame_per_block = config.arch_config.num_frames_per_block
        assert self.num_frame_per_block <= 3
        self.independent_first_frame = False

        self.__post_init__()

        self.layer_names = [
            "blocks",
        ]

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str,
        num_frames: int = 21,
        frame_seqlen: int = 1560,
        num_frame_per_block=1,
        local_attn_size=-1,
    ) -> BlockMask:
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(
            total_length + padded_length, device=device, dtype=torch.long
        )

        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device,
        )

        for tmp in frame_indices:
            ends[tmp : tmp + frame_seqlen * num_frame_per_block] = (
                tmp + frame_seqlen * num_frame_per_block
            )

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            return (
                (kv_idx < ends[q_idx])
                & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))
            ) | (q_idx == kv_idx)

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False,
            device=device,
        )

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames"
            )
            print(block_mask)

        return block_mask

    def _get_rope_embed_kwargs(self, hidden_states: torch.Tensor) -> dict[str, Any]:
        return {}

    def _use_gradient_checkpointing_inference(self) -> bool:
        return True

    @staticmethod
    def _maybe_first_tensor(
        value: torch.Tensor | list[torch.Tensor] | None,
    ) -> torch.Tensor | None:
        if isinstance(value, list):
            if len(value) == 0:
                return None
            return value[0]
        return value

    def _prepare_transformer_inputs(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None,
        start_frame: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor] | None,
        torch.dtype,
        _ForwardShapeInfo,
    ]:
        orig_dtype = hidden_states.dtype
        encoder_hidden_states = self._maybe_first_tensor(encoder_hidden_states)
        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states must not be empty")
        encoder_hidden_states_image = self._maybe_first_tensor(
            encoder_hidden_states_image
        )

        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # Get rotary embeddings
        d = self.hidden_size // self.num_attention_heads
        rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
        rope_kwargs = self._get_rope_embed_kwargs(hidden_states)
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            (
                post_patch_num_frames * get_sp_world_size(),
                post_patch_height,
                post_patch_width,
            ),
            self.hidden_size,
            self.num_attention_heads,
            rope_dim_list,
            dtype=(
                torch.float32
                if current_platform.is_mps() or current_platform.is_musa()
                else torch.float64
            ),
            rope_theta=10000,
            start_frame=start_frame,
            **rope_kwargs,
        )
        if freqs_cos is not None and freqs_cos.device != hidden_states.device:
            freqs_cos = freqs_cos.to(hidden_states.device)
            freqs_sin = freqs_sin.to(hidden_states.device)
        freqs_cis = (
            (freqs_cos.float(), freqs_sin.float()) if freqs_cos is not None else None
        )

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(
                timestep.flatten(), encoder_hidden_states, encoder_hidden_states_image
            )
        )
        timestep_proj = timestep_proj.unflatten(1, (6, self.hidden_size)).unflatten(
            dim=0, sizes=timestep.shape
        )

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1
            )

        encoder_hidden_states = (
            encoder_hidden_states.to(orig_dtype)
            if current_platform.is_mps()
            else encoder_hidden_states
        )

        assert encoder_hidden_states.dtype == orig_dtype

        shape_info = _ForwardShapeInfo(
            batch_size=batch_size,
            num_frames=num_frames,
            post_patch_num_frames=post_patch_num_frames,
            post_patch_height=post_patch_height,
            post_patch_width=post_patch_width,
            p_t=p_t,
            p_h=p_h,
            p_w=p_w,
        )

        return (
            hidden_states,
            encoder_hidden_states,
            timestep_proj,
            temb,
            freqs_cis,
            orig_dtype,
            shape_info,
        )

    @staticmethod
    def _get_layer_cache(caches: Any, block_index: int):
        if caches is None:
            return None
        return caches[block_index]

    def _run_inference_blocks(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_proj: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor] | None,
        kv_cache: Any,
        crossattn_cache: Any,
        current_start: int,
        cache_start: int,
    ) -> torch.Tensor:
        for block_index, block in enumerate(self.blocks):
            if (
                torch.is_grad_enabled()
                and self.gradient_checkpointing
                and self._use_gradient_checkpointing_inference()
            ):
                checkpoint_kwargs = {
                    "kv_cache": self._get_layer_cache(kv_cache, block_index),
                    "current_start": current_start,
                    "cache_start": cache_start,
                    "block_mask": self.block_mask,
                }
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    freqs_cis,
                    **checkpoint_kwargs,
                )
            else:
                block_kwargs = {
                    "kv_cache": self._get_layer_cache(kv_cache, block_index),
                    "crossattn_cache": self._get_layer_cache(
                        crossattn_cache, block_index
                    ),
                    "current_start": current_start,
                    "cache_start": cache_start,
                    "block_mask": self.block_mask,
                }
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    freqs_cis,
                    **block_kwargs,
                )

        return hidden_states

    def _project_to_output(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        timestep: torch.LongTensor,
        shape_info: _ForwardShapeInfo,
    ) -> torch.Tensor:
        temb = temb.unflatten(dim=0, sizes=timestep.shape).unsqueeze(2)
        shift, scale = (self.scale_shift_table.unsqueeze(1) + temb).chunk(2, dim=2)
        hidden_states = self.norm_out(hidden_states, shift, scale)
        hidden_states, _ = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            shape_info.batch_size,
            shape_info.post_patch_num_frames,
            shape_info.post_patch_height,
            shape_info.post_patch_width,
            shape_info.p_t,
            shape_info.p_h,
            shape_info.p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    def _forward_inference(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        kv_cache: Any = None,
        crossattn_cache: Any = None,
        current_start: int = 0,
        cache_start: int = 0,
        start_frame: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        (
            hidden_states,
            encoder_hidden_states,
            timestep_proj,
            temb,
            freqs_cis,
            _,
            shape_info,
        ) = self._prepare_transformer_inputs(
            hidden_states,
            encoder_hidden_states,
            timestep,
            encoder_hidden_states_image,
            start_frame,
        )

        hidden_states = self._run_inference_blocks(
            hidden_states,
            encoder_hidden_states,
            timestep_proj,
            freqs_cis,
            kv_cache,
            crossattn_cache,
            current_start,
            cache_start,
        )

        return self._project_to_output(hidden_states, temb, timestep, shape_info)

    def _forward_train(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        start_frame: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        (
            hidden_states,
            encoder_hidden_states,
            timestep_proj,
            temb,
            freqs_cis,
            _,
            shape_info,
        ) = self._prepare_transformer_inputs(
            hidden_states,
            encoder_hidden_states,
            timestep,
            encoder_hidden_states_image,
            start_frame,
        )

        # Construct blockwise causal attn mask once.
        if self.block_mask is None:
            self.block_mask = self._prepare_blockwise_causal_attn_mask(
                device=hidden_states.device,
                num_frames=shape_info.num_frames,
                frame_seqlen=shape_info.post_patch_height * shape_info.post_patch_width,
                num_frame_per_block=self.num_frame_per_block,
                local_attn_size=self.local_attn_size,
            )

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    freqs_cis,
                    block_mask=self.block_mask,
                )
        else:
            for block in self.blocks:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    freqs_cis,
                    block_mask=self.block_mask,
                )

        return self._project_to_output(hidden_states, temb, timestep, shape_info)

    def forward(self, *args, **kwargs):
        if kwargs.get("kv_cache") is not None:
            return self._forward_inference(*args, **kwargs)
        return self._forward_train(*args, **kwargs)
