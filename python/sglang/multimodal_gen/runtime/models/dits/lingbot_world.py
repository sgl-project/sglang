# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# Adapted from: https://github.com/Robbyant/lingbot-world

# SPDX-License-Identifier: Apache-2.0

import math
from functools import lru_cache
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.dits import LingBotWorldVideoConfig
from sglang.multimodal_gen.runtime.distributed import (
    divide,
    get_sp_group,
    get_sp_parallel_rank,
    get_sp_world_size,
    get_tp_rank,
    get_tp_world_size,
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
    get_ulysses_parallel_world_size,
)
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention, USPAttention
from sglang.multimodal_gen.runtime.layers.elementwise import MulAdd
from sglang.multimodal_gen.runtime.layers.kvcache.causal_attention_cache import (
    CausalSelfAttentionKVCache,
    CrossAttentionKVCache,
)
from sglang.multimodal_gen.runtime.layers.layernorm import (
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
    NDRotaryEmbedding,
    _apply_rotary_emb,
    apply_flashinfer_rope_qk_inplace,
    get_rotary_pos_embed,
)
from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_input_all_to_all,
    _usp_input_all_to_all_varlen,
    _usp_output_all_to_all,
    _usp_output_all_to_all_varlen,
)
from sglang.multimodal_gen.runtime.layers.visual_embedding import (
    PatchEmbed,
    WanCamControlPatchEmbedding,
)
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.models.dits.causal_wanvideo import (
    CausalWanSelfAttention,
    CausalWanTransformer3DModel,
    CausalWanTransformerBlock,
)
from sglang.multimodal_gen.runtime.models.dits.wanvideo import (
    WanI2VCrossAttention,
    WanT2VCrossAttention,
    WanTimeTextImageEmbedding,
    WanTransformer3DModel,
)
from sglang.multimodal_gen.runtime.models.utils import _use_aiter
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_world.constants import (
    LINGBOT_C2WS_PLUCKER_EMB_CACHE,
    LINGBOT_CAM_CONDITIONER_CACHE,
    LINGBOT_ROPE_CACHE,
    LINGBOT_SEQUENCE_SHARD_ROPE_CACHE,
    LINGBOT_TIME_EMBEDDINGS_CACHE,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.realtime.states import (
    get_realtime_causal_dit_state,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.utils import add_prefix

logger = init_logger(__name__)
_is_cuda = current_platform.is_cuda()


def _safe_tensor_version(tensor: torch.Tensor) -> int:
    """Return ``tensor._version``, or ``0`` for inference-mode tensors.

    Tensors created under ``torch.inference_mode`` do not track a version
    counter, so reading ``tensor._version`` raises ``RuntimeError``. The value
    is only used as a cache-invalidation hint for the camera conditioner, so a
    constant fallback is safe for such tensors.
    """
    return 0 if tensor.is_inference() else tensor._version


if _use_aiter:
    from aiter.ops.rope import rope_cached_2c_fwd_inplace


def _compute_sequence_splits(total_len: int, world_size: int) -> list[int]:
    base = total_len // world_size
    remainder = total_len % world_size
    return [base + (1 if rank < remainder else 0) for rank in range(world_size)]


def _sequence_splits_are_uniform(seq_splits: list[int]) -> bool:
    return len(seq_splits) <= 1 or all(
        seq_len == seq_splits[0] for seq_len in seq_splits
    )


def _sequence_shard_tensor(
    x: torch.Tensor, seq_splits: list[int], rank: int
) -> torch.Tensor:
    start = sum(seq_splits[:rank])
    end = start + seq_splits[rank]
    return x[:, start:end, ...].contiguous()


def _sequence_all_gather_varlen(
    x: torch.Tensor,
    seq_splits: list[int],
    group: dist.ProcessGroup,
) -> torch.Tensor:
    rank = get_sp_parallel_rank()
    if _sequence_splits_are_uniform(seq_splits):
        return sequence_model_parallel_all_gather(x.contiguous(), dim=1)

    max_seq = max(seq_splits)
    local_seq = seq_splits[rank]
    if local_seq < max_seq:
        pad_shape = list(x.shape)
        pad_shape[1] = max_seq - local_seq
        pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        x = torch.cat([x, pad], dim=1)
    gathered = [torch.empty_like(x) for _ in seq_splits]
    dist.all_gather(gathered, x.contiguous(), group=group)
    return torch.cat(
        [chunk[:, :seq_len, ...] for chunk, seq_len in zip(gathered, seq_splits)], dim=1
    )


class LingBotWorldCamConditioner(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.cam_injector = MLP(dim, dim, dim, bias=True, act_type="silu")
        self.cam_scale_layer = nn.Linear(dim, dim)
        self.cam_shift_layer = nn.Linear(dim, dim)

    def compute_scale_shift(
        self, c2ws_plucker_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c2ws_hidden_states = self.cam_injector(c2ws_plucker_emb)
        c2ws_hidden_states = c2ws_hidden_states + c2ws_plucker_emb
        cam_scale = self.cam_scale_layer(c2ws_hidden_states)
        cam_shift = self.cam_shift_layer(c2ws_hidden_states)
        return cam_scale, cam_shift

    def forward(
        self,
        hidden_states: torch.Tensor,
        c2ws_plucker_emb: torch.Tensor | None,
        scale_shift: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if c2ws_plucker_emb is None:
            return hidden_states
        if c2ws_plucker_emb.shape != hidden_states.shape:
            raise ValueError(
                "c2ws_plucker_emb shape must match hidden_states shape, "
                f"got {tuple(c2ws_plucker_emb.shape)} vs {tuple(hidden_states.shape)}"
            )
        if scale_shift is None:
            scale_shift = self.compute_scale_shift(c2ws_plucker_emb)
        cam_scale, cam_shift = scale_shift
        return (1.0 + cam_scale) * hidden_states + cam_shift


class LingBotWorldCausalSelfAttention(CausalWanSelfAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ulysses_world_size = max(get_ulysses_parallel_world_size(), 1)
        if self.num_heads % ulysses_world_size != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by ulysses_degree ({ulysses_world_size})."
            )
        self.ulysses_num_heads = self.num_heads // ulysses_world_size
        self.ulysses_attn = LocalAttention(
            num_heads=self.ulysses_num_heads,
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

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, ...],
        block_mask,
        kv_cache: CausalSelfAttentionKVCache | None = None,
        current_start: int = 0,
        cache_start: int | None = None,
        update_cache_only: bool = False,
    ):
        cos, sin = freqs_cis[:2]
        cos_sin_cache = freqs_cis[2] if len(freqs_cis) > 2 else None
        if _is_cuda and q.dim() == 4 and q.shape == k.shape:
            if cos_sin_cache is None:
                cos_sin_cache = torch.cat(
                    [
                        cos.to(dtype=torch.float32).contiguous(),
                        sin.to(dtype=torch.float32).contiguous(),
                    ],
                    dim=-1,
                )
            roped_query, roped_key = apply_flashinfer_rope_qk_inplace(
                q, k, cos_sin_cache, is_neox=False
            )
            roped_query = roped_query.type_as(v)
            roped_key = roped_key.type_as(v)
        else:
            roped_query = _apply_rotary_emb(q, cos, sin, is_neox_style=False).type_as(v)
            roped_key = _apply_rotary_emb(k, cos, sin, is_neox_style=False).type_as(v)
        forward_batch = get_forward_context().forward_batch
        seq_splits = None
        uniform_seq_splits = False
        sequence_shard_enabled = (
            kv_cache is not None
            and forward_batch is not None
            and getattr(forward_batch, "enable_sequence_shard", False)
            and get_ulysses_parallel_world_size() > 1
        )

        if kv_cache is None:
            if sequence_shard_enabled:
                raise NotImplementedError(
                    "LingBot causal sequence sharding currently requires kv_cache-backed inference."
                )
            return super().forward(
                q,
                k,
                v,
                (cos, sin),
                block_mask,
                kv_cache,
                current_start,
                cache_start,
            )

        if sequence_shard_enabled:
            seq_splits = getattr(forward_batch, "sequence_shard_splits", None)
            if seq_splits is None:
                raise ValueError(
                    "LingBot causal sequence sharding requires forward_batch.sequence_shard_splits."
                )
            seq_splits = list(seq_splits)
            uniform_seq_splits = _sequence_splits_are_uniform(seq_splits)
            # Pack Q/K/V to avoid launching three Ulysses all-to-all collectives.
            qkv = torch.cat([roped_query, roped_key, v], dim=-1)
            qkv = (
                _usp_input_all_to_all(qkv, head_dim=2)
                if uniform_seq_splits
                else _usp_input_all_to_all_varlen(qkv, seq_splits, head_dim=2)
            )
            roped_query, roped_key, v = qkv.chunk(3, dim=-1)

        if (
            not sequence_shard_enabled
            and not update_cache_only
            and kv_cache.can_direct_current_attention(roped_key.shape[1])
        ):
            return self.attn(roped_query, roped_key, v)

        cache_head_start = (
            get_tp_rank() * roped_key.shape[2]
            if sequence_shard_enabled
            else self.head_start
        )
        cache_view = kv_cache.update_and_get_attention_kv(
            key=roped_key,
            value=v,
            current_chunk_start=current_start,
            cache_head_start=cache_head_start,
            recent_window_tokens=(
                None
                if update_cache_only
                else getattr(forward_batch, "realtime_causal_kv_sample_tokens", None)
            ),
            debug_name="LingBot KV cache",
        )
        if update_cache_only:
            return v
        attn_impl = self.ulysses_attn if sequence_shard_enabled else self.attn
        x = attn_impl(
            roped_query,
            cache_view.k,
            cache_view.v,
        )
        if sequence_shard_enabled:
            assert seq_splits is not None
            x = (
                _usp_output_all_to_all(x, head_dim=2)
                if uniform_seq_splits
                else _usp_output_all_to_all_varlen(x, seq_splits, head_dim=2)
            )
        return x


class LingBotWorldTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        supported_attention_backends=None,
        prefix: str = "",
        attention_type: str = "original",
        sla_topk: float = 0.1,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()

        self.norm1 = LayerNormScaleShift(
            dim,
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32,
        )
        self.to_q = ColumnParallelLinear(
            dim,
            dim,
            bias=True,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("attn1.to_q", prefix),
        )
        self.to_k = ColumnParallelLinear(
            dim,
            dim,
            bias=True,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("attn1.to_k", prefix),
        )
        self.to_v = ColumnParallelLinear(
            dim,
            dim,
            bias=True,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("attn1.to_v", prefix),
        )
        self.to_out = RowParallelLinear(
            dim,
            dim,
            bias=True,
            reduce_results=True,
            quant_config=quant_config,
            prefix=add_prefix("attn1.to_out.0", prefix),
        )

        tp_size = get_tp_world_size()
        self.local_num_heads = divide(num_heads, tp_size)
        self.hidden_dim = dim
        self.num_attention_heads = num_heads
        self.dim_head = dim // num_heads
        self.tp_rmsnorm = qk_norm == "rms_norm_across_heads" and tp_size > 1

        self.attn1 = USPAttention(
            num_heads=self.local_num_heads,
            head_size=self.dim_head,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            prefix=add_prefix("attn1", prefix),
            quant_config=quant_config,
            is_cross_attention=False,
        )
        if qk_norm == "rms_norm":
            self.norm_q = RMSNorm(self.dim_head, eps=eps)
            self.norm_k = RMSNorm(self.dim_head, eps=eps)
        elif qk_norm == "rms_norm_across_heads":
            self.norm_q = RMSNorm(dim, eps=eps)
            self.norm_k = RMSNorm(dim, eps=eps)
        else:
            raise ValueError(f"Unsupported qk_norm: {qk_norm}")
        if not cross_attn_norm:
            raise ValueError("LingBotWorld requires cross_attn_norm=True")
        self.self_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            eps=eps,
            elementwise_affine=True,
            dtype=torch.float32,
        )

        cross_attn_backends = {
            b for b in supported_attention_backends if not b.is_sparse
        }
        if added_kv_proj_dim is not None:
            self.attn2 = WanI2VCrossAttention(
                dim,
                num_heads,
                qk_norm=qk_norm,
                eps=eps,
                prefix=add_prefix("attn2", prefix),
                supported_attention_backends=cross_attn_backends,
                quant_config=quant_config,
            )
        else:
            self.attn2 = WanT2VCrossAttention(
                dim,
                num_heads,
                qk_norm=qk_norm,
                eps=eps,
                prefix=add_prefix("attn2", prefix),
                supported_attention_backends=cross_attn_backends,
                quant_config=quant_config,
            )
        self.cross_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32,
        )
        self.ffn = MLP(
            dim,
            ffn_dim,
            act_type="gelu_pytorch_tanh",
            prefix=add_prefix("ffn.net", prefix),
            quant_config=quant_config,
        )
        self.mlp_residual = MulAdd()
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.cam_conditioner = LingBotWorldCamConditioner(dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, ...],
        c2ws_plucker_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        orig_dtype = hidden_states.dtype
        if temb.dim() == 4:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        norm_hidden_states = self.norm1(hidden_states, shift_msa, scale_msa)
        query, _ = self.to_q(norm_hidden_states)
        key, _ = self.to_k(norm_hidden_states)
        value, _ = self.to_v(norm_hidden_states)

        if self.tp_rmsnorm:
            query = tensor_parallel_rms_norm(query, self.norm_q)
            key = tensor_parallel_rms_norm(key, self.norm_k)
        else:
            query = self.norm_q(query)
            key = self.norm_k(key)

        query = query.squeeze(1).unflatten(2, (self.local_num_heads, self.dim_head))
        key = key.squeeze(1).unflatten(2, (self.local_num_heads, self.dim_head))
        value = value.squeeze(1).unflatten(2, (self.local_num_heads, self.dim_head))

        cos, sin = freqs_cis
        if _is_cuda and query.shape == key.shape:
            cos_sin_cache = torch.cat(
                [
                    cos.to(dtype=torch.float32).contiguous(),
                    sin.to(dtype=torch.float32).contiguous(),
                ],
                dim=-1,
            )
            query, key = apply_flashinfer_rope_qk_inplace(
                query, key, cos_sin_cache, is_neox=False
            )
        elif _use_aiter:
            query_shape = query.shape
            key_shape = key.shape
            num_tokens = query.shape[:-2].numel()
            q_sbhd = query.view(num_tokens, 1, query_shape[-2], query_shape[-1])
            k_sbhd = key.view(num_tokens, 1, key_shape[-2], key_shape[-1])
            cos_sbhd = cos.contiguous().view(num_tokens, 1, 1, -1)
            sin_sbhd = sin.contiguous().view(num_tokens, 1, 1, -1)
            rope_cached_2c_fwd_inplace(
                q_sbhd,
                k_sbhd,
                cos_sbhd,
                sin_sbhd,
                1,
                True,
                False,
            )
            query = q_sbhd.view(query_shape)
            key = k_sbhd.view(key_shape)
        else:
            query = _apply_rotary_emb(query, cos, sin, is_neox_style=False)
            key = _apply_rotary_emb(key, cos, sin, is_neox_style=False)

        attn_output = self.attn1(query, key, value)
        attn_output = attn_output.flatten(2)
        attn_output, _ = self.to_out(attn_output)
        attn_output = attn_output.squeeze(1)

        null_shift = torch.zeros(
            (1,), device=hidden_states.device, dtype=hidden_states.dtype
        )
        null_scale = torch.zeros(
            (1,), device=hidden_states.device, dtype=hidden_states.dtype
        )
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states, attn_output, gate_msa, null_shift, null_scale
        )
        hidden_states = self.cam_conditioner(
            hidden_states.to(orig_dtype), c2ws_plucker_emb
        )
        norm_hidden_states = self.self_attn_residual_norm.norm(hidden_states).to(
            orig_dtype
        )

        attn_output = self.attn2(
            norm_hidden_states, context=encoder_hidden_states, context_lens=None
        )
        norm_hidden_states, hidden_states = self.cross_attn_residual_norm(
            hidden_states, attn_output, 1, c_shift_msa, c_scale_msa
        )
        norm_hidden_states = norm_hidden_states.to(orig_dtype)
        hidden_states = hidden_states.to(orig_dtype)

        ff_output = self.ffn(norm_hidden_states)
        hidden_states = self.mlp_residual(ff_output, c_gate_msa, hidden_states)
        return hidden_states.to(orig_dtype)


class LingBotWorldTransformer3DModel(CachableDiT, LayerwiseOffloadableModuleMixin):
    _fsdp_shard_conditions = LingBotWorldVideoConfig()._fsdp_shard_conditions
    _compile_conditions = LingBotWorldVideoConfig()._compile_conditions
    _supported_attention_backends = (
        LingBotWorldVideoConfig()._supported_attention_backends
    )
    param_names_mapping = LingBotWorldVideoConfig().param_names_mapping
    reverse_param_names_mapping = LingBotWorldVideoConfig().reverse_param_names_mapping
    lora_param_names_mapping = LingBotWorldVideoConfig().lora_param_names_mapping

    def __init__(
        self,
        config: LingBotWorldVideoConfig,
        hf_config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config)

        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.num_channels_latents = config.num_channels_latents
        self.patch_size = config.patch_size
        self.text_len = config.text_len
        self.patch_embedding = PatchEmbed(
            in_chans=config.in_channels,
            embed_dim=inner_dim,
            patch_size=config.patch_size,
            flatten=False,
        )
        self.patch_embedding_wancamctrl = WanCamControlPatchEmbedding(
            in_chans=6 * 64,
            embed_dim=inner_dim,
            patch_size=config.patch_size,
        )
        self.c2ws_mlp = MLP(inner_dim, inner_dim, inner_dim, bias=True, act_type="silu")
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=config.freq_dim,
            text_embed_dim=config.text_dim,
            image_embed_dim=config.image_dim,
        )
        self.blocks = nn.ModuleList(
            [
                LingBotWorldTransformerBlock(
                    inner_dim,
                    config.ffn_dim,
                    config.num_attention_heads,
                    config.qk_norm,
                    config.cross_attn_norm,
                    config.eps,
                    config.added_kv_proj_dim,
                    self._supported_attention_backends,
                    prefix=f"blocks.{i}",
                    quant_config=quant_config,
                )
                for i in range(config.num_layers)
            ]
        )
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
            prefix="proj_out",
            quant_config=quant_config,
        )
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )
        self.cnt = 0
        self.__post_init__()
        self.sp_size = get_sp_world_size()
        d = self.hidden_size // self.num_attention_heads
        self.rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
        self.rotary_emb = NDRotaryEmbedding(
            rope_dim_list=self.rope_dim_list,
            rope_theta=10000,
            dtype=(
                torch.float64
                if current_platform.is_float64_supported()
                else torch.float32
            ),
        )
        self.layer_names = ["blocks"]

    @lru_cache(maxsize=1)
    def _compute_rope_for_sequence_shard(
        self,
        local_len: int,
        rank: int,
        frame_stride_local: int,
        width_local: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_start = rank * local_len
        token_indices = torch.arange(
            token_start, token_start + local_len, device=device, dtype=torch.long
        )
        t_idx = token_indices // frame_stride_local
        rem = token_indices % frame_stride_local
        h_idx = rem // width_local
        w_idx = rem % width_local
        positions = torch.stack((t_idx, h_idx, w_idx), dim=1)
        return self.rotary_emb.forward_uncached(positions)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        guidance=None,
        c2ws_plucker_emb: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        forward_batch = get_forward_context().forward_batch
        sequence_shard_enabled = (
            forward_batch is not None
            and forward_batch.enable_sequence_shard
            and self.sp_size > 1
        )
        self.enable_teacache = (
            forward_batch is not None and forward_batch.enable_teacache
        )

        orig_dtype = hidden_states.dtype
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]
        if (
            isinstance(encoder_hidden_states_image, list)
            and len(encoder_hidden_states_image) > 0
        ):
            encoder_hidden_states_image = encoder_hidden_states_image[0]
        else:
            encoder_hidden_states_image = None

        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        if not sequence_shard_enabled:
            freqs_cos, freqs_sin = self.rotary_emb.forward_from_grid(
                (
                    post_patch_num_frames * self.sp_size,
                    post_patch_height,
                    post_patch_width,
                ),
                shard_dim=0,
                start_frame=0,
                device=hidden_states.device,
            )
            freqs_cis = (freqs_cos.float(), freqs_sin.float())

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        if c2ws_plucker_emb is not None:
            c2ws_plucker_emb = self.patch_embedding_wancamctrl(
                c2ws_plucker_emb.to(
                    device=hidden_states.device, dtype=hidden_states.dtype
                )
            )
            c2ws_plucker_emb = c2ws_plucker_emb + self.c2ws_mlp(c2ws_plucker_emb)

        seq_len_orig = hidden_states.shape[1]
        seq_shard_pad = 0
        if sequence_shard_enabled:
            if seq_len_orig % self.sp_size != 0:
                seq_shard_pad = self.sp_size - (seq_len_orig % self.sp_size)
                pad = torch.zeros(
                    (batch_size, seq_shard_pad, hidden_states.shape[2]),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                hidden_states = torch.cat([hidden_states, pad], dim=1)
                if c2ws_plucker_emb is not None:
                    c2ws_plucker_emb = torch.cat([c2ws_plucker_emb, pad], dim=1)
            sp_rank = get_sp_group().rank_in_group
            local_seq_len = hidden_states.shape[1] // self.sp_size
            hidden_states = hidden_states.view(
                batch_size, self.sp_size, local_seq_len, hidden_states.shape[2]
            )[:, sp_rank, :, :]
            if c2ws_plucker_emb is not None:
                c2ws_plucker_emb = c2ws_plucker_emb.view(
                    batch_size, self.sp_size, local_seq_len, c2ws_plucker_emb.shape[2]
                )[:, sp_rank, :, :]
            frame_stride = post_patch_height * post_patch_width
            freqs_cos, freqs_sin = self._compute_rope_for_sequence_shard(
                local_seq_len,
                sp_rank,
                frame_stride,
                post_patch_width,
                hidden_states.device,
            )
            freqs_cis = (freqs_cos.float(), freqs_sin.float())

        if timestep.dim() == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(
                timestep,
                encoder_hidden_states,
                encoder_hidden_states_image,
                timestep_seq_len=ts_seq_len,
            )
        )
        timestep_proj = (
            timestep_proj.unflatten(2, (6, -1))
            if ts_seq_len is not None
            else timestep_proj.unflatten(1, (6, -1))
        )
        if sequence_shard_enabled and ts_seq_len is not None:
            if seq_shard_pad > 0:
                pad = torch.zeros(
                    (
                        batch_size,
                        seq_shard_pad,
                        timestep_proj.shape[2],
                        timestep_proj.shape[3],
                    ),
                    dtype=timestep_proj.dtype,
                    device=timestep_proj.device,
                )
                timestep_proj = torch.cat([timestep_proj, pad], dim=1)
            timestep_proj = timestep_proj.view(
                batch_size,
                self.sp_size,
                hidden_states.shape[1],
                timestep_proj.shape[2],
                timestep_proj.shape[3],
            )[:, get_sp_parallel_rank(), :, :, :]

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1
            )
        encoder_hidden_states = (
            encoder_hidden_states.to(orig_dtype)
            if not current_platform.is_amp_supported()
            else encoder_hidden_states
        )

        should_skip_forward = self.should_skip_forward_for_cached_states(
            timestep_proj=timestep_proj, temb=temb
        )
        if should_skip_forward:
            hidden_states = self.retrieve_cached_states(hidden_states)
        else:
            if self.enable_teacache:
                original_hidden_states = hidden_states.clone()
            for block in self.blocks:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    freqs_cis,
                    c2ws_plucker_emb,
                )
            if self.enable_teacache:
                self.maybe_cache_states(hidden_states, original_hidden_states)
        self.cnt += 1

        if sequence_shard_enabled:
            hidden_states = sequence_model_parallel_all_gather(
                hidden_states.contiguous(), dim=1
            )
            if seq_shard_pad > 0:
                hidden_states = hidden_states[:, :seq_len_orig, :]

        if temb.dim() == 3:
            shift, scale = (
                self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)
            ).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        hidden_states = self.norm_out(hidden_states, shift, scale)
        hidden_states, _ = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    maybe_cache_states = WanTransformer3DModel.maybe_cache_states
    should_skip_forward_for_cached_states = (
        WanTransformer3DModel.should_skip_forward_for_cached_states
    )
    retrieve_cached_states = WanTransformer3DModel.retrieve_cached_states


class CausalLingBotWorldTransformerBlock(CausalWanTransformerBlock):
    _use_megatron_tp = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        head_start = self.attn1.head_start
        self.attn1 = LingBotWorldCausalSelfAttention(
            dim=self.hidden_dim,
            num_heads=self.local_num_heads,
            local_attn_size=self.local_attn_size,
            sink_size=self.attn1.sink_size,
            qk_norm=self.attn1.qk_norm,
            eps=self.attn1.eps,
            head_dim=self.dim_head,
            head_start=head_start,
        )
        self.cam_conditioner = LingBotWorldCamConditioner(self.hidden_dim)
        self._fused_qkv_weight = None
        self._fused_qkv_bias = None

    def _can_fuse_qkv_projection(self) -> bool:
        if self._fused_qkv_weight is not None:
            return True

        layers = (self.to_q, self.to_k, self.to_v)
        biases = [layer.bias for layer in layers]
        return (
            all(bias is None for bias in biases)
            or all(bias is not None for bias in biases)
        ) and all(
            getattr(layer, "quant_config", None) is None
            and hasattr(layer, "weight")
            and layer.weight is not None
            for layer in layers
        )

    def fuse_qkv_projection(self) -> bool:
        if not self._can_fuse_qkv_projection():
            return False
        if self._fused_qkv_weight is not None:
            return True

        layers = (self.to_q, self.to_k, self.to_v)
        with torch.no_grad():
            self._fused_qkv_weight = nn.Parameter(
                torch.cat([layer.weight.detach() for layer in layers], dim=0)
                .contiguous()
                .to(self.to_q.weight.device),
                requires_grad=False,
            )
            if all(layer.bias is not None for layer in layers):
                self._fused_qkv_bias = nn.Parameter(
                    torch.cat([layer.bias.detach() for layer in layers], dim=0)
                    .contiguous()
                    .to(self.to_q.weight.device),
                    requires_grad=False,
                )

        return True

    def _project_qkv(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.fuse_qkv_projection():
            qkv = F.linear(hidden_states, self._fused_qkv_weight, self._fused_qkv_bias)
            return qkv.chunk(3, dim=-1)

        query, _ = self.to_q(hidden_states)
        key, _ = self.to_k(hidden_states)
        value, _ = self.to_v(hidden_states)
        return query, key, value

    def _cross_attn_with_cache(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        crossattn_cache: CrossAttentionKVCache | None,
    ) -> torch.Tensor:
        attn2 = self.attn2
        q, _ = attn2.to_q(hidden_states)
        if attn2.tp_rmsnorm:
            q = tensor_parallel_rms_norm(q, attn2.norm_q)
        else:
            q = attn2.norm_q(q)
        q = q.unflatten(2, (attn2.local_num_heads, attn2.head_dim))

        if crossattn_cache is not None and crossattn_cache.is_init:
            k = crossattn_cache.k
            v = crossattn_cache.v
        else:
            k, _ = attn2.to_k(encoder_hidden_states)
            if attn2.tp_rmsnorm:
                k = tensor_parallel_rms_norm(k, attn2.norm_k)
            else:
                k = attn2.norm_k(k)
            k = k.unflatten(2, (attn2.local_num_heads, attn2.head_dim))

            v, _ = attn2.to_v(encoder_hidden_states)
            v = v.unflatten(2, (attn2.local_num_heads, attn2.head_dim))

            if crossattn_cache is not None:
                crossattn_cache.store(k, v)

        hidden_states = attn2.attn(q, k, v)
        hidden_states = hidden_states.flatten(2)
        hidden_states, _ = attn2.to_out(hidden_states)
        return hidden_states

    def _cam_conditioner_scale_shift(
        self,
        c2ws_plucker_emb: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if c2ws_plucker_emb is None:
            return None

        forward_context = get_forward_context()
        forward_batch = forward_context.forward_batch
        if not CausalLingBotWorldTransformer3DModel._should_cache_cam_conditioner(
            forward_batch
        ):
            return self.cam_conditioner.compute_scale_shift(c2ws_plucker_emb)

        cache = CausalLingBotWorldTransformer3DModel._get_request_cache(
            forward_batch, LINGBOT_CAM_CONDITIONER_CACHE
        )
        if cache is None:
            return self.cam_conditioner.compute_scale_shift(c2ws_plucker_emb)

        source_key = (
            c2ws_plucker_emb.data_ptr(),
            tuple(c2ws_plucker_emb.shape),
            tuple(c2ws_plucker_emb.stride()),
            c2ws_plucker_emb.dtype,
            c2ws_plucker_emb.device.type,
            c2ws_plucker_emb.device.index,
            _safe_tensor_version(c2ws_plucker_emb),
        )
        if cache.get("source_key") != source_key:
            cache.clear()
            cache["source_key"] = source_key
            cache["entries"] = {}

        entries = cache["entries"]
        entry_key = id(self.cam_conditioner)
        scale_shift = entries.get(entry_key)
        if scale_shift is None:
            scale_shift = self.cam_conditioner.compute_scale_shift(c2ws_plucker_emb)
            entries[entry_key] = scale_shift
        return scale_shift

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, ...],
        block_mask,
        kv_cache: CausalSelfAttentionKVCache | None = None,
        crossattn_cache: CrossAttentionKVCache | None = None,
        current_start: int = 0,
        cache_start: int | None = None,
        c2ws_plucker_emb: torch.Tensor | None = None,
        cam_conditioner_scale_shift: tuple[torch.Tensor, torch.Tensor] | None = None,
        update_cache_only: bool = False,
    ) -> torch.Tensor:
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        num_frames = temb.shape[1]
        seqlen_per_frame = hidden_states.shape[1] // num_frames
        orig_dtype = hidden_states.dtype
        e = self.scale_shift_table + temb.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = e.chunk(
            6, dim=2
        )
        norm_hidden_states = (
            (
                self.norm1(hidden_states.float()).unflatten(
                    dim=1, sizes=(num_frames, seqlen_per_frame)
                )
                * (1 + scale_msa)
                + shift_msa
            )
            .flatten(1, 2)
            .to(orig_dtype)
        )
        query, key, value = self._project_qkv(norm_hidden_states)
        if self.tp_rmsnorm:
            query = tensor_parallel_rms_norm(query, self.norm_q)
            key = tensor_parallel_rms_norm(key, self.norm_k)
        else:
            query = self.norm_q(query)
            key = self.norm_k(key)
        query = query.squeeze(1).unflatten(2, (self.local_num_heads, self.dim_head))
        key = key.squeeze(1).unflatten(2, (self.local_num_heads, self.dim_head))
        value = value.squeeze(1).unflatten(2, (self.local_num_heads, self.dim_head))

        attn_output = self.attn1(
            query,
            key,
            value,
            freqs_cis,
            block_mask,
            kv_cache,
            current_start,
            cache_start,
            update_cache_only=update_cache_only,
        )
        if update_cache_only:
            return hidden_states
        attn_output = attn_output.flatten(2)
        attn_output, _ = self.to_out(attn_output)
        attn_output = attn_output.squeeze(1)

        residual_zero = torch.zeros(
            (1,), device=hidden_states.device, dtype=hidden_states.dtype
        )
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states, attn_output, gate_msa, residual_zero, residual_zero
        )
        hidden_states = self.cam_conditioner(
            hidden_states.to(orig_dtype),
            c2ws_plucker_emb,
            (
                cam_conditioner_scale_shift
                if cam_conditioner_scale_shift is not None
                else self._cam_conditioner_scale_shift(c2ws_plucker_emb)
            ),
        )
        norm_hidden_states = self.self_attn_residual_norm.norm(hidden_states).to(
            orig_dtype
        )

        attn_output = self._cross_attn_with_cache(
            norm_hidden_states, encoder_hidden_states, crossattn_cache
        )
        norm_hidden_states, hidden_states = self.cross_attn_residual_norm(
            hidden_states, attn_output, 1, c_shift_msa, c_scale_msa
        )
        ff_output = self.ffn(norm_hidden_states.to(orig_dtype))
        hidden_states = self.mlp_residual(
            ff_output, c_gate_msa, hidden_states.to(orig_dtype)
        )
        return hidden_states.to(orig_dtype)


class CausalLingBotWorldTransformer3DModel(CausalWanTransformer3DModel):
    _fsdp_shard_conditions = LingBotWorldVideoConfig()._fsdp_shard_conditions
    _compile_conditions = LingBotWorldVideoConfig()._compile_conditions
    _supported_attention_backends = (
        LingBotWorldVideoConfig()._supported_attention_backends
    )
    param_names_mapping = LingBotWorldVideoConfig().param_names_mapping
    reverse_param_names_mapping = LingBotWorldVideoConfig().reverse_param_names_mapping
    lora_param_names_mapping = LingBotWorldVideoConfig().lora_param_names_mapping

    def __init__(
        self,
        config: LingBotWorldVideoConfig,
        hf_config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config, quant_config=quant_config)
        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.patch_embedding_wancamctrl = WanCamControlPatchEmbedding(
            in_chans=6 * 64,
            embed_dim=inner_dim,
            patch_size=config.patch_size,
        )
        self.c2ws_mlp = MLP(inner_dim, inner_dim, inner_dim, bias=True, act_type="silu")
        self.sp_size = get_sp_world_size()
        d = self.hidden_size // self.num_attention_heads
        self.rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
        self.rotary_emb = NDRotaryEmbedding(
            rope_dim_list=self.rope_dim_list,
            rope_theta=10000,
            dtype=(
                torch.float64
                if current_platform.is_float64_supported()
                else torch.float32
            ),
        )
        self.blocks = nn.ModuleList(
            [
                CausalLingBotWorldTransformerBlock(
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

    def post_load_weights(self) -> None:
        super().post_load_weights()
        for block in self.blocks:
            block.fuse_qkv_projection()

    @lru_cache(maxsize=8)
    def _compute_rope_for_sequence_shard_with_offset(
        self,
        local_len: int,
        token_start: int,
        frame_stride_local: int,
        width_local: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_indices = torch.arange(
            token_start,
            token_start + local_len,
            device=device,
            dtype=torch.long,
        )
        t_idx = token_indices // frame_stride_local
        rem = token_indices % frame_stride_local
        h_idx = rem // width_local
        w_idx = rem % width_local
        positions = torch.stack((t_idx, h_idx, w_idx), dim=1)
        return self.rotary_emb.forward_uncached(positions)

    def _prepare_c2ws_plucker_emb(
        self,
        hidden_states: torch.Tensor,
        c2ws_plucker_emb: torch.Tensor | None,
        forward_batch=None,
    ) -> torch.Tensor | None:
        if c2ws_plucker_emb is None:
            return None
        cache = self._get_request_cache(forward_batch, LINGBOT_C2WS_PLUCKER_EMB_CACHE)
        cache_key = (
            c2ws_plucker_emb.data_ptr(),
            tuple(c2ws_plucker_emb.shape),
            tuple(c2ws_plucker_emb.stride()),
            c2ws_plucker_emb.dtype,
            c2ws_plucker_emb.device.type,
            c2ws_plucker_emb.device.index,
            hidden_states.dtype,
            hidden_states.device.type,
            hidden_states.device.index,
        )
        if cache is not None and cache_key in cache:
            return cache[cache_key]

        c2ws_plucker_emb = self.patch_embedding_wancamctrl(
            c2ws_plucker_emb.to(device=hidden_states.device, dtype=hidden_states.dtype)
        )
        c2ws_plucker_emb = c2ws_plucker_emb + self.c2ws_mlp(c2ws_plucker_emb)
        if cache is not None:
            cache.clear()
            cache[cache_key] = c2ws_plucker_emb
        return c2ws_plucker_emb

    @staticmethod
    def _get_request_cache(forward_batch, name: str) -> dict | None:
        if forward_batch is None:
            return None
        session = getattr(forward_batch, "session", None)
        if session is not None:
            state = get_realtime_causal_dit_state(session)
            return state.runtime_cache.setdefault(name, {})
        extra = getattr(forward_batch, "extra", None)
        if extra is None:
            return None
        return extra.setdefault(name, {})

    @staticmethod
    def _should_cache_cam_conditioner(forward_batch) -> bool:
        return (
            forward_batch is not None
            and getattr(forward_batch, "enable_sequence_shard", False)
            and get_ulysses_parallel_world_size() > 1
        )

    @staticmethod
    def _all_crossattn_caches_initialized(
        crossattn_cache: list[CrossAttentionKVCache] | None,
    ) -> bool:
        return crossattn_cache is not None and all(
            cache.is_init for cache in crossattn_cache
        )

    def _prepare_cached_rope(
        self,
        *,
        forward_batch,
        post_patch_num_frames: int,
        post_patch_height: int,
        post_patch_width: int,
        start_frame: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        cache = self._get_request_cache(forward_batch, LINGBOT_ROPE_CACHE)
        cache_key = (
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            start_frame,
            device.type,
            device.index,
        )
        if cache is not None and cache_key in cache:
            return cache[cache_key]

        freqs_cos, freqs_sin = get_rotary_pos_embed(
            (
                post_patch_num_frames * get_sp_world_size(),
                post_patch_height,
                post_patch_width,
            ),
            self.hidden_size,
            self.num_attention_heads,
            self.rope_dim_list,
            dtype=(
                torch.float64
                if current_platform.is_float64_supported()
                else torch.float32
            ),
            rope_theta=10000,
            start_frame=start_frame,
        )
        freqs_cos = freqs_cos.to(device).float()
        freqs_sin = freqs_sin.to(device).float()
        freqs_cis: tuple[torch.Tensor, ...] = (freqs_cos, freqs_sin)
        if _is_cuda:
            freqs_cis = (
                freqs_cos,
                freqs_sin,
                torch.cat([freqs_cos.contiguous(), freqs_sin.contiguous()], dim=-1),
            )
        if cache is not None:
            cache.clear()
            cache[cache_key] = freqs_cis
        return freqs_cis

    def _prepare_cached_rope_for_sequence_shard(
        self,
        *,
        forward_batch,
        local_seq_len: int,
        token_start: int,
        frame_stride: int,
        post_patch_width: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        cache = self._get_request_cache(
            forward_batch, LINGBOT_SEQUENCE_SHARD_ROPE_CACHE
        )
        cache_key = (
            local_seq_len,
            token_start,
            frame_stride,
            post_patch_width,
            device.type,
            device.index,
        )
        if cache is not None and cache_key in cache:
            return cache[cache_key]

        freqs_cos, freqs_sin = self._compute_rope_for_sequence_shard_with_offset(
            local_seq_len,
            token_start,
            frame_stride,
            post_patch_width,
            device,
        )
        freqs_cos = freqs_cos.float()
        freqs_sin = freqs_sin.float()
        freqs_cis: tuple[torch.Tensor, ...] = (freqs_cos, freqs_sin)
        if _is_cuda:
            freqs_cis = (
                freqs_cos,
                freqs_sin,
                torch.cat([freqs_cos.contiguous(), freqs_sin.contiguous()], dim=-1),
            )
        if cache is not None:
            cache.clear()
            cache[cache_key] = freqs_cis
        return freqs_cis

    def _prepare_condition_embeddings(
        self,
        *,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None,
        crossattn_cache: list[CrossAttentionKVCache] | None,
    ):
        forward_batch = get_forward_context().forward_batch
        temb, timestep_proj = self._prepare_cached_time_embeddings(
            timestep=timestep,
            forward_batch=forward_batch,
        )
        if self._all_crossattn_caches_initialized(crossattn_cache):
            return temb, timestep_proj, encoder_hidden_states, None

        encoder_hidden_states = self.condition_embedder.text_embedder(
            encoder_hidden_states
        )
        if encoder_hidden_states_image is not None:
            assert self.condition_embedder.image_embedder is not None
            encoder_hidden_states_image = self.condition_embedder.image_embedder(
                encoder_hidden_states_image
            )

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image

    def _prepare_cached_time_embeddings(
        self,
        *,
        timestep: torch.LongTensor,
        forward_batch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache = self._get_request_cache(forward_batch, LINGBOT_TIME_EMBEDDINGS_CACHE)
        current_timestep = get_forward_context().current_timestep
        cache_key = (
            current_timestep,
            tuple(timestep.shape),
            timestep.dtype,
            timestep.device.type,
            timestep.device.index,
        )
        if cache is not None and cache_key in cache:
            return cache[cache_key]

        temb = self.condition_embedder.time_embedder(timestep.flatten())
        timestep_proj = self.condition_embedder.time_modulation(temb)
        if cache is not None:
            cache[cache_key] = (temb, timestep_proj)
        return temb, timestep_proj

    def _prepare_cam_conditioner_scale_shifts(
        self,
        c2ws_plucker_emb: torch.Tensor | None,
        forward_batch,
    ) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
        if c2ws_plucker_emb is None:
            return None

        forward_context = get_forward_context()
        if forward_context.current_timestep < 0:
            return None

        if not self._should_cache_cam_conditioner(forward_batch):
            return None

        cache = self._get_request_cache(forward_batch, LINGBOT_CAM_CONDITIONER_CACHE)
        if cache is None:
            return None

        source_key = (
            c2ws_plucker_emb.data_ptr(),
            tuple(c2ws_plucker_emb.shape),
            tuple(c2ws_plucker_emb.stride()),
            c2ws_plucker_emb.dtype,
            c2ws_plucker_emb.device.type,
            c2ws_plucker_emb.device.index,
            _safe_tensor_version(c2ws_plucker_emb),
        )
        if cache.get("source_key") != source_key:
            cache.clear()
            cache["source_key"] = source_key
            cache["entries"] = {}

        entries = cache["entries"]
        scale_shifts = []
        for block in self.blocks:
            entry_key = id(block.cam_conditioner)
            scale_shift = entries.get(entry_key)
            if scale_shift is None:
                scale_shift = block.cam_conditioner.compute_scale_shift(
                    c2ws_plucker_emb
                )
                entries[entry_key] = scale_shift
            scale_shifts.append(scale_shift)
        return scale_shifts

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        kv_cache: list[CausalSelfAttentionKVCache] | None = None,
        crossattn_cache: list[CrossAttentionKVCache] | None = None,
        current_start: int = 0,
        cache_start: int = 0,
        start_frame: int = 0,
        c2ws_plucker_emb: torch.Tensor | None = None,
        skip_final_projection: bool = False,
    ) -> torch.Tensor:
        forward_batch = get_forward_context().forward_batch
        sequence_shard_enabled = (
            forward_batch is not None
            and getattr(forward_batch, "enable_sequence_shard", False)
            and self.sp_size > 1
        )
        if sequence_shard_enabled:
            if get_ring_parallel_world_size() > 1:
                raise NotImplementedError(
                    "LingBot causal sequence sharding currently supports ulysses_degree > 1 with ring_degree = 1 only."
                )
            if get_ulysses_parallel_world_size() <= 1:
                raise NotImplementedError(
                    "LingBot causal sequence sharding currently requires ulysses_degree > 1."
                )
        orig_dtype = hidden_states.dtype
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]
        if (
            isinstance(encoder_hidden_states_image, list)
            and len(encoder_hidden_states_image) > 0
        ):
            encoder_hidden_states_image = encoder_hidden_states_image[0]
        else:
            encoder_hidden_states_image = None

        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        if sequence_shard_enabled:
            seq_shard_splits = _compute_sequence_splits(
                post_patch_num_frames * post_patch_height * post_patch_width,
                self.sp_size,
            )
            forward_batch.sequence_shard_splits = tuple(seq_shard_splits)
        if not sequence_shard_enabled:
            freqs_cis = self._prepare_cached_rope(
                forward_batch=forward_batch,
                post_patch_num_frames=post_patch_num_frames,
                post_patch_height=post_patch_height,
                post_patch_width=post_patch_width,
                start_frame=start_frame,
                device=hidden_states.device,
            )

        hidden_states = self.patch_embedding(hidden_states).flatten(2).transpose(1, 2)
        c2ws_plucker_emb = self._prepare_c2ws_plucker_emb(
            hidden_states, c2ws_plucker_emb, forward_batch
        )
        if sequence_shard_enabled:
            sp_rank = get_sp_parallel_rank()
            seq_shard_splits = list(forward_batch.sequence_shard_splits)
            local_seq_len = seq_shard_splits[sp_rank]
            hidden_states = _sequence_shard_tensor(
                hidden_states, seq_shard_splits, sp_rank
            )
            if c2ws_plucker_emb is not None:
                c2ws_plucker_emb = _sequence_shard_tensor(
                    c2ws_plucker_emb, seq_shard_splits, sp_rank
                )
            frame_stride = post_patch_height * post_patch_width
            token_start = start_frame * frame_stride + sum(seq_shard_splits[:sp_rank])
            freqs_cis = self._prepare_cached_rope_for_sequence_shard(
                forward_batch=forward_batch,
                local_seq_len=local_seq_len,
                token_start=token_start,
                frame_stride=frame_stride,
                post_patch_width=post_patch_width,
                device=hidden_states.device,
            )

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
            self._prepare_condition_embeddings(
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_image=encoder_hidden_states_image,
                crossattn_cache=crossattn_cache,
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
        cam_conditioner_scale_shifts = self._prepare_cam_conditioner_scale_shifts(
            c2ws_plucker_emb, forward_batch
        )

        for block_index, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                freqs_cis,
                block_mask=self.block_mask,
                kv_cache=kv_cache[block_index],
                crossattn_cache=crossattn_cache[block_index],
                current_start=current_start,
                cache_start=cache_start,
                c2ws_plucker_emb=c2ws_plucker_emb,
                cam_conditioner_scale_shift=(
                    None
                    if cam_conditioner_scale_shifts is None
                    else cam_conditioner_scale_shifts[block_index]
                ),
                update_cache_only=skip_final_projection
                and block_index == len(self.blocks) - 1,
            )

        if skip_final_projection:
            return hidden_states

        temb = temb.unflatten(dim=0, sizes=timestep.shape).unsqueeze(2)
        shift, scale = (self.scale_shift_table.unsqueeze(1) + temb).chunk(2, dim=2)
        hidden_states = self.norm_out(hidden_states, shift, scale)
        hidden_states = self.proj_out(hidden_states)
        if sequence_shard_enabled:
            hidden_states = _sequence_all_gather_varlen(
                hidden_states.contiguous(),
                list(forward_batch.sequence_shard_splits),
                get_sp_group().device_group,
            )
        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)


EntryClass = [
    LingBotWorldTransformer3DModel,
    CausalLingBotWorldTransformer3DModel,
]
