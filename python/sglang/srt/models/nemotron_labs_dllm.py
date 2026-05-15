# coding=utf-8
"""
SGLang native implementations of Nemotron Labs Diffusion LMs.

Two architectures:
  - DiffEncoderModel       : Qwen3-based  (nvidia/Nemotron-Diffusion-Research-8B-v0)
  - NemotronLabsDiffusionModel : (nvidia/Nemotron-Labs-Diffusion-8B)

Both are bidirectional (encoder-only attention) masked diffusion language models
that use an iterative denoising loop at inference time (FastDiffuser / LLaDA-style).

Weight layout in HF checkpoints
---------------------------------
  encoder.embed_tokens.weight
  encoder.layers.N.{input,post_attention}_layernorm.weight
  encoder.layers.N.self_attn.{q,k,v,o}_proj.weight
  encoder.layers.N.self_attn.{q,k}_norm.weight   (Qwen only)
  encoder.layers.N.mlp.{gate,up,down}_proj.weight
  encoder.norm.weight
  diffusion_head.weight

SGLang parameter names
  model.*   (encoder.* → model.*)
  diffusion_head.weight  (unchanged)
"""

import logging
from typing import Iterable, Optional, Tuple, Union

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama import LlamaMLP
from sglang.srt.models.qwen2 import Qwen2MLP
from sglang.srt.models.utils import (
    WeightsMapper,
    create_fused_set_kv_buffer_arg,
    enable_fused_set_kv_buffer,
)
from sglang.srt.utils import add_prefix, is_cuda, make_layers

logger = logging.getLogger(__name__)
_is_cuda = is_cuda()

# ---------------------------------------------------------------------------
# DiffEncoderModel (Qwen3-based)
# ---------------------------------------------------------------------------


class QwenDiffuserAttention(nn.Module):
    """
    Bidirectional attention for the Qwen3-based Nemotron Labs diffusion model.

    Differences from standard Qwen3 attention:
      - Uses AttentionType.ENCODER_ONLY (bidirectional / full attention).
      - Applies per-head RMSNorm to Q and K (disable_qk_norm=False in config).
      - Always saves the KV cache during DLLM iterations so that updated
        token embeddings propagate correctly across denoising steps.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_kv_heads = config.num_key_value_heads
        tp_size = get_tensor_model_parallel_world_size()

        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_kv_heads // tp_size)
        self.head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scale = self.head_dim**-0.5

        # Qwen3 uses per-head Q/K RMSNorm unless disable_qk_norm=True
        self.use_qk_norm = not getattr(config, "disable_qk_norm", False)

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_kv_heads,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=getattr(config, "rope_theta", 1_000_000.0),
            rope_scaling=getattr(config, "rope_scaling", None),
        )

        # ENCODER_ONLY → full bidirectional attention (no causal mask)
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scale,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            attn_type=AttentionType.ENCODER_ONLY,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if self.use_qk_norm:
            q = self.q_norm(q.reshape(-1, self.head_dim)).view(q.shape)
            k = self.k_norm(k.reshape(-1, self.head_dim)).view(k.shape)

        # Use fused KV write (like LLaDA2): fuses K/V cache write with rotary computation
        # when on CUDA with bfloat16 KV cache.  During DLLM denoising, out_cache_loc
        # points to the current block's slots, so fused write still rewrites KV correctly.
        can_fuse = (
            self.head_dim == self.rotary_emb.rotary_dim
            and enable_fused_set_kv_buffer(forward_batch)
        )
        q, k = self.rotary_emb(
            positions,
            q,
            k,
            fused_set_kv_buffer_arg=(
                create_fused_set_kv_buffer_arg(v, self.attn, forward_batch)
                if can_fuse
                else None
            ),
        )
        context = self.attn(q, k, v, forward_batch, save_kv_cache=not can_fuse)

        out, _ = self.o_proj(context)
        return out


class QwenDiffuserLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = QwenDiffuserAttention(
            config, layer_id, quant_config, prefix=add_prefix("self_attn", prefix)
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = Qwen2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, forward_batch)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class QwenDiffuserModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: QwenDiffuserLayer(
                config, idx, quant_config, prefix=prefix
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )

        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            hidden_states = (
                self.embed_tokens(input_ids) if input_embeds is None else input_embeds
            )
            residual = None
        else:
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            hidden_states, residual = self.layers[i](
                positions, hidden_states, forward_batch, residual
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        if not forward_batch.forward_mode.is_idle():
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class DiffEncoderModel(nn.Module):
    """
    Qwen3-based Nemotron Labs Diffusion LM (nvidia/Nemotron-Diffusion-Research-8B-v0).

    HF architecture name: DiffEncoderModel
    mask_token_id: 151662
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config

        # The HF checkpoint stores backbone weights under "encoder.*".
        # We name our backbone "model" internally and remap on load.
        self.model = QwenDiffuserModel(
            config,
            quant_config,
            prefix=add_prefix("model", prefix),
        )

        if self.pp_group.is_last_rank:
            self.diffusion_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("diffusion_head", prefix),
            )
        self.logits_processor = LogitsProcessor(config, return_full_logits=True)

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
        )
        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.diffusion_head, forward_batch
            )
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
            (".mlp.gate_up_proj", ".mlp.gate_proj", 0),
            (".mlp.gate_up_proj", ".mlp.up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            # HF stores backbone as "encoder.*"; our backbone is "model.*"
            if name.startswith("encoder."):
                name = "model." + name[len("encoder.") :]

            is_stacked = False
            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                if name in params_dict:
                    weight_loader = getattr(
                        params_dict[name], "weight_loader", default_weight_loader
                    )
                    weight_loader(params_dict[name], loaded_weight, shard_id)
                is_stacked = True
                break

            if is_stacked:
                continue

            if name not in params_dict:
                continue
            weight_loader = getattr(
                params_dict[name], "weight_loader", default_weight_loader
            )
            weight_loader(params_dict[name], loaded_weight)


# ---------------------------------------------------------------------------
# NemotronLabsDiffusionModel
# ---------------------------------------------------------------------------


def _llama4_q_scale(positions: torch.Tensor, beta: float, max_pos: int) -> torch.Tensor:
    """
    Llama-4 per-token attention scale factor applied to Q.
    scale = 1 + beta * log(1 + floor(pos / max_pos))
    Returns shape [seq_len, 1] for broadcasting over heads.
    """
    return (
        1.0 + beta * torch.log(1.0 + torch.floor(positions.float() / max_pos))
    ).unsqueeze(-1)


class NemotronLabsDiffusionAttention(nn.Module):
    """
    Bidirectional attention for the Nemotron Labs diffusion model.

    Differences from QwenDiffuserAttention:
      - No per-head Q/K RMSNorm.
      - Applies Llama-4 per-token Q scaling after RoPE.
      - RoPE uses YaRN rope_scaling from config.rope_parameters.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_kv_heads = config.num_key_value_heads
        tp_size = get_tensor_model_parallel_world_size()

        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_kv_heads // tp_size)
        self.head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scale = self.head_dim**-0.5

        # Llama-4 Q scaling from rope_parameters
        rope_params = getattr(config, "rope_parameters", {}) or {}
        self.llama4_beta: Optional[float] = rope_params.get(
            "llama_4_scaling_beta", None
        )
        self.max_pos = rope_params.get(
            "original_max_position_embeddings",
            getattr(config, "max_position_embeddings", 16384),
        )

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_kv_heads,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        # YaRN RoPE — rope_scaling and rope_parameters hold the same dict
        rope_scaling = getattr(config, "rope_scaling", None) or rope_params or None
        rope_base = rope_params.get(
            "rope_theta", getattr(config, "rope_theta", 1_000_000.0)
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=rope_base,
            rope_scaling=rope_scaling,
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scale,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            attn_type=AttentionType.ENCODER_ONLY,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # qkv.split() produces non-contiguous views. The fused rotary_emb
        # CUDA kernel handles strides, but the eager fallback's .view() needs
        # contiguous tensors.  Only pay the copy cost when NOT in a CUDA graph.
        if not forward_batch.forward_mode.is_dllm_extend():
            q, k = q.contiguous(), k.contiguous()
        q, k = self.rotary_emb(positions, q, k)

        # Llama-4 per-token Q scaling
        if self.llama4_beta is not None:
            scale = _llama4_q_scale(positions, self.llama4_beta, self.max_pos).to(
                q.dtype
            )
            q = q.view(-1, self.num_heads, self.head_dim)
            q = (q * scale.unsqueeze(1)).view(-1, self.num_heads * self.head_dim)

        context = self.attn(q, k, v, forward_batch, save_kv_cache=True)
        out, _ = self.o_proj(context)
        return out


class NemotronLabsDiffusionLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = NemotronLabsDiffusionAttention(
            config, layer_id, quant_config, prefix=add_prefix("self_attn", prefix)
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, forward_batch)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class NemotronLabsDiffusionEncoder(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: NemotronLabsDiffusionLayer(
                config, idx, quant_config, prefix=prefix
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )

        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            hidden_states = (
                self.embed_tokens(input_ids) if input_embeds is None else input_embeds
            )
            residual = None
        else:
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            hidden_states, residual = self.layers[i](
                positions, hidden_states, forward_batch, residual
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        if not forward_batch.forward_mode.is_idle():
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class NemotronLabsDiffusionModel(nn.Module):
    """
    Nemotron Labs Diffusion LM (nvidia/Nemotron-Labs-Diffusion-8B).

    HF architecture name: NemotronLabsDiffusionModel
    mask_token_id: 100
    """

    # Maps HuggingFace checkpoint prefixes to SGLang internal names.
    # Used by ModelOptMixedPrecisionConfig to resolve per-layer quant methods.
    hf_to_sglang_mapper = WeightsMapper(orig_to_new_prefix={"encoder.": "model."})

    # Fused-layer → component mapping for mixed-precision quant resolution.
    # `get_quant_config` reads this and passes it to ModelOptMixedPrecisionConfig
    # so that `_resolve_quant_algo` can find the quant algo for fused layers.
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config

        self.model = NemotronLabsDiffusionEncoder(
            config,
            quant_config,
            prefix=add_prefix("model", prefix),
        )

        if self.pp_group.is_last_rank:
            self.diffusion_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("diffusion_head", prefix),
            )
        self.logits_processor = LogitsProcessor(config, return_full_logits=True)

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
        )
        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.diffusion_head, forward_batch
            )
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
            (".mlp.gate_up_proj", ".mlp.gate_proj", 0),
            (".mlp.gate_up_proj", ".mlp.up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            if name.startswith("encoder."):
                name = "model." + name[len("encoder.") :]

            is_stacked = False
            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                if name in params_dict:
                    weight_loader = getattr(
                        params_dict[name], "weight_loader", default_weight_loader
                    )
                    weight_loader(params_dict[name], loaded_weight, shard_id)
                is_stacked = True
                break

            if is_stacked:
                continue

            if name not in params_dict:
                continue
            weight_loader = getattr(
                params_dict[name], "weight_loader", default_weight_loader
            )
            weight_loader(params_dict[name], loaded_weight)


# Register both model architectures in SGLang's ModelRegistry.
EntryClass = [DiffEncoderModel, NemotronLabsDiffusionModel]
