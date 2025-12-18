# coding=utf-8
"""SGLang native implementation of FastDiffuser model."""
import logging
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
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
from sglang.srt.models.qwen2 import Qwen2MLP  # Reuse MLP from Qwen2
from sglang.srt.models.utils import (
    create_fused_set_kv_buffer_arg,
    enable_fused_set_kv_buffer,
)
from sglang.srt.utils import add_prefix, is_cuda, make_layers

logger = logging.getLogger(__name__)
_is_cuda = is_cuda()


class FastDiffuserAttention(nn.Module):
    """Bidirectional attention for FastDiffuser using SGLang RadixAttention."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_kv_heads = config.num_key_value_heads

        tp_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_size == 0

        if self.total_kv_heads >= tp_size:
            assert self.total_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_kv_heads == 0

        self.num_heads = self.total_num_heads // tp_size
        self.head_dim = config.head_dim or (self.hidden_size // self.total_num_heads)
        self.q_size = self.head_dim * self.num_heads

        self.num_kv_heads = max(1, self.total_kv_heads // tp_size)
        self.kv_size = self.num_kv_heads * self.head_dim

        self.scale = self.head_dim**-0.5

        # Q/K normalization (FastDiffuser uses this)
        self.use_qk_norm = getattr(config, "use_qk_norm", True)

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )

        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        # Rotary embeddings
        self.rotary_dim = self.head_dim
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=getattr(config, "rope_scaling", None),
        )

        # CRITICAL: Use ENCODER_ONLY for bidirectional attention (like LLADA2)
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scale,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            attn_type=AttentionType.ENCODER_ONLY,
            prefix=add_prefix("attn", prefix),
        )

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_by_head = q.reshape(-1, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        k_by_head = k.reshape(-1, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        q = q_by_head.view(q.shape)
        k = k_by_head.view(k.shape)
        return q, k

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
            q, k = self._apply_qk_norm(q, k)

        # CRITICAL: Use fused_set_kv_buffer for proper KV cache management
        q, k = self.rotary_emb(
            positions,
            q,
            k,
            fused_set_kv_buffer_arg=(
                create_fused_set_kv_buffer_arg(
                    value=v,
                    layer=self.attn,
                    forward_batch=forward_batch,
                )
                if enable_fused_set_kv_buffer(forward_batch)
                else None
            ),
        )

        context_layer = self.attn(
            q,
            k,
            v,
            forward_batch,
            save_kv_cache=not enable_fused_set_kv_buffer(forward_batch),
        )

        attn_output, _ = self.o_proj(context_layer)
        return attn_output


class FastDiffuserDecoderLayer(nn.Module):
    """Single FastDiffuser decoder layer (transformer block)."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        hidden_size = config.hidden_size

        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.self_attn = FastDiffuserAttention(
            config,
            layer_id,
            quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

        # Reuse Qwen2MLP with config parameters
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-attention LayerNorm with residual
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Attention
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Pre-MLP LayerNorm with residual
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # MLP
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class FastDiffuserModel(nn.Module):
    """FastDiffuser base model (encoder)."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.hidden_size

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                self.embed_dim,
                quant_config=quant_config,
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: FastDiffuserDecoderLayer(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )

        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(self.embed_dim, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            if not forward_batch.forward_mode.is_idle():
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)
            return hidden_states


class DiffEncoderModel(nn.Module):
    """
    FastDiffuser Diffusion LM - native SGLang implementation.

    This is the top-level model class that will be registered in SGLang's ModelRegistry.
    It wraps FastDiffuserModel (encoder) and adds the diffusion_head.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config

        # Encoder model
        self.model = FastDiffuserModel(
            config,
            quant_config,
            prefix=add_prefix("encoder", prefix),
        )

        # Diffusion head (projection to vocab)
        if self.pp_group.is_last_rank:
            self.diffusion_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("diffusion_head", prefix),
            )

        # CRITICAL: return_full_logits=True for DLLM models
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
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.diffusion_head, forward_batch
            )
        else:
            return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights from checkpoint."""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id) - order matters!
            # QKV must come before gate_up to avoid conflicts
            # QKV shard_ids must be strings "q", "k", "v" for QKVParallelLinear
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
            # Gate/up shard_ids are integers 0, 1 for MergedColumnParallelLinear
            (".mlp.gate_up_proj", ".mlp.gate_proj", 0),
            (".mlp.gate_up_proj", ".mlp.up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            # Handle HF model prefix mapping first
            # HF uses "encoder.X" but we use "model.X"
            if name.startswith("encoder."):
                name = name.replace("encoder.", "model.", 1)

            # Handle weight mapping for stacked parameters
            is_stacked = False
            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name in name:
                    name = name.replace(shard_name, param_name)
                    if name in params_dict:
                        param = params_dict[name]
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight, shard_id)
                        is_stacked = True
                    else:
                        logger.warning(f"Stacked weight {name} not found in model parameters.")
                    break

            if is_stacked:
                continue

            if name not in params_dict:
                # Only warn for unexpected weights, not the ones we intentionally skip
                if not any(skip in name for skip in ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]):
                    logger.warning(f"Weight {name} not found in model parameters. Skipping.")
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


# Register this model in SGLang's ModelRegistry
EntryClass = DiffEncoderModel
