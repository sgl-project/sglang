# SPDX-License-Identifier: Apache-2.0

"""SRT-native BAGEL Qwen2-MoT language model pieces.

This file intentionally keeps the first native BAGEL step narrow: it expresses
the MoT layer shape and runs the understanding branch through normal SRT
ForwardBatch/KV paths. The generation branch modules are present for checkpoint
loading, but their token routing is left to the later G-forward step.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen2 import Qwen2ForCausalLM, Qwen2MLP, Qwen2Model
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix
from sglang.srt.utils.hf_transformers_utils import get_rope_config


class BAGELQwen2MoTAttention(nn.Module):
    """Qwen2 attention with BAGEL MoT branch parameters.

    The default forward is the BAGEL `mode="und"` branch: normal Qwen2 qkv
    projection, QK norm, RoPE, SRT RadixAttention, normal output projection.
    """

    def __init__(
        self,
        config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        rope_theta, rope_scaling = get_rope_config(config)
        head_dim = getattr(config, "head_dim", None)
        self.head_dim = head_dim or config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        self.q_norm = _make_qk_norm(config, self.head_dim)
        self.k_norm = _make_qk_norm(config, self.head_dim)
        self.q_norm_moe_gen = _make_qk_norm(config, self.head_dim)
        self.k_norm_moe_gen = _make_qk_norm(config, self.head_dim)
        self.alt_stream = alt_stream
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            dual_chunk_attention_config=getattr(
                config, "dual_chunk_attention_config", None
            ),
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

        self.qkv_proj_moe_gen = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj_moe_gen", prefix),
        )
        self.o_proj_moe_gen = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj_moe_gen", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        *,
        mode: str = "und",
    ) -> torch.Tensor:
        if mode != "und":
            raise NotImplementedError(
                "BAGEL Qwen2-MoT native gen branch is not implemented yet"
            )

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = apply_qk_norm(
            q=q,
            k=k,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            head_dim=self.head_dim,
            alt_stream=self.alt_stream,
        )
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class BAGELQwen2MoTDecoderLayer(nn.Module):
    """BAGEL Qwen2-MoT decoder layer with native `mode="und"` forward."""

    def __init__(
        self,
        config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.self_attn = BAGELQwen2MoTAttention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
        )
        self.mlp = Qwen2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.mlp_moe_gen = Qwen2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp_moe_gen", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_moe_gen = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm_moe_gen = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class BAGELQwen2MoTModel(Qwen2Model):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=BAGELQwen2MoTDecoderLayer,
            alt_stream=alt_stream,
        )
        self.use_moe = "Mo" in getattr(config, "layer_module", "Qwen2MoTDecoderLayer")
        if self.use_moe and self.pp_group.is_last_rank:
            self.norm_moe_gen = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class BAGELQwen2MoTForCausalLM(Qwen2ForCausalLM):
    """SRT-native BAGEL language model shell for U-forward bring-up."""

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = BAGELQwen2MoTModel(
            config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        self.capture_aux_hidden_states = False

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        return super().load_weights(_iter_bagel_language_model_weights(weights))


def _iter_bagel_language_model_weights(weights: Iterable[Tuple[str, torch.Tensor]]):
    for name, loaded_weight in weights:
        if name.startswith("language_model."):
            yield name[len("language_model.") :], loaded_weight
        elif _is_qwen2_language_model_key(name):
            yield name, loaded_weight


def _is_qwen2_language_model_key(name: str) -> bool:
    return name.startswith(
        (
            "model.",
            "lm_head.",
        )
    )


def _make_qk_norm(config, head_dim: int) -> RMSNorm:
    norm_kwargs = (
        dict(
            weight_dtype=torch.float32,
            cast_x_before_out_mul=True,
        )
        if get_global_server_args().rl_on_policy_target is not None
        else {}
    )
    return RMSNorm(head_dim, eps=config.rms_norm_eps, **norm_kwargs)


EntryClass = [BAGELQwen2MoTForCausalLM]
