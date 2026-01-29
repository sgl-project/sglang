"""
LFM2-MoE (Liquid Foundation Model 2 - Mixture of Experts) implementation for SGLang.

This is a hybrid architecture with attention, ShortConv, and MoE layers:
- Attention layers use standard KV cache (RadixAttention)
- Conv layers use MambaPool for state caching (via HybridReqToTokenPool)
- First `num_dense_layers` use dense MLP, rest use MoE with sigmoid routing

Key MoE characteristics:
- Sigmoid routing (not softmax) - auxiliary-loss-free style
- Expert bias (fp32) affects selection but not weighting
- Post-hoc normalization of top-k weights
"""

from typing import Iterable, Optional, Set, Tuple

import torch
from torch import nn

from sglang.srt.configs.lfm2_moe import Lfm2MoeConfig
from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.attention.mamba.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from sglang.srt.utils import add_prefix, make_layers, set_weight_attrs


class Lfm2MoeMLP(nn.Module):
    """Dense MLP for first N layers (before MoE kicks in)."""

    def __init__(
        self,
        config: Lfm2MoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        # Use MergedColumnParallelLinear for w1/w3 (gate/up projections)
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        out, _ = self.down_proj(x)
        return out


class Lfm2MoeSparseMoeBlock(nn.Module):
    """
    Sparse MoE block with sigmoid routing using optimized FusedMoE.

    Key features:
    - Sigmoid scoring (not softmax) - auxiliary-loss-free style
    - Expert bias (fp32) for load balancing
    - Bias affects selection only, not weighting
    - Uses FusedMoE for efficient batched expert computation
    """

    def __init__(
        self,
        config: Lfm2MoeConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        # Gate (router) - outputs logits for each expert
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )

        # Expert bias (fp32) - affects selection but not weighting
        if config.use_expert_bias:
            self.expert_bias = nn.Parameter(
                torch.zeros(config.num_experts, dtype=torch.float32)
            )
        else:
            self.register_parameter("expert_bias", None)

        # TopK selector with sigmoid scoring
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            layer_id=layer_idx,
            renormalize=config.norm_topk_prob,
            scoring_func="sigmoid",
            correction_bias=self.expert_bias if config.use_expert_bias else None,
        )

        # FusedMoE for efficient batched expert computation
        # Note: We intentionally do NOT pass routed_scaling_factor to FusedMoE.
        # While FusedMoE supports it, passing it there increases numerical
        # differences vs HuggingFace (likely due to different code paths in the
        # Triton runner when scaling_factor != None). We apply it manually below.
        self.experts = FusedMoE(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=layer_idx,
            reduce_results=True,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Optimized expert forward pass using FusedMoE."""
        # Get router logits
        router_logits, _ = self.gate(hidden_states)

        # Select top-k experts with sigmoid scoring
        topk_output = self.topk(hidden_states, router_logits)

        # Run fused expert computation
        final_hidden_states = self.experts(hidden_states, topk_output)

        # Apply routed scaling factor (see __init__ comment for why not in FusedMoE)
        return final_hidden_states * self.routed_scaling_factor


class Lfm2MoeAttention(nn.Module):
    """Grouped-query attention with RoPE and Q/K layernorm."""

    def __init__(
        self,
        config: Lfm2MoeConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.total_num_heads
        self.scaling = self.head_dim**-0.5

        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is not None and "rope_theta" in rope_parameters:
            rope_theta = rope_parameters["rope_theta"]
        else:
            rope_theta = getattr(config, "rope_theta", 1000000.0)

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=getattr(config, "max_position_embeddings", 128000),
            rope_scaling=getattr(config, "rope_scaling", None),
            base=rope_theta,
            is_neox_style=True,
            dtype=torch.get_default_dtype(),
        )

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.out_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("out_proj", prefix),
        )

        self.q_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)

        self.num_local_q_heads = self.qkv_proj.num_heads
        self.num_local_kv_heads = self.qkv_proj.num_kv_heads

        self.attn = RadixAttention(
            num_heads=self.num_local_q_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_local_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        T = hidden_states.shape[0]
        qkv, _ = self.qkv_proj(hidden_states)

        q_size = self.num_local_q_heads * self.head_dim
        kv_size = self.num_local_kv_heads * self.head_dim
        q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)

        q = q.reshape(T, self.num_local_q_heads, self.head_dim)
        k = k.reshape(T, self.num_local_kv_heads, self.head_dim)

        q = self.q_layernorm(q.reshape(-1, self.head_dim)).reshape(
            T, self.num_local_q_heads, self.head_dim
        )
        k = self.k_layernorm(k.reshape(-1, self.head_dim)).reshape(
            T, self.num_local_kv_heads, self.head_dim
        )

        q, k = self.rotary_emb(positions, q, k)

        attn_out = self.attn(q.reshape(T, -1), k.reshape(T, -1), v, forward_batch)
        out, _ = self.out_proj(attn_out)
        return out


class Lfm2MoeShortConv(nn.Module):
    """
    Gated short convolution layer using optimized causal_conv1d kernels.

    Architecture: in_proj -> split(B, C, x) -> Bx -> conv1d -> C*conv_out -> out_proj
    - Supports tensor parallelism: hidden dimension is sharded across TP ranks
    """

    def __init__(
        self,
        config: Lfm2MoeConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.conv_kernel = int(config.conv_L_cache)
        self.use_bias = bool(config.conv_bias)
        self.hidden_size = config.hidden_size

        # Get tensor parallel size for sharding
        self.tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = self.hidden_size // self.tp_size

        # Use MergedColumnParallelLinear so each output (B, C, x) is sharded separately
        self.in_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.hidden_size] * 3,  # B, C, x each get hidden_size
            bias=self.use_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj",
        )
        self.out_proj = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=self.use_bias,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        # Conv weights sharded along hidden dimension: (hidden_size/tp, kernel_size)
        self.conv_weight = nn.Parameter(
            torch.empty(self.hidden_size_per_partition, self.conv_kernel)
        )
        set_weight_attrs(self.conv_weight, {"weight_loader": sharded_weight_loader(0)})
        if self.use_bias:
            self.conv_bias = nn.Parameter(torch.empty(self.hidden_size_per_partition))
            set_weight_attrs(
                self.conv_bias, {"weight_loader": sharded_weight_loader(0)}
            )
        else:
            self.register_parameter("conv_bias", None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_idle():
            return hidden_states

        layer_cache = forward_batch.req_to_token_pool.mamba2_layer_cache(self.layer_idx)
        conv_state = layer_cache.conv[0]
        req_pool_indices = forward_batch.req_pool_indices

        proj, _ = self.in_proj(hidden_states)
        B_gate, C_gate, x = proj.chunk(3, dim=-1)
        Bx = B_gate * x

        if forward_batch.forward_mode.is_decode():
            conv_out = causal_conv1d_update(
                Bx,
                conv_state,
                self.conv_weight,
                self.conv_bias,
                activation=None,
                conv_state_indices=req_pool_indices.to(torch.int32),
            )
        else:
            T = hidden_states.shape[0]
            Bx_t = Bx.transpose(0, 1).contiguous()

            # Build query_start_loc for variable-length sequences
            # causal_conv1d_fn expects [start0, start1, ..., startN, T]
            extend_start_loc = forward_batch.extend_start_loc
            if extend_start_loc is not None and len(extend_start_loc) > 1:
                # Multiple sequences: append T to extend_start_loc
                # Allocate and fill to avoid torch.cat overhead
                query_start_loc = extend_start_loc.new_empty(len(extend_start_loc) + 1)
                query_start_loc[:-1] = extend_start_loc
                query_start_loc[-1] = T
                cache_indices = req_pool_indices.to(torch.int32)
            else:
                # Single sequence: [0, T]
                query_start_loc = hidden_states.new_tensor([0, T], dtype=torch.int32)
                cache_indices = req_pool_indices[:1].to(torch.int32)

            conv_out = causal_conv1d_fn(
                Bx_t,
                self.conv_weight,
                self.conv_bias,
                query_start_loc=query_start_loc,
                cache_indices=cache_indices,
                has_initial_state=None,
                conv_states=conv_state,
                activation=None,
            ).transpose(0, 1)

        output, _ = self.out_proj(C_gate * conv_out)
        return output


class Lfm2MoeDecoderLayer(nn.Module):
    """
    Decoder layer with attention/conv and dense MLP or MoE.

    - Layers 0 to num_dense_layers-1: use Lfm2MoeMLP (dense)
    - Layers num_dense_layers+: use Lfm2MoeSparseMoeBlock (MoE)
    """

    def __init__(
        self,
        config: Lfm2MoeConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_type = config.layer_types[layer_id]
        self.is_attention_layer = self.layer_type == "full_attention"

        self.operator_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

        # Attention or Conv
        if self.is_attention_layer:
            self.self_attn = Lfm2MoeAttention(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("self_attn", prefix),
            )
        else:
            self.conv = Lfm2MoeShortConv(
                config=config,
                layer_idx=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("conv", prefix),
            )

        # Dense MLP or MoE
        if layer_id < config.num_dense_layers:
            self.feed_forward = Lfm2MoeMLP(
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("feed_forward", prefix),
            )
        else:
            self.feed_forward = Lfm2MoeSparseMoeBlock(
                config=config,
                layer_idx=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("feed_forward", prefix),
            )

    def forward(
        self,
        layer_id: int,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not forward_batch.forward_mode.is_idle():
            residual = hidden_states
            normed = self.operator_norm(hidden_states)

            if self.is_attention_layer:
                hidden_states = self.self_attn(positions, normed, forward_batch)
            else:
                hidden_states = self.conv(normed, forward_batch)

            hidden_states = hidden_states + residual
            hidden_states = hidden_states + self.feed_forward(
                self.ffn_norm(hidden_states)
            )

        return hidden_states, residual


class Lfm2MoeModel(nn.Module):
    def __init__(
        self,
        config: Lfm2MoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        # Count attention layers for KV cache sizing
        self.num_attention_layers = sum(
            1 for lt in config.layer_types if lt == "full_attention"
        )

        def get_layer(idx: int, prefix: str, **kwargs):
            return Lfm2MoeDecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.embedding_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = (
            inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        )

        residual = None
        for i in range(len(self.layers)):
            hidden_states, residual = self.layers[i](
                layer_id=i,
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
            )

        return self.embedding_norm(hidden_states)


class Lfm2MoeForCausalLM(nn.Module):
    """LFM2-MoE for causal language modeling."""

    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: Lfm2MoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.pp_group = get_pp_group()
        assert self.pp_group.is_first_rank and self.pp_group.is_last_rank

        self.quant_config = quant_config
        self.model = Lfm2MoeModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            org_num_embeddings=config.vocab_size,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(config)
        self.num_attention_layers = self.model.num_attention_layers

    def get_num_kv_cache_layers(self) -> int:
        return self.num_attention_layers

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        hidden_states = self.model(input_ids, positions, forward_batch, inputs_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_mtp: bool = False
    ) -> Set[str]:
        """Load weights with FusedMoE expert format."""
        stacked_params_mapping = [
            # (param_name, weight_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            # Dense MLP w1/w3 -> gate_up_proj
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
        ]

        # FusedMoE expert params mapping
        # HF format: experts.{expert_id}.w{1,2,3}.weight
        # FusedMoE format: experts.w13_weight, experts.w2_weight
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        embed_tokens_weight = None

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if "embed_tokens.weight" in name:
                embed_tokens_weight = loaded_weight

            # Handle conv weight/bias naming: HF uses conv.conv, we use conv_weight/conv_bias
            if ".conv.conv.weight" in name:
                name = name.replace(".conv.conv.weight", ".conv.conv_weight")
                loaded_weight = loaded_weight.squeeze(1)  # (D, 1, K) -> (D, K)
            if ".conv.conv.bias" in name:
                name = name.replace(".conv.conv.bias", ".conv.conv_bias")

            # Handle dense MLP w2 -> down_proj
            if "feed_forward.w2" in name and "experts" not in name:
                name = name.replace("feed_forward.w2", "feed_forward.down_proj")

            # Handle stacked params (QKV, dense MLP gate_up)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Skip expert weights (handled below)
                if "experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    break
                if name not in params_dict:
                    break
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                # Handle MoE expert weights using FusedMoE format
                # HF format: model.layers.X.feed_forward.experts.Y.wZ.weight
                # FusedMoE format: model.layers.X.feed_forward.experts.w13_weight/w2_weight
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    shard_id,
                ) in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    # Build our parameter name
                    name = name.replace(weight_name, param_name)
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    loaded_params.add(name)
                    break
                else:
                    # Handle regular weights
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)

        # Handle tied lm_head weight
        if "lm_head.weight" not in loaded_params and "lm_head.weight" in params_dict:
            if embed_tokens_weight is not None:
                param = params_dict["lm_head.weight"]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, embed_tokens_weight)
                loaded_params.add("lm_head.weight")

        return loaded_params


EntryClass = [Lfm2MoeForCausalLM]
