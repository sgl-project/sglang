import enum
import logging
from typing import Any, Dict, Iterable, Optional, Tuple, Set

import torch
from torch import nn
import torch.nn.functional as F


from sglang.srt.utils import (
    set_weight_attrs,
    make_layers,
    add_prefix,
)
from sglang.srt.distributed import (
    divide,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.layernorm import RMSNorm, GemmaRMSNorm
from sglang.srt.mem_cache.mamba_cache import (
    MambaCacheParams,
    MambaCacheManager,
)
from sglang.srt.model_loader.weight_utils import (
    sharded_weight_loader,
    default_weight_loader,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.configs.qwen3_hybrid_moe import Qwen3HybridMoeConfig
from sglang.srt.models.qwen2_moe import (
    Qwen2MoeMLP,
    Qwen2MoeSparseMoeBlock,
)
from sglang.srt.layers.attention.mamba.mamba import mamba_v2_sharded_weight_loader
from sglang.srt.layers.attention.mamba.mamba_mixer2 import MambaMixer2, extra_groups_for_head_shards
from einops import rearrange
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

from sglang.srt.layers.attention.mamba.ops.causal_conv1d import (
    causal_conv1d_fn, causal_conv1d_update)

from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule, 
    fused_recurrent_gated_delta_rule_update,
)

logger = logging.getLogger(__name__)

class Qwen3GatedDeltaNet(nn.Module):
    def __init__(
        self, 
        config: Qwen3HybridMoeConfig, 
        layer_id: int
    ) -> None:
        super().__init__()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_id = layer_id
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps

        # FIXME:
        self.time_step_limit = config.time_step_limit
        self.time_step_min = config.time_step_min
        self.time_step_max = config.time_step_max
        self.time_step_floor = config.time_step_floor

                # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            quant_config=None,
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)
        # projection of the input hidden states
        projection_size = self.key_dim * 2 + self.value_dim * 2 + self.num_v_heads * 2
        
        self.in_proj = ColumnParallelLinear(input_size=self.hidden_size,
                                            output_size=projection_size,
                                            bias=False)
        
        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)

        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight, {
                "weight_loader":
                mamba_v2_sharded_weight_loader([
                    query_key_settings,
                    query_key_settings,
                    value_settings,
                ], self.tp_size, self.tp_rank)
            })
        
        # selective projection used to make dt, B and C input dependant

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads // self.tp_size))

        A = torch.empty(divide(self.num_v_heads, self.tp_size), dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias,
                         {"weight_loader": sharded_weight_loader(0)})

        self.share_norm = config.share_norm
        self.norm_before_gate = config.norm_before_gate
        self.output_norm_size = config.output_norm_size
        if self.share_norm:  # original fla
            self.norm = RMSNormGated(
                self.head_v_dim,
                eps=self.layer_norm_epsilon,
                group_size=None,
                norm_before_gate=self.norm_before_gate,
                device=torch.cuda.current_device(),
                dtype=config.torch_dtype,
            )
        else:   # mamba2 or hgrn2 style
            norm_size_map = {
                "head": self.head_v_dim,
                "group": self.value_dim // self.num_k_heads,
            }
            self.norm_size = norm_size_map.get(self.output_norm_size, self.value_dim)
            self.norm = RMSNormGated(
                self.value_dim,
                eps=self.layer_norm_epsilon,
                group_size=self.norm_size,
                norm_before_gate=self.norm_before_gate,
                device=torch.cuda.current_device(),
                dtype=config.torch_dtype,
            )

        # self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        self.out_proj = RowParallelLinear(self.value_dim,
                                          self.hidden_size,
                                          bias=False,
                                          input_is_parallel=True)

    def fix_query_key_value_ordering(self, mixed_qkvzba, ):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvzba`.
        """
        new_tensor_shape = mixed_qkvzba.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            (
                self.head_k_dim + self.head_k_dim +
                (self.head_v_dim + self.head_v_dim + 2) * \
                self.num_v_heads // self.num_k_heads
            ),
        )
        mixed_qkvzba = mixed_qkvzba.view(*new_tensor_shape)

        split_arg_list = [
            self.head_k_dim, self.head_k_dim,
            (
                self.num_v_heads // self.num_k_heads * self.head_v_dim
            ),
            (
                self.num_v_heads // self.num_k_heads * self.head_v_dim
            ),
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads
        ]

        # [b, sq, ng, (hn + hn + np/ng * hn + np/ng + np/ng)]
        # --> [b, sq, ng, hn], [b, sq, ng, hn], [b, sq, ng, np/ng * hn], [b, sq, ng, np/ng * hn], [b, sq, ng, np/ng], [b, sq, ng, np/ng]
        (query, key, value, z, b, a) = torch.split(mixed_qkvzba, split_arg_list, dim=2)

        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.size(0), -1, self.head_v_dim)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b = b.reshape(b.size(0), self.num_v_heads // self.tp_size)
        a = a.reshape(a.size(0), self.num_v_heads // self.tp_size)

        return query, key, value, z, b, a

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        cache_params: Optional[MambaCacheParams] = None,
        sequence_idx: Optional[torch.Tensor] = None,
    ):
        has_initial_states = None
        if forward_batch.extend_prefix_lens is not None:
            has_initial_states = forward_batch.extend_prefix_lens > 0

        # Set up dimensions for reshapes later
        seq_len, _ = hidden_states.shape
        conv_state, recurrent_state = None, None

        # 检查 cache_params 是否为 None
        if cache_params is None:
            raise ValueError("cache_params cannot be None")

        projected_states, _ = self.in_proj(hidden_states)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(projected_states,)
        query, key, value = map(lambda x: rearrange(x, 'l p d -> l (p d)'), (query, key, value))
        mixed_qkv = torch.cat((query, key, value), dim=-1)
        # mixed_qkv = rearrange(mixed_qkv, "b l d -> b d l")

        # 2. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0),
                                               self.conv1d.weight.size(2))

        # TODO(cao1zhg):
        # has_initial_states, attn_metadata.query_start_loc
        has_prefill = forward_batch.forward_mode.is_prefill()

        if has_prefill:
            # - "cache_indices" updates the conv_state cache in positions
            #   pointed to by "mamba_cache_params.state_indices_tensor"
            mixed_qkv = causal_conv1d_fn(
                mixed_qkv.transpose(0, 1),
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=cache_params.conv_state,
                has_initial_state=has_initial_states,
                cache_indices=cache_params.state_indices_tensor,
                query_start_loc=forward_batch.extend_start_loc).transpose(
                    0, 1)[:seq_len]

            # TODO: Why is this needed?
            mixed_qkv = mixed_qkv.contiguous()
        else:
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                cache_params.conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=cache_params.state_indices_tensor)

        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim // self.tp_size, self.key_dim // self.tp_size, self.value_dim // self.tp_size,
            ],
            dim=-1,
        )
        query, key = map(lambda x: rearrange(x, 'l (h d) -> 1 l h d', d=self.head_k_dim), (query, key))
        value = rearrange(value, 'l (h d) -> 1 l h d', d=self.head_v_dim)

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            # NOTE: not use hf repeat kv, since shape is [batch, seqlen, head, headdim]
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        g, beta = map(lambda x: rearrange(x, 'l  d -> 1 l d'), (g, beta))
        
        if has_prefill:
            recurrent_state = cache_params.ssm_state[cache_params.state_indices_tensor]
            core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state=None, # NOTE: In my view, the initial state is not used in training and prefill stage.
                output_final_state=recurrent_state is not None,
                cu_seqlens=forward_batch.extend_start_loc,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )

            if recurrent_state is not None and cache_params is not None:
                last_recurrent_state = last_recurrent_state.to(torch.bfloat16, copy=False)
                cache_params.ssm_state[cache_params.state_indices_tensor] = last_recurrent_state
        else:
            indices = cache_params.state_indices_tensor
            mask = (indices == -1)
            indices[mask] = cache_params.ssm_state.shape[0] - 1

            core_attn_out = fused_recurrent_gated_delta_rule_update(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state_source=cache_params.ssm_state,
                initial_state_indices=indices,
                cu_seqlens=forward_batch.extend_start_loc,
                # head_first=False,
                use_qk_l2norm_in_kernel=True
            )

        if self.share_norm:
            z_shape_og = z.shape
            # reshape input data into 2D tensor
            core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
            z = z.reshape(-1, z.shape[-1])
            core_attn_out = self.norm(core_attn_out, z)
            core_attn_out = core_attn_out.reshape(z_shape_og)
            core_attn_out = rearrange(core_attn_out, '... h d -> ... (h d)')
        else:
            z = rearrange(z, "... h d -> ... (h d)")
            core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
            core_attn_out = self.norm(core_attn_out, z)

        output, _ = self.out_proj(core_attn_out)
        return output
        

class Qwen3HybridLinearDecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3HybridMoeConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = ""
    ) -> None:
        super().__init__()
        self.config = config
        self.linear_attn = Qwen3GatedDeltaNet(config, layer_id)

        # Note: Qwen/Qwen2-57B-A14B-Instruct does not have
        # `mlp_only_layers` in the config.
        mlp_only_layers = ([] if not hasattr(config, "mlp_only_layers") else
                           config.mlp_only_layers)
        if (layer_id not in mlp_only_layers) and (
                config.num_experts > 0 and
            (layer_id + 1) % config.decoder_sparse_step == 0):
            self.mlp = Qwen2MoeSparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config)
        else:
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
            )
        if getattr(config, "use_gemma_rms_norm", getattr(config, "apply_layernorm_1p", False)):
            logger.warning_once("Using Gemma RMSNorm for input normalization and post attn normalization.")
            self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.input_layernorm = RMSNorm(config.hidden_size,  eps=config.rms_norm_eps)
            self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        mamba_cache_params: MambaCacheParams,
        sequence_idx: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        forward_batch = kwargs.get("forward_batch", None)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.linear_attn(hidden_states, forward_batch, mamba_cache_params,
                                   sequence_idx)
        
        # Fully Connected
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(
            hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3HybridMixerDecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3HybridMoeConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = ""
    ) -> None:
        super().__init__()
        self.config = config
        self.mamba2 = MambaMixer2(
            layer_idx=layer_id,
            hidden_size= config.hidden_size,
            ssm_state_size = config.mamba2_state_dim,
            conv_kernel_size = config.mamba2_conv_dim,
            intermediate_size = config.mamba2_expand * config.hidden_size,
            use_conv_bias = config.mamba2_conv_bias,
            use_bias = config.mamba2_proj_bias,
            n_groups=config.mamba2_ngroups,
            num_heads=config.mamba2_nheads,
            head_dim=config.mamba2_head_dim,
            rms_norm_eps=1e-5, # config.rms_norm_eps,
            activation=config.hidden_act,
            chunk_size=config.mamba2_chunk_size,
            quant_config=quant_config)

        # Note: Qwen/Qwen2-57B-A14B-Instruct does not have
        # `mlp_only_layers` in the config.
        mlp_only_layers = ([] if not hasattr(config, "mlp_only_layers") else
                           config.mlp_only_layers)
        if (layer_id not in mlp_only_layers) and (
                config.num_experts > 0 and
            (layer_id + 1) % config.decoder_sparse_step == 0):
            self.mlp = Qwen2MoeSparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config)
        else:
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
            )
        if getattr(config, "use_gemma_rms_norm", getattr(config, "apply_layernorm_1p", False)):
            logger.warning_once("Using Gemma RMSNorm for input normalization and post attn normalization.")
            self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.input_layernorm = RMSNorm(config.hidden_size,  eps=config.rms_norm_eps)
            self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.layer_id = layer_id

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        mamba_cache_params: MambaCacheParams,
        sequence_idx: Optional[torch.Tensor] = None,
        forward_batch: Optional[ForwardBatch] = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.mamba2(hidden_states, mamba_cache_params,
                                   sequence_idx, forward_batch=forward_batch)
        
        # Fully Connected
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(
            hidden_states)
        
        hidden_states = self.mlp(hidden_states)
        
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3HybridAttentionDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3HybridMoeConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.position_embedding_type = getattr(config, "position_embedding_type", "none")
        self.hidden_size = config.hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % self.tp_size == 0
        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= self.tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % self.tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert self.tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.rope_theta = getattr(config, "rope_theta", 10000)
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.rope_scaling = getattr(config, "rope_scaling", None)

        self.attn_output_gate = getattr(config, "attn_output_gate", False)
        if self.attn_output_gate:
            logger.warning_once('using attn output gate!')

        if hasattr(config, "rotary_percent"):
            rotary_dim = self.head_dim * config.rotary_percent
        elif hasattr(config, "partial_rotary_factor"):
            rotary_dim = self.head_dim * config.partial_rotary_factor
        elif hasattr(config, "attn_rotary_emb"):
            rotary_dim = config.attn_rotary_emb  # for backward compatibility
        else:
            rotary_dim = self.head_dim  # default
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=rotary_dim,
            max_position=self.max_position_embeddings,
            rope_scaling=self.rope_scaling,
            base=self.rope_theta,
            is_neox_style=True,
            dtype=torch.get_default_dtype(),  # see impl of get_rope
        )

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
        )

        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim,
                                        config.hidden_size,
                                        bias=False,
                                        quant_config=quant_config)
        
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=f"{prefix}.attn",
        )

        # Note: Qwen/Qwen2-57B-A14B-Instruct does not have
        # `mlp_only_layers` in the config.

        mlp_only_layers = ([] if not hasattr(config, "mlp_only_layers") else
                           config.mlp_only_layers)
        if (layer_id not in mlp_only_layers) and (
                config.num_experts > 0 and
            (layer_id + 1) % config.decoder_sparse_step == 0):
            self.mlp = Qwen2MoeSparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config)
        else:
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
            )
        if getattr(config, "use_gemma_rms_norm", getattr(config, "apply_layernorm_1p", False)):
            logger.warning_once("Using Gemma RMSNorm for input normalization and post attn normalization.")
            self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.input_layernorm = RMSNorm(config.hidden_size,  eps=config.rms_norm_eps)
            self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        if self.config.use_qk_norm:
            if getattr(config, "use_gemma_rms_norm", getattr(config, "apply_layernorm_1p", False)):
                logger.warning_once("Using Gemma RMSNorm for Q normalization and K normalization.")
                self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
                self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            else:
                self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
                self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)


    def self_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        
        if self.attn_output_gate:
            q_gate, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        if self.config.use_qk_norm:
            q = self.q_norm.forward_native(q.view(-1, self.num_heads, self.head_dim))
            assert isinstance(q, torch.Tensor)
            q = q.view(-1, self.num_heads * self.head_dim)
            k = self.k_norm.forward_native(k.view(-1, self.num_kv_heads, self.head_dim))
            assert isinstance(k, torch.Tensor)
            k = k.view(-1, self.num_kv_heads * self.head_dim)

        if self.position_embedding_type == "rope":
            q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v, forward_batch)
        
        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate
        
        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attention(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(
            hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

ALL_DECODER_LAYER_TYPES = {
    "attention": Qwen3HybridAttentionDecoderLayer,
    "mamba": Qwen3HybridMixerDecoderLayer,
    "linear_attention": Qwen3HybridLinearDecoderLayer
}

class Qwen3HybridMoeModel(nn.Module):
    def __init__(
        self,
        config: Qwen3HybridMoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        def get_layer(idx: int, prefix: str):
            layer_class = ALL_DECODER_LAYER_TYPES[
                config.layers_block_type[idx]]
            return layer_class(
                config,
                idx,
                quant_config=quant_config,
                prefix=prefix,
            )

        # self.start_layer, self.end_layer, self.layers = make_layers(
        self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers")
        # self.make_empty_intermediate_tensors = (
        #     make_empty_intermediate_tensors_factory(
        #         ["hidden_states", "residual"], config.hidden_size))
        
        if getattr(config, "use_gemma_rms_norm", getattr(config, "apply_layernorm_1p", False)):
            logger.warning_once("Using Gemma RMSNorm for final normalization.")
            self.norm = GemmaRMSNorm(config.hidden_size,  eps=config.rms_norm_eps)
        else:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.infer_count = 0
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        mamba_cache_params: MambaCacheParams,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # pass a sequence index tensor, that is required for
        # proper continuous batching computation including
        # chunked prefill

        seq_idx = None
        if forward_batch.forward_mode.is_prefill() > 0:
            seq_idx = torch.zeros_like(input_ids, dtype=torch.int32)
            if forward_batch.extend_start_loc is not None:
                for i, (srt, end) in enumerate(
                        zip(
                            forward_batch.extend_start_loc.tolist(),
                            forward_batch.extend_start_loc[1:].tolist(),
                        )):
                    seq_idx[srt:end] = i
            seq_idx.unsqueeze_(0)

        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)
        
        residual = None
        num_attn = 0
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if isinstance(layer, Qwen3HybridAttentionDecoderLayer):
                num_attn += 1

            layer_mamba_cache_params = None
            if isinstance(layer, (Qwen3HybridMixerDecoderLayer, Qwen3HybridLinearDecoderLayer)):
                layer_mamba_cache_params = mamba_cache_params.at_layer_id(
                    i - num_attn)
            
            hidden_states = layer(
                layer_id=i,
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                mamba_cache_params=layer_mamba_cache_params,
                sequence_idx=seq_idx,
                forward_batch=forward_batch,
            )
            
        hidden_states = self.norm(hidden_states)
        
        if hidden_states.shape[0] <= 3:
            self.infer_count += 1
        return hidden_states

class HybridLayerType(enum.Enum):
    full_attention = "attention"
    swa_attention = "swa_attention"
    linear_attention = "linear_attention"
    mamba2 = "mamba"

class Qwen3HybridMoEForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: Qwen3HybridMoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.pp_group = get_pp_group()
        assert self.pp_group.is_first_rank and self.pp_group.is_last_rank
        self.quant_config = quant_config
        self.model = Qwen3HybridMoeModel(
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

        # Used to track and store by the Mamba cache between steps.
        self.mamba_cache: Optional[MambaCacheManager] = None
    
    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        self.init_mamba_cache_once()
        return self.mamba_cache.copy_inputs_before_cuda_graphs(
            input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        self.init_mamba_cache_once()
        return self.mamba_cache.get_seqlen_agnostic_capture_inputs(batch_size)

    def _get_linear_cache_shape(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        config = self.config
        world_size = get_tensor_model_parallel_world_size()
        conv_dim = (config.linear_key_head_dim * config.linear_num_key_heads * 2 + config.linear_value_head_dim * config.linear_num_value_heads)
        conv_state_shape = (
            divide(conv_dim, world_size),
            config.linear_conv_kernel_dim - 1,
        )
        
        temporal_state_shape = (
            divide(config.linear_num_value_heads, world_size),
            config.linear_key_head_dim, 
            config.linear_value_head_dim
        )
        return conv_state_shape, temporal_state_shape

    def _get_mamba_cache_shape(
            self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        world_size = get_tensor_model_parallel_world_size()
        hidden_size = self.config.hidden_size

        conv_state_shape, temporal_state_shape = None, None

        intermediate_size = self.config.mamba2_expand * hidden_size

        # if n_groups is not divisible by world_size, need to extend the shards
        # to ensure all groups needed by a head is sharded along with it
        n_groups = (self.config.mamba2_ngroups + extra_groups_for_head_shards(
            self.config.mamba2_ngroups, world_size))

        # - heads and n_groups are TP-ed
        conv_dim = (intermediate_size +
                    2 * n_groups * self.config.mamba2_state_dim)
        conv_state_shape = (
            divide(conv_dim, world_size),
            self.config.mamba2_conv_dim - 1,
        )

        # These are not TP-ed as they depend on A, dt_bias, D
        # - they are typically small
        #   e.g., (h_heads, d_head, d_state) = (128, 64, 128)
        temporal_state_shape = (
            divide(self.config.mamba2_nheads, world_size),
            self.config.mamba2_head_dim,
            self.config.mamba2_state_dim,
        )
        return conv_state_shape, temporal_state_shape
    
    def init_mamba_cache_once(self):
        if self.mamba_cache is not None:
            return
        layers_block_type_value = self.config.layers_block_type
        if self.config.hybrid_linear_attention:
            num_mamba_layers = sum(type_value == HybridLayerType.linear_attention.value for type_value in layers_block_type_value)
            conv_state_shape, temporal_state_shape = self._get_linear_cache_shape()
        else:
            num_mamba_layers = sum(type_value == HybridLayerType.mamba2.value for type_value in layers_block_type_value)
            conv_state_shape, temporal_state_shape = self._get_mamba_cache_shape()
        self.mamba_cache = MambaCacheManager(
            self.lm_head.weight.dtype, num_mamba_layers,
            conv_state_shape, temporal_state_shape)
        logger.info(f"{self.lm_head.weight.dtype=}")

    def prepare_extend_start_loc(self, forward_batch: ForwardBatch):
        # extend_start_loc shape is batch_size, while hybrid attention need batch_size + 1
        # ugly code, but works :)
        input_ids = forward_batch.input_ids
        if forward_batch.extend_start_loc is not None:
            extend_start_loc = torch.cat([forward_batch.extend_start_loc,
                                                            torch.tensor([input_ids.shape[0]], dtype=torch.int32,
                                                                        device=input_ids.device)],
                                                            dim=0)
        else:
            extend_start_loc = torch.arange(
                input_ids.shape[0] + 1, dtype=torch.int32, device=input_ids.device)
        return extend_start_loc

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        self.init_mamba_cache_once()

        mamba_cache_params = self.mamba_cache.current_run_tensors(**kwargs)

        hidden_states = self.model(input_ids, positions, forward_batch, mamba_cache_params, inputs_embeds)
        
        return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = get_moe_impl_class().make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts)

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if ".self_attn." in name:
                name = name.replace(".self_attn", "")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                
                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                # if is_pp_missing_parameter(name, self):
                #     continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    # if is_pp_missing_parameter(name, self):
                    #     continue
                    # Skip loading extra bias for GPTQ models.
                    if ((name.endswith(".bias") or name.endswith("_bias"))
                            and name not in params_dict):
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader")
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    # if is_pp_missing_parameter(name, self):
                    #     continue

                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

EntryClass = Qwen3HybridMoEForCausalLM