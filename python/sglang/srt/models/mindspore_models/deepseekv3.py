# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
import math
import os
from functools import wraps
from typing import Iterable, List, Optional, Tuple

import mindspore.common.dtype as mstype
import torch
from mindspore import Parameter, Tensor, dtype, jit, mint, mutable, nn, ops

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.distributed.utils import divide
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.mindspore_models.layers import (
    BaseRotaryEmbedding,
    ColParallelLinear,
    DeepseekScalingRotaryEmbedding,
    FusedMoe,
    MLPColParallelLinear,
    MoeReplicatedLinear,
    MsNativeAttnBackend,
    RMSNorm,
    RowParallelLinear,
    SwiGLU,
    VocabParallelEmbedding,
    yarn_get_mscale,
)
from sglang.srt.models.mindspore_models.mindspore_model_base import MindSporeModelBase
from sglang.srt.models.mindspore_models.utils import (
    _get_tp_group_name,
    set_weight_attrs,
    tensor_torch2ms,
)


def transpose_rope_weight(weight, start_dim):
    w1 = weight[..., -start_dim::2, :]
    w2 = weight[..., -start_dim + 1 :: 2, :]
    weight[..., -start_dim:, :] = torch.cat((w1, w2), dim=-2)
    return weight


def reorder_qkv_rope_proj_weight(weight_loader):
    @wraps(weight_loader)
    def wrapper(*args, **kwargs):
        param = args[-2]
        loaded_weight = args[1][:]
        reorder_params = getattr(param, "reorder_params", {})
        if not reorder_params:
            raise ValueError(
                f"reorder_params of param [{param.name}] should not be empty."
            )
        qk_rope_head_dim = reorder_params["qk_rope_head_dim"]
        if "kv_head_dim" in reorder_params:
            kv_head_dim = reorder_params["kv_head_dim"]
            loaded_weight = loaded_weight.reshape(kv_head_dim, -1)
        else:
            num_heads = reorder_params["num_heads"]
            q_head_dim = reorder_params["q_head_dim"]
            loaded_weight = loaded_weight.reshape(num_heads, q_head_dim, -1)

        loaded_weight = transpose_rope_weight(loaded_weight, qk_rope_head_dim)
        if "num_heads" in reorder_params:
            loaded_weight = loaded_weight.reshape(num_heads * q_head_dim, -1)
        weight_loader(param, loaded_weight, **kwargs)

    return wrapper


class DeepseekV3MLP(nn.Cell):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        param_dtype,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.param_dtype = param_dtype

        self.gate_up_proj = MLPColParallelLinear(
            input_size=self.hidden_size,
            output_size=self.intermediate_size * 2,
            param_dtype=self.param_dtype,
            bias=False,
            output_sizes=[self.intermediate_size] * 2,
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            param_dtype=param_dtype,
            bias=False,
            reduce_results=reduce_results,
        )
        self.act_fn = SwiGLU()

    def construct(self, x: Tensor) -> Tensor:
        x = self.gate_up_proj(x)
        x = self.act_fn(x)
        x = self.down_proj(x)
        return x


class DeepseekV3MOE(nn.Cell):
    def __init__(self, config) -> None:
        super().__init__()

        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        self.num_redundant_experts = 1

        self.gate = MoeReplicatedLinear(
            input_size=config.hidden_size,
            output_size=config.n_routed_experts,
            bias=False,
            param_dtype=config.param_dtype,
        )
        self.gate.e_score_correction_bias = Parameter(
            mint.zeros((self.n_routed_experts), dtype=mstype.float32),
            requires_grad=False,
            parallel_optimizer=False,
        )

        self.tp_size = get_tensor_model_parallel_world_size()
        self.ep_size = 1
        self.dp_size = 1

        self.experts = FusedMoe(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            param_dtype=config.param_dtype,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            tp_size=self.tp_size,
            ep_size=self.ep_size,
            dp_size=self.dp_size,
            scoring_func=config.scoring_func,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            num_redundant_experts=self.num_redundant_experts,
        )

        if config.n_shared_experts:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                param_dtype=config.param_dtype,
                reduce_results=self.experts.must_reduce_shared_expert_outputs(),
            )

    def construct(self, hidden_states: Tensor):
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)
        shared_output = (
            self.shared_experts(hidden_states) if self.n_shared_experts else None
        )

        router_logits = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
        if self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
                final_hidden_states=final_hidden_states,
            )
        return final_hidden_states.view(orig_shape)


class DeepseekV3AttentionMLA(nn.Cell):
    def __init__(self, config) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.local_num_heads = self.total_num_heads // self.tp_size
        self.head_dim = (
            config.head_dim
            if hasattr(config, "head_dim")
            else self.hidden_size // self.total_num_heads
        )

        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.kv_head_dim = self.kv_lora_rank + self.qk_rope_head_dim

        self.scaling = 1.0 / math.sqrt(self.q_head_dim)
        if hasattr(config, "rope_scaling"):
            self.rope_scaling = config.rope_scaling
            mscale_all_dim = self.rope_scaling.get("mscale_all_dim", 1)
            scaling_factor = self.rope_scaling.get("factor", 1)
            mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
            self.scaling = self.scaling * mscale * mscale

            extra_kwargs = {
                k: v
                for k, v in self.rope_scaling.items()
                if k
                in (
                    "extrapolation_factor",
                    "attn_factor",
                    "beta_fast",
                    "beta_slow",
                    "mscale",
                    "mscale_all_dim",
                )
            }
            self.rotary_emb = DeepseekScalingRotaryEmbedding(
                head_size=self.qk_rope_head_dim,
                rotary_dim=self.qk_rope_head_dim,
                max_position_embeddings=config.rope_scaling.get(
                    "original_max_position_embeddings", 4096
                ),
                base=config.rope_theta,
                scaling_factor=scaling_factor,
                dtype=config.param_dtype,
                **extra_kwargs,
            )
        else:
            self.rotary_emb = BaseRotaryEmbedding(
                head_size=self.qk_rope_head_dim,
                rotary_dim=self.qk_rope_head_dim,
                max_position_embeddings=config.rope_scaling.get(
                    "original_max_position_embeddings", 4096
                ),
                base=config.rope_theta,
                dtype=config.param_dtype,
            )

        self.attn = MsNativeAttnBackend(
            n_heads=self.total_num_heads // self.tp_size,
            head_dim=self.head_dim,
            n_kv_heads=1,
            scale_value=self.scaling,
            mla_v_dim=self.kv_lora_rank,
        )

        self.q_a_proj = MoeReplicatedLinear(
            input_size=self.hidden_size,
            output_size=self.q_lora_rank,
            bias=False,
            param_dtype=config.param_dtype,
        )

        self.q_a_layernorm = RMSNorm(
            norm_dim=self.q_lora_rank,
            eps=config.rms_norm_eps,
            param_dtype=config.param_dtype,
        )

        # transpose weight avoiding transposition operation
        self.q_b_proj = ColParallelLinear(
            input_size=self.q_lora_rank,
            output_size=self.total_num_heads * self.q_head_dim,
            param_dtype=config.param_dtype,
            bias=False,
        )
        set_weight_attrs(
            self.q_b_proj.weight,
            {
                "reorder_params": {
                    "qk_rope_head_dim": self.qk_rope_head_dim,
                    "num_heads": self.total_num_heads,
                    "q_head_dim": self.q_head_dim,
                },
                "weight_load": reorder_qkv_rope_proj_weight(self.q_b_proj.weight_load),
            },
        )

        # transpose weight avoiding transposition operation
        self.kv_a_proj_with_mqa = MoeReplicatedLinear(
            input_size=self.hidden_size,
            output_size=self.kv_head_dim,
            param_dtype=config.param_dtype,
            bias=False,
        )
        set_weight_attrs(
            self.kv_a_proj_with_mqa.weight,
            {
                "reorder_params": {
                    "qk_rope_head_dim": self.qk_rope_head_dim,
                    "kv_head_dim": self.kv_head_dim,
                },
                "weight_load": reorder_qkv_rope_proj_weight(
                    self.kv_a_proj_with_mqa.weight_load
                ),
            },
        )

        self.kv_a_layernorm = RMSNorm(
            norm_dim=self.kv_lora_rank,
            eps=config.rms_norm_eps,
            param_dtype=config.param_dtype,
        )

        self.kv_b_proj_k = ColParallelLinear(
            input_size=self.kv_lora_rank,
            output_size=self.total_num_heads * self.qk_nope_head_dim,
            param_dtype=config.param_dtype,
            bias=False,
        )

        self.kv_b_proj_v = ColParallelLinear(
            input_size=self.kv_lora_rank,
            output_size=self.total_num_heads * self.v_head_dim,
            param_dtype=config.param_dtype,
            bias=False,
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.v_head_dim,
            output_size=self.hidden_size,
            param_dtype=config.param_dtype,
            bias=False,
        )

        self.tile_kv = ops.Tile()
        self.dim_slice_4d = ops.Slice()
        self.kpe_concat = ops.Concat(1)
        self.pe_concat = ops.Concat(2)
        self.qabsorb_k_matmul = ops.BatchMatMul()
        self.outabsorb_v_matmul = ops.BatchMatMul(transpose_b=True)

    def forward_absorb_prepare(
        self,
        hidden_states,
        positions: Tensor,
        key_cache: Tensor,
        out_cache_loc: Tensor,
        batch_valid_length: Tensor,
        is_prefill: bool,
    ):
        # calculate q
        q = self.q_a_proj(hidden_states)
        norm_q = self.q_a_layernorm(q)
        q = self.q_b_proj(norm_q)
        q = q.view((-1, self.local_num_heads, self.q_head_dim))

        # calculate k(v)
        latent_kv_all = self.kv_a_proj_with_mqa(hidden_states)
        latent_kv, k_pe = mint.split(
            latent_kv_all, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        i_kv = self.kv_a_layernorm(latent_kv)

        # q， k rope
        q_nope, q_pe = mint.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_pe = q_pe.view((-1, self.local_num_heads * self.qk_rope_head_dim))
        q_pe, k_pe = self.rotary_emb(
            positions, q_pe, k_pe, batch_valid_length, is_prefill
        )
        q_pe = q_pe.view((-1, self.local_num_heads, self.qk_rope_head_dim))

        # k reshape_and_cache
        key_states_cache = mint.cat((i_kv, k_pe), 1)
        self.attn(key_states_cache, None, key_cache, None, out_cache_loc)

        return q_nope, q_pe, k_pe, i_kv

    def construct(
        self,
        hidden_states: Tensor,
        positions: Tensor,
        key_cache: Tensor,
        is_prefill: bool,
        out_cache_loc: Tensor,
        attn_mask: Tensor,
        batch_valid_length: Tensor,
        q_seq_lens: Tensor,
        block_tables: Tensor,
    ) -> Tensor:
        q_nope, q_pe, k_pe, i_kv = self.forward_absorb_prepare(
            hidden_states=hidden_states,
            positions=positions,
            key_cache=key_cache,
            out_cache_loc=out_cache_loc,
            batch_valid_length=batch_valid_length,
            is_prefill=is_prefill,
        )

        if is_prefill:
            # q
            query_states = mint.cat((q_nope, q_pe), 2)

            # k
            k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim)
            k_pe = self.tile_kv(k_pe, (1, self.local_num_heads, 1))
            o_k_nope = self.kv_b_proj_k(i_kv)
            k_nope = o_k_nope.view(-1, self.local_num_heads, self.qk_nope_head_dim)
            key_states = self.pe_concat((k_nope, k_pe))

            # v
            o_v = self.kv_b_proj_v(i_kv)
            value_states = o_v.view(-1, self.local_num_heads, self.v_head_dim)
            # It's not necessary. Just fa is not support k != v. V just (t, head, 128)
            value_states = self.pe_concat((value_states, k_pe))

            # attention
            query_states = query_states.view(-1, self.local_num_heads * self.q_head_dim)
            key_states = key_states.view(-1, self.local_num_heads * self.q_head_dim)
            value_states = value_states.view(-1, self.local_num_heads * self.q_head_dim)
            context_layer = self.attn.extend(
                query=query_states,
                key=key_states,
                value=value_states,
                attn_mask=attn_mask,
                q_seq_lens=q_seq_lens,
                batch_valid_length=batch_valid_length,
            )
            context_layer = context_layer.view(
                -1, self.local_num_heads, self.q_head_dim
            )
            context_layer = self.dim_slice_4d(
                context_layer, (0, 0, 0), (-1, self.local_num_heads, self.v_head_dim)
            )
        else:
            # q,  k_absorb
            q_absorb = self.kv_b_proj_k.weight.view(
                self.local_num_heads, self.qk_nope_head_dim, self.kv_lora_rank
            )
            q_nope = self.qabsorb_k_matmul(
                q_nope.transpose(1, 0, 2), q_absorb
            ).transpose(1, 0, 2)
            query_states = self.pe_concat((q_nope, q_pe))
            query_states = query_states.view(
                -1, self.local_num_heads * self.kv_head_dim
            )

            # attention
            context_layer = self.attn.decode(
                query=query_states,
                batch_valid_length=batch_valid_length,
                attn_mask=attn_mask,
                q_seq_lens=q_seq_lens,
                key_cache=key_cache,
                value_cache=key_cache,
                block_tables=block_tables,
            )
            context_layer = context_layer.view(
                -1, self.local_num_heads, self.kv_lora_rank
            )

            # out, v_absorb
            out_absorb = self.kv_b_proj_v.weight.view(
                self.local_num_heads, self.v_head_dim, self.kv_lora_rank
            )
            context_layer = self.outabsorb_v_matmul(
                context_layer.transpose(1, 0, 2), out_absorb
            ).transpose(1, 0, 2)

        attn_out = context_layer.view(-1, self.local_num_heads * self.v_head_dim)
        output = self.o_proj(attn_out)
        return output


class DeepseekV3DecoderLayer(nn.Cell):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = DeepseekV3AttentionMLA(config)

        if (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            self.mlp = DeepseekV3MOE(config)
        else:
            self.mlp = DeepseekV3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                param_dtype=config.param_dtype,
            )
        self.input_layernorm = RMSNorm(
            norm_dim=config.hidden_size,
            eps=config.rms_norm_eps,
            param_dtype=config.param_dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            norm_dim=config.hidden_size,
            eps=config.rms_norm_eps,
            param_dtype=config.param_dtype,
        )

    def construct(
        self,
        positions: Tensor,
        hidden_states: Tensor,
        key_cache: Tensor,
        is_prefill: bool,
        out_cache_loc: Tensor,
        attn_mask: Tensor,
        batch_valid_length: Tensor,
        q_seq_lens: Tensor,
        block_tables: Tensor,
        residual: Optional[Tensor],
    ) -> Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            key_cache=key_cache,
            is_prefill=is_prefill,
            out_cache_loc=out_cache_loc,
            attn_mask=attn_mask,
            batch_valid_length=batch_valid_length,
            q_seq_lens=q_seq_lens,
            block_tables=block_tables,
        )
        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class DeepseekV3Model(nn.Cell):
    def __init__(self, config) -> None:
        super().__init__()
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = VocabParallelEmbedding(config=config)

        self.layers = nn.CellList()
        for layer_idx in range(self.num_hidden_layers):
            layer = DeepseekV3DecoderLayer(config, layer_idx)
            self.layers.append(layer)

        self.norm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, param_dtype=config.param_dtype
        )

    @jit
    def construct(
        self,
        input_ids: Tensor,
        position_ids: Tensor = None,
        key_cache: List[Tensor] = None,
        is_prefill: bool = True,
        out_cache_loc: Tensor = None,
        attention_mask: Tensor = None,
        batch_valid_length: Tensor = None,
        q_seq_lens: Tensor = None,
        block_tables: Tensor = None,
    ):
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer_idx in range(self.num_hidden_layers):
            layer: DeepseekV3DecoderLayer = self.layers[layer_idx]
            hidden_states, residual = layer(
                positions=position_ids,
                hidden_states=hidden_states,
                key_cache=key_cache[layer_idx],
                is_prefill=is_prefill,
                out_cache_loc=out_cache_loc,
                attn_mask=attention_mask,
                batch_valid_length=batch_valid_length,
                q_seq_lens=q_seq_lens,
                block_tables=block_tables,
                residual=residual,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class GatherLastDim(nn.Cell):
    def __init__(self):
        super().__init__()
        tp_group_name = _get_tp_group_name()
        self.all_gather = ops.AllGather(group=tp_group_name)
        self.world_size = get_tensor_model_parallel_world_size()
        self.split = ops.Split(axis=0, output_num=self.world_size)

    def construct(self, input: Tensor) -> Tensor:
        output = self.all_gather(input)
        tensor_list = self.split(output)
        output = mint.cat(tensor_list, dim=-1)
        return output


class DeepseekV3ForCausalLM(MindSporeModelBase):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        setattr(self.config, "param_dtype", dtype.bfloat16)
        self.prev_prefill = False

        self.model = DeepseekV3Model(self.config)

        self.lm_head = ColParallelLinear(
            input_size=self.config.hidden_size,
            output_size=self.config.vocab_size,
            param_dtype=self.config.param_dtype,
            bias=False,
        )

        self.all_gather = GatherLastDim()

        os.environ["MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST"] = (
            "FlashAttentionScore,PagedAttention"
        )
        os.environ["MS_DISABLE_INTERNAL_KERNELS_LIST"] = "RmsNorm"

    def set_model_inputs(self, is_prefill):
        dyn_input_ids = Tensor(shape=[None], dtype=dtype.int32)
        dyn_positions = Tensor(shape=[None], dtype=dtype.int64)

        head_size = self.config.kv_lora_rank + self.config.qk_rope_head_dim
        # use pa, if use ifa, the shape should (None, None, head_size)
        kv_cache_shape = (None, None, None, head_size)

        kv_cache_dtype = self.config.param_dtype

        num_layers = self.config.num_hidden_layers

        dyn_key_cache = Tensor(shape=kv_cache_shape, dtype=kv_cache_dtype)
        dyn_key_caches = mutable([dyn_key_cache for _ in range(num_layers)])

        dyn_out_cache_loc = Tensor(
            shape=[
                None,
            ],
            dtype=dtype.int32,
        )
        dynamic_attention_mask = Tensor(
            shape=[None, None], dtype=self.config.param_dtype
        )
        dyn_batch_valid_length = Tensor(
            shape=[
                None,
            ],
            dtype=dtype.int32,
        )
        dyn_q_seq_lens = Tensor(
            shape=[
                None,
            ],
            dtype=dtype.int32,
        )
        dyn_block_tables = Tensor(shape=[None, None], dtype=dtype.int32)
        self.model.set_inputs(
            input_ids=dyn_input_ids,
            position_ids=dyn_positions,
            attention_mask=dynamic_attention_mask,
            batch_valid_length=dyn_batch_valid_length,
            is_prefill=is_prefill,
            q_seq_lens=dyn_q_seq_lens,
            key_cache=dyn_key_caches,
            out_cache_loc=dyn_out_cache_loc,
            block_tables=dyn_block_tables,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = self.parameters_dict()
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", "gate"),
            ("gate_up_proj", "up_proj", "up"),
        ]

        expert_params_mapping = FusedMoe.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_up_proj_name="up_proj",
            ckpt_down_proj_name="down_proj",
            num_experts=self.config.n_routed_experts,
        )

        loaded_params: set[str] = set()

        for k in params_dict.keys():
            print(f"params_dict:{k}")

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if "kv_b_proj" in name and name not in params_dict:
                k_name = name.replace("kv_b_proj", "kv_b_proj_k")
                v_name = name.replace("kv_b_proj", "kv_b_proj_v")

                loaded_weight = loaded_weight.reshape(
                    self.config.num_attention_heads,
                    self.config.qk_nope_head_dim + self.config.v_head_dim,
                    -1,
                )
                k_weight = loaded_weight[:, : self.config.qk_nope_head_dim, :].reshape(
                    self.config.num_attention_heads * self.config.qk_nope_head_dim, -1
                )
                v_weight = loaded_weight[:, self.config.qk_nope_head_dim :, :].reshape(
                    self.config.num_attention_heads * self.config.qk_nope_head_dim, -1
                )

                if k_name not in params_dict.keys() or v_name not in params_dict.keys():
                    continue

                k_param = params_dict[k_name]
                v_param = params_dict[v_name]

                k_param.weight_load(k_param, k_weight)
                v_param.weight_load(v_param, v_weight)
                loaded_params.add(k_name)
                loaded_params.add(v_name)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if ("mlp.experts" in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                param.weight_load(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                is_expert_weight = False
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    shard_id,
                ) in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    is_expert_weight = True

                    name_mapped = name.replace(weight_name, param_name)

                    if name_mapped not in params_dict.keys():
                        continue

                    param = params_dict[name_mapped]
                    param.weight_load(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )

                    loaded_params.add(name_mapped)
                    break
                else:
                    if is_expert_weight:
                        continue

                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    if name is None:
                        continue

                    if name not in params_dict.keys():
                        print(f"name_mapped: {name} not found in params_dict")
                        continue

                    param = params_dict[name]
                    if hasattr(param, "weight_load"):
                        weight_load = getattr(param, "weight_load")
                        weight_load(param, loaded_weight)
                        param.set_data(param.move_to("Ascend"))
                    else:
                        param.set_data(tensor_torch2ms(loaded_weight).move_to("Ascend"))
                    loaded_params.add(name)
        return loaded_params

    def construct(self, **model_inputs) -> Tensor:
        q_seq_lens = model_inputs["q_seq_lens"]
        is_prefill = model_inputs["is_prefill"]

        if self.prev_prefill != is_prefill:
            self.set_model_inputs(is_prefill)
        self.prev_prefill = is_prefill

        if is_prefill:
            self.model.phase = "prefill"
        else:
            self.model.phase = "increment"

        hidden_state = self.model(**model_inputs)

        # TODO: In pure decode scenarios, cumsum and gather operations will be redundant.
        q_seq_lens = mint.cumsum(q_seq_lens, 0)
        hidden_state = mint.index_select(hidden_state, 0, q_seq_lens - 1)

        logits = self.lm_head(hidden_state)
        logits = self.all_gather(logits)
        logits = ops.cast(logits, dtype.float32)
        logits = mint.reshape(logits, (-1, logits.shape[-1]))
        return logits
