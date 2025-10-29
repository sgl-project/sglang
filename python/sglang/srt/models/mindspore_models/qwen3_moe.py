# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
import logging
import math
import os
from functools import lru_cache
from typing import Iterable, Optional, Tuple, Type, Union

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.ops.operations as P
import numpy as np
import torch
from mindspore import Parameter, Tensor, dtype, jit, mint, mutable, nn, ops

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.srt.distributed.utils import divide
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.mindspore_models.layers import (
    BaseRotaryEmbedding,
    ColParallelLinear,
    FusedMoe,
    MLPColParallelLinear,
    MoeReplicatedLinear,
    MsNativeAttnBackend,
    QKVParallelLinear,
    RMSNorm,
    RowParallelLinear,
    SwiGLU,
    VocabParallelEmbedding,
    YaRNScalingRotaryEmbedding,
)
from sglang.srt.models.mindspore_models.mindspore_model_base import MindSporeModelBase
from sglang.srt.models.mindspore_models.utils import _get_tp_group_name, tensor_torch2ms

logger = logging.getLogger(__name__)

Qwen3Config = None


class Qwen3MoeMLP(nn.Cell):
    def __init__(self, config) -> None:
        super().__init__(config)

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.param_dtype = config.param_dtype

        self.gate_up_proj = MLPColParallelLinear(
            input_size=self.hidden_size,
            output_size=self.intermediate_size * 2,
            param_dtype=self.param_dtype,
            bias=False,
            output_sizes=[self.intermediate_size] * 2,
        )
        self.down_proj = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            param_dtype=config.param_dtype,
            bias=False,
        )
        self.act_fn = SwiGLU()

    def construct(self, x: Tensor) -> Tensor:
        x = self.gate_up_proj(x)
        x = self.act_fn(x)
        x = self.down_proj(x)
        return x


class Qwen3MoeSparseMoeBlock(nn.Cell):
    def __init__(self, config) -> None:
        super().__init__()

        self.ep_size = 1
        self.dp_size = 1
        self.tp_size = get_tensor_model_parallel_world_size()

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}"
            )

        self.experts = FusedMoe(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            param_dtype=config.param_dtype,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            tp_size=self.tp_size,
            ep_size=self.ep_size,
            dp_size=self.dp_size,
            optim_tp_ep_gating_perf=True,
        )

        self.gate = MoeReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            param_dtype=config.param_dtype,
            optim_tp_ep_gating_perf=self.experts.optim_tp_ep_gating_perf,
            expert_start_index=self.experts.expert_start_index,
            expert_end_index=self.experts.expert_end_index,
        )

    def construct(self, hidden_states: Tensor) -> Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        if self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
                final_hidden_states=final_hidden_states,
            )
        return final_hidden_states.view(orig_shape)


class Qwen3MoeAttention(nn.Cell):
    def __init__(self, config) -> None:
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        if hasattr(config, "head_dim"):
            self.head_dim = config.head_dim
        else:
            self.head_dim = config.hidden_size // self.num_heads
        self.q_size = self.head_dim * self.num_heads
        self.kv_size = self.head_dim * self.num_kv_heads
        self.scaling = float(self.head_dim**-0.5)
        self.rope_theta = int(config.rope_theta)
        self.param_dtype = config.param_dtype
        self.max_position = config.max_position_embeddings
        if config.rope_scaling is not None:
            self.rope_type = config.rope_scaling["rope_type"]
            self.rope_factor = config.rope_scaling["factor"]
            self.rope_max_position_embeddings = config.rope_scaling[
                "original_max_position_embeddings"
            ]
        else:
            self.rope_type = "default_rope"

        self.attn = MsNativeAttnBackend(
            config.num_attention_heads // self.tp_size,
            config.head_dim,
            config.num_key_value_heads // self.tp_size,
        )

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_kv_heads,
            bias=config.attention_bias,
            param_dtype=self.param_dtype,
        )
        self.q_norm = RMSNorm(
            norm_dim=config.head_dim,
            eps=config.rms_norm_eps,
            param_dtype=config.param_dtype,
        )
        self.k_norm = RMSNorm(
            norm_dim=config.head_dim,
            eps=config.rms_norm_eps,
            param_dtype=config.param_dtype,
        )
        self.o_proj = RowParallelLinear(
            input_size=self.q_size,
            output_size=self.hidden_size,
            param_dtype=self.param_dtype,
            bias=config.attention_bias,
        )
        self.rotary_emb = None
        if self.rope_type == "yarn":
            self.rotary_emb = YaRNScalingRotaryEmbedding(
                head_size=self.head_dim,
                rotary_dim=self.head_dim,
                max_position_embeddings=self.rope_max_position_embeddings,
                base=self.rope_theta,
                is_neox_style=True,
                scaling_factor=self.rope_factor,
                dtype=self.param_dtype,
            )
        else:
            self.rotary_emb = BaseRotaryEmbedding(
                head_size=self.head_dim,
                rotary_dim=self.head_dim,
                max_position_embeddings=self.max_position,
                base=self.rope_theta,
                dtype=self.param_dtype,
            )

    def construct(
        self,
        hidden_state: Tensor,
        positions: Tensor,
        batch_valid_length: Tensor,
        is_prefill: bool,
        layer_idx: int,
        attn_mask: Tensor,
        q_seq_lens: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        out_cache_loc: Tensor,
        block_tables: Tensor,
    ) -> Tensor:
        token_lens, hidden_dim = hidden_state.shape

        qkv = self.qkv_proj(hidden_state)
        q, k, v = qkv.split(
            [
                self.q_size // self.tp_size,
                self.kv_size // self.tp_size,
                self.kv_size // self.tp_size,
            ],
            dim=-1,
        )

        q = q.view(-1, self.head_dim).contiguous()
        k = k.view(-1, self.head_dim).contiguous()
        v = v.view(-1, self.kv_size // self.tp_size).contiguous()

        q = self.q_norm(q).view(-1, self.q_size // self.tp_size)
        k = self.k_norm(k).view(-1, self.kv_size // self.tp_size)

        q, k = self.rotary_emb(
            positions,
            q,
            k,
            batch_valid_length=batch_valid_length,
            is_prefill=is_prefill,
        )

        k = k.contiguous()

        key_out = self.attn(
            k,
            v,
            key_cache=key_cache,
            value_cache=value_cache,
            out_cache_loc=out_cache_loc,
        )
        q = ops.depend(q, key_out)

        if is_prefill:
            attn_output = self.attn.extend(
                q, k, v, attn_mask, None, None, None, q_seq_lens, batch_valid_length
            )
        else:
            attn_output = self.attn.decode(
                q,
                batch_valid_length,
                attn_mask,
                q_seq_lens,
                key_cache,
                value_cache,
                block_tables,
            )

        output = self.o_proj(attn_output).view(token_lens, -1)
        return output


class Qwen3MoeDecoderLayer(nn.Cell):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3MoeAttention(config=config)
        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        if (layer_idx not in mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3MoeSparseMoeBlock(config=config)
        else:
            self.mlp = Qwen3MoeMLP(config=config)
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
        hidden_state: Tensor,
        residual: Tensor,
        positions: Tensor,
        batch_valid_length: Tensor,
        is_prefill: bool,
        layer_idx: int,
        attn_mask: Tensor,
        q_seq_lens: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        out_cache_loc: Tensor,
        block_tables: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if residual is None:
            residual = hidden_state
            hidden_state = self.input_layernorm(hidden_state)
        else:
            hidden_state, residual = self.input_layernorm(hidden_state, residual)
        hidden_state = self.self_attn(
            hidden_state=hidden_state,
            positions=positions,
            batch_valid_length=batch_valid_length,
            is_prefill=is_prefill,
            layer_idx=layer_idx,
            attn_mask=attn_mask,
            q_seq_lens=q_seq_lens,
            key_cache=key_cache,
            value_cache=value_cache,
            out_cache_loc=out_cache_loc,
            block_tables=block_tables,
        )
        hidden_state, residual = self.post_attention_layernorm(hidden_state, residual)
        hidden_state = self.mlp(hidden_state)

        return hidden_state, residual


class Qwen3MoeModel(nn.Cell):
    r"""
    qwen3 moe model
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.param_dtype = config.param_dtype

        self.embed_tokens = VocabParallelEmbedding(config=config)

        self.layers = nn.CellList()

        for i in range(self.num_hidden_layers):
            layer = Qwen3MoeDecoderLayer(config=config, layer_idx=i)
            self.layers.append(layer)

        self.norm = RMSNorm(
            norm_dim=config.hidden_size,
            eps=config.rms_norm_eps,
            param_dtype=config.param_dtype,
        )

    # pylint: disable=W0613
    @jit
    def construct(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        batch_valid_length=None,
        is_prefill=True,
        q_seq_lens=None,
        key_cache=None,
        value_cache=None,
        out_cache_loc=None,
        block_tables=None,
    ):
        """
        Forward of qwen model.
        """
        hidden_state = self.embed_tokens(input_ids)
        residual = None
        for i in range(self.num_hidden_layers):
            layer = self.layers[i]
            hidden_state, residual = layer(
                hidden_state=hidden_state,
                residual=residual,
                positions=position_ids,
                batch_valid_length=batch_valid_length,
                is_prefill=is_prefill,
                layer_idx=i,
                attn_mask=attention_mask,
                q_seq_lens=q_seq_lens,
                key_cache=key_cache[i],
                value_cache=value_cache[i],
                out_cache_loc=out_cache_loc,
                block_tables=block_tables,
            )

        hidden_state, _ = self.norm(hidden_state, residual)

        return hidden_state


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


class Qwen3MoeForCausalLM(MindSporeModelBase):
    def __init__(
        self,
        config: Qwen3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.prev_prefill = False

        self.config = config
        setattr(self.config, "param_dtype", dtype.bfloat16)
        self.model = Qwen3MoeModel(self.config)

        self.lm_head = ColParallelLinear(
            input_size=self.config.hidden_size,
            output_size=self.config.vocab_size,
            param_dtype=self.config.param_dtype,
            bias=False,
        )
        self.all_gather = GatherLastDim()

        # for best performance of MindSpore for Qwen3
        os.environ["MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST"] = (
            "FlashAttentionScore,PagedAttention"
        )
        os.environ["MS_DISABLE_INTERNAL_KERNELS_LIST"] = "RmsNorm"

    def set_model_inputs(self, is_prefill):
        dyn_input_ids = Tensor(shape=[None], dtype=dtype.int32)
        dyn_position_ids = Tensor(shape=[None], dtype=dtype.int64)

        head_size = self.config.head_dim
        # use pa, if use ifa, the shape should (None, None, head_size)
        kv_cache_shape = (None, None, None, head_size)

        kv_cache_dtype = self.config.param_dtype

        num_layers = self.config.num_hidden_layers

        dyn_key_cache = Tensor(shape=kv_cache_shape, dtype=kv_cache_dtype)
        dyn_value_cache = Tensor(shape=kv_cache_shape, dtype=kv_cache_dtype)
        dyn_key_caches = mutable([dyn_key_cache for _ in range(num_layers)])
        dyn_value_caches = mutable([dyn_value_cache for _ in range(num_layers)])

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
        # dyn_intermediate_tensors = None
        # dyn_inputs_embeds = None
        self.model.set_inputs(
            input_ids=dyn_input_ids,
            position_ids=dyn_position_ids,
            attention_mask=dynamic_attention_mask,
            batch_valid_length=dyn_batch_valid_length,
            is_prefill=is_prefill,
            q_seq_lens=dyn_q_seq_lens,
            key_cache=dyn_key_caches,
            value_cache=dyn_value_caches,
            out_cache_loc=dyn_out_cache_loc,
            block_tables=dyn_block_tables,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        param_dict = self.parameters_dict()
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", "gate"),
            (".gate_up_proj", ".up_proj", "up"),
        ]

        # Params for weights, fp8 weight scale, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoe.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        for name, weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name in param_dict:
                    param = param_dict[name]
                    assert hasattr(param, "weight_load")
                    weight_load = getattr(param, "weight_load")
                    weight_load(param, weight, shard_id)
                    param.set_data(param.move_to("Ascend"))
                    break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    if (
                        name.endswith(",bias") or name.endswith("_bias")
                    ) and name not in param_dict:
                        continue

                    param = param_dict[name]
                    weight_load = param.weight_load
                    weight_load(
                        param,
                        weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    if name in param_dict:
                        param = param_dict[name]
                        if hasattr(param, "weight_load"):
                            weight_load = getattr(param, "weight_load")
                            weight_load(param, weight)
                            param.set_data(param.move_to("Ascend"))
                        else:
                            param.set_data(tensor_torch2ms(weight).move_to("Ascend"))
                        # Make sure the weight is loaded on device, so the kv cache calculation is correct.

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

        # TODO: In pure decode scenarios, cumsum and gather operations will be redundant .
        q_seq_lens = mint.cumsum(q_seq_lens, 0)
        hidden_state = mint.index_select(hidden_state, 0, q_seq_lens - 1)

        logits = self.lm_head(hidden_state)
        logits = self.all_gather(logits)
        logits = ops.cast(logits, dtype.float32)
        logits = mint.reshape(logits, (-1, logits.shape[-1]))
        return logits
