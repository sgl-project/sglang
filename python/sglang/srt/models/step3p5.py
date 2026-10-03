from typing import Any, Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.dp_attention import (
    is_dp_attention_enabled,
)
from sglang.srt.layers.layer_boundary import (
    declare_attn,
    declare_ffn,
    make_stages,
)
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
    reduce_moe_output,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopK
from sglang.srt.layers.moe.utils import (
    RoutingMethodType,
    filter_moe_weight_param_global_expert,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.runtime_context import get_exec, get_parallel, get_stream
from sglang.srt.utils import (
    add_prefix,
    is_cuda,
    make_pp_layers,
)

Step3p5Config = None

_is_cuda = is_cuda()


class Step3p5MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        swiglu_limit: Optional[float] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            reduce_results=reduce_results,
        )
        self.act_fn = SiluAndMul()
        self.limit = swiglu_limit

    def forward(self, x):
        if self.limit is not None:
            gate_up, _ = self.gate_up_proj(x)
            gate, up = gate_up.chunk(2, dim=-1)
            gate = F.silu(gate)
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            output, _ = self.down_proj(gate * up)
        else:
            gate_up, _ = self.gate_up_proj(x)
            x = self.act_fn(gate_up)
            output, _ = self.down_proj(x)
        return output


class Step3p5MoEMLP(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_parallel().tp_size
        self.layer_id = layer_id

        self.need_fp32_gate = config.need_fp32_gate
        self.routed_scaling_factor = config.moe_router_scaling_factor
        self.use_moe_router_bias = config.use_moe_router_bias
        if self.use_moe_router_bias:
            self.router_bias = nn.Parameter(
                torch.zeros(config.moe_num_experts, dtype=torch.float32),
                requires_grad=False,
            )

        if self.tp_size > config.moe_num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.moe_num_experts}."
            )

        self.limit = config.swiglu_limits[layer_id]
        self.limit = self.limit if self.limit > 0 else None

        self.topk = TopK(
            top_k=config.moe_top_k,
            renormalize=True,
            use_grouped_topk=False,
            scoring_func="sigmoid",
            correction_bias=self.router_bias,
            apply_routed_scaling_factor_on_output=False,
            layer_id=layer_id,
        )

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.moe_num_experts
            + get_exec().moe.ep_num_redundant_experts,
            top_k=config.moe_top_k,
            layer_id=layer_id,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            routing_method_type=RoutingMethodType.Renormalize,
            gemm1_clamp_limit=self.limit,
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.moe_num_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )

        if get_moe_a2a_backend().is_deepep():
            # TODO: we will support tp < ep in the future
            self.ep_size = get_parallel().moe_ep_size
            self.moe_num_experts = (
                config.moe_num_experts + get_exec().moe.ep_num_redundant_experts
            )
            self.top_k = config.moe_top_k

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        shared_output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if (
            not get_moe_a2a_backend().is_deepep()
            and not get_moe_a2a_backend().is_ascend_fuseep()
        ):
            return self.forward_normal(hidden_states, shared_output)
        else:
            return self.forward_deepep(hidden_states, forward_batch, shared_output)

    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
            and filter_moe_weight_param_global_expert(
                name, x, self.experts.num_local_experts
            )
        ]

    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        shared_output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (num_tokens, n_experts)
        if self.need_fp32_gate:
            router_logits = torch.matmul(
                hidden_states.to(torch.float32), self.gate.weight.t().to(torch.float32)
            )
        else:
            # router_logits: (batch * sequence_length, n_experts)
            router_logits, _ = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        if hasattr(topk_output, "to_standard"):
            topk_output = topk_output.to_standard(layer_id=self.layer_id)
        if self.routed_scaling_factor != 1.0:
            topk_output = StandardTopKOutput(
                topk_weights=topk_output.topk_weights * self.routed_scaling_factor,
                topk_ids=topk_output.topk_ids,
                router_logits=topk_output.router_logits,
            )
        final_hidden_states = self.experts(hidden_states, topk_output)
        if shared_output is not None:
            final_hidden_states += shared_output
        final_hidden_states = reduce_moe_output(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)

    def forward_deepep(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        shared_output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hidden_states.shape[0] > 0:
            # router_logits: (num_tokens, n_experts)
            router_logits, _ = self.gate(hidden_states)
            topk_output = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=forward_batch.moe_num_token_non_padded(),
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                ),
            )
        else:
            topk_output = self.topk.empty_topk_output(hidden_states.device)
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )
        if shared_output is not None:
            shared_output.add_(final_hidden_states)
            final_hidden_states = shared_output
        return final_hidden_states


class Step3p5Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 1000000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        head_dim: Optional[int] = None,
        max_position_embeddings: int = 32768,
        quant_config: Optional[QuantizationConfig] = None,
        rms_norm_eps: float = None,
        partial_rotary_factor: float = 1.0,
        use_head_wise_attn_gate: bool = False,
        sliding_window_size: int = -1,  # if is -1 ,normal attention,else ,window attention
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        attn_tp_rank = get_parallel().attn_tp_rank
        attn_tp_size = get_parallel().attn_tp_size

        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=rms_norm_eps)

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        self.use_head_wise_attn_gate = use_head_wise_attn_gate
        if self.use_head_wise_attn_gate:
            self.g_proj = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads,
                bias=False,
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
                prefix=add_prefix("g_proj", prefix),
            )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            partial_rotary_factor=partial_rotary_factor,
            is_neox_style=True,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            sliding_window_size=sliding_window_size,  # if is -1 ,normal attention,else ,window attention
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )
        self.alt_stream = alt_stream

    def forward_prepare_native(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_shape, k_shape = q.shape, k.shape
        q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(q_shape)
        k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(k_shape)
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:

        q, k, v = self.forward_prepare_native(
            positions=positions,
            hidden_states=hidden_states,
        )
        if self.use_head_wise_attn_gate:
            gate_states, _ = self.g_proj(hidden_states)
        attn_output = self.attn(q, k, v, forward_batch)
        if self.use_head_wise_attn_gate:
            output = (
                attn_output.view(
                    attn_output.shape[0],
                    self.num_heads,  # TODO: check if this is correct
                    self.head_dim,
                )
                * gate_states.unsqueeze(-1).sigmoid()
            )
            attn_output = output.view(*attn_output.shape)
        output, _ = self.o_proj(attn_output)
        return output


class Step3p5DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Step3p5Config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        layer_types = config.layer_types
        yarn_only_types = config.yarn_only_types
        if layer_types[layer_id] not in yarn_only_types:
            rope_scaling = None
        else:
            rope_scaling = config.rope_scaling
        rope_theta = config.rope_theta
        max_position_embeddings = config.max_position_embeddings
        head_dim = config.head_dim
        moe_layers_set = {int(x) for x in config.moe_layers_enum.split(",")}
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_attention_groups
        self.is_moe_layer = layer_id in moe_layers_set
        self.is_previous_layer_sparse = (layer_id - 1) in moe_layers_set
        self.is_next_layer_sparse = (layer_id + 1) in moe_layers_set
        num_hidden_layers = config.num_hidden_layers

        if (
            config.swiglu_limits_shared
            and config.swiglu_limits_shared[layer_id] is not None
            and config.swiglu_limits_shared[layer_id] != 0
        ):
            swiglu_limit_shared = config.swiglu_limits_shared[layer_id]
        else:
            swiglu_limit_shared = None

        self.sliding_window = -1

        enable_sliding_window = layer_types[layer_id] == "sliding_attention"

        if enable_sliding_window:
            self.sliding_window = config.sliding_window
            self.num_attention_heads = config.attention_other_setting[
                "num_attention_heads"
            ]
            self.num_key_value_heads = config.attention_other_setting[
                "num_attention_groups"
            ]

        self.self_attn = Step3p5Attention(
            hidden_size=self.hidden_size,
            num_heads=self.num_attention_heads,
            num_kv_heads=self.num_key_value_heads,
            layer_id=(
                layer_id
                if layer_id < num_hidden_layers
                else layer_id - num_hidden_layers
            ),
            rope_theta=rope_theta[layer_id],
            rope_scaling=rope_scaling,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            sliding_window_size=self.sliding_window,
            partial_rotary_factor=config.partial_rotary_factors[layer_id],
            quant_config=quant_config,
            rms_norm_eps=config.rms_norm_eps,
            use_head_wise_attn_gate=config.use_head_wise_attn_gate,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
        )
        self.use_moe = False
        if self.is_moe_layer:
            self.moe = Step3p5MoEMLP(
                config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
            # The MoE block adds the shared expert's output to the routed output
            # before its one reduction. Where the routed output arrives complete
            # on each rank's own tokens, the shared expert is not TP-sharded.
            self.share_expert = Step3p5MLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.share_expert_dim,
                swiglu_limit=swiglu_limit_shared,
                quant_config=quant_config,
                prefix=add_prefix("share_expert", prefix),
                reduce_results=False,
                **(
                    dict(tp_rank=0, tp_size=1)
                    if get_moe_a2a_backend().is_deepep()
                    or get_moe_a2a_backend().is_mooncake()
                    or get_moe_a2a_backend().is_nixl()
                    or get_moe_a2a_backend().is_mori()
                    or get_moe_a2a_backend().is_ascend_fuseep()
                    or get_moe_a2a_backend().is_flashinfer()
                    or get_moe_a2a_backend().is_flashinfer_megamoe()
                    or should_use_flashinfer_cutlass_moe_fp4_allgather()
                    else {}
                ),
            )
            self.use_moe = True
        else:
            self.mlp = Step3p5MLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                swiglu_limit=swiglu_limit_shared,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )

        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # An MTP draft is a one-layer model; layer_id still indexes its config.

        self.attn_boundary, self.ffn_boundary = make_stages(
            (declare_attn(), self.input_layernorm),
            (
                declare_ffn(
                    sparse=self.is_moe_layer,
                    next_layer_sparse=self.is_next_layer_sparse,
                ),
                self.post_attention_layernorm,
            ),
            previous=declare_ffn(
                sparse=self.is_previous_layer_sparse,
                next_layer_sparse=self.is_moe_layer,
            )
            if not (0 if is_nextn else layer_id) == 0
            else None,
            terminal=(0 if is_nextn else layer_id)
            == (1 if is_nextn else config.num_hidden_layers) - 1,
        )

        self.layer_id = layer_id

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self Attention
        hidden_states = self.attn_boundary.prepare(
            hidden_states, forward_batch, post_residual_addition=post_residual_addition
        )
        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        # Fully Connected
        hidden_states = self.attn_boundary.finish(hidden_states, forward_batch)
        hidden_states = self.ffn_boundary.prepare(hidden_states, forward_batch)

        if self.use_moe:
            with self.ffn_boundary.exit(forward_batch) as ffn_exit:
                share_output = self.share_expert(hidden_states)
                hidden_states = self.moe(
                    hidden_states, forward_batch, shared_output=share_output
                )
            return ffn_exit.finish(hidden_states)

        with self.ffn_boundary.exit(forward_batch) as ffn_exit:
            hidden_states = self.mlp(hidden_states)
        return ffn_exit.finish(hidden_states)


class Step3p5Model(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.pp_group = get_parallel().pp_group

        alt_stream = get_stream("alt") if _is_cuda else None

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                enable_tp=not is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", prefix),
                params_dtype=(
                    torch.float32
                    if get_exec().deterministic.rl_on_policy_target is not None
                    else None
                ),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.layers, self.start_layer, self.end_layer = make_pp_layers(
            config.num_hidden_layers,
            # 1,
            lambda idx, prefix: Step3p5DecoderLayer(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            ),
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

    def get_input_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        if hasattr(self.config, "scale_emb"):
            return self.get_input_embeddings()(input_ids) * self.config.scale_emb
        else:
            return self.get_input_embeddings()(input_ids)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

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
            residual_batch.start(forward_batch)
        else:
            assert pp_proxy_tensors is not None
            hidden_states = self.layers[self.start_layer].attn_boundary.from_pp(
                pp_proxy_tensors, forward_batch
            )

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states = layer(positions, hidden_states, forward_batch)
            # break
        if not self.pp_group.is_last_rank:
            return residual_batch.to_pp(hidden_states, forward_batch)
        else:
            hidden_states = residual_batch.complete_output(hidden_states, forward_batch)
            hidden_states_before_norm = None
            if not self.pp_group.is_last_rank:
                return residual_batch.to_pp(hidden_states, forward_batch)
            else:
                if hidden_states.shape[0] > 0:
                    # if forward_batch.return_hidden_states_before_norm:
                    hidden_states_before_norm = residual_batch.snapshot(
                        hidden_states, forward_batch
                    )
                    hidden_states = residual_batch.final_norm(
                        hidden_states, forward_batch, self.norm
                    )
            return hidden_states, hidden_states_before_norm


class Step3p5ForCausalLM(nn.Module):
    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.moe_num_experts,
        )

    def __init__(
        self,
        config: Step3p5Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_parallel().pp_group
        self.config = config
        self.quant_config = quant_config
        self.model = Step3p5Model(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        self.tie_word_embeddings = False
        self.num_fused_shared_experts = 0

        # handle the lm head on different pp ranks
        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and self.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    use_attn_tp_group=get_parallel().enable_dp_lm_head,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            # ranks other than the last rank will have a placeholder layer
            self.lm_head = PPMissingLayer()

        # perform weight tying for PP
        if self.pp_group.world_size > 1 and self.tie_word_embeddings:
            if self.pp_group.is_first_rank:
                self.pp_group.send(
                    self.model.embed_tokens.weight, dst=self.pp_group.world_size - 1
                )
            elif self.pp_group.is_last_rank:
                emb_token_weight = self.pp_group.recv(
                    size=self.lm_head.weight.shape,
                    dtype=next(self.model.parameters()).dtype,
                    src=0,
                )
                self.lm_head.weight.copy_(emb_token_weight)

        self.logits_processor = LogitsProcessor(config)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states, hidden_states_before_norm = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                forward_batch,
                hidden_states_before_norm=hidden_states_before_norm,
            )
        else:
            return hidden_states

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        # NOTE:
        # Step3p5 HF checkpoints (e.g. MTP/nextn variants) may include an extra
        # "nextn predict layer" appended after the main decoder layers, such as:
        #   model.layers.<num_hidden_layers>.(eh_proj|enorm|hnorm|transformer.shared_head.*)
        # This implementation currently does NOT instantiate those nextn modules,
        # so we must safely skip them (or load them only when a corresponding
        # nextn model is implemented).

        def _get_layer_id_from_weight_name(weight_name: str) -> Optional[int]:
            # Expected format: "model.layers.<id>...."
            parts = weight_name.split(".")
            if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
                try:
                    return int(parts[2])
                except ValueError:
                    return None
            return None

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        if self.num_fused_shared_experts > 0:
            assert self.num_fused_shared_experts == 1

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.moe_num_experts + self.num_fused_shared_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params = set()

        def match_expert_and_shard_ids(name_path: str, weight_path: str) -> bool:
            name_parts = name_path.split(".")
            weight_parts = weight_path.split(".")
            # Be defensive: some unexpected weight names may not match the shape.
            if len(name_parts) <= 4 or len(weight_parts) <= 2:
                return False
            shard_id_matches = name_parts[4] == weight_parts[2]
            return shard_id_matches

        for name, loaded_weight in weights:
            # Filter nextn layer weights.
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = getattr(self.config, "num_nextn_predict_layers", 0)
                if num_nextn_layers and name.startswith("model.layers."):
                    layer_id = _get_layer_id_from_weight_name(name)
                    if layer_id is not None:
                        if not is_nextn:
                            # Normal load: skip layers appended after the main decoder.
                            if layer_id >= self.config.num_hidden_layers:
                                continue
                        else:
                            # nextn load: only keep the appended nextn layer.
                            # (Only 1 nextn layer is supported by current checkpoints.)
                            if num_nextn_layers != 1:
                                raise ValueError(
                                    "Only 1 nextn layer is supported for Step3p5 checkpoints."
                                )
                            nextn_layer_id = (
                                0
                                if self.config.num_hidden_layers == 1
                                else self.config.num_hidden_layers
                            )
                            if layer_id != nextn_layer_id:
                                # # nextn/MTP load: only keep the appended nextn layers.
                                # # Expected layer ids: [num_hidden_layers, num_hidden_layers + num_nextn_layers).
                                # start = self.config.num_hidden_layers
                                # end = self.config.num_hidden_layers + num_nextn_layers
                                # if not (start <= layer_id < end):
                                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "gate." not in name and "moe" in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    # Extra / unsupported weights (e.g. nextn) should not crash loading.
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if "moe" not in name or "router_bias" in name:
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                else:
                    if "gate." in name:
                        if name not in params_dict:
                            continue
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        weight_loader(param, loaded_weight)
                        loaded_params.add(name)
                        continue

                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if expert_id == self.config.moe_num_experts:
                            continue
                        if not match_expert_and_shard_ids(name, weight_name):
                            continue
                        part_name = weight_name.split(".")[-2]
                        fake_weight_name = name.replace(part_name, weight_name[:-1])
                        actual_param_name = name.replace(part_name + ".", param_name)
                        if actual_param_name not in params_dict:
                            continue
                        param = params_dict[actual_param_name]
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight[expert_id],
                            name,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                        loaded_params.add(actual_param_name)

        # Derived parameters (e.g. blockscale_swizzled from NVFP4 quantization)
        # are computed in process_weights_after_loading, not loaded from checkpoint.
        print_params = {
            p
            for p in set(params_dict.keys()) - loaded_params
            if "blockscale_swizzled" not in p
        }
        assert len(print_params) == 0, f"Some parameters are not loaded: {print_params}"

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


EntryClass = Step3p5ForCausalLM
