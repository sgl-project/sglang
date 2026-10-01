# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import math
from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.configs.beam import BeamConfig
from sglang.srt.distributed import (
    moe_expert_parallel_all_reduce,
    moe_tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
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
    get_moe_runner_backend,
    should_skip_post_experts_all_reduce,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.layers.moe.utils import RoutingMethodType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import RotaryEmbedding
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.utils import add_prefix, make_layers, set_weight_attrs


def apply_ssmax(q: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Scale Q using absolute positions, rounding only after multiplication."""
    scale = torch.exp(0.1 * torch.log1p((positions.float() + 1) / 2048))
    return (q.float() * scale.unsqueeze(-1)).to(q.dtype)


def apply_attention_gate(
    attention_output: torch.Tensor, gate: torch.Tensor, gate_bias: torch.Tensor
) -> torch.Tensor:
    # Bias addition, sigmoid, and multiplication use the activation dtype.
    # The bias has one entry per local query head, broadcast over head_dim.
    multiplier = torch.sigmoid(gate + gate_bias[None, :, None])
    return attention_output * multiplier


def depth_scaled_residual(
    branch: torch.Tensor, residual: torch.Tensor, scales: Tuple[float, float]
) -> torch.Tensor:
    branch_scale, residual_scale = scales
    residual = residual.to(branch.dtype)
    # Round both scaled operands before adding in the activation dtype.
    branch = branch * branch_scale
    residual = residual * residual_scale
    return branch + residual


class BeamRotaryEmbedding(RotaryEmbedding):
    def __init__(self, config: BeamConfig):
        super().__init__(
            head_size=config.head_dim,
            rotary_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_parameters["sliding_attention"]["rope_theta"],
            is_neox_style=True,
            dtype=torch.get_default_dtype(),
        )

    def _torch_compile_forward(self, num_tokens: int):
        # Preserve FP32 CUDA rotation instead of the eager BF16 intermediates.
        return None


class BeamSiluAndMul(SiluAndMul):
    def _torch_compile_forward(self, num_tokens: int):
        # Keep the fused activation/product and its single output cast.
        return None


class BeamUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    def _torch_compile_forward(self, num_tokens: int):
        # The torch-native fallback rounds route weights and omits the runner
        # scale. Keep the CUDA runner, including during single-token compile.
        return None


class BeamMLP(nn.Module):
    def __init__(
        self,
        config: BeamConfig,
        prefix: str = "",
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
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
        self.act_fn = BeamSiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class BeamMoE(nn.Module):
    def __init__(
        self,
        config: BeamConfig,
        layer_id: int,
        prefix: str = "",
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.tp_size = get_parallel().moe_tp_size
        self.ep_size = get_parallel().moe_ep_size
        backend = get_moe_runner_backend()
        supported_runner = (
            backend.is_auto() or backend.is_triton() or backend.is_flashinfer_trtllm()
        )
        if not supported_runner or not get_moe_a2a_backend().is_none():
            raise ValueError(
                "Beam supports the auto, triton, or flashinfer_trtllm MoE runner, "
                "with no MoE all-to-all"
            )
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            params_dtype=torch.float32,
            prefix=add_prefix("gate", prefix),
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.num_experts, dtype=torch.float32), requires_grad=False
        )
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            layer_id=layer_id,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            renormalize=True,
            scoring_func="sigmoid",
            correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=(
                config.routed_scaling_factor if backend.is_flashinfer_trtllm() else None
            ),
            output_format=(
                None if backend.is_flashinfer_trtllm() else TopKOutputFormat.STANDARD
            ),
        )
        # Keep normalized routing coefficients in FP32. The runner applies the
        # routed scale during combination, after weighting each expert output.
        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=layer_id,
            inplace=False,
            routed_scaling_factor=config.routed_scaling_factor,
            routing_method_type=RoutingMethodType.DeepSeekV3,
            quant_config=quant_config,
            quant_method=(
                BeamUnquantizedFusedMoEMethod(
                    use_flashinfer_trtllm_moe=backend.is_flashinfer_trtllm()
                )
                if quant_config is None
                else None
            ),
            prefix=add_prefix("experts", prefix),
        )
        # Some expert methods consume routing weights with scale already applied.
        if self.experts.should_fuse_routed_scaling_factor_in_topk:
            self.topk.topk_config.routed_scaling_factor = config.routed_scaling_factor
            self.topk.topk_config.apply_routed_scaling_factor_on_output = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[0] > 0:
            router_logits, _ = self.gate(hidden_states.float())
            topk_output = self.topk(hidden_states, router_logits)
        else:
            topk_output = self.topk.empty_topk_output(
                hidden_states.device, layer_id=self.layer_id
            )
        output = self.experts(hidden_states, topk_output)
        if self.ep_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=False
        ):
            output = moe_expert_parallel_all_reduce(output)
        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=True
        ):
            output = moe_tensor_model_parallel_all_reduce(output)
        return output


class BeamAttention(nn.Module):
    def __init__(
        self,
        config: BeamConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        rotary_emb: BeamRotaryEmbedding | None = None,
    ):
        super().__init__()
        self.tp_rank = get_parallel().attn_tp_rank
        self.tp_size = get_parallel().attn_tp_size
        if config.num_attention_heads % self.tp_size:
            raise ValueError("Beam query heads must be divisible by attention TP size")
        if config.num_key_value_heads >= self.tp_size:
            if config.num_key_value_heads % self.tp_size:
                raise ValueError("Beam KV heads must be divisible by attention TP size")
        elif self.tp_size % config.num_key_value_heads:
            raise ValueError("Beam attention TP size must be divisible by KV heads")
        self.num_heads = config.num_attention_heads // self.tp_size
        self.num_kv_heads = max(1, config.num_key_value_heads // self.tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # Sliding layers use RoPE; full-attention layers use NoPE with SSMax.
        self.apply_rope = config.layer_types[layer_id] == "sliding_attention"
        self.sliding_window = config.sliding_window - 1 if self.apply_rope else -1
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=False,
            quant_config=quant_config,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            prefix=add_prefix("gate_proj", prefix),
        )
        self.gate_bias = nn.Parameter(torch.empty(self.num_heads), requires_grad=False)
        set_weight_attrs(self.gate_bias, {"weight_loader": self.load_gate_bias})
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            use_dp_attention_reduce=True,
            prefix=add_prefix("o_proj", prefix),
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = (
            (rotary_emb if rotary_emb is not None else BeamRotaryEmbedding(config))
            if self.apply_rope
            else None
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=self.sliding_window,
            prefix=add_prefix("attn", prefix),
        )

    def load_gate_bias(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        default_weight_loader(
            param,
            loaded_weight.narrow(0, self.tp_rank * self.num_heads, self.num_heads),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        if hidden_states.shape[0] == 0:
            return hidden_states
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.reshape(-1, self.head_dim)).view(-1, self.q_size)
        k = self.k_norm(k.reshape(-1, self.head_dim)).view(-1, self.kv_size)
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
        else:
            q = apply_ssmax(q, positions)
        attention_output = self.attn(q, k, v, forward_batch)
        gate, _ = self.gate_proj(hidden_states)
        attention_output = apply_attention_gate(
            attention_output.view(-1, self.num_heads, self.head_dim),
            gate.view(-1, self.num_heads, self.head_dim),
            self.gate_bias,
        ).view(-1, self.q_size)
        output, _ = self.o_proj(attention_output)
        return output


class BeamDecoderLayer(nn.Module):
    def __init__(
        self,
        config: BeamConfig,
        layer_id: int,
        prefix: str = "",
        rotary_emb: BeamRotaryEmbedding | None = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        attn_depth = 2 * (layer_id + 1)
        mlp_depth = attn_depth + 1
        attn_denominator = math.sqrt(attn_depth)
        mlp_denominator = math.sqrt(mlp_depth)
        self.attn_residual_scales = (
            1.0 / attn_denominator,
            math.sqrt(attn_depth - 1) / attn_denominator,
        )
        self.mlp_residual_scales = (
            1.0 / mlp_denominator,
            math.sqrt(mlp_depth - 1) / mlp_denominator,
        )
        self.self_attn = BeamAttention(
            config,
            layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            rotary_emb=rotary_emb,
        )
        self.mlp = (
            BeamMLP(config, prefix=add_prefix("mlp", prefix), quant_config=quant_config)
            if config.mlp_layer_types[layer_id] == "dense"
            else BeamMoE(
                config,
                layer_id,
                prefix=add_prefix("mlp", prefix),
                quant_config=quant_config,
            )
        )
        # Round once, after FP32 normalization and learned-weight multiplication.
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, forward_batch)
        hidden_states = depth_scaled_residual(
            self.post_attention_layernorm(hidden_states),
            residual,
            self.attn_residual_scales,
        )
        residual = hidden_states
        hidden_states = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        return depth_scaled_residual(
            self.post_feedforward_layernorm(hidden_states),
            residual,
            self.mlp_residual_scales,
        )


class BeamModel(nn.Module):
    def __init__(
        self,
        config: BeamConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.pp_group = get_parallel().pp_group
        self.embed_tokens = (
            VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=add_prefix("embed_tokens", prefix),
            )
            if self.pp_group.is_first_rank
            else PPMissingLayer()
        )
        # Reuse the FP32 position cache across this model's rotary layers.
        rotary_emb = (
            BeamRotaryEmbedding(config)
            if "sliding_attention" in config.layer_types
            else None
        )
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: BeamDecoderLayer(
                config, idx, prefix, rotary_emb=rotary_emb, quant_config=quant_config
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        self.norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if self.pp_group.is_last_rank
            else PPMissingLayer()
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        if self.pp_group.is_first_rank:
            hidden_states = (
                self.embed_tokens(input_ids) if input_embeds is None else input_embeds
            )
        else:
            hidden_states = pp_proxy_tensors["hidden_states"]
        for layer_id in range(self.start_layer, self.end_layer):
            hidden_states = self.layers[layer_id](
                positions, hidden_states, forward_batch
            )
        if not self.pp_group.is_last_rank:
            return PPProxyTensors({"hidden_states": hidden_states})
        return self.norm(hidden_states)


class BeamForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: BeamConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        if get_parallel().enable_dp_attention or get_parallel().enable_dp_lm_head:
            raise NotImplementedError(
                "Beam does not yet support attention data parallelism or a "
                "data-parallel LM head"
            )
        if (
            get_exec().moe.enable_eplb
            or get_exec().moe.init_expert_location != "trivial"
            or get_exec().moe.ep_num_redundant_experts > 0
        ):
            raise NotImplementedError(
                "Beam has not been validated with expert relocation or "
                "redundant experts; use trivial expert placement"
            )
        self.config = config
        self.model = BeamModel(config, quant_config, prefix=add_prefix("model", prefix))
        self.start_layer = self.model.start_layer
        self.end_layer = self.model.end_layer
        self.lm_head = (
            ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                params_dtype=torch.bfloat16,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
            if self.model.pp_group.is_last_rank
            else PPMissingLayer()
        )
        # The default head projection returns BF16; LogitsProcessor promotes
        # its output to FP32 before softcap and logprobs.
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
        )
        if not self.model.pp_group.is_last_rank:
            return hidden_states
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_attention_sliding_window_size(self):
        if "sliding_attention" in self.config.layer_types:
            return self.config.sliding_window - 1
        return None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        packed_projections = [
            (".self_attn.q_proj.", ".self_attn.qkv_proj.", "q"),
            (".self_attn.k_proj.", ".self_attn.qkv_proj.", "k"),
            (".self_attn.v_proj.", ".self_attn.qkv_proj.", "v"),
            (".mlp.gate_proj.", ".mlp.gate_up_proj.", 0),
            (".mlp.up_proj.", ".mlp.gate_up_proj.", 1),
        ]
        expert_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )
        params = dict(self.named_parameters())
        loaded = set()
        for name, weight in weights:
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and not self.start_layer <= layer_id < self.end_layer
            ):
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            if (
                name == "model.embed_tokens.weight"
                and not self.model.pp_group.is_first_rank
            ):
                continue
            if (
                name in ("model.norm.weight", "lm_head.weight")
                and not self.model.pp_group.is_last_rank
            ):
                continue
            if name.startswith("lm_head.") and not self.model.pp_group.is_last_rank:
                continue
            for source, target, shard_id in packed_projections:
                if source in name:
                    target_name = name.replace(source, target)
                    param = params[target_name]
                    param.weight_loader(param, weight, shard_id)
                    loaded.add(target_name)
                    break
            else:
                for target, source, expert_id, shard_id in expert_mapping:
                    if source in name:
                        target_name = name.replace(source, target)
                        param = params[target_name]
                        param.weight_loader(
                            param,
                            weight,
                            target_name,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                        loaded.add(target_name)
                        break
                else:
                    # Packed expert checkpoints use [expert, 2*intermediate, hidden]
                    # in gate/up order, or [expert, hidden, intermediate] for down.
                    if name.endswith(".mlp.experts.gate_up_proj"):
                        target_name = name.replace("gate_up_proj", "w13_weight")
                        param = params[target_name]
                        for expert_id, expert_weight in enumerate(weight):
                            gate, up = expert_weight.chunk(2, dim=0)
                            for shard_id, shard in (("w1", gate), ("w3", up)):
                                param.weight_loader(
                                    param,
                                    shard,
                                    target_name,
                                    shard_id=shard_id,
                                    expert_id=expert_id,
                                )
                        loaded.add(target_name)
                    elif name.endswith(".mlp.experts.down_proj"):
                        target_name = name.replace("down_proj", "w2_weight")
                        param = params[target_name]
                        for expert_id, shard in enumerate(weight):
                            param.weight_loader(
                                param,
                                shard,
                                target_name,
                                shard_id="w2",
                                expert_id=expert_id,
                            )
                        loaded.add(target_name)
                    else:
                        param = params[name]
                        loader = getattr(param, "weight_loader", default_weight_loader)
                        loader(param, weight)
                        loaded.add(name)
        return loaded


EntryClass = BeamForCausalLM
