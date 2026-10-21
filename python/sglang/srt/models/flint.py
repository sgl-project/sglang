import sys
from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.configs.flint import SLIDING_ATTENTION, FlintConfig
from sglang.srt.layers.layer_boundary import append_stages, declare_attn, declare_ffn
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.models.qwen2_moe import Qwen2MoeMLP
from sglang.srt.models.qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
    Qwen3MoeSparseMoeBlock,
)
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.runtime_context import attention_backends, get_exec, get_parallel
from sglang.srt.utils import LazyValue, add_prefix, set_weight_attrs
from sglang.srt.utils.hf_transformers_utils import get_rope_config


class FlintMLP(Qwen2MoeMLP):
    def forward(
        self, hidden_states: torch.Tensor, forward_batch: Optional[ForwardBatch] = None
    ) -> torch.Tensor:
        return super().forward(hidden_states)


class FlintAttention(Qwen3MoeAttention):
    def __init__(
        self,
        *,
        layer_type: str,
        skip_rope: bool,
        sliding_window: Optional[int],
        layer_id: int,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__(layer_id=layer_id, prefix=prefix, **kwargs)
        self.skip_rope = skip_rope

        attn_tp_size = get_parallel().attn_tp_size
        attn_tp_rank = get_parallel().attn_tp_rank
        sinks_dtype = (
            torch.float32 if "trtllm_mha" in attention_backends() else torch.bfloat16
        )
        self.softmax_offset = nn.Parameter(
            torch.empty(self.num_heads, dtype=sinks_dtype), requires_grad=False
        )
        set_weight_attrs(
            self.softmax_offset,
            {
                "weight_loader": lambda param, loaded_weight: param.data.copy_(
                    loaded_weight.chunk(attn_tp_size, dim=0)[attn_tp_rank]
                )
            },
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=(
                sliding_window - 1
                if layer_type == SLIDING_ATTENTION and sliding_window is not None
                else -1
            ),
            prefix=add_prefix("attn", prefix),
        )

    def apply_qk_norm_rope(self, qkv, positions, forward_batch):
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = apply_qk_norm(
            q=q,
            k=k,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            head_dim=self.head_dim,
            alt_stream=self.alt_stream,
        )
        if not self.skip_rope:
            q, k = self.rotary_emb(positions, q, k)
        return q, k, v

    def forward_core(self, intermediate_state):
        hidden_states, forward_batch, inner_state = intermediate_state
        if inner_state is None:
            return hidden_states
        q, k, v, fb = inner_state
        attn_output = self.attn(q, k, v, fb, sinks=self.softmax_offset)
        output, _ = self.o_proj(attn_output)
        return output


class FlintSparseMoeBlock(Qwen3MoeSparseMoeBlock):
    def __init__(
        self,
        layer_id: int,
        config: FlintConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__(layer_id, config, quant_config=quant_config, prefix=prefix)
        self.alt_stream = alt_stream
        self.gate.expert_bias = nn.Parameter(
            torch.zeros(config.num_experts, dtype=torch.float32), requires_grad=False
        )
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=False,
            scoring_func="sigmoid",
            correction_bias=self.gate.expert_bias,
            layer_id=layer_id,
        )
        self.shared_expert = FlintMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.shared_expert_intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            reduce_results=False,
            prefix=add_prefix("shared_expert", prefix),
        )

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: Optional[ForwardBatch] = None
    ) -> torch.Tensor:
        if (
            self.alt_stream is not None
            and hidden_states.shape[0] > 0
            and get_is_capture_mode()
            and not torch.compiler.is_compiling()
            and not (
                get_exec().moe.enable_paged_experts and is_in_breakable_cuda_graph()
            )
        ):
            return self._forward_dual_stream(hidden_states, forward_batch)
        shared_output = self.shared_expert(hidden_states)
        return super().forward(hidden_states, forward_batch) + shared_output

    def _forward_dual_stream(
        self, hidden_states: torch.Tensor, forward_batch: Optional[ForwardBatch]
    ) -> torch.Tensor:
        current_stream = torch.cuda.current_stream()
        inplace = self.experts.moe_runner_config.inplace
        shared_input = hidden_states.clone() if inplace else None
        self.alt_stream.wait_stream(current_stream)
        if shared_input is None:
            shared_input = hidden_states.clone()
        shared_output = self.shared_expert(shared_input)
        with torch.cuda.stream(self.alt_stream):
            routed_output = super().forward(hidden_states, forward_batch)
        current_stream.wait_stream(self.alt_stream)
        return routed_output + shared_output


class FlintDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(
        self,
        config: FlintConfig,
        layer_id: int,
        start_layer: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id
        rope_theta, rope_scaling = get_rope_config(config)
        self.rope_theta = rope_theta
        self.self_attn = FlintAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            start_layer=start_layer,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
            layer_type=config.layer_types[layer_id],
            skip_rope=layer_id in config.no_rope_layer_indices,
            sliding_window=config.sliding_window,
        )

        attn, ffn = self.stage_facts(config, layer_id)
        self.is_layer_sparse = ffn.sparse
        if self.is_layer_sparse:
            self.mlp = FlintSparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                alt_stream=alt_stream,
            )
        else:
            self.mlp = FlintMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("mlp", prefix),
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn_boundary, self.ffn_boundary = append_stages(
            (attn, self.input_layernorm),
            (ffn, self.post_attention_layernorm),
        )

    @staticmethod
    def stage_facts(config: FlintConfig, layer_id: int):
        return (
            declare_attn(),
            declare_ffn(
                sparse=layer_id not in config.mlp_only_layers,
                next_layer_sparse=layer_id + 1 not in config.mlp_only_layers,
            ),
        )


class FlintModel(Qwen3MoeModel):
    def __init__(
        self,
        config: FlintConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=FlintDecoderLayer,
        )


class FlintForCausalLM(Qwen3MoeForCausalLM):
    def __init__(
        self,
        config: FlintConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        _require_flashinfer_cuda_norm()
        self.pp_group = get_parallel().pp_group
        self.config = config
        self.quant_config = quant_config
        self.model = FlintModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_parallel().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False
        self.routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: self.model.layers[layer_id].mlp.get_moe_weights()
                for layer_id in range(self.start_layer, self.end_layer)
                if isinstance(self.model.layers[layer_id].mlp, FlintSparseMoeBlock)
            }
        )
        self.attn_cp_size = get_parallel().attn_cp_size
        self.moe_dp_size = get_parallel().moe_dp_size
        assert self.attn_cp_size % self.moe_dp_size == 0, (
            f"attn_cp_size ({self.attn_cp_size}) must be divisible by "
            f"moe_dp_size ({self.moe_dp_size})"
        )

    def get_attention_sliding_window_size(self) -> Optional[int]:
        if self.config.sliding_window is None:
            return None
        return self.config.sliding_window - 1

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_mtp: bool = False
    ):
        weights = (
            (
                name.replace(
                    ".mlp.gate.e_score_correction_bias", ".mlp.gate.expert_bias"
                ),
                tensor,
            )
            for name, tensor in weights
        )
        super().load_weights(weights, is_mtp=is_mtp)


def _require_flashinfer_cuda_norm() -> None:
    norm = sys.modules.get("flashinfer.norm")
    if norm is not None and not norm._USE_CUDA_NORM:
        raise RuntimeError(
            "Flint requires FlashInfer's CUDA RMSNorm; set FLASHINFER_USE_CUDA_NORM=1."
        )


EntryClass = FlintForCausalLM
