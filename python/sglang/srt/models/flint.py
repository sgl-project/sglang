import os
import sys
from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.configs.flint import SLIDING_ATTENTION, FlintConfig
from sglang.srt.layers.layer_boundary import declare_attn, declare_ffn
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import get_layer_id
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
from sglang.srt.models.utils import apply_qk_norm, enable_fused_set_kv_buffer
from sglang.srt.runtime_context import attention_backends, get_exec, get_parallel
from sglang.srt.utils import add_prefix, set_weight_attrs


class FlintMLP(Qwen2MoeMLP):
    def forward(
        self, hidden_states: torch.Tensor, forward_batch: Optional[ForwardBatch] = None
    ) -> torch.Tensor:
        return super().forward(hidden_states)


class FlintAttention(Qwen3MoeAttention):
    def __init__(self, *, skip_rope: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.skip_rope = skip_rope
        self.use_fused_qk_norm_rope = False
        self.use_fused_qk_norm_rope_cpu = False

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

    def apply_qk_norm_rope(self, qkv, positions, forward_batch):
        if not self.skip_rope:
            return super().apply_qk_norm_rope(qkv, positions, forward_batch)
        self._used_fused_qk_norm_rope_last_call = False
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = apply_qk_norm(
            q=q,
            k=k,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            head_dim=self.head_dim,
            alt_stream=self.alt_stream,
        )
        return q, k, v

    def forward_core(self, intermediate_state):
        hidden_states, forward_batch, inner_state = intermediate_state
        if inner_state is None:
            return hidden_states
        q, k, v, fb = inner_state
        save_kv_cache = (
            self.skip_rope
            or self._used_fused_qk_norm_rope_last_call
            or not (
                enable_fused_set_kv_buffer(fb) and self.compatible_with_fused_kv_buffer
            )
        )
        attn_output = self.attn(
            q, k, v, fb, save_kv_cache=save_kv_cache, sinks=self.softmax_offset
        )
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
        self.shared_expert = FlintMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.shared_expert_intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            reduce_results=False,
            prefix=add_prefix("shared_expert", prefix),
        )

    def _make_topk(
        self, config: FlintConfig, layer_id: int, norm_topk_prob: bool
    ) -> TopK:
        self.gate.expert_bias = nn.Parameter(
            torch.zeros(config.num_experts, dtype=torch.float32), requires_grad=False
        )
        return TopK(
            top_k=config.num_experts_per_tok,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=False,
            scoring_func="sigmoid",
            correction_bias=self.gate.expert_bias,
            layer_id=layer_id,
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
        if inplace:
            shared_input = hidden_states.clone()
        self.alt_stream.wait_stream(current_stream)
        if not inplace:
            shared_input = hidden_states.clone()
        shared_output = self.shared_expert(shared_input)
        with torch.cuda.stream(self.alt_stream):
            routed_output = super().forward(hidden_states, forward_batch)
        current_stream.wait_stream(self.alt_stream)
        return routed_output + shared_output


class FlintDecoderLayer(Qwen3MoeDecoderLayer):
    def _make_attention(
        self, config: FlintConfig, layer_id: int, **kwargs
    ) -> FlintAttention:
        is_sliding = (
            config.layer_types[layer_id] == SLIDING_ATTENTION
            and config.sliding_window is not None
        )
        return FlintAttention(
            skip_rope=layer_id in config.no_rope_layer_indices,
            sliding_window_size=config.sliding_window - 1 if is_sliding else -1,
            config=config,
            layer_id=layer_id,
            **kwargs,
        )

    def _make_mlp(
        self,
        config: FlintConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
        alt_stream: Optional[torch.cuda.Stream],
    ) -> nn.Module:
        if self.is_layer_sparse:
            return FlintSparseMoeBlock(
                layer_id=self.layer_id,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            )
        return FlintMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            reduce_results=False,
            prefix=prefix,
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
    model_cls = FlintModel

    def __init__(
        self,
        config: FlintConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        _require_flashinfer_cuda_norm()
        super().__init__(config, quant_config=quant_config, prefix=prefix)

    def get_attention_sliding_window_size(self) -> Optional[int]:
        if self.config.sliding_window is None:
            return None
        return self.config.sliding_window - 1

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_mtp: bool = False
    ):
        param_names = {name for name, _ in self.named_parameters()}

        def renamed_weights():
            for name, tensor in weights:
                name = name.replace(
                    ".mlp.gate.e_score_correction_bias", ".mlp.gate.expert_bias"
                ).replace(".self_attn.sinks", ".self_attn.softmax_offset")
                if (
                    name.startswith("model.layers.")
                    and name.endswith(".self_attn.softmax_offset")
                    and name not in param_names
                    and self._is_local_layer(name)
                ):
                    raise RuntimeError(f"Flint attention sink {name} has no parameter.")
                yield name, tensor

        super().load_weights(renamed_weights(), is_mtp=is_mtp)

    def _is_local_layer(self, name: str) -> bool:
        layer_id = get_layer_id(name)
        return layer_id is None or self.start_layer <= layer_id < self.end_layer


def _require_flashinfer_cuda_norm() -> None:
    norm = sys.modules.get("flashinfer.norm")
    if norm is None:
        return
    use_cuda_norm = getattr(norm, "_USE_CUDA_NORM", None)
    if use_cuda_norm is None:
        use_cuda_norm = os.environ.get("FLASHINFER_USE_CUDA_NORM", "").lower() in (
            "1",
            "true",
        )
    if not use_cuda_norm:
        raise RuntimeError(
            "Flint requires FlashInfer's CUDA RMSNorm; set FLASHINFER_USE_CUDA_NORM=1."
        )


EntryClass = FlintForCausalLM
