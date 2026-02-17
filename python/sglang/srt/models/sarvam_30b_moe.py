"""Inference-only SarvamMoE model compatible with HuggingFace weights for SGLang.
"""

import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_moe_expert_parallel_rank,
    get_moe_expert_parallel_world_size,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    enable_moe_dense_fully_dp,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.moe import (
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.moe.utils import RoutingMethodType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.utils import (
    apply_qk_norm,
    create_fused_set_kv_buffer_arg,
    enable_fused_set_kv_buffer,
)
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    add_prefix,
    is_cuda,
    make_layers,
)

_is_cuda = is_cuda()

if _is_cuda:
    from sgl_kernel import fused_qk_norm_rope

logger = logging.getLogger(__name__)


def compute_yarn_parameters(
    config: PretrainedConfig,
) -> tuple[float, float, float, float]:
    """
    Computes YaRN RoPE scaling parameters for the fused QK norm + RoPE kernel.
    Refer to https://huggingface.co/papers/2309.00071
    
    Returns:
        factor: float, the scaling factor for the RoPE embeddings
        low: float, the lower bound of the dimension range
        high: float, the upper bound of the dimension range
        attention_factor: float, the post-processing scaling factor
    """
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is None:
        return 1.0, 0, 0, 1.0

    base = getattr(config, "rope_theta", 10000.0)
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    dim = int(head_dim * partial_rotary_factor)
    factor = getattr(rope_scaling, "factor", 1.0)
    attention_factor = rope_scaling.get("attention_factor")
    mscale = rope_scaling.get("mscale")
    mscale_all_dim = rope_scaling.get("mscale_all_dim")

    if "original_max_position_embeddings" in rope_scaling:
        original_max_position_embeddings = rope_scaling[
            "original_max_position_embeddings"
        ]
        factor = config.max_position_embeddings / original_max_position_embeddings
    else:
        original_max_position_embeddings = config.max_position_embeddings

    def get_mscale(scale, mscale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    if attention_factor is None:
        if mscale and mscale_all_dim:
            attention_factor = float(
                get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)
            )
        else:
            attention_factor = get_mscale(factor)

    beta_fast = rope_scaling.get("beta_fast") or 32
    beta_slow = rope_scaling.get("beta_slow") or 1

    def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        return (
            dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
        ) / (2 * math.log(base))

    def find_correction_range(
        low_rot, high_rot, dim, base, max_position_embeddings, truncate
    ):
        low = find_correction_dim(low_rot, dim, base, max_position_embeddings)
        high = find_correction_dim(high_rot, dim, base, max_position_embeddings)
        if truncate:
            low = math.floor(low)
            high = math.ceil(high)
        return max(low, 0), min(high, dim - 1)

    truncate = rope_scaling.get("truncate", True)
    low, high = find_correction_range(
        beta_fast, beta_slow, dim, base, original_max_position_embeddings, truncate
    )

    return factor, low, high, attention_factor


class SarvamMoEMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        reduce_results: bool = True,
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            reduce_results=reduce_results,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. Only silu is supported.")
        self.act_fn = SiluAndMul()

    def forward(
        self,
        x,
        forward_batch: ForwardBatch = None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ):
        if x.shape[0] == 0:
            return x
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x, skip_all_reduce=should_allreduce_fusion or use_reduce_scatter)
        return x


class SarvamMoESparseMoeBlock(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.score_function = getattr(config, "score_function", "sigmoid")
        self.alt_stream = alt_stream

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        if getattr(config, "moe_router_enable_expert_bias", False):
            self.expert_bias = nn.Parameter(
                torch.zeros(config.num_experts, dtype=torch.float32),
                requires_grad=False,
            )
        else:
            self.register_buffer(
                "expert_bias",
                torch.zeros(config.num_experts, dtype=torch.float32),
            )

        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            use_grouped_topk=False,
            num_expert_group=getattr(config, "n_group", 1),
            topk_group=getattr(config, "topk_group", 1),
            renormalize=True,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=False,
            scoring_func="sigmoid",
            correction_bias=self.expert_bias.data,
            quant_config=quant_config,
            layer_id=layer_id,
        )

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.num_experts + get_global_server_args().ep_num_redundant_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            routing_method_type=RoutingMethodType.Renormalize,
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )

        if getattr(config, "num_shared_experts", None) and config.num_shared_experts > 0:
            intermediate_size = config.moe_intermediate_size * config.num_shared_experts
            if enable_moe_dense_fully_dp():
                shared_tp_rank, shared_tp_size = 0, 1
            else:
                shared_tp_rank, shared_tp_size = None, None
            self.shared_experts = SarvamMoEMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("shared_experts", prefix),
                reduce_results=False,
                tp_rank=shared_tp_rank,
                tp_size=shared_tp_size,
            )
        else:
            self.shared_experts = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        # Use dual-stream overlap when:
        # 1. We have shared experts
        # 2. alt_stream is available
        # 3. We're in CUDA graph capture mode (for graph compatibility)
        # 4. We have tokens to process
        if (
            self.shared_experts is not None
            and self.alt_stream is not None
            and hidden_states.shape[0] > 0
            and get_is_capture_mode()
        ):
            return self.forward_normal_dual_stream(
                hidden_states, should_allreduce_fusion, use_reduce_scatter
            )
        else:
            return self.forward_normal(
                hidden_states, should_allreduce_fusion, use_reduce_scatter
            )

    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
        ]

    def _forward_shared_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.shared_experts(hidden_states)

    def _forward_router_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits, _ = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        return self.experts(hidden_states, topk_output)

    def forward_normal_dual_stream(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        
        shared_out = self._forward_shared_experts(hidden_states)
        
        with torch.cuda.stream(self.alt_stream):
            final_hidden_states = self._forward_router_experts(hidden_states)
            if self.routed_scaling_factor != 1.0:
                final_hidden_states = final_hidden_states * self.routed_scaling_factor
        
        current_stream.wait_stream(self.alt_stream)
        
        final_hidden_states = final_hidden_states + shared_out
        
        if (
            self.tp_size > 1
            and not should_allreduce_fusion
            and not use_reduce_scatter
            and not should_use_flashinfer_cutlass_moe_fp4_allgather()
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        
        return final_hidden_states.view(num_tokens, hidden_dim)

    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states

        num_tokens, hidden_dim = hidden_states.shape

        identity = hidden_states.clone() if self.shared_experts is not None else hidden_states

        router_logits, _ = self.gate(hidden_states)

        topk_output = self.topk(hidden_states, router_logits)

        final_hidden_states = self.experts(hidden_states, topk_output)

        if self.shared_experts is not None:
            shared_out = self.shared_experts(identity)
            if self.routed_scaling_factor != 1.0:
                shared_out.add_(final_hidden_states, alpha=self.routed_scaling_factor)
            else:
                shared_out.add_(final_hidden_states)
            final_hidden_states = shared_out
        elif self.routed_scaling_factor != 1.0:
            final_hidden_states = final_hidden_states * self.routed_scaling_factor

        if (
            self.tp_size > 1
            and not should_allreduce_fusion
            and not use_reduce_scatter
            and not should_use_flashinfer_cutlass_moe_fp4_allgather()
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)


class SarvamMoEAttention(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.layer_id = layer_id
        self.alt_stream = alt_stream

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        self.total_num_heads = num_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size

        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)

        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=getattr(config, "use_qkv_bias", False),
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=getattr(config, "use_bias", False),
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
        )

        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        self.compatible_with_fused_kv_buffer = True

        self.compatible_with_fused_qk_norm_rope = (
            self.use_qk_norm
            and self.head_dim in (64, 128, 256)
            and _is_cuda
        )
        self.use_fused_qk_norm_rope = (
            get_global_server_args().enable_fused_qk_norm_rope
            and self.compatible_with_fused_qk_norm_rope
        )
        self._used_fused_qk_norm_rope_last_call = False

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

        self._used_fused_kv_buffer_last_call = False

    def forward_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Tuple[Optional[torch.Tensor], ForwardBatch, Optional[Tuple]]:
        """Prepare phase: QKV projection, QK norm, RoPE."""
        if hidden_states.shape[0] == 0:
            return hidden_states, forward_batch, None

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = self.apply_qk_norm_rope(qkv, positions, forward_batch)

        inner_state = (q, k, v, forward_batch)
        return None, forward_batch, inner_state

    def apply_qk_norm_rope(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        use_fused = self.use_fused_qk_norm_rope and qkv.dtype == torch.bfloat16
        
        if use_fused:
            theta = getattr(self.config, "rope_theta", 10000.0)
            positions = (
                positions.view(-1).to(dtype=torch.int32, device=qkv.device).contiguous()
            )
            factor, low, high, attention_factor = compute_yarn_parameters(self.config)
            
            fused_qk_norm_rope(
                qkv,
                self.num_heads,
                self.num_kv_heads,
                self.num_kv_heads,
                self.head_dim,
                self.q_norm.variance_epsilon,
                self.q_norm.weight,
                self.k_norm.weight,
                theta,
                self.rotary_emb.is_neox_style,
                positions,
                factor,
                low,
                high,
                attention_factor,
            )
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            self._used_fused_qk_norm_rope_last_call = True
            self._used_fused_kv_buffer_last_call = False
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            
            if self.use_qk_norm:
                q, k = apply_qk_norm(
                    q=q,
                    k=k,
                    q_norm=self.q_norm,
                    k_norm=self.k_norm,
                    head_dim=self.head_dim,
                    alt_stream=self.alt_stream,
                )
            
            use_fused_kv = (
                enable_fused_set_kv_buffer(forward_batch)
                and self.compatible_with_fused_kv_buffer
            )
            
            if use_fused_kv:
                q, k = self.rotary_emb(
                    positions,
                    q,
                    k,
                    fused_set_kv_buffer_arg=create_fused_set_kv_buffer_arg(
                        value=v,
                        layer=self.attn,
                        forward_batch=forward_batch,
                    ),
                )
                self._used_fused_kv_buffer_last_call = True
            else:
                q, k = self.rotary_emb(positions, q, k)
                self._used_fused_kv_buffer_last_call = False
            
            self._used_fused_qk_norm_rope_last_call = False
        
        return q, k, v

    def forward_core(
        self,
        intermediate_state: Tuple[Optional[torch.Tensor], ForwardBatch, Optional[Tuple]],
    ) -> torch.Tensor:
        hidden_states, forward_batch, inner_state = intermediate_state

        if inner_state is None:
            return hidden_states

        q, k, v, fb = inner_state

        must_save_kv = self._used_fused_qk_norm_rope_last_call
        save_kv_cache = must_save_kv or not self._used_fused_kv_buffer_last_call

        attn_output = self.attn(q, k, v, fb, save_kv_cache=save_kv_cache)
        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        intermediate_state = self.forward_prepare(positions, hidden_states, forward_batch)
        return self.forward_core(intermediate_state)


class SarvamMoEDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.layer_id = layer_id

        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        self.self_attn = SarvamMoEAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
        )

        first_k_dense = getattr(config, "first_k_dense_replace", 0)
        has_moe = getattr(config, "num_experts", None) is not None
        self.is_layer_sparse = has_moe and layer_id >= first_k_dense
        is_previous_layer_sparse = has_moe and (layer_id - 1) >= first_k_dense if layer_id > 0 else False
        is_next_layer_sparse = has_moe and (layer_id + 1) >= first_k_dense if layer_id < config.num_hidden_layers - 1 else False

        if self.is_layer_sparse:
            self.mlp = SarvamMoESparseMoeBlock(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                alt_stream=alt_stream,
            )
        else:
            if enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = SarvamMoEMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                reduce_results=False,
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=(layer_id == config.num_hidden_layers - 1),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )

        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )

        should_allreduce_fusion = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(forward_batch)
        )

        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(forward_batch)

        hidden_states = self.mlp(
            hidden_states, forward_batch, should_allreduce_fusion, use_reduce_scatter
        )

        if not self.is_layer_sparse and self.attn_tp_size > 1 and not use_reduce_scatter and not should_allreduce_fusion:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)

        if should_allreduce_fusion:
            hidden_states._sglang_needs_allreduce_fusion = True
        else:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )

        return hidden_states, residual


    

class SarvamMoEModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        # Create alt_stream for overlapping computation
        self.alt_stream = torch.cuda.Stream() if _is_cuda else None

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("embed_tokens", prefix),
                enable_tp=not is_dp_attention_enabled(),
            )
        else:
            self.embed_tokens = nn.Identity()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: SarvamMoEDecoderLayer(
                config=config,
                quant_config=quant_config,
                layer_id=idx,
                prefix=prefix,
                alt_stream=self.alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix="model.layers",
        )

        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = nn.Identity()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
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
            hidden_states, residual = layer(positions, hidden_states, forward_batch, residual)

        if not self.pp_group.is_last_rank:
            return PPProxyTensors({"hidden_states": hidden_states, "residual": residual})

        if hidden_states.shape[0] != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class SarvamMoEForCausalLM(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = SarvamMoEModel(config, quant_config, add_prefix("model", prefix))

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> LogitsProcessorOutput:
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
        )

        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        else:
            return hidden_states

    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],  
        input_embeds: torch.Tensor = None,
    ) -> Optional[LogitsProcessorOutput]:
        start, end = split_interval
        
        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
            else:
                forward_batch.hidden_states = input_embeds
            forward_batch.residual = None

        for i in range(start, end):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                layer = self.model.layers[i]
                forward_batch.hidden_states, forward_batch.residual = layer(
                    positions,
                    forward_batch.hidden_states,
                    forward_batch,
                    forward_batch.residual,
                )

        if end == self.model.config.num_hidden_layers:
            if forward_batch.residual is None:
                hidden_states = self.model.norm(forward_batch.hidden_states)
            else:
                hidden_states, _ = self.model.norm(
                    forward_batch.hidden_states, forward_batch.residual
                )
            forward_batch.hidden_states = hidden_states
            
            result = self.logits_processor(
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )
        else:
            result = None

        return result

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=getattr(config, "n_group", None),
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        if not hasattr(self, "_cached_params_dict"):
            self._cached_params_dict = dict(self.named_parameters())
        params_dict = self._cached_params_dict

        loaded_counts = {"qkv": 0, "expert": 0, "stacked": 0, "regular": 0, "skipped": 0}

        for name, loaded_weight in weights:
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            if "rotary_emb.inv_freq" in name:
                loaded_counts["skipped"] += 1
                continue

            name = name.replace(".attention.", ".self_attn.")
            name = name.replace(".dense.", ".o_proj.")
            name = name.replace(".word_embeddings.", ".embed_tokens.")
            name = name.replace(".query_layernorm.", ".q_norm.")
            name = name.replace(".key_layernorm.", ".k_norm.")
            name = name.replace(".gate.expert_bias", ".expert_bias")

            if ".query_key_value." in name:
                name = name.replace(".query_key_value.", ".qkv_proj.")
                if name not in params_dict:
                    logger.warning(f"Parameter {name} not found in params_dict")
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_counts["qkv"] += 1
                continue

            is_stacked = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                loaded_counts["stacked"] += 1
                is_stacked = True
                break

            if is_stacked:
                continue

            is_expert = False
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in name:
                    continue

                is_expert = True
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(
                    param,
                    loaded_weight,
                    name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                loaded_counts["expert"] += 1
                break

            if is_expert:
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue
            if name not in params_dict:
                logger.warning(f"Parameter {name} not found in params_dict")
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_counts["regular"] += 1

        logger.info(
            f"Weight loading summary: QKV={loaded_counts['qkv']}, "
            f"Expert={loaded_counts['expert']}, Stacked={loaded_counts['stacked']}, "
            f"Regular={loaded_counts['regular']}, Skipped={loaded_counts['skipped']}"
        )


EntryClass = [SarvamMoEForCausalLM]