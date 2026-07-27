# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from: https://github.com/vllm-project/vllm/blob/0384aa7150c4c9778efca041ffd1beb3ad2bd694/vllm/model_executor/models/kimi_linear.py

import logging
from collections.abc import Iterable
from copy import deepcopy
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import PytorchGELUTanh

from sglang.srt.configs.kimi_k3 import KimiK3Config, KimiK3VisionConfig
from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.distributed import (
    attention_tensor_model_parallel_all_reduce,
    divide,
    get_pp_group,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.attention.fla.fused_norm_gate import FusedRMSNormGated
from sglang.srt.layers.attention.fla.kda import fused_kda_gate
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.communicator import (
    AttentionInputs,
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
    get_attn_tp_context,
)
from sglang.srt.layers.conv import Conv2dLayer
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather_into_tensor,
    attn_tp_reduce_scatter_tensor,
    get_attention_tp_group,
    get_local_dp_buffer,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelBatchedLinear,
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    MergedColumnParallelRepeatedLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    get_embedding_tp_kwargs,
)
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
from sglang.srt.models.kimi_k3_vision_utils import (
    KimiK3Learnable2DInterpPosEmb,
    KimiK3Rope2DPosEmbRepeated,
    apply_rope,
    tpool_patch_merger,
)
from sglang.srt.models.kimi_vl_moonvit import MLP2
from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model
from sglang.srt.models.transformers import maybe_prefix
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import is_npu, make_layers
from sglang.srt.utils.common import BumpAllocator, add_prefix, set_weight_attrs
from sglang.srt.hardware_backend.npu.utils import situ_and_mul, apply_attn_res

logger = logging.getLogger(__name__)


class _NoopRotaryEmbedding(nn.Module):
    """Preserve Kimi MLA's skip_rope semantics on backends that call RoPE."""

    def forward(self, positions, query, key):
        return query, key


class KimiMLAAttention(DeepseekV2AttentionMLA):
    """DeepSeek MLA with K3's optional per-head output gate."""

    def __init__(self, *args, config: KimiLinearConfig, prefix: str = "", **kwargs):
        super().__init__(*args, config=config, prefix=prefix, **kwargs)
        # o_proj does not all-reduce; the caller (KimiDecoderLayer) handles
        # the reduce at the correct point for either the attn_residual path
        # (manual attention_tensor_model_parallel_all_reduce) or the standard
        # LayerCommunicator path.
        self.o_proj.reduce_results = False
        self.o_proj.use_dp_attention_reduce = False
        if is_npu() and self.rotary_emb is None:
            self.rotary_emb = _NoopRotaryEmbedding()
        self.use_output_gate = getattr(config, "mla_use_output_gate", False)
        self._output_gate = None
        if self.use_output_gate:
            self.g_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.v_head_dim,
                bias=False,
                quant_config=self.quant_config,
                prefix=add_prefix("g_proj", prefix),
                tp_rank=get_parallel().attn_tp_rank,
                tp_size=get_parallel().attn_tp_size,
            )

            # All MLA backends funnel their local head output through o_proj.
            # Wrapping that call preserves the optimized NPU attention path and
            # applies K3's gate at the exact point required by the reference.
            ungated_o_proj_forward = self.o_proj.forward

            def gated_o_proj_forward(x: torch.Tensor):
                assert self._output_gate is not None
                return ungated_o_proj_forward(x * self._output_gate)

            self.o_proj.forward = gated_o_proj_forward

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        **kwargs,
    ):
        if hidden_states.shape[0] == 0:
            return hidden_states

        if self.use_output_gate:
            self._output_gate = torch.sigmoid(self.g_proj(hidden_states)[0])

        # DeepSeek MLA's NPU prepare path consumes its q/kv latent projection
        # through the attention TP context. K3 does not use DeepSeek's decoder
        # LayerCommunicator, so publish the input explicitly (the same pattern
        # used by the standalone Kimi K2.5 Eagle MLA layer).
        attn_tp_context = get_attn_tp_context()
        attn_tp_context.set_attn_inputs(
            AttentionInputs(hidden_states, forward_batch, self.prepare_qkv_latent)
        )
        try:
            return super().forward(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
                **kwargs,
            )
        finally:
            attn_tp_context.clear_attn_inputs()


class KimiMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        reduce_results: bool = True,
        activation_situ_beta: float = 1.0,
        activation_situ_linear_beta: Optional[float] = None,
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        if tp_rank is None or tp_size is None:
            if is_dp_attention_enabled():
                tp_rank = get_parallel().attn_tp_rank
                tp_size = get_parallel().attn_tp_size
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
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
            reduce_results=False,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if hidden_act == "situ":
            from sglang.srt.layers.activation import SituAndMul

            # self.act_fn = SituAndMul(
            #     beta=activation_situ_beta,
            #     linear_beta=activation_situ_linear_beta,
            # )
            self.act_fn = situ_and_mul
        elif hidden_act == "silu":
            from sglang.srt.layers.activation import SiluAndMul

            self.act_fn = SiluAndMul()
        else:
            raise ValueError(f"Unsupported activation: {hidden_act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return x
        gate_up, _ = self.gate_up_proj(x)
        return self.down_proj(self.act_fn(gate_up))[0]


class KimiMoE(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        layer_idx: int = 0,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        moe_intermediate_size = config.moe_intermediate_size
        num_experts = config.num_experts
        moe_renormalize = config.moe_renormalize
        self.tp_size = get_parallel().tp_size
        self.routed_scaling_factor = config.routed_scaling_factor
        self.num_shared_experts = config.num_shared_experts
        self.layer_idx = layer_idx
        self.alt_stream = alt_stream
        self.alt_stream = None

        if config.hidden_act not in {"silu", "situ"}:
            raise ValueError(f"Unsupported activation: {config.hidden_act}")

        # Gate always runs at half / full precision for now.
        self.gate = ReplicatedLinear(
            hidden_size,
            num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        self.gate.e_score_correction_bias = nn.Parameter(torch.empty(num_experts))

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_token,
            hidden_size=getattr(
                config, "routed_expert_hidden_size", config.hidden_size
            ),
            intermediate_size=config.moe_intermediate_size,
            layer_id=self.layer_idx,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            activation=config.hidden_act,
            prefix=add_prefix("experts", prefix),
            activation_situ_beta=getattr(config, "activation_situ_beta", 1.0),
            activation_situ_linear_beta=getattr(
                config, "activation_situ_linear_beta", None
            ),
        )

        self.routed_expert_hidden_size = getattr(
            config, "routed_expert_hidden_size", None
        )
        if self.routed_expert_hidden_size is not None:
            self.routed_expert_down_proj = ReplicatedLinear(
                config.hidden_size,
                self.routed_expert_hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("routed_expert_down_proj", prefix),
            )
            self.routed_expert_up_proj = ReplicatedLinear(
                self.routed_expert_hidden_size,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("routed_expert_up_proj", prefix),
            )
            self.routed_expert_norm = RMSNorm(
                self.routed_expert_hidden_size, eps=config.rms_norm_eps
            )

        self.topk = TopK(
            top_k=config.num_experts_per_token,
            renormalize=False,
            use_grouped_topk=True,
            num_expert_group=config.num_expert_group,
            topk_group=config.topk_group,
            correction_bias=self.gate.e_score_correction_bias,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=True,
            # Some Fp4 MoE backends require the output format to be bypassed but the MTP layers are unquantized
            # and requires the output format to be standard. We use quant_config to determine the output format.
            output_format=TopKOutputFormat.STANDARD if quant_config is None else None,
        )

        if self.num_shared_experts is not None:
            intermediate_size = moe_intermediate_size * self.num_shared_experts
            self.shared_experts = KimiMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
                activation_situ_beta=getattr(config, "activation_situ_beta", 1.0),
                activation_situ_linear_beta=getattr(
                    config, "activation_situ_linear_beta", None
                ),
            )

    def _forward_shared_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the TP-sharded shared MLP while keeping MoE tokens scattered."""
        if not (is_dp_attention_enabled() and get_parallel().attn_tp_size > 1):
            return self.shared_experts(hidden_states)

        gathered_hidden_states = get_local_dp_buffer(get_attention_tp_group())
        attn_tp_all_gather_into_tensor(gathered_hidden_states, hidden_states)

        gathered_shared_output = self.shared_experts(gathered_hidden_states)
        shared_output = torch.empty_like(hidden_states)
        attn_tp_reduce_scatter_tensor(shared_output, gathered_shared_output)
        return shared_output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)

        shared_output = None
        routed_hidden_states = hidden_states
        if self.routed_expert_hidden_size is not None and num_tokens > 0:
            routed_hidden_states = self.routed_expert_down_proj(hidden_states)[0]
        elif self.routed_expert_hidden_size is not None:
            routed_hidden_states = hidden_states.new_empty(
                (0, self.routed_expert_hidden_size)
            )

        if (
            self.alt_stream is not None
            and self.num_shared_experts is not None
            and hidden_states.shape[0] > 0
            and get_is_capture_mode()
        ):
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)

            shared_output = self._forward_shared_experts(hidden_states.clone())

            with torch.cuda.stream(self.alt_stream):
                router_logits, _ = self.gate(hidden_states)
                topk_output = self.topk(hidden_states, router_logits)
                final_hidden_states = self.experts(routed_hidden_states, topk_output)

            current_stream.wait_stream(self.alt_stream)
        else:
            if self.num_shared_experts is not None and hidden_states.shape[0] > 0:
                shared_output = self._forward_shared_experts(hidden_states)
            if num_tokens > 0:
                router_logits, _ = self.gate(hidden_states)
                topk_output = self.topk(hidden_states, router_logits)
            else:
                topk_output = self.topk.empty_topk_output(
                    hidden_states.device, layer_id=self.layer_idx
                )
            final_hidden_states = self.experts(routed_hidden_states, topk_output)

        if self.routed_expert_hidden_size is not None and num_tokens > 0:
            final_hidden_states = self.routed_expert_norm(final_hidden_states)
            final_hidden_states = self.routed_expert_up_proj(final_hidden_states)[0]
        elif self.routed_expert_hidden_size is not None:
            final_hidden_states = hidden_states.new_empty((0, hidden_size))

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
        if self.tp_size > 1 and not is_dp_attention_enabled():
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states.view(num_tokens, hidden_size)


class KimiDeltaAttention(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        config: KimiLinearConfig,
        quant_config: Optional[QuantizationConfig] = None,
        rms_norm_eps: float = 1e-5,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.attn_tp_size = get_parallel().attn_tp_size
        self.attn_tp_rank = get_parallel().attn_tp_rank
        self.hidden_size = hidden_size
        self.config = config
        self.head_dim = config.linear_attn_config["head_dim"]
        self.num_heads = config.linear_attn_config["num_heads"]
        self.num_k_heads = config.linear_attn_config["num_heads"]
        self.num_v_heads = config.linear_attn_config["num_heads"]
        self.head_k_dim = config.linear_attn_config["head_dim"]
        self.head_v_dim = config.v_head_dim
        self.layer_idx = layer_idx
        self.prefix = prefix
        self.alt_stream = alt_stream
        assert self.num_heads % self.attn_tp_size == 0
        self.local_num_heads = divide(self.num_heads, self.attn_tp_size)

        projection_size = self.head_dim * self.num_heads
        self.conv_size = config.linear_attn_config["short_conv_kernel_size"]
        self.use_full_rank_gate = config.linear_attn_config.get(
            "use_full_rank_gate", False
        )

        # TODO: support fusion with quant
        self.do_fuse_qkvbfg = quant_config is None and not self.use_full_rank_gate

        if self.do_fuse_qkvbfg:
            # Fuse: q, k, v, beta (column parallel) + f_a, g_a (replicated)
            self.qkvb_sizes = [
                projection_size,
                projection_size,
                projection_size,
                self.num_heads,
            ]
            self.fg_sizes = [self.head_dim, self.head_dim]

            self.fused_qkvbfg_a_proj = MergedColumnParallelRepeatedLinear(
                self.hidden_size,
                self.qkvb_sizes,  # Column parallel
                self.fg_sizes,  # Replicated: f_a, g_a
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkvbfg_a_proj",
            )
            self.split_sizes = [
                3 * projection_size // self.attn_tp_size,  # qkv
                self.num_heads // self.attn_tp_size,  # beta
                2 * self.head_dim,  # f_a, g_a
            ]
            self.fused_fg_b_proj = ColumnParallelBatchedLinear(
                2, self.head_dim, projection_size, dtype=config.dtype
            )
        else:
            # Unfused path: separate QKVParallelLinear
            self.qkv_proj = QKVParallelLinear(
                self.hidden_size,
                self.head_dim,
                self.num_heads,
                self.num_k_heads,
                bias=False,
                quant_config=quant_config,
                tp_rank=self.attn_tp_rank,
                tp_size=self.attn_tp_size,
                v_head_size=self.head_v_dim,
                prefix=f"{prefix}.qkv_proj",
            )

            self.f_a_proj = ReplicatedLinear(
                self.hidden_size,
                self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.f_a_proj",
            )

            self.f_b_proj = ColumnParallelLinear(
                self.head_dim,
                projection_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.f_b_proj",
                tp_rank=self.attn_tp_rank,
                tp_size=self.attn_tp_size,
            )

            self.b_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.b_proj",
                tp_rank=self.attn_tp_rank,
                tp_size=self.attn_tp_size,
            )

            if self.use_full_rank_gate:
                self.g_proj = ColumnParallelLinear(
                    self.hidden_size,
                    projection_size,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.g_proj",
                    tp_rank=self.attn_tp_rank,
                    tp_size=self.attn_tp_size,
                )
            else:
                self.g_a_proj = ReplicatedLinear(
                    self.hidden_size,
                    self.head_dim,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.g_a_proj",
                )
                self.g_b_proj = ColumnParallelLinear(
                    self.head_dim,
                    projection_size,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.g_b_proj",
                    tp_rank=self.attn_tp_rank,
                    tp_size=self.attn_tp_size,
                )

        self.dt_bias = nn.Parameter(
            torch.empty(
                divide(projection_size, self.attn_tp_size), dtype=torch.float32
            )
        )

        def load_attn_shard(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
            loaded_weight = loaded_weight.flatten()
            shard_size = param.numel()
            start = self.attn_tp_rank * shard_size
            param.data.copy_(loaded_weight.narrow(0, start, shard_size).view_as(param))

        set_weight_attrs(self.dt_bias, {"weight_loader": load_attn_shard})

        self.qkv_conv1d = MergedColumnParallelLinear(
            input_size=self.conv_size,
            output_sizes=[projection_size, projection_size, projection_size],
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.qkv_conv1d",
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.qkv_conv1d.weight.data = self.qkv_conv1d.weight.data.unsqueeze(1)

        self.A_log = nn.Parameter(
            torch.empty(1, 1, self.local_num_heads, 1, dtype=torch.float32)
        )

        def load_a_log(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
            # K3 stores A_log as [num_heads], while RadixLinearAttention keeps
            # the local shard as [1, 1, local_num_heads, 1].
            loaded_weight = loaded_weight.flatten()
            start = self.attn_tp_rank * self.local_num_heads
            local_weight = loaded_weight.narrow(0, start, self.local_num_heads)
            param.data.copy_(local_weight.reshape_as(param))

        set_weight_attrs(self.A_log, {"weight_loader": load_a_log})

        self.o_norm = FusedRMSNormGated(
            self.head_dim, eps=rms_norm_eps, activation="sigmoid"
        )
        self.o_proj = RowParallelLinear(
            projection_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            reduce_results=False,
        )

        conv_weights = self.qkv_conv1d.weight.squeeze(1)
        bias = self.qkv_conv1d.bias

        self.attn = RadixLinearAttention(
            layer_id=self.layer_idx,
            num_q_heads=self.num_k_heads // self.attn_tp_size,
            num_k_heads=self.num_k_heads // self.attn_tp_size,
            num_v_heads=self.num_v_heads // self.attn_tp_size,
            head_q_dim=self.head_k_dim,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            conv_weights=conv_weights,
            bias=bias,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
        )

    def forward_qkvbfg(self, hidden_states: torch.Tensor):
        qkv, _ = self.qkv_proj(hidden_states)

        # Compute beta, forget_gate, and g_proj_states
        beta = self.b_proj(hidden_states)[0]
        forget_gate = self.f_b_proj(self.f_a_proj(hidden_states)[0])[0]
        if self.alt_stream is not None:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            with torch.cuda.stream(self.alt_stream):
                if self.use_full_rank_gate:
                    g_proj_states = self.g_proj(hidden_states)[0]
                else:
                    g_proj_states = self.g_b_proj(self.g_a_proj(hidden_states)[0])[0]
        else:
            if self.use_full_rank_gate:
                g_proj_states = self.g_proj(hidden_states)[0]
            else:
                g_proj_states = self.g_b_proj(self.g_a_proj(hidden_states)[0])[0]

        return (
            qkv,
            beta,
            forget_gate,
            g_proj_states,
        )

    def forward_qkvbfg_fused(self, hidden_states: torch.Tensor):
        # Single fused projection for all: qkv + beta + f_a + g_a
        fused_states = self.fused_qkvbfg_a_proj(hidden_states)

        qkv, beta, fg_a_states = torch.split(
            fused_states,
            self.split_sizes,
            dim=-1,
        )

        # use batch matmul to calculate forget_gate and g_proj_states
        forget_gate, g_proj_states = self.fused_fg_b_proj(
            fg_a_states.view(-1, 2, self.head_dim).transpose(0, 1)
        )

        return (
            qkv,
            beta,
            forget_gate,
            g_proj_states,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ) -> None:
        if hidden_states.shape[0] == 0:
            return hidden_states

        if self.do_fuse_qkvbfg:
            mixed_qkv, beta, forget_gate, g_proj_states = self.forward_qkvbfg_fused(
                hidden_states
            )
        else:
            mixed_qkv, beta, forget_gate, g_proj_states = self.forward_qkvbfg(
                hidden_states
            )

        # For prefill: raw gate is passed to chunk_kda_fwd, which fuses gate
        # activation with chunk_local_cumsum (kda_gate_chunk_cumsum kernel).
        # For decode: gate activation is handled inside fused_recurrent kernel.
        beta = beta.float()
        if not forward_batch.forward_mode.is_decode():
            forget_gate = fused_kda_gate(
                forget_gate, self.A_log, self.head_dim, g_bias=self.dt_bias
            )
            beta = beta.sigmoid()
            forget_gate = forget_gate.unsqueeze(0)
        beta = beta.unsqueeze(0)

        core_attn_out = self.attn(
            forward_batch,
            mixed_qkv=mixed_qkv,
            a=forget_gate,
            b=beta,
        )
        if self.alt_stream is not None:
            current_stream = torch.cuda.current_stream()
            current_stream.wait_stream(self.alt_stream)
        norm_gate = g_proj_states.unflatten(
            -1, (-1, self.head_dim)
        )  # ... (h d) -> ... h d
        core_attn_out = self.o_norm(core_attn_out, norm_gate)
        core_attn_out = core_attn_out.squeeze(0).flatten(-2)  # 1 n h d -> n (h d)

        return self.o_proj(core_attn_out)[0]


class KimiDecoderLayer(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.alt_stream = alt_stream
        self.layer_idx = layer_idx

        self.is_moe = config.is_moe
        self.attn_tp_size = get_parallel().attn_tp_size
        self.attn_tp_rank = get_parallel().attn_tp_rank

        if config.is_kda_layer(layer_idx):
            self.self_attn = KimiDeltaAttention(
                layer_idx=layer_idx,
                hidden_size=config.hidden_size,
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                alt_stream=self.alt_stream,
            )
        else:
            self.self_attn = KimiMLAAttention(
                layer_id=layer_idx,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                config=config,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                skip_rope=True,
            )

        if (
            self.is_moe
            and config.num_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            self.block_sparse_moe = KimiMoE(
                config=config,
                quant_config=quant_config,
                layer_idx=layer_idx,
                prefix=f"{prefix}.block_sparse_moe",
                alt_stream=self.alt_stream,
            )
            self.mlp = self.block_sparse_moe
        else:
            self.mlp = KimiMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                activation_situ_beta=getattr(config, "activation_situ_beta", 1.0),
                activation_situ_linear_beta=getattr(
                    config, "activation_situ_linear_beta", None
                ),
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # K3 mixes the running prefix and one residual per attention block with
        # learned scalar projections.  These parameters exist in every K3
        # checkpoint and are not the ordinary two-add Transformer residuals.
        self.use_attn_residuals = (
            getattr(config, "attn_res_block_size", None) is not None
        )
        if self.use_attn_residuals:
            self.attn_res_block_size = config.attn_res_block_size
            self.self_attention_res_norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.mlp_res_norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.self_attention_res_proj = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attention_res_proj",
            )
            self.mlp_res_proj = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp_res_proj",
            )

        # dp-attention
        is_layer_sparse = isinstance(self.mlp, KimiMoE)
        is_previous_layer_sparse = (
            layer_idx > 0
            and config.is_moe
            and config.num_experts is not None
            and (layer_idx - 1) >= config.first_k_dense_replace
            and (layer_idx - 1) % config.moe_layer_freq == 0
        )
        is_next_layer_sparse = (
            (layer_idx + 1) < config.num_hidden_layers
            and config.is_moe
            and config.num_experts is not None
            and (layer_idx + 1) >= config.first_k_dense_replace
            and (layer_idx + 1) % config.moe_layer_freq == 0
        )
        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_idx,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )
        # 第一层是输入是TP-ATTN-FULL
        if layer_idx == 0 and is_dp_attention_enabled() and get_parallel().attn_tp_size > 1:
            self.layer_scatter_modes.layer_input_mode = ScatterMode.TP_ATTN_FULL
        else:
            self.layer_scatter_modes.layer_input_mode = ScatterMode.SCATTERED
        self.layer_scatter_modes.mlp_mode = ScatterMode.SCATTERED
        self.layer_scatter_modes.middle_residual_mode = ScatterMode.SCATTERED
        if layer_idx != config.num_hidden_layers - 1:
            self.layer_scatter_modes.layer_output_mode = ScatterMode.SCATTERED

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=(layer_idx == config.num_hidden_layers - 1),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_attn_residuals:
            return self._forward_attn_residual(
                positions,
                hidden_states,
                forward_batch,
                residual,
                zero_allocator,
            )

        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
        )
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )
        return hidden_states, residual

    def _forward_attn_residual(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        block_residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_sum = hidden_states
        if block_residual is None:
            block_residual = hidden_states.new_empty(
                hidden_states.shape[0], 0, hidden_states.shape[1]
            )

        if block_residual.shape[1] > 0:
            hidden_states = apply_attn_res(
                prefix_sum,
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
            )

        if self.layer_idx % self.attn_res_block_size == 0:
            block_residual = torch.cat(
                (block_residual, prefix_sum.unsqueeze(1)), dim=1
            )
            if self.layer_idx == 0 and is_dp_attention_enabled() and self.attn_tp_size > 1:
                block_residual = block_residual.tensor_split(self.attn_tp_size)[self.attn_tp_rank]
            prefix_sum = None

        # Reuse framework: prepare_attn (input_layernorm + AllGather SCATTERED→TP_ATTN_FULL)
        hidden_states, _ = self.layer_communicator.prepare_attn(
            hidden_states, None, forward_batch
        )

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
        )

        # Reuse framework: prepare_mlp (reduce_scatter TP_ATTN_FULL→SCATTERED, skip layernorm)
        hidden_states, _ = self.layer_communicator.prepare_mlp(
            hidden_states, None, forward_batch, skip_layernorm=True
        )

        # K3-specific: prefix_sum accumulation + apply_attn_res (SCATTERED, token-local)
        prefix_sum = hidden_states if prefix_sum is None else prefix_sum + hidden_states

        hidden_states = apply_attn_res(
            prefix_sum,
            block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP/MoE (SCATTERED)
        if isinstance(self.mlp, KimiMoE):
            hidden_states = self.mlp(hidden_states)
        else:
            # Dense MLP: needs TP_ATTN_FULL
            if self.attn_tp_size > 1 and is_dp_attention_enabled():
                hidden_states_full = get_local_dp_buffer(
                    get_attention_tp_group()
                )
                attn_tp_all_gather_into_tensor(hidden_states_full, hidden_states)
                hidden_states_full = self.mlp(hidden_states_full)
                hidden_states_full = (
                    attention_tensor_model_parallel_all_reduce(hidden_states_full)
                )
                hidden_states = hidden_states_full.tensor_split(self.attn_tp_size)[self.attn_tp_rank]
            else:
                hidden_states = self.mlp(hidden_states)

        prefix_sum = prefix_sum + hidden_states
        return prefix_sum, block_residual


def _apply_attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Module,
    norm: RMSNorm,
) -> torch.Tensor:
    """Apply K3's learned softmax mixing over block residual streams."""
    if prefix_sum.shape[0] == 0:
        return prefix_sum

    score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
    output = torch.empty_like(prefix_sum)

    # Attention residuals grow to several hidden-size streams.  Materializing
    # all fp32 normalized streams at once costs multiple GiB for long prefills.
    # The mixing is token-local, so compute it in bounded token chunks and fold
    # the RMS scale into the score after the hidden-dimension reduction.
    token_chunk_size = 256
    for start in range(0, prefix_sum.shape[0], token_chunk_size):
        end = min(start + token_chunk_size, prefix_sum.shape[0])
        values = torch.cat(
            (
                block_residual[start:end],
                prefix_sum[start:end].unsqueeze(1),
            ),
            dim=1,
        )
        values_float = values.float()
        variance = values_float.square().mean(-1, keepdim=True)
        inv_rms = torch.rsqrt(variance + norm.variance_epsilon)
        scores = torch.matmul(values_float, score_weight) * inv_rms.squeeze(-1)
        probabilities = scores.softmax(-1).unsqueeze(1)
        output[start:end].copy_(
            torch.matmul(probabilities, values_float).squeeze(1).to(values.dtype)
        )

    return output


class KimiLinearModel(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        self.config = config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()
        self.dspark_layers_to_capture: Optional[list[int]] = None

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
                **get_embedding_tp_kwargs(),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.alt_stream = torch.cuda.Stream()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: KimiDecoderLayer(
                layer_idx=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=self.alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=f"{prefix}.layers",
        )

        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.use_attn_residuals = (
                getattr(config, "attn_res_block_size", None) is not None
            )
            if self.use_attn_residuals:
                self.output_attn_res_norm = RMSNorm(
                    config.hidden_size, eps=config.rms_norm_eps
                )
                self.output_attn_res_proj = ReplicatedLinear(
                    config.hidden_size,
                    1,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.output_attn_res_proj",
                )
        else:
            self.norm = PPMissingLayer()
            self.use_attn_residuals = False

        attn_tp_size = get_parallel().attn_tp_size
        assert (
            config.num_attention_heads % attn_tp_size == 0
        ), "num_attention_heads must be divisible by attn_tp_size"

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: torch.Tensor | None = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual = None
            if self.use_attn_residuals:
                residual = hidden_states.new_empty(
                    hidden_states.shape[0], 0, hidden_states.shape[1]
                )
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        total_num_layers = self.end_layer - self.start_layer
        device = hidden_states.device
        zero_allocator = BumpAllocator(
            buffer_size=total_num_layers * 2,
            dtype=torch.float32,
            device=device,
        )
        # TODO: capture aux hidden states
        aux_hidden_states = []
        for i in range(self.start_layer, self.end_layer):
            ctx = get_global_expert_distribution_recorder().with_current_layer(i)
            with ctx:
                layer = self.layers[i]
                hidden_states, residual = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                    residual=residual,
                    zero_allocator=zero_allocator,
                )
            if (
                self.dspark_layers_to_capture is not None
                and i in self.dspark_layers_to_capture
            ):
                captured_hidden = self._dspark_capture_stream(
                    i, hidden_states, residual
                )
                if is_dp_attention_enabled() and get_parallel().attn_tp_size > 1:
                    attn_tp_size = get_parallel().attn_tp_size
                    expected_num_tokens = captured_hidden.shape[0] * attn_tp_size
                    if expected_num_tokens != forward_batch.input_ids.shape[0]:
                        raise ValueError(
                            "DSpark aux hidden token mismatch before attention-TP "
                            f"allgather: local={captured_hidden.shape[0]}, "
                            f"attn_tp_size={attn_tp_size}, "
                            f"expected_full={expected_num_tokens}, "
                            f"forward_tokens={forward_batch.input_ids.shape[0]}"
                        )
                    gathered_hidden = captured_hidden.new_empty(
                        (expected_num_tokens, *captured_hidden.shape[1:])
                    )
                    attn_tp_all_gather_into_tensor(gathered_hidden, captured_hidden)
                    captured_hidden = gathered_hidden
                aux_hidden_states.append(captured_hidden)

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            if hidden_states.shape[0] != 0:
                if self.use_attn_residuals:
                    hidden_states = apply_attn_res(
                        hidden_states,
                        residual,
                        self.output_attn_res_proj,
                        self.output_attn_res_norm,
                    )
                    hidden_states = self.norm(hidden_states)
                elif residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)
                # when dp-attention and attn-tp != 1, the output is Scattered,
                # should do AllGather to make it TP-ATTN-FULL for final output
                if is_dp_attention_enabled() and get_parallel().attn_tp_size > 1:
                    gathered = get_local_dp_buffer(
                        get_attention_tp_group()
                    )
                    attn_tp_all_gather_into_tensor(gathered, hidden_states)
                    hidden_states = gathered
        if self.dspark_layers_to_capture is not None:
            return hidden_states, aux_hidden_states
        if len(aux_hidden_states) == 0:
            return hidden_states

    def _dspark_capture_stream(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return the pre-norm stream consumed after ``layer_idx``."""
        if not self.use_attn_residuals:
            return hidden_states if residual is None else hidden_states + residual
        if layer_idx + 1 < self.end_layer:
            next_layer = self.layers[layer_idx + 1]
            return apply_attn_res(
                hidden_states,
                residual,
                next_layer.self_attention_res_proj,
                next_layer.self_attention_res_norm,
            )
        return apply_attn_res(
            hidden_states,
            residual,
            self.output_attn_res_proj,
            self.output_attn_res_norm,
        )


class KimiK3ForCausalLM(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = KimiLinearModel(
            config, quant_config, prefix=maybe_prefix(prefix, "model")
        )
        self.pp_group = get_pp_group()
        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
                use_attn_tp_group=is_dp_attention_enabled(),
            )
        else:
            self.lm_head = PPMissingLayer()
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(config=config, logit_scale=logit_scale)
        self.capture_aux_hidden_states = False

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_dspark_layers_to_capture(self, layer_ids: list[int]) -> None:
        if self.pp_group.world_size > 1:
            raise NotImplementedError("DSPARK aux hidden capture requires PP=1.")
        if not self.pp_group.is_last_rank:
            return
        if layer_ids is None:
            raise ValueError("DSPARK requires explicit layer_ids for aux hidden capture.")
        self.capture_aux_hidden_states = True
        self.model.dspark_layers_to_capture = list(layer_ids)

    @property
    def start_layer(self) -> int:
        return self.model.start_layer

    @property
    def end_layer(self) -> int:
        return self.model.end_layer

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        mm_input_embeds = getattr(forward_batch, "mm_input_embeds", None)

        if mm_input_embeds is not None:
            logger.warning(
                "[K3_CAUSAL_MM_FORWARD] mode=%s, "
                "input_embeds_shape=%s, mm_input_embeds_shape=%s, same_object=%s",
                forward_batch.forward_mode,
                tuple(input_embeds.shape) if input_embeds is not None else None,
                tuple(mm_input_embeds.shape),
                input_embeds is mm_input_embeds,
            )
        if input_embeds is None:
            input_embeds = inputs_embeds
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors,
        )
        if self.pp_group.is_last_rank:
            aux_hidden_states = None
            if self.capture_aux_hidden_states:
                hidden_states, aux_hidden_states = hidden_states
            return self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                forward_batch,
                aux_hidden_states,
            )
        else:
            return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        # Kimi-K3 is distributed as a multimodal checkpoint.  This class only
        # owns the text model, whose tensors are nested below ``language_model``
        # in the checkpoint but live at the top level in this module.
        language_model_prefix = "language_model."
        multimodal_prefixes = ("vision_tower.", "mm_projector.")

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            # Fused path
            (".fused_qkvbfg_a_proj", ".q_proj", 0),
            (".fused_qkvbfg_a_proj", ".k_proj", 1),
            (".fused_qkvbfg_a_proj", ".v_proj", 2),
            (".fused_qkvbfg_a_proj", ".b_proj", 3),
            (".fused_qkvbfg_a_proj", ".f_a_proj", 4),
            (".fused_qkvbfg_a_proj", ".g_a_proj", 5),
            (".fused_fg_b_proj", ".f_b_proj", 0),
            (".fused_fg_b_proj", ".g_b_proj", 1),
            # Unfused path: separate qkv_proj (when do_fuse_qkvbfg=False)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            # qkv conv fuse
            (".qkv_conv1d", ".q_conv1d", 0),
            (".qkv_conv1d", ".k_conv1d", 1),
            (".qkv_conv1d", ".v_conv1d", 2),
        ]
        if self.config.is_moe:
            # Params for weights, fp8 weight scales, fp8 activation scales
            # (param_name, weight_name, expert_id, shard_id)
            expert_params_mapping = FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=self.config.num_experts,
            )
        else:
            expert_params_mapping = []
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for args in weights:
            name, loaded_weight = args[:2]
            kwargs = args[2] if len(args) > 2 else {}

            if name.startswith(language_model_prefix):
                name = name[len(language_model_prefix) :]
            elif name.startswith(multimodal_prefixes):
                # Vision weights are consumed by the multimodal wrapper.  The
                # language-only SRT implementation must not try to load them.
                continue

            if name.startswith("model.layers."):
                layer_id = int(name.split(".")[2])
                if layer_id >= self.config.num_hidden_layers:
                    continue

            if "rotary_emb.inv_freq" in name:
                continue

            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue

            # DeepSeek MLA fuses q_a and kv_a into one replicated projection,
            # while K3 checkpoints keep the two FLOAT tensors separate.
            if ".self_attn.q_a_proj." in name or ".self_attn.kv_a_proj_with_mqa." in name:
                layer_id = int(name.split(".")[2])
                if not self.config.is_kda_layer(layer_id):
                    if ".self_attn.q_a_proj." in name:
                        fused_name = name.replace(
                            ".self_attn.q_a_proj.",
                            ".self_attn.fused_qkv_a_proj_with_mqa.",
                        )
                        output_offset = 0
                    else:
                        fused_name = name.replace(
                            ".self_attn.kv_a_proj_with_mqa.",
                            ".self_attn.fused_qkv_a_proj_with_mqa.",
                        )
                        output_offset = self.config.q_lora_rank
                    param = params_dict[fused_name]
                    param.data.narrow(
                        0, output_offset, loaded_weight.shape[0]
                    ).copy_(loaded_weight)
                    loaded_params.add(fused_name)
                    continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if (".experts." in name) and name not in params_dict:
                    continue
                # Check if this mapping targets a fused projection (only apply fusion check to fused params)
                if param_name in {".fused_qkvbfg_a_proj", ".fused_fg_b_proj"}:
                    layer_id = int(name.split(".")[2])
                    if not self.config.is_kda_layer(layer_id):
                        continue
                    layer = self.model.layers[layer_id].self_attn
                    # Only load to fused projection if fusion is enabled
                    if not getattr(layer, "do_fuse_qkvbfg", False):
                        continue
                if weight_name in {".q_proj", ".k_proj", ".v_proj"}:
                    layer_id = int(name.split(".")[2])
                    if not self.config.is_kda_layer(layer_id):
                        continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # if is_pp_missing_parameter(name, self):
                #     continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for idx, (param_name, weight_name, expert_id, shard_id) in enumerate(
                    expert_params_mapping
                ):
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # if is_pp_missing_parameter(name, self):
                    #     continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        expert_id=expert_id,
                        shard_id=shard_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias")
                        and name not in params_dict
                        and not self.config.is_linear_attn
                    ):  # noqa: E501
                        continue
                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                    # if is_pp_missing_parameter(name, self):
                    #     continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, **kwargs)
            loaded_params.add(name)

        full_attention_layer_ids = [
            layer_id
            for layer_id in range(self.config.num_hidden_layers)
            if not self.config.is_kda_layer(layer_id)
        ]
        for layer_id in full_attention_layer_ids:
            self_attn = self.model.layers[layer_id].self_attn
            w_kc, w_vc = self_attn.kv_b_proj.weight.unflatten(
                0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
            ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
            self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
            self_attn.w_vc = w_vc.contiguous().transpose(1, 2)
            if hasattr(self_attn.kv_b_proj, "weight_scale"):
                self_attn.w_scale = self_attn.kv_b_proj.weight_scale


class KimiK3VisionEncoderLayer(nn.Module):
    """K3 MoonViT block with 1024-wide residuals and 1536-wide QKV."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        qkv_hidden_size: int,
        mlp_dim: int,
        *,
        activation=F.gelu,
        attn_bias: bool = False,
        linear_bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.norm0 = nn.RMSNorm(hidden_dim)
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.mlp = MLP2(
            [hidden_dim, mlp_dim, hidden_dim],
            activation,
            bias=linear_bias,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.attn = VisionAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            projection_size=qkv_hidden_size,
            head_dim=qkv_hidden_size // num_heads,
            use_qkv_parallel=True,
            qkv_bias=attn_bias,
            proj_bias=attn_bias,
            flatten_batch=True,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            use_data_parallel=use_data_parallel,
            customized_position_embedding_applier=apply_rope,
            use_dp_attention_reduce=is_dp_attention_enabled(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.attn(
            self.norm0(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=rope_freqs_cis,
        )
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(self.norm1(hidden_states))


class KimiK3VisionPatchEmbed(nn.Module):
    def __init__(
        self,
        out_dim: int,
        patch_size: int | Sequence = 14,
        pos_emb_height: int = 64,
        pos_emb_width: int = 64,
        pos_emb_time: int = 4,
        interpolation_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = tuple(patch_size)
        self.proj = Conv2dLayer(
            3,
            out_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        self.pos_emb = KimiK3Learnable2DInterpPosEmb(
            height=pos_emb_height,
            width=pos_emb_width,
            num_frames=pos_emb_time,
            dim=out_dim,
            interpolation_mode=interpolation_mode,
        )

    def forward(
        self, pixel_values: torch.Tensor, grid_thws: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.proj(pixel_values).view(pixel_values.size(0), -1)
        return self.pos_emb(hidden_states, grid_thws)


class KimiK3VisionEncoder(nn.Module):
    def __init__(
        self,
        config: KimiK3VisionConfig,
        *,
        use_data_parallel: bool,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ) -> None:
        super().__init__()
        hidden_size = config.vt_hidden_size
        qkv_hidden_size = config.qkv_hidden_size
        num_heads = config.vt_num_attention_heads
        self.rope_2d = KimiK3Rope2DPosEmbRepeated(
            qkv_hidden_size // num_heads, 512, 512
        )
        self.blocks = nn.ModuleList(
            [
                KimiK3VisionEncoderLayer(
                    num_heads=num_heads,
                    hidden_dim=hidden_size,
                    qkv_hidden_size=qkv_hidden_size,
                    mlp_dim=config.vt_intermediate_size,
                    activation=PytorchGELUTanh(),
                    attn_bias=config.attn_bias,
                    linear_bias=config.linear_bias,
                    quant_config=quant_config,
                    prefix=add_prefix(f"blocks.{layer_idx}", prefix),
                    use_data_parallel=use_data_parallel,
                )
                for layer_idx in range(config.vt_num_hidden_layers)
            ]
        )
        self.final_layernorm = nn.RMSNorm(hidden_size)

    def forward(
        self, hidden_states: torch.Tensor, grid_thws: torch.Tensor
    ) -> torch.Tensor:
        rope_freqs_cis = self.rope_2d.get_freqs_cis(
            grid_thws=grid_thws, device=hidden_states.device
        )
        lengths = torch.cat(
            (
                torch.zeros(
                    1, dtype=grid_thws.dtype, device=grid_thws.device
                ),
                grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2],
            )
        )
        cu_seqlens = lengths.cumsum(dim=0, dtype=torch.int32)
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rope_freqs_cis=rope_freqs_cis,
            )
        return self.final_layernorm(hidden_states)


class KimiK3VisionTower(nn.Module):
    def __init__(
        self,
        config: KimiK3VisionConfig,
        *,
        use_data_parallel: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "vision_tower",
    ) -> None:
        super().__init__()
        config = deepcopy(config)
        self.config = config
        self.merge_kernel_size = config.merge_kernel_size
        self.patch_size = config.patch_size
        self.merge_type = config.merge_type
        if config.pos_emb_type != "divided_fixed":
            raise ValueError(
                f"Unsupported K3 vision pos_emb_type: {config.pos_emb_type}"
            )
        if self.merge_type != "sd2_tpool":
            raise ValueError(
                f"Unsupported K3 vision merge_type: {self.merge_type}"
            )
        if config.norm_type != "rmsnorm":
            raise ValueError(
                f"Unsupported K3 vision norm_type: {config.norm_type}"
            )
        if config.mlp_type != "mlp2":
            raise ValueError(
                f"Unsupported K3 vision mlp_type: {config.mlp_type}"
            )
        if config.patch_embed_proj_bias:
            raise ValueError("K3 vision patch embedding bias is not supported")
        if config.qkv_hidden_size % config.vt_num_attention_heads != 0:
            raise ValueError(
                "qkv_hidden_size must be divisible by vt_num_attention_heads"
            )
        self.patch_embed = KimiK3VisionPatchEmbed(
            out_dim=config.vt_hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=config.init_pos_emb_height,
            pos_emb_width=config.init_pos_emb_width,
            pos_emb_time=config.init_pos_emb_time,
            interpolation_mode=config.pos_emb_interpolation_mode,
        )
        self.encoder = KimiK3VisionEncoder(
            config,
            use_data_parallel=use_data_parallel,
            quant_config=quant_config,
            prefix=add_prefix("encoder", prefix),
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thws: Optional[torch.Tensor] = None,
        grid_thw: Optional[torch.Tensor] = None,
    ) -> list[torch.Tensor]:
        if grid_thws is None:
            grid_thws = grid_thw
        if grid_thws is None:
            raise ValueError("K3 vision forward requires grid_thws")
        if grid_thws.ndim != 2 or grid_thws.size(1) != 3:
            raise ValueError(f"Expected grid_thws with shape [N, 3]: {grid_thws}")
        hidden_states = self.patch_embed(pixel_values, grid_thws)
        hidden_states = self.encoder(hidden_states, grid_thws).squeeze(0)
        return tpool_patch_merger(
            hidden_states,
            grid_thws,
            merge_kernel_size=self.merge_kernel_size,
        )


class KimiK3MultiModalProjector(nn.Module):
    """K3 PatchMergerMLPV2 projector with checkpoint-compatible names."""

    def __init__(
        self,
        config: KimiK3VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "mm_projector",
    ) -> None:
        super().__init__()
        merge_h, merge_w = config.merge_kernel_size
        vision_hidden_size = getattr(config, "mm_hidden_size", None)
        if vision_hidden_size is None:
            vision_hidden_size = getattr(
                config, "hidden_size", getattr(config, "vt_hidden_size")
            )
        text_hidden_size = config.text_hidden_size
        self.hidden_size = vision_hidden_size * merge_h * merge_w
        projector_quant_config = (
            quant_config if isinstance(quant_config, ModelSlimConfig) else None
        )
        self.proj = nn.ModuleList(
            [
                ReplicatedLinear(
                    self.hidden_size,
                    self.hidden_size,
                    bias=False,
                    quant_config=projector_quant_config,
                    prefix=add_prefix("proj.0", prefix),
                ),
                nn.GELU(),
                ReplicatedLinear(
                    self.hidden_size,
                    text_hidden_size,
                    bias=False,
                    quant_config=projector_quant_config,
                    prefix=add_prefix("proj.2", prefix),
                ),
            ]
        )
        self.post_norm = nn.RMSNorm(
            text_hidden_size, eps=getattr(config, "projector_ln_eps", 1e-5)
        )

        quant_description = getattr(quant_config, "quant_description", {})
        rot_proj_prefix = f"{prefix}.rot_proj"
        rot_proj_weight_key = f"{rot_proj_prefix}.weight"

        self.use_rot_proj = rot_proj_weight_key in quant_description
        if self.use_rot_proj:
            self.rot_proj = ReplicatedLinear(
                text_hidden_size,
                text_hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=rot_proj_prefix,
            )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = image_features.reshape(-1, self.hidden_size)
        hidden_states, _ = self.proj[0](hidden_states)
        hidden_states = self.proj[1](hidden_states)
        hidden_states, _ = self.proj[2](hidden_states)
        hidden_states = self.post_norm(hidden_states)

        if self.use_rot_proj:
            hidden_states, _ = self.rot_proj(hidden_states)
        return hidden_states

class KimiK3ForConditionalGeneration(nn.Module):
    """Kimi-K3 multimodal wrapper.

    The image embedding merge is intentionally added separately; this wrapper
    establishes the native module hierarchy and complete checkpoint loading.
    """

    def __init__(
        self,
        config: KimiK3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        text_config = getattr(config, "text_config", config)
        vision_config = getattr(config, "vision_config", None)
        quant_description = getattr(quant_config, "quant_description", {})
        uses_wrapper_quant_prefix = any(
            isinstance(name, str) and name.startswith("language_model.")
            for name in quant_description
        )
        language_prefix = (
            maybe_prefix(prefix, "language_model")
            if uses_wrapper_quant_prefix
            else prefix
        )
        self.language_model = KimiK3ForCausalLM(
            text_config,
            quant_config,
            prefix=language_prefix,
        )
        self.pp_group = self.language_model.pp_group

        self.vision_tower = None
        self.mm_projector = None
        self.use_data_parallel = False
        server_args = get_global_server_args()
        multimodal_enabled = bool(
            server_args is not None and server_args.enable_multimodal
        )
        if (
            vision_config is not None
            and multimodal_enabled
            and not getattr(config, "language_only", False)
        ):
            use_data_parallel = bool(
                server_args is not None and server_args.mm_enable_dp_encoder
            )
            self.use_data_parallel = use_data_parallel
            self.vision_tower = KimiK3VisionTower(
                vision_config,
                use_data_parallel=use_data_parallel,
                quant_config=(
                    quant_config
                    if isinstance(quant_config, ModelSlimConfig)
                    else None
                ),
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            self.mm_projector = KimiK3MultiModalProjector(
                vision_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "mm_projector"),
            )

    @property
    def model(self):
        return self.language_model

    def __setattr__(self, name, value):
        if name == "model":
            return
        super().__setattr__(name, value)

    @property
    def lm_head(self):
        return self.language_model.lm_head

    @property
    def start_layer(self) -> int:
        return self.language_model.start_layer

    @property
    def end_layer(self) -> int:
        return self.language_model.end_layer

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_dspark_layers_to_capture(self, layer_ids: list[int]) -> None:
        self.language_model.set_dspark_layers_to_capture(layer_ids)

    def get_image_feature(
        self, items: List[MultimodalDataItem]
    ) -> torch.Tensor:
        if self.vision_tower is None or self.mm_projector is None:
            raise RuntimeError(
                "Kimi-K3 vision modules are not initialized; "
                "start the server with --enable-multimodal"
            )
        device = self.vision_tower.device
        target_dtype = self.vision_tower.dtype
        pixel_values = torch.cat(
            [item.feature for item in items], dim=0
        ).to(device=device, dtype=target_dtype)
        image_grid_thws = []
        for item in items:
            grid_thw = item.model_specific_data.get("image_grid_thw")
            if grid_thw is None:
                grid_thw = item.model_specific_data.get("grid_thws")
            if grid_thw is None:
                raise ValueError("Kimi-K3 image item is missing grid_thws")
            image_grid_thws.append(grid_thw)
        grid_thws = torch.cat(image_grid_thws, dim=0).to(device)

        if self.use_data_parallel:
            image_embeds = run_dp_sharded_mrope_vision_model(
                self.vision_tower,
                pixel_values,
                grid_thws.tolist(),
                rope_type="rope_2d",
            )
        else:
            image_embeds = torch.cat(
                self.vision_tower(pixel_values, grid_thws), dim=0
            )
        image_features = self.mm_projector(image_embeds)
        logger.debug(f"Kimi-K3 image features: {image_features.shape}, "
              f"grid_thws: {grid_thws.tolist()}")
        return image_features

    def pad_input_ids(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={Modality.IMAGE: self.get_image_feature},
            positions=positions,
            pp_proxy_tensors=pp_proxy_tensors,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load multimodal tensors inline and stream text tensors to the LM."""
        multimodal_params = dict(self.named_parameters(remove_duplicate=False))
        loaded_multimodal_params: set[str] = set()

        def stream_language_weights():
            for args in weights:
                name, loaded_weight = args[:2]
                kwargs = args[2] if len(args) > 2 else {}
                if name.startswith(("vision_tower.", "mm_projector.")):
                    if self.vision_tower is None or self.mm_projector is None:
                        continue
                    target_name = (
                        name.replace(".wqkv.", ".attn.qkv_proj.")
                        .replace(".wo.", ".attn.proj.")
                    )
                    if target_name not in multimodal_params:
                        raise ValueError(
                            "Kimi-K3 multimodal weight has no target parameter: "
                            f"source={name}, target={target_name}"
                        )
                    param = multimodal_params[target_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, **kwargs)
                    loaded_multimodal_params.add(target_name)
                    continue

                language_name = name.removeprefix("language_model.")
                if len(args) > 2:
                    yield language_name, loaded_weight, kwargs
                else:
                    yield language_name, loaded_weight

        self.language_model.load_weights(stream_language_weights())

        if self.vision_tower is not None and self.mm_projector is not None:
            expected_multimodal_params = {
                name
                for name in multimodal_params
                if name.startswith(("vision_tower.", "mm_projector."))
            }
            missing = sorted(
                expected_multimodal_params - loaded_multimodal_params
            )
            logger.info(
                "Kimi-K3 multimodal weight load: loaded=%d expected=%d missing=%d",
                len(loaded_multimodal_params),
                len(expected_multimodal_params),
                len(missing),
            )
            if missing:
                logger.warning(
                    "Kimi-K3 multimodal parameters without checkpoint weights: %s",
                    missing[:100],
                )


EntryClass = KimiK3ForConditionalGeneration
