# Copyright 2023-2024 SGLang Team
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

import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.batch_overlap.two_batch_overlap import model_forward_maybe_tbo
from sglang.srt.configs.model_config import get_mimo_v2_fused_qkv_expected_tp_size
from sglang.srt.distributed import (
    get_pp_group,
    moe_tensor_model_parallel_all_reduce,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
    enable_moe_dense_fully_dp,
)
from sglang.srt.layers.cp.utils import enable_cp_v2, is_cp_v2_active
from sglang.srt.layers.dp_attention import (
    get_moe_cp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
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
from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE, get_moe_impl_class
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.utils.cp_utils import (
    can_cp_split,
    cp_all_gather_rerange_output,
    cp_split_and_rebuild_data,
    cp_split_and_rebuild_position,
    is_prefill_context_parallel_enabled,
    prepare_context_parallel_metadata,
)
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
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
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    kv_cache_scales_loader,
)
from sglang.srt.models.mimo_audio import AudioEncoderMixin, MiMoAudioEncoderConfig
from sglang.srt.models.mimo_vl import MiMoVisionTransformer, MiMoVLVisionConfig
from sglang.srt.runtime_context import get_forward, get_parallel, get_server_args
from sglang.srt.utils import (
    LazyValue,
    add_prefix,
    is_non_idle_and_non_empty,
    make_layers,
)

MiMoV2Config = None

logger = logging.getLogger(__name__)

CP_V2_LOCAL_PAD_MIN_TOKENS = 128
CP_V2_LOCAL_PAD_ALIGNMENT = 8


def _get_cp_v2_local_pad_size(num_tokens: int, cp_v2_active: bool) -> int:
    if (
        not cp_v2_active
        or num_tokens < CP_V2_LOCAL_PAD_MIN_TOKENS
        or num_tokens % CP_V2_LOCAL_PAD_ALIGNMENT == 0
    ):
        return 0
    return (
        math.ceil(num_tokens / CP_V2_LOCAL_PAD_ALIGNMENT) * CP_V2_LOCAL_PAD_ALIGNMENT
        - num_tokens
    )


def _get_cp_v2_tp_pad_size(
    num_tokens: int,
    forward_batch: Optional[ForwardBatch],
    *,
    is_moe_full: bool = False,
) -> int:
    if forward_batch is None or not is_cp_v2_active(forward_batch):
        return 0

    target_num_tokens = num_tokens
    cp_metadata = getattr(forward_batch, "attn_cp_metadata", None)
    per_rank_tokens = getattr(cp_metadata, "per_rank_actual_token", None)
    if per_rank_tokens:
        max_rank_tokens = max(int(token) for token in per_rank_tokens)
        if is_moe_full and num_tokens > max_rank_tokens:
            target_num_tokens = max(
                target_num_tokens,
                max_rank_tokens * get_moe_cp_size(),
            )
        else:
            target_num_tokens = max(target_num_tokens, max_rank_tokens)

    return (
        target_num_tokens
        + _get_cp_v2_local_pad_size(target_num_tokens, cp_v2_active=True)
        - num_tokens
    )


def _get_mimo_v2_qkv_linear(param):
    weight_loader = getattr(param, "weight_loader", None)
    linear = getattr(weight_loader, "__self__", None)
    return linear if isinstance(linear, QKVParallelLinear) else None


def _block_dequantize_fp8_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: List[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    block_n, block_k = block_size
    n, k = weight.shape
    scale = scale.to(device=weight.device)
    scale = scale.repeat_interleave(block_n, dim=0).repeat_interleave(block_k, dim=1)
    scale = scale[:n, :k]
    return (weight.to(torch.float32) * scale).to(dtype)


def _block_quantize_fp8_weight(
    weight: torch.Tensor,
    block_size: List[int],
    fp8_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    block_n, block_k = block_size
    n, k = weight.shape
    padded_n = math.ceil(n / block_n) * block_n
    padded_k = math.ceil(k / block_k) * block_k
    weight_padded = torch.zeros(
        (padded_n, padded_k),
        dtype=torch.float32,
        device=weight.device,
    )
    weight_padded[:n, :k] = weight.to(torch.float32)

    n_blocks = padded_n // block_n
    k_blocks = padded_k // block_k
    weight_view = weight_padded.view(n_blocks, block_n, k_blocks, block_k)
    scale = (weight_view.abs().amax(dim=(1, 3)) / torch.finfo(fp8_dtype).max).clamp(
        min=1e-12
    )
    qweight = (
        (weight_view / scale[:, None, :, None])
        .clamp(
            min=torch.finfo(fp8_dtype).min,
            max=torch.finfo(fp8_dtype).max,
        )
        .to(fp8_dtype)
    )

    return (
        qweight.view(padded_n, padded_k)[:n, :k].contiguous(),
        scale.to(torch.float32).contiguous(),
    )


def _mimo_v2_fused_qkv_group_sizes(
    qkv_linear: QKVParallelLinear, groups_per_rank: int
) -> Tuple[int, int, int]:
    return (
        qkv_linear.q_proj_shard_size // groups_per_rank,
        qkv_linear.kv_proj_shard_size // groups_per_rank,
        qkv_linear.v_proj_shard_size // groups_per_rank,
    )


def _finalize_mimo_v2_fused_qkv_repacked(qkv_linear: QKVParallelLinear) -> bool:
    if getattr(qkv_linear, "_mimo_v2_fused_qkv_repacked", False):
        return True
    if not getattr(qkv_linear, "_mimo_v2_fused_qkv_weight_loaded", False):
        return False
    if not hasattr(qkv_linear, "_mimo_v2_fused_qkv_scale_interleaved"):
        return False

    groups_per_rank = qkv_linear._mimo_v2_fused_qkv_groups_per_rank
    block_size = qkv_linear._mimo_v2_fused_qkv_block_size
    scale_interleaved = qkv_linear._mimo_v2_fused_qkv_scale_interleaved
    scale_param = qkv_linear._mimo_v2_fused_qkv_scale_param

    q_group_size, k_group_size, v_group_size = _mimo_v2_fused_qkv_group_sizes(
        qkv_linear, groups_per_rank
    )
    group_weight_size = q_group_size + k_group_size + v_group_size
    group_scale_size = math.ceil(group_weight_size / block_size[0])

    weight = qkv_linear.weight.data
    if weight.shape[0] != group_weight_size * groups_per_rank:
        return False

    q_parts: List[torch.Tensor] = []
    k_parts: List[torch.Tensor] = []
    v_parts: List[torch.Tensor] = []
    for group_idx in range(groups_per_rank):
        weight_start = group_idx * group_weight_size
        scale_start = group_idx * group_scale_size
        group_weight = weight[weight_start : weight_start + group_weight_size]
        group_scale = scale_interleaved[scale_start : scale_start + group_scale_size]
        group_dequant = _block_dequantize_fp8_weight(
            group_weight,
            group_scale,
            block_size,
            qkv_linear.orig_dtype,
        )
        q_part, k_part, v_part = group_dequant.split(
            [q_group_size, k_group_size, v_group_size], dim=0
        )
        q_parts.append(q_part)
        k_parts.append(k_part)
        v_parts.append(v_part)

    repacked = torch.cat([*q_parts, *k_parts, *v_parts], dim=0).contiguous()
    if tuple(repacked.shape) != tuple(qkv_linear.weight.shape):
        return False

    qweight, qscale = _block_quantize_fp8_weight(
        repacked,
        block_size,
        qkv_linear.weight.dtype,
    )
    if tuple(qweight.shape) != tuple(qkv_linear.weight.shape):
        return False
    if tuple(qscale.shape) != tuple(scale_param.shape):
        return False

    qkv_linear.weight.data = qweight.to(device=qkv_linear.weight.device)
    scale_param.data = qscale.to(device=scale_param.device, dtype=scale_param.dtype)
    qkv_linear._mimo_v2_fused_qkv_repacked = True
    return True


def _load_mimo_v2_fused_qkv_collapsed_tp(
    name: str,
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    expected_fused_tp_size: int,
) -> bool:
    qkv_linear = _get_mimo_v2_qkv_linear(param)
    if qkv_linear is None:
        return False

    tp_size = get_parallel().attn_tp_size
    tp_rank = get_parallel().attn_tp_rank
    groups_per_rank = expected_fused_tp_size // tp_size
    if groups_per_rank <= 1:
        return False

    group_chunks = loaded_weight.chunk(expected_fused_tp_size, dim=0)
    rank_start = tp_rank * groups_per_rank
    local_interleaved = torch.cat(
        group_chunks[rank_start : rank_start + groups_per_rank], dim=0
    )

    if param is qkv_linear.weight:
        if tuple(local_interleaved.shape) != tuple(param.shape):
            return False
        default_weight_loader(param, local_interleaved)
        qkv_linear._mimo_v2_fused_qkv_groups_per_rank = groups_per_rank
        qkv_linear._mimo_v2_fused_qkv_weight_loaded = True
        _finalize_mimo_v2_fused_qkv_repacked(qkv_linear)
        return True

    if not hasattr(qkv_linear, "weight") or not hasattr(qkv_linear, "weight_scale_inv"):
        return False
    if param is not qkv_linear.weight_scale_inv:
        return False

    block_size = getattr(
        qkv_linear.quant_method.quant_config, "weight_block_size", None
    )
    if block_size is None:
        return False

    q_group_size, k_group_size, v_group_size = _mimo_v2_fused_qkv_group_sizes(
        qkv_linear, groups_per_rank
    )
    group_weight_size = q_group_size + k_group_size + v_group_size
    group_scale_size = math.ceil(group_weight_size / block_size[0])

    expected_scale_rows = group_scale_size * groups_per_rank
    if local_interleaved.shape[0] != expected_scale_rows:
        return False
    qkv_linear._mimo_v2_fused_qkv_groups_per_rank = groups_per_rank
    qkv_linear._mimo_v2_fused_qkv_block_size = block_size
    qkv_linear._mimo_v2_fused_qkv_scale_param = param
    qkv_linear._mimo_v2_fused_qkv_scale_interleaved = local_interleaved.detach().clone()
    _finalize_mimo_v2_fused_qkv_repacked(qkv_linear)
    return True


def load_mimo_v2_qkv_proj_weight(
    name, param, loaded_weight, expected_fused_tp_size: Optional[int] = None
):
    tp_size = get_parallel().attn_tp_size
    if expected_fused_tp_size is not None:
        if expected_fused_tp_size % tp_size != 0:
            raise ValueError(
                f"MiMoV2 fused qkv_proj checkpoint is TP={expected_fused_tp_size}-"
                f"interleaved; got incompatible attention tp_size={tp_size} while "
                f"loading {name}. The checkpoint TP size must be divisible by the "
                "runtime attention TP size."
            )
        if (
            expected_fused_tp_size // tp_size > 1
            and _load_mimo_v2_fused_qkv_collapsed_tp(
                name, param, loaded_weight, expected_fused_tp_size
            )
        ):
            return

    if loaded_weight.shape == param.shape:
        # The checkpoint already stores this rank's qkv_proj shard.
        default_weight_loader(param, loaded_weight)
        return

    if loaded_weight.ndim != param.ndim or loaded_weight.shape[1:] != param.shape[1:]:
        raise ValueError(
            f"qkv_proj weight {name}: unexpected shape {tuple(loaded_weight.shape)}; "
            f"expected sharded {tuple(param.shape)}"
        )

    tp_rank = get_parallel().attn_tp_rank

    qkv_weight_loader = getattr(param, "weight_loader", None)
    if qkv_weight_loader is not None:
        if expected_fused_tp_size is None:
            qkv_weight_loader(param, loaded_weight)
            return
        if (
            name.endswith(".weight_scale_inv")
            and loaded_weight.shape[0] >= param.shape[0]
        ):
            qkv_weight_loader(param, loaded_weight)
            return

    fused_shape = (param.shape[0] * tp_size, *param.shape[1:])
    if tuple(loaded_weight.shape) != fused_shape:
        raise ValueError(
            f"qkv_proj weight {name}: unexpected shape {tuple(loaded_weight.shape)}; "
            f"expected fused {fused_shape} or sharded {tuple(param.shape)}"
        )

    default_weight_loader(param, loaded_weight.chunk(tp_size, dim=0)[tp_rank])


class MiMoV2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size

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
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(
        self,
        x,
        forward_batch: ForwardBatch = None,
    ):
        if (self.tp_size == 1) and x.shape[0] == 0:
            return x

        local_pad_size = _get_cp_v2_tp_pad_size(x.shape[0], forward_batch)
        if local_pad_size > 0:
            original_num_tokens = x.shape[0]
            x = F.pad(x, (0, 0, 0, local_pad_size))
        else:
            original_num_tokens = None

        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        if original_num_tokens is not None:
            x = x[:original_num_tokens]
        return x


class MoEGate(ReplicatedLinear):
    def __init__(
        self,
        config,
        quant_config,
        prefix: str = "",
        is_nextn: bool = False,
    ):
        super().__init__(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            quant_config=None,
            prefix=prefix,
        )
        self.is_nextn = is_nextn
        self.dtype = torch.float32
        if config.topk_method == "noaux_tc":
            correction_bias_dtype = (
                torch.bfloat16
                if quant_config is not None
                and quant_config.get_name() == "modelopt_fp4"
                and get_moe_runner_backend().is_flashinfer_trtllm()
                else self.dtype
            )
            self.e_score_correction_bias = nn.Parameter(
                torch.empty((config.n_routed_experts), dtype=correction_bias_dtype)
            )
        else:
            self.e_score_correction_bias = None

    def forward(self, hidden_states):
        logits, _ = super().forward(hidden_states)
        return logits


class MiMoV2MoE(nn.Module):

    def __init__(
        self,
        config: MiMoV2Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        is_nextn: bool = False,
    ):
        super().__init__()
        self.tp_size = get_parallel().moe_tp_size

        self.config = config
        self.layer_id = layer_id

        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        self.gate = MoEGate(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("gate", prefix),
            is_nextn=is_nextn,
        )

        experts_type = get_moe_impl_class(quant_config)
        self.experts = experts_type(
            num_experts=config.n_routed_experts
            + get_server_args().ep_num_redundant_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=self.layer_id,
            quant_config=quant_config,
            routed_scaling_factor=1.0,
            prefix=add_prefix("experts", prefix),
        )

        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            correction_bias=self.gate.e_score_correction_bias,
            scoring_func=config.scoring_func,
            quant_config=quant_config,
            routed_scaling_factor=1.0,
            apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk,
            # Some Fp4 MoE backends require the output format to be bypassed but the MTP layers are unquantized
            # and requires the output format to be standard. We use quant_config to determine the output format.
            output_format=TopKOutputFormat.STANDARD if quant_config is None else None,
        )

        # todo : implement tbo forward needed
        if (
            get_moe_a2a_backend().is_deepep()
            or get_moe_a2a_backend().is_mooncake()
            or get_moe_a2a_backend().is_ascend_fuseep()
        ):
            # TODO: we will support tp < ep in the future
            self.ep_size = get_parallel().moe_ep_size
            self.num_experts = (
                config.n_routed_experts + get_server_args().ep_num_redundant_experts
            )
            self.renormalize = config.norm_topk_prob
            self.topk_group = config.topk_group
            self.num_expert_group = config.n_group
            self.correction_bias = (
                self.gate.e_score_correction_bias.data
                if self.gate.e_score_correction_bias is not None
                else None
            )

        self._enable_a2a_moe = (
            get_moe_a2a_backend().is_deepep()
            or get_moe_a2a_backend().is_mooncake()
            or get_moe_a2a_backend().is_ascend_fuseep()
        )

    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
        ]

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
    ) -> torch.Tensor:
        if not self._enable_a2a_moe:
            return self.forward_normal(
                hidden_states,
                forward_batch,
            )
        else:
            return self.forward_deepep(hidden_states, forward_batch)

    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
    ) -> torch.Tensor:
        local_pad_size = _get_cp_v2_tp_pad_size(
            hidden_states.shape[0], forward_batch, is_moe_full=True
        )
        if local_pad_size > 0:
            original_num_tokens = hidden_states.shape[0]
            hidden_states = F.pad(hidden_states, (0, 0, 0, local_pad_size))
        else:
            original_num_tokens = None

        if hidden_states.shape[0] > 0:
            # router_logits: (num_tokens, n_experts)
            router_logits = self.gate(hidden_states)
            topk_output = self.topk(hidden_states, router_logits)
        else:
            topk_output = self.topk.empty_topk_output(hidden_states.device)

        final_hidden_states = self.experts(hidden_states, topk_output)

        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=True,
        ):
            final_hidden_states = moe_tensor_model_parallel_all_reduce(
                final_hidden_states
            )

        if original_num_tokens is not None:
            final_hidden_states = final_hidden_states[:original_num_tokens]
        return final_hidden_states

    def forward_deepep(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        if hidden_states.shape[0] > 0:
            # router_logits: (num_tokens, n_experts)
            router_logits = self.gate(hidden_states)
            topk_output = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=forward_batch.num_token_non_padded,
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                ),
            )
        else:
            topk_output = self.topk.empty_topk_output(hidden_states.device)

        final_hidden_states = self.experts(
            hidden_states=hidden_states, topk_output=topk_output
        )

        return final_hidden_states

    def op_gate(self, state):
        if is_non_idle_and_non_empty(
            state.forward_batch.forward_mode, state.hidden_states_mlp_input
        ):
            # router_logits: (num_tokens, n_experts)
            state.router_logits = self.gate(state.hidden_states_mlp_input)
        else:
            state.router_logits = None

    def op_select_experts(self, state):
        router_logits = state.pop("router_logits")
        hidden_states = state.hidden_states_mlp_input
        if router_logits is not None:
            with get_global_expert_distribution_recorder().with_current_layer(
                self.layer_id
            ):
                state.topk_output = self.topk(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    num_token_non_padded=state.forward_batch.num_token_non_padded,
                    expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                        layer_id=self.layer_id,
                    ),
                )
        else:
            state.topk_output = self.topk.empty_topk_output(hidden_states.device)

    def op_dispatch_a(self, state):
        if self.ep_size > 1:
            self.experts.dispatcher.dispatch_a(
                hidden_states=state.pop("hidden_states_mlp_input"),
                topk_output=state.pop("topk_output"),
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )

    def op_dispatch_b(self, state):
        if self.ep_size > 1:
            with get_global_expert_distribution_recorder().with_current_layer(
                self.layer_id
            ):
                state.dispatch_output = self.experts.dispatcher.dispatch_b(
                    tbo_subbatch_index=state.get("tbo_subbatch_index"),
                )

    def op_experts(self, state):
        state.combine_input = self.experts.run_moe_core(
            dispatch_output=state.dispatch_output,
        )

    def op_combine_a(self, state):
        if self.ep_size > 1:
            self.experts.dispatcher.combine_a(
                combine_input=state.pop("combine_input"),
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )
            state.pop("dispatch_output")

    def op_combine_b(self, state):
        if self.ep_size > 1:
            state.hidden_states_after_combine = self.experts.dispatcher.combine_b(
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )

    def op_output(self, state):
        state.hidden_states_mlp_output = state.pop("hidden_states_after_combine")


class MiMoV2Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        v_head_dim: Optional[int] = None,
        v_scale: Optional[float] = None,
        sliding_window_size: int = -1,  # if is -1 ,normal attention,else ,window attention
        attention_bias: bool = False,
        attention_sink_bias: bool = False,
        layer_id: int = 0,
        rope_theta: float = 1000000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 32768,
        quant_config: Optional[QuantizationConfig] = None,
        partial_rotary_factor: float = 1.0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        attn_tp_rank = get_parallel().attn_tp_rank
        attn_tp_size = get_parallel().attn_tp_size

        self.total_num_heads = num_heads
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
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim if v_head_dim is not None else head_dim

        self.q_size = self.num_heads * self.head_dim
        self.k_size = self.num_kv_heads * self.head_dim
        self.v_size = self.num_kv_heads * self.v_head_dim

        self.v_scale = v_scale

        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            v_head_size=self.v_head_dim,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
            skip_block_quant_check=True,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.v_head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            partial_rotary_factor=partial_rotary_factor,
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            v_head_dim=self.v_head_dim,
            sliding_window_size=sliding_window_size,  # if is -1 ,normal attention,else ,window attention
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

        self.attention_sink_bias = (
            torch.nn.Parameter(torch.empty(self.num_heads), requires_grad=False)
            if attention_sink_bias
            else None
        )

    def _qkv_proj_with_cp_v2_padding(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        qkv_pad_size = _get_cp_v2_local_pad_size(
            hidden_states.shape[0], is_cp_v2_active(forward_batch)
        )
        if qkv_pad_size == 0:
            return self.qkv_proj(hidden_states)

        padded_hidden_states = F.pad(hidden_states, (0, 0, 0, qkv_pad_size))
        qkv, bias = self.qkv_proj(padded_hidden_states)
        return qkv[: hidden_states.shape[0]], bias

    def op_prepare(self, state):
        state.attn_intermediate_state = self.forward_prepare(
            positions=state.positions,
            hidden_states=state.pop("hidden_states_after_comm_pre_attn"),
            forward_batch=state.forward_batch,
        )

    def op_core(self, state):
        state.hidden_states_after_attn = self.forward_core(
            state.pop("attn_intermediate_state")
        )

    def forward_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        if hidden_states.shape[0] == 0:
            return hidden_states, forward_batch, None
        qkv, _ = self._qkv_proj_with_cp_v2_padding(hidden_states, forward_batch)
        q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)

        q, k = self.rotary_emb(positions, q, k)
        if self.v_scale is not None:
            v = v * self.v_scale

        inner_state = q, k, v, forward_batch
        return None, forward_batch, inner_state

    def forward_core(self, intermediate_state):
        hidden_states, forward_batch, inner_state = intermediate_state
        if inner_state is None:
            return hidden_states
        attn_output = self.attn(
            *inner_state,
            sinks=self.attention_sink_bias,
        )
        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self._qkv_proj_with_cp_v2_padding(hidden_states, forward_batch)
        q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)

        # [t, h, dr]
        q, k = self.rotary_emb(positions, q, k)
        # [t, h, d]

        if self.v_scale is not None:
            v = v * self.v_scale
        attn_output = self.attn(q, k, v, forward_batch, sinks=self.attention_sink_bias)
        output, _ = self.o_proj(attn_output)
        return output


class MiMoV2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: MiMoV2Config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id

        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        # In v5, rope_scaling is a property alias for rope_parameters and returns
        # a standardized dict even when there's no actual scaling.  Treat the
        # "default" (no-op) type as None so factory.py uses plain RotaryEmbedding.
        if (
            isinstance(rope_scaling, dict)
            and rope_scaling.get("rope_type") == "default"
        ):
            rope_scaling = None
        max_position_embeddings = getattr(
            config,
            "context_len",
            getattr(config, "max_position_embeddings", 32768),
        )

        if self.is_swa_layer():
            self.self_attn = MiMoV2Attention(
                hidden_size=self.hidden_size,
                num_heads=config.swa_num_attention_heads,
                num_kv_heads=config.swa_num_key_value_heads,
                head_dim=config.swa_head_dim,
                v_head_dim=getattr(config, "swa_v_head_dim", None),
                v_scale=getattr(config, "attention_value_scale", None),
                sliding_window_size=config.sliding_window_size,
                attention_bias=config.attention_bias,
                attention_sink_bias=getattr(
                    config, "add_swa_attention_sink_bias", False
                ),
                layer_id=layer_id,
                rope_theta=getattr(config, "swa_rope_theta", rope_theta),
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                partial_rotary_factor=getattr(config, "partial_rotary_factor", 1.0),
                prefix=add_prefix("self_attn", prefix),
            )
        else:
            self.self_attn = MiMoV2Attention(
                hidden_size=self.hidden_size,
                num_heads=self.config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                v_head_dim=getattr(config, "v_head_dim", None),
                v_scale=getattr(config, "attention_value_scale", None),
                sliding_window_size=-1,  # normal attention
                attention_bias=config.attention_bias,
                attention_sink_bias=getattr(
                    config, "add_full_attention_sink_bias", False
                ),
                layer_id=layer_id,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                partial_rotary_factor=getattr(config, "partial_rotary_factor", 1.0),
                prefix=add_prefix("self_attn", prefix),
            )

        self.is_layer_sparse = self.is_moe_layer(layer_id)
        is_previous_layer_sparse = self.is_moe_layer(layer_id - 1)
        is_next_layer_sparse = self.is_moe_layer(layer_id + 1)

        if self.is_layer_sparse:
            self.mlp = MiMoV2MoE(
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                layer_id=layer_id,
            )
        else:
            if enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = MiMoV2MLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.layernorm_epsilon
        )

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
            is_last_layer=(self.layer_id == self.config.num_hidden_layers - 1),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
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

        fuse_mlp_allreduce = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )

        # For DP with padding, reduce scatter can be used instead of all-reduce.
        mlp_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        with get_forward().scoped(
            fuse_mlp_allreduce=fuse_mlp_allreduce,
            mlp_reduce_scatter=mlp_reduce_scatter,
        ):
            hidden_states = self.mlp(hidden_states, forward_batch)

        if fuse_mlp_allreduce:
            hidden_states._sglang_needs_allreduce_fusion = True
        else:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )

        return hidden_states, residual

    def is_moe_layer(self, layer_idx: int) -> bool:
        return (
            hasattr(self.config, "moe_layer_freq")
            and 0 <= layer_idx < len(self.config.moe_layer_freq)
            and not isinstance(self.config.moe_layer_freq, int)
            and self.config.moe_layer_freq[layer_idx]
        )

    def is_swa_layer(self) -> bool:
        return self.config.hybrid_layer_pattern[self.layer_id] == 1

    def op_comm_prepare_attn(
        self,
        state,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        tbo_subbatch_index: Optional[int] = None,
    ):
        state.hidden_states_after_comm_pre_attn, state.residual_after_input_ln = (
            self.layer_communicator.prepare_attn(hidden_states, residual, forward_batch)
        )
        state.update(
            dict(
                forward_batch=forward_batch,
                positions=positions,
                tbo_subbatch_index=tbo_subbatch_index,
            )
        )

    def op_comm_prepare_mlp(self, state):
        state.hidden_states_mlp_input, state.residual_after_comm_pre_mlp = (
            self.layer_communicator.prepare_mlp(
                state.pop("hidden_states_after_attn"),
                state.pop("residual_after_input_ln"),
                state.forward_batch,
            )
        )

    def op_comm_postprocess_layer(self, state):
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            state.pop("hidden_states_mlp_output"),
            state.pop("residual_after_comm_pre_mlp"),
            state.forward_batch,
        )

        output = dict(
            positions=state.positions,
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=state.forward_batch,
            tbo_subbatch_index=state.tbo_subbatch_index,
        )

        state.clear(
            expect_keys={
                "positions",
                "forward_batch",
                "tbo_subbatch_index",
            }
        )
        return output


class MiMoV2Model(nn.Module):
    def __init__(
        self,
        config: MiMoV2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = MiMoV2DecoderLayer,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                use_attn_tp_group=is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Use the provided decoder layer type or default to MiMoV2DecoderLayer
        decoder_layer_type = decoder_layer_type or MiMoV2DecoderLayer
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            layer_fn=lambda idx, prefix: decoder_layer_type(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        else:
            self.norm = PPMissingLayer(return_tuple=True)
        self.attn_cp_size = get_parallel().attn_cp_size

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
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        if (
            is_prefill_context_parallel_enabled()
            and not enable_cp_v2()
            and not is_cp_v2_active(forward_batch)
            and forward_batch.forward_mode.is_context_parallel_extend()
            and forward_batch.attn_cp_metadata is not None
        ):
            if self.pp_group.is_first_rank:
                hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)
            positions = cp_split_and_rebuild_position(forward_batch, positions)

        if forward_batch.can_run_tbo:
            tbo_start_layer = self.start_layer
            tbo_end_layer = self.end_layer

            # skip first layer for TBO when starting from layer 0
            if self.start_layer == 0:
                layer = self.layers[0]
                hidden_states, residual = layer(
                    positions, hidden_states, forward_batch, residual
                )
                tbo_start_layer = tbo_start_layer + 1

            hidden_states, residual = model_forward_maybe_tbo(
                layers=self.layers[tbo_start_layer:tbo_end_layer],
                enable_tbo=True,
                input_data_scatter_mode=(
                    ScatterMode.model_input_output()
                    if tbo_start_layer == self.start_layer
                    else self.layers[
                        tbo_start_layer - 1
                    ].layer_scatter_modes.layer_output_mode
                ),
                positions=positions,
                forward_batch=forward_batch,
                hidden_states=hidden_states,
                residual=residual,
            )
        else:
            for i in range(self.start_layer, self.end_layer):
                layer = self.layers[i]
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    forward_batch,
                    residual,
                )

        hidden_states_before_norm = None
        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            if hidden_states.shape[0] > 0:
                if forward_batch.return_hidden_states_before_norm:
                    hidden_states_before_norm = (
                        hidden_states if residual is None else hidden_states + residual
                    )
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)

        if (
            self.pp_group.is_last_rank
            and not enable_cp_v2()
            and not is_cp_v2_active(forward_batch)
            and is_prefill_context_parallel_enabled()
            and forward_batch.forward_mode.is_context_parallel_extend()
            and forward_batch.attn_cp_metadata is not None
        ):
            hidden_states = cp_all_gather_rerange_output(
                hidden_states,
                self.attn_cp_size,
                forward_batch,
                torch.cuda.current_stream(),
            )
            if hidden_states_before_norm is not None:
                hidden_states_before_norm = cp_all_gather_rerange_output(
                    hidden_states_before_norm,
                    self.attn_cp_size,
                    forward_batch,
                    torch.cuda.current_stream(),
                )

        return hidden_states, hidden_states_before_norm

    # If this function is called, it should always initialize KV cache scale
    # factors (or else raise an exception). Thus, handled exceptions should
    # make sure to leave KV cache scale factors in a known good (dummy) state
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        attn_tp_rank = get_parallel().attn_tp_rank
        attn_tp_size = get_parallel().attn_tp_size
        for layer_idx, scaling_factor in kv_cache_scales_loader(
            quantization_param_path,
            attn_tp_rank,
            attn_tp_size,
            self.config.num_hidden_layers,
            self.config.__class__.model_type,
        ):
            if not isinstance(self.layers[layer_idx], nn.Identity):
                layer_self_attn = self.layers[layer_idx].self_attn
            if hasattr(layer_self_attn.attn, "k_scale"):
                layer_self_attn.attn.k_scale = scaling_factor
                layer_self_attn.attn.v_scale = scaling_factor
            else:
                raise RuntimeError(
                    "Self attention has no KV cache scaling " "factor attribute!"
                )


class MiMoV2ForCausalLM(nn.Module, AudioEncoderMixin):
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

    # Prefixes for weight routing in encoder_only/language_only modes
    _LANGUAGE_WEIGHT_PREFIXES = ("model.", "lm_head.")
    _VISION_WEIGHT_PREFIXES = ("visual.", "vision_model.")
    # ``audio_`` already covers ``audio_encoder.`` so a single prefix is enough.
    _AUDIO_WEIGHT_PREFIXES = ("audio_",)
    _AUDIO_WEIGHT_SUBSTRING = "speech_embeddings"
    _MIN_LEGACY_CP_TOKENS = 64

    def __init__(
        self,
        config: MiMoV2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self._encoder_processor = None  # lazy-created in preprocess_mm_for_encoder

        if not self.config.encoder_only:
            self.model = MiMoV2Model(
                config, quant_config=quant_config, prefix=add_prefix("model", prefix)
            )

            if self.pp_group.is_last_rank:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                    use_attn_tp_group=get_server_args().enable_dp_lm_head,
                )
            else:
                self.lm_head = PPMissingLayer()
        else:
            self.model = None
            self.lm_head = None

        self.logits_processor = (
            LogitsProcessor(config) if not self.config.encoder_only else None
        )
        self.attn_cp_size = get_parallel().attn_cp_size
        self.attn_cp_rank = get_parallel().attn_cp_rank

        vision_config = getattr(config, "vision_config", None)
        audio_config = getattr(config, "audio_config", None)
        self._is_multimodal = (
            not self.config.language_only
            and vision_config is not None
            and audio_config is not None
        )
        # In full multimodal or encoder-only mode, build encoders so P can fall
        # back to local encoding when an EPD encoder is unreachable. In
        # language-only mode, keep only the language tower.
        if self._is_multimodal:
            if hasattr(vision_config, "to_dict"):
                vision_config = vision_config.to_dict()
            if hasattr(audio_config, "to_dict"):
                audio_config = audio_config.to_dict()

            self.visual = MiMoVisionTransformer(
                MiMoVLVisionConfig.from_dict(vision_config),
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=None,
                prefix=add_prefix("visual", prefix),
            )
            self.build_audio_encoder(MiMoAudioEncoderConfig(**audio_config))

        self._routed_experts_weights_of_layer = LazyValue(
            lambda: (
                {
                    layer_id: layer.mlp.get_moe_weights()
                    for layer_id, layer in enumerate(self.model.layers)
                    if isinstance(layer.mlp, MiMoV2MoE)
                }
                if self.model is not None
                else {}
            )
        )

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    def get_input_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert (
            self.model is not None
        ), "get_input_embedding() is not available in encoder_only mode"
        return self.model.get_input_embedding(input_ids)

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def preprocess_mm_for_encoder(self, mm_data, modality, config):
        if self._encoder_processor is None:
            from sglang.srt.multimodal.processors.mimo_v2 import MiMoProcessor

            self._encoder_processor = MiMoProcessor.from_hf_config(
                self.config, mm_config=config
            )
        return self._encoder_processor.preprocess_for_encoder(mm_data, modality)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        image_grid_thw = torch.cat([item.image_grid_thw for item in items], dim=0)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()
        return self.visual(pixel_values, grid_thw=image_grid_thw)

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        video_grid_thw = torch.cat([item.video_grid_thw for item in items], dim=0)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert video_grid_thw.dim() == 2, video_grid_thw.dim()
        return self.visual(pixel_values, grid_thw=video_grid_thw)

    @torch.inference_mode()
    def encode_video_audio(self, mm_inputs: Dict) -> Optional[torch.Tensor]:
        # EPD-side hook: encode audio tracks pulled from videos and trim to the
        # interleaved per-video segments produced by MiMoProcessor (segment
        # starts / lens / per_video_num_units). Returns None if there is no
        # audio to encode. The server passes the result through to the receiver
        # under aux_data["video_audio_embedding"].
        import numpy as np

        audio_features = mm_inputs.get("video_audio_features")
        if not audio_features:
            return None

        def _as_tensor(data):
            if isinstance(data, torch.Tensor):
                return data
            if isinstance(data, np.ndarray):
                return torch.tensor(data)
            if isinstance(data, list) and data and isinstance(data[0], np.ndarray):
                return torch.tensor(np.array(data))
            if isinstance(data, list) and data and isinstance(data[0], (int, float)):
                return torch.tensor(data)
            return data

        audio_feature_lens = mm_inputs["video_audio_feature_lens"]
        audio_item = MultimodalDataItem.from_dict(
            {
                "modality": Modality.AUDIO,
                "feature": _as_tensor(audio_features),
            }
        )
        audio_item.set("audio_feature_lens", _as_tensor(audio_feature_lens))

        audio_embedding = self.get_audio_feature([audio_item]).cpu()
        if audio_embedding.ndim != 2:
            audio_embedding = audio_embedding.reshape(-1, audio_embedding.shape[-1])

        segment_lens_flat = mm_inputs["video_audio_segment_lens_flat"]
        segment_starts_flat = mm_inputs["video_audio_segment_starts_flat"]
        per_video_num_units = mm_inputs["video_audio_per_video_num_units"]
        per_video_audio_token_lens = (
            audio_feature_lens.tolist()
            if hasattr(audio_feature_lens, "tolist")
            else list(audio_feature_lens)
        )

        trimmed_chunks = []
        emb_offset = 0
        unit_idx = 0
        audio_video_idx = 0
        for num_units in per_video_num_units:
            if num_units <= 0:
                continue
            vid_audio_len = per_video_audio_token_lens[audio_video_idx]
            for _ in range(num_units):
                start = segment_starts_flat[unit_idx]
                seg_len = segment_lens_flat[unit_idx]
                trimmed_chunks.append(
                    audio_embedding[emb_offset + start : emb_offset + start + seg_len]
                )
                unit_idx += 1
            emb_offset += vid_audio_len
            audio_video_idx += 1

        return (
            torch.cat(trimmed_chunks, dim=0) if trimmed_chunks else audio_embedding[:0]
        )

    def get_input_embeddings(self) -> Optional[nn.Embedding]:
        return self.model.embed_tokens if self.model is not None else None

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        assert (
            not self.config.encoder_only
        ), "forward() should not be called in encoder_only mode"

        real_num_tokens = getattr(forward_batch, "num_token_non_padded_cpu", None)
        if real_num_tokens is None:
            real_num_tokens = len(input_ids)
        if (
            is_prefill_context_parallel_enabled()
            and not enable_cp_v2()
            and not is_cp_v2_active(forward_batch)
            and int(real_num_tokens) >= self._MIN_LEGACY_CP_TOKENS
        ):
            if can_cp_split(len(input_ids), self.attn_cp_size, forward_batch):
                seq_lens_cpu = forward_batch.seq_lens_cpu
                if hasattr(seq_lens_cpu, "tolist"):
                    seq_lens_cpu = seq_lens_cpu.tolist()
                forward_batch.attn_cp_metadata = prepare_context_parallel_metadata(
                    len(input_ids),
                    self.attn_cp_rank,
                    self.attn_cp_size,
                    seq_lens_cpu,
                    extend_seqs_len=forward_batch.extend_seq_lens_cpu,
                )

        if self._is_multimodal:
            hidden_states, hidden_states_before_norm = general_mm_embed_routine(
                input_ids=input_ids,
                forward_batch=forward_batch,
                language_model=self.model,
                multimodal_model=self,
                positions=positions,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        else:
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
        return self.model.start_layer if self.model is not None else 0

    @property
    def end_layer(self):
        return self.model.end_layer if self.model is not None else 0

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        stacked_params_mapping_vit = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = DeepEPMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )

        params_dict = dict(self.named_parameters())
        skipped_mtp_weights = False

        for name, loaded_weight in weights:
            is_vision_weight = name.startswith(self._VISION_WEIGHT_PREFIXES)
            is_audio_weight = (
                name.startswith(self._AUDIO_WEIGHT_PREFIXES)
                or self._AUDIO_WEIGHT_SUBSTRING in name
            )

            if not self._is_multimodal and (is_vision_weight or is_audio_weight):
                continue

            if self.config.encoder_only and name.startswith(
                self._LANGUAGE_WEIGHT_PREFIXES
            ):
                continue

            if self._is_multimodal and is_audio_weight:
                if name.startswith("audio_encoder."):
                    name = name[len("audio_encoder.") :]
                name = self.remap_audio_weight_name(name)
                if "input_local_transformer" not in name:
                    name = name.replace("self_attn.out_proj", "self_attn.attn.proj")
                    audio_stacked = False
                    for param_name, weight_name, shard_id in [
                        ("self_attn.attn.qkv_proj", "self_attn.q_proj", "q"),
                        ("self_attn.attn.qkv_proj", "self_attn.k_proj", "k"),
                        ("self_attn.attn.qkv_proj", "self_attn.v_proj", "v"),
                    ]:
                        if weight_name in name:
                            name = name.replace(weight_name, param_name)
                            if name not in params_dict:
                                break
                            param = params_dict[name]
                            weight_loader = param.weight_loader
                            weight_loader(param, loaded_weight, shard_id)
                            audio_stacked = True
                            break
                    if audio_stacked:
                        continue
                if name not in params_dict:
                    logger.warning(
                        f"Audio param {name} not found in params_dict, skipping"
                    )
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if self._AUDIO_WEIGHT_SUBSTRING in name:
                    weight_loader(param, loaded_weight[: param.shape[0], :])
                else:
                    weight_loader(param, loaded_weight)
                continue

            if self._is_multimodal and "visual" in name:
                name = name.replace("vision_model.", "")
                name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")
                match_stacked_vit = False
                for param_name, weight_name, shard_id in stacked_params_mapping_vit:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        match_stacked_vit = True
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    match_stacked_vit = True
                    break
                if match_stacked_vit:
                    continue
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

                if name.endswith("patch_embed.proj.weight"):
                    patch_embed = self.get_submodule(name.rsplit(".", 2)[0])
                    if hasattr(patch_embed, "sync_proj_weight_linear_format"):
                        patch_embed.sync_proj_weight_linear_format()
                continue

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

            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue

            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                if self.pp_group.world_size > 1 and self.pp_group.is_last_rank:
                    # Handle pp weight tying here
                    # find the embed_tokens.weight in the weights
                    embed_token_weights = next(
                        filter(lambda x: x[0] == "model.embed_tokens.weight", weights)
                    )[1]
                    loaded_weight = embed_token_weights
                else:
                    continue

            if "mtp" in name:
                if not skipped_mtp_weights:
                    logger.info(
                        "Skipping draft-only MiMo-V2 MTP weights while loading the "
                        "target model; MiMoV2MTP loads these weights in the draft "
                        "model runner."
                    )
                    skipped_mtp_weights = True
                continue

            # Support fused qkv_proj checkpoint (Pro format)
            if "qkv_proj" in name:
                if name in params_dict:
                    param = params_dict[name]
                    expected_fused_tp_size = get_mimo_v2_fused_qkv_expected_tp_size(
                        self.config
                    )
                    load_mimo_v2_qkv_proj_weight(
                        name, param, loaded_weight, expected_fused_tp_size
                    )
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if (
                    "compression_attention" in name
                    or "hybrid_softmax_attention" in name
                    or "compressed_softmax_attn" in name
                ):
                    continue
                if weight_name not in name:
                    continue
                if ("mlp.experts." in name) and name not in params_dict:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    if name in params_dict.keys():
                        param = params_dict[name]
                        if "attention_sink_bias" in name:
                            start = get_parallel().attn_tp_rank * param.numel()
                            param.data.copy_(
                                loaded_weight[start : start + param.numel()]
                            )
                        else:
                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            weight_loader(param, loaded_weight)
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")

    def get_embed_and_head(self):
        assert (
            self.model is not None and self.lm_head is not None
        ), "get_embed_and_head() is not available in encoder_only mode"
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        assert (
            self.model is not None and self.lm_head is not None
        ), "set_embed_and_head() is not available in encoder_only mode"
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        if self.model is not None:
            self.model.load_kv_cache_scales(quantization_param_path)

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=getattr(config, "n_routed_experts", 1),
            num_groups=getattr(config, "n_group", None),
        )


# Keep the old Flash architecture name loadable while new configs use MiMoV2ForCausalLM.
class MiMoV2FlashForCausalLM(MiMoV2ForCausalLM):
    pass


EntryClass = [MiMoV2ForCausalLM, MiMoV2FlashForCausalLM]
