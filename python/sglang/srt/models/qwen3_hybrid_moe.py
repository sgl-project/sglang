import enum
import logging
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.qwen3_hybrid_moe import Qwen3HybridMoeConfig
from sglang.srt.distributed import (
    divide,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.attention.fla.layernorm_gated import RMSNorm as RMSNormGated
from sglang.srt.layers.attention.mamba.mamba import mamba_v2_sharded_weight_loader
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from sglang.srt.models.qwen2_moe import Qwen2MoeMLP, Qwen2MoeSparseMoeBlock
from sglang.srt.utils import add_prefix, is_cuda, make_layers, set_weight_attrs

logger = logging.getLogger(__name__)
_is_cuda = is_cuda()

import triton
import triton.language as tl


@triton.jit
def fused_qkvzba_split_reshape_cat_kernel(
    mixed_qkv,
    z,
    b,
    a,
    mixed_qkvzba,
    NUM_HEADS_QK: tl.constexpr,
    NUM_HEADS_V: tl.constexpr,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
):
    i_bs, i_qk = tl.program_id(0), tl.program_id(1)
    QKVZBA_DIM_T: tl.constexpr = (
        HEAD_QK * 2
        + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V * 2
        + NUM_HEADS_V // NUM_HEADS_QK * 2
    )
    QKV_DIM_T: tl.constexpr = HEAD_QK * 2 + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V
    q_end: tl.constexpr = HEAD_QK
    blk_q_ptr = (
        mixed_qkvzba
        + i_bs * NUM_HEADS_QK * QKVZBA_DIM_T
        + i_qk * QKVZBA_DIM_T
        + tl.arange(0, q_end)
    )
    k_end: tl.constexpr = q_end + HEAD_QK
    blk_k_ptr = (
        mixed_qkvzba
        + i_bs * NUM_HEADS_QK * QKVZBA_DIM_T
        + i_qk * QKVZBA_DIM_T
        + tl.arange(q_end, k_end)
    )
    v_end: tl.constexpr = k_end + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V
    blk_v_ptr = (
        mixed_qkvzba
        + i_bs * NUM_HEADS_QK * QKVZBA_DIM_T
        + i_qk * QKVZBA_DIM_T
        + tl.arange(k_end, v_end)
    )
    z_end: tl.constexpr = v_end + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V
    blk_z_ptr = (
        mixed_qkvzba
        + i_bs * NUM_HEADS_QK * QKVZBA_DIM_T
        + i_qk * QKVZBA_DIM_T
        + tl.arange(v_end, z_end)
    )
    blk_q_st_ptr = (
        mixed_qkv
        + i_bs * NUM_HEADS_QK * QKV_DIM_T
        + i_qk * HEAD_QK
        + tl.arange(0, HEAD_QK)
    )
    blk_k_st_ptr = (
        mixed_qkv
        + i_bs * NUM_HEADS_QK * QKV_DIM_T
        + NUM_HEADS_QK * HEAD_QK
        + i_qk * HEAD_QK
        + tl.arange(0, HEAD_QK)
    )
    blk_v_st_ptr = (
        mixed_qkv
        + i_bs * NUM_HEADS_QK * QKV_DIM_T
        + NUM_HEADS_QK * HEAD_QK * 2
        + i_qk * HEAD_V * NUM_HEADS_V // NUM_HEADS_QK
        + tl.arange(0, HEAD_V * NUM_HEADS_V // NUM_HEADS_QK)
    )
    blk_z_st_ptr = (
        z
        + i_bs * NUM_HEADS_V * HEAD_V
        + i_qk * HEAD_V * NUM_HEADS_V // NUM_HEADS_QK
        + tl.arange(0, HEAD_V * NUM_HEADS_V // NUM_HEADS_QK)
    )
    tl.store(blk_q_st_ptr, tl.load(blk_q_ptr))
    tl.store(blk_k_st_ptr, tl.load(blk_k_ptr))
    tl.store(blk_v_st_ptr, tl.load(blk_v_ptr))
    tl.store(blk_z_st_ptr, tl.load(blk_z_ptr))
    b_end: tl.constexpr = z_end + NUM_HEADS_V // NUM_HEADS_QK
    a_end: tl.constexpr = b_end + NUM_HEADS_V // NUM_HEADS_QK
    for i in tl.static_range(z_end, b_end):
        blk_b_ptr = (
            mixed_qkvzba + i_bs * NUM_HEADS_QK * QKVZBA_DIM_T + i_qk * QKVZBA_DIM_T + i
        )
        blk_b_st_ptr = (
            b + i_bs * NUM_HEADS_V + i_qk * NUM_HEADS_V // NUM_HEADS_QK + (i - z_end)
        )
        tl.store(blk_b_st_ptr, tl.load(blk_b_ptr))
    for i in tl.static_range(b_end, a_end):
        blk_a_ptr = (
            mixed_qkvzba + i_bs * NUM_HEADS_QK * QKVZBA_DIM_T + i_qk * QKVZBA_DIM_T + i
        )
        blk_a_st_ptr = (
            a + i_bs * NUM_HEADS_V + i_qk * NUM_HEADS_V // NUM_HEADS_QK + (i - b_end)
        )
        tl.store(blk_a_st_ptr, tl.load(blk_a_ptr))


def fused_qkvzba_split_reshape_cat(
    mixed_qkvzba,
    num_heads_qk,
    num_heads_v,
    head_qk,
    head_v,
):
    batch, seq_len = mixed_qkvzba.shape[0], 1
    qkv_dim_t = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    mixed_qkv = torch.empty(
        [batch * seq_len, qkv_dim_t],
        dtype=mixed_qkvzba.dtype,
        device=mixed_qkvzba.device,
    )
    z = torch.empty(
        [batch * seq_len, num_heads_v, head_v],
        dtype=mixed_qkvzba.dtype,
        device=mixed_qkvzba.device,
    )
    b = torch.empty(
        [batch * seq_len, num_heads_v],
        dtype=mixed_qkvzba.dtype,
        device=mixed_qkvzba.device,
    )
    a = torch.empty_like(b)
    grid = (batch * seq_len, num_heads_qk)
    fused_qkvzba_split_reshape_cat_kernel[grid](
        mixed_qkv,
        z,
        b,
        a,
        mixed_qkvzba,
        num_heads_qk,
        num_heads_v,
        head_qk,
        head_v,
        num_warps=1,
        num_stages=3,
    )
    return mixed_qkv, z, b, a


# g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
@triton.jit
def fused_gdn_gating_kernel(
    g,
    A_log,
    a,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> torch.Tensor:
    batch, num_heads = a.shape
    seq_len = 1
    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty_like(a, dtype=torch.float32)
    fused_gdn_gating_kernel[grid](
        g, A_log, a, dt_bias, seq_len, num_heads, beta, threshold, 8, num_warps=1
    )
    return g


class Qwen3GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen3HybridMoeConfig, layer_id: int) -> None:
        super().__init__()
        self.config = config
        self.attn_tp_rank = get_attention_tp_rank()
        self.attn_tp_size = get_attention_tp_size()
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
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)
        # projection of the input hidden states
        projection_size = self.key_dim * 2 + self.value_dim * 2 + self.num_v_heads * 2

        self.in_proj = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=projection_size,
            bias=False,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
        )

        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)

        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight,
            {
                "weight_loader": mamba_v2_sharded_weight_loader(
                    [
                        query_key_settings,
                        query_key_settings,
                        value_settings,
                    ],
                    self.attn_tp_size,
                    self.attn_tp_rank,
                )
            },
        )

        # selective projection used to make dt, B and C input dependent

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads // self.attn_tp_size))

        A = torch.empty(
            divide(self.num_v_heads, self.attn_tp_size), dtype=torch.float32
        ).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        set_weight_attrs(
            self.A_log, {"weight_loader": sharded_weight_loader(0, self.attn_tp_rank)}
        )
        set_weight_attrs(
            self.dt_bias, {"weight_loader": sharded_weight_loader(0, self.attn_tp_rank)}
        )

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
        else:  # mamba2 or hgrn2 style
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
        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_results=False,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
        )

    def fix_query_key_value_ordering(
        self,
        mixed_qkvzba,
    ):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvzba`.
        """
        new_tensor_shape = mixed_qkvzba.size()[:-1] + (
            self.num_k_heads // self.attn_tp_size,
            (
                self.head_k_dim
                + self.head_k_dim
                + (self.head_v_dim + self.head_v_dim + 2)
                * self.num_v_heads
                // self.num_k_heads
            ),
        )
        mixed_qkvzba = mixed_qkvzba.view(*new_tensor_shape)

        split_arg_list = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads,
        ]

        # [b, sq, ng, (hn + hn + np/ng * hn + np/ng + np/ng)]
        # --> [b, sq, ng, hn], [b, sq, ng, hn], [b, sq, ng, np/ng * hn], [b, sq, ng, np/ng * hn], [b, sq, ng, np/ng], [b, sq, ng, np/ng]
        (query, key, value, z, b, a) = torch.split(mixed_qkvzba, split_arg_list, dim=2)

        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.size(0), -1, self.head_v_dim)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b = b.reshape(b.size(0), self.num_v_heads // self.attn_tp_size)
        a = a.reshape(a.size(0), self.num_v_heads // self.attn_tp_size)

        return query, key, value, z, b, a

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        seq_len, _ = hidden_states.shape
        has_prefill = forward_batch.forward_mode.is_prefill()

        projected_states, _ = self.in_proj(hidden_states)

        if self.num_v_heads // self.num_k_heads in [1, 2, 4] and not has_prefill:
            mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat(
                projected_states,
                triton.cdiv(self.num_k_heads, self.attn_tp_size),
                triton.cdiv(self.num_v_heads, self.attn_tp_size),
                self.head_k_dim,
                self.head_v_dim,
            )
        else:
            query, key, value, z, b, a = self.fix_query_key_value_ordering(
                projected_states,
            )
            query, key, value = map(
                lambda x: x.reshape(x.shape[0], -1), (query, key, value)
            )
            mixed_qkv = torch.cat((query, key, value), dim=-1)
        # mixed_qkv = rearrange(mixed_qkv, "b l d -> b d l")

        # 2. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        kwargs = {
            "mixed_qkv": mixed_qkv,
            "conv_weights": conv_weights,
            "bias": self.conv1d.bias,
            "activation": self.activation,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "attention_tp_size": self.attn_tp_size,
            "head_k_dim": self.head_k_dim,
            "head_v_dim": self.head_v_dim,
            "a": a,
            "b": b,
            "A_log": self.A_log,
            "dt_bias": self.dt_bias,
            "layer_id": self.layer_id,
            "seq_len": seq_len,
            "z": z,
        }

        core_attn_out = forward_batch.attn_backend.forward(
            q=None,
            k=None,
            v=None,
            layer=None,
            forward_batch=forward_batch,
            **kwargs,
        )

        if self.share_norm:
            z_shape_og = z.shape
            # reshape input data into 2D tensor
            core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
            z = z.reshape(-1, z.shape[-1])
            core_attn_out = self.norm(core_attn_out, z)
            core_attn_out = core_attn_out.reshape(z_shape_og)
            core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:-2], -1)
        else:
            z = z.reshape(*z.shape[:-2], -1)
            core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:-2], -1)
            core_attn_out = self.norm(core_attn_out, z)

        output, _ = self.out_proj(core_attn_out)
        return output


class Qwen3HybridLinearDecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3HybridMoeConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.linear_attn = Qwen3GatedDeltaNet(config, layer_id)

        # Qwen3HybridMoE all layers are sparse and have no nextn now
        self.is_layer_sparse = True
        is_previous_layer_sparse = True
        self.layer_id = layer_id

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
        )

        if self.is_layer_sparse:
            self.mlp = Qwen2MoeSparseMoeBlock(
                layer_id=layer_id, config=config, quant_config=quant_config
            )
        else:
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
            )
        if getattr(
            config, "use_gemma_rms_norm", getattr(config, "apply_layernorm_1p", False)
        ):
            logger.warning_once(
                "Using Gemma RMSNorm for input normalization and post attn normalization."
            )
            self.input_layernorm = GemmaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_attention_layernorm = GemmaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        **kwargs,
    ):
        forward_batch = kwargs.get("forward_batch", None)

        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )

        if not forward_batch.forward_mode.is_idle():
            hidden_states = self.linear_attn(
                hidden_states,
                forward_batch,
            )
        # Fully Connected
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )

        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )
        hidden_states = self.mlp(hidden_states, forward_batch, use_reduce_scatter)

        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )

        return hidden_states, residual


class Qwen3HybridAttentionDecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3HybridMoeConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "none"
        )
        self.hidden_size = config.hidden_size
        self.attn_tp_rank = get_attention_tp_rank()
        self.attn_tp_size = get_attention_tp_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % self.attn_tp_size == 0
        self.num_heads = self.total_num_heads // self.attn_tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= self.attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % self.attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert self.attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.attn_tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = getattr(config, "rope_theta", 10000)
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.rope_scaling = getattr(config, "rope_scaling", None)
        self.layer_id = layer_id

        self.attn_output_gate = getattr(config, "attn_output_gate", False)
        if self.attn_output_gate:
            logger.warning_once("using attn output gate!")

        if hasattr(config, "rotary_percent"):
            rotary_dim = int(self.head_dim * config.rotary_percent)
        elif hasattr(config, "partial_rotary_factor"):
            rotary_dim = int(self.head_dim * config.partial_rotary_factor)
        elif hasattr(config, "attn_rotary_emb"):
            rotary_dim = int(config.attn_rotary_emb)  # for backward compatibility
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
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=f"{prefix}.attn",
        )

        # Qwen3HybridMoE all layers are sparse and have no nextn now
        self.is_layer_sparse = True
        is_previous_layer_sparse = True

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
        )

        if self.is_layer_sparse:
            self.mlp = Qwen2MoeSparseMoeBlock(
                layer_id=layer_id, config=config, quant_config=quant_config
            )
        else:
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
            )
        if getattr(
            config, "use_gemma_rms_norm", getattr(config, "apply_layernorm_1p", False)
        ):
            logger.warning_once(
                "Using Gemma RMSNorm for input normalization and post attn normalization."
            )
            self.input_layernorm = GemmaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_attention_layernorm = GemmaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

        if self.config.use_qk_norm:
            if getattr(
                config,
                "use_gemma_rms_norm",
                getattr(config, "apply_layernorm_1p", False),
            ):
                logger.warning_once(
                    "Using Gemma RMSNorm for Q normalization and K normalization."
                )
                self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
                self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            else:
                self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
                self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
        )

        self.alt_stream = alt_stream

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # overlap qk norm
        if self.alt_stream is not None and get_is_capture_mode():
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            q_by_head = q.reshape(-1, self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            with torch.cuda.stream(self.alt_stream):
                k_by_head = k.reshape(-1, self.head_dim)
                k_by_head = self.k_norm(k_by_head)
            current_stream.wait_stream(self.alt_stream)
        else:
            q_by_head = q.reshape(-1, self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            k_by_head = k.reshape(-1, self.head_dim)
            k_by_head = self.k_norm(k_by_head)
        q = q_by_head.view(q.shape)
        k = k_by_head.view(k.shape)
        return q, k

    def self_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)

        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if self.config.use_qk_norm:
            q, k = self._apply_qk_norm(q, k)

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
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ):
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )

        if not forward_batch.forward_mode.is_idle():
            hidden_states = self.self_attention(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        # Fully Connected
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )
        hidden_states = self.mlp(hidden_states, forward_batch, use_reduce_scatter)

        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )

        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "attention": Qwen3HybridAttentionDecoderLayer,
    "linear_attention": Qwen3HybridLinearDecoderLayer,
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

        alt_stream = torch.cuda.Stream() if _is_cuda else None

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            enable_tp=not is_dp_attention_enabled(),
        )

        def get_layer(idx: int, prefix: str):
            layer_class = ALL_DECODER_LAYER_TYPES[config.layers_block_type[idx]]
            return layer_class(
                config,
                idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            )

        # self.start_layer, self.end_layer, self.layers = make_layers(
        self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )

        if getattr(
            config, "use_gemma_rms_norm", getattr(config, "apply_layernorm_1p", False)
        ):
            logger.warning_once("Using Gemma RMSNorm for final normalization.")
            self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.infer_count = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        # mamba_cache_params: MambaCacheParams,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # pass a sequence index tensor, that is required for
        # proper continuous batching computation including
        # chunked prefill
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                layer_id=i,
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
            )

        if not forward_batch.forward_mode.is_idle():
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

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
            use_attn_tp_group=global_server_args_dict["enable_dp_lm_head"],
        )
        self.lm_head = self.lm_head.float()
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        hidden_states = self.model(input_ids, positions, forward_batch, inputs_embeds)

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_mtp: bool = False
    ) -> Set[str]:
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
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:

            if is_mtp and name not in [
                "fc.weight",
                "pre_fc_norm_embedding.weight",
                "pre_fc_norm_hidden.weight",
            ]:
                name = f"model.{name}"

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
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader")
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
                    # if is_pp_missing_parameter(name, self):
                    #     continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=None,
        )


EntryClass = Qwen3HybridMoEForCausalLM
