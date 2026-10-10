import logging
from array import array
from contextlib import nullcontext
from typing import Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from sglang.kernels.ops.attention.fla.fused_norm_gate import FusedRMSNormGated
from sglang.kernels.ops.layernorm.mhc import hc_contract
from sglang.kernels.ops.layernorm.mhc import hc_post as _hc_post_fn
from sglang.kernels.ops.layernorm.mhc import hc_pre as _hc_pre_fn
from sglang.srt.batch_overlap.two_batch_overlap import (
    model_forward_stages,
)
from sglang.srt.configs.glm5_next import Glm5NextConfig, Glm5NextTextConfig
from sglang.srt.configs.model_config import is_deepseek_dsa
from sglang.srt.distributed.utils import divide
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import (
    get_global_expert_distribution_recorder,
)
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.attention import vision_utils
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.aux_hidden_states import AuxHiddenStateList
from sglang.srt.layers.conv import Conv2dLayer
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layer_boundary import (
    PLAIN_RESIDUAL_OPS,
    MHCState,
    SumGroup,
    append_stages,
    declare_attn,
    declare_ffn,
    get_attn_tp_context,
    is_dense_ffn_fully_dp,
)
from sglang.srt.layers.layer_boundary.residual import access as residual_access
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
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
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.utils import (
    get_moe_a2a_backend,
    get_moe_runner_backend,
    is_shared_experts_fusion_disabled,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils.common import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    get_embedding_tp_kwargs,
)
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
    prepare_mm_inputs,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    PPProxyTensors,
)
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc import (
    apply_mhc_post_pre_boundary,
    is_cross_layer_mhc_fusion_enabled,
    try_aiter_mhc_pre_quant,
)
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.srt.models.deepseek_common.utils import (
    _device_sm,
    _is_cuda,
    _use_aiter_gfx95,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
from sglang.srt.models.deepseek_v2 import DeepseekV2MLP as Glm5NextMLP
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE as Glm5NextMoE
from sglang.srt.models.glm4v import glm4v_vision_reduces_over_attn_tp
from sglang.srt.models.glm_ocr import (
    GlmOcrRMSNorm,
    GlmOcrVisionBlock,
    GlmOcrVisionMLP,
    GlmOcrVisionModel,
    GlmOcrVisionPatchEmbed,
    GlmOcrVisionPatchMerger,
)
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.multimodal.mm_utils import (
    run_dp_presharded_mrope_vision_model,
    run_dp_sharded_mrope_vision_model,
)
from sglang.srt.runtime_context import (
    get_lora,
    get_mm,
    get_parallel,
    get_spec,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.common import (
    BumpAllocator,
    LazyValue,
    add_prefix,
    log_info_on_rank0,
    make_pp_layers,
    set_weight_attrs,
)

if _use_aiter_gfx95:
    from sglang.srt.layers.rocm_linear_utils import (
        get_dsv3_gemm_output_zero_allocator_size,
    )

logger = logging.getLogger(__name__)

# Matches DeepSeek-V4's _MHC_POST_MULT_VALUE; the fused and unfused boundaries
# must agree on it.
_MHC_POST_MULT_VALUE = 2.0

# Conservative cap, not the crossover: at GLM-5.3-Flash's hc_mult=4 and
# hidden_size=4096 the fusion wins to 16 tokens and reaches parity at 24, with
# 17-23 unmeasured. Past it the pre-norm GEMM drops mhc_pre's split-K kernel.
_MHC_FUSED_BOUNDARY_MAX_TOKENS = 16
_MHC_FUSED_BOUNDARY_LARGE_TOKENS = (4096, 8192, 16384, 131072)
_MHC_QUANT_BOUNDARY_TOKENS = (
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
)


@torch.compile
def swiglu_clamped(y: torch.Tensor, limit: float):
    gate, up = torch.chunk(y, 2, dim=-1)
    gate = torch.clamp(gate, max=limit)
    up = torch.clamp(up, min=-limit, max=limit)
    return F.silu(gate) * up


class Glm5NextVisionMLP(GlmOcrVisionMLP):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        swiglu_limit: float,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            bias=bias,
            quant_config=quant_config,
            prefix=prefix,
            use_data_parallel=use_data_parallel,
        )
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor):
        gate_up, _ = self.gate_up_proj(x)
        x = swiglu_clamped(gate_up, self.swiglu_limit)
        x, _ = self.down_proj(x)
        return x


class Glm5NextVisionPatchMerger(GlmOcrVisionPatchMerger):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        swiglu_limit: float,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__(
            d_model=d_model,
            context_dim=context_dim,
            quant_config=quant_config,
            bias=bias,
            prefix=prefix,
            use_data_parallel=use_data_parallel,
        )
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor):
        x, _ = self.proj(x)
        x = self.extra_activation_func(self.post_projection_norm(x))
        gate_up, _ = self.gate_up_proj(x)
        x = swiglu_clamped(gate_up, self.swiglu_limit)
        x, _ = self.down_proj(x)
        return x


class Glm5NextVisionBlock(GlmOcrVisionBlock):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        num_heads: int,
        swiglu_limit: float,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        attn_qkv_bias: bool = True,
        num_dummy_heads: int = 0,
        rms_norm_eps: float = 1e-5,
        use_data_parallel: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        self.norm1 = RMSNorm(dim, eps=rms_norm_eps)
        self.norm2 = RMSNorm(dim, eps=rms_norm_eps)
        self.attn = VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            use_qkv_parallel=True,
            qkv_bias=attn_qkv_bias,
            proj_bias=True,
            qk_normalization_by_head_size=True,
            flatten_batch=True,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            num_dummy_heads=num_dummy_heads,
            use_data_parallel=use_data_parallel,
            use_dp_attention_reduce=glm4v_vision_reduces_over_attn_tp(
                use_data_parallel
            ),
        )
        self.mlp = Glm5NextVisionMLP(
            dim,
            intermediate_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
            use_data_parallel=use_data_parallel,
            swiglu_limit=swiglu_limit,
        )


class Glm5NextVisionModel(GlmOcrVisionModel):
    def __init__(
        self,
        vision_config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.out_hidden_size = vision_config.out_hidden_size
        self.intermediate_size = vision_config.intermediate_size
        self.use_data_parallel = use_data_parallel

        self.patch_embed = GlmOcrVisionPatchEmbed(
            patch_size=vision_config.patch_size,
            temporal_patch_size=vision_config.temporal_patch_size,
            in_channels=vision_config.in_channels,
            hidden_size=self.hidden_size,
        )

        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = get_rope(
            head_size=head_dim,
            rotary_dim=head_dim // 2,
            max_position=8192,
            base=10000.0,
            is_neox_style=True,
        )

        self.blocks = nn.ModuleList(
            [
                Glm5NextVisionBlock(
                    dim=self.hidden_size,
                    intermediate_dim=self.intermediate_size,
                    num_heads=self.num_heads,
                    quant_config=quant_config,
                    prefix=add_prefix(f"blocks.{layer_idx}", prefix),
                    rms_norm_eps=vision_config.rms_norm_eps,
                    attn_qkv_bias=vision_config.attention_bias,
                    use_data_parallel=use_data_parallel,
                    swiglu_limit=vision_config.swiglu_limit,
                )
                for layer_idx in range(vision_config.depth)
            ]
        )
        projection_intermediate_size = getattr(
            vision_config, "projection_intermediate_size", None
        )
        self.merger = Glm5NextVisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=(
                projection_intermediate_size
                if projection_intermediate_size is not None
                else vision_config.intermediate_size
            ),
            quant_config=quant_config,
            bias=False,
            prefix=add_prefix("merger", prefix),
            use_data_parallel=use_data_parallel,
            swiglu_limit=vision_config.swiglu_limit,
        )

        # These non-overlapping patches are equivalent to unfold + linear and
        # avoid MIOpen's expensive per-shape convolution search on ROCm.
        self.downsample = Conv2dLayer(
            in_channels=vision_config.hidden_size,
            out_channels=vision_config.out_hidden_size,
            kernel_size=vision_config.spatial_merge_size,
            stride=vision_config.spatial_merge_size,
            disable_linear=False,
        )
        self.post_layernorm = GlmOcrRMSNorm(
            vision_config.hidden_size, eps=vision_config.rms_norm_eps
        )


class Glm5NextLinearAttention(nn.Module):
    _PACKED_MODULES_MAPPING = {
        "fused_qkvbfg_a_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
            "b_proj",
            "f_a_proj",
            "g_a_proj",
        ],
        "fused_bfg_a_proj": ["b_proj", "f_a_proj", "g_a_proj"],
        "fused_fg_b_proj": ["f_b_proj", "g_b_proj"],
    }

    @classmethod
    def _can_fuse_proj(
        cls,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
        *fused_projs: str,
    ) -> bool:
        if get_lora().enable_lora or get_lora().lora_paths:
            return False
        if quant_config is None:
            return True
        if quant_config.get_name() not in {
            "fp8",
            "mxfp8",
            "modelopt_fp8",
            "modelopt_fp4",
            "modelopt_mixed",
            "quark",
        }:
            return False

        source_projs = [
            proj
            for fused_proj in fused_projs
            for proj in cls._PACKED_MODULES_MAPPING[fused_proj]
        ]
        if "fused_qkvbfg_a_proj" in fused_projs:
            source_projs.append("qkv_proj")
        return all(
            quant_config.is_linear_unquantized(f"{prefix}.{proj}")
            for proj in source_projs
        )

    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        config: Glm5NextTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        rms_norm_eps: float = 1e-5,
        prefix: str = "",
        reduce_results: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        from sglang.srt.runtime_context import (
            linear_attn_parallel_group,
            linear_attn_tp_size,
        )

        head_shard_size = linear_attn_tp_size()
        head_parallel_group = linear_attn_parallel_group()

        self.hidden_size = hidden_size
        self.config = config
        self.head_dim = config.linear_attn_config["head_dim"]
        self.num_heads = config.linear_attn_config["num_heads"]
        self.num_k_heads = config.linear_attn_config["num_heads"]
        self.num_v_heads = config.linear_attn_config["num_heads"]
        self.head_k_dim = config.linear_attn_config["head_dim"]
        self.head_v_dim = config.linear_attn_config["head_dim"]
        self.layer_idx = layer_idx
        self.prefix = prefix
        assert self.num_heads % head_shard_size == 0
        self.local_num_heads = divide(self.num_heads, head_shard_size)

        projection_size = self.head_dim * self.num_heads
        self.conv_size = config.linear_attn_config["short_conv_kernel_size"]

        self.do_fuse_qkvbfg = self._can_fuse_proj(
            quant_config, prefix, "fused_qkvbfg_a_proj", "fused_fg_b_proj"
        )
        self.fuse_bfg = not self.do_fuse_qkvbfg and self._can_fuse_proj(
            quant_config, prefix, "fused_bfg_a_proj", "fused_fg_b_proj"
        )
        if self.do_fuse_qkvbfg:
            self.qkvb_sizes = [
                projection_size,
                projection_size,
                projection_size,
                self.num_heads,
            ]
            self.fg_sizes = [self.head_dim, self.head_dim]

            self.fused_qkvbfg_a_proj = MergedColumnParallelRepeatedLinear(
                self.hidden_size,
                self.qkvb_sizes,
                self.fg_sizes,
                quant_config=None,
                prefix=f"{prefix}.fused_qkvbfg_a_proj",
                parallel_group=head_parallel_group,
            )
            self.split_sizes = [
                3 * projection_size // head_shard_size,
                self.num_heads // head_shard_size,
                2 * self.head_dim,
            ]
            self.fused_fg_b_proj = ColumnParallelBatchedLinear(
                2,
                self.head_dim,
                projection_size,
                dtype=self.fused_qkvbfg_a_proj.params_dtype,
                parallel_group=head_parallel_group,
            )
        else:
            self.qkv_proj = QKVParallelLinear(
                self.hidden_size,
                self.head_dim,
                self.num_heads,
                self.num_k_heads,
                bias=False,
                quant_config=quant_config,
                parallel_group=head_parallel_group,
                prefix=f"{prefix}.qkv_proj",
            )

            if self.fuse_bfg:
                self.fused_bfg_a_proj = MergedColumnParallelRepeatedLinear(
                    self.hidden_size,
                    [self.num_heads],
                    [self.head_dim, self.head_dim],
                    quant_config=None,
                    prefix=f"{prefix}.fused_bfg_a_proj",
                    parallel_group=head_parallel_group,
                )
                self.bfg_split_sizes = [self.local_num_heads, 2 * self.head_dim]
                self.fused_fg_b_proj = ColumnParallelBatchedLinear(
                    2,
                    self.head_dim,
                    projection_size,
                    dtype=self.fused_bfg_a_proj.params_dtype,
                    parallel_group=head_parallel_group,
                )
            else:
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
                    parallel_group=head_parallel_group,
                )

                self.b_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.num_heads,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.b_proj",
                    parallel_group=head_parallel_group,
                )

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
                    parallel_group=head_parallel_group,
                )

        self.dt_bias = nn.Parameter(
            torch.empty(divide(projection_size, head_shard_size), dtype=torch.float32)
        )

        set_weight_attrs(
            self.dt_bias,
            {
                "weight_loader": sharded_weight_loader(
                    0, parallel_group=head_parallel_group
                )
            },
        )

        self.qkv_conv1d = MergedColumnParallelLinear(
            input_size=self.conv_size,
            output_sizes=[projection_size, projection_size, projection_size],
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.qkv_conv1d",
            parallel_group=head_parallel_group,
        )
        # ColumnParallelLinear's loader cannot reshape conv1d weights, so add the
        # singleton dimension after construction.
        self.qkv_conv1d.weight.data = self.qkv_conv1d.weight.data.unsqueeze(1)

        self.A_log = nn.Parameter(
            torch.empty(1, 1, self.local_num_heads, 1, dtype=torch.float32)
        )
        set_weight_attrs(
            self.A_log,
            {
                "weight_loader": sharded_weight_loader(
                    2, parallel_group=head_parallel_group
                )
            },
        )

        self.o_norm = FusedRMSNormGated(
            self.head_dim, eps=rms_norm_eps, activation="sigmoid"
        )
        self.o_proj = RowParallelLinear(
            projection_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            reduce_results=reduce_results,
            parallel_group=head_parallel_group,
        )

        conv_weights = self.qkv_conv1d.weight.squeeze(1)
        bias = self.qkv_conv1d.bias

        self.attn = RadixLinearAttention(
            layer_id=self.layer_idx,
            num_q_heads=self.local_num_heads,
            num_k_heads=self.local_num_heads,
            num_v_heads=self.local_num_heads,
            head_q_dim=self.head_k_dim,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            conv_weights=conv_weights,
            bias=bias,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
        )

        self.attn.lower_bound = config.linear_attn_config.get("gate_lower_bound", None)

    def forward_qkvbfg(self, hidden_states: torch.Tensor, forward_batch: ForwardBatch):
        qkv, _ = self.qkv_proj(hidden_states)

        if self.fuse_bfg:
            fused_states = self.fused_bfg_a_proj(hidden_states)
            beta, fg_a_states = torch.split(fused_states, self.bfg_split_sizes, dim=-1)
            forget_gate, g_proj_states = self.fused_fg_b_proj(
                fg_a_states.view(-1, 2, self.head_dim).transpose(0, 1)
            )
        else:
            beta = self.b_proj(hidden_states)[0]
            forget_gate = self.f_b_proj(self.f_a_proj(hidden_states)[0])[0]
            g_proj_states = self.g_b_proj(self.g_a_proj(hidden_states)[0])[0]

        return (
            qkv,
            beta,
            forget_gate,
            g_proj_states,
        )

    def forward_qkvbfg_fused(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ):
        fused_states = self.fused_qkvbfg_a_proj(hidden_states)

        qkv, beta, fg_a_states = torch.split(fused_states, self.split_sizes, dim=-1)

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
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_idle():
            return hidden_states

        from sglang.srt.layers.cp.interleave import (
            cp_interleave_to_sequence_order,
            cp_sequence_to_interleave_order,
        )

        gathered_rows = hidden_states.shape[0]
        hidden_states = cp_interleave_to_sequence_order(hidden_states, forward_batch)

        if self.do_fuse_qkvbfg:
            mixed_qkv, beta, forget_gate, g_proj_states = self.forward_qkvbfg_fused(
                hidden_states, forward_batch
            )
        else:
            mixed_qkv, beta, forget_gate, g_proj_states = self.forward_qkvbfg(
                hidden_states, forward_batch
            )

        if not forward_batch.forward_mode.is_decode():
            forget_gate = forget_gate.unsqueeze(0)
        beta = beta.unsqueeze(0)

        core_attn_out = self.attn(
            forward_batch,
            mixed_qkv=mixed_qkv,
            a=forget_gate,
            b=beta,
        )

        norm_gate = g_proj_states.unflatten(-1, (-1, self.head_dim))
        core_attn_out = self.o_norm(core_attn_out, norm_gate)
        core_attn_out = core_attn_out.squeeze(0).flatten(-2)

        return cp_sequence_to_interleave_order(
            self.o_proj(core_attn_out)[0], forward_batch, gathered_rows
        )


class Glm5NextDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Glm5NextTextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        moe_quant_config_override: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        rope_theta = config.rope_theta
        rope_scaling = config.rope_scaling
        max_position_embeddings = config.max_position_embeddings
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        )
        self.layer_id = layer_id
        self.is_nextn = is_nextn
        self.is_linear_attn = config.is_kda_layer(layer_id)

        if self.is_linear_attn:
            self.self_attn = Glm5NextLinearAttention(
                layer_idx=layer_id,
                hidden_size=config.hidden_size,
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                rms_norm_eps=config.rms_norm_eps,
                reduce_results=False,
            )
        else:
            self.self_attn = DeepseekV2AttentionMLA(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                layer_id=layer_id,
                reduce_results=False,
                prefix=add_prefix("self_attn", prefix),
                alt_stream=alt_stream,
                is_nextn=is_nextn,
                skip_rope=True,
            )

        if config.q_lora_rank is None and envs.SGLANG_USE_AG_AFTER_QLORA.get():
            raise ValueError(
                "SGLANG_USE_AG_AFTER_QLORA only supports the model with q_lora_rank"
            )

        self.is_layer_sparse = self._is_layer_sparse(layer_id, is_nextn=is_nextn)
        is_next_layer_sparse = self._is_layer_sparse(layer_id + 1, is_nextn=False)

        if self.is_layer_sparse:
            self.mlp = Glm5NextMoE(
                config=config,
                quant_config=moe_quant_config_override or quant_config,
                prefix=add_prefix("mlp", prefix),
                layer_id=self.layer_id,
                alt_stream=alt_stream,
                is_nextn=is_nextn,
                reduce_results=False,
            )
        else:
            mlp_parallel_group = "replicated" if is_dense_ffn_fully_dp() else "tp"
            self.mlp = Glm5NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                parallel_group=mlp_parallel_group,
                swiglu_limit=config.swiglu_limit,
                reduce_results=False,
                allow_fused_down=False,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        if self.config.mhc:
            hc_mult = config.hc_mult
            mix_hc = (2 + hc_mult) * hc_mult
            hc_dim = hc_mult * config.hidden_size

            # Keep mHC params directly on the decoder layer so their names match
            # the checkpoint verbatim.
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
            self.hc_attn_fn = nn.Parameter(
                torch.empty(mix_hc, hc_dim, dtype=torch.float32)
            )

            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
            self.hc_ffn_fn = nn.Parameter(
                torch.empty(mix_hc, hc_dim, dtype=torch.float32)
            )
            for stage in ("attn", "ffn"):
                self.register_buffer(f"_hc_{stage}_fn_packed", None, persistent=False)
                setattr(self, f"_hc_{stage}_fn_packed_source", None)

        terminal = layer_id == (1 if is_nextn else config.num_hidden_layers) - 1
        residual = PLAIN_RESIDUAL_OPS
        if self.config.mhc:
            fuse_mhc_boundaries = is_cross_layer_mhc_fusion_enabled()
            residual = MHCState(
                hc_mult=config.hc_mult,
                hc_attn_pre=self.hc_attn_pre,
                hc_ffn_pre=self.hc_ffn_pre,
                hc_post=self.hc_post,
                hc_ffn_post_pre=(self.hc_ffn_post_pre if fuse_mhc_boundaries else None),
                hc_attn_post_pre=(
                    self.hc_attn_post_pre if fuse_mhc_boundaries else None
                ),
                hc_attn_pre_quant=(
                    self.hc_attn_pre_quant if fuse_mhc_boundaries else None
                ),
                defer_ffn_update=fuse_mhc_boundaries and not terminal,
                is_last_layer=terminal,
            ).residual_ops()
        self.attn_boundary, self.ffn_boundary = append_stages(
            (
                declare_attn(
                    read=residual.attn_readout,
                    update=residual.attn_update,
                    tp_group=SumGroup.TP
                    if self.is_linear_attn and get_parallel().enable_cp_tp_group_sharing
                    else SumGroup.ATTN_TP,
                ),
                self.input_layernorm,
                {
                    "qkv_latent_func": self.self_attn.prepare_qkv_latent
                    if not self.is_linear_attn
                    else None,
                },
            ),
            (
                declare_ffn(
                    sparse=self.is_layer_sparse,
                    next_layer_sparse=is_next_layer_sparse,
                    read=residual.ffn_readout,
                    update=residual.ffn_update,
                ),
                self.post_attention_layernorm,
            ),
        )

    def _hc_pre(
        self, hc_fn, hc_scale, hc_base, hidden_states, out_norm_weight, out_norm_eps
    ):
        return _hc_pre_fn(
            x=hidden_states,
            hc_fn=hc_fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            hc_mult=self.config.hc_mult,
            rms_eps=self.config.rms_norm_eps,
            hc_eps=self.config.hc_eps,
            sinkhorn_iters=self.config.hc_sinkhorn_iters,
            post_mult_value=_MHC_POST_MULT_VALUE,
            hc_norm_weight=None,
            out_norm_weight=out_norm_weight,
            out_norm_eps=out_norm_eps,
        )

    def hc_attn_pre(self, hidden_states, out_norm_weight, out_norm_eps):
        return self._hc_pre(
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            hidden_states,
            out_norm_weight,
            out_norm_eps,
        )

    def hc_ffn_pre(self, hidden_states, out_norm_weight, out_norm_eps):
        return self._hc_pre(
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            hidden_states,
            out_norm_weight,
            out_norm_eps,
        )

    def _get_hc_fn_packed(self, stage: str):
        weight = getattr(self, f"hc_{stage}_fn")
        packed_name = f"_hc_{stage}_fn_packed"
        source_name = f"_hc_{stage}_fn_packed_source"
        version = None if weight.is_inference() else weight._version
        source = getattr(self, source_name)
        source_is_valid = (
            source is not None
            and weight is source[0]
            and weight.data_ptr() == source[1]
            and (source[2] is None or version == source[2])
        )
        packed = getattr(self, packed_name)
        if packed is None or not source_is_valid:
            from aiter.ops.mhc import mhc_shuffle_fn

            packed = mhc_shuffle_fn(weight)
            setattr(self, packed_name, packed)
            setattr(
                self,
                source_name,
                (
                    weight,
                    weight.data_ptr(),
                    version,
                ),
            )
        return packed

    def _hc_post_pre(
        self,
        stage,
        hidden_states,
        residual,
        h_res,
        h_post,
        out_norm_weight,
        out_norm_eps,
        is_prefill,
    ):
        assert self.config.mhc, "mHC post/pre fusion requires config.mhc=True"
        num_tokens, hidden_size = hidden_states.shape
        hc_mult = self.config.hc_mult
        use_large_m_fused = (
            is_prefill
            and _use_aiter_gfx95
            and hidden_size == 4096
            and hc_mult == 4
            and num_tokens in _MHC_FUSED_BOUNDARY_LARGE_TOKENS
        )
        if num_tokens > _MHC_FUSED_BOUNDARY_MAX_TOKENS and not use_large_m_fused:
            return None
        hc_fn = (
            self._get_hc_fn_packed(stage)
            if use_large_m_fused
            else getattr(self, f"hc_{stage}_fn")
        )
        return_quant = (
            stage == "attn"
            and use_large_m_fused
            and self._can_use_hc_attn_prequant(num_tokens)
        )
        fused = apply_mhc_post_pre_boundary(
            layer_input=hidden_states,
            residual=residual.view(num_tokens, hc_mult, hidden_size),
            post=h_post.view(num_tokens, hc_mult),
            comb=h_res.view(num_tokens, hc_mult, hc_mult),
            hc_fn=hc_fn,
            hc_scale=getattr(self, f"hc_{stage}_scale"),
            hc_base=getattr(self, f"hc_{stage}_base"),
            hc_mult=hc_mult,
            rms_eps=self.config.rms_norm_eps,
            hc_eps=self.config.hc_eps,
            hc_post_mult=_MHC_POST_MULT_VALUE,
            sinkhorn_iters=self.config.hc_sinkhorn_iters,
            norm_weight=out_norm_weight,
            norm_eps=out_norm_eps,
            fn_transpose=True,
            force_fused=use_large_m_fused,
            w_preshuffle_bf16=use_large_m_fused,
            return_quant=return_quant,
        )
        if fused is None:
            return None
        if len(fused) == 6:
            next_residual, layer_input, post, comb, norm_fused, prequant = fused
        else:
            next_residual, layer_input, post, comb, norm_fused = fused
            prequant = None
        result = (
            layer_input,
            next_residual.reshape(num_tokens, -1),
            comb.reshape(num_tokens, hc_mult * hc_mult),
            post.reshape(num_tokens, hc_mult),
            norm_fused,
        )
        return result if prequant is None else (*result, prequant)

    def _can_use_hc_attn_prequant(self, num_tokens: int) -> bool:
        consumer = getattr(self.self_attn, "can_consume_mhc_prequant", None)
        return (
            self.is_linear_attn
            and _use_aiter_gfx95
            and num_tokens in _MHC_QUANT_BOUNDARY_TOKENS
            and callable(consumer)
            and consumer(num_tokens)
        )

    def hc_attn_pre_quant(self, hidden_states, out_norm_weight, out_norm_eps):
        num_tokens, hc_hidden_size = hidden_states.shape
        hidden_size = hc_hidden_size // self.config.hc_mult
        if (
            hidden_size != 4096
            or out_norm_weight is None
            or not self._can_use_hc_attn_prequant(num_tokens)
        ):
            return None
        quantized = try_aiter_mhc_pre_quant(
            residual=hidden_states.view(num_tokens, self.config.hc_mult, hidden_size),
            hc_fn=self.hc_attn_fn,
            hc_scale=self.hc_attn_scale,
            hc_base=self.hc_attn_base,
            norm_weight=out_norm_weight,
            rms_eps=self.config.rms_norm_eps,
            hc_eps=self.config.hc_eps,
            hc_post_mult=_MHC_POST_MULT_VALUE,
            sinkhorn_iters=self.config.hc_sinkhorn_iters,
            norm_eps=out_norm_eps,
        )
        if quantized is None:
            return None
        layer_input, post, comb, norm_fused, prequant = quantized
        return (
            layer_input,
            comb.reshape(num_tokens, self.config.hc_mult**2),
            post.reshape(num_tokens, self.config.hc_mult),
            norm_fused,
            prequant,
        )

    def hc_attn_post_pre(
        self,
        hidden_states,
        residual,
        h_res,
        h_post,
        out_norm_weight,
        out_norm_eps,
        is_prefill,
    ):
        return self._hc_post_pre(
            "attn",
            hidden_states,
            residual,
            h_res,
            h_post,
            out_norm_weight,
            out_norm_eps,
            is_prefill,
        )

    def hc_ffn_post_pre(
        self,
        hidden_states,
        residual,
        h_res,
        h_post,
        out_norm_weight,
        out_norm_eps,
        is_prefill,
    ):
        return self._hc_post_pre(
            "ffn",
            hidden_states,
            residual,
            h_res,
            h_post,
            out_norm_weight,
            out_norm_eps,
            is_prefill,
        )

    def hc_post(self, hidden_states, residual, h_res, h_post):
        assert self.config.mhc, "hc_post is only valid when config.mhc=True"
        return _hc_post_fn(
            x=hidden_states,
            residual=residual,
            h_post=h_post,
            h_res=h_res,
            hc_mult=self.config.hc_mult,
        )

    def _is_layer_sparse(self, layer_id: int, is_nextn: bool) -> bool:
        return is_nextn or (
            self.config.n_routed_experts is not None
            and layer_id >= self.config.first_k_dense_replace
            and layer_id % self.config.moe_layer_freq == 0
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: Optional[BumpAllocator] = None,
        gemm_output_zero_allocator: BumpAllocator = None,
        prev_topk_indices: Optional[torch.Tensor] = None,
        capture_output=None,
    ):
        hidden_states_orig = residual_access.buffer(hidden_states)

        hidden_states = self.attn_boundary.prepare(
            hidden_states, forward_batch, capture=capture_output
        )

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
            input_on_attn_tp_slices=(self.attn_boundary.input_on_attn_tp_slices),
            prev_topk_indices=prev_topk_indices,
        )
        if isinstance(hidden_states, tuple):
            hidden_states, topk_indices = hidden_states
        else:
            topk_indices = None
        get_attn_tp_context().clear_attn_inputs()

        hidden_states = self.attn_boundary.finish(hidden_states, forward_batch)
        hidden_states = self.ffn_boundary.prepare(hidden_states, forward_batch)

        if isinstance(self.mlp, Glm5NextMLP):
            gemm_output_zero_allocator = None

        if (
            isinstance(self.mlp, Glm5NextMoE)
            and not self.mlp.experts.moe_runner_config.inplace
            and not torch.compiler.is_compiling()
            and hidden_states_orig.shape[0] == hidden_states.shape[0]
        ):
            from sglang.srt.layers.moe.moe_runner.base import moe_output_buffer_ctx

            _mlp_ctx = moe_output_buffer_ctx(hidden_states_orig)
        else:
            _mlp_ctx = nullcontext()

        with _mlp_ctx:
            hidden_states = self.mlp(
                hidden_states,
                forward_batch,
                gemm_output_zero_allocator,
            )
        hidden_states = self.ffn_boundary.finish(hidden_states, forward_batch)

        return (hidden_states, topk_indices)


class Glm5NextModel(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: Glm5NextTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.first_k_dense_replace = config.first_k_dense_replace
        self.pp_group = get_parallel().pp_group

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                **get_embedding_tp_kwargs(),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.alt_stream = (
            torch.cuda.Stream()
            if (
                _is_cuda
                or envs.SGLANG_NPU_USE_MULTI_STREAM.get()
                or envs.SGLANG_ROCM_USE_MULTI_STREAM.get()
            )
            else None
        )

        self.layers, self.start_layer, self.end_layer = make_pp_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Glm5NextDecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=self.alt_stream,
            ),
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        self.gemm_output_zero_allocator_size = 0
        if (
            _use_aiter_gfx95
            and config.n_routed_experts == 256
            and self.embed_tokens.embedding_dim == 7168
        ):
            num_moe_layers = sum(
                [
                    1
                    for i in range(len(self.layers))
                    if isinstance(self.layers[i].mlp, Glm5NextMoE)
                ]
            )

            allocate_size = 0
            for i in range(len(self.layers)):
                if isinstance(self.layers[i].mlp, Glm5NextMoE):
                    a2a_backend = get_moe_a2a_backend()
                    is_a2a_moe = (
                        a2a_backend.is_deepep()
                        or a2a_backend.is_mori()
                        or a2a_backend.is_mooncake()
                    )
                    tp_size = 1 if is_a2a_moe else get_parallel().tp_size
                    intermediate_size = (
                        config.moe_intermediate_size * config.n_shared_experts
                    )
                    share_expert_output_size_per_partition = divide(
                        intermediate_size * 2, tp_size
                    )
                    allocate_size = share_expert_output_size_per_partition
                    break

            self.gemm_output_zero_allocator_size = (
                get_dsv3_gemm_output_zero_allocator_size(
                    config.n_routed_experts,
                    num_moe_layers,
                    allocate_size,
                    self.embed_tokens.embedding_dim,
                )
            )
        self.layers_to_capture = []
        self.dflash_capture = False
        if (
            get_moe_a2a_backend().is_deepep()
            or get_moe_a2a_backend().is_mooncake()
            or get_moe_a2a_backend().is_deepep_v2()
        ):
            self.enable_a2a_moe = True
        else:
            self.enable_a2a_moe = False

    def get_input_embeddings(self) -> torch.Tensor:
        return self.embed_tokens

    def _prepare_aux_hidden_state(self, aux_hidden_state: torch.Tensor) -> torch.Tensor:
        if self.dflash_capture and self.config.mhc:
            aux_hidden_state = hc_contract(aux_hidden_state, self.config.hc_mult)
        return aux_hidden_state

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        total_num_layers = self.end_layer - self.start_layer
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
        device = hidden_states.device
        zero_allocator = BumpAllocator(
            buffer_size=total_num_layers * 2 * (2 if forward_batch.can_run_tbo else 1),
            dtype=torch.float32,
            device=device,
        )

        has_gemm_output_zero_allocator = hasattr(
            self, "gemm_output_zero_allocator_size"
        )

        gemm_output_zero_allocator = (
            BumpAllocator(
                buffer_size=self.gemm_output_zero_allocator_size,
                dtype=torch.float32,
                device=device,
            )
            if has_gemm_output_zero_allocator
            and self.gemm_output_zero_allocator_size > 0
            else None
        )

        normal_start_layer = self.start_layer
        normal_end_layer = self.end_layer
        if forward_batch.can_run_tbo and not self.dflash_capture:
            if (
                self.first_k_dense_replace > normal_start_layer
                and self.first_k_dense_replace < normal_end_layer
            ):
                normal_end_layer = self.first_k_dense_replace
            elif self.first_k_dense_replace < normal_start_layer:
                normal_end_layer = normal_start_layer = 0
        aux_hidden_states = AuxHiddenStateList()
        topk_indices = None
        for i in range(normal_start_layer, normal_end_layer):
            # NOTE: torch dynamo does not support graph break in context manager
            ctx = (
                nullcontext()
                if check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE)
                else get_global_expert_distribution_recorder().with_current_layer(i)
            )
            with ctx:

                def capture_output(aux_hidden_state, *, owned=False):
                    aux_hidden_state = self._prepare_aux_hidden_state(aux_hidden_state)
                    owned = owned or (self.dflash_capture and self.config.mhc)
                    if self.enable_a2a_moe and i > self.first_k_dense_replace:
                        group = get_parallel().attn_tp_group
                        aux_hidden_state = group.all_gather(aux_hidden_state, dim=0)
                        owned = owned or group.world_size > 1
                    aux_hidden_states.capture(aux_hidden_state, owned=owned)

                layer = self.layers[i]
                (hidden_states, topk_indices) = layer(
                    positions,
                    hidden_states,
                    forward_batch,
                    zero_allocator,
                    gemm_output_zero_allocator,
                    prev_topk_indices=topk_indices,
                    capture_output=capture_output
                    if i in self.layers_to_capture
                    else None,
                )

        if normal_end_layer != self.end_layer:
            hidden_states = model_forward_stages(
                layers=self.layers[normal_end_layer : self.end_layer],
                enable_tbo=True,
                positions=positions,
                forward_batch=forward_batch,
                hidden_states=hidden_states,
                zero_allocator=zero_allocator,
            )

        if not self.pp_group.is_last_rank:
            return residual_batch.to_pp(hidden_states, forward_batch)
        else:
            hidden_states = residual_batch.complete_output(hidden_states, forward_batch)
            if not forward_batch.forward_mode.is_idle():
                hidden_states = residual_batch.final_norm(
                    hidden_states, forward_batch, self.norm
                )

        if len(aux_hidden_states) == 0:
            return hidden_states
        return hidden_states, aux_hidden_states


class Glm5NextForConditionalGeneration(nn.Module):
    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_substr={
            "model.visual": "visual",
        },
        orig_to_new_prefix={
            "model.language_model.": "model.",
        },
        orig_to_new_suffix={".attn.qkv": ".attn.qkv_proj"},
    )
    packed_modules_mapping = {
        "fused_qkv_a_proj_with_mqa": ["q_a_proj", "kv_a_proj_with_mqa"],
        **Glm5NextLinearAttention._PACKED_MODULES_MAPPING,
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "qkv_conv1d": ["q_conv1d", "k_conv1d", "v_conv1d"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: Glm5NextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        vision_utils.update_vit_attn_dummy_heads_config(config)
        self.mm_config = config
        text_config = config.text_config
        self.encoder_only = bool(getattr(config, "encoder_only", False))
        self.language_only = bool(getattr(config, "language_only", False))

        self.fuse_qkv_a_proj = (
            not self.encoder_only
            and getattr(text_config, "q_lora_rank", None) is not None
        )

        self.pp_group = get_parallel().pp_group
        self.config = text_config
        self.quant_config = quant_config
        self.use_dsa = is_deepseek_dsa(text_config)
        self.num_fused_shared_experts = 0
        self.model = None
        self.lm_head = None
        self.logits_processor = None
        if not self.encoder_only:
            self.determine_num_fused_shared_experts()
            self.model = Glm5NextModel(
                text_config, quant_config, prefix=add_prefix("model", prefix)
            )
            if self.pp_group.is_last_rank:
                if self.pp_group.world_size == 1 and text_config.tie_word_embeddings:
                    self.lm_head = self.model.embed_tokens
                else:
                    self.lm_head = ParallelLMHead(
                        text_config.vocab_size,
                        text_config.hidden_size,
                        quant_config=quant_config,
                        prefix=add_prefix("lm_head", prefix),
                        use_attn_tp_group=get_parallel().enable_dp_lm_head,
                    )
            else:
                self.lm_head = PPMissingLayer()
            self.logits_processor = LogitsProcessor(text_config)

        self._routed_experts_weights_of_layer = LazyValue(
            lambda: (
                {
                    layer_id: layer.mlp.get_moe_weights()
                    for layer_id, layer in enumerate(self.model.layers)
                    if isinstance(layer.mlp, Glm5NextMoE)
                }
                if self.model is not None
                else {}
            )
        )
        self.capture_aux_hidden_states = False

        if not self.encoder_only:
            get_attn_tp_context().init_context(
                getattr(text_config, "q_lora_rank", None),
                self.use_dsa,
                text_config.mhc,
            )

        self.use_data_parallel = get_mm().mm_enable_dp_encoder
        self.visual = None
        if not self.language_only:
            self.visual = Glm5NextVisionModel(
                config.vision_config,
                quant_config=quant_config,
                prefix=add_prefix("visual", prefix),
                use_data_parallel=self.use_data_parallel,
            )
        self.is_mrope_enabled = not self.encoder_only and "mrope_section" in (
            self.config.rope_scaling or {}
        )

    def get_input_embeddings(self) -> nn.Embedding:
        if self.model is None:
            raise AttributeError(
                "get_input_embeddings() is not available in encoder-only mode"
            )
        return self.model.embed_tokens

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    @classmethod
    def shared_experts_fusion_disable_reason(cls, hf_config, quant_config):
        # Kept in lockstep with the wrapper gate below: a divergence drops the
        # shared-expert weights and runs the fused slot uninitialized.
        text_config = getattr(hf_config, "text_config", hf_config)
        if not getattr(text_config, "n_shared_experts", None):
            return "No shared experts are defined in the config."
        if getattr(text_config, "n_shared_experts", None) != 1:
            return "Shared experts fusion requires exactly one shared expert."
        if quant_config is not None and quant_config.get_name() == "modelopt_fp4":
            first_sparse_layer = getattr(text_config, "first_k_dense_replace", 0)
            for layer_id in range(first_sparse_layer, text_config.num_hidden_layers):
                moe_prefix = f"model.layers.{layer_id}.mlp"
                if quant_config.is_layer_excluded(
                    f"{moe_prefix}.shared_experts"
                ) and not quant_config.is_layer_excluded(f"{moe_prefix}.experts"):
                    return (
                        "ModelOpt FP4 keeps shared experts unquantized while routed "
                        "experts are quantized."
                    )
        if get_parallel().moe_ep_size > 1:
            return (
                "Shared experts fusion is not supported together with expert "
                "parallelism yet."
            )

        if _is_cuda:
            if _device_sm is not None and _device_sm < 80:
                return "Shared experts fusion requires SM80 or newer GPUs."
            if get_moe_a2a_backend().is_deepep():
                return (
                    "Shared experts fusion is not supported when Deepep MoE backend "
                    "is enabled."
                )
        else:
            if not _use_aiter_gfx95:
                return "HIP shared experts fusion requires AITER on a gfx950 device."
            if not get_moe_a2a_backend().is_none():
                return (
                    "HIP shared experts fusion is not supported when an MoE A2A "
                    "backend is enabled."
                )
            parallel = get_parallel()
            if (
                parallel.tp_size not in (4, 8)
                or parallel.moe_tp_size != parallel.tp_size
            ):
                return "HIP shared experts fusion is validated only for TP4 and TP8."
            if (
                getattr(text_config, "model_type", None) != "glm5_next_text"
                or getattr(text_config, "hidden_size", None) != 4096
                or getattr(text_config, "moe_intermediate_size", None) != 2048
                or getattr(text_config, "n_routed_experts", None) != 288
                or getattr(text_config, "num_experts_per_tok", None) != 8
            ):
                return (
                    "HIP shared experts fusion is validated only for the "
                    "GLM-5.3-Flash E=288/topk=8 expert geometry."
                )
            if (
                getattr(text_config, "hidden_act", None) != "silu"
                or getattr(text_config, "swiglu_limit", None) != 10.0
            ):
                return (
                    "HIP shared experts fusion requires the validated "
                    "clamped SiLU G1U1 activation."
                )
            if quant_config is not None and quant_config.get_name() == "quark":
                return cls._quark_mxfp4_shared_fusion_disable_reason(
                    text_config, quant_config
                )
            if (
                quant_config is None
                or quant_config.get_name() != "fp8"
                or not getattr(quant_config, "is_checkpoint_fp8_serialized", False)
                or getattr(quant_config, "weight_block_size", None) != [128, 128]
                or getattr(quant_config, "activation_scheme", None) != "dynamic"
            ):
                return (
                    "HIP shared experts fusion requires serialized dynamic "
                    "128x128 block-FP8 experts."
                )
            ignored_layers = getattr(quant_config, "ignored_layers", [])
            packed_mapping = getattr(quant_config, "packed_modules_mapping", {})
            first_sparse_layer = getattr(text_config, "first_k_dense_replace", 0)
            for layer_id in range(first_sparse_layer, text_config.num_hidden_layers):
                moe_prefix = f"model.layers.{layer_id}.mlp"
                if is_layer_skipped(
                    f"{moe_prefix}.experts", ignored_layers, packed_mapping
                ) or is_layer_skipped(
                    f"{moe_prefix}.shared_experts", ignored_layers, packed_mapping
                ):
                    return (
                        "HIP shared experts fusion requires routed and shared "
                        "experts to use the same block-FP8 layout."
                    )
        return None

    @staticmethod
    def _quark_mxfp4_shared_fusion_disable_reason(text_config, quant_config):
        from sglang.srt.layers.quantization.quark.utils import (
            deep_compare,
            should_ignore_layer,
        )

        reason = (
            "HIP shared experts fusion requires routed and shared experts to "
            "use the same Quark MXFP4 layout."
        )
        if not quant_config.can_fuse_shared_expert():
            return reason
        lookup_stub = torch.nn.Module()
        first_sparse_layer = getattr(text_config, "first_k_dense_replace", 0)
        for layer_id in range(first_sparse_layer, text_config.num_hidden_layers):
            moe_prefix = f"model.layers.{layer_id}.mlp"
            routed_name = f"{moe_prefix}.experts"
            shared_names = [
                f"{moe_prefix}.shared_experts.{proj}"
                for proj in ("gate_up_proj", "down_proj")
            ]
            if any(
                should_ignore_layer(
                    name,
                    ignore=quant_config.exclude_layers,
                    fused_mapping=quant_config.packed_modules_mapping,
                )
                for name in (routed_name, *shared_names)
            ):
                return reason
            try:
                routed = quant_config._find_matched_config(routed_name, lookup_stub)
                shared = [
                    quant_config._find_matched_config(name, lookup_stub)
                    for name in shared_names
                ]
            except ValueError:
                return reason
            if not quant_config._is_mx_fp4(
                routed.get("weight"), routed.get("input_tensors")
            ) or not all(deep_compare(routed, cfg) for cfg in shared):
                return reason
        return None

    def determine_num_fused_shared_experts(self):
        self.num_fused_shared_experts = (
            0 if is_shared_experts_fusion_disabled() else self.config.n_shared_experts
        )
        if self.num_fused_shared_experts == 0:
            return
        assert self.num_fused_shared_experts == 1, (
            f"Only 1 fused shared expert is supported for {type(self).__name__}"
        )
        log_info_on_rank0(logger, "Shared experts fusion optimization enabled.")

    def wants_prefill_autotune(self) -> bool:
        backend = get_moe_runner_backend()
        return (
            (backend.is_flashinfer_trtllm() or backend.is_flashinfer_trtllm_routed())
            and get_moe_a2a_backend().is_none()
            and not is_dp_attention_enabled()
        )

    def autotune_prefill_kernels(self, num_tokens: int, *, dtype: torch.dtype) -> int:
        seen = set()
        for module in self.model.modules():
            if not isinstance(module, Glm5NextMoE):
                continue
            experts = module.experts
            key = (experts.w13_weight.shape, experts.w2_weight.shape)
            if key in seen:
                continue
            seen.add(key)
            hidden_states = torch.randn(
                (num_tokens, module.gate.weight.shape[1]),
                dtype=dtype,
                device=experts.w13_weight.device,
            )
            router_logits = module.gate(hidden_states)
            experts(hidden_states, module.topk(hidden_states, router_logits))
            del hidden_states, router_logits
        if envs.SGLANG_FLASHINFER_AUTOTUNE_EXTEND.get():
            return 0
        return len(seen)

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        if layer_ids is None:
            self.capture_aux_hidden_states = True
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]
        else:
            self.capture_aux_hidden_states = True
            self.model.layers_to_capture = [val + 1 for val in layer_ids]

    def set_dflash_layers_to_capture(self, layer_ids: List[int]):
        if not self.pp_group.is_last_rank:
            return

        if layer_ids is None:
            raise ValueError(
                "DFLASH requires explicit layer_ids for aux hidden capture."
            )

        self.capture_aux_hidden_states = True
        self.model.dflash_capture = True
        # Capturing before layer k + 1 gives the completed output of layer k.
        self.model.layers_to_capture = [val + 1 for val in layer_ids]

    def pad_input_ids(self, input_ids: array, mm_inputs: MultimodalInputs) -> array:
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()

        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(
                self.visual, pixel_values, image_grid_thw.tolist(), rope_type="rope_3d"
            )
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        return image_embeds

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        video_grid_thw = torch.concat([item.video_grid_thw for item in items], dim=0)

        temp_frames_hw = []
        for t, h, w in video_grid_thw:
            repeated_row = (
                torch.tensor([1, h.item(), w.item()]).unsqueeze(0).repeat(t, 1)
            )
            temp_frames_hw.append(repeated_row)
        flattened_video_grid_thw = torch.cat(temp_frames_hw, dim=0)

        assert pixel_values.dim() == 2, pixel_values.dim()
        assert video_grid_thw.dim() == 2, video_grid_thw.dim()
        if items and getattr(items[0], "dp_decode_sharded", False):
            if len(items) != 1:
                raise ValueError("DP-sharded video decode requires one video item")
            dp_meta = items[0].dp_meta
            height = int(video_grid_thw[0][1])
            width = int(video_grid_thw[0][2])
            global_grid = [[1, height, width]] * int(dp_meta["n_units"])
            return run_dp_presharded_mrope_vision_model(
                self.visual,
                pixel_values,
                flattened_video_grid_thw.tolist(),
                global_grid,
                dp_meta["gpu_sample_counts"],
            )

        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(
                self.visual,
                pixel_values,
                flattened_video_grid_thw.tolist(),
                rope_type="rope_3d",
            )
        video_embeds = self.visual(pixel_values, grid_thw=flattened_video_grid_thw)
        return video_embeds

    def prepare_cp_inputs(self, forward_batch, **kwargs):
        """Finish multimodal embedding injection before the runner shards tokens."""
        input_embeds = kwargs.pop("input_embeds", None)
        positions = (
            forward_batch.mrope_positions
            if self.is_mrope_enabled
            else forward_batch.positions
        )
        if input_embeds is None or forward_batch.contains_mm_inputs():
            input_embeds = prepare_mm_inputs(
                forward_batch.input_ids,
                forward_batch,
                self.model,
                kwargs,
                multimodal_model=self,
            )
        return input_embeds, positions, kwargs

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            if input_embeds is not None and not forward_batch.contains_mm_inputs():
                # Preserve input_embeds / replace_embeds supplied by the runner.
                # The MM routine otherwise performs a fresh token lookup.
                hidden_states = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    forward_batch=forward_batch,
                    input_embeds=input_embeds,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
            else:
                hidden_states = general_mm_embed_routine(
                    input_ids=input_ids,
                    forward_batch=forward_batch,
                    language_model=self.model,
                    multimodal_model=self,
                    positions=positions,
                    pp_proxy_tensors=pp_proxy_tensors,
                )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
            )
        else:
            return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        if is_nextn:
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                assert num_nextn_layers == 1, "Only 1 nextn layer is supported"
                nextn_layer_id = (
                    0
                    if self.config.num_hidden_layers == 1
                    else self.config.num_hidden_layers
                )
            else:
                raise ValueError("num_nextn_predict_layers is not in the config")

        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            (".fused_qkvbfg_a_proj", ".q_proj", 0),
            (".fused_qkvbfg_a_proj", ".k_proj", 1),
            (".fused_qkvbfg_a_proj", ".v_proj", 2),
            (".fused_qkvbfg_a_proj", ".b_proj", 3),
            (".fused_qkvbfg_a_proj", ".f_a_proj", 4),
            (".fused_qkvbfg_a_proj", ".g_a_proj", 5),
            (".fused_fg_b_proj", ".f_b_proj", 0),
            (".fused_fg_b_proj", ".g_b_proj", 1),
            (".fused_bfg_a_proj", ".b_proj", 0),
            (".fused_bfg_a_proj", ".f_a_proj", 1),
            (".fused_bfg_a_proj", ".g_a_proj", 2),
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".qkv_conv1d", ".q_conv1d", 0),
            (".qkv_conv1d", ".k_conv1d", 1),
            (".qkv_conv1d", ".v_conv1d", 2),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts + self.num_fused_shared_experts,
        )

        if is_nextn:
            nextn_layer_prefix = f"model.layers.{nextn_layer_id}"
            nextn_spec_weight_names = [
                "shared_head.norm",
                "eh_proj",
                "enorm",
                "hnorm",
            ]

        fuse_qkv_a_proj = getattr(self, "fuse_qkv_a_proj", False)
        cached_a_proj: dict[str, torch.Tensor] = {} if fuse_qkv_a_proj else None
        qc = self.quant_config
        if qc is not None and qc.get_name() in {"awq", "awq_marlin", "moe_wna16"}:
            fused_cat_dim = 1
        else:
            fused_cat_dim = 0

        params_dict = dict(self.named_parameters())

        def maybe_map_fp8_block_scale_name(name: str) -> str:
            # Quark stores dequantization scales without native block-FP8's
            # "_inv" suffix, including fused w13/w2 expert parameters.
            if name not in params_dict and name.endswith("weight_scale"):
                candidate = name + "_inv"
                if candidate in params_dict:
                    return candidate
            return name

        weight_names = []
        for name, loaded_weight in weights:
            is_visual_weight = "visual" in name
            if getattr(self, "encoder_only", False) and not is_visual_weight:
                continue
            if getattr(self, "language_only", False) and is_visual_weight:
                continue

            if "language_model." in name:
                name = name.replace("language_model.", "")
            if "model.visual." in name:
                name = name.replace("model.visual.", "visual.")

            if "visual" in name:
                name = name.replace("attn.qkv.", "attn.qkv_proj.")
                loaded_weight = vision_utils.pad_vit_attn_dummy_heads(
                    self.mm_config, name, loaded_weight
                )

            weight_names.append(name)

            if self.num_fused_shared_experts > 0 and "mlp.shared_experts" in name:
                name = name.replace(
                    "mlp.shared_experts",
                    f"mlp.experts.{self.config.n_routed_experts}",
                )

            if not is_nextn:
                if hasattr(self.config, "num_nextn_predict_layers"):
                    num_nextn_layers = self.config.num_nextn_predict_layers
                    if num_nextn_layers > 0 and name.startswith("model.layers"):
                        name_list = name.split(".")
                        if (
                            len(name_list) >= 3
                            and int(name_list[2]) >= self.config.num_hidden_layers
                        ):
                            continue
            else:
                if not name.startswith(nextn_layer_prefix):
                    continue

                if "shared_head.head" in name or "embed_tokens" in name:
                    continue

                is_decoder = True
                for weight_name in nextn_spec_weight_names:
                    if weight_name in name:
                        name = name.replace(nextn_layer_prefix, "model")
                        is_decoder = False
                        break
                if is_decoder:
                    name = name.replace(nextn_layer_prefix, "model.decoder")

            if "rotary_emb.inv_freq" in name:
                continue
            if "hc_head" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                candidate = name.replace(weight_name, param_name)
                candidate = maybe_map_fp8_block_scale_name(candidate)
                if (
                    param_name
                    in {
                        ".fused_qkvbfg_a_proj",
                        ".fused_bfg_a_proj",
                        ".fused_fg_b_proj",
                        ".qkv_proj",
                        ".qkv_conv1d",
                    }
                    and candidate not in params_dict
                ):
                    continue
                name = candidate
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name = name.replace(weight_name, param_name)
                    name = maybe_map_fp8_block_scale_name(name)
                    if name not in params_dict:
                        continue
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
                    if is_expert_weight:
                        continue
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    if fuse_qkv_a_proj and (
                        "q_a_proj" in name or "kv_a_proj_with_mqa" in name
                    ):
                        cached_a_proj[name] = loaded_weight
                        q_a_proj_name = (
                            name
                            if "q_a_proj" in name
                            else name.replace("kv_a_proj_with_mqa", "q_a_proj")
                        )
                        kv_a_proj_name = (
                            name
                            if "kv_a_proj_with_mqa" in name
                            else name.replace("q_a_proj", "kv_a_proj_with_mqa")
                        )
                        if (
                            q_a_proj_name in cached_a_proj
                            and kv_a_proj_name in cached_a_proj
                        ):
                            fused_weight = torch.cat(
                                [
                                    cached_a_proj[q_a_proj_name],
                                    cached_a_proj[kv_a_proj_name],
                                ],
                                dim=fused_cat_dim,
                            )
                            target = (
                                name.replace("q_a_proj", "fused_qkv_a_proj_with_mqa")
                                if "q_a_proj" in name
                                else name.replace(
                                    "kv_a_proj_with_mqa",
                                    "fused_qkv_a_proj_with_mqa",
                                )
                            )
                            target = maybe_map_fp8_block_scale_name(target)
                            if target in params_dict:
                                param = params_dict[target]
                                weight_loader = getattr(
                                    param, "weight_loader", default_weight_loader
                                )
                                weight_loader(param, fused_weight)
                            cached_a_proj.pop(q_a_proj_name, None)
                            cached_a_proj.pop(kv_a_proj_name, None)
                        continue

                    name = maybe_map_fp8_block_scale_name(name)
                    if name not in params_dict:
                        continue

                    if name.endswith(".A_log") and loaded_weight.dim() == 1:
                        loaded_weight = loaded_weight.view(1, 1, -1, 1)

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

        if getattr(self, "encoder_only", False):
            run_post = False
        elif is_nextn:
            decoder_attn = getattr(self.model.decoder, "self_attn", None)
            run_post = decoder_attn is not None and hasattr(decoder_attn, "kv_b_proj")
        else:
            run_post = True
        if run_post:
            DeepseekV2WeightLoaderMixin.post_load_weights(
                self, is_nextn=is_nextn, weight_names=weight_names
            )

    def post_load_weights(self, is_nextn: bool = False, weight_names=None):
        if self.encoder_only:
            return
        DeepseekV2WeightLoaderMixin.post_load_weights(
            self, is_nextn=is_nextn, weight_names=weight_names
        )

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        if self.model is None:
            raise AttributeError(
                "load_kv_cache_scales() is not available in encoder-only mode"
            )
        if callable(getattr(self.model, "load_kv_cache_scales", None)):
            self.model.load_kv_cache_scales(quantization_param_path)
        else:
            logger.warning(
                f"{self.model.__class__} does not support loading scaling factors."
            )

    def get_embed_and_head(self):
        if self.model is None or self.lm_head is None:
            raise AttributeError(
                "get_embed_and_head() is not available in encoder-only mode"
            )
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        if self.model is None or self.lm_head is None:
            raise AttributeError(
                "set_embed_and_head() is not available in encoder-only mode"
            )
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        config = getattr(config, "text_config", config)
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=None,
        )


EntryClass = [Glm5NextForConditionalGeneration]
