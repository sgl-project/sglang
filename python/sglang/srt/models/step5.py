# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Step5 target model, shared decoder components and native weight loading."""

from __future__ import annotations

import json
import logging
import re
import weakref
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import torch
from torch import nn

from sglang.kernels.ops.activation import SiluClampAndMulOp
from sglang.kernels.ops.attention.step5 import options as step5_options
from sglang.kernels.ops.attention.step5.qknorm_rope import (
    step5_qknorm_rope,
    step5_qknorm_rope_gated,
)
from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul
from sglang.kernels.ops.moe.step5_router_topk import Step5RouterTopKOp
from sglang.kernels.spec import KernelBackend
from sglang.srt.configs.step5 import (
    Step5Config,
    get_step5_position_padding,
    parse_step5_attention_profile,
    resolve_step5_layer_selection,
)
from sglang.srt.distributed import (
    attention_tensor_model_parallel_all_reduce,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.attention.step5.indexer import Step5Indexer
from sglang.srt.layers.dp_attention import (
    dp_gather_replicate,
    dp_scatter,
    get_global_dp_buffer,
    world_dp_gather_enabled,
)
from sglang.srt.layers.layer_boundary import (
    append_stages,
    declare_attn,
    declare_ffn,
    layer_stack,
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
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.mega_moe import forward_mega_moe_local_experts
from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
    FlashInferTrtllmDeferredFinalizeOutput,
    finalize_flashinfer_trtllm_deferred_output,
)
from sglang.srt.layers.moe.topk import TopK, TopKOutput, TopKOutputFormat
from sglang.srt.layers.moe.utils import (
    RoutingMethodType,
    get_moe_a2a_backend,
    get_moe_runner_backend,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptNvFp4FusedMoEMethod,
)
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.step5_allreduce import (
    Step5BoundaryFusions,
    Step5DeferredMoE,
)
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.lora.utils import RoutedExpertLoRATargets
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    get_tc_piecewise_forward_context,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.runtime_context import (
    get_exec,
    get_forward,
    get_lora,
    get_model,
    get_parallel,
    get_platform,
    get_stream,
)
from sglang.srt.utils import add_prefix, set_weight_attrs


class Step5MLP(nn.Module):
    """BF16 SwiGLU with post-SiLU clamp and boundary-owned TP reduction."""

    def __init__(
        self,
        *,
        role: Literal["dense", "shared"],
        hidden_size: int,
        intermediate_size: int,
        swiglu_limit: float,
        tp_rank: int,
        tp_size: int,
        prefix: str = "",
        use_dp_attention_reduce: bool = False,
    ) -> None:
        super().__init__()
        self.role = role
        self.hidden_size = hidden_size
        self.tp_size = tp_size
        self.act_fn = SiluClampAndMulOp(limit=swiglu_limit)
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size, intermediate_size],
            bias=False,
            gather_output=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=add_prefix("gate_up_proj", prefix),
            parallel_group="attn_tp",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            input_is_parallel=True,
            params_dtype=torch.bfloat16,
            quant_config=None,
            # Stage boundaries sum dense and shared-expert TP partials.
            # The attention-DP path has no stage boundaries and sums here.
            reduce_results=use_dp_attention_reduce,
            use_dp_attention_reduce=use_dp_attention_reduce,
            prefix=add_prefix("down_proj", prefix),
            parallel_group="attn_tp",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return BF16 TP partials, except on the boundary-free attention-DP path."""
        gate_up, _ = self.gate_up_proj(hidden_states)
        activation = self.act_fn(gate_up)
        output, _ = self.down_proj(activation)
        return output


class Step5Router(nn.Module):
    """BF16 gate with FP32 logits, bias and routing weights."""

    def __init__(
        self,
        *,
        layer_id: int,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        routed_scaling_factor: float,
        prefix: str = "",
        allow_routed_experts_capture: bool = True,
        kernel_routing: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_routing = kernel_routing
        self.gate = ReplicatedLinear(
            input_size=hidden_size,
            output_size=num_experts,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )
        self.router_bias = nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32), requires_grad=False
        )
        set_weight_attrs(self.router_bias, {"weight_loader": default_weight_loader})
        self.routing_op = Step5RouterTopKOp(
            num_experts=num_experts,
            top_k=top_k,
            routed_scaling_factor=routed_scaling_factor,
        )
        self.topk = TopK(
            top_k=top_k,
            layer_id=layer_id,
            use_grouped_topk=False,
            renormalize=True,
            num_fused_shared_experts=0,
            custom_routing_function=None if kernel_routing else self._route,
            custom_routing_handles_padding=(
                # TODO(yhyang201): frozen SGLANG_OPT_STEP5_ROUTER_PADDING; prune dead paths.
                not kernel_routing and step5_options.ROUTER_PADDING
            ),
            scoring_func="sigmoid",
            correction_bias=self.router_bias if kernel_routing else None,
            quant_config=None,
            routed_scaling_factor=None,
            apply_routed_scaling_factor_on_output=False,
            output_format=TopKOutputFormat.BYPASSED
            if kernel_routing
            else TopKOutputFormat.STANDARD,
            allow_routed_experts_capture=allow_routed_experts_capture,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """BF16 operands with FP32 output; CPU promotion is diagnostic only."""
        if hidden_states.device.type == "cpu":
            return torch.nn.functional.linear(
                hidden_states.float(), self.gate.weight.float()
            )
        return torch.mm(hidden_states, self.gate.weight.t(), out_dtype=torch.float32)

    def _route(
        self,
        *,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        num_token_non_padded: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.routing_op(
            gating_output, self.router_bias, num_token_non_padded=num_token_non_padded
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        num_token_non_padded: torch.Tensor | None = None,
        expert_location_dispatch_info: ExpertLocationDispatchInfo | None = None,
    ) -> TopKOutput:
        logits = self.compute_logits(hidden_states)
        return self.topk(
            hidden_states,
            logits,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )


def _step5_keep_bf16_experts(quant_config, prefix: str) -> bool:
    if quant_config is None:
        return True
    ignored = quant_config.ignored_layers
    w13_skipped = is_layer_skipped(f"{prefix}.w13_weight", ignored)
    w2_skipped = is_layer_skipped(f"{prefix}.w2_weight", ignored)
    if w13_skipped != w2_skipped:
        raise ValueError(
            f"Step5 expert matrices must have matching precision: {prefix}"
        )
    return w13_skipped and not envs.SGLANG_STEP5_QUANTIZE_BF16_MOE.get()


def _step5_bf16_trtllm_supported() -> bool:
    if not get_platform().is_sm100:
        return False
    from flashinfer.fused_moe import core as fi_moe

    # Only FlashInfer builds with the SwigluStep patch apply the post-SiLU clamp.
    return getattr(fi_moe, "TRTLLM_BF16_SWIGLU_STEP_SUPPORTED", False)


_STEP5_EXPERT_LAYER = re.compile(r"layers\.(\d+)\.moe\.experts")
_STEP5_EXPERT_TENSOR = re.compile(r"(?:^|\.)layers\.(\d+)\.moe\.experts\.\d+\.")
_STEP5_PACKED_NVFP4_TENSOR = re.compile(
    r"(?:^|\.)layers\.(\d+)\.moe\.(?:gate|up|down)_proj\.weight_scale_2$"
)
_STEP5_NVFP4_EXPERT_SOURCES = {
    f"{projection}_proj.{suffix}": (f"{target}{param_suffix}", shard)
    for projection, target, shard in (
        ("gate", "w13", "w1"),
        ("up", "w13", "w3"),
        ("down", "w2", "w2"),
    )
    for suffix, param_suffix in (
        ("weight", "_weight"),
        ("weight_scale", "_weight_scale"),
        ("weight_scale_2", "_weight_scale_2"),
        ("input_scale", "_input_scale"),
    )
}
_STEP5_NVFP4_PACKED_SOURCES = {
    f"{projection}_proj.{suffix}": (f"{target}{param_suffix}", shard)
    for projection, target, shard in (
        ("gate", "w13", "w1"),
        ("up", "w13", "w3"),
        ("down", "w2", "w2"),
    )
    for suffix, param_suffix in (
        ("weight", "_weight"),
        ("weight_scale", "_weight_scale"),
        ("weight_scale_2", "_weight_scale_2"),
        ("input_scale", "_input_scale"),
    )
}
_STEP5_NVFP4_PARAM_SOURCES = {
    target: tuple(
        source
        for source, (candidate, _) in _STEP5_NVFP4_EXPERT_SOURCES.items()
        if candidate == target
    )
    for target in dict.fromkeys(
        target for target, _ in _STEP5_NVFP4_EXPERT_SOURCES.values()
    )
}


def _step5_routed_expert_row(name: str, num_experts: int) -> tuple[str, int] | None:
    # Trainers send `model.layers.L.moe.experts.E.<proj>`, one row of the packed `moe.<proj>`.
    head, marker, tail = name.rpartition(".moe.experts.")
    expert, _, projection = tail.partition(".")
    if (
        not marker
        or not projection
        or not expert.isdecimal()
        or str(int(expert)) != expert
    ):
        return None
    if not 0 <= int(expert) < num_experts:
        return None
    return f"{head}.moe.{projection}", int(expert)


def _step5_indexed_nvfp4_layers(checkpoint_path) -> dict[int, str] | None:
    if not checkpoint_path:
        return None
    index_path = Path(checkpoint_path) / "model.safetensors.index.json"
    if not index_path.is_file():
        return None
    layers: dict[int, str] = {}
    for name in json.loads(index_path.read_text())["weight_map"]:
        match = _STEP5_EXPERT_TENSOR.search(name)
        if match is not None:
            layers[int(match.group(1))] = "per_expert"
            continue
        match = _STEP5_PACKED_NVFP4_TENSOR.search(name)
        if match is not None:
            layers[int(match.group(1))] = "packed"
    return layers


def step5_nvfp4_expert_layouts(
    config: Step5Config,
    quant_config: QuantizationConfig | None,
    checkpoint_path: str | Path | None = None,
) -> dict[int, str]:
    """MoE layers whose routed experts are ModelOpt NVFP4, with their checkpoint layout.

    ModelOpt exports the experts either per expert (``experts.E.gate_proj.weight``) or
    packed per projection (``moe.gate_proj.weight`` with a leading expert axis); the
    checkpoint index decides when it exists, otherwise the ModelOpt ignore list names the
    layers whose experts stayed BF16/FP8 and the per-expert layout is assumed.
    """
    if not isinstance(quant_config, ModelOptFp4Config):
        return {}
    indexed = _step5_indexed_nvfp4_layers(checkpoint_path)
    if indexed is not None:
        return {
            layer: indexed[layer] for layer in config.moe_layer_list if layer in indexed
        }
    excluded = set()
    for pattern in quant_config.exclude_modules:
        match = _STEP5_EXPERT_LAYER.search(pattern)
        if match is not None:
            excluded.add(int(match.group(1)))
    return {
        layer: "per_expert" for layer in config.moe_layer_list if layer not in excluded
    }


def step5_nvfp4_expert_layers(
    config: Step5Config,
    quant_config: QuantizationConfig | None,
    checkpoint_path: str | Path | None = None,
) -> frozenset[int]:
    return frozenset(step5_nvfp4_expert_layouts(config, quant_config, checkpoint_path))


def step5_nvfp4_packed_source(
    name: str, layouts: Mapping[int, str]
) -> tuple[str, str] | None:
    """Fused parameter and shard id for a packed per-projection NVFP4 expert tensor."""
    head, marker, suffix = name.rpartition(".moe.")
    source = _STEP5_NVFP4_PACKED_SOURCES.get(suffix) if marker else None
    if source is None or layouts.get(_step5_layer_index(head)) != "packed":
        return None
    param_name, shard_id = source
    return f"{head}.moe.experts.{param_name}", shard_id


def step5_check_nvfp4_expert_scales(experts: FusedMoE, prefix: str) -> None:
    """The fused kernels dequantise gate and up with one global scale per expert."""
    scale = experts.w13_weight_scale_2
    if (
        scale.dim() == 2
        and scale.shape[1] == 2
        and not torch.equal(scale[:, 0], scale[:, 1])
    ):
        raise ValueError(
            f"{prefix}: gate and up NVFP4 weight_scale_2 differ for "
            f"{int((scale[:, 0] != scale[:, 1]).sum())} experts"
        )


def step5_expert_quant_config(
    quant_config: QuantizationConfig | None,
    layer_id: int,
    nvfp4_expert_layers: frozenset[int],
) -> QuantizationConfig | None:
    if (
        not isinstance(quant_config, ModelOptFp4Config)
        or layer_id in nvfp4_expert_layers
    ):
        return quant_config
    # Experts the NVFP4 export left in BF16 follow the FP8 checkpoint's BF16 policy;
    # SGLANG_STEP5_QUANTIZE_BF16_MOE=1 restores load-time FP8 quantization.
    if not envs.SGLANG_STEP5_QUANTIZE_BF16_MOE.get():
        return None
    return Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
    )


def step5_nvfp4_expert_source(name: str) -> tuple[str, str, int] | None:
    """Map a per-expert ModelOpt tensor name to (fused param name, shard id, expert id)."""
    head, marker, tail = name.rpartition(".moe.experts.")
    if not marker:
        return None
    expert_text, _, source = tail.partition(".")
    mapped = _STEP5_NVFP4_EXPERT_SOURCES.get(source)
    if mapped is None or not expert_text.isdecimal():
        return None
    param_name, shard_id = mapped
    return f"{head}{marker}{param_name}", shard_id, int(expert_text)


def _step5_layer_index(prefix: str) -> int | None:
    _, _, index = prefix.rpartition(".")
    return int(index) if index.isdecimal() else None


class Step5MoE(nn.Module):
    """Native routed experts plus a BF16 shared MLP with public TP/DP reduction."""

    def __init__(
        self,
        *,
        layer_id: int,
        hidden_size: int,
        num_experts: int,
        intermediate_size: int,
        shared_intermediate_size: int,
        top_k: int,
        routed_scaling_factor: float,
        swiglu_limit: float,
        shared_swiglu_limit: float,
        quant_config: QuantizationConfig | None,
    ):
        super().__init__()
        parallel = get_parallel()
        self.tp_rank = parallel.tp_rank
        self.num_experts = num_experts
        self.attn_dp_size = parallel.attn_dp_size
        self.attn_tp_size = parallel.attn_tp_size
        self.attn_tp_rank = parallel.attn_tp_rank
        self.layer_id = layer_id
        self.prefix = f"model.layers.{layer_id}.moe"
        self.use_nvfp4_experts = isinstance(quant_config, ModelOptFp4Config)
        self.keep_bf16_experts = (
            not self.use_nvfp4_experts
            and _step5_keep_bf16_experts(quant_config, f"{self.prefix}.experts")
        )
        self.bf16_trtllm = self.keep_bf16_experts and _step5_bf16_trtllm_supported()
        self.use_mega_moe = (
            get_moe_a2a_backend().is_megamoe() and not self.keep_bf16_experts
        )
        if self.use_nvfp4_experts and self.use_mega_moe:
            raise ValueError("Step5 Mega MoE requires FP8 routed experts")
        self.kernel_routing = (
            get_moe_runner_backend().value == "flashinfer_trtllm"
            and not self.use_mega_moe
            and not self.keep_bf16_experts
            and not self.use_nvfp4_experts
        )
        self.swiglu_limit = swiglu_limit
        if self.kernel_routing and get_exec().features.enable_return_routed_experts:
            raise ValueError("Step5 kernel routing cannot return routed experts")
        self.router = Step5Router(
            layer_id=layer_id,
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            routed_scaling_factor=routed_scaling_factor,
            prefix=self.prefix,
            kernel_routing=self.kernel_routing,
        )
        self.share_expert = Step5MLP(
            role="shared",
            hidden_size=hidden_size,
            intermediate_size=shared_intermediate_size,
            swiglu_limit=shared_swiglu_limit,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            use_dp_attention_reduce=self.attn_dp_size > 1,
            prefix=f"{self.prefix}.share_expert",
        )
        if self.use_nvfp4_experts:
            quant_method = ModelOptNvFp4FusedMoEMethod(quant_config)
        elif self.keep_bf16_experts:
            # Checkpoint BF16 experts apply the post-SiLU clamp in TRTLLM when FlashInfer
            # has SwigluStep, else in the CUDA Triton runner; never in CUTLASS.
            quant_method = UnquantizedFusedMoEMethod(
                use_flashinfer_trtllm_moe=self.bf16_trtllm,
                use_flashinfer_cutlass=False,
            )
        else:
            quant_method = Fp8MoEMethod(quant_config)
        self.experts = FusedMoE(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            top_k=top_k,
            num_fused_shared_experts=0,
            params_dtype=torch.bfloat16,
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{self.prefix}.experts",
            activation="silu",
            apply_router_weight_on_input=False,
            use_presharded_weights=False,
            inplace=False,
            no_combine=False,
            routed_scaling_factor=routed_scaling_factor
            if self.kernel_routing
            else None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=swiglu_limit,
            swiglu_limit=None,
            use_weight_loader_fused=False,
            with_bias=False,
            is_gated=True,
            gate_up_interleaved=False,
            fp8_use_rn_division=(
                not self.keep_bf16_experts and get_moe_runner_backend().is_triton()
            ),
            quant_method=quant_method,
            routing_method_type=RoutingMethodType.SigmoidBiasOrderedRenorm
            if self.kernel_routing
            else None,
            silu_clamp_mode="post",
            fc1_output_dtype="bf16",
        )
        weight = self.router.gate.weight
        self.use_allreduce_dp_gather = (
            weight.is_cuda
            and torch.version.hip is None
            and torch.cuda.get_device_capability(weight.device) == (10, 3)
            and hidden_size == 4096
            and parallel.nnodes == 1
            and parallel.pp_size == 1
            and parallel.attn_tp_size == 4
            and (parallel.tp_size, parallel.attn_dp_size, parallel.moe_ep_size)
            == (8, 2, 8)
            and get_exec().comm.flashinfer_allreduce_fusion_backend == "trtllm"
            and not get_exec().comm.enable_symm_mem
            and not get_exec().comm.enable_quant_communications
            and not get_exec().deterministic.enable_deterministic_inference
        )
        self.fuse_shared_finalize = (
            # TODO(yhyang201): frozen SGLANG_OPT_STEP5_MOE_FINALIZE_SHARED; prune dead paths.
            step5_options.MOE_FINALIZE_SHARED
            if step5_options.MOE_FINALIZE_SHARED is not None
            else (
                weight.is_cuda
                and torch.version.hip is None
                and torch.cuda.get_device_capability(weight.device) == (10, 3)
                and hidden_size == 4096
                and intermediate_size == 1536
                and top_k == 8
                and (parallel.tp_size, parallel.moe_ep_size) in ((8, 8), (4, 1))
                and self.attn_dp_size == 1
                and isinstance(quant_config, Fp8Config)
                and quant_config.is_checkpoint_fp8_serialized
                and quant_config.weight_block_size == [128, 128]
            )
        )
        if self.keep_bf16_experts:
            self.fuse_shared_finalize = False
            logging.getLogger(__name__).info(
                "Step5 layer %d: preserving checkpoint BF16 MoE experts (%s)",
                layer_id,
                "flashinfer_trtllm"
                if self.bf16_trtllm
                else get_moe_runner_backend().value,
            )
        # DP shared reduction must finish before the in-place gather.
        # TODO(yhyang201): frozen SGLANG_STEP5_SHARED_OVERLAP_MAX_ROWS; prune dead paths.
        self.overlap_max_rows = step5_options.SHARED_OVERLAP_MAX_ROWS
        self.alt_stream = (
            get_stream("alt")
            if self.overlap_max_rows > 0 and self.attn_dp_size == 1
            else None
        )
        self.dp_alt_stream = get_stream("alt") if self.use_allreduce_dp_gather else None
        self.dp_overlap_max_rows = 128

    def check_routed_expert_lora_support(self) -> None:
        """Routed-expert LoRA runs in FusedMoEWithLoRA's Triton hooks; MegaMoE and the
        FlashInfer runners (including the deferred-finalize path) never call them."""
        backend = get_moe_runner_backend()
        unsupported = [
            name
            for name, used in (
                ("MegaMoE", self.use_mega_moe),
                ("NVFP4 experts", self.use_nvfp4_experts),
                ("FlashInfer TRT-LLM BF16 experts", self.bf16_trtllm),
                (f"the {backend.value} MoE runner", not backend.is_triton()),
            )
            if used
        ]
        if unsupported:
            raise ValueError(
                f"Step5 routed-expert LoRA needs --moe-runner-backend triton; layer "
                f"{self.layer_id} uses {', '.join(unsupported)}"
            )

    def _can_use_allreduce_dp_gather(self, gathered, forward_batch):
        comm = get_parallel().tp_group.ca_comm
        return (
            self.use_allreduce_dp_gather
            and not world_dp_gather_enabled()
            and forward_batch.dp_padding_mode is not None
            and forward_batch.dp_padding_mode.is_max_len()
            and gathered.dtype == torch.bfloat16
            and 0 < gathered.shape[0] <= 512
            and gathered.shape[0] % 8 == 0
            and isinstance(comm, CustomAllReduceV2)
            and not comm.disabled
            and comm.should_custom_ar(gathered)
        )

    def _forward_dp(self, hidden_states, forward_batch):
        gathered = get_global_dp_buffer(get_parallel().tp_group)
        overlap = (
            self.dp_alt_stream is not None
            and self._can_use_allreduce_dp_gather(gathered, forward_batch)
            and 0 < hidden_states.shape[0] <= self.dp_overlap_max_rows
            and get_parallel().attn_tp_group._can_use_flashinfer_allreduce(
                hidden_states
            )
            and get_parallel().tp_group._can_use_flashinfer_allreduce(gathered)
            and (
                forward_batch.forward_mode.is_decode()
                or forward_batch.forward_mode.is_target_verify()
            )
        )
        if not overlap:
            shared = self.share_expert(hidden_states)
            return shared, self._forward_dp_routed(
                hidden_states, forward_batch=forward_batch
            )

        current = torch.cuda.current_stream()
        self.dp_alt_stream.wait_stream(current)
        hidden_states.record_stream(self.dp_alt_stream)
        # The leader-preserving gather leaves hidden_states untouched. Shared
        # experts and their TP4 reduction can overlap the routed EP8 branch;
        # the two reductions own distinct FlashInfer group workspaces.
        with torch.cuda.stream(self.dp_alt_stream):
            shared = self.share_expert(hidden_states)
        routed = self._forward_dp_routed(hidden_states, forward_batch=forward_batch)
        current.wait_stream(self.dp_alt_stream)
        shared.record_stream(current)
        return shared, routed

    def _forward_dp_routed(
        self, hidden_states: torch.Tensor, *, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        # DP gather may overwrite nonleader inputs; compute shared output first.
        gathered = get_global_dp_buffer(get_parallel().tp_group)
        # Small B300 DP2 gathers use custom allreduce v2 instead of an
        # attention-TP reduce-scatter followed by a global allgather. Retain
        # each DP leader as the source, even if other attention ranks differ.
        # Only globally shared geometry may select the collective sequence.
        use_all_reduce = self._can_use_allreduce_dp_gather(gathered, forward_batch)
        dp_gather_replicate(
            gathered, hidden_states, forward_batch, use_all_reduce=use_all_reduce
        )
        # Every EP rank sees the same gathered rows. MegaMoE consumes global
        # expert IDs and filters to this rank's experts; ordinary runners use
        # StandardDispatcher. Both return partials for the single reduction below.
        topk = self.router(gathered)
        if self.use_mega_moe:
            routed = forward_mega_moe_local_experts(
                self.experts,
                gathered,
                topk.topk_ids,
                topk.topk_weights,
                group=get_parallel().moe_tp_group.device_group,
                activation_clamp=self.swiglu_limit,
            )
        else:
            routed = self.experts(gathered, topk)
        routed = tensor_model_parallel_all_reduce(routed)
        local = torch.empty_like(hidden_states)
        dp_scatter(local, routed, forward_batch, fuse_zero_copy=use_all_reduce)
        return local

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        num_token_non_padded: torch.Tensor | None = None,
        forward_batch: ForwardBatch | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | FlashInferTrtllmDeferredFinalizeOutput]:
        """Return shared and routed outputs; routed output may defer finalization."""
        if self.attn_dp_size > 1 and forward_batch is None:
            raise ValueError("Step5 DP MoE requires native per-forward DP metadata")
        if self.attn_dp_size > 1:
            return self._forward_dp(hidden_states, forward_batch)
        overlap = (
            self.alt_stream is not None
            and 0 < hidden_states.shape[0] <= self.overlap_max_rows
            and forward_batch is not None
            and (
                forward_batch.forward_mode.is_decode()
                or forward_batch.forward_mode.is_target_verify()
            )
        )
        if overlap:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            # Keep collectives and routed workspace on the main stream.
            hidden_states.record_stream(self.alt_stream)
        else:
            shared_output = self.share_expert(hidden_states)
        topk = self.router(hidden_states, num_token_non_padded=num_token_non_padded)
        if self.use_mega_moe:
            routed_output = forward_mega_moe_local_experts(
                self.experts,
                hidden_states,
                topk.topk_ids,
                topk.topk_weights,
                group=get_parallel().moe_tp_group.device_group,
                activation_clamp=self.swiglu_limit,
            )
        elif (
            self.fuse_shared_finalize
            and hidden_states.dtype == torch.bfloat16
            and get_moe_runner_backend().value
            in ("flashinfer_trtllm_routed", "flashinfer_trtllm")
        ):
            routed_output = self.experts.forward_deferred_finalize(hidden_states, topk)
        else:
            routed_output = self.experts(hidden_states, topk)
        if overlap:
            with torch.cuda.stream(self.alt_stream):
                shared_output = self.share_expert(hidden_states)
            current_stream.wait_stream(self.alt_stream)
            shared_output.record_stream(current_stream)
        return shared_output, routed_output


_STEP5_ATTENTION_OWNERS: weakref.WeakValueDictionary[int, Step5Attention] = (
    weakref.WeakValueDictionary()
)


def _step5_attention_core(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    gate: torch.Tensor,
    output: torch.Tensor,
    owner_key: int,
    use_gate_epilogue: bool,
) -> None:
    forward_batch = get_tc_piecewise_forward_context().forward_batch
    owner = _STEP5_ATTENTION_OWNERS[owner_key]
    layer = owner.attn
    num_tokens = forward_batch.global_num_token_non_padded_cpu
    backend = get_attn_backend()
    kwargs = {}
    if use_gate_epilogue:
        kwargs["head_gate"] = gate[:num_tokens]
    if owner.is_dsa:
        metadata = backend.get_indexer_metadata(layer.layer_id, forward_batch)
        # The index plan includes DP-alignment rows, but not the larger graph
        # bucket tail. Its extent must also size the indexer and sparse queries.
        if metadata is None:
            output.zero_()
            return
        num_tokens = metadata.num_input_tokens
        kwargs["index_selection"] = owner.indexer(
            hidden_states[:num_tokens], positions[:num_tokens], metadata
        )
    out_cache_loc = forward_batch.out_cache_loc
    forward_batch.out_cache_loc = out_cache_loc[:num_tokens]
    try:
        ret = backend.forward(
            q[:num_tokens],
            k[:num_tokens].view(-1, layer.tp_k_head_num, layer.qk_head_dim),
            v[:num_tokens].view(-1, layer.tp_v_head_num, layer.v_head_dim),
            layer,
            forward_batch,
            save_kv_cache=True,
            **kwargs,
        )
    finally:
        forward_batch.out_cache_loc = out_cache_loc
    output[:num_tokens].copy_(ret.reshape(num_tokens, output.shape[-1]))
    output[num_tokens:].zero_()


_breakable_step5_attention_core = eager_on_graph(True)(_step5_attention_core)


class Step5Attention(nn.Module):
    """SWA/DSA attention with model-owned rotary caches and an optional Indexer."""

    def __init__(
        self,
        *,
        config: Step5Config,
        layer_id: int,
        tp_rank: int,
        tp_size: int,
        prefix: str,
        use_dp_attention_reduce: bool = False,
        profile=None,
    ) -> None:
        super().__init__()
        layer_type = config.layer_types_with_mtp[layer_id]
        self.is_dsa = layer_type == "full_attention"
        profile = profile or parse_step5_attention_profile(config)
        self.profile = profile
        self.rotary_emb = get_rope(
            head_size=config.head_dim,
            rotary_dim=config.head_dim,
            partial_rotary_factor=config.partial_rotary_factors[layer_id],
            max_position=config.max_position_embeddings + get_step5_position_padding(),
            base=config.rope_theta[layer_id],
            rope_scaling=config.rope_scaling if self.is_dsa else None,
            is_neox_style=True,
            dtype=torch.bfloat16,
            cache_dtype=torch.bfloat16,
        )
        self.head_dim = profile.head_dim
        self.num_heads = config.num_attention_heads // tp_size
        self.eps = config.rms_norm_eps
        self.qkv_proj = QKVParallelLinear(
            hidden_size=profile.hidden_size,
            head_size=profile.head_dim,
            total_num_heads=profile.num_heads,
            total_num_kv_heads=profile.num_kv_heads,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=add_prefix("qkv_proj", prefix),
            parallel_group="attn_tp",
        )
        self.g_proj = ColumnParallelLinear(
            input_size=profile.hidden_size,
            output_size=profile.num_heads,
            bias=False,
            gather_output=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=add_prefix("g_proj", prefix),
            parallel_group="attn_tp",
        )
        self.q_norm = GemmaRMSNorm(
            profile.head_dim, eps=self.eps, weight_dtype=torch.float32
        )
        self.k_norm = GemmaRMSNorm(
            profile.head_dim, eps=self.eps, weight_dtype=torch.float32
        )
        self.o_proj = RowParallelLinear(
            input_size=profile.num_heads * profile.head_dim,
            output_size=profile.hidden_size,
            bias=False,
            input_is_parallel=True,
            params_dtype=torch.bfloat16,
            quant_config=None,
            reduce_results=False,
            use_dp_attention_reduce=use_dp_attention_reduce,
            parallel_group="attn_tp",
            prefix=add_prefix("o_proj", prefix),
        )
        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=profile.head_dim,
            scaling=profile.head_dim**-0.5,
            num_kv_heads=1,
            layer_id=layer_id,
            quant_config=None,
            prefix=add_prefix("attn", prefix),
            # Window includes self; the backend takes left distance.
            sliding_window_size=-1 if self.is_dsa else config.sliding_window - 1,
        )
        if self.is_dsa:
            self.indexer = Step5Indexer(
                layer_id=layer_id,
                hidden_size=profile.hidden_size,
                index_head_dim=profile.index_head_dim,
                num_index_heads=profile.index_num_heads,
                index_tp_size=profile.provider_groups,
                tp_rank=tp_rank,
                tp_size=tp_size,
                rotary_dim=config.sparse_config["sparse_indexer_rope_dim"],
                rope_theta=config.rope_theta[layer_id],
                rope_scaling=config.rope_scaling,
                max_position_embeddings=config.max_position_embeddings,
                eps=self.eps,
                prefix=prefix,
            )
            # Checkpoint-only parameter; unused by weighted ReLU scoring.
            self.ssmax_s = nn.Parameter(
                torch.zeros(profile.num_heads, dtype=torch.float32), requires_grad=False
            )

        _STEP5_ATTENTION_OWNERS[id(self)] = self
        self.register_buffer("qkv_gate_weight", None, persistent=False)
        self._qkv_gate_layer = None
        weight = self.qkv_proj.weight
        if (
            weight.is_cuda
            and torch.version.hip is None
            and torch.cuda.get_device_capability(weight.device)[0] == 10
            and weight.shape in ((1920, 4096), (3456, 4096))
            and not get_lora().enable_lora
            and get_model().weight_cache_mode == "off"
            and get_model().load_format
            in ("auto", "pt", "safetensors", "npcache", "dummy")
            and get_exec().offload.cpu_offload_gb == 0
            and get_exec().offload.offload_group_size <= 0
        ):
            self.qkv_gate_weight = torch.empty(
                weight.shape[0] + self.num_heads,
                weight.shape[1],
                dtype=weight.dtype,
                device=weight.device,
            )
            self.qkv_proj.weight.data = self.qkv_gate_weight[: weight.shape[0]]
            self.g_proj.weight.data = self.qkv_gate_weight[weight.shape[0] :]
            self._qkv_gate_layer = SimpleNamespace(weight=self.qkv_gate_weight)
        self.alt_stream = get_stream("alt") if self.qkv_proj.weight.is_cuda else None
        self.dsa_overlap_max_rows = (
            256
            if self.is_dsa
            and self.qkv_proj.weight.is_cuda
            and torch.version.hip is None
            and torch.cuda.get_device_capability(self.qkv_proj.weight.device) == (10, 3)
            and self.qkv_proj.weight.shape in ((1920, 4096), (3456, 4096))
            else 255
        )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        if self.qkv_gate_weight is not None:
            rows = self.qkv_proj.weight.shape[0]
            if (
                self.qkv_proj.weight.data_ptr() != self.qkv_gate_weight.data_ptr()
                or self.g_proj.weight.data_ptr()
                != self.qkv_gate_weight[rows].data_ptr()
            ):
                self.qkv_gate_weight = None
                self._qkv_gate_layer = None
            else:
                self._qkv_gate_layer = SimpleNamespace(weight=self.qkv_gate_weight)
        return self

    def forward(
        self,
        *,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if is_in_breakable_cuda_graph():
            return self._forward_breakable_extend(positions, hidden_states)
        fused_projection = self.qkv_gate_weight is not None
        backend = get_attn_backend()
        use_alt_stream = (
            (not fused_projection or self.is_dsa)
            and self.alt_stream is not None
            and hidden_states.is_cuda
            and 0 < hidden_states.shape[0] <= self.dsa_overlap_max_rows
            and (
                forward_batch.forward_mode.is_decode()
                or forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend_v2()
            )
        )
        use_indexer_alt_stream = use_alt_stream and self.is_dsa
        gate_on_main = (
            not fused_projection
            and use_indexer_alt_stream
            # TODO(yhyang201): frozen SGLANG_OPT_STEP5_INDEXER_WEIGHT_FIRST; prune dead paths.
            and step5_options.INDEXER_WEIGHT_FIRST
        )
        defer_side_work = use_alt_stream and torch.cuda.is_current_stream_capturing()
        gate = None
        selection = None
        idle = forward_batch.forward_mode.is_idle()
        if self.is_dsa and not idle:
            indexer_metadata = backend.get_indexer_metadata(
                self.attn.layer_id, forward_batch
            )
        if use_alt_stream:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            hidden_states.record_stream(self.alt_stream)

        def run_side_work():
            nonlocal gate
            with torch.cuda.stream(self.alt_stream):
                if not fused_projection and not gate_on_main:
                    gate, _ = self.g_proj(hidden_states)
                if use_indexer_alt_stream:
                    return self.indexer(
                        hidden_states,
                        positions,
                        indexer_metadata,
                        # TODO(yhyang201): frozen SGLANG_OPT_STEP5_INDEXER_WEIGHT_FIRST; prune dead paths.
                        weights_first=step5_options.INDEXER_WEIGHT_FIRST,
                    )
            return None

        if fused_projection:
            qkv = self.qkv_proj.quant_method.apply(self._qkv_gate_layer, hidden_states)
        else:
            qkv, _ = self.qkv_proj(hidden_states)
        if use_alt_stream:
            # Keep eager submission order. During capture, record the side
            # branch just before its join so replay retains the main lane.
            # The fork above stays in place, preserving graph overlap.
            selection = run_side_work if defer_side_work else run_side_work()
        elif not fused_projection:
            gate, _ = self.g_proj(hidden_states)
        use_gate_epilogue = (
            not self.is_dsa
            # TODO(yhyang201): frozen SGLANG_OPT_STEP5_SWA_FA4_GATE_EPILOGUE; prune dead paths.
            and step5_options.SWA_FA4_GATE_EPILOGUE
            and backend.supports_swa_head_gate()
        )
        use_dsa_gate_epilogue = (
            self.is_dsa
            and not torch.compiler.is_compiling()
            and not get_exec().deterministic.enable_deterministic_inference
            # TODO(yhyang201): frozen SGLANG_OPT_STEP5_HEAD_GATE; prune dead paths.
            and step5_options.HEAD_GATE
            and forward_batch.forward_mode.is_target_verify()
            and hidden_states.shape[0] == 4
            and hidden_states.dtype == torch.bfloat16
            and backend.supports_dsa_head_gate()
        )
        kv_write = None
        # Idle dummy rows must not write KV, including through fused store.
        if (
            not self.is_dsa
            and not idle
            # TODO(yhyang201): frozen SGLANG_OPT_STEP5_SWA_FUSED_QKNORM_STORE; prune dead paths.
            and step5_options.SWA_FUSED_QKNORM_STORE
        ):
            candidate = backend.swa_kv_write_target(self.attn)
            if (
                candidate is not None
                and candidate[2].shape == (qkv.shape[0],)
                and candidate[2].dtype == torch.int64
            ):
                kv_write = candidate
        save_kv_cache = kv_write is None and not idle
        if fused_projection:
            q, k, v, gate = step5_qknorm_rope_gated(
                qkv,
                self.q_norm.weight,
                self.k_norm.weight,
                positions,
                self.rotary_emb.cos_sin_cache,
                eps=self.eps,
                kv_write=kv_write,
            )
        else:
            q, k, v = step5_qknorm_rope(
                qkv,
                self.q_norm.weight,
                self.k_norm.weight,
                positions,
                self.rotary_emb.cos_sin_cache,
                eps=self.eps,
                kv_write=kv_write,
            )
        if self.is_dsa:
            if not idle and not use_indexer_alt_stream:
                selection = self.indexer(hidden_states, positions, indexer_metadata)

            def prepare_selection_and_gate():
                nonlocal gate
                # The backend invokes this after its main KV write and before
                # joining the indexer stream. Keep the original graph fork so
                # indexer projections overlap QKV without waiting for the gate.
                result = selection() if callable(selection) else selection
                gate, _ = self.g_proj(hidden_states)
                return result

            # Read the gate after resolving selection and joining the indexer,
            # whether it was projected on the main or alternate stream.
            gate_kwargs = (
                {"head_gate_getter": lambda: gate} if use_dsa_gate_epilogue else {}
            )
            output = self.attn(
                q,
                k,
                v,
                forward_batch,
                save_kv_cache=save_kv_cache,
                index_selection=(
                    prepare_selection_and_gate if gate_on_main else selection
                ),
                index_stream=self.alt_stream if use_indexer_alt_stream else None,
                **gate_kwargs,
            )
        elif use_gate_epilogue:
            if defer_side_work:
                run_side_work()
            if use_alt_stream:
                current_stream.wait_stream(self.alt_stream)
                gate.record_stream(current_stream)
            output = self.attn(
                q, k, v, forward_batch, head_gate=gate, save_kv_cache=save_kv_cache
            )
        else:
            output = self.attn(q, k, v, forward_batch, save_kv_cache=save_kv_cache)
        if not (use_gate_epilogue or use_dsa_gate_epilogue):
            if defer_side_work and not self.is_dsa:
                run_side_work()
            if use_alt_stream and not fused_projection:
                # DSA already joined this branch before sparse attention.
                # Recording another join after it would migrate the main
                # CUDA graph replay chain again.
                if not (defer_side_work and self.is_dsa):
                    current_stream.wait_stream(self.alt_stream)
                gate.record_stream(current_stream)
            output = self._apply_head_gate(output, gate)
        output, _ = self.o_proj(output.flatten(1))
        return output

    def _apply_head_gate(
        self, output: torch.Tensor, gate: torch.Tensor
    ) -> torch.Tensor:
        # TODO(yhyang201): frozen SGLANG_OPT_STEP5_HEAD_GATE; prune dead paths.
        if step5_options.HEAD_GATE:
            return fused_sigmoid_mul(
                output.reshape(-1, self.num_heads * self.head_dim),
                gate.unsqueeze(-1).expand(-1, -1, self.head_dim),
                round_gate=True,
            )
        return output.reshape(
            -1, self.num_heads, self.head_dim
        ) * gate.sigmoid().unsqueeze(-1)

    def _forward_breakable_extend(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if self.qkv_gate_weight is not None:
            q, k, v, gate = step5_qknorm_rope_gated(
                self.qkv_proj.quant_method.apply(self._qkv_gate_layer, hidden_states),
                self.q_norm.weight,
                self.k_norm.weight,
                positions,
                self.rotary_emb.cos_sin_cache,
                eps=self.eps,
                kv_write=None,
            )
        else:
            qkv, _ = self.qkv_proj(hidden_states)
            gate, _ = self.g_proj(hidden_states)
            q, k, v = step5_qknorm_rope(
                qkv,
                self.q_norm.weight,
                self.k_norm.weight,
                positions,
                self.rotary_emb.cos_sin_cache,
                eps=self.eps,
                kv_write=None,
            )
        use_gate_epilogue = (
            not self.is_dsa
            # TODO(yhyang201): frozen SGLANG_OPT_STEP5_SWA_FA4_GATE_EPILOGUE; prune dead paths.
            and step5_options.SWA_FA4_GATE_EPILOGUE
            and get_attn_backend().supports_swa_head_gate()
        )
        backend = get_attn_backend()
        if not self.is_dsa and getattr(backend, "swa_graph_active", False):
            forward_batch = get_tc_piecewise_forward_context().forward_batch
            kwargs = {"head_gate": gate} if use_gate_epilogue else {}
            output = backend.forward_swa_in_graph(
                q, k, v, self.attn, forward_batch, **kwargs
            )
        else:
            output = q.new_empty((q.shape[0], self.num_heads * self.head_dim))
            _breakable_step5_attention_core(
                q,
                k,
                v,
                hidden_states,
                positions,
                gate,
                output,
                id(self),
                use_gate_epilogue,
            )
        if not use_gate_epilogue:
            output = self._apply_head_gate(output, gate)
        output, _ = self.o_proj(output.flatten(1))
        return output


class Step5DecoderLayer(nn.Module):
    """Stage boundaries own residuals, decoder norms and deferred reduction."""

    def __init__(
        self,
        *,
        config: Step5Config,
        layer_id: int,
        quant_config: QuantizationConfig | None,
        tp_rank: int,
        tp_size: int,
        prefix: str,
        use_dp_attention_reduce: bool = False,
        profile=None,
    ) -> None:
        super().__init__()
        # Native Gemma loading casts FP32 checkpoint weights and updates 1 + weight.
        self.use_dp_attention_reduce = use_dp_attention_reduce
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.self_attn = Step5Attention(
            config=config,
            layer_id=layer_id,
            tp_rank=tp_rank,
            tp_size=tp_size,
            use_dp_attention_reduce=use_dp_attention_reduce,
            profile=profile,
            prefix=add_prefix("self_attn", prefix),
        )
        self.use_moe = layer_id in config.moe_layer_list
        self.input_layernorm = GemmaRMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            weight_dtype=torch.bfloat16,
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            weight_dtype=torch.bfloat16,
        )
        if self.use_moe:
            self.moe = Step5MoE(
                layer_id=layer_id,
                hidden_size=config.hidden_size,
                num_experts=config.moe_num_experts,
                intermediate_size=config.moe_intermediate_size,
                shared_intermediate_size=config.share_expert_dim,
                top_k=config.moe_top_k,
                routed_scaling_factor=config.moe_router_scaling_factor,
                swiglu_limit=config.swiglu_limits[layer_id],
                shared_swiglu_limit=config.swiglu_limits_shared[layer_id],
                quant_config=quant_config,
            )
        else:
            self.mlp = Step5MLP(
                role="dense",
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                swiglu_limit=config.swiglu_limits_shared[layer_id],
                tp_rank=tp_rank,
                tp_size=tp_size,
                prefix=f"{prefix}.mlp",
                use_dp_attention_reduce=use_dp_attention_reduce,
            )

        self.boundary_fusions = Step5BoundaryFusions(self.input_layernorm)
        if not use_dp_attention_reduce:
            if get_exec().comm.boundary_reduction != "ar":
                raise ValueError("Step5 requires --boundary-reduction ar")
            attn, ffn = self.stage_facts(config, layer_id)
            self.attn_boundary, self.ffn_boundary = append_stages(
                (
                    attn,
                    self.input_layernorm,
                    {"fusions": self.boundary_fusions},
                ),
                (
                    ffn,
                    self.post_attention_layernorm,
                    {"fusions": self.boundary_fusions},
                ),
            )

    @staticmethod
    def stage_facts(config: Step5Config, layer_id: int):
        # Local MegaMoE experts consume attention-TP rows and return TP partials.
        return (
            declare_attn(),
            declare_ffn(
                sparse=layer_id in config.moe_layer_list
                and not get_moe_a2a_backend().is_megamoe()
            ),
        )

    def _forward_ffn(
        self, *, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor | Step5DeferredMoE:
        if not self.use_moe:
            return self.mlp(hidden_states)
        if self.use_dp_attention_reduce:
            shared, routed = self.moe(hidden_states, forward_batch=forward_batch)
            return routed + shared
        shared, routed = self.moe(
            hidden_states,
            num_token_non_padded=forward_batch.num_token_non_padded,
            forward_batch=forward_batch,
        )
        if isinstance(routed, FlashInferTrtllmDeferredFinalizeOutput):
            if get_forward().defer_moe_finalize and self.boundary_fusions.can_fuse_moe(
                shared, routed
            ):
                return Step5DeferredMoE(routed, shared)
            combined = finalize_flashinfer_trtllm_deferred_output(
                routed, shared, output_dtype=torch.bfloat16, round_routed=True
            )
        else:
            if self.moe.use_mega_moe and get_forward().defer_moe_finalize:
                deferred = self.boundary_fusions.defer_combined_moe(routed, shared)
                if deferred is not None:
                    return deferred
            combined = routed + shared
        # Stage boundaries complete or carry the routed + shared TP sum,
        # including under LoRA and at the terminal layer. Return partials here.
        return combined

    def forward(
        self,
        *,
        positions,
        hidden_states,
        forward_batch,
        captured_last_layer_outputs=None,
    ):
        if self.use_dp_attention_reduce:
            # Routed experts own their EP gather/scatter; shared and dense
            # experts compute on attention-TP rows. No global FFN gather.
            stream = residual_batch.stream_of(forward_batch)
            hidden_states, residual = stream.export(hidden_states)
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
            if captured_last_layer_outputs is not None:
                captured_last_layer_outputs.capture(residual, owned=False)
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
            hidden_states = attention_tensor_model_parallel_all_reduce(hidden_states)
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
            with get_forward().scoped(
                mlp_reduce_scatter=False,
                defer_moe_finalize=False,
            ):
                hidden_states = self._forward_ffn(
                    hidden_states=hidden_states, forward_batch=forward_batch
                )
            from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_ADD
            from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream

            hidden_states, forward_batch.residual_stream = ResidualStream.from_handoff(
                hidden_states, residual, PLAIN_ADD
            )
            return hidden_states

        hidden_states = self.attn_boundary.prepare(
            hidden_states, forward_batch, capture_gathered=captured_last_layer_outputs
        )
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states = self.attn_boundary.finish(hidden_states, forward_batch)
        hidden_states = self.ffn_boundary.prepare(hidden_states, forward_batch)
        with self.ffn_boundary.exit(forward_batch) as ffn_exit:
            hidden_states = self._forward_ffn(
                hidden_states=hidden_states, forward_batch=forward_batch
            )
        return ffn_exit.finish(hidden_states)


class Step5Model(nn.Module):
    """Own decoder layers and shared RoPE buffers; construct on the final device."""

    def __init__(
        self,
        *,
        config: Step5Config,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.attention_profile = parse_step5_attention_profile(config)
        self.nvfp4_expert_layouts = step5_nvfp4_expert_layouts(
            config, quant_config, get_model().model_path
        )
        self.nvfp4_expert_layers = frozenset(self.nvfp4_expert_layouts)
        parallel = get_parallel()
        use_dp_attention_reduce = parallel.attn_dp_size > 1
        self.step5_layer_selection = resolve_step5_layer_selection(
            config=config, draft_model_idx=None
        )
        self.start_layer = self.step5_layer_selection.start_layer
        self.end_layer = self.step5_layer_selection.end_layer
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            params_dtype=torch.bfloat16,
            quant_config=None,
            use_attn_tp_group=use_dp_attention_reduce,
            prefix=add_prefix("embed_tokens", prefix),
        )
        with layer_stack(
            previous_layers=[
                lambda i=i: Step5DecoderLayer.stage_facts(config, i)
                for i in range(self.start_layer - 1, -1, -1)
            ],
            next_layers=[
                lambda i=i: Step5DecoderLayer.stage_facts(config, i)
                for i in range(self.end_layer, config.num_hidden_layers)
            ],
        ):
            self.layers = nn.ModuleList(
                Step5DecoderLayer(
                    profile=self.attention_profile,
                    config=config,
                    quant_config=step5_expert_quant_config(
                        quant_config, layer_id, self.nvfp4_expert_layers
                    ),
                    layer_id=layer_id,
                    tp_rank=parallel.attn_tp_rank,
                    tp_size=parallel.attn_tp_size,
                    use_dp_attention_reduce=use_dp_attention_reduce,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                )
                for layer_id in range(self.start_layer, self.end_layer)
            )
        self.norm = GemmaRMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            weight_dtype=torch.float32,
        )

    def get_input_embeddings(self) -> VocabParallelEmbedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        ).bfloat16()
        residual_batch.start(forward_batch)
        for layer in self.layers:
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        # Final norm and MTP capture consume the materialized BF16 residual sum.
        hidden_states, residual = residual_batch.stream_of(forward_batch).export(
            hidden_states
        )
        forward_batch.residual_stream = None
        hidden_states = hidden_states + residual
        normalized = self.norm(hidden_states, backend=KernelBackend.TORCH)
        return normalized, hidden_states.float()


class Step5ForCausalLM(nn.Module, RoutedExpertLoRATargets):
    kv_cache_dtype_full_attention_only = True

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.moe_num_experts,
        )

    def __init__(
        self,
        config: Step5Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        valid_vocab_size: int,
    ) -> None:
        super().__init__()
        if (
            type(valid_vocab_size) is not int
            or not 0 < valid_vocab_size <= config.vocab_size
        ):
            raise ValueError(
                "Step5 requires a resolved effective vocabulary at construction"
            )
        self.config = config
        self.checkpoint_path = get_model().model_path
        self.quant_config = quant_config
        self.model = Step5Model(
            config=config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.layer_selection = self.model.step5_layer_selection
        self.nvfp4_expert_layers = self.model.nvfp4_expert_layers
        self.nvfp4_expert_layouts = self.model.nvfp4_expert_layouts
        self.tie_word_embeddings = False
        self.num_fused_shared_experts = 0
        # Only the checkpoint load must cover every target weight; later load_weights calls
        # are weight updates and, as for other models, may stream any subset.
        self._checkpoint_loaded = False
        self._checkpoint_duplicate_names = frozenset()
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_parallel().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(
            config, valid_vocab_size=valid_vocab_size
        )

    @property
    def start_layer(self):
        return self.layer_selection.start_layer

    @property
    def end_layer(self):
        return self.layer_selection.end_layer

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def get_hidden_dim(self, module_name: str, layer_idx: int) -> tuple[int, int]:
        """LoRA (input, output) widths; an MoE layer's gate_up/down is its shared expert."""
        config = self.config
        if module_name == "qkv_proj":
            heads = config.num_attention_heads + 2 * config.num_key_value_heads
            return config.hidden_size, heads * config.head_dim
        if module_name == "o_proj":
            return config.num_attention_heads * config.head_dim, config.hidden_size
        intermediate = (
            config.share_expert_dim
            if layer_idx in config.moe_layer_list
            else config.intermediate_size
        )
        if module_name == "gate_up_proj":
            return config.hidden_size, 2 * intermediate
        if module_name == "down_proj":
            return intermediate, config.hidden_size
        if module_name == "gate_up_proj_moe":
            return config.hidden_size, 2 * config.moe_intermediate_size
        if module_name == "down_proj_moe":
            return config.moe_intermediate_size, config.hidden_size
        # The head-wise gate (g_proj) stays frozen.
        raise ValueError(f"Step5 LoRA does not adapt {module_name}")

    def lora_targets_routed_experts(self, target_modules: Iterable[str]) -> bool:
        """gate_up_proj/down_proj also name the dense MLPs and the shared expert, so
        only a target under the routed experts (an `experts` path component, as in
        `moe.experts.gate_proj` or `experts.down_proj`), or all linear layers, adapts them."""
        adapts = any(
            target in ("all", "all-linear") or "experts" in target.split(".")
            for target in target_modules
        )
        if adapts:
            for layer in self.model.layers:
                if layer.use_moe:
                    layer.moe.check_routed_expert_lora_support()
        return adapts

    @staticmethod
    def _checkpoint_duplicate_owners(expected, model_path) -> dict[str, str]:
        # names stored in more than one shard -> the shard the index names
        from safetensors import safe_open

        if not model_path:
            return {}
        root = Path(model_path)
        index_path = root / "model.safetensors.index.json"
        if not index_path.is_file():
            return {}
        owners = json.loads(index_path.read_text())["weight_map"]
        occurrences = {}
        for filename in set(owners.values()):
            with safe_open(root / filename, framework="pt", device="cpu") as shard:
                for name in expected.intersection(shard.keys()):
                    view = shard.get_slice(name)
                    signature = (view.get_dtype(), tuple(view.get_shape()))
                    if name in occurrences and occurrences[name][0] != signature:
                        raise ValueError(
                            f"Step5 duplicate dtype/shape mismatch: {name}"
                        )
                    previous = occurrences.get(name, (signature, []))[1]
                    occurrences[name] = (signature, [*previous, filename])
        for name in expected.intersection(owners):
            if owners[name] not in occurrences.get(name, (None, []))[1]:
                raise ValueError(f"Step5 indexed owner does not contain weight: {name}")
        duplicates = {}
        for name, (_, files) in occurrences.items():
            if len(files) > 1:
                if owners.get(name) not in files:
                    raise ValueError(f"Step5 duplicate has no indexed owner: {name}")
                duplicates[name] = owners[name]
        return duplicates

    @staticmethod
    def _indexed_duplicate_weights(weights, duplicates, model_path):
        # Native iterators do not retain shard provenance. For duplicate names,
        # read the indexed owner explicitly rather than depend on arrival order.
        from safetensors import safe_open

        root = Path(model_path) if model_path else None
        emitted = set()
        for name, weight in weights:
            if name not in duplicates:
                yield name, weight
                continue
            if name in emitted:
                continue
            with safe_open(
                root / duplicates[name], framework="pt", device="cpu"
            ) as shard:
                canonical = shard.get_tensor(name)
                if weight.dtype != canonical.dtype or weight.shape != canonical.shape:
                    raise ValueError(
                        f"Step5 streamed duplicate dtype/shape mismatch: {name}"
                    )
                yield name, canonical
            emitted.add(name)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = (
            ("self_attn.qkv_proj.weight", "self_attn.q_proj.weight", "q"),
            ("self_attn.qkv_proj.weight", "self_attn.k_proj.weight", "k"),
            ("self_attn.qkv_proj.weight", "self_attn.v_proj.weight", "v"),
            ("mlp.gate_up_proj.weight", "mlp.gate_proj.weight", 0),
            ("mlp.gate_up_proj.weight", "mlp.up_proj.weight", 1),
            (
                "moe.share_expert.gate_up_proj.weight",
                "moe.share_expert.gate_proj.weight",
                0,
            ),
            (
                "moe.share_expert.gate_up_proj.weight",
                "moe.share_expert.up_proj.weight",
                1,
            ),
        )

        expert_params_mapping = {
            f"{projection}_proj.{suffix}": (f"{target}{scale_suffix}", shard)
            for projection, target, shard in (
                ("gate", "w13_weight", "w1"),
                ("up", "w13_weight", "w3"),
                ("down", "w2_weight", "w2"),
            )
            for suffix, scale_suffix in (
                ("weight", ""),
                ("weight_scale_inv", "_scale_inv"),
            )
        }

        params_mapping = {
            "moe.gate.weight": "moe.router.gate.weight",
            "moe.router_bias": "moe.router.router_bias",
            **{
                f"self_attn.sparse_indexer_{suffix}": f"self_attn.indexer.sparse_indexer_{suffix}"
                for suffix in (
                    "q.weight",
                    "k.weight",
                    "z.weight",
                    "w.weight",
                    "q_norm.weight",
                    "k_norm.weight",
                    "k_norm.bias",
                )
            },
        }

        params_dict = dict(self.named_parameters())
        config = self.config
        # Expand actual parameters into external shards to detect partial QKV/MLP loads.
        expected = set()
        for name in params_dict:
            head, marker, target = name.rpartition(".moe.experts.")
            if marker and _step5_layer_index(head) in self.nvfp4_expert_layers:
                sources = _STEP5_NVFP4_PARAM_SOURCES.get(target)
                if (
                    sources
                    and self.nvfp4_expert_layouts[_step5_layer_index(head)] == "packed"
                ):
                    expected.update(f"{head}.moe.{source}" for source in sources)
                elif sources:
                    expected.update(
                        f"{head}{marker}{expert_id}.{source}"
                        for expert_id in range(config.moe_num_experts)
                        for source in sources
                    )
                continue
            sources = [
                name[: -len(target)] + source
                for target, source, _ in stacked_params_mapping
                if name.endswith("." + target)
            ]
            sources += [
                name[: -len(target)] + source
                for source, target in params_mapping.items()
                if name.endswith("." + target)
            ]
            sources += [
                name[: -len("moe.experts." + target)] + "moe." + source
                for source, (target, _) in expert_params_mapping.items()
                if name.endswith(".moe.experts." + target)
            ]
            expected.update(sources or [name])
        seen, original_names = set(), set()
        checkpoint_depth = config.num_hidden_layers + config.num_nextn_predict_layers
        checkpoint_load = not self._checkpoint_loaded
        if checkpoint_load:
            duplicates = self._checkpoint_duplicate_owners(
                expected, self.checkpoint_path
            )
            # a disk reload streams both copies again, with no shard provenance
            self._checkpoint_duplicate_names = frozenset(duplicates)
            weights = self._indexed_duplicate_weights(
                weights, duplicates, self.checkpoint_path
            )
        first_copies = {}
        for original_name, loaded_weight in weights:
            name = original_name
            if name.startswith("vision_model.") or name == "vit_large_projector.weight":
                continue
            if name.startswith("model.layers."):
                _, _, index, local = name.split(".", 3)
                if (
                    not index.isascii()
                    or not index.isdecimal()
                    or str(int(index)) != index
                ):
                    raise ValueError(f"Invalid Step5 checkpoint layer: {name}")
                if config.num_hidden_layers <= int(index) < checkpoint_depth:
                    continue
                if local.startswith("share_expert."):
                    name = f"model.layers.{index}.moe.{local}"
            expert_ids, seen_key = range(config.moe_num_experts), name
            if name not in expected and not checkpoint_load:
                row = _step5_routed_expert_row(name, config.moe_num_experts)
                if row is not None and row[0] in expected:
                    # one expert's row of the packed tensor; duplicates are per expert
                    name, expert_ids, seen_key = row[0], (row[1],), original_name
                    loaded_weight = loaded_weight.unsqueeze(0)
            if name not in expected:
                raise ValueError(f"Unknown Step5 target weight: {name}")
            if not checkpoint_load and name in self._checkpoint_duplicate_names:
                if name in first_copies:
                    if not torch.equal(first_copies[name], loaded_weight):
                        raise ValueError(
                            f"Step5 duplicate copies differ: {original_name}"
                        )
                    continue
                first_copies[name] = loaded_weight
            if seen_key in seen:
                raise ValueError(f"Duplicate Step5 target weight: {original_name}")
            seen.add(seen_key)
            original_names.add(original_name)
            nvfp4_source = step5_nvfp4_expert_source(name)
            if nvfp4_source is not None:
                param_key, shard_id, expert_id = nvfp4_source
                param = params_dict[param_key]
                param.weight_loader(
                    param,
                    loaded_weight,
                    param_key,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                continue
            packed_source = step5_nvfp4_packed_source(name, self.nvfp4_expert_layouts)
            if packed_source is not None:
                param_key, shard_id = packed_source
                param = params_dict[param_key]
                for expert_id in range(config.moe_num_experts):
                    param.weight_loader(
                        param,
                        loaded_weight[expert_id],
                        param_key,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                continue
            for weight_name, (param_name, shard_id) in expert_params_mapping.items():
                suffix = ".moe." + weight_name
                if not name.endswith(suffix):
                    continue
                name = name[: -len(suffix)] + ".moe.experts." + param_name
                param = params_dict[name]
                if (
                    weight_name.endswith(".weight")
                    and loaded_weight.dtype == torch.float8_e4m3fn
                    and param.dtype != torch.float8_e4m3fn
                ):
                    # Without its scales, copying FP8 into BF16 silently drops the dequantization.
                    raise ValueError(
                        f"Refusing FP8 weight for {param.dtype} Step5 experts: {original_name}. "
                        "Send these experts in their served precision."
                    )
                # Native loaders own EP/TP slicing; quantize only resident experts.
                quantize_bf16 = (
                    weight_name.endswith(".weight")
                    and loaded_weight.dtype == torch.bfloat16
                    and param.dtype == torch.float8_e4m3fn
                )
                if quantize_bf16:
                    # Weight updates re-quantize BF16 into FP8 experts; at checkpoint load the
                    # experts' precision must come from the checkpoint or the explicit opt-in.
                    if (
                        checkpoint_load
                        and not envs.SGLANG_STEP5_QUANTIZE_BF16_MOE.get()
                    ):
                        raise ValueError(
                            f"Refusing implicit BF16-to-FP8 expert quantization: {original_name}. "
                            "Declare the BF16 experts in quantization_config.ignored_layers "
                            "or explicitly set SGLANG_STEP5_QUANTIZE_BF16_MOE=1."
                        )
                    experts = self.get_submodule(name.rsplit(".", 1)[0])
                    block_n, block_k = experts.quant_method.weight_block_size
                    scale_name = name + "_scale_inv"
                    scale_param = params_dict[scale_name]
                    generated_scale = original_name + "_scale_inv"
                    if generated_scale in seen:
                        raise ValueError(
                            f"BF16 expert weight has checkpoint scales: {original_name}"
                        )
                    seen.add(generated_scale)
                for row, expert_id in enumerate(expert_ids):
                    weight = loaded_weight[row]
                    if quantize_bf16:
                        local_id = experts._map_global_expert_id_to_local_expert_id(
                            expert_id
                        )
                        if not 0 <= local_id < experts.num_local_experts:
                            continue
                        n, k = weight.shape
                        blocks = weight.to(
                            device=param.device, dtype=torch.float32
                        ).reshape(n // block_n, block_n, k // block_k, block_k)
                        # Round the loading-time scale division before the FP8 cast.
                        scale = (
                            blocks.abs().amax(dim=(1, 3)).clamp_min(1e-12).double()
                            / 448.0
                        ).float()
                        weight = (
                            (blocks / scale[:, None, :, None])
                            .clamp(-448.0, 448.0)
                            .to(torch.float8_e4m3fn)
                            .reshape(n, k)
                        )
                        scale_param.weight_loader(
                            scale_param,
                            scale,
                            scale_name,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    param.weight_loader(
                        param,
                        weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                break
            else:
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if not name.endswith("." + weight_name):
                        continue
                    name = name[: -len(weight_name)] + param_name
                    param = params_dict[name]
                    param.weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    for weight_name, param_name in params_mapping.items():
                        if name.endswith("." + weight_name):
                            name = name[: -len(weight_name)] + param_name
                            break
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

        missing = expected - seen
        if checkpoint_load and missing:
            raise ValueError(
                f"Step5 target is missing {len(missing)} weights: {sorted(missing)[:16]}"
            )
        self._checkpoint_loaded = True
        for layer in self.model.layers:
            if layer.use_moe and layer.moe.use_nvfp4_experts:
                step5_check_nvfp4_expert_scales(layer.moe.experts, layer.moe.prefix)
        return original_names

    def post_load_weights(self) -> None:
        # Loaders that bypass load_weights (dummy, sharded state, remote) call this once
        # their weights are in place, so the next load_weights is an update.
        self._checkpoint_loaded = True

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
    ):
        hidden_states, hidden_states_before_norm = self.model(
            input_ids, positions, forward_batch, input_embeds
        )
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            hidden_states_before_norm=hidden_states_before_norm,
        )


EntryClass = Step5ForCausalLM
