"""Ascend MoE runner backend with NPU‑specific ops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.distributed.communication_op import tensor_model_parallel_all_gather
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.moe import standard_dispatch as ascend_dispatch
from sglang.srt.hardware_backend.npu.moe.activation import (
    AllGatherActivationWrapper,
    NPUGeluAndMul,
    NPUSitu,
    NPUSituMXFP8Quant,
    NPUSwiglu,
    NPUSwigluDeepEPKernel,
    NPUSwigluMxfp8Quant,
    NPUSwigluOAI,
    NPUSwigluQuant,
    NPUSwigluStepAndMul,
    NPUSwigluWithLimit,
)
from sglang.srt.hardware_backend.npu.moe.routing import (
    INT8_QUANT_MODE,
    npu_moe_finalize_routing,
    npu_moe_init_routing,
    quant_mode_dtype,
)
from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUMXFP8MoEMethod,
    NPUW4A4MXFP4MoEMethod,
    NPUW4A8Int8MoEMethod,
    NPUW4A8MXFP4MoEMethod,
    NPUW8A8Int8MoEMethod,
)
from sglang.srt.runtime_context import get_parallel


def _uses_fused_gmm1(kernel, config: MoeRunnerConfig) -> bool:
    """Whether gmm1 runs matmul+swiglu+requant fused (no separate activation)."""
    # The fused gmm1 kernel (npu_grouped_matmul_swiglu_quant_v2) applies plain
    # SiLU with no clamp. Models whose activation is SiLU-with-clamp
    # (swiglu_limit, e.g. DSV4) must keep the unfused path so the clamp is
    # applied (NPUSwigluMxfp8Quant / NPUSwigluStepAndMul below).
    has_clamp = config.swiglu_limit is not None and config.swiglu_limit > 0

    if isinstance(kernel, NPUW4A4MXFP4MoEMethod):
        if has_clamp:
            raise NotImplementedError(
                "NPUW4A4MXFP4MoEMethod does not support swiglu_limit."
            )
        # Fuse only plain SiLU under ascend_tp. Keep the runner's unfused
        # activation for SiTU, custom alpha/clamp, and unvalidated DeepEP.
        return (
            config.activation == "silu"
            and config.gemm1_alpha is None
            and config.gemm1_clamp_limit is None
            and not get_moe_a2a_backend().is_deepep()
        )

    if config.activation == "situ":
        return False
    if not isinstance(kernel, (NPUMXFP8MoEMethod, NPUW4A8MXFP4MoEMethod)):
        return False
    if has_clamp:
        return False  # Keep the separate activation to preserve the clamp.
    if isinstance(kernel, NPUMXFP8MoEMethod):
        return True
    return kernel.use_fused_gmm1


def _reshape_received_scale(received_scale: torch.Tensor) -> torch.Tensor:
    """Normalize communicated MX scales, including zero-row receives."""
    if received_scale.ndim == 3 and received_scale.shape[-1] == 2:
        return received_scale
    if received_scale.ndim != 2 or received_scale.shape[1] % 2 != 0:
        raise RuntimeError(
            f"invalid communicated MoE scale shape: {tuple(received_scale.shape)}"
        )
    return received_scale.reshape(
        received_scale.shape[0], received_scale.shape[1] // 2, 2
    )


from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_post_permute,
    register_pre_permute,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.ascend_ops import (
        AscendOpsCombineInput,
        AscendOpsDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLCombineInput,
        DeepEPLLDispatchOutput,
        DeepEPNormalCombineInput,
        DeepEPNormalDispatchOutput,
    )

from sglang.srt.layers.moe.utils import (
    MoeRunnerBackend,
    get_moe_a2a_backend,
)


# ---------------------------------------------------------------------------
# Runner IO dataclasses
# ---------------------------------------------------------------------------
@dataclass
class AscendRunnerInput(RunnerInput):
    """Input bundle for the NPU runner."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]  # None for unquant
    expert_tokens: torch.Tensor
    group_list_type: int  # 0 or 1 (passed to NPU ops)

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.ASCEND


@dataclass
class AscendRunnerOutput(RunnerOutput):
    """Output bundle from the NPU runner."""

    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.ASCEND


# ---------------------------------------------------------------------------
# Main runner core
# ---------------------------------------------------------------------------
class AscendRunnerCore(MoeRunnerCore):
    runner_backend = MoeRunnerBackend.ASCEND

    def __init__(self, config: MoeRunnerConfig):
        from sglang.srt.hardware_backend.npu.quantization.block_fp8_moe import (
            _NPUBlockFp8MoEKernel,
        )

        super().__init__(config)

        kernel = config.layer.w2_kernel

        if _uses_fused_gmm1(kernel, config):
            # Fused methods (MXFP8, W4A8, and ascend_tp W4A4) fold
            # gate/up + swiglu + requant into gmm1, so there is no separate
            # activation step — run() skips it. Left None on purpose so that
            # reaching for it fails loudly instead of silently applying an
            # unfused swiglu to already-requantised activations. This holds
            # for each transport: routing may quantize the input, while BF16
            # dispatcher output is quantized before gmm1 when scales are absent.
            self.activation = None
        elif (
            isinstance(kernel, (NPUW4A8MXFP4MoEMethod, NPUMXFP8MoEMethod))
            and config.swiglu_limit is not None
            and config.swiglu_limit > 0
        ):
            self.activation = NPUSwigluMxfp8Quant(config.swiglu_limit)
        elif (
            isinstance(kernel, _NPUBlockFp8MoEKernel)
            and config.swiglu_limit is not None
            and config.swiglu_limit > 0
        ):
            self.activation = NPUSwigluWithLimit(config.swiglu_limit)
        elif get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_ascend_ops():
            # DeepEP / Ascend Ops path: the dispatcher hands over tokens grouped
            # by expert, so use a unified kernel that decides quantisation.
            is_quant_kernel = isinstance(
                kernel, (NPUW4A8Int8MoEMethod, NPUW8A8Int8MoEMethod)
            )
            if config.activation == "situ":
                beta = config.gemm1_alpha if config.gemm1_alpha is not None else 4.0
                if (
                    isinstance(kernel, NPUW4A8MXFP4MoEMethod)
                    and envs.SGLANG_NPU_MOE_SITU_MXFP8_FUSED.get()
                ):
                    if config.gemm1_clamp_limit is None:
                        raise ValueError(
                            "fused SiTU MXFP8 quantization requires gemm1_clamp_limit"
                        )
                    self.activation = NPUSituMXFP8Quant(
                        beta=beta,
                        linear_beta=config.gemm1_clamp_limit,
                    )
                else:
                    self.activation = NPUSitu(
                        need_quant=is_quant_kernel,
                        beta=beta,
                        linear_beta=config.gemm1_clamp_limit,
                    )
            else:
                self.activation = NPUSwigluDeepEPKernel(
                    need_quant=is_quant_kernel,
                    alpha=config.gemm1_alpha,
                    limit=config.gemm1_clamp_limit,
                )
        else:
            # Non-DeepEP, no-A2A routing path
            # 1. Choose the base activation according to the quant method
            if isinstance(kernel, (NPUW4A8Int8MoEMethod, NPUW8A8Int8MoEMethod)):
                inner = NPUSwigluQuant()
            elif config.activation == "situ":
                # Grouped SiTU (Kimi-K3). need_quant=False: the MXFP4 / BF16
                # gmm2 requantizes the activations itself, so no quant is
                # fused here. Matches the DeepEP branch below.
                inner = NPUSitu(
                    need_quant=False,
                    beta=config.gemm1_alpha if config.gemm1_alpha is not None else 4.0,
                    linear_beta=config.gemm1_clamp_limit,
                )
            else:
                if config.activation == "npu_swiglu_oai":
                    # NPUSwigluOAI requires the runner config to pass
                    # gemm1_alpha and gemm1_clamp_limit to the triton kernel.
                    inner = NPUSwigluOAI(moe_runner_config=config)
                elif config.activation == "silu":
                    if config.gemm1_clamp_limit is not None:
                        inner = NPUSwigluStepAndMul(
                            clamp_limit=config.gemm1_clamp_limit
                        )
                    else:
                        inner = NPUSwiglu()
                else:
                    inner = NPUGeluAndMul()

            # 2. If the quant method (GGUF) needs TP all‑gather, wrap the activation
            if getattr(config, "use_tp_all_gather_activation", False):
                self.activation = AllGatherActivationWrapper(inner, dim=-1)
            else:
                self.activation = inner

    def run(
        self,
        runner_input: AscendRunnerInput,
        quant_info: AscendQuantInfo,
        running_state: dict,
        hooks: Optional[Any] = None,
    ) -> AscendRunnerOutput:
        """
        Execute the MoE layer using NPU‑specific grouped matmul ops.
        """
        x = runner_input.hidden_states
        original_dtype = torch.float16 if x.dtype == torch.float16 else torch.bfloat16
        expert_tokens = runner_input.expert_tokens
        group_list_type = runner_input.group_list_type

        w13_kernel = self.config.layer.w13_kernel

        if _uses_fused_gmm1(w13_kernel, self.config):
            # --- w13 projection + activation, fused into one kernel ---
            # The fused gmm1 returns activations already requantised for gmm2,
            # so there is no separate activation step to run.
            hidden_states, pertoken_scale = w13_kernel.apply_fused_gmm1_swiglu(
                quant_info,
                x,
                expert_tokens,
                pertoken_scale=runner_input.hidden_states_scale,
                group_list_type=group_list_type,
            )
        else:
            # --- w13 (gate & up) projection ---
            hidden_states = w13_kernel.apply(
                quant_info,
                x,
                expert_tokens,
                pertoken_scale=runner_input.hidden_states_scale,
                output_dtype=original_dtype,
                weight_prefix="w13",
                group_list_type=group_list_type,
            )

            # --- Activation ---
            # Grouped-row activations require dispatch metadata.
            if isinstance(
                self.activation,
                (
                    NPUSwigluDeepEPKernel,
                    NPUSitu,
                    NPUSituMXFP8Quant,
                    NPUSwigluMxfp8Quant,
                ),
            ):
                hidden_states, pertoken_scale = self.activation._apply_activation(
                    hidden_states,
                    group_list=expert_tokens,
                    group_list_type=group_list_type,
                )
            else:
                hidden_states, pertoken_scale = self.activation._apply_activation(
                    hidden_states
                )

        # --- w2 (down) projection ---
        hidden_states = self.config.layer.w2_kernel.apply(
            quant_info,
            hidden_states,
            expert_tokens,
            pertoken_scale=pertoken_scale,
            output_dtype=original_dtype,
            weight_prefix="w2",
            group_list_type=group_list_type,
        )
        return AscendRunnerOutput(hidden_states=hidden_states)


# ---------------------------------------------------------------------------
# QuantInfo
# ---------------------------------------------------------------------------
@dataclass
class AscendQuantInfo(MoeQuantInfo):
    """Quantization payload for Ascend."""

    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    w13_weight_scale: Optional[torch.Tensor] = None
    w2_weight_scale: Optional[torch.Tensor] = None
    w13_weight_offset: Optional[torch.Tensor] = None
    w2_weight_offset: Optional[torch.Tensor] = None
    w13_weight_bias: Optional[torch.Tensor] = None
    w2_weight_bias: Optional[torch.Tensor] = None
    w13_scale_bias: Optional[torch.Tensor] = None
    w2_scale_bias: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Pre/Post permute hooks
# ---------------------------------------------------------------------------


def _routing_layer(runner_config: MoeRunnerConfig):
    """The layer the Ascend routing state lives on.

    ``AscendRunnerCore`` reads ``config.layer`` for its kernels, so the permute
    hooks can rely on it too; fail loudly instead of silently routing nothing.
    """
    layer = getattr(runner_config, "layer", None)
    if layer is None:
        raise RuntimeError("Ascend MoE routing requires runner_config.layer to be set.")
    return layer


@register_pre_permute("standard", "ascend")
def pre_permute_standard_to_ascend(
    dispatch_output: StandardDispatchOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AscendRunnerInput:
    """Sort tokens by expert and (optionally) quantize them in the same op."""
    # Imported here, not at module scope: ``topk`` pulls in eplb/observability/
    # disaggregation, which imports ``configs.model_config`` while it is still
    # initializing (model_config -> quantization -> ... -> this module).
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    topk_output = dispatch_output.topk_output
    if not TopKOutputChecker.format_is_standard(topk_output):
        raise NotImplementedError(
            "Ascend MoE routing requires a standard top-k output, got "
            f"{type(topk_output).__name__}."
        )
    if dispatch_output.hidden_states_pre_quant is not None:
        raise NotImplementedError(
            "Ascend MoE routing quantizes inside npu_moe_init_routing_v2 and "
            "cannot consume a pre-quantized routed input."
        )

    layer = _routing_layer(runner_config)
    hidden_states = dispatch_output.hidden_states
    topk_weights = topk_output.topk_weights.to(hidden_states.dtype)
    topk_ids = topk_output.topk_ids.to(torch.int32)

    quant_mode, tp_all_gather, group_list_type = ascend_dispatch.get_routing(layer)

    routed = npu_moe_init_routing(
        hidden_states,
        topk_ids,
        runner_config.num_experts,
        quant_mode=quant_mode,
        active_num=topk_ids.numel(),
    )

    running_state["ascend_routing"] = {
        "expanded_row_idx": routed.expanded_row_idx,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "tp_all_gather": tp_all_gather,
    }

    return AscendRunnerInput(
        hidden_states=routed.hidden_states,
        hidden_states_scale=routed.scale,
        expert_tokens=routed.expert_tokens,
        group_list_type=group_list_type,
    )


@register_pre_permute("ascend_ops", "ascend")
def pre_permute_ascend_ops_to_ascend(
    dispatch_output: AscendOpsDispatchOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AscendRunnerInput:
    """Regroup the communicated rows into local-expert order."""
    mode = dispatch_output.exec_mode
    if mode == "alltoallv":
        return _pre_permute_alltoallv(dispatch_output, runner_config, running_state)
    if mode == "allgather":
        return _pre_permute_allgather(dispatch_output, runner_config, running_state)
    if mode == "mc2":
        # Decode MC2 dispatch already groups rows per expert.
        if dispatch_output.forward_state is None:
            raise RuntimeError("MC2 dispatch output is missing forward state")
        running_state["ascend_ops_forward_state"] = dispatch_output.forward_state
        return AscendRunnerInput(
            hidden_states=dispatch_output.hidden_states,
            hidden_states_scale=dispatch_output.hidden_states_scale,
            expert_tokens=dispatch_output.expert_tokens.to(torch.int64),
            group_list_type=dispatch_output.group_list_type,
        )
    raise NotImplementedError(f"unknown ascend_ops exec mode {mode!r}")


def _pre_permute_alltoallv(
    dispatch_output: AscendOpsDispatchOutput,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AscendRunnerInput:
    """Regroup the a2a-received rows into local-expert order.

    A second scale-reusing routing pass does the regrouping; with a single
    local expert the wire order already is expert order.
    """
    received = dispatch_output.hidden_states
    received_scale = dispatch_output.hidden_states_scale
    recv_counts = dispatch_output.recv_counts
    if recv_counts is None:
        raise RuntimeError("alltoallv dispatch output is missing recv_counts")
    num_local_experts = runner_config.num_local_experts
    quant_mode = dispatch_output.quant_mode

    local_counts = recv_counts.reshape(-1)
    local_ids = (
        torch.arange(num_local_experts, dtype=torch.int32, device=received.device)
        .repeat(recv_counts.shape[0], 1)
        .reshape(-1)
    )
    local_ids = torch.repeat_interleave(local_ids, local_counts).view(-1, 1)

    if quant_mode not in (-1, INT8_QUANT_MODE) and received_scale is not None:
        received_scale = _reshape_received_scale(received_scale)

    if num_local_experts > 1 and received.shape[0] > 0:
        routed = npu_moe_init_routing(
            received,
            local_ids,
            num_local_experts,
            quant_mode=quant_mode,
            scale=received_scale,
        )
        local_permuted = routed.hidden_states
        local_expanded_row_idx = routed.expanded_row_idx
        local_scale = routed.scale
    else:
        local_permuted = received
        local_expanded_row_idx = torch.arange(
            received.shape[0], dtype=torch.int32, device=received.device
        )
        local_scale = received_scale

    if quant_mode != -1 and num_local_experts > 1:
        local_permuted = local_permuted.view(quant_mode_dtype(quant_mode))

    state = dispatch_output.forward_state
    if state is None:
        raise RuntimeError("alltoallv dispatch output is missing forward state")
    state.local_expanded_row_idx = local_expanded_row_idx
    running_state["ascend_ops_forward_state"] = state
    return AscendRunnerInput(
        hidden_states=local_permuted,
        hidden_states_scale=local_scale,
        expert_tokens=recv_counts.sum(dim=0).to(torch.int64),
        group_list_type=dispatch_output.group_list_type,
    )


def _pre_permute_allgather(
    dispatch_output: AscendOpsDispatchOutput,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AscendRunnerInput:
    """Keep only the local-expert routes and group them by expert.

    Non-local routes get weight 0 and the sentinel expert id
    ``num_local_experts`` — never -1, which corrupts expert counts even at
    weight zero.
    """
    num_local_experts = runner_config.num_local_experts
    topk_ids = dispatch_output.topk_ids
    local_ids = topk_ids - runner_config.moe_ep_rank * num_local_experts
    is_local = (local_ids >= 0) & (local_ids < num_local_experts)
    topk_weights = dispatch_output.topk_weights.masked_fill(~is_local, 0)
    local_ids = local_ids.masked_fill(~is_local, num_local_experts)

    routed = npu_moe_init_routing(
        dispatch_output.hidden_states,
        local_ids,
        num_local_experts,
        quant_mode=dispatch_output.quant_mode,
        scale=dispatch_output.hidden_states_scale,
        active_num=local_ids.numel(),
    )
    # Dropped routes must not read garbage: point their reverse-mapping rows
    # at the last capacity row, which the post-permute zeroes before finalize.
    expanded_row_idx = routed.expanded_row_idx.masked_fill(
        ~is_local.reshape(-1), local_ids.numel() - 1
    )

    state = dispatch_output.forward_state
    if state is None:
        raise RuntimeError("allgather dispatch output is missing forward state")
    state.topk_weights = topk_weights
    state.expanded_row_idx = expanded_row_idx
    state.topk_ids = local_ids
    state.expert_tokens = routed.expert_tokens
    state.num_routes = local_ids.numel()
    running_state["ascend_ops_forward_state"] = state
    return AscendRunnerInput(
        hidden_states=routed.hidden_states,
        hidden_states_scale=routed.scale,
        expert_tokens=routed.expert_tokens,
        group_list_type=dispatch_output.group_list_type,
    )


@register_pre_permute("deepep_normal", "ascend")
def pre_permute_deepep_normal_to_ascend(
    dispatch_output: DeepEPNormalDispatchOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AscendRunnerInput:
    (
        hidden_states,
        hidden_states_scale,
        topk_ids,
        topk_weights,
        num_recv_tokens_per_expert,
    ) = dispatch_output
    group_list = torch.tensor(
        num_recv_tokens_per_expert,
        dtype=torch.int64,
        device=hidden_states.device,
    )
    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights

    return AscendRunnerInput(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        expert_tokens=group_list,
        group_list_type=1,
    )


@register_pre_permute("deepep_ll", "ascend")
def pre_permute_deepep_ll_to_ascend(
    dispatch_output: DeepEPLLDispatchOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AscendRunnerInput:
    (
        hidden_states,
        hidden_states_scale,
        topk_ids,
        topk_weights,
        group_list,
        _,
    ) = dispatch_output
    group_list = group_list.to(torch.int64)
    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights
    return AscendRunnerInput(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        expert_tokens=group_list,
        group_list_type=1,
    )


@register_post_permute("ascend", "standard")
def post_permute_ascend_to_standard(
    runner_output: AscendRunnerOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:
    """Weighted top-k sum of the grouped expert output back into ``[T, H]``."""
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
    )

    state = running_state.pop("ascend_routing")
    final_hidden_states = npu_moe_finalize_routing(
        runner_output.hidden_states,
        state["expanded_row_idx"],
        topk_weights=state["topk_weights"],
        topk_ids=state["topk_ids"],
    )
    if state["tp_all_gather"] and get_parallel().tp_size > 1:
        # GGUF keeps full, non-intermediate-sharded expert weights, so the
        # hidden dim must be gathered back after finalize.
        final_hidden_states = tensor_model_parallel_all_gather(
            final_hidden_states, dim=-1
        )

    return StandardCombineInput(hidden_states=final_hidden_states)


@register_post_permute("ascend", "ascend_ops")
def post_permute_ascend_to_ascend_ops(
    runner_output: AscendRunnerOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AscendOpsCombineInput:
    """Undo the pre-permute's regrouping before the dispatcher's combine."""
    from sglang.srt.layers.moe.token_dispatcher.ascend_ops import (
        AscendOpsCombineInput,
        _AllGatherState,
        _AllToAllVState,
    )

    state = running_state.pop("ascend_ops_forward_state", None)
    if state is None:
        raise RuntimeError("Ascend Ops dispatch output is missing forward state")
    if isinstance(state, _AllToAllVState):
        local_expanded_row_idx = state.local_expanded_row_idx
        if local_expanded_row_idx is None:
            raise RuntimeError("alltoallv forward state is missing local row map")
        hidden_states = runner_output.hidden_states
        if runner_config.num_local_experts > 1 and hidden_states.shape[0] > 0:
            hidden_states = npu_moe_finalize_routing(
                hidden_states, local_expanded_row_idx
            )
    elif isinstance(state, _AllGatherState):
        hidden_states = _allgather_weighted_sum(runner_output, runner_config, state)
    else:
        # MC2 / decode: the combine op applies the top-k weights itself.
        hidden_states = runner_output.hidden_states
    return AscendOpsCombineInput(
        hidden_states=hidden_states,
        forward_state=state,
    )


def _allgather_weighted_sum(
    runner_output: AscendRunnerOutput,
    runner_config: MoeRunnerConfig,
    state,
) -> torch.Tensor:
    """Weighted top-k sum over the full gathered route capacity.

    Finalize mode 2 requires the full ``[global_tokens * top_k, hidden]``
    shape; unused capacity rows are zeroed so dropped routes never read NaN.
    """
    if (
        state.num_routes is None
        or state.expert_tokens is None
        or state.expanded_row_idx is None
        or state.topk_weights is None
        or state.topk_ids is None
    ):
        raise RuntimeError("allgather forward state is incomplete")
    hidden_states = runner_output.hidden_states
    expert_output = torch.zeros(
        (state.num_routes, runner_config.hidden_size),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    expert_output[: hidden_states.shape[0]].copy_(hidden_states)
    unused_rows = (
        torch.arange(expert_output.shape[0], device=expert_output.device)
        >= state.expert_tokens.sum()
    )
    expert_output.masked_fill_(unused_rows[:, None], 0)
    # Keep weighted expert sums in FP32 across ranks: rounding each rank's
    # partial output to BF16 before the ReduceScatter loses precision.
    return npu_moe_finalize_routing(
        expert_output,
        state.expanded_row_idx,
        topk_weights=state.topk_weights,
        topk_ids=state.topk_ids,
    )


@register_post_permute("ascend", "deepep_normal")
def post_permute_ascend_to_deepep_normal(
    runner_output: AscendRunnerOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepEPNormalCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPNormalCombineInput

    return DeepEPNormalCombineInput(
        hidden_states=runner_output.hidden_states,
        topk_ids=running_state["topk_ids"],
        topk_weights=running_state["topk_weights"],
    )


@register_post_permute("ascend", "deepep_ll")
def post_permute_ascend_to_deepep_ll(
    runner_output: AscendRunnerOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepEPLLCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLCombineInput

    return DeepEPLLCombineInput(
        hidden_states=runner_output.hidden_states,
        topk_ids=running_state["topk_ids"],
        topk_weights=running_state["topk_weights"],
    )
