"""Composite SharedEP decode backend with a DeepEP/DeepGEMM prefill fallback."""

from __future__ import annotations

from typing import NamedTuple

import torch
import triton.language as tl

from sglang.kernels.ops.attention.dsv4 import silu_and_mul_contig_post_quant
from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
    invoke_fused_moe_kernel,
)
from sglang.srt.layers.dp_attention import (
    get_dp_global_num_tokens,
    get_is_extend_in_batch,
)
from sglang.srt.layers.moe.moe_runner.base import (
    MoeRunnerConfig,
    PermuteMethodPool,
    register_fused_func,
)
from sglang.srt.layers.moe.moe_runner.deep_gemm import (
    DeepGemmMoeQuantInfo,
    DeepGemmRunnerCore,
)
from sglang.srt.layers.moe.shared_ep.kernels import (
    prepare_routes,
    quantize_pack_input,
)
from sglang.srt.layers.moe.shared_ep.layout import SharedEpLayout
from sglang.srt.layers.moe.shared_ep.profiles import (
    RELEASE_MAX_TOKENS_PER_RANK,
    SharedEpProfile,
    select_profile,
)
from sglang.srt.layers.moe.shared_ep.state import (
    SharedEpState,
    create_shared_ep_state,
)
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    CombineInputChecker,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPDispatcher
from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
from sglang.srt.layers.moe.topk import TopKOutput, TopKOutputChecker
from sglang.srt.layers.moe.utils import get_deepep_mode, get_moe_runner_backend
from sglang.srt.runtime_context import get_parallel, get_resources


class SharedEpDispatchOutput(NamedTuple):
    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor
    topk_output: TopKOutput
    state: SharedEpState
    profile: SharedEpProfile
    num_tokens: int
    local_expert_start: int

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.STANDARD


def compact_intermediate_capacity(
    *,
    num_tokens: int,
    world_size: int,
    top_k: int,
    num_local_experts: int,
    block_size: int,
) -> int:
    """Graph-static upper bound for this rank's padded local-expert routes."""

    valid_routes = num_tokens * world_size * top_k
    experts_with_routes = min(valid_routes, num_local_experts)
    return valid_routes + experts_with_routes * (block_size - 1)


def decode_intermediate_capacity(profile: SharedEpProfile) -> int:
    """Worst-case EP-local route capacity for uneven owner batches."""

    return compact_intermediate_capacity(
        num_tokens=profile.max_tokens_per_rank,
        world_size=profile.ep_size,
        top_k=profile.top_k,
        num_local_experts=profile.num_local_experts,
        block_size=profile.block_size_m,
    )


def _validate_decode_capacity(profile: SharedEpProfile) -> None:
    global_num_tokens = get_dp_global_num_tokens()
    if global_num_tokens is None or len(global_num_tokens) != profile.ep_size:
        raise RuntimeError(
            "SharedEP requires the complete DP-Attention token-count vector "
            "before decode publication"
        )
    largest = max(global_num_tokens)
    if largest > profile.max_tokens_per_rank:
        failed_rank = global_num_tokens.index(largest)
        raise ValueError(
            "SharedEP decode capacity exceeded: "
            f"rank {failed_rank} has {largest} local tokens, "
            f"capacity is {profile.max_tokens_per_rank}"
        )


def _get_shared_state(
    config: MoeRunnerConfig,
    profile: SharedEpProfile,
) -> SharedEpState:
    """Return the process-lifetime VMM state shared by all MoE layers."""

    resources = get_resources().buffers
    key = f"shared_ep_{profile.name}_ep{profile.ep_size}"
    state = resources.get(key)
    if state is None:
        ep_group = get_parallel().moe_ep_group
        layout = SharedEpLayout.build(
            hidden_size=profile.hidden_size,
            top_k=profile.top_k,
            max_tokens_per_rank=profile.max_tokens_per_rank,
        )
        state = create_shared_ep_state(
            layout=layout,
            cpu_group=ep_group.cpu_group,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        resources[key] = state
    expected = (
        config.hidden_size,
        config.top_k,
        profile.max_tokens_per_rank,
    )
    actual = (
        state.layout.hidden_size,
        state.layout.top_k,
        state.layout.max_tokens_per_rank,
    )
    if actual != expected:
        raise RuntimeError(
            f"SharedEP process state does not match the layer: {actual=} {expected=}"
        )
    return state


class SharedEpDispatcher(BaseDispatcher):
    def __init__(
        self,
        config: MoeRunnerConfig,
        fallback_dispatcher: BaseDispatcher | None = None,
    ):
        super().__init__()
        parallel = get_parallel()
        if parallel.moe_ep_size != 8:
            raise ValueError(f"SharedEP requires EP8, got EP{parallel.moe_ep_size}")
        capability = torch.cuda.get_device_capability()
        self.profile = select_profile(
            config,
            capability=capability,
            ep_size=parallel.moe_ep_size,
            block_shape=(128, 128),
            max_tokens_per_rank=RELEASE_MAX_TOKENS_PER_RANK,
        )
        self.config = config
        self.state = _get_shared_state(self.config, self.profile)
        self.local_expert_start = parallel.moe_ep_rank * self.profile.num_local_experts
        self.fallback_dispatcher = fallback_dispatcher

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        if get_is_extend_in_batch():
            if self.fallback_dispatcher is None:
                raise RuntimeError("SharedEP requires a DeepEP prefill fallback")
            return self.fallback_dispatcher.dispatch(
                hidden_states=hidden_states,
                topk_output=topk_output,
            )
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise TypeError("SharedEP decode requires standard Top-K output")

        state = self.state
        num_tokens = hidden_states.shape[0]
        _validate_decode_capacity(self.profile)
        if num_tokens > self.profile.max_tokens_per_rank:
            raise ValueError(
                "SharedEP decode capacity exceeded: "
                f"{num_tokens} > {self.profile.max_tokens_per_rank} local tokens"
            )
        quantize_pack_input(
            state.local_input,
            source=hidden_states,
            source_ids=topk_output.topk_ids,
            source_weights=topk_output.topk_weights,
            group_size=self.profile.block_shape[1],
        )
        state.input_epoch.publish()
        return SharedEpDispatchOutput(
            hidden_states=state.global_input.activations,
            hidden_states_scale=state.global_input.scales,
            topk_output=topk_output,
            state=state,
            profile=self.profile,
            num_tokens=num_tokens,
            local_expert_start=self.local_expert_start,
        )

    def combine(self, combine_input: CombineInput) -> torch.Tensor:
        if CombineInputChecker.format_is_standard(combine_input):
            return combine_input.hidden_states
        if self.fallback_dispatcher is None:
            raise RuntimeError("SharedEP received a fallback output without fallback")
        return self.fallback_dispatcher.combine(combine_input=combine_input)

    def set_quant_config(self, quant_config: dict) -> None:
        super().set_quant_config(quant_config)
        if self.fallback_dispatcher is not None:
            self.fallback_dispatcher.set_quant_config(quant_config)


def create_shared_ep_dispatcher(
    config: MoeRunnerConfig,
    *,
    group,
) -> SharedEpDispatcher:
    """Build the decode backend and its non-overlapped DeepEP prefill fallback."""

    if not get_moe_runner_backend().is_deep_gemm():
        raise ValueError(
            "shared_ep requires --moe-runner-backend deep_gemm "
            "for its prefill fallback"
        )
    fallback_dispatcher = DeepEPDispatcher(
        group=group,
        router_topk=config.top_k,
        permute_fusion=True,
        num_experts=config.num_experts,
        num_local_experts=config.num_local_experts,
        hidden_size=config.hidden_size,
        params_dtype=config.params_dtype,
        deepep_mode=get_deepep_mode(),
        async_finish=True,
        return_recv_hook=True,
    )
    return SharedEpDispatcher(
        config,
        fallback_dispatcher=fallback_dispatcher,
    )


def _run_deep_gemm_fallback(
    dispatch_output,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
):
    running_state = {}
    dispatch_format = dispatch_output.format.value
    pre_permute = PermuteMethodPool.get_pre_permute(
        dispatch_format,
        "deep_gemm",
    )
    runner_input = pre_permute(
        dispatch_output,
        quant_info,
        runner_config,
        running_state,
    )
    runner_output = DeepGemmRunnerCore(runner_config).run(
        runner_input,
        quant_info,
        running_state,
    )
    post_permute = PermuteMethodPool.get_post_permute(
        "deep_gemm",
        dispatch_format,
    )
    return post_permute(
        runner_output,
        quant_info,
        runner_config,
        running_state,
    )


def _validate_decode_weights(
    dispatch_output: SharedEpDispatchOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> None:
    profile = dispatch_output.profile
    if not isinstance(quant_info, DeepGemmMoeQuantInfo):
        raise TypeError("SharedEP requires DeepGemm FP8 weight metadata")
    if not quant_info.use_fp8 or quant_info.is_fp4_experts or quant_info.use_mxfp8:
        raise ValueError("SharedEP release supports block-FP8 experts only")
    if tuple(quant_info.block_shape or ()) != profile.block_shape:
        raise ValueError(
            f"SharedEP requires block shape {profile.block_shape}, "
            f"got {quant_info.block_shape}"
        )
    expected_w13 = (
        profile.num_local_experts,
        profile.intermediate_size * 2,
        profile.hidden_size,
    )
    expected_w2 = (
        profile.num_local_experts,
        profile.hidden_size,
        profile.intermediate_size,
    )
    if tuple(quant_info.w13_weight.shape) != expected_w13:
        raise ValueError(
            f"SharedEP W13 shape is {tuple(quant_info.w13_weight.shape)}, "
            f"expected {expected_w13}"
        )
    if tuple(quant_info.w2_weight.shape) != expected_w2:
        raise ValueError(
            f"SharedEP W2 shape is {tuple(quant_info.w2_weight.shape)}, "
            f"expected {expected_w2}"
        )
    if quant_info.w13_scale is None or quant_info.w2_scale is None:
        raise ValueError("SharedEP FP8 weights require W13 and W2 scales")
    if runner_config.activation != "silu" or not runner_config.is_gated:
        raise ValueError("SharedEP release supports gated SiLU experts only")
    if runner_config.gemm1_alpha is not None:
        raise ValueError("SharedEP release does not support gemm1_alpha")
    if runner_config.gemm1_clamp_limit is not None:
        raise ValueError("SharedEP release does not support gemm1_clamp_limit")


@register_fused_func("shared_ep", "deep_gemm")
def run_shared_ep(
    dispatch_output,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    if not isinstance(dispatch_output, SharedEpDispatchOutput):
        return _run_deep_gemm_fallback(
            dispatch_output,
            quant_info,
            runner_config,
        )
    _validate_decode_weights(dispatch_output, quant_info, runner_config)

    profile = dispatch_output.profile
    state = dispatch_output.state
    routes = prepare_routes(
        profile,
        state.global_input.topk_ids,
        state.global_input.topk_weights,
        state.input_epoch.allocation.local_storage,
        state.input_epoch.epoch,
        local_expert_start=dispatch_output.local_expert_start,
    )
    route_ids = routes.local_ids.view(-1, profile.top_k)
    route_weights = routes.local_weights.view(-1, profile.top_k)
    capacity = decode_intermediate_capacity(profile)
    gate_up = torch.empty(
        (capacity, profile.intermediate_size * 2),
        dtype=torch.bfloat16,
        device=dispatch_output.hidden_states.device,
    )
    shared_activations = dispatch_output.hidden_states.view(
        -1,
        profile.hidden_size,
    )
    shared_scales = dispatch_output.hidden_states_scale.view(
        -1,
        profile.hidden_size // profile.block_shape[1],
    )
    invoke_fused_moe_kernel(
        A=shared_activations,
        B=quant_info.w13_weight,
        bias=None,
        C=gate_up,
        A_scale=shared_scales,
        B_scale=quant_info.w13_scale,
        B_zp=None,
        topk_weights=route_weights,
        topk_ids=route_ids,
        sorted_token_ids=routes.sorted_token_ids,
        expert_ids=routes.expert_ids,
        num_tokens_post_padded=routes.num_tokens_post_padded,
        mul_routed_weight=False,
        top_k=profile.top_k,
        config=profile.w13_kernel_config(dispatch_output.num_tokens),
        compute_type=tl.bfloat16,
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=list(profile.block_shape),
        filter_expert=True,
        c_sorted=True,
        a_is_prequantized=True,
    )

    down_fp8 = torch.empty(
        (capacity, profile.intermediate_size),
        dtype=torch.float8_e4m3fn,
        device=gate_up.device,
    )
    down_scale = torch.empty(
        (capacity, profile.intermediate_size // profile.block_shape[1]),
        dtype=torch.float32,
        device=gate_up.device,
    )
    silu_and_mul_contig_post_quant(
        input=gate_up,
        output=down_fp8,
        output_scale=down_scale,
        quant_group_size=profile.block_shape[1],
        swiglu_limit=runner_config.swiglu_limit,
        valid_rows=routes.num_tokens_post_padded,
    )
    shared_output = state.global_output.view(-1, profile.hidden_size)
    invoke_fused_moe_kernel(
        A=down_fp8,
        B=quant_info.w2_weight,
        bias=None,
        C=shared_output,
        A_scale=down_scale,
        B_scale=quant_info.w2_scale,
        B_zp=None,
        topk_weights=route_weights,
        topk_ids=route_ids,
        sorted_token_ids=routes.sorted_token_ids,
        expert_ids=routes.expert_ids,
        num_tokens_post_padded=routes.num_tokens_post_padded,
        mul_routed_weight=True,
        top_k=1,
        config=profile.w2_kernel_config(dispatch_output.num_tokens),
        compute_type=tl.bfloat16,
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=list(profile.block_shape),
        filter_expert=True,
        a_use_tma=True,
        b_use_tma=True,
        a_is_prequantized=True,
    )
    state.output_epoch.publish()
    state.output_epoch.wait_all()

    # The next layer cannot reuse this output until every owner has reduced it:
    # each owner publishes the next input only after this sum, and the next
    # expert phase waits for all input publications.
    output = state.local_output[: dispatch_output.num_tokens].sum(dim=1)
    return StandardCombineInput(hidden_states=output)
