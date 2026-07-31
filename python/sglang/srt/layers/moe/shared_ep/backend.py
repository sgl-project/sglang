"""Shared-object EP backend with direct decode and pull-cache prefill consumers."""

from __future__ import annotations

from typing import Literal, NamedTuple

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
    register_fused_func,
)
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.moe.shared_ep.kernels import (
    prepare_routes,
    quantize_pack_input,
)
from sglang.srt.layers.moe.shared_ep.layout import SharedEpLayout
from sglang.srt.layers.moe.shared_ep.profiles import (
    RELEASE_MAX_PREFILL_TOKENS_PER_RANK,
    RELEASE_MAX_TOKENS_PER_RANK,
    SharedEpProfile,
    make_pull_cache_prefill_profile,
    resolve_prefill_capacity,
    select_profile,
)
from sglang.srt.layers.moe.shared_ep.pull_cache_prefill import (
    PullCache,
    allocate_pull_cache,
    invoke_pull_cache_w13,
    make_pull_cache_prefill_plan,
    pull_cache_rows,
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
from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
from sglang.srt.layers.moe.topk import TopKOutput, TopKOutputChecker
from sglang.srt.runtime_context import get_parallel, get_resources, get_schedule


class SharedEpDispatchOutput(NamedTuple):
    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor
    topk_output: TopKOutput
    state: SharedEpState
    profile: SharedEpProfile
    num_tokens: int
    local_expert_start: int
    phase: Literal["decode", "prefill"]
    pull_cache: PullCache | None

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


def intermediate_capacity(profile: SharedEpProfile) -> int:
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
    key = (
        f"shared_ep_{profile.name}_ep{profile.ep_size}"
        f"_capacity{profile.max_tokens_per_rank}"
    )
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


def _get_shared_pull_cache(profile: SharedEpProfile) -> PullCache:
    resources = get_resources().buffers
    key = (
        f"shared_ep_pull_cache_{profile.name}_ep{profile.ep_size}"
        f"_capacity{profile.max_tokens_per_rank}"
    )
    cache = resources.get(key)
    if cache is None:
        plan = make_pull_cache_prefill_plan(
            owners=profile.ep_size,
            source_tokens_per_owner=profile.max_tokens_per_rank,
            hidden_size=profile.hidden_size,
            top_k=profile.top_k,
            num_local_experts=profile.num_local_experts,
            expert_alignment=profile.block_size_m,
        )
        cache = allocate_pull_cache(
            plan,
            active_rows=plan.cache_rows,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        resources[key] = cache
    return cache


def select_shared_ep_phase(
    num_tokens: int,
    *,
    is_prefill: bool,
    prefill_capacity: int,
    supports_pull_cache_prefill: bool,
) -> Literal["decode", "prefill"]:
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    if prefill_capacity <= 0:
        raise ValueError(f"prefill_capacity must be positive, got {prefill_capacity}")
    if not is_prefill:
        return "decode"
    if not supports_pull_cache_prefill:
        raise ValueError("SharedEP profile does not support prefill")
    if num_tokens <= prefill_capacity:
        return "prefill"
    raise ValueError(
        "SharedEP prefill capacity exceeded: "
        f"{num_tokens} > {prefill_capacity} local tokens per rank"
    )


class SharedEpDispatcher(BaseDispatcher):
    def __init__(
        self,
        config: MoeRunnerConfig,
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
        if self.profile.supports_pull_cache_prefill:
            prefill_capacity = resolve_prefill_capacity(
                get_schedule().chunked_prefill_size,
                release_max_tokens=RELEASE_MAX_PREFILL_TOKENS_PER_RANK,
            )
            self.prefill_profile = make_pull_cache_prefill_profile(
                self.profile,
                prefill_capacity,
            )
            if self.prefill_profile is None:
                raise RuntimeError("SharedEP pull-cache prefill profile is missing")
            self.prefill_state = _get_shared_state(self.config, self.prefill_profile)
            self.prefill_cache = _get_shared_pull_cache(self.prefill_profile)
        else:
            self.prefill_profile = self.profile
            self.prefill_state = None
            self.prefill_cache = None
        self.local_expert_start = parallel.moe_ep_rank * self.profile.num_local_experts

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        is_prefill = get_is_extend_in_batch()
        global_num_tokens = get_dp_global_num_tokens()
        if global_num_tokens is None or len(global_num_tokens) != self.profile.ep_size:
            raise RuntimeError(
                "SharedEP requires the complete DP-Attention token-count vector "
                "before publication"
            )
        phase = select_shared_ep_phase(
            max(global_num_tokens),
            is_prefill=is_prefill,
            prefill_capacity=self.prefill_profile.max_tokens_per_rank,
            supports_pull_cache_prefill=self.profile.supports_pull_cache_prefill,
        )
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise TypeError("SharedEP requires standard Top-K output")

        if phase == "prefill":
            if self.prefill_state is None or self.prefill_cache is None:
                raise RuntimeError("SharedEP prefill consumer was not admitted")
            profile = self.prefill_profile
            state = self.prefill_state
            pull_cache = self.prefill_cache
        else:
            profile = self.profile
            state = self.state
            pull_cache = None
        num_tokens = hidden_states.shape[0]
        if phase == "decode":
            _validate_decode_capacity(profile)
        if num_tokens > profile.max_tokens_per_rank:
            raise ValueError(
                f"SharedEP {phase} capacity exceeded: "
                f"{num_tokens} > {profile.max_tokens_per_rank} local tokens"
            )
        quantize_pack_input(
            state.local_input,
            source=hidden_states,
            source_ids=topk_output.topk_ids,
            source_weights=topk_output.topk_weights,
            group_size=profile.block_shape[1],
        )
        state.input_epoch.publish()
        return SharedEpDispatchOutput(
            hidden_states=state.global_input.activations,
            hidden_states_scale=state.global_input.scales,
            topk_output=topk_output,
            state=state,
            profile=profile,
            num_tokens=num_tokens,
            local_expert_start=self.local_expert_start,
            phase=phase,
            pull_cache=pull_cache,
        )

    def combine(self, combine_input: CombineInput) -> torch.Tensor:
        if CombineInputChecker.format_is_standard(combine_input):
            return combine_input.hidden_states
        raise TypeError("SharedEP combine requires StandardCombineInput")


def create_shared_ep_dispatcher(
    config: MoeRunnerConfig,
) -> SharedEpDispatcher:
    """Build the standalone SharedEP dispatcher."""

    return SharedEpDispatcher(config)


def _validate_weights(
    dispatch_output: SharedEpDispatchOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> None:
    profile = dispatch_output.profile
    if not isinstance(quant_info, TritonMoeQuantInfo):
        raise TypeError("SharedEP requires Triton FP8 weight metadata")
    if (
        not quant_info.use_fp8_w8a8
        or quant_info.use_mxfp8
        or quant_info.use_int8_w8a8
        or quant_info.use_int8_w8a16
        or quant_info.use_int4_w4a16
    ):
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


@register_fused_func("shared_ep", "triton")
def run_shared_ep(
    dispatch_output,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    if not isinstance(dispatch_output, SharedEpDispatchOutput):
        raise TypeError(
            "SharedEP fused execution requires SharedEpDispatchOutput, "
            f"got {type(dispatch_output).__name__}"
        )
    _validate_weights(dispatch_output, quant_info, runner_config)

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
    capacity = intermediate_capacity(profile)
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
    if dispatch_output.phase == "prefill":
        pull_cache = dispatch_output.pull_cache
        if pull_cache is None or pull_cache.active_rows != capacity:
            raise RuntimeError("SharedEP prefill pull-cache capacity mismatch")
        pull_cache_rows(
            source_activations=shared_activations,
            source_scales=shared_scales,
            sorted_token_ids=routes.sorted_token_ids,
            num_tokens_post_padded=routes.num_tokens_post_padded,
            cache=pull_cache,
            top_k=profile.top_k,
            source_route_capacity=shared_activations.shape[0] * profile.top_k,
        )
        invoke_pull_cache_w13(
            cache=pull_cache,
            weight=quant_info.w13_weight,
            weight_scale=quant_info.w13_scale,
            output=gate_up,
            expert_ids=routes.expert_ids,
            num_tokens_post_padded=routes.num_tokens_post_padded,
            config=profile.w13_kernel_config(dispatch_output.num_tokens),
            block_shape=profile.block_shape,
        )
    else:
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
