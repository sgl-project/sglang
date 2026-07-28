"""Composite SharedEP decode backend with platform-native prefill fallback."""

from __future__ import annotations

import os
from typing import Any, NamedTuple

import torch
import torch.distributed as dist

from sglang.srt.layers.dp_attention import (
    get_dp_global_num_tokens,
)
from sglang.srt.layers.moe.moe_runner.base import (
    MoeRunnerConfig,
    register_fused_func,
)
from sglang.srt.layers.moe.moe_runner.shared_ep import (
    SharedEpQuantCapability,
    SharedEpQuantInfo,
    SharedEpQuantization,
    SharedEpScaleLayout,
    SharedEpWeightLayout,
)
from sglang.srt.layers.moe.shared_ep.lanes import (
    SharedEpLaneProtocol,
    compute_shared_ep_lane_protocol,
    shared_ep_state_resource_key,
    validate_shared_ep_model_namespace,
)
from sglang.srt.layers.moe.shared_ep.layout import (
    SharedEpInputFormat,
    SharedEpLayout,
)
from sglang.srt.layers.moe.shared_ep.profiles import (
    RELEASE_MAX_TOKENS_PER_RANK,
    SharedEpProfile,
    select_profile,
)
from sglang.srt.layers.moe.shared_ep.runtime import (
    SharedEpRuntimeHooks,
    get_shared_ep_runtime_hooks,
)
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    CombineInputChecker,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
from sglang.srt.layers.moe.topk import TopKOutput, TopKOutputChecker
from sglang.srt.layers.moe.utils import (
    DeepEPMode,
    get_moe_runner_backend,
    get_shared_ep_prefill_backend,
)
from sglang.srt.runtime_context import (
    get_forward,
    get_parallel,
    get_resources,
    get_server_args,
)
from sglang.srt.utils import is_cuda, is_hip


class SharedEpDispatchOutput(NamedTuple):
    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor | None
    topk_output: TopKOutput
    state: Any
    profile: SharedEpProfile
    num_tokens: int
    local_expert_start: int

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.STANDARD

    @property
    def is_shared_ep_decode(self) -> bool:
        return True


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
    if global_num_tokens is None:
        global_num_tokens = get_forward().shared_ep_global_num_tokens
    if global_num_tokens is None:
        # Eager DP-attention does not populate the gathered-buffer metadata
        # when SharedEP bypasses that buffer. Admission caps the global request
        # count at EP * max_tokens_per_rank and every owner validates its local
        # row count below, so the fixed-capacity route scan remains safe.
        return
    global_num_tokens = tuple(int(value) for value in global_num_tokens)
    if not global_num_tokens:
        return
    largest = max(global_num_tokens)
    if largest > profile.max_tokens_per_rank:
        failed_rank = global_num_tokens.index(largest)
        raise ValueError(
            "SharedEP decode capacity exceeded: "
            f"rank {failed_rank} has {largest} local tokens, "
            f"capacity is {profile.max_tokens_per_rank}"
        )


def _get_device_profile_capability() -> tuple[tuple[int, int], str]:
    """Validate the release device and map it to the registered profile key."""

    if is_hip():
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        arch = getattr(properties, "gcnArchName", "").split(":", 1)[0]
        if arch != "gfx950":
            raise ValueError(
                f"ROCm SharedEP initial release requires gfx950, got {arch or 'unknown'}"
            )
        return (9, 5), "rocm:gfx950"

    if is_cuda():
        capability = torch.cuda.get_device_capability()
        if capability != (9, 0):
            raise ValueError(
                "CUDA SharedEP initial release requires SM90, got "
                f"SM{capability[0]}{capability[1]}"
            )
        return capability, "cuda:sm90"

    raise ValueError("SharedEP requires NVIDIA CUDA or AMD ROCm")


def _profile_quantization(
    config: MoeRunnerConfig,
) -> tuple[SharedEpQuantization, tuple[int, int]]:
    """Select only the exact DSV4-Pro shape into the MXFP4 profile."""

    dsv4_pro = (
        config.hidden_size,
        config.intermediate_size_per_partition,
        config.top_k,
        config.num_experts,
        config.num_local_experts,
    ) == (7168, 3072, 6, 384, 48)
    if dsv4_pro:
        return SharedEpQuantization.MXFP4, (1, 32)
    return SharedEpQuantization.BLOCK_FP8, (128, 128)


def _synchronize_admission_stage(
    cpu_group,
    *,
    stage: str,
    descriptor: tuple | None,
    local_error: BaseException | None,
) -> None:
    """Publish framework failures before any rank can enter a GPU wait loop."""

    world_size = dist.get_world_size(group=cpu_group)
    rank = dist.get_rank(group=cpu_group)
    local_result = (
        descriptor,
        None if local_error is None else f"{type(local_error).__name__}: {local_error}",
    )
    results: list[tuple[tuple | None, str | None] | None] = [None] * world_size
    dist.all_gather_object(results, local_result, group=cpu_group)

    for failed_rank, result in enumerate(results):
        if result is None:
            error_text = "rank did not publish an admission result"
        else:
            _, error_text = result
        if error_text is not None:
            message = (
                f"SharedEP {stage} admission failed on rank {failed_rank}: {error_text}"
            )
            if failed_rank == rank:
                raise RuntimeError(message) from local_error
            raise RuntimeError(message)

    descriptors = [result[0] for result in results if result is not None]
    if descriptors and any(value != descriptors[0] for value in descriptors[1:]):
        raise RuntimeError(
            f"SharedEP {stage} admission differs across ranks: {descriptors}"
        )


def _admit_shared_ep_framework(
    config: MoeRunnerConfig,
    parallel,
) -> tuple[SharedEpProfile, SharedEpRuntimeHooks, Any]:
    cpu_group = parallel.moe_ep_group.cpu_group
    profile = None
    runtime_hooks = None
    descriptor = None
    local_error = None
    try:
        if parallel.moe_ep_size != 8:
            raise ValueError(f"SharedEP requires EP8, got EP{parallel.moe_ep_size}")
        if parallel.attn_dp_size != 8:
            raise ValueError(f"SharedEP requires DP8, got DP{parallel.attn_dp_size}")
        capability, device_name = _get_device_profile_capability()
        quantization, block_shape = _profile_quantization(config)
        profile = select_profile(
            config,
            capability=capability,
            ep_size=parallel.moe_ep_size,
            block_shape=block_shape,
            max_tokens_per_rank=RELEASE_MAX_TOKENS_PER_RANK,
            quantization=quantization,
        )
        runtime_hooks = get_shared_ep_runtime_hooks()
        lane_protocol = compute_shared_ep_lane_protocol(get_server_args())
        model_namespace = validate_shared_ep_model_namespace(
            config.shared_ep_model_namespace
        )
        descriptor = (
            device_name,
            runtime_hooks.name,
            profile.admission_tuple(),
            model_namespace,
            lane_protocol.tbo_width,
            lane_protocol.generation_width,
        )
    except BaseException as error:
        local_error = error

    _synchronize_admission_stage(
        cpu_group,
        stage="framework",
        descriptor=descriptor,
        local_error=local_error,
    )
    assert profile is not None and runtime_hooks is not None
    return profile, runtime_hooks, cpu_group


def _get_shared_state(
    config: MoeRunnerConfig,
    profile: SharedEpProfile,
    runtime_hooks: SharedEpRuntimeHooks,
    *,
    model_namespace: str,
    lane_id: int,
) -> Any:
    """Return one process-lifetime VMM lane shared by sequential MoE layers."""

    resources = get_resources().buffers
    key = shared_ep_state_resource_key(
        runtime_name=runtime_hooks.name,
        profile_name=profile.name,
        ep_size=profile.ep_size,
        model_namespace=model_namespace,
        lane_id=lane_id,
    )
    state = resources.get(key)
    if state is None:
        ep_group = get_parallel().moe_ep_group
        use_mxfp4 = profile.quantization is SharedEpQuantization.MXFP4
        layout = SharedEpLayout.build(
            hidden_size=profile.hidden_size,
            top_k=profile.top_k,
            max_tokens_per_rank=profile.max_tokens_per_rank,
            input_format=(
                SharedEpInputFormat.BF16 if use_mxfp4 else SharedEpInputFormat.BLOCK_FP8
            ),
            direct_output=use_mxfp4,
        )
        state = runtime_hooks.create_state(
            layout=layout,
            cpu_group=ep_group.cpu_group,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        resources[key] = state
    expected = (
        config.hidden_size,
        config.top_k,
        profile.max_tokens_per_rank,
        (
            SharedEpInputFormat.BF16
            if profile.quantization is SharedEpQuantization.MXFP4
            else SharedEpInputFormat.BLOCK_FP8
        ),
        profile.quantization is SharedEpQuantization.MXFP4,
    )
    actual = (
        state.layout.hidden_size,
        state.layout.top_k,
        state.layout.max_tokens_per_rank,
        state.layout.input_format,
        state.layout.direct_output,
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
        *,
        model_namespace: str | None = None,
        lane_id: int = 0,
        admission: tuple[SharedEpProfile, SharedEpRuntimeHooks, Any] | None = None,
    ):
        super().__init__()
        parallel = get_parallel()
        self.profile, self.runtime_hooks, self.cpu_group = (
            admission
            if admission is not None
            else _admit_shared_ep_framework(config, parallel)
        )
        self.config = config
        configured_namespace = validate_shared_ep_model_namespace(
            config.shared_ep_model_namespace
        )
        if model_namespace is not None and model_namespace != configured_namespace:
            raise ValueError(
                "SharedEP dispatcher namespace differs from its admitted model "
                f"config: {model_namespace!r} != {configured_namespace!r}"
            )
        self.model_namespace = configured_namespace
        self.lane_id = int(lane_id)
        self.state = _get_shared_state(
            self.config,
            self.profile,
            self.runtime_hooks,
            model_namespace=self.model_namespace,
            lane_id=self.lane_id,
        )
        self.local_expert_start = parallel.moe_ep_rank * self.profile.num_local_experts
        self.fallback_dispatcher = fallback_dispatcher
        self.expert_mask_gpu = getattr(
            fallback_dispatcher,
            "expert_mask_gpu",
            None,
        )
        self._decode_quant_admitted = False
        self._stage = "initial"
        self._active_uses_shared_ep: bool | None = None

    def set_fallback_dispatcher(self, fallback_dispatcher: BaseDispatcher) -> None:
        self.fallback_dispatcher = fallback_dispatcher
        self.expert_mask_gpu = getattr(
            fallback_dispatcher,
            "expert_mask_gpu",
            None,
        )

    def _require_stage(self, expected: str) -> None:
        if self._stage != expected:
            raise RuntimeError(
                f"SharedEP lane {self.lane_id} is already in stage "
                f"{self._stage!r}; expected {expected!r}. Concurrent writers "
                "must select different state lanes."
            )

    @staticmethod
    def _use_shared_ep_decode() -> bool:
        # This flag is set from ForwardBatch.forward_mode before eager execution
        # and graph capture. Unlike is_extend_in_batch, it remains false for
        # TARGET_VERIFY even when decode-graph capture normalizes DP metadata.
        return bool(get_forward().shared_ep_is_decode) or (
            os.environ.get("SGLANG_SHARED_EP_DIRECT_SMALL_BATCH", "0") == "1"
        )

    def _require_fallback(self) -> BaseDispatcher:
        if self.fallback_dispatcher is None:
            raise RuntimeError(
                "SharedEP requires a platform materialized fallback outside "
                "plain decode"
            )
        return self.fallback_dispatcher

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        self._require_stage("initial")
        use_shared_ep = self._use_shared_ep_decode()
        if use_shared_ep:
            output = self._dispatch_shared_ep(hidden_states, topk_output)
        else:
            output = self._require_fallback().dispatch(
                hidden_states=hidden_states,
                topk_output=topk_output,
            )
        self._active_uses_shared_ep = use_shared_ep
        self._stage = "after_dispatch_b"
        return output

    def _dispatch_shared_ep(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ) -> SharedEpDispatchOutput:
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise TypeError("SharedEP decode requires standard Top-K output")
        if not self._decode_quant_admitted:
            raise RuntimeError(
                "SharedEP decode quantization was not admitted during weight setup"
            )

        state = self.state
        num_tokens = hidden_states.shape[0]
        _validate_decode_capacity(self.profile)
        if num_tokens > self.profile.max_tokens_per_rank:
            raise ValueError(
                "SharedEP decode capacity exceeded: "
                f"{num_tokens} > {self.profile.max_tokens_per_rank} local tokens"
            )
        if self.profile.quantization is SharedEpQuantization.MXFP4:
            from sglang.srt.layers.moe.shared_ep.fp4 import (
                publish_bf16_owner_input,
            )

            publish_bf16_owner_input(
                state.local_input,
                source=hidden_states,
                source_ids=topk_output.topk_ids,
                source_weights=topk_output.topk_weights,
            )
            # A missing/invalid route must reduce as zero rather than reusing a
            # contribution from the previous epoch.
            state.local_output[:num_tokens].zero_()
        else:
            from sglang.srt.layers.moe.shared_ep.kernels import quantize_pack_input

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
        self._require_stage("after_dispatch_b")
        try:
            if self._active_uses_shared_ep:
                if not CombineInputChecker.format_is_standard(combine_input):
                    raise TypeError(
                        "SharedEP decode produced a non-standard combine input"
                    )
                return combine_input.hidden_states
            return self._require_fallback().combine(combine_input=combine_input)
        finally:
            self._active_uses_shared_ep = None
            self._stage = "initial"

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ) -> None:
        """Publish/launch dispatch while retaining graph-static lane ownership."""

        self._require_stage("initial")
        use_shared_ep = self._use_shared_ep_decode()
        if use_shared_ep:
            self._staged_dispatch_output = self._dispatch_shared_ep(
                hidden_states,
                topk_output,
            )
        else:
            self._fallback_num_tokens = hidden_states.shape[0]
            self._require_fallback().dispatch_a(
                hidden_states=hidden_states,
                topk_output=topk_output,
            )
        self._active_uses_shared_ep = use_shared_ep
        self._stage = "after_dispatch_a"

    def dispatch_b(self):
        self._require_stage("after_dispatch_a")
        if self._active_uses_shared_ep:
            output = self._staged_dispatch_output
            del self._staged_dispatch_output
        else:
            output = self._require_fallback().dispatch_b()
        self._stage = "after_dispatch_b"
        return output

    def combine_a(self, combine_input: CombineInput) -> None:
        self._require_stage("after_dispatch_b")
        if self._active_uses_shared_ep:
            if not CombineInputChecker.format_is_standard(combine_input):
                raise TypeError("SharedEP decode produced a non-standard combine input")
            self._staged_combined_output = combine_input.hidden_states
        else:
            self._require_fallback().combine_a(combine_input=combine_input)
        self._stage = "after_combine_a"

    def combine_b(self) -> torch.Tensor:
        self._require_stage("after_combine_a")
        try:
            if self._active_uses_shared_ep:
                output = self._staged_combined_output
                del self._staged_combined_output
                return output
            output = self._require_fallback().combine_b()
            num_tokens = self._fallback_num_tokens
            del self._fallback_num_tokens
            return output[:num_tokens]
        finally:
            self._active_uses_shared_ep = None
            self._stage = "initial"

    def set_quant_config(self, quant_config: dict) -> None:
        descriptor = None
        local_error = None
        try:
            descriptor = self._validate_quant_config(quant_config)
            if self.fallback_dispatcher is not None:
                self.fallback_dispatcher.set_quant_config(quant_config)
        except BaseException as error:
            local_error = error

        _synchronize_admission_stage(
            self.cpu_group,
            stage="quantization",
            descriptor=descriptor,
            local_error=local_error,
        )
        super().set_quant_config(quant_config)
        self._decode_quant_admitted = True

    def _validate_quant_config(self, quant_config: dict) -> tuple:
        profile = self.profile
        if profile.quantization is SharedEpQuantization.MXFP4:
            return self._validate_mxfp4_quant_config(quant_config)
        if quant_config.get("shared_ep_quantization") != "block_fp8":
            raise ValueError("SharedEP supports block-FP8 expert weights only")
        if quant_config.get("shared_ep_weight_layout") != (
            SharedEpWeightLayout.CANONICAL.value
        ):
            raise ValueError(
                "SharedEP decode requires canonical weights; AITER-shuffled "
                "weights are fallback-only"
            )
        block_shape = tuple(quant_config.get("block_shape") or ())
        if block_shape != profile.block_shape:
            raise ValueError(
                f"SharedEP requires block shape {profile.block_shape}, "
                f"got {block_shape}"
            )

        weight_dtype = quant_config.get("weight_dtype")
        fp8_dtypes = {
            torch.float8_e4m3fn,
            getattr(torch, "float8_e4m3fnuz", torch.float8_e4m3fn),
        }
        if weight_dtype not in fp8_dtypes:
            raise TypeError(f"SharedEP requires FP8 expert weights, got {weight_dtype}")

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
        expected_s13 = (
            profile.num_local_experts,
            profile.intermediate_size * 2 // profile.block_shape[0],
            profile.hidden_size // profile.block_shape[1],
        )
        expected_s2 = (
            profile.num_local_experts,
            profile.hidden_size // profile.block_shape[0],
            profile.intermediate_size // profile.block_shape[1],
        )
        observed_shapes = (
            tuple(quant_config.get("w13_shape") or ()),
            tuple(quant_config.get("w2_shape") or ()),
            tuple(quant_config.get("w13_scale_shape") or ()),
            tuple(quant_config.get("w2_scale_shape") or ()),
        )
        expected_shapes = (expected_w13, expected_w2, expected_s13, expected_s2)
        if observed_shapes != expected_shapes:
            raise ValueError(
                "SharedEP canonical weight/scale shapes do not match the "
                f"{profile.name} profile: observed={observed_shapes}, "
                f"expected={expected_shapes}"
            )
        return (
            "block_fp8",
            SharedEpWeightLayout.CANONICAL.value,
            str(weight_dtype),
            block_shape,
            observed_shapes,
        )

    def _validate_mxfp4_quant_config(self, quant_config: dict) -> tuple:
        profile = self.profile
        if quant_config.get("shared_ep_quantization") != (
            SharedEpQuantization.MXFP4.value
        ):
            raise ValueError(
                "The DSV4-Pro SharedEP profile requires canonical MXFP4 experts"
            )
        if quant_config.get("shared_ep_weight_layout") != (
            SharedEpWeightLayout.CANONICAL.value
        ):
            raise ValueError(
                "SharedEP MXFP4 decode requires canonical unswizzled weights"
            )
        if quant_config.get("shared_ep_scale_layout") != (
            SharedEpScaleLayout.CANONICAL.value
        ):
            raise ValueError(
                "SharedEP MXFP4 decode requires canonical unswizzled E8M0 scales"
            )
        block_shape = tuple(quant_config.get("block_shape") or ())
        if block_shape != profile.block_shape:
            raise ValueError(
                f"SharedEP MXFP4 requires block shape {profile.block_shape}, "
                f"got {block_shape}"
            )
        if quant_config.get("weight_group_size") != 32:
            raise ValueError("SharedEP MXFP4 requires weight group size 32")
        if quant_config.get("scale_format") != "e8m0":
            raise ValueError("SharedEP MXFP4 requires E8M0 weight scales")

        weight_dtype = quant_config.get("weight_dtype")
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        valid_weight_dtypes = {torch.uint8}
        if fp4_dtype is not None:
            valid_weight_dtypes.add(fp4_dtype)
        if weight_dtype not in valid_weight_dtypes:
            raise TypeError(
                "SharedEP MXFP4 requires packed OCP E2M1 expert weights, "
                f"got {weight_dtype}"
            )
        scale_dtype = quant_config.get("weight_scale_dtype")
        e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
        valid_scale_dtypes = {torch.uint8}
        if e8m0_dtype is not None:
            valid_scale_dtypes.add(e8m0_dtype)
        if scale_dtype not in valid_scale_dtypes:
            raise TypeError(
                f"SharedEP MXFP4 requires E8M0 scale bytes, got {scale_dtype}"
            )

        expected_shapes = (
            (
                profile.num_local_experts,
                profile.intermediate_size * 2,
                profile.hidden_size // 2,
            ),
            (
                profile.num_local_experts,
                profile.hidden_size,
                profile.intermediate_size // 2,
            ),
            (
                profile.num_local_experts,
                profile.intermediate_size * 2,
                profile.hidden_size // 32,
            ),
            (
                profile.num_local_experts,
                profile.hidden_size,
                profile.intermediate_size // 32,
            ),
        )
        observed_shapes = (
            tuple(quant_config.get("w13_shape") or ()),
            tuple(quant_config.get("w2_shape") or ()),
            tuple(quant_config.get("w13_scale_shape") or ()),
            tuple(quant_config.get("w2_scale_shape") or ()),
        )
        if observed_shapes != expected_shapes:
            raise ValueError(
                "SharedEP canonical MXFP4 weight/scale shapes do not match "
                f"{profile.name}: observed={observed_shapes}, "
                f"expected={expected_shapes}"
            )
        return (
            SharedEpQuantization.MXFP4.value,
            SharedEpWeightLayout.CANONICAL.value,
            SharedEpScaleLayout.CANONICAL.value,
            str(weight_dtype),
            str(scale_dtype),
            block_shape,
            observed_shapes,
            bool(quant_config.get("fallback_uses_duplicate_tensors", False)),
        )


class SharedEpLaneDispatcher(BaseDispatcher):
    """Graph-static dispatcher facade over disjoint SharedEP state lanes."""

    def __init__(
        self,
        inners: list[SharedEpDispatcher],
        lane_protocol: SharedEpLaneProtocol,
    ):
        super().__init__()
        if len(inners) != lane_protocol.lane_count:
            raise ValueError(
                "SharedEP lane dispatcher received the wrong number of inner "
                f"dispatchers: {len(inners)} != {lane_protocol.lane_count}"
            )
        self._inners = tuple(inners)
        self.lane_protocol = lane_protocol

    @property
    def expert_mask_gpu(self):
        return self._inners[0].expert_mask_gpu

    @property
    def inner_dispatchers(self) -> tuple[SharedEpDispatcher, ...]:
        return self._inners

    def _inner(
        self,
        *,
        tbo_subbatch_index: int | None = None,
    ) -> SharedEpDispatcher:
        lane_id = self.lane_protocol.lane_id(
            generation_index=int(get_forward().shared_ep_generation),
            tbo_subbatch_index=tbo_subbatch_index,
        )
        return self._inners[lane_id]

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        return self._inner().dispatch(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )

    def combine(self, combine_input: CombineInput) -> torch.Tensor:
        return self._inner().combine(combine_input=combine_input)

    def dispatch_a(
        self,
        *,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        tbo_subbatch_index: int | None = None,
    ) -> None:
        self._inner(tbo_subbatch_index=tbo_subbatch_index).dispatch_a(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )

    def dispatch_b(self, *, tbo_subbatch_index: int | None = None):
        return self._inner(tbo_subbatch_index=tbo_subbatch_index).dispatch_b()

    def combine_a(
        self,
        *,
        combine_input: CombineInput,
        tbo_subbatch_index: int | None = None,
    ) -> None:
        self._inner(tbo_subbatch_index=tbo_subbatch_index).combine_a(
            combine_input=combine_input
        )

    def combine_b(
        self,
        *,
        tbo_subbatch_index: int | None = None,
    ) -> torch.Tensor:
        return self._inner(tbo_subbatch_index=tbo_subbatch_index).combine_b()

    def set_quant_config(self, quant_config: dict) -> None:
        super().set_quant_config(quant_config)
        for inner in self._inners:
            inner.set_quant_config(quant_config)

    def set_overlap_args(self, combine_overlap_args, meta_overlap_args: dict) -> None:
        super().set_overlap_args(combine_overlap_args, meta_overlap_args)
        for inner in self._inners:
            inner.set_overlap_args(combine_overlap_args, meta_overlap_args)

    def clear_overlap_args(self) -> None:
        super().clear_overlap_args()
        for inner in self._inners:
            inner.clear_overlap_args()


def _create_shared_ep_prefill_dispatcher(
    config: MoeRunnerConfig,
    *,
    group,
    instance_id: int = 0,
):
    fallback_backend = get_shared_ep_prefill_backend()
    if fallback_backend.is_aiter():
        from sglang.srt.layers.moe.token_dispatcher.moriep import MoriEPDispatcher

        fallback_cls = MoriEPDispatcher
        platform_kwargs = {"instance_id": int(instance_id)}
    else:
        from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPDispatcher

        fallback_cls = DeepEPDispatcher
        platform_kwargs = {}

    fallback_dispatcher = fallback_cls(
        group=group,
        router_topk=config.top_k,
        permute_fusion=True,
        num_experts=config.num_experts,
        num_local_experts=config.num_local_experts,
        hidden_size=config.hidden_size,
        params_dtype=config.params_dtype,
        # Prefill, TARGET_VERIFY, and DRAFT_EXTEND require the materialized
        # dispatcher. Do not let decode-graph capture's normalized
        # is_extend_in_batch=False select the low-latency implementation.
        deepep_mode=DeepEPMode.NORMAL,
        async_finish=True,
        return_recv_hook=True,
        **platform_kwargs,
    )
    return fallback_dispatcher


def create_shared_ep_dispatcher(
    config: MoeRunnerConfig,
    *,
    group,
) -> SharedEpLaneDispatcher:
    """Build graph-static SharedEP lanes plus materialized platform fallbacks."""

    fallback_backend = get_shared_ep_prefill_backend()
    if get_moe_runner_backend() is not fallback_backend:
        raise ValueError(
            "shared_ep requires --moe-runner-backend "
            f"{fallback_backend.value} on this platform"
        )

    lane_protocol = compute_shared_ep_lane_protocol(get_server_args())
    model_namespace = validate_shared_ep_model_namespace(
        config.shared_ep_model_namespace
    )
    # Admit every rank before constructing communication fallbacks, then create
    # all VMM/epoch lanes in deterministic lane-ID order on every rank.
    admission = _admit_shared_ep_framework(config, get_parallel())
    inners = [
        SharedEpDispatcher(
            config,
            model_namespace=model_namespace,
            lane_id=lane_id,
            admission=admission,
        )
        for lane_id in range(lane_protocol.lane_count)
    ]
    for inner in inners:
        inner.set_fallback_dispatcher(
            _create_shared_ep_prefill_dispatcher(
                config,
                group=group,
                instance_id=inner.lane_id,
            )
        )
    return SharedEpLaneDispatcher(inners, lane_protocol)


def _validate_decode_weights(
    dispatch_output: SharedEpDispatchOutput,
    quant_info: SharedEpQuantInfo,
    runner_config: MoeRunnerConfig,
) -> None:
    profile = dispatch_output.profile
    if not isinstance(quant_info, SharedEpQuantInfo):
        raise TypeError("SharedEP requires runner-neutral SharedEpQuantInfo")
    quant_info.require_decode_capability(SharedEpQuantCapability.CANONICAL_BLOCK_FP8)
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


@register_fused_func("shared_ep", "aiter")
@register_fused_func("shared_ep", "deep_gemm")
def run_shared_ep(
    dispatch_output,
    quant_info: SharedEpQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    if not isinstance(dispatch_output, SharedEpDispatchOutput):
        raise TypeError(
            "The SharedEP fused path received a prefill dispatch object; "
            "MoeRunner must route it through the native fallback runner"
        )
    if dispatch_output.profile.quantization is SharedEpQuantization.MXFP4:
        from sglang.srt.layers.moe.shared_ep.fp4 import run_shared_ep_mxfp4

        return run_shared_ep_mxfp4(dispatch_output, quant_info, runner_config)
    _validate_decode_weights(dispatch_output, quant_info, runner_config)

    import triton.language as tl

    from sglang.kernels.ops.attention.dsv4 import (
        silu_and_mul_contig_post_quant,
    )
    from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
        invoke_fused_moe_kernel,
    )
    from sglang.srt.layers.moe.shared_ep.kernels import prepare_routes

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
    use_tma = is_cuda()
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
        a_use_tma=use_tma,
        b_use_tma=use_tma,
        a_sorted=not use_tma,
        a_is_prequantized=True,
    )
    state.output_epoch.publish()
    state.output_epoch.wait_all()

    # The next layer cannot reuse this output until every owner has reduced it:
    # each owner publishes the next input only after this sum, and the next
    # expert phase waits for all input publications.
    output = state.local_output[: dispatch_output.num_tokens].sum(dim=1)
    return StandardCombineInput(hidden_states=output)
