from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

import torch
import torch.nn.functional as F
import triton.language as tl

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_fused_func,
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend, get_moe_padding_size
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_hip,
    is_xpu,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


_is_hip = is_hip()
_is_cuda = is_cuda()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_xpu = is_xpu()


if _is_cuda:
    from sgl_kernel import gelu_and_mul, moe_sum_reduce, silu_and_mul
elif _is_cpu and _is_cpu_amx_available:
    pass
elif _is_hip:
    from sgl_kernel import gelu_and_mul, silu_and_mul

    if _use_aiter:
        try:
            from aiter import moe_sum
        except ImportError:
            raise ImportError("aiter is required when SGLANG_USE_AITER is set to True")
elif _is_xpu:
    from sgl_kernel import moe_sum_reduce, silu_and_mul

# Try to import vllm_ops for non-CUDA/HIP/XPU platforms
_has_vllm_ops = False
if not _is_cuda and not _is_hip and not _is_xpu:
    try:
        from vllm import _custom_ops as vllm_ops

        _has_vllm_ops = True
    except ImportError:
        # Fallback: vllm not available, will use native PyTorch implementations
        _has_vllm_ops = False


if _is_cuda or _is_hip or _is_xpu:
    from sgl_kernel import (  # noqa: F401
        moe_align_block_size as sgl_moe_align_block_size,
    )


@dataclass
class TritonRunnerInput(RunnerInput):

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    sorted_token_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON


@dataclass
class TritonRunnerOutput(RunnerOutput):

    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON


@dataclass
class TritonMoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    b13: Optional[torch.Tensor] = None
    b2: Optional[torch.Tensor] = None
    use_fp8_w8a8: bool = False
    use_int8_w8a8: bool = False
    use_int8_w8a16: bool = False
    use_int4_w4a16: bool = False
    per_channel_quant: bool = False
    w13_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None
    w13_zp: Optional[torch.Tensor] = None
    w2_zp: Optional[torch.Tensor] = None
    a13_scale: Optional[torch.Tensor] = None
    a2_scale: Optional[torch.Tensor] = None
    block_shape: Optional[List[int]] = None


class TritonRunnerCore(MoeRunnerCore):

    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)

    def run(
        self,
        runner_input: TritonRunnerInput,
        quant_info: TritonMoeQuantInfo,
        running_state: dict,
        hooks: Optional[Any] = None,
    ) -> TritonRunnerOutput:

        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
            _swiglu_gpt_oss_sigmoid_alpha,
            _swiglu_silu_clamp_mul,
            moe_sum_reduce_torch_compile,
        )
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels import (
            act_and_mul_triton,
            invoke_fused_moe_kernel,
            moe_sum_reduce_triton,
        )

        hidden_states = runner_input.hidden_states
        topk_weights = runner_input.topk_weights
        topk_ids = runner_input.topk_ids
        sorted_token_ids = runner_input.sorted_token_ids
        expert_ids = runner_input.expert_ids
        num_tokens_post_padded = runner_input.num_tokens_post_padded

        w13 = quant_info.w13_weight
        w2 = quant_info.w2_weight
        b13 = quant_info.b13
        b2 = quant_info.b2
        a13_scale = quant_info.a13_scale
        a2_scale = quant_info.a2_scale
        w13_scale = quant_info.w13_scale
        w2_scale = quant_info.w2_scale
        w13_zp = quant_info.w13_zp
        w2_zp = quant_info.w2_zp
        block_shape = quant_info.block_shape
        per_channel_quant = quant_info.per_channel_quant
        use_fp8_w8a8 = quant_info.use_fp8_w8a8
        use_int8_w8a8 = quant_info.use_int8_w8a8
        use_int8_w8a16 = quant_info.use_int8_w8a16
        use_int4_w4a16 = quant_info.use_int4_w4a16

        activation = self.config.activation
        is_gated = self.config.is_gated
        no_combine = self.config.no_combine
        inplace = self.config.inplace
        gemm1_alpha = self.config.gemm1_alpha
        gemm1_limit = self.config.gemm1_clamp_limit
        routed_scaling_factor = self.config.routed_scaling_factor
        apply_router_weight_on_input = self.config.apply_router_weight_on_input
        filter_expert = (
            self.config.num_experts is None
            or self.config.num_experts != self.config.num_local_experts
        )

        config = running_state["config"]
        down_config = running_state.get("down_config")
        down_moe_use_tma = running_state.get("down_moe_use_tma", False)

        num_tokens = hidden_states.shape[0]
        E, N, _ = w13.shape
        topk = topk_ids.shape[1]
        compute_type = (
            tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16
        )

        padded_tokens = (
            min(num_tokens * topk, E + 1) * (config["BLOCK_SIZE_M"] - 1)
            if down_moe_use_tma
            else 0
        )
        total_tokens = num_tokens * topk + padded_tokens

        if no_combine:
            assert not inplace
            out_hidden_states = torch.empty(
                (num_tokens, topk, w2.shape[1]),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        elif inplace:
            out_hidden_states = hidden_states
        else:
            out_hidden_states = torch.empty_like(hidden_states)

        use_fused_moe_sum_all_reduce = (
            get_global_server_args().enable_fused_moe_sum_all_reduce
            and (not no_combine)
            and (topk > 2)
            and (not use_int8_w8a16)
            and (not use_int4_w4a16)
        )

        intermediate_cache1 = torch.empty(
            (total_tokens, N),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        invoke_fused_moe_kernel(
            hidden_states,
            w13,
            b13,
            intermediate_cache1,
            a13_scale,
            w13_scale,
            w13_zp,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight_on_input,
            topk,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            c_sorted=down_moe_use_tma,
            filter_expert=filter_expert,
        )

        if hooks and hooks.after_gate_up:
            hooks.after_gate_up(
                hidden_states, intermediate_cache1, topk_weights, topk_ids
            )

        intermediate_cache2 = torch.empty(
            (total_tokens, N // 2),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        # Activation function with multiplication
        if activation == "silu" and is_gated:
            if gemm1_alpha is not None:
                assert gemm1_limit is not None
                intermediate_cache2 = _swiglu_gpt_oss_sigmoid_alpha(
                    intermediate_cache1.view(-1, N), gemm1_alpha, gemm1_limit
                )
            elif gemm1_limit is not None:
                intermediate_cache2 = _swiglu_silu_clamp_mul(
                    intermediate_cache1.view(-1, N), gemm1_limit
                )
            elif _is_cuda or _is_hip or _is_xpu:
                if not filter_expert:
                    silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
                else:
                    act_and_mul_triton(
                        intermediate_cache1.view(-1, N),
                        intermediate_cache2,
                        config,
                        topk_ids,
                        expert_ids,
                        down_moe_use_tma,
                        activation,
                    )
            else:
                if _has_vllm_ops:
                    vllm_ops.silu_and_mul(
                        intermediate_cache2, intermediate_cache1.view(-1, N)
                    )
                else:
                    x = intermediate_cache1.view(-1, N)
                    d = x.shape[-1] // 2
                    intermediate_cache2.copy_(F.silu(x[..., :d]) * x[..., d:])
        elif activation == "gelu" and is_gated:
            assert gemm1_alpha is None, "gemm1_alpha is not supported for gelu"
            assert gemm1_limit is None, "gemm1_limit is not supported for gelu"
            if _is_cuda or _is_hip:
                if not filter_expert:
                    gelu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
                else:
                    act_and_mul_triton(
                        intermediate_cache1.view(-1, N),
                        intermediate_cache2,
                        config,
                        topk_ids,
                        expert_ids,
                        down_moe_use_tma,
                        activation,
                    )
            else:
                if _has_vllm_ops:
                    vllm_ops.gelu_and_mul(
                        intermediate_cache2, intermediate_cache1.view(-1, N)
                    )
                else:
                    x = intermediate_cache1.view(-1, N)
                    d = x.shape[-1] // 2
                    intermediate_cache2.copy_(F.gelu(x[..., :d]) * x[..., d:])
        # Activation function without multiplication
        elif activation == "silu" and not is_gated:
            intermediate_cache2 = F.silu(intermediate_cache1.view(-1, N))
        elif activation == "gelu" and not is_gated:
            intermediate_cache2 = F.gelu(intermediate_cache1.view(-1, N))
        elif activation == "relu2" and not is_gated:
            intermediate_cache2 = torch.square(F.relu(intermediate_cache1.view(-1, N)))
        else:
            raise ValueError(f"Unsupported activation: {activation=}, with {is_gated=}")

        del intermediate_cache1

        intermediate_cache3 = torch.empty(
            (num_tokens, topk, w2.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        # LoRA hooks force intermediate_cache3 so the hook can modify it before reduction.
        _use_intermediate = not no_combine and (topk != 1 or hooks)

        out_slice = None
        if use_fused_moe_sum_all_reduce:
            out_slice = out_hidden_states
            out_slice.zero_()

        invoke_fused_moe_kernel(
            intermediate_cache2,
            w2,
            b2,
            (
                out_slice
                if use_fused_moe_sum_all_reduce
                else (
                    intermediate_cache3
                    if _use_intermediate
                    else out_hidden_states.unsqueeze(0)
                )
            ),
            a2_scale,
            w2_scale,
            w2_zp,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            down_config or config,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            a_use_tma=down_moe_use_tma,
            b_use_tma=down_moe_use_tma,
            filter_expert=filter_expert,
            fuse_sum_all_reduce=use_fused_moe_sum_all_reduce,
            router_topk=topk,
        )

        if hooks and hooks.after_down:
            hooks.after_down(
                intermediate_cache2, intermediate_cache3, topk_weights, topk_ids
            )

        del intermediate_cache2

        if routed_scaling_factor is None:
            routed_scaling_factor = 1.0

        if no_combine:
            pass
        elif _is_cuda:
            if use_fused_moe_sum_all_reduce:
                if routed_scaling_factor != 1.0:
                    assert out_slice is not None
                    out_slice.mul_(routed_scaling_factor)
            elif topk == 1 and routed_scaling_factor == 1.0 and not _use_intermediate:
                pass  # we wrote directly into out_hidden_states
            elif topk == 2 and routed_scaling_factor == 1.0:
                torch.add(
                    intermediate_cache3[:, 0],
                    intermediate_cache3[:, 1],
                    out=out_hidden_states,
                ).squeeze(dim=1)
            else:
                # According to micro benchmark results, torch.compile can get better performance for small token.
                if num_tokens <= 32:
                    moe_sum_reduce_torch_compile(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states,
                        routed_scaling_factor,
                    )
                else:
                    moe_sum_reduce(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states,
                        routed_scaling_factor,
                    )
        elif _is_hip:
            if _use_aiter:
                moe_sum(
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states,
                )
            else:
                if num_tokens <= 32:
                    moe_sum_reduce_torch_compile(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states,
                        routed_scaling_factor,
                    )
                else:
                    moe_sum_reduce_triton(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states,
                        routed_scaling_factor,
                    )
        elif _is_xpu:
            moe_sum_reduce(
                intermediate_cache3.view(*intermediate_cache3.shape),
                out_hidden_states,
                routed_scaling_factor,
            )
        else:
            if _has_vllm_ops:
                vllm_ops.moe_sum(
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states,
                )
            else:
                moe_sum_reduce_triton(
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states,
                    routed_scaling_factor,
                )

        del intermediate_cache3

        return TritonRunnerOutput(
            hidden_states=out_hidden_states,
        )

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON


@register_fused_func("none", "triton")
def fused_experts_none_to_triton(
    dispatch_output: StandardDispatchOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    output = fused_experts(
        hidden_states=dispatch_output.hidden_states,
        w1=quant_info.w13_weight,
        w2=quant_info.w2_weight,
        topk_output=dispatch_output.topk_output,
        moe_runner_config=runner_config,
        b1=quant_info.b13,
        b2=quant_info.b2,
        use_fp8_w8a8=quant_info.use_fp8_w8a8,
        use_int8_w8a8=quant_info.use_int8_w8a8,
        use_int8_w8a16=quant_info.use_int8_w8a16,
        use_int4_w4a16=quant_info.use_int4_w4a16,
        per_channel_quant=quant_info.per_channel_quant,
        w1_scale=quant_info.w13_scale,
        w2_scale=quant_info.w2_scale,
        w1_zp=quant_info.w13_zp,
        w2_zp=quant_info.w2_zp,
        a1_scale=quant_info.a13_scale,
        a2_scale=quant_info.a2_scale,
        block_shape=quant_info.block_shape,
    )

    return StandardCombineInput(
        hidden_states=output,
    )


@register_pre_permute("standard", "triton")
def pre_permute_standard_to_triton(
    dispatch_output: StandardDispatchOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> TritonRunnerInput:

    # NOTE: this is dead code as a fused func for standard format is registered.
    # This is left here for testing and examples.

    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import _down_moe_use_tma
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config import (
        get_config_dtype_str,
        try_get_optimal_moe_config,
    )
    from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import (
        moe_align_block_size,
    )
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    hidden_states, topk_output = (
        dispatch_output.hidden_states,
        dispatch_output.topk_output,
    )

    assert TopKOutputChecker.format_is_standard(topk_output)

    num_tokens = hidden_states.shape[0]
    num_local_experts = runner_config.num_local_experts

    if (
        not (quant_info.use_fp8_w8a8 or quant_info.use_int8_w8a8)
        or quant_info.block_shape is not None
        or _use_aiter
    ):
        padding_size = 0
    else:
        padding_size = get_moe_padding_size(_use_aiter)

    config_dtype = get_config_dtype_str(
        use_fp8_w8a8=quant_info.use_fp8_w8a8,
        use_int8_w8a8=quant_info.use_int8_w8a8,
        use_int8_w8a16=quant_info.use_int8_w8a16,
        use_int4_w4a16=quant_info.use_int4_w4a16,
        dtype=hidden_states.dtype,
    )

    config, (down_config, _) = try_get_optimal_moe_config(
        quant_info.w13_weight.shape,
        (
            num_local_experts,
            quant_info.w2_weight.shape[1],
            quant_info.w2_weight.shape[2] - padding_size,
        ),
        topk_output.topk_ids.shape[1],
        config_dtype,
        num_tokens,
        block_shape=quant_info.block_shape,
        per_channel_quant=quant_info.per_channel_quant,
        return_down_config=True,
    )

    down_moe_use_tma = (
        _down_moe_use_tma()
        and down_config is not None
        and down_config.pop("USE_TMA", False)
    )

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_output.topk_ids, config["BLOCK_SIZE_M"], num_local_experts
    )

    running_state["config"] = config
    running_state["down_config"] = down_config
    running_state["down_moe_use_tma"] = down_moe_use_tma

    return TritonRunnerInput(
        hidden_states=hidden_states,
        topk_weights=topk_output.topk_weights,
        topk_ids=topk_output.topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
    )


@register_post_permute("triton", "standard")
def post_permute_triton_to_standard(
    runner_output: TritonRunnerOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:

    # NOTE: this is dead code as a fused func for standard format is registered.
    # This is left here for testing and examples.

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    return StandardCombineInput(
        hidden_states=runner_output.hidden_states,
    )
