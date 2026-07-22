from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

import torch

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
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils import is_cuda, is_gfx95_supported, is_hip

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
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
    use_mxfp8: bool = False
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
        # A dispatched-EP rank can receive zero tokens for its local experts
        # (e.g. mori normal dispatch truncated to valid==0). moe_align then
        # yields num_tokens_post_padded==0 and the fused_moe grid launches with
        # zero blocks -> hipErrorInvalidConfiguration. Short-circuit to a
        # correct empty output. shape[0] is host-side (no D2H sync); under
        # cuda-graph capture the buffer is never truncated so this never fires.
        if runner_input.hidden_states.shape[0] == 0:
            out = runner_input.hidden_states.new_empty(
                (0, quant_info.w2_weight.shape[1])
            )
            return TritonRunnerOutput(hidden_states=out)

        if quant_info.use_mxfp8 and is_hip() and is_gfx95_supported():
            from sglang.kernels.ops.moe.mxfp8_moe_amd_gfx95 import (
                fused_experts_mxfp8,
            )

            out = fused_experts_mxfp8(
                runner_input.hidden_states,
                quant_info.w13_weight,
                quant_info.w2_weight,
                runner_input.topk_weights,
                runner_input.topk_ids,
                quant_info.w13_scale,
                quant_info.w2_scale,
                b1=quant_info.b13,
                b2=quant_info.b2,
                activation=self.config.activation,
                is_gated=self.config.is_gated,
                no_combine=self.config.no_combine,
                inplace=self.config.inplace,
                apply_router_weight_on_input=self.config.apply_router_weight_on_input,
                routed_scaling_factor=self.config.routed_scaling_factor,
                gemm1_alpha=self.config.gemm1_alpha,
                gemm1_limit=self.config.gemm1_clamp_limit,
                swiglu_limit=self.config.swiglu_limit,
                gate_up_interleaved=self.config.gate_up_interleaved,
            )
            return TritonRunnerOutput(hidden_states=out)

        if quant_info.use_mxfp8 and is_cuda():
            raise NotImplementedError(
                "Triton MoE runner does not support NVIDIA MXFP8; use "
                "--moe-runner-backend deep_gemm (or flashinfer_trtllm/cutlass)."
            )

        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            _fused_moe_kernel_sequence,
        )

        filter_expert = (
            self.config.num_experts is None
            or self.config.num_experts != self.config.num_local_experts
        )
        # Dense-compacted mori input (see pre_permute_deepep_normal_to_triton)
        # carries no invalid (-1) expert ids, so force the unfiltered kernel path
        # (identical to EP=1) rather than the filter_expert branch.
        if running_state.get("mori_dense", False):
            filter_expert = False

        out = _fused_moe_kernel_sequence(
            runner_input.hidden_states,
            quant_info.w13_weight,
            quant_info.w2_weight,
            runner_input.topk_weights,
            runner_input.topk_ids,
            runner_input.sorted_token_ids,
            runner_input.expert_ids,
            runner_input.num_tokens_post_padded,
            running_state["config"],
            running_state.get("down_config"),
            running_state.get("down_moe_use_tma", False),
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
            activation=self.config.activation,
            is_gated=self.config.is_gated,
            no_combine=self.config.no_combine,
            inplace=self.config.inplace,
            apply_router_weight_on_input=self.config.apply_router_weight_on_input,
            # Dense-compacted mori input already carries routed_scaling_factor in
            # topk_weights (applied at gather), so the kernel must not re-apply it.
            # Passing 1.0 lets the topk==1 direct-write reduction guard fire
            # instead of scaling a never-written intermediate buffer into output.
            routed_scaling_factor=(
                1.0
                if running_state.get("mori_dense", False)
                else self.config.routed_scaling_factor
            ),
            gemm1_alpha=self.config.gemm1_alpha,
            gemm1_limit=self.config.gemm1_clamp_limit,
            filter_expert=filter_expert,
            hooks=hooks,
            swiglu_limit=self.config.swiglu_limit,
        )

        return TritonRunnerOutput(hidden_states=out)

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON


@register_fused_func("none", "triton")
def fused_experts_none_to_triton(
    dispatch_output: StandardDispatchOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    if quant_info.use_mxfp8 and is_hip() and is_gfx95_supported():
        from sglang.kernels.ops.moe.mxfp8_moe_amd_gfx95 import (
            fused_experts_mxfp8,
        )

        topk_weights, topk_ids, _ = dispatch_output.topk_output
        output = fused_experts_mxfp8(
            hidden_states=dispatch_output.hidden_states,
            w1=quant_info.w13_weight,
            w2=quant_info.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            w1_scale=quant_info.w13_scale,
            w2_scale=quant_info.w2_scale,
            b1=quant_info.b13,
            b2=quant_info.b2,
            activation=runner_config.activation,
            is_gated=runner_config.is_gated,
            no_combine=runner_config.no_combine,
            inplace=runner_config.inplace,
            apply_router_weight_on_input=runner_config.apply_router_weight_on_input,
            routed_scaling_factor=runner_config.routed_scaling_factor,
            gemm1_alpha=runner_config.gemm1_alpha,
            gemm1_limit=runner_config.gemm1_clamp_limit,
            swiglu_limit=runner_config.swiglu_limit,
            gate_up_interleaved=runner_config.gate_up_interleaved,
        )
    else:
        if quant_info.use_mxfp8 and is_cuda():
            raise NotImplementedError(
                "Triton MoE runner does not support NVIDIA MXFP8; use "
                "--moe-runner-backend deep_gemm (or flashinfer_trtllm/cutlass)."
            )
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            fused_experts,
        )

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

    # Registered fallback for format-conversion tests and examples.

    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        _prepare_fused_moe_run,
    )
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    hidden_states, topk_output = (
        dispatch_output.hidden_states,
        dispatch_output.topk_output,
    )

    assert TopKOutputChecker.format_is_standard(topk_output)

    (
        config,
        down_config,
        down_moe_use_tma,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    ) = _prepare_fused_moe_run(
        hidden_states,
        quant_info.w13_weight,
        quant_info.w2_weight,
        topk_output.topk_ids,
        use_fp8_w8a8=quant_info.use_fp8_w8a8,
        use_int8_w8a8=quant_info.use_int8_w8a8,
        use_int8_w8a16=quant_info.use_int8_w8a16,
        use_int4_w4a16=quant_info.use_int4_w4a16,
        per_channel_quant=quant_info.per_channel_quant,
        block_shape=quant_info.block_shape,
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

    # Registered fallback for format-conversion tests and examples.

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    return StandardCombineInput(
        hidden_states=runner_output.hidden_states,
    )


@register_pre_permute("deepep_normal", "triton")
def pre_permute_deepep_normal_to_triton(
    dispatch_output: Any,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> TritonRunnerInput:
    # DeepEP/Mori "normal" dispatch delivers tokens already routed to their
    # expert-home rank; topk_ids index the local experts (invalid slots are
    # skipped by moe_align + the kernel's filter_expert). The triton runner
    # core needs the aligned (sorted_token_ids/expert_ids/num_tokens_post_padded)
    # layout, so build it here from the same helper the standard path uses.
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        _prepare_fused_moe_run,
    )

    hidden_states = dispatch_output.hidden_states
    topk_ids = dispatch_output.topk_ids
    topk_weights = dispatch_output.topk_weights

    # DeepEP/mori deliver hidden_states in a dispatcher-owned buffer (for mori
    # an external, RDMA-registered, read-only-mapped region). The triton runner
    # core writes MoE output in-place into hidden_states when config.inplace is
    # True (the default), which faults on that buffer ("write to read-only
    # page"). Force a fresh output allocation for this dispatch format.
    runner_config.inplace = False

    # Mori packs origin routing (used by combine) and dispatches into a
    # fixed-size buffer; cap it to the configured input budget so the kernel
    # does not read padding rows past the valid tokens.
    is_mori = hasattr(dispatch_output, "origin_topk_ids")

    if is_mori:
        from sglang.srt.runtime_context import get_parallel

        # Mori normal dispatch delivers a fixed-size buffer with the valid
        # received tokens front-packed at [0, totalRecvTokenNum) and zero
        # padding after. We keep the full (static) buffer shape and only remap
        # ids with tensor ops: any data-dependent truncation would change the
        # shape and break cuda-graph capture/replay on the decode path (which
        # runs under a graph). topk_ids carry GLOBAL expert ids in
        # [0, num_experts); the triton runner's moe_align/kernel index this
        # rank's LOCAL experts, so remap to [0, num_local_experts) and route
        # non-local experts (and padding rows) to -1, the EP filtered-expert id
        # (moe_align/kernel skip -1 slots; w1 only holds local experts 0..nle-1,
        # so any out-of-range id like num_local_experts would index w1 OOB).
        # Each rank thus computes only its local experts' weighted partial
        # (padding rows have zero activations -> zero output); mori combine sums
        # the partials across ranks and reads only [0, totalRecvTokenNum),
        # discarding the padded tail.
        nle = runner_config.num_local_experts
        ep_rank = get_parallel().moe_ep_rank

        # The triton fused_moe kernel (unlike aiter) has no valid-row mask and
        # would run moe_align + both GEMMs over the whole fixed dispatch buffer,
        # including the zero-padded tail past the valid front-packed tokens.
        # num_recv_tokens_per_expert is a host python list, so sum() is a
        # host-side count (no D2H sync). On the eager path (prefill runs eager
        # here) bound work to the valid rows by truncation; under cuda-graph
        # capture (decode) shapes must stay static, so keep the full buffer.
        nre = getattr(dispatch_output, "num_recv_tokens_per_expert", None)
        # mori combine reads the expert output back through the same fixed-size
        # dispatch buffer it built src_info against, so it expects hidden with
        # the full pre-dispatch row count. Record it here so post_permute can
        # zero-pad the (truncated) kernel output back to that shape before
        # handing it to combine.
        running_state["mori_full_rows"] = int(hidden_states.shape[0])
        _did_trunc = False
        if nre is not None and not torch.cuda.is_current_stream_capturing():
            valid = int(sum(nre))
            hidden_states = hidden_states[:valid]
            topk_ids = topk_ids[:valid]
            topk_weights = topk_weights[:valid]
            _did_trunc = True

        local = topk_ids.to(torch.int64) - ep_rank * nle
        topk_ids = torch.where(
            (local >= 0) & (local < nle),
            local,
            torch.full_like(local, -1),
        ).to(torch.int32)

        # Dense-compaction: gather only the valid (row, local_expert) pairs into
        # a dense topk=1 batch with no -1 ids, so the kernel runs the unfiltered
        # path (filter_expert=False, identical to EP=1) and the down-GEMM writes
        # each pair's weighted expert output directly to its output row. This
        # avoids the HIP filter_expert reduction path; each pair's output is
        # scatter-added back per token in post_permute. Only on the truncated
        # eager path -- cuda-graph capture keeps the static -1 buffer shape.
        if _did_trunc:
            valid_mask = topk_ids >= 0
            pair_rows = valid_mask.nonzero(as_tuple=True)[0]
            dense_experts = topk_ids[valid_mask].to(torch.int32).view(-1, 1)
            dense_weights = topk_weights[valid_mask].view(-1, 1)
            dense_hidden = hidden_states.index_select(0, pair_rows)
            running_state["mori_dense"] = True
            running_state["mori_dense_pair_rows"] = pair_rows
            running_state["mori_dense_out_rows"] = int(hidden_states.shape[0])
            hidden_states = dense_hidden
            topk_ids = dense_experts
            topk_weights = dense_weights
        else:
            running_state["mori_dense"] = False

        running_state["combine_topk_ids"] = dispatch_output.origin_topk_ids
        running_state["combine_topk_weights"] = dispatch_output.origin_topk_weights
    else:
        running_state["combine_topk_ids"] = topk_ids
        running_state["combine_topk_weights"] = topk_weights
    running_state["combine_is_mori"] = is_mori

    (
        config,
        down_config,
        down_moe_use_tma,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    ) = _prepare_fused_moe_run(
        hidden_states,
        quant_info.w13_weight,
        quant_info.w2_weight,
        topk_ids,
        use_fp8_w8a8=quant_info.use_fp8_w8a8,
        use_int8_w8a8=quant_info.use_int8_w8a8,
        use_int8_w8a16=quant_info.use_int8_w8a16,
        use_int4_w4a16=quant_info.use_int4_w4a16,
        per_channel_quant=quant_info.per_channel_quant,
        block_shape=quant_info.block_shape,
        ignore_invalid_expert=is_mori and not running_state.get("mori_dense", False),
    )

    running_state["config"] = config
    running_state["down_config"] = down_config
    running_state["down_moe_use_tma"] = down_moe_use_tma

    return TritonRunnerInput(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
    )


@register_post_permute("triton", "deepep_normal")
def post_permute_triton_to_deepep_normal(
    runner_output: TritonRunnerOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
):
    if running_state["combine_is_mori"]:
        from sglang.srt.layers.moe.token_dispatcher.moriep import (
            MoriEPNormalCombineInput,
        )

        cls = MoriEPNormalCombineInput

        hidden_states = runner_output.hidden_states
        # Un-compact the dense topk=1 batch: each pair's weighted expert output
        # (kernel already applied the routing weight) scatter-adds back to its
        # token row, reproducing the per-token top-k reduction.
        if running_state.get("mori_dense", False):
            pair_rows = running_state["mori_dense_pair_rows"]
            out_rows = running_state["mori_dense_out_rows"]
            scattered = hidden_states.new_zeros((out_rows, hidden_states.shape[1]))
            scattered.index_add_(0, pair_rows, hidden_states)
            hidden_states = scattered
        # The kernel ran over the truncated [valid, dim] buffer; mori combine
        # reassembles through the full fixed-size dispatch buffer, so pad the
        # output back to the original pre-dispatch row count (real rows front-
        # packed, zero tail) to match combine's src_info addressing.
        full_rows = running_state.get("mori_full_rows")
        if full_rows is not None and hidden_states.shape[0] < full_rows:
            padded = hidden_states.new_zeros((full_rows, hidden_states.shape[1]))
            padded[: hidden_states.shape[0]] = hidden_states
            hidden_states = padded

        return cls(
            hidden_states=hidden_states,
            topk_ids=running_state["combine_topk_ids"],
            topk_weights=running_state["combine_topk_weights"],
        )
    else:
        from sglang.srt.layers.moe.token_dispatcher.deepep import (
            DeepEPNormalCombineInput,
        )

        cls = DeepEPNormalCombineInput

    return cls(
        hidden_states=runner_output.hidden_states,
        topk_ids=running_state["combine_topk_ids"],
        topk_weights=running_state["combine_topk_weights"],
    )
