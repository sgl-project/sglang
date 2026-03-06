from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch

from sglang.srt.layers.moe.cutlass_moe_params import (
    CutlassMoEParams,
    CutlassMoEQuantType,
    CutlassMoEType,
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
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils import is_cuda, is_sm100_supported

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLCombineInput,
        DeepEPLLDispatchOutput,
        DeepEPNormalCombineInput,
        DeepEPNormalDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )

if is_cuda():
    from sgl_kernel import (
        apply_shuffle_mul_sum,
        es_sm100_mxfp8_blockscaled_grouped_quant,
        get_cutlass_w4a8_moe_mm_data,
        prepare_moe_input,
        scaled_fp4_experts_quant,
        shuffle_rows,
    )
    from sgl_kernel.gemm import sgl_per_tensor_quant_fp8

from sglang.srt.layers.moe.cutlass_utils import (
    cutlass_fused_experts_fp8,
    cutlass_moe_fp4,
    w4a8_moe,
)


@dataclass
class CutlassRunnerInput(RunnerInput):
    gate_up_input: torch.Tensor
    # Standard mode fields
    a_map: Optional[torch.Tensor] = None
    rep_aux: Optional[torch.Tensor] = None
    # DeepEP mode fields
    masked_m: Optional[torch.Tensor] = None
    expected_m: Optional[int] = None

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.CUTLASS


@dataclass
class CutlassRunnerOutput(RunnerOutput):
    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.CUTLASS


@dataclass
class CutlassMoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    w13_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None
    w1_blockscale: Optional[torch.Tensor] = None
    w2_blockscale: Optional[torch.Tensor] = None
    w1_alpha: Optional[torch.Tensor] = None
    w2_alpha: Optional[torch.Tensor] = None
    use_mxfp8: Optional[bool] = False
    enable_es: Optional[Tuple[bool, bool]] = (False, False)
    w13_input_scale: Optional[torch.Tensor] = None
    w2_input_scale: Optional[torch.Tensor] = None
    params: Optional[CutlassMoEParams] = None
    deepep_ll_or_deepep_normal: Optional[CutlassMoEType] = None


class CutlassRunnerCore(MoeRunnerCore):
    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)

        if not is_cuda():  # TODO (Jonahcb): maybe move this to server args
            raise RuntimeError("Cutlass runner requires CUDA support.")
        if self.config.activation not in ("silu", None):
            raise ValueError("Cutlass runner currently supports SiLU activation only.")

    def run(
        self,
        runner_input: CutlassRunnerInput,
        quant_info: CutlassMoeQuantInfo,
        running_state: Dict[str, torch.Tensor],
    ) -> CutlassRunnerOutput:
        quant_type = quant_info.params.quant_type
        deepep_ll_or_deepep_normal = quant_info.deepep_ll_or_deepep_normal

        if quant_type == CutlassMoEQuantType.W4A8 or deepep_ll_or_deepep_normal in [
            CutlassMoEType.DeepEP_LL,
            CutlassMoEType.DeepEP_Normal,
        ]:
            down_output = w4a8_moe(
                a=runner_input.gate_up_input,
                w1_q=quant_info.w13_weight,
                w2_q=quant_info.w2_weight,
                w1_scale=quant_info.w13_scale,
                w2_scale=quant_info.w2_scale,
                topk_ids=running_state["topk_ids"],
                params=quant_info.params,
                a1_scale=quant_info.w13_input_scale,
                a2_scale=quant_info.w2_input_scale,
                masked_m=runner_input.masked_m,
                deepep_ll_or_deepep_normal=deepep_ll_or_deepep_normal,
            )
            if deepep_ll_or_deepep_normal in [
                CutlassMoEType.DeepEP_LL,
                CutlassMoEType.DeepEP_Normal,
            ]:
                return CutlassRunnerOutput(hidden_states=down_output)

        elif quant_type == CutlassMoEQuantType.BlockscaledFP8:
            down_output = cutlass_fused_experts_fp8(
                a=runner_input.gate_up_input,
                w1_q=quant_info.w13_weight,
                w2_q=quant_info.w2_weight,
                w1_scale=quant_info.w13_scale,
                w2_scale=quant_info.w2_scale,
                topk_ids=running_state["topk_ids"],
                params=quant_info.params,
                rep_a1_scales=runner_input.rep_aux,
                use_mxfp8=quant_info.use_mxfp8,
                blockscale_offsets=running_state.get("blockscale_offsets", None),
                max_blockscale=running_state.get("max_blockscale", None),
                enable_es=quant_info.enable_es,
            )

        elif quant_type == CutlassMoEQuantType.BlockscaledFP4:
            down_output = cutlass_moe_fp4(
                a=runner_input.gate_up_input,
                rep_aux=runner_input.rep_aux,
                w1_fp4=quant_info.w13_weight,
                w1_blockscale=quant_info.w1_blockscale,
                w1_alphas=quant_info.w1_alpha,
                a2_gscale=quant_info.w2_input_scale,
                w2_fp4=quant_info.w2_weight,
                w2_blockscale=quant_info.w2_blockscale,
                w2_alphas=quant_info.w2_alpha,
                topk_ids=running_state["topk_ids"],
                params=quant_info.params,
            )

        return CutlassRunnerOutput(hidden_states=down_output)

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.CUTLASS


@register_pre_permute("standard", "cutlass")
def pre_permute_standard_to_cutlass(
    dispatch_output: StandardDispatchOutput,
    quant_info: CutlassMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: Dict[str, torch.Tensor],
) -> CutlassRunnerInput:
    # Common validation assertions for all CUTLASS MoE types
    assert (
        dispatch_output.topk_output.topk_weights.shape
        == dispatch_output.topk_output.topk_ids.shape
    ), "topk shape mismatch"
    assert (
        quant_info.w13_weight.shape[0] == quant_info.w2_weight.shape[0]
    ), "Expert number mismatch"
    assert (
        quant_info.w13_weight.shape[0] == quant_info.w13_scale.shape[0]
    ), "w1 scales expert number mismatch"
    assert (
        quant_info.w13_weight.shape[0] == quant_info.w2_scale.shape[0]
    ), "w2 scales expert number mismatch"

    hidden_states, topk_output = (
        dispatch_output.hidden_states,
        dispatch_output.topk_output,
    )
    topk_weights, topk_ids, _ = topk_output

    device = hidden_states.device

    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)

    running_state["topk_weights"] = topk_weights
    running_state["topk_ids"] = topk_ids

    if quant_info.params.quant_type == CutlassMoEQuantType.BlockscaledFP8:
        # Extract variables from quant_info
        a = hidden_states
        w1_q = quant_info.w13_weight
        w2_q = quant_info.w2_weight
        w1_scale = quant_info.w13_scale
        w2_scale = quant_info.w2_scale
        expert_offsets = quant_info.params.expert_offsets
        problem_sizes1 = quant_info.params.problem_sizes1
        problem_sizes2 = quant_info.params.problem_sizes2

        enable_es = quant_info.enable_es
        use_mxfp8 = quant_info.use_mxfp8

        if is_cuda:
            from sglang.srt.layers.quantization.fp8_kernel import (
                sglang_per_token_group_quant_fp8,
            )
        es_up, es_down = enable_es
        num_experts = w1_q.size(0)
        m = a.size(0)
        k = w1_q.size(1)
        n = w2_q.size(1)

        topk = topk_ids.size(1)
        device = a.device

        # FP8-specific asserts
        from sglang.srt.layers.quantization.fp8_utils import cutlass_fp8_supported

        if not cutlass_fp8_supported():
            raise RuntimeError("CUTLASS FP8 kernels are not available on this system.")
        assert w1_q.dtype == torch.float8_e4m3fn
        assert w2_q.dtype == torch.float8_e4m3fn
        assert hidden_states.dtype in [
            torch.half,
            torch.bfloat16,
        ], "Invalid input dtype"
        assert a.shape[1] == w1_q.shape[1], "Hidden size mismatch w1"
        assert w1_q.shape[2] == w2_q.shape[1] * 2, "Hidden size mismatch w2"
        assert w1_q.shape[0] == w2_q.shape[0], "Weights expert number mismatch"

        if use_mxfp8:
            assert (
                es_up and es_down
            ), "MXFP8 requires expert-specialization for both GEMMs"
            assert is_sm100_supported(), "MXFP8 requires SM100"
            assert k % 32 == 0, "MXFP8 requires hidden size to be divisible by 32"
            assert n % 32 == 0, "MXFP8 requires intermediate size to be divisible by 32"
            assert w1_scale.dtype == torch.uint8, "MXFP8 w1_scale must be uint8"
            assert w2_scale.dtype == torch.uint8, "MXFP8 w2_scale must be uint8"
            expected_w1_scale_shape = (
                num_experts,
                w1_q.shape[1] // 32,
                w1_q.shape[2],
            )
            expected_w2_scale_shape = (
                num_experts,
                w2_q.shape[1] // 32,
                w2_q.shape[2],
            )
            assert (
                w1_scale.shape == expected_w1_scale_shape
            ), f"MXFP8 w1_scale must be {expected_w1_scale_shape}, got {w1_scale.shape}"
            assert (
                w2_scale.shape == expected_w2_scale_shape
            ), f"MXFP8 w2_scale must be {expected_w2_scale_shape}, got {w2_scale.shape}"

            mxfp8_blockscale_align = 128
            total_tokens = m * topk
            nonzero_experts = min(num_experts, total_tokens)
            max_total = total_tokens + (mxfp8_blockscale_align - 1) * nonzero_experts
            max_blockscale = (
                (max_total + mxfp8_blockscale_align - 1) // mxfp8_blockscale_align
            ) * mxfp8_blockscale_align

            running_state["max_blockscale"] = max_blockscale

        blockscale_offsets = None
        if use_mxfp8 and (es_up or es_down):
            blockscale_offsets = torch.empty(
                (num_experts + 1,), dtype=torch.int32, device=device
            )

        prepare_moe_input(
            topk_ids,
            expert_offsets,
            problem_sizes1,
            problem_sizes2,
            a_map,
            c_map,
            num_experts,
            n,
            k,
            blockscale_offsets,
        )

        if use_mxfp8 and es_up:
            rep_a = shuffle_rows(a, a_map, (m * topk, k))
            rep_a_q = torch.empty_like(rep_a, dtype=torch.float8_e4m3fn)
            rep_a1_scales = torch.empty(
                (max_blockscale, k // 32), dtype=torch.uint8, device=device
            )
            es_sm100_mxfp8_blockscaled_grouped_quant(
                rep_a,
                problem_sizes1,
                expert_offsets[:-1],
                blockscale_offsets[:-1],
                rep_a_q,
                rep_a1_scales,
            )
        else:
            a_q, a1_scale = sglang_per_token_group_quant_fp8(a, 128)
            rep_a_q = shuffle_rows(a_q, a_map, (m * topk, k))
            rep_a1_scales = shuffle_rows(a1_scale, a_map, (m * topk, int(k / 128)))

        # ----------------------------------------------------------
        # Return the prepared input data
        gateup_input = rep_a_q  # The quantized and shuffled input
        rep_aux = rep_a1_scales  # The scales for the quantized input

        running_state["blockscale_offsets"] = blockscale_offsets

        running_state["c_map"] = c_map

        return CutlassRunnerInput(
            gate_up_input=gateup_input,
            a_map=a_map,
            rep_aux=rep_aux,
        )

    elif quant_info.params.quant_type == CutlassMoEQuantType.BlockscaledFP4:
        # FP4-specific asserts
        assert quant_info.w13_weight.dtype == torch.uint8, "weight 1 must be uint8"
        assert quant_info.w2_weight.dtype == torch.uint8, "weight 2 must be uint8"
        assert (
            quant_info.w13_weight.ndim == 3
            and quant_info.w2_weight.ndim == 3
            and quant_info.w1_blockscale.ndim == 3
            and quant_info.w2_blockscale.ndim == 3
        ), "All Weights must be of rank 3 for cutlass_moe_fp4"

        m_a, k_a = hidden_states.shape
        e_w1, nx2_w1, half_k_w1 = quant_info.w13_weight.shape
        e_w2, k_w2, half_n_w2 = quant_info.w2_weight.shape

        assert e_w1 == e_w2 and e_w1 == quant_info.params.num_experts, (
            "Number of experts must match",
            " between weights.",
        )
        assert (
            k_a // 2 == half_k_w1 and quant_info.params.hidden_size == k_w2
        ), "Hidden size mismatch between a, w1 and w2"
        assert (
            nx2_w1 == quant_info.params.intermediate_size_per_partition * 2
            and half_n_w2 == quant_info.params.intermediate_size_per_partition // 2
        ), ("mismatch in " "expected `n`")
        assert 2 * half_k_w1 == k_w2, "Hidden size mismatch w2 and w1"
        assert hidden_states.dtype in [
            torch.half,
            torch.bfloat16,
        ], "Invalid input dtype"

        params = quant_info.params

        prepare_moe_input(
            topk_ids,
            params.expert_offsets,
            params.problem_sizes1,
            params.problem_sizes2,
            a_map,
            c_map,
            params.num_experts,
            params.intermediate_size_per_partition,
            params.hidden_size,
            params.blockscale_offsets,
        )

        gateup_input, rep_a_blockscale = scaled_fp4_experts_quant(
            hidden_states,
            quant_info.w13_input_scale,
            params.expert_offsets,
            params.blockscale_offsets,
            topk_ids.shape[1],
            expert_map=a_map,
        )

        rep_aux = rep_a_blockscale

    elif quant_info.params.quant_type == CutlassMoEQuantType.W4A8:
        # w4a8-specific checks
        assert quant_info.w13_weight.dtype == torch.int8
        assert quant_info.w2_weight.dtype == torch.int8
        assert (
            dispatch_output.hidden_states.shape[1] // 2
            == quant_info.w13_weight.shape[2]
        ), "Hidden size mismatch w1"  # divide by two due to byte packing
        assert (
            quant_info.w13_weight.shape[2] * 2 == quant_info.w2_weight.shape[1]
        ), "Hidden size mismatch w2"  # multiply by two due to byte packing
        assert (
            quant_info.params.a_strides1.shape[0] == quant_info.w13_weight.shape[0]
        ), "A Strides 1 expert number mismatch"
        assert (
            quant_info.params.b_strides1.shape[0] == quant_info.w13_weight.shape[0]
        ), "B Strides 1 expert number mismatch"
        assert (
            quant_info.params.a_strides2.shape[0] == quant_info.w2_weight.shape[0]
        ), "A Strides 2 expert number mismatch"
        assert (
            quant_info.params.b_strides2.shape[0] == quant_info.w2_weight.shape[0]
        ), "B Strides 2 expert number mismatch"
        from sglang.srt.distributed.parallel_state import (
            get_moe_expert_parallel_world_size,
        )
        from sglang.srt.layers.moe.ep_moe.kernels import (
            cutlass_w4_run_moe_ep_preproess,
            pre_reorder_for_cutlass_moe,
        )

        num_local_experts = quant_info.w13_weight.size(0)
        m = hidden_states.size(0)
        k = quant_info.w13_weight.size(2) * 2  # w1_q is transposed and packed
        n = quant_info.w2_weight.size(2) * 2  # w2_q is transposed and packed
        topk = topk_ids.size(1)

        running_state["num_local_experts"] = num_local_experts

        if runner_config.apply_router_weight_on_input:
            assert (
                topk == 1
            ), "apply_router_weight_on_input is only implemented for topk=1"

        if get_moe_expert_parallel_world_size() > 1:
            topk_ids = torch.where(topk_ids == -1, num_local_experts, topk_ids)

        src2dst = cutlass_w4_run_moe_ep_preproess(topk_ids)
        running_state["src2dst"] = src2dst

        gateup_input = torch.empty(
            (m * topk, k),
            device=device,
            dtype=torch.float8_e4m3fn,
        )

        pre_reorder_for_cutlass_moe(
            hidden_states,
            gateup_input,
            src2dst,
            topk_ids,
            quant_info.w13_input_scale,
            num_local_experts,
            topk,
            m,
            k,
        )

        # NOTE: a_map and c_map are not used in the get_cutlass_w4a8_moe_mm_data kernel,
        # they are kept to allow for a quick switch of the permutation logic
        # from the current triton kernel implementation to the cutlass-based one if needed.

        get_cutlass_w4a8_moe_mm_data(
            topk_ids,
            quant_info.params.expert_offsets,
            quant_info.params.problem_sizes1,
            quant_info.params.problem_sizes2,
            a_map,
            c_map,
            num_local_experts,
            n,
            k,
        )

        rep_aux = None

        running_state["c_map"] = c_map

    return CutlassRunnerInput(
        gate_up_input=gateup_input,
        a_map=a_map,
        rep_aux=rep_aux,
    )


@register_post_permute("cutlass", "standard")
def post_permute_cutlass_to_standard(
    runner_output: CutlassRunnerOutput,
    quant_info: CutlassMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: Dict[str, torch.Tensor],
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    down_output = runner_output.hidden_states
    quant_type = quant_info.params.quant_type

    # Optional no-combine path: return (M, topk, K) without reduction
    if runner_config.no_combine:
        num_tokens = running_state["topk_ids"].shape[0]
        topk = running_state["c_map"].shape[0] // num_tokens
        hidden_size = down_output.shape[1]
        reordered = shuffle_rows(
            down_output,
            running_state["c_map"],
            (num_tokens * topk, hidden_size),
        ).view(num_tokens, topk, hidden_size)
        if not runner_config.apply_router_weight_on_input:
            reordered.mul_(
                running_state["topk_weights"]
                .view(num_tokens, topk, 1)
                .to(reordered.dtype)
            )

        down_output = reordered

    if quant_type == CutlassMoEQuantType.BlockscaledFP8:
        num_tokens = running_state["topk_ids"].shape[0]
        hidden_size = down_output.shape[1]
        result = torch.empty(
            (num_tokens, hidden_size),
            device=down_output.device,
            dtype=down_output.dtype,
        )
        apply_shuffle_mul_sum(
            down_output,
            result,
            running_state["c_map"],
            running_state["topk_weights"].to(result.dtype),
        )
        hidden_states = result

        if runner_config.routed_scaling_factor is not None:
            hidden_states.mul_(runner_config.routed_scaling_factor)
    elif quant_type == CutlassMoEQuantType.BlockscaledFP4:
        num_tokens = running_state["topk_ids"].shape[0]
        topk = running_state["c_map"].shape[0] // num_tokens
        hidden_size = down_output.shape[1]

        reordered = shuffle_rows(
            down_output,
            running_state["c_map"],
            (num_tokens * topk, hidden_size),
        )
        reordered = reordered.view(num_tokens, topk, hidden_size)
        if not runner_config.apply_router_weight_on_input:
            reordered.mul_(
                running_state["topk_weights"]
                .view(num_tokens, topk, 1)
                .to(reordered.dtype)
            )
        hidden_states = reordered.sum(dim=1).to(down_output.dtype)

        if runner_config.routed_scaling_factor is not None:
            hidden_states.mul_(runner_config.routed_scaling_factor)
    elif quant_type == CutlassMoEQuantType.W4A8:
        from sglang.srt.layers.moe.ep_moe.kernels import post_reorder_for_cutlass_moe

        num_tokens = running_state["topk_ids"].shape[0]
        hidden_size = down_output.shape[1]
        topk = running_state["topk_ids"].size(1)

        result = torch.empty(
            (num_tokens, hidden_size),
            device=down_output.device,
            dtype=down_output.dtype,
        )

        post_reorder_for_cutlass_moe(
            down_output,
            result,
            running_state["src2dst"],
            running_state["topk_ids"],
            running_state["topk_weights"],
            running_state["num_local_experts"],
            topk,
            num_tokens,
            hidden_size,
            runner_config.routed_scaling_factor or 1.0,
        )

        hidden_states = result

    return StandardCombineInput(hidden_states=hidden_states)


@register_pre_permute("deepep_ll", "cutlass")
def pre_permute_deepep_ll_to_cutlass(
    dispatch_output: DeepEPLLDispatchOutput,
    quant_info: CutlassMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: Dict[str, torch.Tensor],
) -> CutlassRunnerInput:
    # Common validation assertions for all CUTLASS MoE types
    assert (
        dispatch_output.topk_weights.shape == dispatch_output.topk_ids.shape
    ), "topk shape mismatch"
    assert (
        quant_info.w13_weight.shape[2] * 2 == quant_info.w2_weight.shape[1]
    ), "Hidden size mismatch w2"
    assert (
        quant_info.w13_weight.shape[0] == quant_info.w2_weight.shape[0]
    ), "Expert number mismatch"
    assert (
        quant_info.w13_weight.shape[0] == quant_info.w13_scale.shape[0]
    ), "w1 scales expert number mismatch"
    assert (
        quant_info.w13_weight.shape[0] == quant_info.w2_scale.shape[0]
    ), "w2 scales expert number mismatch"
    # w4a8-specific checks
    assert quant_info.w13_weight.dtype == torch.int8
    assert quant_info.w2_weight.dtype == torch.int8
    assert (
        quant_info.params.a_strides1.shape[0] == quant_info.w13_weight.shape[0]
    ), "A Strides 1 expert number mismatch"
    assert (
        quant_info.params.b_strides1.shape[0] == quant_info.w13_weight.shape[0]
    ), "B Strides 1 expert number mismatch"
    assert (
        quant_info.params.a_strides2.shape[0] == quant_info.w2_weight.shape[0]
    ), "A Strides 2 expert number mismatch"
    assert (
        quant_info.params.b_strides2.shape[0] == quant_info.w2_weight.shape[0]
    ), "B Strides 2 expert number mismatch"
    from sglang.srt.layers.moe.ep_moe.kernels import (
        deepep_ll_get_cutlass_w4a8_moe_mm_data,
    )

    # deep_ep_ll-specific checks
    assert (
        dispatch_output.hidden_states.shape[2] // 2 == quant_info.w13_weight.shape[2]
    ), "Hidden size mismatch w1"

    hidden_states, _, topk_ids, topk_weights, masked_m, expected_m = dispatch_output

    # Store for post_permute
    running_state["topk_weights"] = topk_weights
    running_state["topk_ids"] = topk_ids

    num_experts = quant_info.w13_weight.size(0)
    k = quant_info.w13_weight.size(2) * 2  # w1_q is transposed and packed
    n = quant_info.w2_weight.size(2) * 2  # w2_q is transposed and packed

    device = hidden_states.device

    quant_info.params.problem_sizes1, quant_info.params.problem_sizes2 = (
        deepep_ll_get_cutlass_w4a8_moe_mm_data(
            masked_m,
            quant_info.params.problem_sizes1,
            quant_info.params.problem_sizes2,
            num_experts,
            n,
            k,
        )
    )

    gateup_input = torch.empty(
        hidden_states.shape, dtype=torch.float8_e4m3fn, device=device
    )
    sgl_per_tensor_quant_fp8(
        hidden_states, gateup_input, quant_info.w13_input_scale.float(), True
    )

    return CutlassRunnerInput(
        gate_up_input=gateup_input,
        masked_m=masked_m,
        expected_m=expected_m,
    )


@register_post_permute("cutlass", "deepep_ll")
def post_permute_cutlass_to_deepep_ll(
    runner_output: CutlassRunnerOutput,
    quant_info: CutlassMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: Dict[str, torch.Tensor],
) -> DeepEPLLCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLCombineInput

    return DeepEPLLCombineInput(
        hidden_states=runner_output.hidden_states,
        topk_ids=running_state["topk_ids"],
        topk_weights=running_state["topk_weights"],
    )


@register_pre_permute("deepep_normal", "cutlass")
def pre_permute_deepep_normal_to_cutlass(
    dispatch_output: DeepEPNormalDispatchOutput,
    quant_info: CutlassMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: Dict[str, torch.Tensor],
) -> CutlassRunnerInput:
    # Common validation assertions for all CUTLASS MoE types
    assert (
        dispatch_output.topk_weights.shape == dispatch_output.topk_ids.shape
    ), "topk shape mismatch"
    assert (
        quant_info.w13_weight.shape[2] * 2 == quant_info.w2_weight.shape[1]
    ), "Hidden size mismatch w2"
    assert (
        quant_info.w13_weight.shape[0] == quant_info.w2_weight.shape[0]
    ), "Expert number mismatch"
    assert (
        quant_info.w13_weight.shape[0] == quant_info.w13_scale.shape[0]
    ), "w1 scales expert number mismatch"
    assert (
        quant_info.w13_weight.shape[0] == quant_info.w2_scale.shape[0]
    ), "w2 scales expert number mismatch"
    # w4a8-specific checks
    assert quant_info.w13_weight.dtype == torch.int8
    assert quant_info.w2_weight.dtype == torch.int8
    assert (
        quant_info.params.a_strides1.shape[0] == quant_info.w13_weight.shape[0]
    ), "A Strides 1 expert number mismatch"
    assert (
        quant_info.params.b_strides1.shape[0] == quant_info.w13_weight.shape[0]
    ), "B Strides 1 expert number mismatch"
    assert (
        quant_info.params.a_strides2.shape[0] == quant_info.w2_weight.shape[0]
    ), "A Strides 2 expert number mismatch"
    assert (
        quant_info.params.b_strides2.shape[0] == quant_info.w2_weight.shape[0]
    ), "B Strides 2 expert number mismatch"
    from sglang.srt.layers.moe.ep_moe.kernels import (
        deepep_permute_triton_kernel,
        deepep_run_moe_deep_preprocess,
    )

    # deep_ep_normal-specific checks
    assert (
        dispatch_output.hidden_states.shape[1] // 2 == quant_info.w13_weight.shape[2]
    ), "Hidden size mismatch w1"

    hidden_states, _, topk_ids_, topk_weights, _ = dispatch_output

    # Store state for post_permute
    running_state["topk_weights"] = topk_weights
    running_state["topk_ids"] = topk_ids_

    num_experts = quant_info.w13_weight.size(0)
    k = quant_info.w13_weight.size(2) * 2  # w1_q is transposed and packed
    n = quant_info.w2_weight.size(2) * 2  # w2_q is transposed and packed
    topk = topk_ids_.size(1)
    device = hidden_states.device

    reorder_topk_ids, src2dst, _ = deepep_run_moe_deep_preprocess(
        topk_ids_, num_experts
    )
    num_total_tokens = reorder_topk_ids.numel()
    gateup_input_pre_reorder = torch.empty(
        (int(num_total_tokens), hidden_states.shape[1]),
        device=device,
        dtype=hidden_states.dtype,
    )

    deepep_permute_triton_kernel[(hidden_states.shape[0],)](
        hidden_states,
        gateup_input_pre_reorder,
        src2dst,
        topk_ids_.to(torch.int64),
        None,
        topk,
        hidden_states.shape[1],
        BLOCK_SIZE=512,
    )
    gateup_input = torch.empty(
        gateup_input_pre_reorder.shape, dtype=torch.float8_e4m3fn, device=device
    )
    sgl_per_tensor_quant_fp8(
        gateup_input_pre_reorder, gateup_input, quant_info.w13_input_scale.float(), True
    )
    del gateup_input_pre_reorder
    local_topk_ids = topk_ids_
    local_topk_ids = (
        torch.where(local_topk_ids == -1, num_experts, topk_ids_).to(torch.int32)
    ).contiguous()

    a_map = torch.empty((local_topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((local_topk_ids.numel()), dtype=torch.int32, device=device)

    quant_info.params.problem_sizes1.zero_()
    quant_info.params.problem_sizes2.zero_()
    # Safely zero offsets too if the kernel calculates them via cumsum
    quant_info.params.expert_offsets.zero_()

    get_cutlass_w4a8_moe_mm_data(
        local_topk_ids,
        quant_info.params.expert_offsets,
        quant_info.params.problem_sizes1,
        quant_info.params.problem_sizes2,
        a_map,
        c_map,
        num_experts,
        n,
        k,
    )

    # Store additional state for the main function
    running_state["src2dst"] = src2dst
    running_state["c_map"] = c_map

    return CutlassRunnerInput(
        gate_up_input=gateup_input,
        a_map=a_map,
    )


@register_post_permute("cutlass", "deepep_normal")
def post_permute_cutlass_to_deepep_normal(
    runner_output: CutlassRunnerOutput,
    quant_info: CutlassMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: Dict[str, torch.Tensor],
) -> DeepEPNormalCombineInput:
    from sglang.srt.layers.moe.ep_moe.kernels import deepep_post_reorder_triton_kernel
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPNormalCombineInput

    c2 = runner_output.hidden_states
    src2dst = running_state["src2dst"]
    topk_ids_ = running_state["topk_ids"]
    topk_weights = running_state["topk_weights"]
    topk = topk_ids_.size(1)
    hidden_size = c2.shape[1]  # Use c2.shape[1] directly

    num_tokens = src2dst.shape[0] // topk
    output = torch.empty(
        (num_tokens, hidden_size),
        device=c2.device,
        dtype=torch.bfloat16,
    )

    deepep_post_reorder_triton_kernel[(num_tokens,)](
        c2,
        output,
        src2dst,
        topk_ids_,
        topk_weights,
        topk,
        hidden_size,
        BLOCK_SIZE=512,
    )

    return DeepEPNormalCombineInput(
        hidden_states=output,
        topk_ids=topk_ids_,
        topk_weights=topk_weights,
    )
