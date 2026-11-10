from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.moe.sm70_nvfp4_moe_decode import sm70_nvfp4_moe_decode_max_rows
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    RunnerInput,
    RunnerOutput,
    register_fused_func,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


@triton.jit
def _unpack_packed_topk_kernel(packed_ptr, ids_ptr, w_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    packed = tl.load(
        packed_ptr + offs, mask=mask
    )  # int32 (id << 16) | bf16-weight-bits
    tl.store(ids_ptr + offs, packed >> 16, mask=mask)  # expert id (high 16 bits)
    wbits = (packed & 0xFFFF).to(tl.int16)  # bf16 weight bits (low 16 bits)
    tl.store(
        w_ptr + offs, wbits.to(tl.bfloat16, bitcast=True).to(tl.float32), mask=mask
    )


def _fused_unpack_packed_topk(packed: torch.Tensor):
    """Single-launch inverse of the fused topk pack ((id << 16) | bf16-weight-bits).

    Returns (topk_ids int32, topk_weights float32). Collapses the ~5 elementwise
    ops (shift / mask / int16 / bitcast / cast) the torch reference emits per call
    into one Triton launch. Mirrors _pack_topk_kernel (ops/lora/moe/trtllm_lora_temp/topk_pack).
    """
    packed = packed.contiguous()
    ids = torch.empty_like(packed, dtype=torch.int32)
    w = torch.empty(packed.shape, dtype=torch.float32, device=packed.device)
    numel = packed.numel()
    if numel:
        BLOCK = 1024
        _unpack_packed_topk_kernel[(triton.cdiv(numel, BLOCK),)](
            packed, ids, w, numel, BLOCK=BLOCK
        )
    return ids, w


@dataclass
class MarlinRunnerInput(RunnerInput):
    """Input bundle passed to the Marlin runner core."""

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.MARLIN


@dataclass
class MarlinRunnerOutput(RunnerOutput):
    """Output bundle returned from the Marlin runner core."""

    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.MARLIN


@dataclass
class MarlinMoeQuantInfo(MoeQuantInfo):
    """Quantization payload consumed by the Marlin backend."""

    w13_qweight: torch.Tensor
    w2_qweight: torch.Tensor
    w13_scales: torch.Tensor
    w2_scales: torch.Tensor
    w13_g_idx_sort_indices: Optional[torch.Tensor]
    w2_g_idx_sort_indices: Optional[torch.Tensor]
    weight_bits: int

    # GPTQ specific (Optional)
    w13_g_idx: Optional[torch.Tensor] = None
    w2_g_idx: Optional[torch.Tensor] = None
    is_k_full: bool = True

    # AWQ specific (Optional)
    w13_qzeros: Optional[torch.Tensor] = None
    w2_qzeros: Optional[torch.Tensor] = None

    # Optional
    expert_map: Optional[torch.Tensor] = None
    global_num_experts: int = -1
    # ModelOpt NVFP4 specific (one FP32 scale per expert).
    w13_global_scale: Optional[torch.Tensor] = None
    w2_global_scale: Optional[torch.Tensor] = None
    w13_bias: Optional[torch.Tensor] = None
    w2_bias: Optional[torch.Tensor] = None


@register_fused_func("none", "marlin")
def fused_experts_none_to_marlin(
    dispatch_output: StandardDispatchOutput,
    quant_info: MarlinMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output

    # The fused gate+topk kernel (SGLANG_OPT_USE_FUSED_GATE_TOPK) emits a
    # PackedTopKOutput -- int32 (expert_id << 16) | bf16-weight-bits -- for the
    # FlashInfer trtllm routed kernel, which unpacks device-side. The Marlin
    # runner reads topk_ids / topk_weights separately, so unpack it here.
    if not hasattr(topk_output, "topk_weights"):
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        topk_ids, topk_weights = _fused_unpack_packed_topk(topk_output.packed_topk_ids)
        topk_output = StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=topk_output.router_logits,
        )

    # Qwen3.8 Flash Next's TP4 decode shape is too skinny for Marlin's
    # block-padded grouped GEMM: M<=4 routes ten rows through I=160 experts.
    # On Volta, stream the existing Marlin-repacked NVFP4 tensors directly and
    # split K widely enough to occupy all 80 SMs. Keep Marlin for prefill and
    # every other model/shape.
    routed_scale = runner_config.routed_scaling_factor
    use_sm70_nvfp4_decode = (
        hidden_states.dtype == torch.float16
        and hidden_states.ndim == 2
        and 1 <= hidden_states.shape[0] <= sm70_nvfp4_moe_decode_max_rows()
        and hidden_states.shape[1] == 2560
        and tuple(topk_output.topk_ids.shape) == (hidden_states.shape[0], 10)
        and tuple(quant_info.w13_qweight.shape) == (512, 160, 640)
        and tuple(quant_info.w2_qweight.shape) == (512, 10, 5120)
        and quant_info.weight_bits == 4
        and quant_info.w13_qzeros is None
        and quant_info.w2_qzeros is None
        and quant_info.w13_scales.dtype == torch.float8_e4m3fn
        and quant_info.w2_scales.dtype == torch.float8_e4m3fn
        and quant_info.w13_global_scale is not None
        and quant_info.w2_global_scale is not None
        and quant_info.expert_map is None
        and runner_config.num_experts == runner_config.num_local_experts == 512
        and runner_config.swiglu_limit is None
        and runner_config.activation == "silu"
        and runner_config.is_gated
        and not runner_config.apply_router_weight_on_input
        and not runner_config.no_combine
        and runner_config.gemm1_alpha is None
        and runner_config.gemm1_beta is None
        and runner_config.gemm1_clamp_limit is None
        and runner_config.gate_up_input_scale == 1.0
        and runner_config.wide_output_scale == 1.0
        and (routed_scale is None or routed_scale == 1.0)
    )
    if use_sm70_nvfp4_decode:
        from sglang.kernels.ops.moe.sm70_nvfp4_moe_decode import (
            sm70_nvfp4_moe_decode,
            sm70_nvfp4_moe_decode_available,
        )

        if sm70_nvfp4_moe_decode_available():
            output = sm70_nvfp4_moe_decode(
                hidden_states,
                quant_info.w13_qweight,
                quant_info.w2_qweight,
                quant_info.w13_scales,
                quant_info.w2_scales,
                quant_info.w13_global_scale,
                quant_info.w2_global_scale,
                topk_output.topk_ids.view(-1),
                topk_output.topk_weights.view(-1),
            )
            return StandardCombineInput(hidden_states=output)

    # GLM-5.3-Flash: hidden 4096, top-8; EP or TP-sliced experts. Load replaced
    # the Marlin expert pack with the lane-major HMMA pack (same bytes). Decode
    # reads that pack. Prefill unpacks it back into the shared scratch so
    # Marlin still sees its own layout.
    from sglang.kernels.ops.moe.sm70_glm_nvfp4_moe_decode import (
        glm_hmma_experts_packed,
    )

    w13_qweight = quant_info.w13_qweight
    w2_qweight = quant_info.w2_qweight
    glm_hmma_packed = (
        hidden_states.dtype == torch.float16
        and hidden_states.ndim == 2
        and hidden_states.shape[1] == 4096
        and tuple(topk_output.topk_ids.shape) == (hidden_states.shape[0], 8)
        and glm_hmma_experts_packed(w13_qweight, w2_qweight)
        and quant_info.weight_bits == 4
        and quant_info.w13_qzeros is None
        and quant_info.w2_qzeros is None
        and quant_info.w13_bias is None
        and quant_info.w2_bias is None
        and quant_info.w13_scales.dtype == torch.float8_e4m3fn
        and quant_info.w2_scales.dtype == torch.float8_e4m3fn
        and quant_info.w13_global_scale is not None
        and quant_info.w2_global_scale is not None
        and runner_config.activation == "silu"
        and runner_config.is_gated
        and not runner_config.apply_router_weight_on_input
        and not runner_config.no_combine
        and runner_config.gemm1_alpha is None
        and runner_config.gemm1_beta is None
        and runner_config.gemm1_clamp_limit is None
        and runner_config.gate_up_input_scale == 1.0
        and runner_config.wide_output_scale == 1.0
    )
    if glm_hmma_packed:
        from sglang.kernels.ops.moe.sm70_glm_nvfp4_moe_decode import (
            materialize_glm_marlin,
            sm70_glm_nvfp4_moe_decode,
            sm70_glm_nvfp4_moe_decode_available,
        )

        if sm70_glm_nvfp4_moe_decode_available() and 1 <= hidden_states.shape[0] <= 4:
            from sglang.srt.layers.moe.sm70_shared_expert_fold import (
                take_shared_expert,
            )

            output = sm70_glm_nvfp4_moe_decode(
                hidden_states,
                w13_qweight,
                w2_qweight,
                quant_info.w13_scales,
                quant_info.w2_scales,
                quant_info.w13_global_scale,
                quant_info.w2_global_scale,
                topk_output.topk_ids.reshape(-1),
                topk_output.topk_weights.reshape(-1),
                0.0
                if runner_config.swiglu_limit is None
                else float(runner_config.swiglu_limit),
                1.0 if routed_scale is None else float(routed_scale),
                shared=take_shared_expert(),
            )
            return StandardCombineInput(hidden_states=output)
        w13_qweight, w2_qweight = materialize_glm_marlin(w13_qweight, w2_qweight)

    # DeepSeek-V4.1-Flash decode: Marlin's M=1 grouped GEMM is still ~5x
    # above a bandwidth-bound GEMV. Stream the already-repacked MXFP4
    # tensors. Prefill M>>4 and every other shape stay on Marlin.
    from sglang.kernels.ops.moe.sm70_dsv41_mxfp4_moe_decode import (
        sm70_dsv41_mxfp4_moe_decode,
        sm70_dsv41_mxfp4_moe_decode_eligible,
    )

    if sm70_dsv41_mxfp4_moe_decode_eligible(
        hidden_states,
        quant_info.w13_qweight,
        quant_info.w2_qweight,
        quant_info.w13_scales,
        quant_info.w2_scales,
        topk_output.topk_ids,
        weight_bits=quant_info.weight_bits,
        w13_qzeros=quant_info.w13_qzeros,
        w2_qzeros=quant_info.w2_qzeros,
        w13_global_scale=quant_info.w13_global_scale,
        w2_global_scale=quant_info.w2_global_scale,
        w13_bias=quant_info.w13_bias,
        w2_bias=quant_info.w2_bias,
        is_gated=runner_config.is_gated,
        activation=runner_config.activation,
        gate_up_input_scale=runner_config.gate_up_input_scale,
        wide_output_scale=runner_config.wide_output_scale,
        gemm1_alpha=runner_config.gemm1_alpha,
        gemm1_clamp_limit=runner_config.gemm1_clamp_limit,
        swiglu_limit=runner_config.swiglu_limit,
        apply_router_weight_on_input=runner_config.apply_router_weight_on_input,
        no_combine=runner_config.no_combine,
    ):
        output = sm70_dsv41_mxfp4_moe_decode(
            hidden_states,
            quant_info.w13_qweight,
            quant_info.w2_qweight,
            quant_info.w13_scales,
            quant_info.w2_scales,
            topk_output.topk_ids,
            topk_output.topk_weights,
            swiglu_limit=runner_config.swiglu_limit,
        )
        return StandardCombineInput(hidden_states=output)

    if runner_config.apply_router_weight_on_input or runner_config.no_combine:
        raise NotImplementedError(
            "Marlin MoE runner does not implement apply_router_weight_on_input "
            "or no_combine; fused_marlin_moe would need to scale the input rows by the "
            "router weight / return uncombined per-expert outputs."
        )
    # The alpha activation computes (up + 1), as Triton does, so beta 1.0 is exact.
    if runner_config.gemm1_beta not in (None, 1.0):
        raise NotImplementedError(
            "Marlin MoE runner only implements gemm1_beta=1.0, "
            f"got {runner_config.gemm1_beta}."
        )
    if (
        runner_config.gemm1_clamp_limit is not None
        and runner_config.gemm1_alpha is None
    ):
        raise NotImplementedError(
            "Marlin MoE runner does not implement a limit-only gemm1_clamp_limit; "
            "fused_marlin_moe clamps gate before silu, Triton clamps silu(gate) after it."
        )

    if runner_config.is_gated:
        assert runner_config.activation in {
            "silu",
            "situ",
        }, f"Only gated SiLU/SiTU is supported, got {runner_config.activation}."
    elif runner_config.activation not in {"silu", "relu2"}:
        raise ValueError(
            f"Unsupported Marlin MoE activation: {runner_config.activation}"
        )

    # Marlin always applies the router weight inside its down GEMM.
    assert (
        not runner_config.no_combine or runner_config.no_combine_keep_router_weight
    ), "Marlin MoE supports no_combine only with no_combine_keep_router_weight"

    # Use a per-call workspace so captured graphs cannot alias Marlin's
    # inter-block reduction locks and deadlock during capture.
    workspace = marlin_make_workspace(hidden_states.device, max_blocks_per_sm=4)

    marlin_hidden_states = hidden_states
    # Avoid aliasing the MoE input buffer until Marlin output semantics are
    # fully validated across shared-expert and overlap paths.
    marlin_inplace = False
    if (
        quant_info.weight_bits == 4
        and quant_info.w13_qzeros is None
        and quant_info.w2_qzeros is None
        and quant_info.w13_scales.dtype == torch.float8_e8m0fnu
        and quant_info.w2_scales.dtype == torch.float8_e8m0fnu
        and hidden_states.dtype == torch.float16
    ):
        # Hopper sgl-kernel never instantiates FP16+E8M0, so upcast to BF16.
        # SM70 marlin_v100 *does* run FP16+E8M0; a BF16 upcast on V100 is
        # wrong (no BF16 tensor cores) and would hide the real kernel path.
        is_sm70 = torch.cuda.get_device_capability(hidden_states.device)[0] == 7
        if not is_sm70:
            marlin_hidden_states = hidden_states.to(torch.bfloat16)
            marlin_inplace = False

    output = fused_marlin_moe(
        hidden_states=marlin_hidden_states,
        w1=w13_qweight,
        w2=w2_qweight,
        w1_scale=quant_info.w13_scales,
        w2_scale=quant_info.w2_scales,
        gating_output=topk_output.router_logits,
        topk_weights=topk_output.topk_weights,
        topk_ids=topk_output.topk_ids,
        global_num_experts=quant_info.global_num_experts,
        expert_map=quant_info.expert_map,
        is_expert_parallel=(
            runner_config.num_experts != runner_config.num_local_experts
        ),
        g_idx1=quant_info.w13_g_idx,
        g_idx2=quant_info.w2_g_idx,
        sort_indices1=quant_info.w13_g_idx_sort_indices,
        sort_indices2=quant_info.w2_g_idx_sort_indices,
        w1_zeros=quant_info.w13_qzeros,
        w2_zeros=quant_info.w2_qzeros,
        w1_global_scale=quant_info.w13_global_scale,
        w2_global_scale=quant_info.w2_global_scale,
        w1_bias=quant_info.w13_bias,
        w2_bias=quant_info.w2_bias,
        workspace=workspace,
        num_bits=quant_info.weight_bits,
        is_k_full=quant_info.is_k_full,
        inplace=marlin_inplace,
        routed_scaling_factor=runner_config.routed_scaling_factor,
        clamp_limit=(
            runner_config.gemm1_clamp_limit
            if runner_config.gemm1_alpha is not None
            else runner_config.swiglu_limit
        ),
        gemm1_alpha=runner_config.gemm1_alpha,
        activation=runner_config.activation,
        is_gated=runner_config.is_gated,
        no_combine=runner_config.no_combine,
        gate_up_input_scale=runner_config.gate_up_input_scale,
        wide_output_scale=runner_config.wide_output_scale,
    ).to(hidden_states.dtype)

    return StandardCombineInput(
        hidden_states=output,
    )


# ===== TO BE REFACTORED ====
from sglang.srt.lora.marlin_lora_temp import sgl_backend  # noqa: E402,F401

# ===== END TO BE REFACTORED ====
