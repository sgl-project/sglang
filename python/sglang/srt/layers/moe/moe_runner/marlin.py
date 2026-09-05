from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

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
        CombineInput,
        DeepEPLLDispatchOutput,
        DeepEPNormalDispatchOutput,
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
    into one Triton launch. Mirrors _pack_topk_kernel (trtllm_lora_temp/topk_pack).
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
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

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

    return StandardCombineInput(
        hidden_states=_run_marlin(
            hidden_states,
            topk_output.topk_ids,
            topk_output.topk_weights,
            quant_info,
            runner_config,
            # Standard EP dispatch already maps non-local routes to -1.
            is_ep=(
                runner_config.num_local_experts is not None
                and runner_config.num_experts is not None
                and runner_config.num_local_experts < runner_config.num_experts
            ),
        )
    )


def _run_marlin(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    quant_info: MarlinMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    *,
    is_ep: bool = False,
    expected_m: Optional[int] = None,
    expert_num_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace

    if hidden_states.shape[0] == 0:
        return torch.empty_like(hidden_states)

    if runner_config.is_gated:
        assert runner_config.activation in {
            "silu",
            "situ",
        }, f"Only gated SiLU/SiTU is supported, got {runner_config.activation}."
    elif runner_config.activation not in {"silu", "relu2"}:
        raise ValueError(
            f"Unsupported Marlin MoE activation: {runner_config.activation}"
        )

    # Use a per-call workspace so captured graphs cannot alias Marlin's
    # inter-block reduction locks and deadlock during capture.
    workspace = marlin_make_workspace(hidden_states.device, max_blocks_per_sm=4)

    marlin_hidden_states = hidden_states
    if (
        quant_info.weight_bits == 4
        and quant_info.w13_qzeros is None
        and quant_info.w2_qzeros is None
        and quant_info.w13_scales.dtype == torch.float8_e8m0fnu
        and quant_info.w2_scales.dtype == torch.float8_e8m0fnu
        and hidden_states.dtype == torch.float16
    ):
        # MXFP4(E8M0) Marlin kernels are only numerically valid on the bf16
        # activation path. The fp16 + E8M0 path is intentionally not generated
        # in sgl-kernel, so upcast activations here and cast the result back.
        marlin_hidden_states = hidden_states.to(torch.bfloat16)

    output = fused_marlin_moe(
        hidden_states=marlin_hidden_states,
        w1=quant_info.w13_qweight,
        w2=quant_info.w2_qweight,
        w1_scale=quant_info.w13_scales,
        w2_scale=quant_info.w2_scales,
        gating_output=None,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        global_num_experts=quant_info.global_num_experts,
        expert_map=quant_info.expert_map,
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
        inplace=False,
        routed_scaling_factor=runner_config.routed_scaling_factor,
        clamp_limit=(
            runner_config.gemm1_clamp_limit
            if runner_config.gemm1_alpha is not None
            else runner_config.swiglu_limit
        ),
        gemm1_alpha=runner_config.gemm1_alpha,
        activation=runner_config.activation,
        is_gated=runner_config.is_gated,
        is_ep=is_ep,
        expected_m=expected_m,
        expert_num_tokens=expert_num_tokens,
    ).to(hidden_states.dtype)

    return output


def validate_deepep_marlin(
    config: MoeRunnerConfig, *, lora_enabled: bool = False
) -> None:
    """Validate the resolved runner, including quantization's automatic choice."""
    from sglang.srt.arg_groups.model_override_base import resolved_view
    from sglang.srt.runtime_context import get_server_args
    from sglang.srt.utils import is_cuda

    if not is_cuda():
        raise ValueError("Marlin + DeepEP requires NVIDIA CUDA.")
    if config.runner_backend != MoeRunnerBackend.MARLIN:
        raise ValueError("DeepEP supports marlin, not experimental_sgl_marlin.")
    if config.params_dtype != torch.bfloat16:
        raise ValueError("Marlin + DeepEP requires --dtype bfloat16.")
    if lora_enabled:
        raise ValueError("Marlin + DeepEP does not support --enable-lora.")
    if config.num_fused_shared_experts:
        raise ValueError("Marlin + DeepEP requires separate shared-expert computation.")
    if (config.is_gated and config.activation not in {"silu", "situ"}) or (
        not config.is_gated and config.activation not in {"silu", "relu2"}
    ):
        raise ValueError(f"Unsupported Marlin activation: {config.activation}.")
    if config.apply_router_weight_on_input or config.no_combine:
        raise ValueError("Marlin + DeepEP requires output routing weights and combine.")

    view = resolved_view(get_server_args())
    for option in (
        "enable_lora",
        "enable_eplb",
        "elastic_ep_backend",
        "enable_single_batch_overlap",
        "enable_two_batch_overlap",
        "enable_waterfill",
        "ep_num_redundant_experts",
    ):
        if getattr(view, option):
            raise ValueError(
                f"Marlin + DeepEP does not support --{option.replace('_', '-')}."
            )
    if view.init_expert_location != "trivial":
        raise ValueError("Marlin + DeepEP requires --init-expert-location trivial.")
    if view.deepep_dispatcher_output_dtype not in {"auto", "bf16"}:
        raise ValueError(
            "Marlin + DeepEP requires --deepep-dispatcher-output-dtype bf16 or auto."
        )


@triton.jit
def _deepep_ll_copy_kernel(
    expert_rows,
    compact_rows,
    counts,
    ids,
    CAPACITY: tl.constexpr,
    HIDDEN: tl.constexpr,
    EXPERTS: tl.constexpr,
    PACK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    expert = tl.program_id(0)
    row = tl.program_id(1) * 8 + tl.arange(0, 8)
    experts = tl.arange(0, BLOCK_E)
    sizes = tl.load(counts + experts, experts < EXPERTS, 0)
    start = tl.sum(tl.where(experts < expert, sizes, 0))
    valid = (row < CAPACITY) & (row < tl.load(counts + expert))
    col = tl.arange(0, BLOCK_H)
    expert_offset = (expert * CAPACITY + row[:, None]) * HIDDEN + col[None, :]
    compact_offset = (start + row[:, None]) * HIDDEN + col[None, :]
    mask = valid[:, None] & (col[None, :] < HIDDEN)
    if PACK:
        value = tl.load(expert_rows + expert_offset, mask, 0)
        tl.store(compact_rows + compact_offset, value, mask)
        tl.store(ids + start + row, expert, valid)
    else:
        value = tl.load(compact_rows + compact_offset, mask, 0)
        # DeepEP combine reads only dispatched rows through its handle.
        tl.store(expert_rows + expert_offset, value, mask)


def _deepep_ll_pack(hidden_states: torch.Tensor, masked_m: torch.Tensor, topk: int):
    """Compact valid expert rows into a graph-safe worst-case routing capacity."""
    experts, capacity, hidden = hidden_states.shape
    # Capacity already includes all source ranks. A source token visits at
    # most topk experts, so allocating capacity for every expert wastes space.
    rows = capacity * min(experts, topk)
    compact = hidden_states.new_empty((rows, hidden))
    ids = torch.full((rows, 1), -1, device=hidden_states.device, dtype=torch.int32)
    weights = torch.ones((rows, 1), device=hidden_states.device, dtype=torch.float32)
    if capacity:
        _deepep_ll_copy_kernel[(experts, triton.cdiv(capacity, 8))](
            hidden_states,
            compact,
            masked_m,
            ids,
            capacity,
            hidden,
            experts,
            True,
            triton.next_power_of_2(hidden),
            triton.next_power_of_2(experts),
        )
    return compact, ids, weights


def _deepep_ll_unpack(
    compact: torch.Tensor, hidden_states: torch.Tensor, counts: torch.Tensor
):
    output = torch.empty_like(hidden_states)
    experts, capacity, hidden = hidden_states.shape
    if capacity:
        _deepep_ll_copy_kernel[(experts, triton.cdiv(capacity, 8))](
            output,
            compact,
            counts,
            counts,
            capacity,
            hidden,
            experts,
            False,
            triton.next_power_of_2(hidden),
            triton.next_power_of_2(experts),
        )
    return output


@register_fused_func("deepep", "marlin")
def fused_experts_deepep_to_marlin(
    dispatch_output: DeepEPNormalDispatchOutput | DeepEPLLDispatchOutput,
    quant_info: MarlinMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> CombineInput:
    from sglang.srt.layers.moe.token_dispatcher import (
        DeepEPLLCombineInput,
        DeepEPNormalCombineInput,
        DispatchOutputFormat,
    )

    hidden_states = dispatch_output.hidden_states
    if (
        hidden_states.dtype != torch.bfloat16
        or dispatch_output.hidden_states_scale is not None
    ):
        raise ValueError("Marlin + DeepEP expects BF16 activations without scales.")

    # Dispatch has already assigned local experts. A second global-to-local
    # mapping would silently select another rank's weights.
    quant_info = replace(quant_info, expert_map=None, global_num_experts=-1)
    if dispatch_output.format == DispatchOutputFormat.DEEPEP_NORMAL:
        output = _run_marlin(
            hidden_states,
            dispatch_output.topk_ids,
            dispatch_output.topk_weights,
            quant_info,
            runner_config,
            is_ep=True,
        )
        return DeepEPNormalCombineInput(
            output,
            dispatch_output.topk_ids,
            dispatch_output.topk_weights,
        )
    if dispatch_output.format == DispatchOutputFormat.DEEPEP_LL:
        compact, topk_ids, topk_weights = _deepep_ll_pack(
            hidden_states, dispatch_output.masked_m, dispatch_output.topk_ids.shape[1]
        )
        output = _run_marlin(
            compact,
            topk_ids,
            topk_weights,
            quant_info,
            runner_config,
            is_ep=True,
            expected_m=dispatch_output.expected_m,
            expert_num_tokens=dispatch_output.masked_m,
        )
        # DeepEP applies the original routing weights while combining. These
        # expert outputs were computed with unit weights, not the source top-k.
        return DeepEPLLCombineInput(
            _deepep_ll_unpack(output, hidden_states, dispatch_output.masked_m),
            dispatch_output.topk_ids,
            dispatch_output.topk_weights,
        )
    raise ValueError(f"Unsupported Marlin dispatch format: {dispatch_output.format}")


# ===== TO BE REFACTORED ====
from sglang.srt.lora.marlin_lora_temp import sgl_backend  # noqa: E402,F401

# ===== END TO BE REFACTORED ====
