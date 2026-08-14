"""b12x W4A16 fused MoE for SM120/SM121.

The FlashInfer CUTLASS MXFP4 path splits one expert layer into seven launches:
build-expert-maps, expand-input-rows, compute-strides, FC1, activation, FC2 and
finalize. b12x fuses FC1, the gated activation and FC2 into a single kernel and
gathers the routed rows by index instead of materializing a permuted copy, so a
layer costs one launch instead of seven. It is also W4A16 rather than W4A8:
activations stay bf16 instead of being quantized to MXFP8.

Weights are prepared once at load time (see ``Mxfp4B12xMoEMethod``) and the
scratch buffer is planned there too, so this module only binds views -- which is
what makes it safe to capture in a CUDA graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    register_fused_func,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


@dataclass
class B12xMoeQuantInfo(MoeQuantInfo):
    """Payload for the b12x W4A16 fused MoE.

    ``experts`` owns the only copy of the expert weights: the checkpoint tensors
    are released once it is built, so this dataclass carries none of its own.
    """

    # b12x expert-weight package, built at load time.
    experts: Any
    # b12x scratch plan, planned at load time.
    plan: Any
    # Caller-owned scratch, allocated at load time so it never lands in a CUDA
    # graph's private memory pool.
    scratch: torch.Tensor
    # Set when the quant method already bound everything it needs; the two b12x
    # generations take different arguments, so the version-specific call lives
    # there rather than here.
    launch: Any = None
    # True only when expert parallelism is on: that is the only case where the
    # dispatcher rewrites non-local experts (and padded rows) to -1, which b12x
    # would otherwise run as garbage. At EP=1 the guard would cost five tiny
    # kernels per layer for nothing -- torch.topk cannot produce negatives.
    ep_guard: bool = False


@register_fused_func("none", "b12x")
def fused_experts_none_to_b12x(
    dispatch_output: StandardDispatchOutput,
    quant_info: MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    """Run one expert layer through the b12x W4A16 fused kernel."""
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    assert isinstance(
        quant_info, B12xMoeQuantInfo
    ), f"Unexpected quant_info type for b12x: {type(quant_info)}"
    if runner_config.activation != "silu":
        raise NotImplementedError(
            f"b12x supports activation='silu', got {runner_config.activation!r}."
        )
    if runner_config.apply_router_weight_on_input:
        raise NotImplementedError(
            "b12x applies the router weights in the FC2 epilogue; "
            "apply_router_weight_on_input is not supported."
        )

    x = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    if TopKOutputChecker.format_is_bypassed(topk_output):
        topk_output = topk_output.to_standard()

    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
        out = torch.empty(
            x.shape[0], x.shape[-1], dtype=torch.bfloat16, device=x.device
        )

    # HashTopK already emits int32 ids and float32 weights, so these casts are
    # free no-ops kept as a contract check for other topk implementations.
    topk_ids = topk_output.topk_ids.to(torch.int32)
    topk_weights = topk_output.topk_weights.to(torch.float32)
    if quant_info.ep_guard:
        # With EP the dispatcher rewrites non-local experts -- and padded rows --
        # to -1. FlashInfer CUTLASS skips out-of-range ids; b12x instead runs
        # them and writes ~1e4-per-element garbage into those rows.
        invalid = topk_ids < 0
        topk_ids = topk_ids.clamp_min(0)
        topk_weights = topk_weights.masked_fill(invalid, 0.0)

    if quant_info.launch is not None:
        quant_info.launch(x, topk_ids, topk_weights, out)
    else:
        from b12x.moe import fused_moe

        binding = fused_moe.bind(
            quant_info.plan,
            scratch=quant_info.scratch,
            a=x,
            experts=quant_info.experts,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            output=out,
        )
        fused_moe.run(binding=binding)

    return StandardCombineInput(hidden_states=out)
