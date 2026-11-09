"""SAIL DeepGEMM pre-permute adapters.

The shared pre-permute turns a DeepEP dispatch buffer into the grouped GEMM
layout once, for every platform. SAIL's DeepEP wire format differs in two ways
the shared implementation cannot express:

* the activation scale block size has to be derived from the dispatched scale
  width (MXFP4 packs two values per byte, so its group covers 32 values), and
* tokens are scattered by ``ep_scatter_sail``, which pads to the per-expert
  valid counts and treats an INT8 activation as one whole-row quantization
  group.

Both variants register as platform-specific overrides of the same
``(dispatch format, runner backend)`` pair, so the shared implementation keeps
no SAIL branch and every other platform keeps using it unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe.moe_runner.base import (
    MoeRunnerConfig,
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.moe.moe_runner.deep_gemm import (
    DeepGemmMoeQuantInfo,
    DeepGemmRunnerInput,
    copy_list_to_gpu_no_ce,
)
from sglang.srt.layers.moe.moe_runner.deep_gemm import (
    post_permute_deep_gemm_to_deepep_normal as _shared_post_permute,
)
from sglang.srt.layers.offloader import get_offloader
from sglang.srt.runtime_context import get_exec
from sglang.srt.utils import dispose_tensor

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmRunnerOutput
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLDispatchOutput,
        DeepEPNormalCombineInput,
        DeepEPNormalDispatchOutput,
    )

# Key under which this module registers its permute overrides. ``MoeRunner``
# passes it on every lookup, so no other platform sees these entries.
PERMUTE_PLATFORM = "sail"


@register_pre_permute("deepep_ll", "deep_gemm", platform=PERMUTE_PLATFORM)
def pre_permute_deepep_ll_to_deep_gemm(
    dispatch_output: DeepEPLLDispatchOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepGemmRunnerInput:
    hidden_states, hidden_states_scale, topk_ids, topk_weights, masked_m, expected_m = (
        dispatch_output
    )

    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights
    running_state["hidden_states_shape"] = hidden_states.shape
    running_state["hidden_states_dtype"] = hidden_states.dtype
    running_state["hidden_states_device"] = hidden_states.device
    activation_scale_block_size = 128
    if hidden_states_scale is not None:
        if hidden_states.dtype == torch.uint8:
            # MXFP4 data packs two values per byte, so the dispatched quantization
            # group covers 32 values.
            activation_scale_block_size = 32
        else:
            scale_width = hidden_states_scale.shape[-1]
            hidden_size = hidden_states.shape[-1]
            assert scale_width > 0 and hidden_size % scale_width == 0
            activation_scale_block_size = hidden_size // scale_width
    return DeepGemmRunnerInput(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        use_masked_gemm=True,
        masked_m=masked_m,
        expected_m=expected_m,
        activation_scale_block_size=activation_scale_block_size,
    )


@register_pre_permute("deepep_normal", "deep_gemm", platform=PERMUTE_PLATFORM)
def pre_permute_deepep_normal_to_deep_gemm(
    dispatch_output: DeepEPNormalDispatchOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepGemmRunnerInput:
    from sglang.kernels.ops.moe.ep_moe_kernels import ep_scatter_sail

    (
        hidden_states,
        hidden_states_scale,
        topk_ids,
        topk_weights,
        num_recv_tokens_per_expert,
    ) = dispatch_output
    assert runner_config.activation in ("silu", "situ")

    all_tokens = sum(num_recv_tokens_per_expert)
    running_state["all_tokens"] = all_tokens

    K = hidden_states.shape[1]

    hidden_states_shape = hidden_states.shape
    hidden_states_device = hidden_states.device
    hidden_states_dtype = hidden_states.dtype

    running_state["hidden_states_shape"] = hidden_states_shape
    running_state["hidden_states_device"] = hidden_states_device
    running_state["hidden_states_dtype"] = hidden_states_dtype
    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights

    # Deterministic inference zero-fills the scatter buffers: expert-alignment
    # padding leaves slots that ep_scatter_sail never writes, and pad garbage in
    # input_tensor would leak batch-dependent values into the grouped GEMM.
    # The scale buffer only matters for quantized activations sharing this
    # pre-permute (ep_scatter_sail skips scales entirely for BF16 dispatch).
    deterministic = get_exec().deterministic.enable_deterministic_inference
    buffer_init = torch.zeros if deterministic else torch.empty

    input_tensor = buffer_init(
        (all_tokens, K),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    block_k = 128
    is_block_wise = quant_info.block_shape is not None
    if hidden_states_scale is None:
        input_tensor_scale = None
    elif deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
        input_tensor_scale = torch.zeros(
            (hidden_states_scale.shape[-1], all_tokens),
            device=hidden_states.device,
            dtype=hidden_states_scale.dtype,
        ).transpose(0, 1)
    else:
        # The dispatch quantization group size is independent of the weight
        # quantization; follow the actual wire scale width.
        scale_width = hidden_states_scale.shape[-1]
        assert scale_width > 0 and K % scale_width == 0
        block_k = K // scale_width
        if quant_info.use_int8:
            assert not is_block_wise and scale_width == 1
        if quant_info.use_mxfp4:
            assert hidden_states_scale.dtype == torch.uint8 and block_k == 16
        input_tensor_scale = buffer_init(
            (all_tokens, scale_width),
            device=hidden_states.device,
            dtype=hidden_states_scale.dtype,
        )

    m_indices = buffer_init(all_tokens, device=hidden_states.device, dtype=torch.int32)
    output_index = torch.empty_like(topk_ids)

    if get_offloader().forbid_copy_engine_usage:
        num_recv_tokens_per_expert_gpu = copy_list_to_gpu_no_ce(
            num_recv_tokens_per_expert
        )
    else:
        num_recv_tokens_per_expert_gpu = torch.tensor(
            num_recv_tokens_per_expert,
            dtype=torch.int32,
            pin_memory=True,
            device="cpu",
        ).cuda(non_blocking=True)
    expert_start_loc = torch.empty_like(num_recv_tokens_per_expert_gpu)

    ep_scatter_sail(
        hidden_states,
        topk_ids.to(torch.int32),
        num_recv_tokens_per_expert_gpu,
        expert_start_loc,
        input_tensor,
        m_indices,
        output_index,
        hidden_states_scale,
        input_tensor_scale,
        BLOCK_E=1,
        BLOCK_D=block_k,
        is_block_wise=is_block_wise,
        num_valid_tokens_per_expert=num_recv_tokens_per_expert_gpu,
    )
    dispose_tensor(hidden_states)
    if hidden_states_scale is not None:
        dispose_tensor(hidden_states_scale)

    running_state["output_index"] = output_index
    # Without m_rows the nopad grouped GEMM rescans m_indices to recompute the
    # per-expert row counts; ep_scatter_sail already consumed exactly those
    # counts (BLOCK_E=1, no alignment padding), so passing them through skips
    # the scan without changing numerics.
    running_state["m_rows"] = num_recv_tokens_per_expert_gpu

    return DeepGemmRunnerInput(
        hidden_states=input_tensor,
        hidden_states_scale=input_tensor_scale,
        use_masked_gemm=False,
        m_indices=m_indices,
        activation_scale_block_size=block_k,
    )


@register_post_permute("deep_gemm", "deepep_normal", platform=PERMUTE_PLATFORM)
def post_permute_deep_gemm_to_deepep_normal(
    runner_output: DeepGemmRunnerOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepEPNormalCombineInput:
    """Widen the gathered shape back for MXFP4.

    The dispatch carries MXFP4 two values per byte, so the bf16 shape the
    dispatcher recorded is half the width the gather buffer needs.
    """
    if quant_info.use_mxfp4:
        M, K = running_state["hidden_states_shape"]
        running_state["hidden_states_shape"] = (M, K * 2)
    return _shared_post_permute(runner_output, quant_info, runner_config, running_state)
