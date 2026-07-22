"""FP8 MoE dispatch using LoRA-capable block-scale wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        FlashInferTrtllmFp8MoeQuantInfo,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


def fused_experts_fp8_sgl(
    dispatch_output: StandardDispatchOutput,
    quant_info: FlashInferTrtllmFp8MoeQuantInfo,
    runner_config: MoeRunnerConfig,
    use_routed_topk: bool = False,
) -> StandardCombineInput:
    # Lazy (call-time) imports so this module never triggers the flashinfer_trtllm
    # <-> quantization import cycle at load time.
    from flashinfer.fused_moe import Fp8QuantizationType

    from sglang.kernels.ops.moe.trtllm_lora_temp.topk_pack import fused_pack_topk
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        get_tp_group,
        is_allocation_symmetric,
        next_power_of_2,
        per_token_group_quant_fp8,
        scaled_fp8_quant,
        trtllm_fp8_per_tensor_scale_moe_wrapper,
        use_symmetric_memory,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.layers.moe.utils import RoutingMethodType
    from sglang.srt.lora.trtllm_lora_temp.experimental_sgl_trtllm_moe import (
        sgl_trtllm_fp8_block_scale_moe_wrapper as trtllm_fp8_block_scale_moe_wrapper,
    )
    from sglang.srt.lora.trtllm_lora_temp.experimental_sgl_trtllm_moe import (
        sgl_trtllm_fp8_block_scale_routed_moe_wrapper as trtllm_fp8_block_scale_routed_moe_wrapper,
    )

    _SUPPORTED_FP8_ACTIVATIONS = {"silu", "relu2"}
    assert runner_config.activation in _SUPPORTED_FP8_ACTIVATIONS, (
        f"Only {_SUPPORTED_FP8_ACTIVATIONS} are supported for FP8 MoE, "
        f"got '{runner_config.activation}'."
    )
    assert not runner_config.no_combine, "no_combine is not supported for flashinfer."

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    if TopKOutputChecker.format_is_bypassed(topk_output):
        router_logits = topk_output.router_logits
        topk_config = topk_output.topk_config
        correction_bias = (
            None
            if topk_config.correction_bias is None
            else topk_config.correction_bias.to(hidden_states.dtype)
        )
    else:
        router_logits = None
        topk_config = None
        correction_bias = None

    routing_method_type = quant_info.routing_method_type
    fp8_quantization_type = (
        Fp8QuantizationType.MxFp8
        if quant_info.use_mxfp8
        else Fp8QuantizationType.DeepSeekFp8
    )
    use_shuffled_weight = quant_info.use_mxfp8

    if quant_info.block_quant:
        assert quant_info.weight_block_k is not None
        assert quant_info.w13_weight_scale_inv is not None
        assert quant_info.w2_weight_scale_inv is not None

        if quant_info.use_mxfp8:
            assert quant_info.weight_block_k == 32
            from flashinfer import mxfp8_quantize

            a_q, a_sf = mxfp8_quantize(hidden_states, False)
            # FlashInfer TRT-LLM MxFP8 expects token-major activation scales:
            # [num_tokens, hidden_size // 32] (no transpose).
            a_sf_t = a_sf.view(torch.uint8).reshape(hidden_states.shape[0], -1)
        else:
            a_q, a_sf = per_token_group_quant_fp8(
                hidden_states, quant_info.weight_block_k
            )
            a_sf_t = a_sf.t().contiguous()

        # Allocate output inside symmetric memory context
        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            symm_output = torch.empty(
                hidden_states.shape[0],
                hidden_states.shape[1],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        # Move kernel call outside context manager to avoid graph breaks
        # during torch.compile for piecewise cuda graph.
        # Use custom op wrapper for torch.compile compatibility.
        if use_routed_topk:
            assert (
                runner_config.top_k is not None
            ), "runner_config.top_k is required for flashinfer_trtllm_routed."
            assert TopKOutputChecker.format_is_standard(topk_output)
            packed_topk_ids = fused_pack_topk(
                topk_ids=topk_output.topk_ids,
                topk_weights=topk_output.topk_weights,
            )

            output = trtllm_fp8_block_scale_routed_moe_wrapper(
                topk_ids=packed_topk_ids,
                routing_bias=None,
                hidden_states=a_q,
                hidden_states_scale=a_sf_t,
                gemm1_weights=quant_info.w13_weight,
                gemm1_weights_scale=quant_info.w13_weight_scale_inv,
                gemm2_weights=quant_info.w2_weight,
                gemm2_weights_scale=quant_info.w2_weight_scale_inv,
                num_experts=quant_info.global_num_experts,
                top_k=runner_config.top_k,
                n_group=None,
                topk_group=None,
                intermediate_size=quant_info.intermediate_size,
                local_expert_offset=quant_info.local_expert_offset,
                local_num_experts=quant_info.local_num_experts,
                routed_scaling_factor=(
                    runner_config.routed_scaling_factor
                    if runner_config.routed_scaling_factor is not None
                    else 1.0
                ),
                routing_method_type=(
                    RoutingMethodType.TopK
                    if routing_method_type == RoutingMethodType.DeepSeekV3
                    else routing_method_type
                ),
                use_shuffled_weight=use_shuffled_weight,
                tune_max_num_tokens=next_power_of_2(a_q.shape[0]),
                fp8_quantization_type=int(fp8_quantization_type),
                activation_type=quant_info.activation_type,
            )
        else:
            assert TopKOutputChecker.format_is_bypassed(topk_output)

            output = trtllm_fp8_block_scale_moe_wrapper(
                routing_logits=router_logits,
                routing_bias=correction_bias,
                hidden_states=a_q,
                hidden_states_scale=a_sf_t,
                gemm1_weights=quant_info.w13_weight,
                gemm1_weights_scale=quant_info.w13_weight_scale_inv,
                gemm2_weights=quant_info.w2_weight,
                gemm2_weights_scale=quant_info.w2_weight_scale_inv,
                num_experts=quant_info.global_num_experts,
                top_k=topk_config.top_k,
                n_group=topk_config.num_expert_group,
                topk_group=topk_config.topk_group,
                intermediate_size=quant_info.intermediate_size,
                local_expert_offset=quant_info.local_expert_offset,
                local_num_experts=quant_info.local_num_experts,
                routed_scaling_factor=(
                    runner_config.routed_scaling_factor
                    if runner_config.routed_scaling_factor is not None
                    else 1.0
                ),
                routing_method_type=routing_method_type,
                use_shuffled_weight=use_shuffled_weight,
                tune_max_num_tokens=next_power_of_2(a_q.shape[0]),
                fp8_quantization_type=int(fp8_quantization_type),
                activation_type=quant_info.activation_type,
            )
        # TODO: Once https://github.com/flashinfer-ai/flashinfer/issues/2703 is fixed, pass output to moe kernel and remove this copy.
        symm_output.copy_(output)
        output = symm_output
    else:
        assert TopKOutputChecker.format_is_bypassed(topk_output)
        assert quant_info.w13_input_scale is not None
        assert quant_info.output1_scales_scalar is not None
        assert quant_info.output1_scales_gate_scalar is not None
        assert quant_info.output2_scales_scalar is not None

        a_q, _ = scaled_fp8_quant(hidden_states, quant_info.w13_input_scale)
        routing_bias_cast = (
            None if correction_bias is None else correction_bias.to(torch.bfloat16)
        )

        # Allocate output inside symmetric memory context
        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            symm_output = torch.empty(
                hidden_states.shape[0],
                hidden_states.shape[1],
                dtype=torch.bfloat16,
                device=hidden_states.device,
            )

        # Move kernel call outside context manager to avoid graph breaks
        # during torch.compile for piecewise cuda graph.
        # Use custom op wrapper for torch.compile compatibility.

        router_logits = router_logits.to(torch.bfloat16)

        output = trtllm_fp8_per_tensor_scale_moe_wrapper(
            routing_logits=router_logits,
            routing_bias=routing_bias_cast,
            hidden_states=a_q,
            gemm1_weights=quant_info.w13_weight,
            output1_scales_scalar=quant_info.output1_scales_scalar,
            output1_scales_gate_scalar=quant_info.output1_scales_gate_scalar,
            gemm2_weights=quant_info.w2_weight,
            output2_scales_scalar=quant_info.output2_scales_scalar,
            num_experts=quant_info.global_num_experts,
            top_k=topk_config.top_k,
            n_group=topk_config.num_expert_group,
            topk_group=topk_config.topk_group,
            intermediate_size=int(quant_info.w2_weight.shape[2]),
            local_expert_offset=quant_info.local_expert_offset,
            local_num_experts=quant_info.local_num_experts,
            routed_scaling_factor=(
                runner_config.routed_scaling_factor
                if runner_config.routed_scaling_factor is not None
                else 1.0
            ),
            use_routing_scales_on_input=False,
            routing_method_type=routing_method_type,
            tune_max_num_tokens=next_power_of_2(a_q.shape[0]),
            activation_type=quant_info.activation_type,
        )
        symm_output.copy_(output)
        output = symm_output

    return StandardCombineInput(hidden_states=output)
