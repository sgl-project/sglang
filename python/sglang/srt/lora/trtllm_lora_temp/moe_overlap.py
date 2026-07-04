"""Two-stream MoE LoRA dispatch (O1).

Monkey-patches ``fused_experts_none_to_experimental_sgl_trtllm_fp8_lora`` in
``layers/moe/moe_runner/flashinfer_trtllm.py`` (when
``SGLANG_LORA_TWO_STREAM=1``) so the gate_up LoRA shrink+expand runs on a
side stream concurrent with the main-stream FP8 quant.

Batches that don't qualify for two-stream (prefill / non-virtual-lora /
batch without active LoRA) fall through to the saved-original function so
their behavior is byte-identical to the unpatched code path.
"""

import torch

from sglang.srt.lora.trtllm_lora_temp import (
    get_lora_side_stream,
    get_original_bf16_moe_lora_func,
    get_original_fp4_moe_lora_func,
    get_original_moe_lora_func,
    is_two_stream_active,
)

# GEMM1-LoRA overlap: keep LoRA-ready events recorded during cuda-graph capture alive so the
# captured cross-stream wait (resolved inside the trtllm op before activation) isn't torn down
# before graph instantiation. Only appended while capturing; eager runs rely on CUDA's
# deferred cudaEventDestroy.
_LORA_OVERLAP_EVENTS: list = []


def fused_experts_none_to_experimental_sgl_trtllm_fp8_lora_two_stream(
    dispatch_output,
    quant_info,
    runner_config,
    lora_info,
):
    """Drop-in replacement for the like-named function in flashinfer_trtllm.py.

    Two-stream fast path: only fires when the batch is decode-shaped AND uses
    virtual-experts LoRA. Everything else delegates to the original function.
    """
    hidden_states = dispatch_output.hidden_states

    use_virtual_lora_store = bool(
        lora_info.lora_use_virtual_experts and lora_info.max_lora_rank > 0
    )
    # Two-stream requires virtual-experts LoRA AND a decode-shaped batch.
    # Fall back to the original implementation for anything else (prefill,
    # non-virtual LoRA, non-LoRA capture, etc.).
    if not (use_virtual_lora_store and is_two_stream_active(hidden_states)):
        return get_original_moe_lora_func()(
            dispatch_output, quant_info, runner_config, lora_info
        )

    # ---- two-stream fast path ----
    from flashinfer.fused_moe import Fp8QuantizationType

    from sglang.jit_kernel.trtllm_lora_temp import (
        trtllm_fp8_block_scale_routed_moe_lora,
    )
    from sglang.jit_kernel.trtllm_lora_temp.topk_pack import fused_pack_topk
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        use_symmetric_memory,
    )
    from sglang.srt.layers.dp_attention import is_allocation_symmetric
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.layers.moe.utils import RoutingMethodType
    from sglang.srt.layers.quantization.fp8_kernel import per_token_group_quant_fp8
    from sglang.srt.lora.trtllm_lora_temp.shared_add_overlap import (
        maybe_overlap_staged_shared_add,
    )
    from sglang.kernels.ops.moe.trtllm_lora_temp.virtual_experts import merged_experts_fused_moe_lora_add
    from sglang.srt.utils.common import next_power_of_2

    assert runner_config.activation == "silu" and runner_config.is_gated, (
        "experimental_sgl_trtllm LoRA currently supports the gated SwiGLU FP8 "
        "Qwen path only."
    )
    assert quant_info.block_quant and not quant_info.use_mxfp8, (
        "experimental_sgl_trtllm LoRA currently supports DeepSeekFp8 block-quant "
        "checkpoints only."
    )
    assert quant_info.weight_block_k is not None
    assert quant_info.w13_weight_scale_inv is not None
    assert quant_info.w2_weight_scale_inv is not None

    topk_output = dispatch_output.topk_output
    assert TopKOutputChecker.format_is_standard(topk_output)
    assert runner_config.top_k is not None

    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights
    token_lora_mapping = lora_info.token_lora_mapping
    fused_lora_routing_cache: dict = {}

    side_stream = get_lora_side_stream()

    # EP-aware LoRA: under MoE EP each rank computes the delta only for its owned experts
    # (passed via local_expert_offset/local_num_experts below). gate_up_delta stays
    # new_empty even though non-owned [token, k] slots are then left unwritten -- the
    # trtllm MoE is itself EP-aware, so those slots never feed the all-reduced output.
    gate_up_delta_shape = (
        hidden_states.shape[0],
        runner_config.top_k,
        quant_info.w13_weight.shape[1],
    )
    gate_up_delta = hidden_states.new_empty(gate_up_delta_shape)

    def _run_gate_up_lora():
        merged_experts_fused_moe_lora_add(
            output=gate_up_delta,
            hidden_states=hidden_states,
            lora_a=lora_info.gate_up_lora_a_weights,
            lora_b=lora_info.gate_up_lora_b_weights,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=token_lora_mapping,
            mul_routed_weight=False,
            experts_shared_outer_loras_a=lora_info.experts_shared_outer_loras,
            experts_shared_outer_loras_b=False,
            routing_cache=fused_lora_routing_cache,
            fuse_add_to_output=False,
            use_direct_expand_add=lora_info.max_lora_rank <= 64,
            local_expert_offset=quant_info.local_expert_offset,
            local_num_experts=quant_info.local_num_experts,
            intermediate_buffer=gate_up_lora_intermediate,
        )

    # GEMM1-LoRA overlap: fire the gate_up LoRA on the side stream + record an event; the
    # trtllm op waits on it right before activation (the only consumer of gate_up_delta), so
    # permute+GEMM1 overlap the side-stream LoRA shrink/expand instead of joining before the
    # whole op.
    lora_event = torch.cuda.Event()

    # Hoist every side-chain allocation onto the MAIN stream (cuda-graph
    # allocator safety -- see the "routing" stage in virtual_experts.py):
    # pre-warm the routing cache and pre-allocate the shrink intermediate here,
    # so the side-stream block below launches kernels only.
    merged_experts_fused_moe_lora_add(
        output=gate_up_delta,
        hidden_states=hidden_states,
        lora_a=lora_info.gate_up_lora_a_weights,
        lora_b=lora_info.gate_up_lora_b_weights,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        token_lora_mapping=token_lora_mapping,
        mul_routed_weight=False,
        experts_shared_outer_loras_a=lora_info.experts_shared_outer_loras,
        experts_shared_outer_loras_b=False,
        routing_cache=fused_lora_routing_cache,
        stage="routing",
        local_expert_offset=quant_info.local_expert_offset,
        local_num_experts=quant_info.local_num_experts,
    )
    gate_up_lora_intermediate = hidden_states.new_empty(
        (
            hidden_states.shape[0],
            topk_ids.shape[1],
            lora_info.gate_up_lora_a_weights.shape[2],
        )
    )

    # O1 fork — gate_up shrink/expand on side stream concurrent with the main-stream
    # per-token-group FP8 quant + the trtllm op's permute+GEMM1 below.
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        _run_gate_up_lora()
        lora_event.record()

    # Fuse the per-token scale transpose into the quant kernel: column-major scales make
    # the `.t()` a free view, dropping the standalone ~2us transpose+copy. The trtllm MoE
    # kernel wants the [K, M]-contiguous scale, which `.t()` of the column-major buffer is
    # exactly -- byte/shape-identical to the old `a_sf.t().contiguous()`.
    a_q, a_sf = per_token_group_quant_fp8(
        hidden_states, quant_info.weight_block_k, column_major_scales=True
    )
    a_sf_t = a_sf.t()

    activation_lora_input = torch.empty(
        (hidden_states.shape[0], runner_config.top_k, quant_info.intermediate_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    # SGLANG_OPT_LORA_FUSED_TOPK_PACK: the routed pack may already have been produced
    # fused inside the gating kernel (StandardTopKOutput.packed_topk_ids) — including
    # the padded-region id=-1 mask. Fall back to the separate pack otherwise.
    packed_topk_ids = getattr(topk_output, "packed_topk_ids", None)
    if packed_topk_ids is None:
        packed_topk_ids = fused_pack_topk(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
        direct_down_output = torch.empty(
            hidden_states.shape[0],
            hidden_states.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

    # No pre-op join: the trtllm op waits on lora_event right before its activation kernel,
    # so permute+GEMM1 run concurrent with the side-stream LoRA. Keep the event alive through
    # cuda-graph capture so the captured cross-stream wait isn't torn down before instantiation.
    if torch.cuda.is_current_stream_capturing():
        _LORA_OVERLAP_EVENTS.append(lora_event)
    lora_ready_handle = lora_event.cuda_event

    # Down-LoRA/finalize overlap (env-gated): the op records gemm2_done_event right after the
    # base down GEMM (before finalize); the side stream waits on it and runs ONLY the down-proj
    # LoRA shrink (gemm A) + routing prep concurrent with finalizeKernel. The expand-add
    # (gemm B) atomic-adds into `output` -- which finalize WRITES concurrently -- so it stays
    # on the main stream after the op (post-finalize), exactly like the serial path.
    # DISABLED: the down/finalize overlap is bench-verified net-neutral-to-negative AND
    # corrupts the base/decode path — the captured gemm2_done cross-stream event under
    # cuda-graph replay perturbs no-active-LoRA (base) requests (qwen base gsm8k 0.81 -> 0.56
    # with it on; bisect-confirmed). The serial down-LoRA path below is used unconditionally.
    down_overlap = False
    gemm2_done_handle = 0
    if down_overlap:
        gemm2_done_event = torch.cuda.Event()
        # Materialize the underlying cudaEvent (torch creates it lazily on first record) so
        # .cuda_event is a real handle; the op re-records it after GEMM2.
        gemm2_done_event.record()
        gemm2_done_handle = gemm2_done_event.cuda_event

    moe_result = trtllm_fp8_block_scale_routed_moe_lora(
        topk_ids=packed_topk_ids,
        routing_bias=None,
        hidden_states=a_q,
        hidden_states_scale=a_sf_t,
        gemm1_weights=quant_info.w13_weight,
        gemm1_weights_scale=quant_info.w13_weight_scale_inv,
        gemm2_weights=quant_info.w2_weight,
        gemm2_weights_scale=quant_info.w2_weight_scale_inv,
        gate_up_lora_delta=gate_up_delta,
        activation_lora_input=activation_lora_input,
        lora_ready_event=lora_ready_handle,
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
            if quant_info.routing_method_type == RoutingMethodType.DeepSeekV3
            else quant_info.routing_method_type
        ),
        use_shuffled_weight=False,
        do_finalize=True,
        output=direct_down_output,
        tune_max_num_tokens=next_power_of_2(a_q.shape[0]),
        fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8,
        activation_type=quant_info.activation_type,
        gemm2_done_event=gemm2_done_handle,
    )

    output = moe_result

    def _run_down_lora(
        out, stage="all", intermediate_buffer=None, expand_wait_event=None
    ):
        return merged_experts_fused_moe_lora_add(
            output=out,
            hidden_states=activation_lora_input.view(-1, quant_info.intermediate_size),
            lora_a=lora_info.down_lora_a_weights,
            lora_b=lora_info.down_lora_b_weights,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=token_lora_mapping,
            mul_routed_weight=True,
            experts_shared_outer_loras_a=False,
            experts_shared_outer_loras_b=lora_info.experts_shared_outer_loras,
            routing_cache=fused_lora_routing_cache,
            fuse_add_to_output=False,
            fuse_sum_all_reduce=True,
            use_direct_expand_add=lora_info.max_lora_rank <= 64,
            local_expert_offset=quant_info.local_expert_offset,
            local_num_experts=quant_info.local_num_experts,
            stage=stage,
            intermediate_buffer=intermediate_buffer,
            expand_wait_event=expand_wait_event,
        )

    # Shared-add overlap: the trtllm op above already finalized `output`, so the
    # staged shared-expert add (if any) can run on the producer (main) stream
    # concurrent with the down-LoRA shrink below; the expand waits on it via
    # expand_wait_event before atomic-adding into the same buffer.
    shared_add_done = maybe_overlap_staged_shared_add(output)

    if down_overlap:
        # Fork at "base down GEMM done": ONLY the shrink (gemm A) + routing prep run on the
        # side stream, concurrent with the main-stream finalizeKernel. The expand-add (gemm B)
        # joins back on the MAIN stream after the op -- i.e. strictly after finalize wrote
        # `output` -- and atomic-adds into it exactly like the serial path: same kernels,
        # same buffers, identical numerics; the shrink just starts earlier.
        # The shrink intermediate is allocated HERE (main = consumer stream of the expand),
        # per the SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC lesson on side-stream allocations.
        down_intermediate = hidden_states.new_empty(
            (
                hidden_states.shape[0],
                topk_ids.shape[1],
                lora_info.down_lora_a_weights.shape[2],
            )
        )
        side_stream.wait_event(gemm2_done_event)
        shrink_done_event = torch.cuda.Event()
        with torch.cuda.stream(side_stream):
            _run_down_lora(
                output, stage="shrink", intermediate_buffer=down_intermediate
            )
            shrink_done_event.record()
        if torch.cuda.is_current_stream_capturing():
            _LORA_OVERLAP_EVENTS.append(gemm2_done_event)
            _LORA_OVERLAP_EVENTS.append(shrink_done_event)
        torch.cuda.current_stream().wait_event(shrink_done_event)
        _run_down_lora(
            output,
            stage="expand",
            intermediate_buffer=down_intermediate,
            expand_wait_event=shared_add_done,
        )
    else:
        _run_down_lora(output, expand_wait_event=shared_add_done)
    return StandardCombineInput(hidden_states=output)


def fused_experts_none_to_experimental_sgl_trtllm_fp4_lora_two_stream(
    dispatch_output,
    quant_info,
    runner_config,
    lora_info,
):
    """Two-stream NVFP4 sibling of the FP8 two-stream MoE LoRA dispatch.

    Fires only for virtual-experts LoRA + decode-shaped batches; everything else
    delegates to the saved-original single-stream FP4 dispatch (byte-identical).
    """
    hidden_states = dispatch_output.hidden_states

    use_virtual_lora_store = bool(
        lora_info.lora_use_virtual_experts and lora_info.max_lora_rank > 0
    )
    if not (use_virtual_lora_store and is_two_stream_active(hidden_states)):
        return get_original_fp4_moe_lora_func()(
            dispatch_output, quant_info, runner_config, lora_info
        )

    # ---- two-stream fast path ----
    from sglang.jit_kernel.trtllm_lora_temp import (
        trtllm_fp4_block_scale_routed_moe_lora,
    )
    from sglang.jit_kernel.trtllm_lora_temp.topk_pack import fused_pack_topk
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        use_symmetric_memory,
    )
    from sglang.srt.layers.dp_attention import is_allocation_symmetric
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.kernels.ops.moe.trtllm_lora_temp.virtual_experts import merged_experts_fused_moe_lora_add

    assert (
        runner_config.activation == "silu" and runner_config.is_gated
    ), "experimental_sgl_trtllm NVFP4 LoRA currently supports the gated SwiGLU path only."
    topk_output = dispatch_output.topk_output
    assert TopKOutputChecker.format_is_standard(topk_output)
    assert runner_config.top_k is not None

    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights
    token_lora_mapping = lora_info.token_lora_mapping
    fused_lora_routing_cache: dict = {}

    # Down-proj LoRA runs serially on the main stream (after the trtllm op) by default. The old
    # side-stream down-overlap was removed: bench-verified net-neutral-to-negative (the extra
    # side-stream all-reduce cancels any overlap gain), AND its act_ready_event cross-stream sync
    # corrupted decode state under sustained heavy LoRA load (cuda-graph replay -> persistent
    # garbage). SGLANG_OPT_LORA_DOWN_FINALIZE_OVERLAP=1 re-introduces a more conservative variant:
    # fork at gemm2_done (not act_ready), and only the SHRINK (gemm A) overlaps the finalize
    # kernel -- the expand-add (gemm B) stays on the MAIN stream post-finalize (it writes the
    # same `output` finalize writes), so its kernels/numerics match the serial path exactly.
    inter = quant_info.intermediate_size_per_partition
    side_stream = get_lora_side_stream()

    gate_up_delta = hidden_states.new_empty(
        (hidden_states.shape[0], runner_config.top_k, quant_info.w13_weight.shape[1])
    )

    def _run_gate_up_lora():
        merged_experts_fused_moe_lora_add(
            output=gate_up_delta,
            hidden_states=hidden_states,
            lora_a=lora_info.gate_up_lora_a_weights,
            lora_b=lora_info.gate_up_lora_b_weights,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=token_lora_mapping,
            mul_routed_weight=False,
            experts_shared_outer_loras_a=lora_info.experts_shared_outer_loras,
            experts_shared_outer_loras_b=False,
            routing_cache=fused_lora_routing_cache,
            fuse_add_to_output=False,
            use_direct_expand_add=lora_info.max_lora_rank <= 64,
            local_expert_offset=quant_info.local_expert_offset,
            local_num_experts=quant_info.local_num_experts,
            intermediate_buffer=gate_up_lora_intermediate,
        )

    # Hoist every side-chain allocation onto the MAIN stream (cuda-graph
    # allocator safety -- see the "routing" stage in virtual_experts.py).
    merged_experts_fused_moe_lora_add(
        output=gate_up_delta,
        hidden_states=hidden_states,
        lora_a=lora_info.gate_up_lora_a_weights,
        lora_b=lora_info.gate_up_lora_b_weights,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        token_lora_mapping=token_lora_mapping,
        mul_routed_weight=False,
        experts_shared_outer_loras_a=lora_info.experts_shared_outer_loras,
        experts_shared_outer_loras_b=False,
        routing_cache=fused_lora_routing_cache,
        stage="routing",
        local_expert_offset=quant_info.local_expert_offset,
        local_num_experts=quant_info.local_num_experts,
    )
    gate_up_lora_intermediate = hidden_states.new_empty(
        (
            hidden_states.shape[0],
            topk_ids.shape[1],
            lora_info.gate_up_lora_a_weights.shape[2],
        )
    )

    # O1-fp4 fork: gate_up shrink/expand on the side stream, concurrent with the
    # FP4 op's permute + gate_up GEMM1 below. The op waits on lora_event right
    # before its activation kernel (the only consumer of gate_up_delta).
    lora_event = torch.cuda.Event()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        _run_gate_up_lora()
        lora_event.record()

    activation_lora_input = torch.empty(
        (hidden_states.shape[0], runner_config.top_k, inter),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    packed_topk_ids = fused_pack_topk(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )
    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
        direct_down_output = torch.empty(
            hidden_states.shape[0],
            hidden_states.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

    # Keep the event alive through cuda-graph capture so the captured wait inside
    # the FP4 op isn't torn down before instantiation (eager relies on deferred destroy).
    if torch.cuda.is_current_stream_capturing():
        _LORA_OVERLAP_EVENTS.append(lora_event)
    lora_ready_handle = lora_event.cuda_event

    # Down-LoRA/finalize overlap (env-gated, see the comment above): record gemm2_done inside
    # the op; the side stream runs only the down-LoRA shrink + routing prep concurrent with
    # finalize; the expand-add joins back on the main stream after the op.
    # DISABLED: the down/finalize overlap is bench-verified net-neutral-to-negative AND
    # corrupts the base/decode path — the captured gemm2_done cross-stream event under
    # cuda-graph replay perturbs no-active-LoRA (base) requests (qwen base gsm8k 0.81 -> 0.56
    # with it on; bisect-confirmed). The serial down-LoRA path below is used unconditionally.
    down_overlap = False
    gemm2_done_handle = 0
    if down_overlap:
        gemm2_done_event = torch.cuda.Event()
        # Materialize the underlying cudaEvent (torch creates it lazily on first record) so
        # .cuda_event is a real handle; the op re-records it after the down GEMM.
        gemm2_done_event.record()
        gemm2_done_handle = gemm2_done_event.cuda_event

    output = trtllm_fp4_block_scale_routed_moe_lora(
        topk_ids=packed_topk_ids,
        routing_bias=None,
        hidden_states=hidden_states,
        hidden_states_scale=None,
        gemm1_weights=quant_info.w13_weight,
        gemm1_weights_scale=quant_info.w13_weight_scale.view(torch.float8_e4m3fn),
        gemm2_weights=quant_info.w2_weight,
        gemm2_weights_scale=quant_info.w2_weight_scale.view(torch.float8_e4m3fn),
        output1_scales_scalar=quant_info.g1_scale_c,
        output1_scales_gate_scalar=quant_info.g1_alphas,
        output2_scales_scalar=quant_info.g2_alphas,
        gate_up_lora_delta=gate_up_delta,
        activation_lora_input=activation_lora_input,
        lora_ready_event=lora_ready_handle,
        num_experts=quant_info.global_num_experts,
        top_k=runner_config.top_k,
        intermediate_size=inter,
        local_expert_offset=quant_info.local_expert_offset,
        local_num_experts=quant_info.local_num_experts,
        routed_scaling_factor=(
            runner_config.routed_scaling_factor
            if runner_config.routed_scaling_factor is not None
            else 1.0
        ),
        routing_method_type=quant_info.routing_method_type,
        do_finalize=True,
        output=direct_down_output,
        gemm2_done_event=gemm2_done_handle,
    )

    def _run_down_lora(out, stage="all", intermediate_buffer=None):
        return merged_experts_fused_moe_lora_add(
            output=out,
            hidden_states=activation_lora_input.view(-1, inter),
            lora_a=lora_info.down_lora_a_weights,
            lora_b=lora_info.down_lora_b_weights,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=token_lora_mapping,
            mul_routed_weight=True,
            experts_shared_outer_loras_a=False,
            experts_shared_outer_loras_b=lora_info.experts_shared_outer_loras,
            routing_cache=fused_lora_routing_cache,
            fuse_add_to_output=False,
            fuse_sum_all_reduce=True,
            use_direct_expand_add=lora_info.max_lora_rank <= 64,
            local_expert_offset=quant_info.local_expert_offset,
            local_num_experts=quant_info.local_num_experts,
            stage=stage,
            intermediate_buffer=intermediate_buffer,
        )

    if down_overlap:
        # Fork at "base down GEMM done": shrink (gemm A) + routing prep on the side stream
        # concurrent with the main-stream finalize; the expand-add (gemm B) joins back on the
        # MAIN stream after the op (strictly post-finalize) -- serial-path kernels/numerics.
        # Intermediate allocated on main (= consumer stream of the expand), per the
        # SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC lesson.
        down_intermediate = hidden_states.new_empty(
            (
                hidden_states.shape[0],
                topk_ids.shape[1],
                lora_info.down_lora_a_weights.shape[2],
            )
        )
        side_stream.wait_event(gemm2_done_event)
        shrink_done_event = torch.cuda.Event()
        with torch.cuda.stream(side_stream):
            _run_down_lora(
                output, stage="shrink", intermediate_buffer=down_intermediate
            )
            shrink_done_event.record()
        if torch.cuda.is_current_stream_capturing():
            _LORA_OVERLAP_EVENTS.append(gemm2_done_event)
            _LORA_OVERLAP_EVENTS.append(shrink_done_event)
        torch.cuda.current_stream().wait_event(shrink_done_event)
        _run_down_lora(output, stage="expand", intermediate_buffer=down_intermediate)
    else:
        _run_down_lora(output)
    return StandardCombineInput(hidden_states=output)


def fused_experts_none_to_experimental_sgl_trtllm_bf16_lora_two_stream(
    dispatch_output,
    quant_info,
    runner_config,
    lora_info,
):
    """Two-stream BF16 sibling of the FP8/FP4 two-stream MoE LoRA dispatches.

    O1-bf16 fork: the gate_up LoRA shrink/expand runs on the side stream
    concurrent with the bf16 op's routing + permute + gate_up GEMM; the op
    waits on ``lora_ready_event`` right before its activation kernel (the only
    consumer of ``gate_up_delta``). Fires only for virtual-experts LoRA +
    decode-shaped batches; everything else delegates to the saved-original
    single-stream bf16 dispatch (byte-identical). Down-LoRA stays serial on
    the main stream — the down/finalize overlap was bench-verified
    net-neutral-to-negative on the FP8/FP4 paths and corrupted the base
    decode path under cuda-graph replay (see the comment in the FP4 variant).
    """
    hidden_states = dispatch_output.hidden_states

    use_virtual_lora_store = bool(
        lora_info.lora_use_virtual_experts and lora_info.max_lora_rank > 0
    )
    if not (use_virtual_lora_store and is_two_stream_active(hidden_states)):
        return get_original_bf16_moe_lora_func()(
            dispatch_output, quant_info, runner_config, lora_info
        )

    # ---- two-stream fast path ----
    from sglang.jit_kernel.trtllm_lora_temp import trtllm_bf16_routed_moe_lora
    from sglang.jit_kernel.trtllm_lora_temp.topk_pack import fused_pack_topk
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        use_symmetric_memory,
    )
    from sglang.srt.layers.dp_attention import is_allocation_symmetric
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        get_activation_type,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.layers.moe.utils import RoutingMethodType
    from sglang.kernels.ops.moe.trtllm_lora_temp.virtual_experts import merged_experts_fused_moe_lora_add

    assert (
        runner_config.activation == "silu" and runner_config.is_gated
    ), "experimental_sgl_trtllm BF16 LoRA currently supports the gated SwiGLU path only."
    topk_output = dispatch_output.topk_output
    assert TopKOutputChecker.format_is_standard(topk_output)
    assert runner_config.top_k is not None

    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights
    token_lora_mapping = lora_info.token_lora_mapping
    fused_lora_routing_cache: dict = {}

    inter = runner_config.intermediate_size_per_partition
    side_stream = get_lora_side_stream()

    gate_up_delta = hidden_states.new_empty(
        (hidden_states.shape[0], runner_config.top_k, 2 * inter)
    )

    def _run_gate_up_lora():
        merged_experts_fused_moe_lora_add(
            output=gate_up_delta,
            hidden_states=hidden_states,
            lora_a=lora_info.gate_up_lora_a_weights,
            lora_b=lora_info.gate_up_lora_b_weights,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=token_lora_mapping,
            mul_routed_weight=False,
            experts_shared_outer_loras_a=lora_info.experts_shared_outer_loras,
            experts_shared_outer_loras_b=False,
            routing_cache=fused_lora_routing_cache,
            fuse_add_to_output=False,
            use_direct_expand_add=lora_info.max_lora_rank <= 64,
            local_expert_offset=quant_info.local_expert_offset,
            local_num_experts=runner_config.num_local_experts,
            intermediate_buffer=gate_up_lora_intermediate,
        )

    # O1-bf16 fork: gate_up shrink/expand on the side stream, concurrent with the
    # bf16 op's routing + permute + gate_up GEMM below. The op waits on lora_event
    # right before its activation kernel (the only consumer of gate_up_delta).
    lora_event = torch.cuda.Event()

    # Hoist every side-chain allocation onto the MAIN stream (cuda-graph allocator
    # safety -- see the "routing" stage in virtual_experts.py): pre-warm the routing
    # cache and pre-allocate the shrink intermediate here, so the side-stream block
    # below launches kernels only. Without this, the routing tensors + shrink
    # intermediate get allocated inside the side-stream context during cuda-graph
    # capture, where cross-stream tracking is off -> pool blocks reused with no graph
    # edge -> '!!!!' decode corruption at max-loras>=2 (matches the fp8/fp4 fix).
    merged_experts_fused_moe_lora_add(
        output=gate_up_delta,
        hidden_states=hidden_states,
        lora_a=lora_info.gate_up_lora_a_weights,
        lora_b=lora_info.gate_up_lora_b_weights,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        token_lora_mapping=token_lora_mapping,
        mul_routed_weight=False,
        experts_shared_outer_loras_a=lora_info.experts_shared_outer_loras,
        experts_shared_outer_loras_b=False,
        routing_cache=fused_lora_routing_cache,
        stage="routing",
        local_expert_offset=quant_info.local_expert_offset,
        local_num_experts=runner_config.num_local_experts,
    )
    gate_up_lora_intermediate = hidden_states.new_empty(
        (
            hidden_states.shape[0],
            topk_ids.shape[1],
            lora_info.gate_up_lora_a_weights.shape[2],
        )
    )
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        _run_gate_up_lora()
        lora_event.record()

    activation_lora_input = torch.empty(
        (hidden_states.shape[0], runner_config.top_k, inter),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    packed_topk_ids = getattr(topk_output, "packed_topk_ids", None)
    if packed_topk_ids is None:
        packed_topk_ids = fused_pack_topk(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

    routing_method_type = runner_config.routing_method_type
    if routing_method_type is None:
        routing_method_type = RoutingMethodType.Default
    elif routing_method_type == RoutingMethodType.DeepSeekV3:
        routing_method_type = RoutingMethodType.TopK

    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
        direct_down_output = torch.empty(
            hidden_states.shape[0],
            hidden_states.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

    # Keep the event alive through cuda-graph capture so the captured wait inside
    # the bf16 op isn't torn down before instantiation (eager relies on deferred destroy).
    if torch.cuda.is_current_stream_capturing():
        _LORA_OVERLAP_EVENTS.append(lora_event)
    lora_ready_handle = lora_event.cuda_event

    output = trtllm_bf16_routed_moe_lora(
        topk_ids=packed_topk_ids,
        routing_bias=None,
        hidden_states=hidden_states,
        gemm1_weights=quant_info.gemm1_weights,
        gemm2_weights=quant_info.gemm2_weights,
        gate_up_lora_delta=gate_up_delta,
        activation_lora_input=activation_lora_input,
        num_experts=quant_info.global_num_experts,
        top_k=runner_config.top_k,
        intermediate_size=inter,
        local_expert_offset=quant_info.local_expert_offset,
        local_num_experts=runner_config.num_local_experts,
        routed_scaling_factor=(
            runner_config.routed_scaling_factor
            if runner_config.routed_scaling_factor is not None
            else 1.0
        ),
        routing_method_type=routing_method_type,
        do_finalize=True,
        output=direct_down_output,
        activation_type=get_activation_type(
            runner_config.activation, is_gated=runner_config.is_gated
        ),
        lora_ready_event=lora_ready_handle,
        # Down-LoRA/finalize overlap intentionally NOT wired (gemm2_done_event=0):
        # bench-verified net-neutral-to-negative on FP8/FP4 and corrupts the base
        # decode path under cuda-graph replay. Serial down-LoRA below.
        gemm2_done_event=0,
    )

    merged_experts_fused_moe_lora_add(
        output=output,
        hidden_states=activation_lora_input.view(-1, inter),
        lora_a=lora_info.down_lora_a_weights,
        lora_b=lora_info.down_lora_b_weights,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        token_lora_mapping=token_lora_mapping,
        mul_routed_weight=True,
        experts_shared_outer_loras_a=False,
        experts_shared_outer_loras_b=lora_info.experts_shared_outer_loras,
        routing_cache=fused_lora_routing_cache,
        fuse_add_to_output=False,
        fuse_sum_all_reduce=True,
        use_direct_expand_add=lora_info.max_lora_rank <= 64,
        local_expert_offset=quant_info.local_expert_offset,
        local_num_experts=runner_config.num_local_experts,
    )
    return StandardCombineInput(hidden_states=output)
