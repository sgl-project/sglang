"""Temporary optimized shared-sink LoRA execution for Inkling."""

from __future__ import annotations

import torch

from sglang.srt.environ import envs
from sglang.srt.models.inkling_common.kernels.comm import symm_mem_all_reduce


def allow_inkling_moe_two_stream(
    shared_experts, routed_experts, num_tokens: int
) -> bool:
    """Gate routed/shared overlap when the current batch has LoRA work."""

    from sglang.srt.lora.trtllm_lora_temp.environ import lora_envs
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        get_is_capture_mode,
    )

    lora_backend = getattr(shared_experts, "lora_backend", None)
    if lora_backend is None:
        lora_backend = getattr(routed_experts, "lora_backend", None)
    batch_info = getattr(lora_backend, "batch_info", None)
    has_lora_work = get_is_capture_mode() or bool(
        getattr(batch_info, "has_active_lora", False)
    )
    return not has_lora_work or (
        lora_envs.SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC.get() and num_tokens <= 32
    )


def apply_multi_lora(
    layer,
    inputs: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    base_output: torch.Tensor,
    *,
    stack_num: int,
) -> torch.Tensor:
    if getattr(layer.lora_backend, "name", None) != "triton" or not inputs.is_cuda:
        raise RuntimeError("Multi-slot dense LoRA requires Triton on CUDA")
    from sglang.kernels.ops.gemm.trtllm_lora_temp.sgemm_lora_a import (
        shared_sink_sgemm_lora_a_fwd,
    )
    from sglang.kernels.ops.gemm.trtllm_lora_temp.sgemm_lora_b import (
        shared_sink_sgemm_lora_b_fwd,
    )

    padded_rank = stack_num > 1
    batch_info = layer.lora_backend._sgemm_info()
    shrink = shared_sink_sgemm_lora_a_fwd(
        inputs,
        lora_a,
        batch_info,
        stack_num=stack_num,
        padded_rank=padded_rank,
    )
    return shared_sink_sgemm_lora_b_fwd(
        shrink,
        lora_b,
        batch_info,
        base_output,
        apply_scaling=False,
        padded_rank=padded_rank,
    )


def _shared_sink_routing(layer, num_tokens: int, device: torch.device):
    cached = layer._lora_routing_cache.get(num_tokens)
    if cached is None or cached[0].device != device:
        topk_ids = torch.arange(
            layer.n_shared_experts, dtype=torch.int32, device=device
        ).repeat(num_tokens, 1)
        topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
        cached = (topk_ids, topk_weights)
        layer._lora_routing_cache[num_tokens] = cached
    return cached


def _apply_per_expert_lora(
    layer,
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
) -> None:
    from sglang.kernels.ops.moe.virtual_experts import (
        merged_experts_fused_moe_lora_add,
    )

    num_tokens = output.shape[0]
    topk_ids, topk_weights = _shared_sink_routing(
        layer, num_tokens, hidden_states.device
    )
    token_lora_mapping = layer.lora_backend.batch_info.moe_lora_info
    token_lora_mapping = token_lora_mapping.token_lora_mapping[:num_tokens]
    merged_experts_fused_moe_lora_add(
        output=output,
        hidden_states=hidden_states,
        lora_a=lora_a,
        lora_b=lora_b,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        token_lora_mapping=token_lora_mapping,
        mul_routed_weight=False,
        experts_shared_outer_loras_a=False,
        experts_shared_outer_loras_b=False,
    )


def forward_with_lora(
    layer,
    x_td: torch.Tensor,
    gammas_ts: torch.Tensor,
    linearized_weights: tuple[torch.Tensor, torch.Tensor],
    use_reduce_scatter: bool,
) -> torch.Tensor:
    w13_lin, w2_lin = linearized_weights
    t = x_td.shape[0]
    n = layer.n_shared_experts
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        get_is_capture_mode,
    )

    run_lora = get_is_capture_mode() or bool(
        getattr(layer.lora_backend.batch_info, "has_active_lora", False)
    )
    overlap_gate_up = (
        envs.SGLANG_OPT_USE_INKLING_MULTI_STREAM_OVERLAP.get()
        and run_lora
        and x_td.is_cuda
        and layer.experts_shared_outer_loras
    )
    if overlap_gate_up:
        device_major = torch.cuda.get_device_capability(x_td.device)[0]
        max_overlap_tokens = {9: 32, 10: 16}.get(device_major, 0)
        overlap_gate_up = t <= max_overlap_tokens

    gate_up_shrink = None
    gate_up_side_stream = None
    gate_up_batch_info = None
    if overlap_gate_up:
        from sglang.srt.lora.trtllm_lora_temp import get_lora_side_stream

        assert layer._w1_delta is not None
        consumer_stream = torch.cuda.current_stream()
        gate_up_side_stream = get_lora_side_stream()
        gate_up_side_stream.wait_stream(consumer_stream)
        single_gate_up = layer._w1_delta.shape[0] == 1
        if single_gate_up:
            a_gate_up = layer.gate_up_lora_a_weights[0, 0]
            gate_up_shrink = x_td.new_empty((t, a_gate_up.shape[0]))
        with torch.cuda.stream(gate_up_side_stream):
            if single_gate_up:
                torch.mm(x_td, a_gate_up.T, out=gate_up_shrink)
            else:
                from sglang.kernels.ops.gemm.trtllm_lora_temp.sgemm_lora_a import (
                    shared_sink_sgemm_lora_a_fwd,
                )

                gate_up_batch_info = layer.lora_backend._sgemm_info()
                gate_up_shrink = shared_sink_sgemm_lora_a_fwd(
                    x_td,
                    layer.gate_up_lora_a_weights[:, 0],
                    gate_up_batch_info,
                    stack_num=2,
                    padded_rank=True,
                    out_alloc_stream=consumer_stream,
                )

    y = torch.mm(x_td, w13_lin.T).view(t, n, -1)
    if run_lora:
        if not layer.experts_shared_outer_loras:
            _apply_per_expert_lora(
                layer,
                x_td,
                y,
                layer.gate_up_lora_a_weights,
                layer.gate_up_lora_b_weights,
            )
        elif overlap_gate_up:
            assert layer._w1_delta is not None
            assert gate_up_side_stream is not None
            assert gate_up_shrink is not None
            torch.cuda.current_stream().wait_stream(gate_up_side_stream)
            if layer._w1_delta.shape[0] == 1:
                y_flat = y.view(t, -1)
                y_flat.addmm_(gate_up_shrink, layer._w1_delta[0].T)
                y = y_flat.view(t, n, -1)
            else:
                from sglang.kernels.ops.gemm.trtllm_lora_temp.sgemm_lora_b import (
                    shared_sink_sgemm_lora_b_fwd,
                )

                assert gate_up_batch_info is not None
                y = shared_sink_sgemm_lora_b_fwd(
                    gate_up_shrink,
                    layer._w1_delta,
                    gate_up_batch_info,
                    y.view(t, -1),
                    apply_scaling=False,
                    padded_rank=True,
                ).view(t, n, -1)
        elif layer._w1_delta.shape[0] == 1:
            a_gate_up = layer.gate_up_lora_a_weights[0, 0]
            shrink = torch.mm(x_td, a_gate_up.T)
            y_flat = y.view(t, -1)
            y_flat.addmm_(shrink, layer._w1_delta[0].T)
            y = y_flat.view(t, n, -1)
        else:
            y = apply_multi_lora(
                layer,
                x_td,
                layer.gate_up_lora_a_weights[:, 0],
                layer._w1_delta,
                y.view(t, -1),
                stack_num=2,
            ).view(t, n, -1)

    act = layer._swiglu(y, gammas_ts)
    out_td = torch.mm(act.reshape(t, -1), w2_lin)
    if run_lora:
        if not layer.experts_shared_outer_loras:
            delta = torch.zeros(
                (t, n, out_td.shape[-1]), dtype=out_td.dtype, device=out_td.device
            )
            _apply_per_expert_lora(
                layer,
                act.reshape(t * n, -1),
                delta,
                layer.down_lora_a_weights,
                layer.down_lora_b_weights,
            )
            out_td.add_(delta.sum(dim=1))
        else:
            assert layer._a_cat is not None
            act_flat = act.reshape(t, -1)
            if layer._a_cat.shape[0] == 1:
                b_down = layer.down_lora_b_weights[0, 0]
                shrink = torch.mm(act_flat, layer._a_cat[0].T)
                out_td.addmm_(shrink, b_down.T)
            else:
                out_td = apply_multi_lora(
                    layer,
                    act_flat,
                    layer._a_cat,
                    layer.down_lora_b_weights[:, 0],
                    out_td,
                    stack_num=1,
                )
    if not use_reduce_scatter and layer.tp_group is not None:
        out_td = symm_mem_all_reduce(out_td, layer.tp_group)
    return out_td
