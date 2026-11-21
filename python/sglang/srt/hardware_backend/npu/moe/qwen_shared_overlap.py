"""Schedule Qwen's replicated shared MLP at the NPU routed-expert milestones."""
from contextlib import ExitStack

import torch
from sgl_kernel_npu.activation.fused_sigmoid_mul import fused_sigmoid_mul_broadcast

from sglang.srt.hardware_backend.npu.cmo import get_share_stream, set_share_stream
from sglang.srt.layers.moe.token_dispatcher.ascend_ops import AscendOpsDispatcher
from sglang.srt.model_executor.forward_mode import ForwardMode


def supports_staged_shared(block, hidden, batch):
    mode = getattr(batch, "_original_forward_mode", None) or batch.forward_mode
    return (
        mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND_V2)
        and 0 < hidden.shape[0] <= 128
        and hidden.dtype == torch.bfloat16
        and hidden.shape[-1] == 2560
        and block.shared_expert is not None
        and block.shared_expert_gate is not None
        and isinstance(getattr(block.experts, "dispatcher", None), AscendOpsDispatcher)
        and getattr(getattr(block.shared_expert.gate_up_proj, "weight", None), "dtype", None) == torch.bfloat16
        and getattr(getattr(block.shared_expert.down_proj, "weight", None), "dtype", None) == torch.bfloat16
    )


def staged_shared_forward(block, hidden, logits):
    main = torch.npu.current_stream()
    side = get_share_stream()
    if side is None:
        side = torch.npu.Stream()
        set_share_stream(side)
    mlp = block.shared_expert
    dispatcher = block.experts.dispatcher
    # Keep the shared branch independent of any routed activation reuse.
    shared_input = hidden.clone()
    side.wait_stream(main)
    with torch.npu.stream(side):
        shared_input.record_stream(side)
        gate_up, _ = mlp.gate_up_proj(shared_input)
    activated = output = None

    def after_dispatch(*_):
        nonlocal activated
        side.wait_stream(main)
        with torch.npu.stream(side):
            activated = mlp.act_fn(gate_up)

    def before_combine(*_):
        nonlocal output
        if activated is None:
            after_dispatch()
        side.wait_stream(main)
        with torch.npu.stream(side):
            down, _ = mlp.down_proj(activated)
            gate = block.shared_expert_gate(shared_input)
            output = fused_sigmoid_mul_broadcast(down, gate)

    try:
        topk = block.topk(hidden, logits)
        with ExitStack() as hooks:
            hooks.callback(dispatcher.register_post_dispatch_hook(after_dispatch).remove)
            hooks.callback(dispatcher.register_pre_combine_hook(before_combine).remove)
            routed = block.experts(hidden_states=hidden, topk_output=topk)
        if output is None:
            before_combine()
    finally:
        main.wait_stream(side)
    output.record_stream(main)
    return routed.add_(output)
