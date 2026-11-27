"""TOP bindings for the GLM (DeepSeek-V2 family) gate, MoE block and model admission."""


def top_glm_gate_enabled(*, is_deepseek_v4=False):
    """GLM's BF16 gate GEMM; DeepSeek-V4 keeps the deterministic FP32-output gate."""
    from sglang.srt.true_on_policy.moe import should_use_slot_order_moe_combine

    return not is_deepseek_v4 and should_use_slot_order_moe_combine()


def top_glm_gate_logits(hidden_states, weight):
    import torch
    import torch.nn.functional as F

    if hidden_states.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise NotImplementedError("[top] GLM router requires BF16 operands and logits")
    # Preserve the reference gate's BF16 rounding before sigmoid routing.
    return F.linear(hidden_states, weight, None)


def top_glm_router_cls(*, is_deepseek_v4=False):
    """TopGLMRouter under TOP for GLM, None otherwise."""
    from sglang.srt.true_on_policy.moe import should_use_slot_order_moe_combine

    if is_deepseek_v4 or not should_use_slot_order_moe_combine():
        return None
    from sglang.srt.true_on_policy.glm_router import TopGLMRouter

    return TopGLMRouter


def configure_top_glm_moe(*, experts, num_fused_shared_experts):
    """Whether DeepseekV2MoE combines its slots on the owner (TOP); refuses slots the combine cannot take."""
    from sglang.srt.true_on_policy.moe import should_use_slot_order_moe_combine

    slot_combine = should_use_slot_order_moe_combine()
    if slot_combine and (
        num_fused_shared_experts or experts.should_fuse_routed_scaling_factor_in_topk
    ):
        raise NotImplementedError(
            "[top] GLM scaling/shared slots conflict with the owner-side combine"
        )
    return slot_combine


def top_glm_combine(final_hidden_states, shared_output):
    # Dispatch/combine already folded routed scaling into FP32 weights.
    return (
        final_hidden_states
        if shared_output is None
        else final_hidden_states + shared_output
    )


def validate_top_glm(config, quant_config):
    from sglang.srt.true_on_policy import is_true_on_policy_enabled

    if is_true_on_policy_enabled():
        from sglang.srt.true_on_policy.glm import validate_glm

        validate_glm(config, quant_config)


def top_allows_min_latency_q_b_gemm():
    from sglang.srt.true_on_policy import is_true_on_policy_enabled

    # Small-batch fused GEMM does not match the paired TOP matmul.
    return not is_true_on_policy_enabled()
