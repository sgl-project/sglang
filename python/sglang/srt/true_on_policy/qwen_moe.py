"""TOP bindings for the Qwen3 MoE sparse block: admission, the shared EP1 forward and DeepEP routing."""

from sglang.srt.true_on_policy.config import is_true_on_policy_enabled


class TopQwenMoe:
    """Qwen3MoeSparseMoeBlock's TOP program; build() returns None when TOP is off."""

    @classmethod
    def build(cls, *, config, quant_config, ep_size):
        if not is_true_on_policy_enabled():
            return None
        return cls(config=config, quant_config=quant_config, ep_size=ep_size)

    def __init__(self, *, config, quant_config, ep_size):
        from sglang.srt.runtime_context import get_exec, get_parallel
        from sglang.srt.true_on_policy.moe import refuse_routed_experts_capture

        refuse_routed_experts_capture("Qwen3 MoE")
        self.ep1 = ep_size == 1
        self.top_k = config.num_experts_per_tok
        if not self.ep1:
            from sglang.srt.true_on_policy.moe_ep import validate_qwen_moe

            validate_qwen_moe(config, quant_config, block="Qwen3MoeSparseMoeBlock")
            return
        from sglang.srt.layers.moe.utils import get_moe_runner_backend

        if get_parallel().moe_dp_size != 1:
            raise NotImplementedError("[top] Qwen MoE EP1 requires no MoE DP")
        if quant_config is not None:
            raise NotImplementedError(
                "[top] Qwen MoE requires unquantized BF16 experts"
            )
        if not get_moe_runner_backend().is_deep_gemm():
            raise NotImplementedError("[top] Qwen MoE requires DeepGEMM experts")
        if getattr(config, "hidden_act", "silu") != "silu":
            raise NotImplementedError("[top] Qwen MoE requires SiLU experts")
        if get_exec().moe.enable_eplb or get_exec().moe.ep_num_redundant_experts:
            raise NotImplementedError("[top] EP1 requires fixed expert ownership")
        if not getattr(config, "norm_topk_prob", True):
            raise NotImplementedError(
                "[top] Qwen MoE requires normalized top-k weights"
            )

    def forward(self, hidden_states, *, gate_weight, experts):
        """The shared EP1 forward; None when the DeepEP path runs."""
        if not self.ep1:
            return None
        from sglang.srt.true_on_policy.moe import local_experts, route_qwen

        weights, ids = route_qwen(hidden_states, gate_weight, self.top_k)
        return local_experts(
            hidden_states, experts.w13_weight, experts.w2_weight, weights, ids
        )

    def deepep_topk(self, hidden_states, *, gate_weight):
        from sglang.srt.layers.moe.topk import StandardTopKOutput
        from sglang.srt.true_on_policy.moe import route_qwen

        weights, ids = route_qwen(hidden_states, gate_weight, self.top_k)
        return StandardTopKOutput(weights, ids, None)
