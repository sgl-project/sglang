"""Shared TOP EP operation constraints and qualified expert configurations."""

import msgspec

from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.moe.utils import get_deepep_mode, get_moe_runner_backend
from sglang.srt.runtime_context import get_exec, get_server_args
from sglang.srt.true_on_policy.contracts import TRUE_ON_POLICY_V1, data_parallel_layout
from sglang.srt.true_on_policy.qwen_moe_router import SHARED_GATE_HIDDEN_SIZES


def validate_one_expert_domain(args) -> None:
    """Experts form one EP domain across the engine; the op registry owns the EP degree."""
    if args.moe_dp_size != 1 or data_parallel_layout(args)[1] != 1:
        raise NotImplementedError(
            "[top] EP requires one expert domain: no MoE DP and no attention-DP replicas"
        )


def validate_ep_runtime():
    """Validate the TOP EP runtime and transport; the op registry admits the layout."""
    args = get_server_args()
    if args.true_on_policy_contract != TRUE_ON_POLICY_V1:
        raise NotImplementedError("[top] EP requires the true_on_policy_v1 contract")
    if not get_moe_a2a_backend().is_deepep():
        raise NotImplementedError(
            "[top] EP requires deepep transport with DeepGEMM experts"
        )
    if not get_moe_runner_backend().is_deep_gemm():
        raise NotImplementedError("[top] EP requires DeepGEMM experts")
    if not deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
        raise NotImplementedError("[top] EP requires enabled JIT DeepGEMM")
    if (
        args.dtype not in ("auto", "bfloat16")
        or args.deepep_dispatcher_output_dtype != "bf16"
    ):
        # A server-wide FP8 wire also covers BF16-expert layers and quantizes prefill with stock's rule (1e-10 floor,
        # FP32 amax/448 scales on Hopper); TOP's FP8 experts turn on FP8 in both DeepEP modes per layer (slot_order_deepep).
        raise NotImplementedError(
            "[top] EP requires BF16 model math and --deepep-dispatcher-output-dtype bf16"
        )
    validate_one_expert_domain(args)
    moe = get_exec().moe
    if (
        moe.enable_eplb
        or moe.init_expert_location != "trivial"
        or moe.ep_num_redundant_experts
    ):
        raise NotImplementedError("[top] EP requires fixed contiguous expert ownership")


class QualifiedExperts(msgspec.Struct, frozen=True, kw_only=True):
    """Expert configuration a TOP MoE block was qualified on."""

    name: str
    block: str  # the SGLang sparse MoE block whose TOP binding runs it
    # (hidden, expert intermediate, experts, top-k); None admits any size of model_type.
    shape: tuple[int, int, int, int] | None
    model_type: str | None = None


QUALIFIED_EXPERTS = (
    QualifiedExperts(
        name="Qwen3-30B-A3B",
        block="Qwen3MoeSparseMoeBlock",
        shape=(2048, 768, 128, 8),
    ),
    # Every Qwen3.5/3.6 MoE size runs the same row-wise expert program; only the counts differ.
    QualifiedExperts(
        name="Qwen3.5/3.6 MoE",
        block="Qwen2MoeSparseMoeBlock",
        shape=None,
        model_type="qwen3_5_moe_text",
    ),
    QualifiedExperts(
        name="Qwen3.8-Flash-Next",
        block="Qwen2MoeSparseMoeBlock",
        shape=(2560, 640, 512, 10),
    ),
)


def _qualified_experts(config, *, block) -> QualifiedExperts:
    shape = (
        config.hidden_size,
        config.moe_intermediate_size,
        config.num_experts,
        config.num_experts_per_tok,
    )
    qualified = [q for q in QUALIFIED_EXPERTS if q.block == block]
    match = next(
        (
            q
            for q in qualified
            if q.shape == shape
            or (q.shape is None and q.model_type == config.model_type)
        ),
        None,
    )
    if match is None or not getattr(config, "norm_topk_prob", True):
        names = ", ".join(q.name for q in qualified)
        raise NotImplementedError(f"[top] expected the {names} expert configuration")
    if match.shape is None and config.hidden_size not in SHARED_GATE_HIDDEN_SIZES:
        raise NotImplementedError(
            f"[top] the TOP shared-expert gate is qualified for hidden sizes {SHARED_GATE_HIDDEN_SIZES}"
        )
    return match


def validate_qwen_moe(config, quant_config, *, block):
    """Admit BF16 SwiGLU experts qualified on ``block`` and their EP runtime."""
    if quant_config is not None or getattr(config, "hidden_act", "silu") != "silu":
        raise NotImplementedError(
            "[top] Qwen MoE requires unquantized BF16 SwiGLU experts"
        )
    _qualified_experts(config, block=block)
    validate_ep_runtime()
    args = get_server_args()
    if args.kv_cache_dtype not in ("auto", "bf16", "bfloat16"):
        raise NotImplementedError("[top] Qwen MoE requires BF16 KV cache")
    if get_deepep_mode().value != "auto":
        raise NotImplementedError(
            "[top] Qwen MoE requires normal prefill and low-latency decode (auto)"
        )
