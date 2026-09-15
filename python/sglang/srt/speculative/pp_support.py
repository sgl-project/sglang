"""Model allowlists for pipeline-parallel speculative prefill."""

QWEN35_PP_MTP_PREFILL_ARCHITECTURES = frozenset(
    {
        "Qwen3_5ForCausalLM",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }
)


def supports_qwen35_pp_mtp_prefill(model_architecture: str) -> bool:
    return model_architecture in QWEN35_PP_MTP_PREFILL_ARCHITECTURES
