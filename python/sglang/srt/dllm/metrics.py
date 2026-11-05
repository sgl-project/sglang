"""Request-level dLLM work, independent of token-aligned denoising step maps."""


def make_dllm_metrics(forward_count: int, completion_tokens: int) -> dict:
    return {
        "forward_count": forward_count,
        "completion_tokens": completion_tokens,
        "tokens_per_forward": (
            completion_tokens / forward_count if forward_count else 0.0
        ),
    }
