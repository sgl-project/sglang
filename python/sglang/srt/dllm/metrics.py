"""Cumulative request work, independent of token-aligned denoising step maps."""


def make_dllm_metrics(
    forward_count: int,
    completion_tokens: int,
    *,
    prompt_forward_count: int,
    generation_forward_count: int,
    recompute_forward_count: int,
) -> dict:
    return {
        "forward_count": forward_count,
        "prompt_forward_count": prompt_forward_count,
        "generation_forward_count": generation_forward_count,
        "recompute_forward_count": recompute_forward_count,
        "completion_tokens": completion_tokens,
        "tokens_per_forward": (
            completion_tokens / generation_forward_count
            if generation_forward_count
            else 0.0
        ),
    }
