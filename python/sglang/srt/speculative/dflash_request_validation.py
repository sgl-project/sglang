from __future__ import annotations

from typing import Optional

from sglang.srt.sampling.sampling_params import TOP_K_ALL


def validate_dflash_request_options(
    *,
    return_logprob: bool,
    return_hidden_states: bool,
    sampling_params,
    enable_overlap: bool,
    enable_deterministic_inference: bool = False,
    sampling_backend: str | None = None,
) -> Optional[str]:
    if return_logprob:
        return "DFLASH speculative decoding does not support return_logprob yet."

    if enable_overlap and return_hidden_states:
        return "DFLASH speculative decoding does not support return_hidden_states yet."

    if (
        sampling_params.json_schema is not None
        or sampling_params.regex is not None
        or sampling_params.ebnf is not None
        or sampling_params.structural_tag is not None
    ):
        return (
            "DFLASH speculative decoding does not support "
            "grammar-constrained decoding yet."
        )

    sampling_seed = getattr(sampling_params, "sampling_seed", None)
    top_k = getattr(sampling_params, "top_k", None)
    if (
        sampling_seed is not None
        and top_k is not None
        and int(top_k) > 1
        and sampling_backend != "pytorch"
    ):
        if enable_deterministic_inference and _is_unfiltered_non_greedy(
            sampling_params
        ):
            return None
        if enable_deterministic_inference:
            return (
                "DFLASH speculative decoding with seeded filtered sampling "
                "requires --sampling-backend pytorch."
            )
        return (
            "DFLASH speculative decoding with seeded non-greedy sampling "
            "requires --sampling-backend pytorch unless deterministic inference is enabled."
        )

    return None


def _is_unfiltered_non_greedy(sampling_params) -> bool:
    return (
        int(getattr(sampling_params, "top_k", TOP_K_ALL)) == TOP_K_ALL
        and float(getattr(sampling_params, "top_p", 1.0)) == 1.0
        and float(getattr(sampling_params, "min_p", 0.0)) == 0.0
    )
