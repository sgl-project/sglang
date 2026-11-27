from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


_HY4_FLASH_ARCHITECTURE = "HYV4FlashForConditionalGeneration"


def resolve_hy4_dfly_sampling_defaults(
    sampling_params: dict,
    *,
    request_sampling: dict,
    architectures: Sequence[str],
) -> dict:
    """Use greedy Hy4 verification unless the request chose a sampling policy.

    Callers must gate this on DFLY and retain explicit-field provenance before
    merging protocol or generation_config defaults.
    """
    if not any(
        arch in ("HYV4ForCausalLM", "HYV4VLForConditionalGeneration")
        for arch in architectures
    ):
        return sampling_params
    if any(
        request_sampling.get(name) is not None
        for name in ("temperature", "top_k", "top_p", "min_p")
    ):
        return sampling_params
    return sampling_params | {"temperature": 0.0, "top_k": 1}


def dfly_targets_hy4_flash(architectures: Optional[Sequence[str]]) -> bool:
    return bool(architectures) and architectures[0] == _HY4_FLASH_ARCHITECTURE


def validate_dfly_request(
    req: Req, *, enable_overlap: bool, targets_hy4_flash: bool
) -> Optional[str]:
    if targets_hy4_flash:
        from sglang.srt.speculative.dflash_utils import validate_dflash_request

        return validate_dflash_request(
            req, enable_overlap, bound_runtime=False, supports_grammar=False
        )
    return _validate_hy4_dfly_request(req)


def _validate_hy4_dfly_request(req: Req) -> Optional[str]:
    """Error for request features the greedy DFly verify cannot honour."""
    sampling_params = req.sampling_params
    if sampling_params.top_k != 1:
        return (
            "DFLY speculative decoding verifies greedily; send temperature=0 "
            "or top_k=1."
        )
    has_grammar = req.grammar is not None or any(
        getattr(sampling_params, field) is not None
        for field in ("json_schema", "regex", "ebnf", "structural_tag")
    )
    if has_grammar:
        return "DFLY speculative decoding does not support grammar decoding."
    if (
        sampling_params.frequency_penalty != 0.0
        or sampling_params.presence_penalty != 0.0
        or sampling_params.repetition_penalty != 1.0
        or sampling_params.min_new_tokens > 0
        or sampling_params.logit_bias is not None
    ):
        return "DFLY speculative decoding does not support penalties or logit bias."
    if req.custom_logit_processor:
        return "DFLY speculative decoding does not support custom logit processors."
    if req.return_logprob:
        return "DFLY speculative decoding does not support returned logprobs."
    if req.return_hidden_states_mode.need_capture():
        return "DFLY speculative decoding does not support return_hidden_states."
    return None
