"""Request-admission validation for UNO speculative decoding."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


def validate_uno_request(req: Req) -> Optional[str]:
    """Return an error for request features that UNO cannot execute."""

    sampling_params = req.sampling_params

    if sampling_params.min_p > 0.0:
        return "UNO speculative decoding does not support min_p sampling."

    has_grammar = req.grammar is not None or any(
        getattr(sampling_params, field) is not None
        for field in ("json_schema", "regex", "ebnf", "structural_tag")
    )
    if has_grammar:
        return "UNO speculative decoding does not support grammar decoding."

    if req.return_logprob:
        return "UNO speculative decoding does not support returned logprobs."

    if req.return_hidden_states_mode.need_capture():
        return "UNO speculative decoding does not support return_hidden_states."

    has_penalties = (
        sampling_params.frequency_penalty != 0.0
        or sampling_params.presence_penalty != 0.0
        or sampling_params.repetition_penalty != 1.0
        or sampling_params.min_new_tokens > 0
    )
    if has_penalties:
        return "UNO speculative decoding does not support sampling penalties."

    if sampling_params.logit_bias is not None:
        return "UNO speculative decoding does not support logit_bias."

    if req.custom_logit_processor:
        return "UNO speculative decoding does not support custom logit processors."

    if req.lora_id is not None:
        return "UNO speculative decoding does not support request-selectable LoRA."

    return None
