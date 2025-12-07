from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import GenerationBatchResult

logger = logging.getLogger(__name__)


def validate_input_length(
    input_ids: List[int],
    max_req_input_len: int,
    allow_auto_truncate: bool,
    revised_input_token_num: int = None,
    is_session_validation: bool = False,
) -> Optional[str]:
    """Validate and potentially truncate input length.

    Args:
        input_ids: The input_ids to validate
        max_req_input_len: Maximum allowed input length
        allow_auto_truncate: Whether to truncate long inputs
        revised_input_token_num: revised input token counts for comparison.
            If not provided,fall back to the input_ids length
        is_session_validation: If the validation is for the aggregated session
            request in scheduler or a single request in tokenizer.

    Returns:
        Error message if validation fails, None if successful
    """
    request_type_str = (
        "aggregated session request" if is_session_validation else "request"
    )
    length_limit_type_str = (
        "maximum allowed length" if is_session_validation else "model's context length"
    )

    input_token_num = (
        revised_input_token_num if revised_input_token_num else len(input_ids)
    )

    if input_token_num >= max_req_input_len:
        if allow_auto_truncate:
            logger.warning(
                f"Length of the {request_type_str} is longer than the KV cache pool "
                "size or the max context length. Truncated. "
                f"{input_token_num=}, {max_req_input_len=}."
            )
            input_ids = input_ids[:max_req_input_len]
            return None
        else:
            error_msg = (
                f"Input length ({input_token_num} tokens) of the {request_type_str} "
                f"exceeds the {length_limit_type_str} ({max_req_input_len} tokens). "
                f"Use a shorter input or enable --allow-auto-truncate."
            )
            return error_msg

    return None


def get_logprob_dict_from_result(result: GenerationBatchResult) -> dict:

    logits_output = result.logits_output
    assert logits_output is not None

    return {
        "extend_input_len_per_req": result.extend_input_len_per_req,
        "extend_logprob_start_len_per_req": result.extend_logprob_start_len_per_req,
        "next_token_logprobs": result.logits_output.next_token_logprobs,
        "next_token_top_logprobs_val": result.logits_output.next_token_top_logprobs_val,
        "next_token_top_logprobs_idx": result.logits_output.next_token_top_logprobs_idx,
        "next_token_token_ids_logprobs_val": result.logits_output.next_token_token_ids_logprobs_val,
        "next_token_token_ids_logprobs_idx": result.logits_output.next_token_token_ids_logprobs_idx,
        "input_token_logprobs": result.logits_output.input_token_logprobs,
        "input_top_logprobs_val": result.logits_output.input_top_logprobs_val,
        "input_top_logprobs_idx": result.logits_output.input_top_logprobs_idx,
        "input_token_ids_logprobs_val": result.logits_output.input_token_ids_logprobs_val,
        "input_token_ids_logprobs_idx": result.logits_output.input_token_ids_logprobs_idx,
    }


def get_logprob_from_pp_outputs(
    next_pp_outputs: PPProxyTensors,
) -> tuple[LogitsProcessorOutput, list[int], list[int]]:
    logits_output = LogitsProcessorOutput(
        # Do not send logits and hidden states because they are large
        next_token_logits=None,
        hidden_states=None,
        next_token_logprobs=next_pp_outputs["next_token_logprobs"],
        next_token_top_logprobs_val=next_pp_outputs["next_token_top_logprobs_val"],
        next_token_top_logprobs_idx=next_pp_outputs["next_token_top_logprobs_idx"],
        next_token_token_ids_logprobs_val=next_pp_outputs[
            "next_token_token_ids_logprobs_val"
        ],
        next_token_token_ids_logprobs_idx=next_pp_outputs[
            "next_token_token_ids_logprobs_idx"
        ],
        input_token_logprobs=next_pp_outputs["input_token_logprobs"],
        input_top_logprobs_val=next_pp_outputs["input_top_logprobs_val"],
        input_top_logprobs_idx=next_pp_outputs["input_top_logprobs_idx"],
        input_token_ids_logprobs_val=next_pp_outputs["input_token_ids_logprobs_val"],
        input_token_ids_logprobs_idx=next_pp_outputs["input_token_ids_logprobs_idx"],
    )
    extend_input_len_per_req = next_pp_outputs["extend_input_len_per_req"]
    extend_logprob_start_len_per_req = next_pp_outputs[
        "extend_logprob_start_len_per_req"
    ]

    return logits_output, extend_input_len_per_req, extend_logprob_start_len_per_req
