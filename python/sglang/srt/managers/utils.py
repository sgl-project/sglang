import logging
from http import HTTPStatus
from typing import Optional

from sglang.srt.managers.env_vars import CLIP_MAX_NEW_TOKENS_ESTIMATION
from sglang.srt.managers.io_struct import BatchStrOut
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req

logger = logging.getLogger(__name__)


def pack_err_batch_str_output(req: Req) -> BatchStrOut:
    decode_ids, read_offset = req.init_incremental_detokenize()
    return BatchStrOut(
        rids=[req.rid],
        finished_reasons=[req.finished_reason.to_json()],
        output_strs=[req.decoded_text],
        output_ids=[decode_ids],
        prompt_tokens=[len(req.origin_input_ids)],
        completion_tokens=[len(req.output_ids)],
        cached_tokens=[req.cached_tokens],
        spec_verify_ct=[req.spec_verify_ct],
        input_token_logprobs_val=[req.input_token_logprobs_val],
        input_token_logprobs_idx=[req.input_token_logprobs_idx],
        output_token_logprobs_val=[req.output_token_logprobs_val],
        output_token_logprobs_idx=[req.output_token_logprobs_idx],
        input_top_logprobs_val=[req.input_top_logprobs_val],
        input_top_logprobs_idx=[req.input_top_logprobs_idx],
        output_top_logprobs_val=[req.output_top_logprobs_val],
        output_top_logprobs_idx=[req.output_top_logprobs_idx],
        input_token_ids_logprobs_val=[req.input_token_ids_logprobs_val],
        input_token_ids_logprobs_idx=[req.input_token_ids_logprobs_idx],
        output_token_ids_logprobs_val=[req.output_token_ids_logprobs_val],
        output_token_ids_logprobs_idx=[req.output_token_ids_logprobs_idx],
        output_hidden_states=[req.hidden_states],
    )


def validate_input_length(
    req: Req,
    max_req_input_len: int,
    allow_auto_truncate: bool,
    cur_rem_tokens_len: int,
    could_wait: bool,
) -> Optional[str]:
    """Validate and potentially truncate input length.

    Args:
        req: The request containing input_ids to validate
        max_req_input_len: Maximum allowed input length
        allow_auto_truncate: Whether to truncate long inputs

    Returns:
        Error message if validation fails, None if successful
    """
    if len(req.origin_input_ids) >= max_req_input_len:
        if allow_auto_truncate:
            logger.warning(
                "Request length is longer than the KV cache pool size or "
                "the max context length. Truncated. "
                f"{len(req.origin_input_ids)=}, {max_req_input_len=}."
            )
            req.origin_input_ids = req.origin_input_ids[:max_req_input_len]
            return None
        else:
            error_msg = (
                f"Input length ({len(req.origin_input_ids)} tokens) exceeds "
                f"the maximum allowed length ({max_req_input_len} tokens). "
                f"Use a shorter input or enable --allow-auto-truncate."
            )
            return error_msg

    req_max_new_tokens = (
        req.sampling_params.max_new_tokens
        if req.sampling_params.max_new_tokens is not None
        else 0
    )
    cur_need_tokens = len(req.origin_input_ids) + min(
        req_max_new_tokens, CLIP_MAX_NEW_TOKENS_ESTIMATION
    )
    if not could_wait:
        if cur_need_tokens > cur_rem_tokens_len:
            error_msg = (
                f"Current need ({cur_need_tokens} tokens) exceeds "
                f"the current allowed ({cur_rem_tokens_len} tokens). "
                f"Use a shorter context req or increase max-waiting-requests."
            )
            logger.error(error_msg)
            req.finished_reason = FINISH_ABORT(
                error_msg, HTTPStatus.FORBIDDEN, "Request is Forbidden"
            )
            return error_msg
    return None
