import logging
from http import HTTPStatus
from typing import Optional

from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
from sglang.srt.managers.io_struct import (
    BatchTokenIDOut
)

logger = logging.getLogger(__name__)


def pack_err_batch_tokenid_out(req: Req):
    decode_ids, read_offset = req.init_incremental_detokenize()
    return BatchTokenIDOut(rids=[req.rid],
                           finished_reasons=[req.finished_reason.to_json()],
                           vids=[req.vid],
                           decoded_texts=[req.decoded_text],
                           decode_ids=[decode_ids],
                           read_offsets=[read_offset],
                           output_ids=[req.output_ids],
                           skip_special_tokens=[req.sampling_params.skip_special_tokens],
                           spaces_between_special_tokens=[req.sampling_params.spaces_between_special_tokens],
                           no_stop_trim=[req.sampling_params.no_stop_trim],
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
                           output_top_logprobs_idx=[req.output_top_logprobs_idx]
                          )


def validate_input_length(
    req: Req, max_req_input_len: int, allow_auto_truncate: bool
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
            logger.error(error_msg)
            req.finished_reason = FINISH_ABORT(
                error_msg, HTTPStatus.BAD_REQUEST, "BadRequestError"
            )
            return error_msg

    return None
