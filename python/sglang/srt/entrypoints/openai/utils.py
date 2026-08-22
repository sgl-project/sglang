import logging
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    Union,
)

import torch

from sglang.srt.entrypoints.openai.protocol import (
    CachedTokensDetails,
    ChatCompletionRequest,
    CompletionRequest,
    LogProbs,
    SpecTokensDetails,
    StreamOptions,
)

logger = logging.getLogger(__name__)


def to_openai_style_logprobs(
    input_token_logprobs=None,
    output_token_logprobs=None,
    input_top_logprobs=None,
    output_top_logprobs=None,
):
    ret_logprobs = LogProbs()

    def append_token_logprobs(token_logprobs):
        for logprob, _, token_text in token_logprobs:
            ret_logprobs.tokens.append(token_text)
            ret_logprobs.token_logprobs.append(logprob)

            # Not supported yet
            ret_logprobs.text_offset.append(-1)

    def append_top_logprobs(top_logprobs):
        for tokens in top_logprobs:
            if tokens is not None:
                ret_logprobs.top_logprobs.append(
                    {token[2]: token[0] for token in tokens}
                )
            else:
                ret_logprobs.top_logprobs.append(None)

    if input_token_logprobs is not None:
        append_token_logprobs(input_token_logprobs)
    if output_token_logprobs is not None:
        append_token_logprobs(output_token_logprobs)
    if input_top_logprobs is not None:
        append_top_logprobs(input_top_logprobs)
    if output_top_logprobs is not None:
        append_top_logprobs(output_top_logprobs)

    return ret_logprobs


class _TokenLogprob(Protocol):
    """Shared shape of the per-token logprob entries of both APIs."""

    token: str


_TokenLogprobT = TypeVar("_TokenLogprobT", bound=_TokenLogprob)


def align_token_logprobs_to_text(
    entries: Sequence[_TokenLogprobT],
    strip_input: str,
    cleaned: str,
    kept_spans: Sequence[tuple[int, int]],
) -> Optional[List[_TokenLogprobT]]:
    """Keep entries whose positional spans survive text sanitization.

    The generated tokens must either equal ``strip_input`` or end with it;
    the latter accounts for reasoning tokens emitted before content. Entries
    are retained only when their full positional span lies inside one kept
    span. Return ``None`` when exact alignment is impossible, since a partial
    array is more misleading than omitting logprobs entirely.
    """
    full_parts = []
    for entry in entries:
        if entry.token is None:
            return None
        full_parts.append(entry.token)
    full = "".join(full_parts)
    if full == strip_input:
        offset = 0
    elif strip_input and full.endswith(strip_input):
        offset = len(full) - len(strip_input)
    else:
        return None

    aligned: List[_TokenLogprobT] = []
    cursor = 0
    for entry in entries:
        token = entry.token
        end = cursor + len(token)
        shifted_start = cursor - offset
        shifted_end = end - offset
        if (
            token
            and end > offset
            and any(
                span_start <= shifted_start and shifted_end <= span_end
                for span_start, span_end in kept_spans
            )
        ):
            aligned.append(entry)
        cursor = end
    return aligned if "".join(entry.token for entry in aligned) == cleaned else None


def process_hidden_states_from_ret(
    ret_item: Dict[str, Any],
    request: Union[
        ChatCompletionRequest,
        CompletionRequest,
    ],
) -> Optional[List]:
    """Process hidden states from a ret item in non-streaming response.

    Args:
        ret_item: Response item containing meta_info
        request: The original request object

    Returns:
        Processed hidden states for the last token, or None
    """
    if not request.return_hidden_states:
        return None

    hidden_states = ret_item["meta_info"].get("hidden_states", None)
    return process_hidden_states_for_response(
        hidden_states, request.return_hidden_states
    )


def process_hidden_states_for_response(
    hidden_states: Optional[List],
    return_hidden_states: Union[bool, Literal["last"]],
) -> Optional[List]:
    """Format scheduler hidden states for OpenAI API responses."""
    if not return_hidden_states or hidden_states is None:
        return None
    if return_hidden_states == "last":
        return hidden_states
    return hidden_states[-1] if len(hidden_states) > 1 else []


def should_include_usage(
    stream_options: StreamOptions | None, stream_response_default_include_usage: bool
) -> tuple[bool, bool]:
    # When stream_options are specified in the request
    if stream_options:
        include_usage = (
            stream_options.include_usage or stream_response_default_include_usage
        )
        continuous_usage_stats = bool(stream_options.continuous_usage_stats)
    else:
        include_usage, continuous_usage_stats = (
            stream_response_default_include_usage,
            False,
        )
    return include_usage, continuous_usage_stats


def process_routed_experts_from_ret(
    ret_item: Dict[str, Any],
    request: Union[
        ChatCompletionRequest,
        CompletionRequest,
    ],
) -> Optional[str]:
    """Process routed experts from a ret item in non-streaming response."""
    if not getattr(request, "return_routed_experts", False):
        return None
    return ret_item["meta_info"].get("routed_experts", None)


def cached_tokens_details_from_dict(
    details: Dict[str, Any],
) -> CachedTokensDetails:
    """Convert a raw cached_tokens_details dict to a CachedTokensDetails object."""
    if "storage" in details:
        return CachedTokensDetails(
            device=details.get("device", 0),
            host=details.get("host", 0),
            storage=details.get("storage", 0),
            storage_backend=details.get("storage_backend"),
        )
    else:
        return CachedTokensDetails(
            device=details.get("device", 0),
            host=details.get("host", 0),
        )


def process_cached_tokens_details_from_ret(
    ret_item: Dict[str, Any],
    request: Union[
        ChatCompletionRequest,
        CompletionRequest,
    ],
) -> Optional[CachedTokensDetails]:
    """Process cached tokens details from a ret item in non-streaming response."""
    if not request.return_cached_tokens_details:
        return None

    details = ret_item["meta_info"].get("cached_tokens_details", None)
    if details is None:
        return None

    return cached_tokens_details_from_dict(details)


def spec_tokens_details_from_meta_info(
    meta_info: Dict[str, Any],
) -> Optional[SpecTokensDetails]:
    """Build speculative decoding details from canonical or legacy metrics."""
    details = dict(meta_info)

    metric_keys = (
        "spec_accept_rate",
        "spec_accept_length",
        "spec_cap_length",
        "spec_block_accept_length",
        "spec_num_correct_drafts",
        "spec_num_proposed_drafts",
        "spec_verify_ct",
        "spec_correct_drafts_histogram",
        "spec_cap_lens_histogram",
    )
    if not any(key in details for key in metric_keys):
        return None

    return SpecTokensDetails(
        spec_accept_rate=details.get("spec_accept_rate") or 0.0,
        spec_accept_length=details.get("spec_accept_length") or 0.0,
        spec_cap_length=details.get("spec_cap_length") or 0.0,
        spec_block_accept_length=details.get("spec_block_accept_length") or 0.0,
        spec_num_correct_drafts=details.get("spec_num_correct_drafts") or 0,
        spec_num_proposed_drafts=details.get("spec_num_proposed_drafts") or 0,
        spec_verify_ct=details.get("spec_verify_ct") or 0,
        spec_correct_drafts_histogram=details.get("spec_correct_drafts_histogram")
        or [],
        spec_cap_lens_histogram=details.get("spec_cap_lens_histogram") or [],
    )


def process_spec_tokens_details_from_ret(
    ret_item: Dict[str, Any],
    request: Union[
        ChatCompletionRequest,
        CompletionRequest,
    ],
) -> Optional[SpecTokensDetails]:
    """Process speculative decoding details from a response item."""
    if not getattr(request, "return_spec_tokens_details", False):
        return None
    return spec_tokens_details_from_meta_info(ret_item["meta_info"])


def convert_embeds_to_tensors(
    embeds: Optional[Union[List[Optional[List[List[float]]]], List[List[float]]]],
) -> Optional[List[Optional[List[torch.Tensor]]]]:
    """Convert nested float lists from the HTTP API to lists of tensors.

    Accepts either:
      - None -> returns None
      - List[List[float]] (single input) -> [[tensor, ...]]
      - List[Optional[List[List[float]]]] (batch) -> [Optional[List[tensor]], ...]
    Each innermost List[float] becomes a 1-D torch.Tensor.
    Per-input None entries are preserved (no overrides for that input).
    """
    if embeds is None:
        return None
    if len(embeds) == 0:
        return []
    # Find first non-None entry to detect nesting depth
    first_non_none = next((e for e in embeds if e is not None), None)
    if first_non_none is None:
        # All entries are None
        return [None] * len(embeds)
    # Detect nesting depth by checking the first non-None entry:
    # - Single input [num_replacements][hidden_size]: first element is List[float]
    # - Batch [num_inputs][num_replacements][hidden_size]: first element is List[List[float]]
    if not first_non_none or not isinstance(first_non_none[0], list):
        # Single input: each entry is a float vector
        return [[torch.tensor(vec, dtype=torch.float32) for vec in embeds]]
    # Otherwise it's batch: [num_inputs][num_replacements][hidden_size]
    return [
        (
            [torch.tensor(vec, dtype=torch.float32) for vec in per_input]
            if per_input is not None
            else None
        )
        for per_input in embeds
    ]
