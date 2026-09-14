import logging
from typing import Any, Dict, List, Literal, Optional, Union

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

# GPT-2 style byte-level BPE decoder table (char -> raw byte). Byte-level BPE
# vocab tokens are stored as a printable-char mapping of the raw UTF-8 bytes
# (see openai/gpt-2 bytes_to_unicode); converting a token id back to its raw
# bytes must go through this table, NOT through `token.encode()` on the
# detokenized display string (that loses fragmentary bytes as U+FFFD).
_BYTE_DECODER: Dict[str, int] = {}


def _build_byte_decoder() -> Dict[str, int]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord(chr(0xA1)), ord(chr(0xAC)) + 1))
        + list(range(ord(chr(0xAE)), ord(chr(0xFF)) + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(cs, bs))


def token_id_to_bytes(tokenizer, token_id) -> Optional[List[int]]:
    """Raw UTF-8 bytes for a byte-level-BPE token id.

    Returns the token's original bytes via the GPT-2 byte decoder, or None when
    the token is not byte-level representable (e.g. special ids / non byte BPE),
    so callers can fall back to the detokenized display string.
    """
    global _BYTE_DECODER
    if not _BYTE_DECODER:
        _BYTE_DECODER = _build_byte_decoder()
    try:
        piece = tokenizer.convert_ids_to_tokens(token_id)
    except Exception:
        return None
    if piece is None:
        return None
    out = bytearray()
    for ch in piece:
        b = _BYTE_DECODER.get(ch)
        if b is None:
            return None
        out.append(b)
    if not out:
        return None
    return list(out)


def to_openai_style_logprobs(
    input_token_logprobs=None,
    output_token_logprobs=None,
    input_top_logprobs=None,
    output_top_logprobs=None,
    tokenizer=None,
):
    """Convert engine logprob triples to an OpenAI ``LogProbs`` object.

    Each engine logprob item is a ``(logprob, token_id, token_text)`` triple.
    ``token_text`` is a detokenized *display* string that loses fragmentary
    byte-level tokens (a lone byte of a 4-byte char decodes to U+FFFD).  The
    legacy completions surface has no per-token ``bytes`` field, so when
    ``tokenizer`` is provided we render fragments losslessly as latin-1
    (one char per raw byte), keeping the string channel reversible.
    """
    ret_logprobs = LogProbs()

    def append_token_logprobs(token_logprobs):
        for logprob, token_id, token_text in token_logprobs:
            token_text = _lossless_token_text(tokenizer, token_id, token_text)
            ret_logprobs.tokens.append(token_text)
            ret_logprobs.token_logprobs.append(logprob)

            # Not supported yet
            ret_logprobs.text_offset.append(-1)

    def append_top_logprobs(top_logprobs):
        for tokens in top_logprobs:
            if tokens is not None:
                ret_logprobs.top_logprobs.append(
                    {
                        _lossless_token_text(tokenizer, token_id, token_text): logprob
                        for logprob, token_id, token_text in tokens
                    }
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


def _lossless_token_text(tokenizer, token_id, token_text):
    """Return a lossless display string for one engine logprob triple.

    Fragmentary byte-level tokens decode to U+FFFD in the display string.  When
    we can recover the true raw bytes from the token id (byte-level BPE), render
    them as latin-1 so every byte round-trips; otherwise keep the display text.
    """
    if token_text is not None and "\ufffd" not in token_text:
        return token_text
    if tokenizer is None or token_id is None:
        return token_text if token_text is not None else ""
    raw = token_id_to_bytes(tokenizer, token_id)
    if raw is None:
        return token_text if token_text is not None else ""
    try:
        return bytes(raw).decode("latin-1")
    except Exception:
        return token_text if token_text is not None else ""


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
