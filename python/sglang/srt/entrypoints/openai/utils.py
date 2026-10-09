import logging
from typing import Any, Literal

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
_BYTE_DECODER: dict[str, int] = {}

# Tokenizer-level cache: once verified, we know whether *all* tokens from a
# given tokenizer can safely use the byte decoder. Avoids per-token checks.
_BYTE_LEVEL_TOKENIZERS: set = set()


def _build_byte_decoder() -> dict[str, int]:
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


def _is_byte_level_tokenizer(tokenizer) -> bool:
    """Heuristically determine whether *tokenizer* uses GPT-2 byte-level BPE.

    Only GPT-2 family (GPT2Tokenizer, Llama, Qwen, etc.) store vocabulary
    tokens as the ``bytes_to_unicode`` printable-char mapping. SentencePiece
    tokenizers (Mistral, Gemma, T5) store raw Unicode pieces, so applying the
    byte decoder to them corrupts multi-byte characters.

    We probe by checking ``is_byte_level`` (HuggingFace fast tokenizers) or
    by verifying that a known multi-byte character (é, U+00E9) round-trips:
    GPT-2 encodes it as a single byte-level piece ``chr(233)`` whose byte
    decoder output [233] does NOT form valid UTF-8, while SentencePiece stores
    the full character ``é`` whose UTF-8 is [195, 169].
    """
    tid = id(tokenizer)
    if tid in _BYTE_LEVEL_TOKENIZERS:
        return True

    # Fast path: HuggingFace fast tokenizers expose is_byte_level.
    is_bl = getattr(tokenizer, "is_byte_level", None)
    if isinstance(is_bl, bool):
        if is_bl:
            _BYTE_LEVEL_TOKENIZERS.add(tid)
        return is_bl

    # Slow path: probe with a known é token.
    global _BYTE_DECODER
    if not _BYTE_DECODER:
        _BYTE_DECODER = _build_byte_decoder()
    try:
        vocab_size = len(tokenizer.get_vocab())
        # Sample a few tokens to check if all chars are byte-decodable.
        sample_ids = [0, 1, 2, 3, vocab_size // 2, vocab_size - 2]
        for sid in sample_ids:
            if sid < 0 or sid >= vocab_size:
                continue
            piece = tokenizer.convert_ids_to_tokens(sid)
            if piece is None or not piece:
                continue
            # If any char in the piece is NOT in the byte decoder table,
            # this tokenizer does NOT use byte-level encoding.
            if any(ch not in _BYTE_DECODER for ch in piece):
                return False
        # All sampled tokens are byte-decodable → likely byte-level BPE.
        _BYTE_LEVEL_TOKENIZERS.add(tid)
        return True
    except Exception:
        return False


def token_id_to_bytes(tokenizer, token_id) -> list[int] | None:
    """Raw bytes for a byte-level-BPE token id.

    Returns the token's original bytes via the GPT-2 byte decoder, or None when
    the token is not byte-level representable (e.g. special ids / non byte BPE
    tokenizers like SentencePiece), so callers can fall back to the detokenized
    display string.
    """
    if not _is_byte_level_tokenizer(tokenizer):
        return None
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
    we can recover the true raw bytes from the token id (byte-level BPE), we
    validate that the recovered bytes do NOT form valid UTF-8 (a real fragment
    never does), then render them as latin-1 so every byte round-trips.

    Three safeguards address the reviewer's concerns:
    1. Non-byte-level tokenizers (SentencePiece/Mistral) are detected and
       skipped, so multi-byte characters like é are NOT corrupted to [233].
    2. Legitimate U+FFFD text (e.g. GPT-2 token 4210 = bytes [239,191,189])
       round-trips as valid UTF-8, so we keep the original display text.
    3. The latin-1 representation is only applied to genuine fragments (bytes
       that fail UTF-8 decode), avoiding key collisions in top_logprobs.
    """
    if token_text is not None and "\ufffd" not in token_text:
        return token_text
    if tokenizer is None or token_id is None:
        return token_text if token_text is not None else ""
    raw = token_id_to_bytes(tokenizer, token_id)
    if raw is None:
        return token_text if token_text is not None else ""
    # Only treat as a fragment if the recovered bytes do NOT form valid UTF-8.
    # A complete token whose display text happens to contain U+FFFD (e.g. token
    # 4210 = bytes [239,191,189] = valid UTF-8 for U+FFFD) must be left alone.
    try:
        bytes(raw).decode("utf-8")
        # Valid UTF-8 → this is NOT a fragment; keep the original display text.
        return token_text if token_text is not None else ""
    except UnicodeDecodeError:
        pass
    # Genuine fragment: render as latin-1 (one char per byte, lossless).
    try:
        return bytes(raw).decode("latin-1")
    except Exception:
        return token_text if token_text is not None else ""


def process_hidden_states_from_ret(
    ret_item: dict[str, Any],
    request: ChatCompletionRequest | CompletionRequest,
) -> list | None:
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
    hidden_states: list | None,
    return_hidden_states: bool | Literal["last"],
) -> list | None:
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
    ret_item: dict[str, Any],
    request: ChatCompletionRequest | CompletionRequest,
) -> str | None:
    """Process routed experts from a ret item in non-streaming response."""
    if not getattr(request, "return_routed_experts", False):
        return None
    return ret_item["meta_info"].get("routed_experts", None)


def cached_tokens_details_from_dict(
    details: dict[str, Any],
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
    ret_item: dict[str, Any],
    request: ChatCompletionRequest | CompletionRequest,
) -> CachedTokensDetails | None:
    """Process cached tokens details from a ret item in non-streaming response."""
    if not request.return_cached_tokens_details:
        return None

    details = ret_item["meta_info"].get("cached_tokens_details", None)
    if details is None:
        return None

    return cached_tokens_details_from_dict(details)


def spec_tokens_details_from_meta_info(
    meta_info: dict[str, Any],
) -> SpecTokensDetails | None:
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
    ret_item: dict[str, Any],
    request: ChatCompletionRequest | CompletionRequest,
) -> SpecTokensDetails | None:
    """Process speculative decoding details from a response item."""
    if not getattr(request, "return_spec_tokens_details", False):
        return None
    return spec_tokens_details_from_meta_info(ret_item["meta_info"])


def convert_embeds_to_tensors(
    embeds: list[list[list[float]] | None] | list[list[float]] | None,
) -> list[list[torch.Tensor] | None] | None:
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
