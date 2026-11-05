"""Utilities for JSON serialization in HTTP responses."""

from typing import Any, Dict, Optional, Tuple

import orjson
from fastapi.responses import Response
from pydantic import BaseModel

# Keep response serialization behavior consistent across endpoints:
# - Support non-string dictionary keys used in some metadata payloads.
# - Support numpy scalars/arrays without pre-conversion.
ORJSON_RESPONSE_OPTIONS = orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY

# orjson.Fragment needs orjson >= 3.9, and python/pyproject.toml does not pin orjson.
CUMULATIVE_STREAM_ENCODER_AVAILABLE = hasattr(orjson, "Fragment")

# The lists a cumulative native /generate stream chunk repeats in full. Within
# one request the TokenizerManager only appends to them or leaves them as is.
_CUMULATIVE_CHUNK_KEYS = ("output_ids", "prompt_token_ids")
_CUMULATIVE_META_INFO_KEYS = (
    "input_token_logprobs",
    "output_token_logprobs",
    "input_top_logprobs",
    "output_top_logprobs",
    "input_token_ids_logprobs",
    "output_token_ids_logprobs",
    "input_top_logprobs_val_flat",
    "input_top_logprobs_idx_flat",
    "output_token_sampling_mask",
    "output_token_sampling_logprobs",
)


def dumps_json(content: Any) -> bytes:
    """Serialize content to JSON bytes using SGLang's ORJSON options."""
    return orjson.dumps(content, option=ORJSON_RESPONSE_OPTIONS)


class CumulativeStreamEncoder:
    """Encodes the chunks of one native ``/generate`` stream to exactly
    ``dumps_json(chunk)``.

    Without incremental streaming output, every chunk repeats each request's
    whole ``output_ids`` and logprob lists. The encoder keeps each list's JSON
    from the request's previous chunk and encodes only the entries appended
    since. A list that is not the same list object, or is shorter, or whose
    last previously encoded entry is a different object, is encoded in full.
    Entries rewritten in place are not detected, which is why only lists that
    are append-only within a request are cached.
    """

    __slots__ = ("_requests",)

    def __init__(self):
        # Keyed by the chunk's "index", which batch streams set per request.
        self._requests: Dict[Optional[int], Dict[str, tuple]] = {}

    def encode(self, chunk: Any) -> bytes:
        if type(chunk) is not dict:
            return dumps_json(chunk)
        index = chunk.get("index")
        encoded_lists = self._requests.get(index)
        meta_info = chunk.get("meta_info")
        if (
            encoded_lists is None
            and type(meta_info) is dict
            and meta_info.get("finish_reason") is not None
        ):
            return dumps_json(chunk)
        if encoded_lists is None:
            encoded_lists = self._requests[index] = {}
        chunk = _with_encoded_lists(chunk, _CUMULATIVE_CHUNK_KEYS, encoded_lists)
        meta_info = chunk.get("meta_info")
        if type(meta_info) is dict:
            chunk["meta_info"] = _with_encoded_lists(
                meta_info, _CUMULATIVE_META_INFO_KEYS, encoded_lists
            )
            if meta_info.get("finish_reason") is not None:
                # The request's last chunk; release its encoded lists.
                del self._requests[index]
        return dumps_json(chunk)


def _with_encoded_lists(
    source: Dict[Any, Any], keys: Tuple[str, ...], encoded_lists: Dict[str, tuple]
) -> Dict[Any, Any]:
    # A shallow copy: the caller's dicts and lists stay untouched.
    target = dict(source)
    for key in keys:
        value = source.get(key)
        if type(value) is not list:
            continue
        if value:
            target[key] = orjson.Fragment(
                _encode_cumulative_list(
                    encoded_lists=encoded_lists, key=key, value=value
                )
            )
        else:
            # Cheaper to encode than to cache; it may also have shrunk to empty.
            encoded_lists.pop(key, None)
    return target


def _encode_cumulative_list(
    *, encoded_lists: Dict[str, tuple], key: str, value: list
) -> bytes:
    # value is not empty. Entry: (list, entries encoded, last of them, their JSON).
    entry = encoded_lists.get(key)
    if entry is not None:
        cached_list, num_encoded, last_encoded, encoded = entry
        num_entries = len(value)
        if (
            cached_list is value
            and num_entries >= num_encoded
            and value[num_encoded - 1] is last_encoded
        ):
            if num_entries > num_encoded:
                tail = dumps_json(value[num_encoded:])
                # orjson output is compact: "[a,b]" + "[c]" -> "[a,b,c]".
                encoded = b"".join(
                    (memoryview(encoded)[:-1], b",", memoryview(tail)[1:])
                )
                encoded_lists[key] = (value, num_entries, value[-1], encoded)
            return encoded
    encoded = dumps_json(value)
    encoded_lists[key] = (value, len(value), value[-1], encoded)
    return encoded


class SGLangORJSONResponse(Response):
    """ORJSON response with SGLang-specific serialization options."""

    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return dumps_json(content)


def orjson_response(content: Any, status_code: int = 200) -> Response:
    """Create a JSON response with stable ORJSON serialization options."""
    return SGLangORJSONResponse(content=content, status_code=status_code)


def model_json_response(content: Any) -> Any:
    """Serialize a pydantic result in one pass, skipping FastAPI's jsonable_encoder walk."""
    if not isinstance(content, BaseModel):
        return content
    return Response(
        content=content.model_dump_json(by_alias=True),
        media_type="application/json",
    )
