"""Utilities for JSON serialization in HTTP responses."""

from typing import Any

import orjson
from fastapi.responses import Response

# Keep response serialization behavior consistent across endpoints:
# - Support non-string dictionary keys used in some metadata payloads.
# - Support numpy scalars/arrays without pre-conversion.
ORJSON_RESPONSE_OPTIONS = orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY


def dumps_json(content: Any) -> bytes:
    """Serialize content to JSON bytes using SGLang's ORJSON options."""
    return orjson.dumps(content, option=ORJSON_RESPONSE_OPTIONS)


class SGLangORJSONResponse(Response):
    """ORJSON response with SGLang-specific serialization options."""

    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return dumps_json(content)


def orjson_response(content: Any, status_code: int = 200) -> Response:
    """Create a JSON response with stable ORJSON serialization options."""
    return SGLangORJSONResponse(content=content, status_code=status_code)
