"""Utilities for JSON serialization in HTTP responses."""

from typing import Any

import orjson
from fastapi.responses import Response
from pydantic import BaseModel

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


def model_json_response(content: Any) -> Any:
    """Render a pydantic endpoint result in one pass; pass anything else through.

    Returning the model itself makes FastAPI run ``jsonable_encoder``: a
    ``model_dump`` followed by a pure-Python walk over the dumped payload, which
    dominates large responses such as top-k logprobs. The JSON values match
    FastAPI's (aliases applied), except that non-finite floats become null, as
    in ``dumps_json``, where Starlette would raise.
    """
    if not isinstance(content, BaseModel):
        return content
    return Response(
        content=content.model_dump_json(by_alias=True),
        media_type="application/json",
    )
