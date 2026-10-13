"""Response declarations for the Python server's OpenAI completion routes."""

from typing import Any

from pydantic import BaseModel

from sglang.srt.entrypoints.openai.protocol import ErrorResponse


def _sse_event_schema(stream_model: type[BaseModel]) -> dict[str, Any]:
    """Describe one parsed SSE event referencing an independently registered model.

    This OpenAPI 3.1 convention describes a parsed event, not the complete HTTP
    body. Consumers must decode JSON data and validate contentSchema explicitly;
    ordinary JSON Schema validation treats contentSchema as an annotation.
    """
    return {
        "type": "object",
        "description": (
            "One parsed SSE data event, not the complete stream. OpenAPI 3.1 "
            "publishes this per-event convention under schema, not itemSchema. "
            "Decode JSON data and validate its contentSchema explicitly. "
            "The generator ends with data: [DONE]; a disconnected stream may "
            "end earlier. Comment-only frames are not data events."
        ),
        "required": ["data"],
        "properties": {
            "data": {
                "type": "string",
                "anyOf": [
                    {"const": "[DONE]"},
                    {
                        "contentMediaType": "application/json",
                        "contentSchema": {
                            "$ref": f"#/components/schemas/{stream_model.__name__}"
                        },
                    },
                    {
                        "contentMediaType": "application/json",
                        "contentSchema": {
                            "type": "object",
                            "required": ["error"],
                            "properties": {
                                "error": {"$ref": "#/components/schemas/ErrorResponse"}
                            },
                        },
                    },
                ],
            },
            "event": {"type": "string"},
            "id": {"type": "string"},
            "retry": {"type": "integer", "minimum": 0},
        },
    }


def completion_response_kwargs(
    response_model: type[BaseModel], stream_model: type[BaseModel]
) -> dict[str, Any]:
    """Build route kwargs registering native unary, chunk and error schemas.

    Supply the native models used by the completion serving layer. Callers must
    return already serialized Response objects (including StreamingResponse),
    as these routes do through model_json_response, to bypass response_model
    validation and preserve native serialization. SSE describes parsed events
    using the OpenAPI 3.1 convention documented by _sse_event_schema.
    Other errors remain unspecified because middleware can use different bodies.
    """
    return {
        "response_model": response_model,
        "responses": {
            200: {
                # FastAPI registers this model and its dependencies, but also
                # assigns it to JSON; openapi_extra restores the unary reference.
                "model": stream_model,
                "description": "A JSON completion, or SSE data events when stream=true.",
                "content": {
                    "text/event-stream": {"schema": _sse_event_schema(stream_model)}
                },
            },
            400: {
                "model": ErrorResponse,
                "description": "Invalid request, including body validation failures.",
            },
            500: {
                "model": ErrorResponse,
                "description": "Internal error handled by the OpenAI serving layer.",
            },
            # FastAPI omits its automatic 422 when a default response is declared.
            # These routes' RequestValidationError handler actually returns 400.
            "default": {
                "description": (
                    "Other responses, including configuration-dependent errors such "
                    "as authentication failures. Their bodies are not specified here."
                ),
            },
        },
        "openapi_extra": {
            "responses": {
                "200": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": f"#/components/schemas/{response_model.__name__}"
                            }
                        }
                    }
                }
            }
        },
    }
