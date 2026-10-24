"""Shared wire envelope for OpenAI-compatible HTTP and streaming errors."""

from typing import Optional


def openai_error(
    message: str,
    status_code: int = 400,
    err_type: Optional[str] = None,
    param: Optional[str] = None,
    code: Optional[str] = None,
) -> dict:
    # HTTP status belongs on the response, not in the SDK's string error.code.
    # Legacy callers pass class names, status names or numeric strings as type.
    # Retain explicit protocol types; normalize those legacy values by status.
    error_type = {
        401: "authentication_error",
        403: "permission_error",
        429: "rate_limit_error",
    }.get(
        status_code, "server_error" if status_code >= 500 else "invalid_request_error"
    )
    if err_type and err_type.endswith("_error"):
        error_type = err_type
    if code is None:
        # An SSE error travels under HTTP 200. Retain distinctions needed by
        # protocol adapters using stable string codes, never numeric statuses.
        code = {
            404: "not_found",
            408: "request_timeout",
            413: "request_too_large",
            503: "service_unavailable",
        }.get(status_code)
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }
