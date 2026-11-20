"""Size-only placeholders for request content in error and watchdog logs."""

from typing import Any

from sglang.srt.runtime_context import get_context, get_observability


def log_request_content_on_error() -> bool:
    # A process with no published config redacts.
    if not get_context().is_config_namespace_published("observability"):
        return False
    return get_observability().log_request_content_on_error


def _redacted(value: Any) -> str:
    try:
        return f"<redacted len={len(value)}>"
    except TypeError:
        return "<redacted>"


def redact_content(value: Any) -> str:
    if value is None or log_request_content_on_error():
        return repr(value)
    return _redacted(value)


def redact_text(value: Any) -> str:
    if value is None or log_request_content_on_error():
        return str(value)
    return _redacted(value)


def redact_exception(e: BaseException) -> str:
    # Compiler and parser errors quote the input they rejected.
    if log_request_content_on_error():
        return repr(e)
    return f"{type(e).__name__}({_redacted(str(e))})"
