"""Override object fields based on _HEADER_OVERRIDES from header values.

This mechanism allows upstream callers to leave the body opaque
(no parse/merge/re-serialize).
"""

from fastapi import HTTPException

# request header -> (target attribute, value type)
_HEADER_OVERRIDES = {
    "x-override-rid": ("rid", str),
    "x-override-bootstrap-host": ("bootstrap_host", str),
    "x-override-bootstrap-port": ("bootstrap_port", int),
    "x-override-bootstrap-room": ("bootstrap_room", int),
    "x-override-conversation-id": ("conversation_id", str),
    "x-override-routed-dp-rank": ("routed_dp_rank", int),
    "x-override-disagg-prefill-dp-rank": ("disagg_prefill_dp_rank", int),
    "x-override-priority": ("priority", int),
}

# Same header the vLLM OpenAI server accepts, and the one sgl-router already
# uses for sticky session routing.
SESSION_ID_HEADER = "x-session-id"


def apply_session_id_header(obj, headers) -> None:
    """Fall back to ``X-Session-ID`` when the body carries no session identity.

    The body always wins: this only fills a missing ``session_id``, so a caller
    that sets the field explicitly is unaffected. ``session_params`` describes a
    scheduler-managed session and is mutually exclusive with ``session_id``, so
    the header is ignored when ``session_params`` is present.
    """
    if getattr(obj, "session_id", None) is not None:
        return
    if getattr(obj, "session_params", None) is not None:
        return
    value = headers.get(SESSION_ID_HEADER)
    if value:
        obj.session_id = value


def apply_header_overrides(obj, headers) -> None:
    """Override request based on header values. Fail the request when any override has issues."""
    for header, (attr, cast) in _HEADER_OVERRIDES.items():
        value = headers.get(header)
        if value is None:
            continue
        try:
            setattr(obj, attr, cast(value))
        except ValueError as e:
            raise HTTPException(
                status_code=400, detail=f"invalid {header} header {value!r}: {e}"
            ) from e
