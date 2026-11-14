"""Override object fields based on _HEADER_OVERRIDES from header values.

This mechanism allows upstream callers to leave the body opaque
(no parse/merge/re-serialize).
"""

import re

import http_sf
from fastapi import HTTPException, Request

from sglang.srt.runtime_context import get_schedule, get_serving

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


def resolve_request_priority(
    priority: int | None,
    raw_request: Request | None,
    *,
    body_priority_is_set: bool,
) -> int | None:
    """Resolve native priority headers without changing body or legacy overrides.

    The converted priority already includes any opt-in vendor override. RFC 9218
    requires ignoring malformed dictionaries and invalid urgency values.
    """
    if (
        body_priority_is_set
        or priority is not None
        or raw_request is None
        or not get_schedule().enable_priority_scheduling
        or get_serving().allow_header_priority is False
    ):
        return priority

    headers = raw_request.headers
    values = headers.getlist("x-priority")
    if values:
        if len(values) != 1 or not re.fullmatch(r"[+-]?[0-9]+", values[0].strip(" \t")):
            raise HTTPException(
                status_code=400, detail="X-Priority must be a single integer header"
            )
        try:
            return int(values[0])
        except ValueError as e:
            raise HTTPException(
                status_code=400, detail="X-Priority integer is too large"
            ) from e

    values = headers.getlist("priority")
    if values:
        try:
            parameters = http_sf.parse(
                ", ".join(values).encode("ascii"), tltype="dictionary"
            )
        except ValueError:
            return priority
        urgency, _ = parameters.get("u", (None, {}))
        if type(urgency) is int and 0 <= urgency <= 7:
            return urgency * 10
    return priority
