"""Read request fields from HTTP header values.

Two independent mechanisms:

- ``apply_header_overrides`` writes header values straight onto declared fields.
  It also steers placement and scheduling, so it stays behind
  ``SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES`` and, being applied last, wins over
  anything below.
- ``resolve_rid_from_headers`` reads the standard ``x-request-id`` header. It is
  honored on every request and only contributes an identity, which the caller
  resolves against the body rid.

Both allow upstream callers to leave the body opaque (no parse/merge/re-serialize).
"""

from typing import List, Optional, Union

from fastapi import HTTPException
from starlette.datastructures import Headers

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


def resolve_rid_from_headers(headers: Headers) -> Optional[Union[List[str], str]]:
    """Resolve the rids carried by the x-request-id header.

    The header is list-valued: values may be comma separated on one line or split
    across repeated lines, which RFC 9110 5.3 makes equivalent. Every line is read
    and blank entries are dropped, so the two spellings are interchangeable.

    A request may name itself once, once per batch item, or once per sample; the
    counts a batch accepts are enforced during normalization. The readable way to
    name every sample of a batch is one line per batch item, holding that item's
    n ids in order:

        x-request-id: req-a-0, req-a-1, req-a-2     # first input, n = 3
        x-request-id: req-b-0, req-b-1, req-b-2     # second input

    which is the same request as a single line reading "req-a-0, ..., req-b-2".

    A lone value is returned as a string; several are returned as a list.
    """
    rids = []
    for name, value in headers.items():
        if name.lower() != "x-request-id":
            continue
        rids.extend(entry.strip() for entry in value.split(",") if entry.strip())

    if not rids:
        return None

    return rids[0] if len(rids) == 1 else rids
