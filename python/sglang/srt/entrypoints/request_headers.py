"""Request-header handling shared by every entrypoint.

Two mechanisms live here:

* ``extract_routed_dp_rank``: the ``X-Data-Parallel-Rank`` header, read
  unconditionally on every generation route so a DP-aware gateway can pin a
  request to a data-parallel rank without touching the body.
* ``apply_header_overrides``: the ``x-override-*`` family, gated behind
  ``SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES``, which lets upstream callers leave
  the body opaque (no parse/merge/re-serialize).
"""

import logging
from typing import Optional

from fastapi import HTTPException

logger = logging.getLogger(__name__)

ROUTED_DP_RANK_HEADER = "x-data-parallel-rank"

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


def extract_routed_dp_rank(
    headers, body_routed_dp_rank: Optional[int] = None
) -> Optional[int]:
    """The DP rank a request is pinned to: the ``X-Data-Parallel-Rank`` header
    when present (case-insensitive), else ``body_routed_dp_rank``.

    The header wins over the body so a gateway that routes on rank can pin a
    request without parsing and re-serializing its payload. A header that is
    not an integer is a 400: silently ignoring it would let a mistyped pin
    degrade to the engine's own load balancing with no error anywhere.
    """
    if headers is None:
        return body_routed_dp_rank

    header_value = headers.get(ROUTED_DP_RANK_HEADER)
    if header_value is None:
        return body_routed_dp_rank

    try:
        header_dp_rank = int(header_value)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid X-Data-Parallel-Rank header: must be an integer, got '{header_value}'",
        )
    if body_routed_dp_rank is not None and header_dp_rank != body_routed_dp_rank:
        logger.debug(
            f"X-Data-Parallel-Rank header ({header_dp_rank}) overrides "
            f"body routed_dp_rank ({body_routed_dp_rank})"
        )
    return header_dp_rank


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
