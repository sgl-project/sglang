# SPDX-License-Identifier: Apache-2.0
"""Rank-local control endpoints shared by KVCR storage and direct delivery.

Only callers with an explicitly homogeneous source topology may apply a rank
offset. A router hint is advisory; endpoint validation does not establish a
source lease or prove that a transfer completed.
"""

from urllib.parse import urlsplit


def rank_offset(config) -> int:
    """TP is the innermost rank within one advertised DP replica."""
    return config.tp_rank + config.tp_size * (
        config.attn_cp_rank + config.attn_cp_size * config.pp_rank
    )


def rank_endpoint(endpoint: str, offset: int) -> str | None:
    """Offset an advertised TCP endpoint, or reject an ambiguous address."""
    if not isinstance(endpoint, str) or type(offset) is not int:
        return None
    try:
        source = urlsplit(endpoint)
        port = source.port
        if (
            source.scheme != "tcp"
            or not source.hostname
            or port is None
            or not 1 <= port <= 65535
            or not 1 <= port + offset <= 65535
            or source.path
            or source.query
            or source.fragment
            or source.username is not None
        ):
            return None
    except ValueError:
        return None
    host = source.hostname
    if ":" in host:
        host = f"[{host}]"
    return f"tcp://{host}:{port + offset}"
