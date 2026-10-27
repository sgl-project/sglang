# SPDX-License-Identifier: Apache-2.0
"""Rank-local control endpoints shared by KVCR storage and direct delivery.

Only callers with an explicitly homogeneous source topology may apply a rank
offset. A router hint is advisory; endpoint validation does not establish a
source lease or prove that a transfer completed.
"""

import ipaddress
from urllib.parse import urlsplit


def control_address(config, extra_config) -> tuple[str, int]:
    """Resolve the same rank-local address for discovery and channel startup."""
    host = extra_config.get("control_advertise_host", "127.0.0.1")
    if not isinstance(host, str) or not host:
        raise ValueError("KVCR requires an advertisable control host")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if any(char.isspace() or char in "/@?#:[]" for char in host):
            raise ValueError("KVCR requires a host, not a URL or malformed address")
    else:
        if address.is_unspecified:
            raise ValueError("KVCR cannot advertise an unspecified control host")
        host = f"[{address.compressed}]" if address.version == 6 else address.compressed
    base_port = extra_config.get("control_port", 32100)
    if type(base_port) is not int or not 0 < base_port <= 65535:
        raise ValueError("KVCR control_port must be a valid integer port")
    stride = config.tp_size * config.attn_cp_size * config.pp_size
    port = base_port + config.dp_rank * stride + rank_offset(config)
    if not 0 < port <= 65535:
        raise ValueError("KVCR control-port range is invalid")
    return host, port


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
