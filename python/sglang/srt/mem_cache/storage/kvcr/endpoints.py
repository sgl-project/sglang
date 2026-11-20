# SPDX-License-Identifier: Apache-2.0
"""Rank-local control endpoints for KVCR storage.

Only callers with an explicitly homogeneous source topology may apply a rank
offset. A router hint is advisory; endpoint validation does not establish a
source lease or prove that a transfer completed.
"""

import ipaddress
import socket
from urllib.parse import urlsplit


def control_address(config, extra_config) -> tuple[str, int]:
    """Resolve the same rank-local address for discovery and channel startup."""
    host = extra_config.get("control_advertise_host", "127.0.0.1")
    if not isinstance(host, str) or not host:
        raise ValueError("KVCR requires an advertisable control host")
    if host == "*":
        raise ValueError("KVCR cannot advertise an unspecified control host")
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
    first_port = base_port + config.dp_rank * stride
    last_port = first_port + stride - 1
    port = first_port + rank_offset(config)
    if not 0 < first_port <= port <= last_port <= 65535:
        raise ValueError("KVCR control-port range is invalid")
    return host, port


def require_routable_host(host: str) -> None:
    """Reject a resolved control host other workers cannot connect to.

    Takes a host as control_address() returns it. Mirrors the advertise-host
    check of Dynamo's KV state agent, which subscribes from another host:
    loopback, unspecified, wildcard and localhost names fail, including
    abbreviated IPv4 forms such as 127.1 and IPv4-mapped IPv6. DNS names are
    not resolved.
    """
    literal = host[1:-1] if host.startswith("[") else host
    if literal == "*" or literal.rstrip(".").lower() == "localhost":
        raise ValueError("KVCR control_advertise_host must not be loopback")
    try:
        address = ipaddress.ip_address(literal)
    except ValueError:
        try:
            address = ipaddress.IPv4Address(socket.inet_aton(literal))
        except OSError:
            return
    if address.version == 6 and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    if address.is_loopback or address.is_unspecified:
        raise ValueError("KVCR control_advertise_host must not be loopback")


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
