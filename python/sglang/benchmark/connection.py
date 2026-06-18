"""Resolve the base URL an HTTP benchmark client sends to.

Shared by every bench client so they apply the same precedence (an explicit
base URL wins over host/port) and the same IPv6-correct URL formatting. Kept
separate from ``endpoint.py`` so lightweight clients do not pull in the server
launch stack just to build a URL.
"""

from sglang.srt.utils.network import NetworkAddress


def resolve_base_url(base_url: str, host: str, port: int) -> str:
    """Return the base URL to send requests to: ``base_url`` if set, otherwise
    ``http://host:port`` (IPv6 brackets handled via NetworkAddress)."""
    if base_url:
        return base_url
    return NetworkAddress(host, port).to_url()
