from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from typing import Tuple


def _is_ipv6(host: str) -> bool:
    """Check whether *host* is a valid IPv6 address (without brackets)."""
    try:
        ipaddress.IPv6Address(host)
        return True
    except ValueError:
        return False


def _wrap(host: str) -> str:
    """Wrap an IPv6 address in brackets; pass IPv4/hostname through."""
    return f"[{host}]" if _is_ipv6(host) else host


def _parse_port(s: str) -> int:
    try:
        port = int(s)
    except ValueError:
        raise ValueError(f"Invalid port number: {s!r}")
    if not (0 <= port <= 65535):
        raise ValueError(f"Port out of range (0-65535): {port}")
    return port


@dataclass(frozen=True)
class NetworkAddress:
    host: str
    port: int

    @property
    def is_ipv6(self) -> bool:
        return _is_ipv6(self.host)

    @property
    def family(self) -> socket.AddressFamily:
        return socket.AF_INET6 if self.is_ipv6 else socket.AF_INET

    def to_url(self, scheme: str = "http") -> str:
        """``http://127.0.0.1:30000`` or ``http://[::1]:30000``."""
        return f"{scheme}://{_wrap(self.host)}:{self.port}"

    def to_tcp(self) -> str:
        """``tcp://`` endpoint for ZMQ / torch distributed."""
        return self.to_url("tcp")

    def to_host_port_str(self) -> str:
        """``host:port`` string for gRPC listen address, session IDs, logs."""
        return f"{_wrap(self.host)}:{self.port}"

    def to_bind_tuple(self) -> Tuple[str, int]:
        """Raw ``(host, port)`` tuple for ``socket.bind()`` / ``socket.connect()``.

        Returns the *unwrapped* host — sockets need the raw address, not
        the bracketed form.
        """
        return (self.host, self.port)

    @staticmethod
    def parse(addr: str) -> NetworkAddress:
        """Parse a ``host:port`` string into a ``NetworkAddress``.

        Accepted formats::

            [::1]:8000          → NetworkAddress("::1", 8000)
            127.0.0.1:8000      → NetworkAddress("127.0.0.1", 8000)
            my-hostname:8000    → NetworkAddress("my-hostname", 8000)

        IPv6 addresses **must** be bracketed.  Bare ``::1:8000`` is
        ambiguous and will raise ``ValueError``.

        Raises:
            ValueError: If the string cannot be unambiguously parsed.
        """
        if not addr:
            raise ValueError("Empty address string")

        # --- Bracketed IPv6: [addr]:port ---
        if addr.startswith("["):
            close = addr.find("]")
            if close == -1:
                raise ValueError(f"Missing closing bracket in IPv6 address: {addr!r}")
            host = addr[1:close]
            if not _is_ipv6(host):
                raise ValueError(f"Invalid IPv6 address inside brackets: {host!r}")
            rest = addr[close + 1 :]
            if not rest.startswith(":") or len(rest) < 2:
                raise ValueError(
                    f"Expected ':port' after closing bracket, got: {rest!r}"
                )
            return NetworkAddress(host, _parse_port(rest[1:]))

        # --- Plain host:port (IPv4 / hostname) ---
        if ":" not in addr:
            raise ValueError(f"Missing port in address (expected host:port): {addr!r}")
        host, port_str = addr.rsplit(":", 1)
        if not host:
            raise ValueError(f"Empty host in address: {addr!r}")
        # Guard against bare IPv6 slipping through
        if ":" in host and _is_ipv6(host):
            raise ValueError(
                f"Bare IPv6 address without brackets is ambiguous: {addr!r}. "
                f"Use [{host}]:{port_str} instead."
            )
        return NetworkAddress(host, _parse_port(port_str))

    @staticmethod
    def from_parts(host: str, port: int) -> NetworkAddress:
        """Create from separate host and port, stripping brackets if present.

        Useful when the host may come from user input that already has
        brackets (e.g. ``[::1]``).
        """
        if host.startswith("[") and host.endswith("]"):
            host = host[1:-1]
        return NetworkAddress(host, port)

    def __str__(self) -> str:
        return self.to_host_port_str()

    def __repr__(self) -> str:
        return f"NetworkAddress({self.host!r}, {self.port})"
