from __future__ import annotations

import ipaddress
import logging
import os
import socket
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import psutil
import zmq

logger = logging.getLogger(__name__)


def get_open_port() -> int:
    port = os.getenv("SGLANG_PORT")
    if port is not None:
        port = int(port)
        while True:
            if is_port_available(port):
                return port
            logger.info("Port %d is already in use, trying port %d", port, port + 1)
            port += 1
    sock = try_bind_socket()
    port = sock.getsockname()[1]
    sock.close()
    return port


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def find_process_using_port(port: int) -> Optional[psutil.Process]:
    for conn in psutil.net_connections(kind="inet"):
        if conn.laddr.port == port:
            try:
                return psutil.Process(conn.pid)
            except psutil.NoSuchProcess:
                # It could happen by race condition (the proc dies when psutil.Process is called).
                pass

    return None


def wait_port_available(
    port: int, port_name: str, timeout_s: int = 30, raise_exception: bool = True
) -> bool:
    for i in range(timeout_s):
        if is_port_available(port):
            return True

        if i > 10 and i % 5 == 0:
            process = find_process_using_port(port)
            if process is None:
                logger.warning(
                    f"The port {port} is in use, but we could not find the process that uses it."
                )

            pid = process.pid
            error_message = f"{port_name} is used by a process already. {process.name()=}' {process.cmdline()=} {process.status()=} {pid=}"
            logger.info(
                f"port {port} is in use. Waiting for {i} seconds for {port_name} to be available. {error_message}"
            )
        time.sleep(0.1)

    if raise_exception:
        raise ValueError(
            f"{port_name} at {port} is not available in {timeout_s} seconds. {error_message}"
        )
    return False


def _get_addrinfos_for_bind(host=None, port=0):
    """Return deduplicated addrinfo tuples for binding (one per address family).

    Args:
        host: Bind address. None (with AI_PASSIVE) resolves to wildcard
              addresses (0.0.0.0 / ::) suitable for accepting on all interfaces.
        port: Port number. 0 lets the OS assign an available ephemeral port.

    Flags:
        AI_ADDRCONFIG — only return families actually configured on this host.
        AI_PASSIVE    — return wildcard addresses suitable for bind().

    Falls back to AF_INET if getaddrinfo fails (e.g. DNS misconfiguration).
    """
    try:
        infos = socket.getaddrinfo(
            host,
            port,
            socket.AF_UNSPEC,
            socket.SOCK_STREAM,
            0,
            socket.AI_ADDRCONFIG | socket.AI_PASSIVE,
        )
        deduped = []
        seen_families = set()
        for info in infos:
            if info[0] not in seen_families:
                seen_families.add(info[0])
                deduped.append(info)
        # Prefer IPv4 so that callers without an explicit host get consistent
        # behaviour across platforms (some OSes list IPv6 first).
        deduped.sort(key=lambda x: (x[0] != socket.AF_INET,))
        return deduped
    except socket.gaierror:
        fallback_host = "0.0.0.0" if host is None else host
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (fallback_host, port))]


def try_bind_socket(host=None, port=0, *, reuse_addr=True, listen=False):
    """Bind a TCP socket on the first available address family (IPv4/IPv6).

    Iterates over address families returned by _get_addrinfos_for_bind and
    returns the first socket that successfully binds.

    Args:
        host: Bind address. None binds to all interfaces (0.0.0.0 / ::).
        port: Port number. 0 lets the OS assign an available ephemeral port;
              use sock.getsockname()[1] to retrieve the assigned port.
        reuse_addr: Set SO_REUSEADDR to allow quick port reuse after close.
        listen: Call listen(1) after bind, making the socket ready to accept.

    Returns:
        The bound socket. Caller is responsible for closing it.

    Raises:
        OSError: If bind fails on all configured address families.
    """
    for family, socktype, proto, _, sockaddr in _get_addrinfos_for_bind(host, port):
        sock = socket.socket(family, socktype, proto)
        try:
            if family == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            if reuse_addr:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(sockaddr)
            if listen:
                sock.listen(1)
            return sock
        except (OSError, OverflowError):
            sock.close()
    raise OSError(f"Could not bind port {port} on any configured address family")


def is_port_available(port):
    """Return whether a port is available on all configured address families."""
    try:
        for family, socktype, proto, _, sockaddr in _get_addrinfos_for_bind(port=port):
            sock = socket.socket(family, socktype, proto)
            try:
                if family == socket.AF_INET6:
                    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(sockaddr)
            finally:
                sock.close()
        return True
    except (OSError, OverflowError):
        return False


def get_free_port():
    sock = try_bind_socket()
    port = sock.getsockname()[1]
    sock.close()
    return port


def bind_port(port):
    """Bind to a specific port, assuming it's available."""
    return try_bind_socket(port=port, listen=True)


def get_zmq_socket_on_host(
    context: zmq.Context,
    socket_type: zmq.SocketType,
    host: Optional[str] = None,
) -> Tuple[int, zmq.Socket]:
    """Create and configure a ZeroMQ socket.

    Args:
        context: ZeroMQ context to create the socket from.
        socket_type: Type of ZeroMQ socket to create.
        host: Optional host to bind/connect to, without "tcp://" prefix. If None, binds to "tcp://*".

    Returns:
        Tuple of (port, socket) where port is the randomly assigned TCP port.
    """
    socket = context.socket(socket_type)
    # Bind to random TCP port, auto-wrapping IPv6 and setting zmq.IPV6 flag
    config_socket(socket, socket_type)
    if host:
        if is_valid_ipv6_address(host):
            socket.setsockopt(zmq.IPV6, 1)
            bind_host = f"tcp://[{host}]"
        else:
            bind_host = f"tcp://{host}"
    else:
        bind_host = "tcp://*"
    port = socket.bind_to_random_port(bind_host)
    return port, socket


def config_socket(socket, socket_type: zmq.SocketType):
    mem = psutil.virtual_memory()
    total_mem = mem.total / 1024**3
    available_mem = mem.available / 1024**3
    if total_mem > 32 and available_mem > 16:
        buf_size = int(0.5 * 1024**3)
    else:
        buf_size = -1

    def set_send_opt():
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, buf_size)

    def set_recv_opt():
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)

    if socket_type == zmq.PUSH:
        set_send_opt()
    elif socket_type == zmq.PULL:
        set_recv_opt()
    elif socket_type in [zmq.DEALER, zmq.REQ, zmq.REP]:
        set_send_opt()
        set_recv_opt()
    else:
        raise ValueError(f"Unsupported socket type: {socket_type}")


def get_local_ip_by_nic(interface: str = None) -> Optional[str]:
    if not (interface := interface or os.environ.get("SGLANG_LOCAL_IP_NIC", None)):
        return None
    try:
        import netifaces
    except ImportError as e:
        raise ImportError(
            "Environment variable SGLANG_LOCAL_IP_NIC requires package netifaces, please install it through 'pip install netifaces'"
        ) from e

    try:
        addresses = netifaces.ifaddresses(interface)
        if netifaces.AF_INET in addresses:
            for addr_info in addresses[netifaces.AF_INET]:
                ip = addr_info.get("addr")
                if ip and ip != "127.0.0.1" and ip != "0.0.0.0":
                    return ip
        if netifaces.AF_INET6 in addresses:
            for addr_info in addresses[netifaces.AF_INET6]:
                ip = addr_info.get("addr")
                if ip and not ip.startswith("fe80::") and ip != "::1":
                    return ip.split("%")[0]
    except (ValueError, OSError) as e:
        logger.warning(
            f"{e} Can not get local ip from NIC. Please verify whether SGLANG_LOCAL_IP_NIC is set correctly."
        )
    return None


def get_local_ip_by_remote() -> Optional[str]:
    # Google's public DNS servers, used to discover the local IP.
    # UDP connect doesn't send packets; it just selects the right source address.
    # https://developers.google.com/speed/public-dns/docs/using#addresses
    # Try IPv4 first, then IPv6. getaddrinfo on a literal IP returns exactly
    # one result, so we unpack directly instead of looping.
    for dns_host, dns_port in [("8.8.8.8", 80), ("2001:4860:4860::8888", 80)]:
        try:
            family, socktype, proto, _, sockaddr = socket.getaddrinfo(
                dns_host,
                dns_port,
                socket.AF_UNSPEC,
                socket.SOCK_DGRAM,
                0,
                socket.AI_ADDRCONFIG,
            )[0]
            with socket.socket(family, socktype, proto) as s:
                s.connect(sockaddr)
                return s.getsockname()[0]
        except (socket.gaierror, OSError):
            continue

    # Fallback: resolve the local hostname to an IP address via /etc/hosts or DNS.
    # Unreliable — many machines resolve hostname to 127.0.0.1, so we skip loopback.
    try:
        hostname = socket.gethostname()
        ip = socket.getaddrinfo(
            hostname, None, socket.AF_UNSPEC, 0, 0, socket.AI_ADDRCONFIG
        )[0][4][0]
        if ip and ip not in ("127.0.0.1", "0.0.0.0", "::1"):
            return ip
    except Exception:
        pass

    logger.warning("Can not get local ip by remote")
    return None


def get_local_ip_auto(fallback: str = None) -> str:
    """
    Automatically detect the local IP address using multiple fallback strategies.

    This function attempts to obtain the local IP address through several methods.
    If all methods fail, it returns the specified fallback value or raises an exception.

    Args:
        fallback (str, optional): Fallback IP address to return if all detection
            methods fail. For server applications, explicitly set this to
            "0.0.0.0" (IPv4) or "::" (IPv6) to bind to all available interfaces.
            Defaults to None.

    Returns:
        str: The detected local IP address, or the fallback value if detection fails.

    Raises:
        ValueError: If IP detection fails and no fallback value is provided.

    Note:
        The function tries detection methods in the following order:
        1. Direct IP detection via get_ip()
        2. Network interface enumeration via get_local_ip_by_nic()
        3. Remote connection method via get_local_ip_by_remote()
    """
    # Try environment variable
    host_ip = os.getenv("SGLANG_HOST_IP", "") or os.getenv("HOST_IP", "")
    if host_ip:
        return host_ip
    logger.debug("get_ip failed")
    # Fallback
    if ip := get_local_ip_by_nic():
        return ip
    logger.debug("get_local_ip_by_nic failed")
    # Fallback
    if ip := get_local_ip_by_remote():
        return ip
    logger.debug("get_local_ip_by_remote failed")
    if fallback:
        return fallback
    raise ValueError("Can not get local ip")


def get_zmq_socket(
    context: zmq.Context,
    socket_type: zmq.SocketType,
    endpoint: Optional[str] = None,
    bind: bool = True,
) -> Union[zmq.Socket, Tuple[int, zmq.Socket]]:
    """Create and configure a ZeroMQ socket.

    Args:
        context: ZeroMQ context to create the socket from.
        socket_type: Type of ZeroMQ socket to create.
        endpoint: Optional endpoint to bind/connect to. If None, binds to a random TCP port.
        bind: Whether to bind (True) or connect (False) to the endpoint. Ignored if endpoint is None.

    Returns:
        If endpoint is None: Tuple of (port, socket) where port is the randomly assigned TCP port.
        If endpoint is provided: The configured ZeroMQ socket.
    """
    socket = context.socket(socket_type)

    if endpoint is None:
        # Bind to random TCP port
        config_socket(socket, socket_type)
        port = socket.bind_to_random_port("tcp://*")
        return port, socket
    else:
        # Handle IPv6 if endpoint contains brackets
        if endpoint.find("[") != -1:
            socket.setsockopt(zmq.IPV6, 1)

        config_socket(socket, socket_type)

        if bind:
            socket.bind(endpoint)
        else:
            socket.connect(endpoint)

        return socket


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

    def __post_init__(self):
        # Auto-strip IPv6 brackets so callers can pass "[::1]" or "::1"
        if self.host.startswith("[") and self.host.endswith("]"):
            object.__setattr__(self, "host", self.host[1:-1])

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

    @staticmethod
    def resolve_host(host: str) -> str:
        """Return *host* as-is if it's an IP, otherwise DNS-resolve to one."""
        try:
            ipaddress.ip_address(host)
            return host
        except ValueError:
            pass
        try:
            return socket.getaddrinfo(
                host, None, socket.AF_UNSPEC, 0, 0, socket.AI_ADDRCONFIG
            )[0][4][0]
        except socket.gaierror as e:
            raise ValueError(f"Cannot resolve host {host!r}: {e}") from e

    def resolved(self) -> NetworkAddress:
        """DNS-resolve hostname to IP; return self if already an IP."""
        ip = self.resolve_host(self.host)
        return self if ip == self.host else NetworkAddress(ip, self.port)

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

    def __str__(self) -> str:
        return self.to_host_port_str()

    def __repr__(self) -> str:
        return f"NetworkAddress({self.host!r}, {self.port})"
