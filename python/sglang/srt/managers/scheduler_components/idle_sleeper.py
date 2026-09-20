from __future__ import annotations

from typing import TYPE_CHECKING

import zmq

if TYPE_CHECKING:
    from sglang.srt.rust_server.server import RustServer


class IdleSleeper:
    """
    In setups which have long inactivity periods it is desirable to reduce
    system power consumption when sglang does nothing. This would lead not only
    to power savings, but also to more CPU thermal headroom when a request
    eventually comes. This is important in cases when multiple GPUs are connected
    as each GPU would otherwise pin one thread at 100% CPU usage.

    The simplest solution is to use zmq.Poller on all sockets that may receive
    data that needs handling immediately.

    Parking is all this class does, and only the rank owning those sockets can
    do it; idle memory reclamation belongs to the scheduler.
    """

    def __init__(self, sockets):
        self.poller = zmq.Poller()
        for s in sockets:
            self.poller.register(s, zmq.POLLIN)

    def maybe_sleep(self):
        self.poller.poll(1000)


class RustServerIdleSleeper:
    """Idle sleeper for the embedded Rust server.

    The Rust ingress is an in-process request ring, not a zmq socket.
    Instead park directly on the ring: ``wait_request`` blocks until
    a request is pushed — the request ring wakes the parked thread
    the instant a producer pushes, so there's no added latency for real
    requests — or the timeout elapses.
    """

    def __init__(self, rust_server: RustServer, timeout_ms: int = 1000):
        self.rust_server = rust_server
        self.timeout_ms = timeout_ms

    def maybe_sleep(self):
        self.rust_server.wait_request(self.timeout_ms)
