"""Scheduler-control transport through the existing DP broadcast topology."""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import zmq

from sglang.srt.managers.io_struct import BaseReq, sock_send
from sglang.srt.utils.network import (
    NetworkAddress,
    get_local_ip_auto,
    get_zmq_socket,
    get_zmq_socket_on_host,
)

if TYPE_CHECKING:
    from sglang.srt.rust_extensions._server import Server


class RustControlTransport:
    """Only the scheduler thread touches these sockets; Rust owns pending RPCs."""

    def __init__(
        self,
        server: Server,
        controller_endpoint: str,
        dp_rank: int,
        *,
        cross_node: bool,
    ) -> None:
        if not controller_endpoint:
            raise ValueError("Rust DP controls require a controller endpoint")
        self.server = server
        self.dp_rank = dp_rank
        self.context = zmq.Context(1)
        self.controller = get_zmq_socket(
            self.context, zmq.PUSH, controller_endpoint, bind=False
        )
        self.controller.setsockopt(zmq.SNDTIMEO, 1000)
        host = get_local_ip_auto() if cross_node else "127.0.0.1"
        port, self.replies = get_zmq_socket_on_host(self.context, zmq.PULL, host)
        self.endpoint = NetworkAddress(host, port).to_tcp()
        self.reply_sockets: dict[str, zmq.Socket] = {}
        self.decoder = msgspec.msgpack.Decoder(tuple[str, int, bytes])

    def broadcast(self, request: BaseReq) -> None:
        request.http_worker_ipc = self.endpoint
        sock_send(self.controller, request)

    def send_result(self, request: BaseReq, payload: bytes) -> None:
        endpoint = request.http_worker_ipc
        if not endpoint:
            raise ValueError("Rust DP control is missing its reply endpoint")
        if endpoint == self.endpoint:
            self.server.push_control_result_part(request.rid, self.dp_rank, payload)
            return
        socket = self.reply_sockets.get(endpoint)
        if socket is None:
            socket = get_zmq_socket(self.context, zmq.PUSH, endpoint, bind=False)
            socket.setsockopt(zmq.SNDTIMEO, 1000)
            self.reply_sockets[endpoint] = socket
        socket.send(msgspec.msgpack.encode((request.rid, self.dp_rank, payload)))

    def drain_replies(self) -> None:
        while True:
            try:
                data = self.replies.recv(zmq.NOBLOCK)
            except zmq.Again:
                return
            rid, rank, payload = self.decoder.decode(data)
            self.server.push_control_result_part(rid, rank, payload)

    def close(self) -> None:
        self.controller.close(linger=0)
        self.replies.close(linger=0)
        for socket in self.reply_sockets.values():
            socket.close(linger=0)
        self.reply_sockets.clear()
        self.context.term()
