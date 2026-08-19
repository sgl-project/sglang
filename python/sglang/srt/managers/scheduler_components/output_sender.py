from typing import Optional, Union

import zmq

from sglang.srt.managers.io_struct import BaseBatchReq, BaseReq, sock_send


class SenderWrapper:
    def __init__(self, socket: zmq.Socket):
        self.socket = socket

    def send_output(
        self,
        output: Union[BaseReq, BaseBatchReq],
        recv_obj: Optional[object] = None,
    ):
        if self.socket is None:
            return

        http_worker_ipc = getattr(recv_obj, "http_worker_ipc", None)
        if (
            isinstance(output, BaseReq)
            and http_worker_ipc is not None
            and output.http_worker_ipc is None
        ):
            # Scheduler Req is not a BaseReq but carries the same return route.
            output.http_worker_ipc = http_worker_ipc

        sock_send(self.socket, output)

    def try_send_output(
        self,
        output: Union[BaseReq, BaseBatchReq],
        recv_obj: Optional[Union[BaseReq, BaseBatchReq]] = None,
    ) -> bool:
        """Best-effort send for liveness signals that must not block the scheduler."""
        if self.socket is None:
            return True

        if (
            isinstance(recv_obj, BaseReq)
            and recv_obj.http_worker_ipc is not None
            and output.http_worker_ipc is None
        ):
            output.http_worker_ipc = recv_obj.http_worker_ipc

        try:
            sock_send(self.socket, output, flags=zmq.NOBLOCK)
        except zmq.Again:
            return False
        return True
