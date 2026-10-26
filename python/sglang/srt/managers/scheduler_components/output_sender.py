from typing import Optional, Union

import zmq

from sglang.srt.managers.io_struct import (
    BaseBatchReq,
    BaseReq,
    sock_send,
    sock_send_with_timestamp,
)


class SenderWrapper:
    def __init__(self, socket: zmq.Socket, *, timestamped: bool = False):
        self.socket = socket
        self.timestamped = timestamped

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

        if self.timestamped:
            sock_send_with_timestamp(self.socket, output)
        else:
            sock_send(self.socket, output)
