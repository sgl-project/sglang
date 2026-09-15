from typing import Callable, Optional, Union

import zmq

from sglang.srt.managers.io_struct import BaseBatchReq, BaseReq, sock_send


class SenderWrapper:
    def __init__(self, socket: zmq.Socket):
        self.socket = socket
        self.output_handler: Optional[
            Callable[[Union[BaseReq, BaseBatchReq]], bool]
        ] = None

    def send_output(
        self,
        output: Union[BaseReq, BaseBatchReq],
        recv_obj: Optional[object] = None,
    ):
        http_worker_ipc = getattr(recv_obj, "http_worker_ipc", None)
        if (
            isinstance(output, BaseReq)
            and http_worker_ipc is not None
            and output.http_worker_ipc is None
        ):
            # Scheduler Req is not a BaseReq but carries the same return route.
            output.http_worker_ipc = http_worker_ipc

        if self.output_handler is not None and self.output_handler(output):
            return
        if self.socket is not None:
            sock_send(self.socket, output)
