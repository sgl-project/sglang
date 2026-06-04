from typing import Optional, Union

import zmq

from sglang.srt.managers.io_struct import BaseBatchReqIpc, BaseReqIpc, sock_send


class SenderWrapper:
    def __init__(self, socket: zmq.Socket):
        self.socket = socket

    def send_output(
        self,
        output: Union[BaseReqIpc, BaseBatchReqIpc],
        recv_obj: Optional[Union[BaseReqIpc, BaseBatchReqIpc]] = None,
    ):
        if self.socket is None:
            return

        if (
            isinstance(recv_obj, BaseReqIpc)
            and recv_obj.http_worker_ipc is not None
            and output.http_worker_ipc is None
        ):
            # handle communicator reqs for multi-http worker case
            output.http_worker_ipc = recv_obj.http_worker_ipc

        sock_send(self.socket, output)
