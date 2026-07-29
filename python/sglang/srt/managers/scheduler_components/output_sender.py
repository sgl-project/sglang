from typing import Optional, Union

import zmq

from sglang.srt.managers.io_struct import BaseBatchReq, BaseReq


class SenderWrapper:
    def __init__(self, socket: zmq.Socket):
        self.socket = socket

    def send_output(
        self,
        output: Union[BaseReq, BaseBatchReq],
        recv_obj: Optional[Union[BaseReq, BaseBatchReq]] = None,
    ):
        if self.socket is None:
            return

        if (
            isinstance(recv_obj, BaseReq)
            and recv_obj.http_worker_ipc is not None
            and output.http_worker_ipc is None
        ):
            # handle communicator reqs for multi-http worker case
            output.http_worker_ipc = recv_obj.http_worker_ipc

        self.socket.send_pyobj(output)
