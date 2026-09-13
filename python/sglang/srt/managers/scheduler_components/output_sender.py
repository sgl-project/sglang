import os
from socket import gethostname
from typing import Optional, Union

import zmq

from sglang.srt.managers.io_struct import (
    BaseBatchReq,
    BaseReq,
    TokenizerControlBackendResultReq,
    msgpack_encode,
    sock_send,
)
from sglang.srt.managers.tokenizer_control import CONTROL_RETURN_PREFIX


class SenderWrapper:
    def __init__(self, socket: zmq.Socket):
        self.socket = socket
        self.worker_id = f"{gethostname()}:{os.getpid()}"

    def send_output(
        self,
        output: Union[BaseReq, BaseBatchReq],
        recv_obj: Optional[object] = None,
    ):
        if self.socket is None:
            return

        http_worker_ipc = getattr(recv_obj, "http_worker_ipc", None)
        if http_worker_ipc and http_worker_ipc.startswith(CONTROL_RETURN_PREFIX):
            sock_send(
                self.socket,
                TokenizerControlBackendResultReq(
                    operation_id=http_worker_ipc[len(CONTROL_RETURN_PREFIX) :],
                    worker_id=self.worker_id,
                    payload=msgpack_encode(output),
                ),
            )
            return
        if (
            isinstance(output, BaseReq)
            and http_worker_ipc is not None
            and output.http_worker_ipc is None
        ):
            # Scheduler Req is not a BaseReq but carries the same return route.
            output.http_worker_ipc = http_worker_ipc

        sock_send(self.socket, output)
