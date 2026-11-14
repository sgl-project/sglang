from typing import Callable, Optional, Union

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


# An in-process consumer of the scheduler's outputs (see `Scheduler.__init__`).
OutputSink = Callable[[Union[BaseReq, BaseBatchReq]], None]


class CallbackSender:
    """Hands each output to an in-process callback instead of a socket."""

    def __init__(self, sink: OutputSink):
        self.sink = sink

    def send_output(
        self,
        output: Union[BaseReq, BaseBatchReq],
        recv_obj: Optional[object] = None,
    ):
        self.sink(output)
