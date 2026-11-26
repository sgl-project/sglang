from typing import Callable, Optional, Union

import zmq

from sglang.srt.managers.io_struct import BaseBatchReq, BaseReq, sock_send


class SenderWrapper:
    def __init__(self, socket: zmq.Socket):
        self.socket = socket
        self._pending_output: Optional[Callable[[], BaseBatchReq]] = None

    def defer_output(self, build_output: Callable[[], BaseBatchReq]) -> None:
        """Keep at most one builder, which must own all of its output state."""
        if self.socket is not None:
            self.flush_output()
            self._pending_output = build_output

    def flush_output(self) -> None:
        if self._pending_output is not None:
            sock_send(self.socket, self._pending_output())
            self._pending_output = None

    def send_output(
        self,
        output: Union[BaseReq, BaseBatchReq],
        recv_obj: Optional[object] = None,
    ):
        if self.socket is None:
            return

        # Terminal outputs, state releases and control replies must not overtake
        # the previous token delta, even before the scheduler's next forward.
        self.flush_output()
        http_worker_ipc = getattr(recv_obj, "http_worker_ipc", None)
        if (
            isinstance(output, BaseReq)
            and http_worker_ipc is not None
            and output.http_worker_ipc is None
        ):
            # Scheduler Req is not a BaseReq but carries the same return route.
            output.http_worker_ipc = http_worker_ipc

        sock_send(self.socket, output)
