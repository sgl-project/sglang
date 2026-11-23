import threading
from typing import Optional, Union

import zmq

from sglang.srt.managers.io_struct import BaseBatchReq, BaseReq, sock_send


class SenderWrapper:
    def __init__(self, socket: zmq.Socket):
        self.socket = socket
        # Set once a second thread sends on this socket too.
        self._lock: Optional[threading.Lock] = None

    def enable_thread_safety(self) -> None:
        """Serialize sends with a lock: a zmq socket is not thread-safe."""
        if self._lock is None:
            self._lock = threading.Lock()

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

        lock = self._lock
        if lock is None:
            sock_send(self.socket, output)
        else:
            with lock:
                sock_send(self.socket, output)
