from __future__ import annotations

from typing import Any, Protocol


class SchedulerSender(Protocol):
    """Type for the tokenizer-process side of the tokenizer->scheduler IPC.

    Single-worker mode: zmq.asyncio.Socket directly.
    Multi-HTTP-worker mode: SenderWrapper (multi_tokenizer_mixin.py) which
    wraps the socket and stamps http_worker_ipcs onto BaseBatchReq objects.
    Both satisfy this Protocol.
    """

    def send_pyobj(self, obj: Any) -> None: ...
