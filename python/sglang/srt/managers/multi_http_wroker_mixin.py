from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import zmq

from sglang.srt.managers.io_struct import MultiHttpWorkerObjWrapper
from sglang.srt.utils import get_zmq_socket

if TYPE_CHECKING:
    from sglang.srt.managers.detokenizer_manager import DetokenizerManager
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


class MultiHttpWorkerSenderWrapper:
    def __init__(self, sender: zmq.Socket, worker_id: int):
        self._sender = sender
        self.worker_id = worker_id

    def send_pyobj(self, obj: Any):
        # Wrap the request object with the worker id
        obj = MultiHttpWorkerObjWrapper(self.worker_id, obj)
        self._sender.send_pyobj(obj)


class MultiHttpWorkerTokenizerMixin:
    def maybe_init_multi_http_worker(self: TokenizerManager):
        """Init multi http worker related settings as a sub tokenizer in multi tokenizer manager mode"""
        self.is_sub_tokenizer = self.server_args.num_http_workers > 1

        if not self.is_sub_tokenizer:
            return

        # rebind the socket to the multi http worker collector
        self.worker_id = os.getpid()
        self.send_to_scheduler.close(linger=0)
        self.send_to_scheduler = get_zmq_socket(
            self.context, zmq.PUSH, self.port_args.multi_http_worker_ipc_name, False
        )
        self.send_to_scheduler = MultiHttpWorkerSenderWrapper(
            self.send_to_scheduler, self.worker_id
        )


class MultiHttpWorkerDetokenizerMixin:
    pass


class MultiHttpWorkerCollector:
    pass
