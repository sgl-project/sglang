from abc import ABC
from typing import Any, Type, List, Tuple, Callable

import zmq
from sglang.srt.utils import get_zmq_socket


class Sender(ABC):
    def send(self, obj):
        raise NotImplementedError


class Receiver(ABC):
    def recv(self):
        raise NotImplementedError


class _ZMQSender(Sender):
    def __init__(self, endpoint: str):
        self._socket = get_zmq_socket(zmq.Context(1), zmq.PUSH, endpoint)

    def send(self, obj):
        self._socket.send_pyobj(obj)


class _ZMQReceiver(Receiver):
    def __init__(self, endpoint: str):
        self._socket = get_zmq_socket(zmq.Context(1), zmq.PULL, endpoint)

    def recv(self):
        return self._socket.recv_pyobj()


class TypeBasedDispatcher:
    def __init__(self, mapping: List[Tuple[Type, Callable]]):
        self._mapping = mapping

    def __call__(self, obj: Any):
        for ty, fn in self._mapping:
            if isinstance(obj, ty):
                return fn(obj)
        raise ValueError(f"Invalid request: {obj}")
