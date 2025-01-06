from abc import ABC
from typing import Any, Type, List, Tuple, Callable


class Sender(ABC):
    def send(self, obj):
        raise NotImplementedError


class Receiver(ABC):
    def recv(self):
        raise NotImplementedError


class ReceiverDispatcher:
    def __init__(self, receiver: Receiver, target: Any, mapping: List[Tuple[Type, Callable]]):
        self._receiver = receiver
        self._target = target
        self._mapping = mapping

    def recv_and_dispatch(self):
        TODO


class _ZMQSender(Sender):
    def send(self, obj):
        TODO


class _ZMQReceiver(Receiver):
    def recv(self):
        TODO
