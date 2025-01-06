from abc import ABC
from typing import Any, Type, List, Tuple, Callable


class Sender(ABC):
    def send(self, obj):
        raise NotImplementedError


class Receiver(ABC):
    def recv(self):
        raise NotImplementedError


class _ZMQSender(Sender):
    def send(self, obj):
        TODO


class _ZMQReceiver(Receiver):
    def recv(self):
        TODO


class TypeBasedDispatcher:
    def __init__(self, mapping: List[Tuple[Type, Callable]]):
        self._mapping = mapping

    def __call__(self, obj: Any):
        for ty, fn in self._mapping:
            if isinstance(obj, ty):
                fn(obj)
