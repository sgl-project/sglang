from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, List, Optional

from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass
class ForwardContext:
    def __init__(self):
        self.forward_batch = None
        self.attention_layer = None

    def set_forward_batch(self, forward_batch: ForwardBatch):
        self.forward_batch = forward_batch

    def set_attention_layers(self, layers: List[Any]):
        self.attention_layers = layers


_forward_context: Optional[ForwardContext] = None


def get_forward_context() -> Optional[ForwardContext]:
    if _forward_context is None:
        return None
    return _forward_context


@contextmanager
def set_forward_context(forward_batch: ForwardBatch, attention_layers: List[Any]):
    global _forward_context
    prev_forward_context = _forward_context
    _forward_context = ForwardContext()
    _forward_context.set_forward_batch(forward_batch)
    _forward_context.set_attention_layers(attention_layers)
    try:
        yield
    finally:
        _forward_context = prev_forward_context
