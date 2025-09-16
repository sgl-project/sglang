from dataclasses import dataclass
from typing import Optional, Any
from contextlib import contextmanager
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

@dataclass
class ForwardContext:
    def __init__(self):
        self.forward_batch = None
        self.attention_layer = None

    def set_forward_batch(self, forward_batch: ForwardBatch):
        self.forward_batch = forward_batch

    def set_attention_layer(self, layer: Any):
        self.attention_layer = layer

_forward_context: Optional[ForwardContext] = None

def get_forward_context() -> Optional[ForwardContext]:
    if _forward_context is None:
        raise RuntimeError("Forward context not found")
    return _forward_context

@contextmanager
def set_forward_context(forward_batch: ForwardBatch):
    global _forward_context
    prev_forward_context = _forward_context
    _forward_context = ForwardContext()
    _forward_context.set_forward_batch(forward_batch)
    try:
        yield
    finally:
        _forward_context = prev_forward_context

@contextmanager
def set_forward_attention_layer(layer: Any):
    global _forward_context
    prev_forward_context = _forward_context
    _forward_context.set_attention_layer(layer)
    try:
        yield
    finally:
        _forward_context = prev_forward_context