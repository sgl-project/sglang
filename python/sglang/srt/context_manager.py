from dataclasses import dataclass
from typing import Optional
from contextlib import contextmanager
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

@dataclass
class ForwardContext:
    def __init__(self, forward_batch: ForwardBatch):
        self.forward_batch = forward_batch

_forward_context: Optional[ForwardContext] = None

def get_forward_context() -> Optional[ForwardContext]:
    if _forward_context is None:
        raise RuntimeError("Forward context not found")
    return _forward_context

@contextmanager
def set_forward_context(forward_batch: ForwardBatch):
    global _forward_context
    prev_forward_context = _forward_context
    _forward_context = ForwardContext(forward_batch)
    try:
        yield
    finally:
        _forward_context = prev_forward_context