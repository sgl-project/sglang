from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch

_in_piecewise_cuda_graph = False
_in_pcg_torch_compile = False
_pcg_capture_stream = None


def is_in_piecewise_cuda_graph():
    return _in_piecewise_cuda_graph


def is_in_pcg_torch_compile():
    return _in_pcg_torch_compile


def get_pcg_capture_stream():
    return _pcg_capture_stream


@contextmanager
def enable_piecewise_cuda_graph_compile():
    global _in_pcg_torch_compile
    _in_pcg_torch_compile = True
    yield
    _in_pcg_torch_compile = False


@contextmanager
def enable_piecewise_cuda_graph():
    global _in_piecewise_cuda_graph
    _in_piecewise_cuda_graph = True

    yield

    _in_piecewise_cuda_graph = False


@contextmanager
def set_pcg_capture_stream(stream: torch.cuda.Stream):
    global _pcg_capture_stream
    _pcg_capture_stream = stream
    yield
    _pcg_capture_stream = None


@dataclass
class ForwardContext:
    def __init__(self):
        self.forward_batch = None
        self.attention_layer = None
        self.quant_config = None
        self.moe_layers = None

    def set_forward_batch(self, forward_batch: ForwardBatch):
        self.forward_batch = forward_batch

    def set_attention_layers(self, layers: List[Any]):
        self.attention_layers = layers

    def set_quant_config(self, quant_config: Any):
        self.quant_config = quant_config

    def set_moe_layers(self, layers: List[Any]):
        self.moe_layers = layers


_forward_context: Optional[ForwardContext] = None


def get_forward_context() -> Optional[ForwardContext]:
    if _forward_context is None:
        return None
    return _forward_context


@contextmanager
def set_forward_context(
    forward_batch: ForwardBatch,
    attention_layers: List[Any],
    quant_config: Any,
    moe_layers: List[Any],
):
    global _forward_context
    _forward_context = ForwardContext()
    _forward_context.set_forward_batch(forward_batch)
    _forward_context.set_attention_layers(attention_layers)
    _forward_context.set_quant_config(quant_config)
    _forward_context.set_moe_layers(moe_layers)
    try:
        yield
    finally:
        _forward_context = None
