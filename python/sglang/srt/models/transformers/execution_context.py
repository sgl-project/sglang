# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from itertools import count
from weakref import WeakValueDictionary

import torch


@dataclass(frozen=True)
class TransformersExecutionContext:
    forward_batch: object

    def num_token_non_padded(self):
        return self.forward_batch.moe_num_token_non_padded()


_EXECUTION_CONTEXT = ContextVar("transformers_execution_context", default=None)
_LAYER_HANDLES = count()
_LAYERS = WeakValueDictionary()


def register_execution_layer(layer) -> str:
    handle = str(next(_LAYER_HANDLES))
    _LAYERS[handle] = layer
    return handle


def get_execution_layer(handle: str):
    try:
        return _LAYERS[handle]
    except KeyError as exc:
        raise RuntimeError("Transformers execution layer is no longer alive") from exc


def get_transformers_execution_context() -> TransformersExecutionContext:
    context = _EXECUTION_CONTEXT.get()
    if context is None:
        raise RuntimeError("Transformers forward requires an execution context")
    return context


@contextmanager
def transformers_execution_context(forward_batch):
    if torch.compiler.is_compiling():
        yield
        return
    token = _EXECUTION_CONTEXT.set(TransformersExecutionContext(forward_batch))
    try:
        yield
    finally:
        _EXECUTION_CONTEXT.reset(token)


def wrap_forward_with_context(forward):
    @wraps(forward)
    def wrapped(*args, **kwargs):
        batch = kwargs.get("forward_batch")
        if batch is None:
            if len(args) < 3:
                raise TypeError("Transformers forward requires forward_batch")
            batch = args[2]
        with transformers_execution_context(batch):
            return forward(*args, **kwargs)

    return wrapped
