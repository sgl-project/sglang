"""Run the worker's existing weight-sharing function during draft loading."""

from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch import nn

_sharing = ContextVar("draft_weight_sharing", default=None)


@contextmanager
def draft_weight_sharing(sharing):
    token = _sharing.set(sharing)
    try:
        yield
    finally:
        _sharing.reset(token)


def get_draft_weight_sharing():
    return _sharing.get()


def can_skip_vocab_loading(model):
    from sglang.srt.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
        VocabParallelEmbedding,
    )

    return all(
        type(module.quant_method) is UnquantizedEmbeddingMethod
        for module in model.modules()
        if isinstance(module, VocabParallelEmbedding)
    )


def _skip_weight(*args, **kwargs):
    pass


def unloaded_shared_weight(weight):
    if weight is None:
        return None
    # Run the real setters with storage-free parameters. Loading must never
    # write into the target; the same setters bind it after loading completes.
    placeholder = nn.Parameter(
        torch.empty_like(weight, device="meta"), requires_grad=False
    )
    placeholder.__dict__.update(weight.__dict__)
    placeholder.weight_loader = _skip_weight
    placeholder._shared_draft_weight = True
    return placeholder
