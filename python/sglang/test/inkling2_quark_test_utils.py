"""Test helpers for the Inkling-2 Quark exports: the quantization_config of the four
exports, the tensors of a tiny export, the real attention at the dimensions of
Inkling-2-Mini, and a recorder of hidden copies of the packed weights."""

import torch
from torch.utils import _pytree
from torch.utils._python_dispatch import TorchDispatchMode


class Materialization(TorchDispatchMode):
    """Records (op, shape, dtype) of every output with new storage of an ATen op that
    reads one of the watched tensors: a cast or a copy, not a view."""

    def __init__(self, *watched):
        super().__init__()
        self.storages = {
            t.untyped_storage().data_ptr() for t in watched if t is not None
        }
        self.copies = []

    def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        leaves = _pytree.tree_leaves((args, kwargs))
        reads = any(
            isinstance(t, torch.Tensor)
            and t.untyped_storage().data_ptr() in self.storages
            for t in leaves
        )
        if reads:
            for o in _pytree.tree_leaves(out):
                if (
                    isinstance(o, torch.Tensor)
                    and o.untyped_storage().data_ptr() not in self.storages
                ):
                    self.copies.append((str(func), tuple(o.shape), str(o.dtype)))
        return out
