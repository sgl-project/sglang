import torch
from torch.utils._pytree import tree_map


class _WrapperTensor(torch.Tensor):
    @classmethod
    def __new__(cls, inner: torch.Tensor):
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            inner.shape,
            strides=inner.stride(),
            storage_offset=inner.storage_offset(),
            dtype=inner.dtype,
            device=inner.device,
            layout=inner.layout,
            requires_grad=inner.requires_grad,
        )
        r._inner = inner
        return r

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(e):
            if isinstance(e, cls):
                return e._unwrap()
            else:
                return e

        return func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

    def _unwrap(self):
        raise NotImplementedError


class DisposableTensor(_WrapperTensor):
    pass


class LazyTensor(_WrapperTensor):
    pass
