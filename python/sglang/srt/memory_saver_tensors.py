import torch
from torch.utils._pytree import tree_map


class _WrapperTensor(torch.Tensor):
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(e):
            if isinstance(e, cls):
                return e._unwrap()
            else:
                return e

        return func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

    def __repr__(self):
        return "WrapperTensor:" + repr(self._unwrap())

    def __str__(self):
        return "WrapperTensor:" + str(self._unwrap())

    def _unwrap(self):
        raise NotImplementedError


class DisposableTensor(_WrapperTensor):
    @staticmethod
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

    def dispose(self):
        self._inner = None

    def _unwrap(self):
        assert self._inner is not None, "Cannot use a DisposableTensor that is already disposed"
        return self._inner


class LazyTensor(_WrapperTensor):
    @staticmethod
    def __new__(cls, *args, **kwargs):
        r = torch.Tensor._make_wrapper_subclass(cls, *args, **kwargs)
        r._create_args = args
        r._create_kwargs = kwargs
        r._inner = None
        return r

    def _unwrap(self):
        if self._inner is None:
            self._inner = torch.empty(*self._create_args, **self._create_kwargs)
        return self._inner
