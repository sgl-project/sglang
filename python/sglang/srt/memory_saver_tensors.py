import torch


class _WrapperTensor(torch.Tensor):
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


class DisposableTensor(_WrapperTensor):
    pass


class LazyTensor(_WrapperTensor):
    pass
