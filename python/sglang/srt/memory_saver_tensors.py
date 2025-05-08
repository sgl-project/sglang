import torch


class _WrapperTensor(torch.Tensor):
    pass


class DisposableTensor(_WrapperTensor):
    pass


class LazyTensor(_WrapperTensor):
    pass
