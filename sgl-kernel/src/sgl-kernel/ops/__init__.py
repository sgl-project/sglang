from .warp_reduce_cuda import reduce as _reduce


def warp_reduce(input_tensor):
    return _reduce(input_tensor)
